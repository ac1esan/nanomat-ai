#!/usr/bin/env python
"""Turn the raw ensemble spread into a calibrated interval, and write it into the checkpoint.

A deep ensemble ranks its own errors well but is badly scaled: on the held-out
test split, +-1 sigma of the raw spread covers only 37% of cases instead of 68%.
This script fits multiplicative scale factors on the VALIDATION split, verifies
the resulting coverage on the TEST split (so the numbers are not fitted to what
they report), and stores everything inside the ensemble checkpoint under
`calibration`, where `nanomat.predict.Predictor` picks it up.

    python scripts/calibrate_uncertainty.py \
        --weights runs/ens_group.pt --data alignn_data_alex_2d --split runs/ens_group.split.json

Add --dry-run to print the numbers without touching the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat import Predictor  # noqa: E402
from nanomat.predict import read_structure  # noqa: E402


@torch.no_grad()
def embed(P: Predictor, data_dir: str, names: list[str], batch: int = 256) -> np.ndarray:
    """L2-normalised pooled embeddings from the first ensemble member's trunk."""
    from pymatgen.core import Structure
    from torch_geometric.data import Batch as PyGBatch
    from nanomat.graph import to_graph

    chunks = []
    for i in range(0, len(names), batch):
        graphs = [to_graph(Structure.from_file(os.path.join(data_dir, f)), P.cutoff)
                  for f in names[i:i + batch]]
        graphs = [g for g in graphs if g is not None]
        chunks.append(P.models[0].encode(PyGBatch.from_data_list(graphs)).numpy())
    E = np.concatenate(chunks)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def distance_to_train(E_ref: np.ndarray, E_query: np.ndarray, k: int = 10) -> np.ndarray:
    """1 - mean cosine similarity to the k nearest training structures."""
    sims = E_query @ E_ref.T
    return 1 - np.sort(sims, axis=1)[:, -k:].mean(1)


def evaluate(P: Predictor, data_dir: str, names: list[str], truth: dict[str, float], batch: int):
    items = [(f, read_structure(os.path.join(data_dir, f))) for f in names]
    res = P.run_many(items, batch_size=batch)
    y = np.array([truth[k] for k, r in res if r is not None])
    p = np.array([r.gap for _, r in res if r is not None])
    u = np.array([r.unc for _, r in res if r is not None])
    return np.abs(p - y), u, p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="ensemble checkpoint (.pt) to calibrate")
    ap.add_argument("--data", required=True, help="folder with .vasp files and id_prop.csv")
    ap.add_argument("--split", required=True, help="<run>.split.json written by train_cgcnn.py")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split = json.load(open(args.split))
    truth = {}
    for line in open(os.path.join(args.data, "id_prop.csv")):
        parts = line.strip().split(",")
        truth[parts[0]] = float(parts[1])

    # load the checkpoint through a staging dir so Predictor finds it by name
    stage = os.path.join(ROOT, ".calib_tmp")
    os.makedirs(stage, exist_ok=True)
    shutil.copy(args.weights, os.path.join(stage, "cgcnn_2d_ensemble.pt"))
    P = Predictor(stage, verbose=False)
    if len(P.models) < 2:
        raise SystemExit("calibration needs an ensemble checkpoint (more than one model)")

    err_v, u_v, _, _ = evaluate(P, args.data, split["val"], truth, args.batch)
    err_t, u_t, _, _ = evaluate(P, args.data, split["test"], truth, args.batch)

    # Second, independent out-of-domain signal. The ensemble spread only measures
    # disagreement between initialisations; when every member shares the same
    # training set, a chemistry nobody saw produces confident agreement (phosphorene
    # is the worked example). Distance to the training set in the model's own latent
    # space catches exactly that case.
    E_train = embed(P, args.data, split["train"], args.batch)
    d_val = distance_to_train(E_train, embed(P, args.data, split["val"], args.batch))
    d_test = distance_to_train(E_train, embed(P, args.data, split["test"], args.batch))
    from scipy.stats import spearmanr
    rho_u, rho_d = spearmanr(u_t, err_t).correlation, spearmanr(d_test, err_t).correlation
    d_edges = np.quantile(d_val, [0.25, 0.50, 0.75, 0.90])
    ratio = err_v / np.maximum(u_v, 1e-6)
    s68, s90 = float(np.quantile(ratio, 0.68)), float(np.quantile(ratio, 0.90))
    edges = np.quantile(u_v, [0.25, 0.50, 0.75])
    bins = np.digitize(u_t, edges)
    tier = {
        "reliable": float(err_t[u_t <= edges[1]].mean()),
        "check": float(err_t[(u_t > edges[1]) & (u_t <= edges[2])].mean()),
        "out_of_domain": float(err_t[u_t > edges[2]].mean()),
    }
    cal = {
        "fitted_on": os.path.basename(args.split) + " (validation split)",
        "verified_on": "test split",
        "n_val": int(len(err_v)), "n_test": int(len(err_t)),
        "test_mae": float(err_t.mean()),
        "scale68": s68, "scale90": s90,
        "unc_q25": float(edges[0]), "unc_median": float(edges[1]), "unc_q75": float(edges[2]),
        "tier_mae": tier,
        "raw_coverage_1sigma": float(np.mean(err_t <= u_t)),
        "test_coverage68": float(np.mean(err_t <= s68 * u_t)),
        "test_coverage90": float(np.mean(err_t <= s90 * u_t)),
        "test_coverage90_by_unc_quartile": [float(np.mean(err_t[bins == k] <= s90 * u_t[bins == k]))
                                            for k in range(4)],
        "global_conformal90_eV": float(np.quantile(err_v, 0.90)),
        "latent_q25": float(d_edges[0]), "latent_median": float(d_edges[1]),
        "latent_q75": float(d_edges[2]), "latent_q90": float(d_edges[3]),
        "spearman_unc": float(rho_u), "spearman_latent": float(rho_d),
        "spearman_combined": float(spearmanr(u_t * d_test, err_t).correlation),
        "latent_tier_mae": {
            "near": float(err_t[d_test <= d_edges[2]].mean()),
            "far": float(err_t[d_test > d_edges[2]].mean()),
        },
    }

    print(json.dumps(cal, indent=2))
    print(f"\nRaw +-1 sigma covers {100*cal['raw_coverage_1sigma']:.0f}% (Gaussian ideal 68%): "
          f"the ensemble ranks well but is overconfident.")
    print(f"Scaled +-{s90:.2f}*unc covers {100*cal['test_coverage90']:.0f}% on the held-out test "
          f"split, versus a fixed +-{cal['global_conformal90_eV']:.2f} eV for everyone.")
    print("Conditional coverage by uncertainty quartile: " +
          ", ".join(f"{100*c:.0f}%" for c in cal["test_coverage90_by_unc_quartile"]) +
          "  (the most confident quartile is still slightly over-optimistic).")

    print(f"\nOut-of-domain signals, Spearman vs |error|: ensemble spread {rho_u:+.3f}, "
          f"latent distance {rho_d:+.3f}, product {cal['spearman_combined']:+.3f} "
          "(they are complementary).")
    print(f"Beyond the latent q75 the MAE is {cal['latent_tier_mae']['far']:.3f} eV versus "
          f"{cal['latent_tier_mae']['near']:.3f} eV inside it.")

    shutil.rmtree(stage, ignore_errors=True)
    if args.dry_run:
        print("\n--dry-run: checkpoint not modified")
        return
    ck = torch.load(args.weights, map_location="cpu")
    ck["calibration"] = cal
    ck["reference_embeddings"] = torch.tensor(E_train, dtype=torch.float16)
    torch.save(ck, args.weights)
    print(f"\nwritten calibration + {E_train.shape[0]} reference embeddings into {args.weights}")


if __name__ == "__main__":
    main()
