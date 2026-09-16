#!/usr/bin/env python
"""Compare two ensembles on one test set: is the trade worth making?

Bagging makes every member train on a resample, so a chemistry represented once is
missed by roughly a third of them and they disagree. That should make the spread
honest about sparse data — which plain ensembling cannot be, since all members share
one training set. The price is that each member sees about 64% of the structures.

Accuracy alone cannot decide this. The spread exists to be a filter, so the question
is whether it ranks error better, not only whether the mean is closer. This prints
both, plus the case that motivated the whole thing: phosphorene, where the plain
ensemble reported 0.82 eV against an experimental 2.0 with a spread of 0.035 eV.

    python scripts/compare_ensembles.py --a runs/ens_group.pt --b runs/ens_bagged.pt \
        --data alignn_data_alex_2d --split runs/ens_group.split.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat import Predictor  # noqa: E402
from nanomat.predict import read_structure  # noqa: E402

REFERENCES = ["MoS2", "MoSe2", "WS2", "WSe2", "hBN", "phosphorene", "graphene"]


def load(path: str, tag: str) -> Predictor:
    stage = os.path.join(ROOT, f".cmp_{tag}")
    os.makedirs(stage, exist_ok=True)
    shutil.copy(path, os.path.join(stage, "cgcnn_2d_ensemble.pt"))
    return Predictor(stage, verbose=False)


def evaluate(P: Predictor, data: str, names: list[str], truth: dict[str, float]) -> dict:
    from scipy.stats import spearmanr
    items = [(f, read_structure(os.path.join(data, f))) for f in names]
    res = P.run_many(items, batch_size=256)
    y = np.array([truth[k] for k, r in res if r is not None])
    p = np.array([r.gap for _, r in res if r is not None])
    u = np.array([r.unc for _, r in res if r is not None])
    err = np.abs(p - y)
    q = np.quantile(u, [0.25, 0.5, 0.75])
    bins = np.digitize(u, q)
    return {
        "n": int(len(y)),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "spearman": float(spearmanr(u, err).correlation),
        "median_unc": float(np.median(u)),
        "mae_by_unc_quartile": [float(err[bins == k].mean()) for k in range(4)],
        # how much worse the least confident quartile is than the most confident:
        # the whole point of a trust filter is that this ratio is large
        "quartile_ratio": float(err[bins == 3].mean() / max(err[bins == 0].mean(), 1e-9)),
    }


def references(P: Predictor) -> dict:
    out = {}
    for name in REFERENCES:
        path = os.path.join(ROOT, "examples", f"{name}.vasp")
        if not os.path.exists(path):
            continue
        r = P.run(read_structure(path))
        out[name] = {"gap": round(r.gap, 3), "unc": round(r.unc, 4),
                     "verdict": r.verdict.split(" (")[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline checkpoint (plain ensemble)")
    ap.add_argument("--b", required=True, help="checkpoint to judge (bagged)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--label-a", default="plain")
    ap.add_argument("--label-b", default="bagged")
    ap.add_argument("--json")
    args = ap.parse_args()

    split = json.load(open(args.split))
    truth = {}
    for line in open(os.path.join(args.data, "id_prop.csv")):
        parts = line.strip().split(",")
        truth[parts[0]] = float(parts[1])

    out = {}
    for tag, path, label in (("a", args.a, args.label_a), ("b", args.b, args.label_b)):
        P = load(path, tag)
        out[label] = {"test": evaluate(P, args.data, split["test"], truth),
                      "references": references(P)}
        shutil.rmtree(os.path.join(ROOT, f".cmp_{tag}"), ignore_errors=True)

    A, B = out[args.label_a], out[args.label_b]
    print(f"{'':22s} {args.label_a:>12s} {args.label_b:>12s}")
    print("-" * 48)
    for key, fmt in (("mae", "{:.4f}"), ("rmse", "{:.4f}"), ("spearman", "{:+.3f}"),
                     ("median_unc", "{:.4f}"), ("quartile_ratio", "{:.2f}x")):
        print(f"{key:22s} {fmt.format(A['test'][key]):>12s} {fmt.format(B['test'][key]):>12s}")
    print(f"\nMAE by uncertainty quartile (the filter working means these rise steeply):")
    print(f"  {args.label_a:>10s}: " + "  ".join(f"{v:.3f}" for v in A["test"]["mae_by_unc_quartile"]))
    print(f"  {args.label_b:>10s}: " + "  ".join(f"{v:.3f}" for v in B["test"]["mae_by_unc_quartile"]))

    print(f"\nreference monolayers — gap and spread:")
    print(f"{'material':13s} {args.label_a:>22s} {args.label_b:>22s}")
    for name in REFERENCES:
        if name not in A["references"]:
            continue
        a, b = A["references"][name], B["references"][name]
        print(f"{name:13s} {a['gap']:8.2f} ± {a['unc']:.3f} {a['verdict'][:6]:>6s} "
              f"{b['gap']:8.2f} ± {b['unc']:.3f} {b['verdict'][:6]:>6s}")

    pa = A["references"].get("phosphorene")
    pb = B["references"].get("phosphorene")
    if pa and pb:
        print(f"\nPhosphorene is the case that motivated bagging: one elemental-phosphorus "
              f"structure\nin the whole training set, so every plain member learned from it and "
              f"they agreed.\nIts spread goes {pa['unc']:.3f} -> {pb['unc']:.3f} eV, a factor of "
              f"{pb['unc'] / max(pa['unc'], 1e-9):.1f}.")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
