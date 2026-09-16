#!/usr/bin/env python
"""Fit the two corrections that turn a PBE-level prediction into something measurable.

The model is trained on PBE gaps, which are not what anyone measures. There are two
different physical targets, and conflating them is the mistake this script exists to
avoid:

  quasiparticle gap  — what you would measure in photoemission, and what a transport
                       or device calculation wants. Approximated here by HSE06.
  optical gap        — the absorption onset, which is the quasiparticle gap minus the
                       exciton binding energy. In a 2D monolayer that binding energy
                       is several tenths of an eV, so the two differ a lot.

Both are fitted against the model's own output rather than against PBE, so each
absorbs the functional difference and the model's own bias in one step.

  quasiparticle: 32 structures from JARVIS dft_2d that carry `hse_gap`, restricted
                 to ones the tool does not flag as out-of-domain. Validated
                 leave-one-out.
  optical:       the reference monolayers in examples/ with measured optical gaps.
                 Only five usable points, so this one stays a rough estimate.

    python scripts/fit_gap_corrections.py              # fit and report
    python scripts/fit_gap_corrections.py --write      # also store in the checkpoint

Run without a proxy if one intercepts figshare:
    env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy python scripts/...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat import Predictor  # noqa: E402
from nanomat.predict import read_structure  # noqa: E402

ENSEMBLE = os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.pt")
# measured optical gaps, same references validate_experiment.py uses
OPTICAL = {"MoS2": 1.88, "MoSe2": 1.55, "WS2": 2.00, "WSe2": 1.65,
           "hBN": 6.00, "phosphorene": 2.00, "graphene": 0.00}


def loo_mae(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out error of a linear fit: refit without each point, predict it."""
    errs = []
    for i in range(len(x)):
        k = np.arange(len(x)) != i
        a, b = np.polyfit(x[k], y[k], 1)
        errs.append(abs(a * x[i] + b - y[i]))
    return float(np.mean(errs))


def fit_quasiparticle(P: Predictor) -> dict:
    from jarvis.core.atoms import Atoms
    from jarvis.db.figshare import data as jarvis_data

    num = lambda v: pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    raw = jarvis_data("dft_2d")
    rows = []
    for e in raw:
        hse = num(e.get("hse_gap"))
        if pd.isna(hse) or hse <= 0.01:
            continue
        rows.append({"jid": e.get("jid"), "hse": float(hse), "atoms": e["atoms"]})
    print(f"dft_2d entries carrying an HSE gap: {len(rows)}")

    items = [(r["jid"], Atoms.from_dict(r["atoms"]).pymatgen_converter()) for r in rows]
    res = dict(P.run_many(items, batch_size=128))
    keep_x, keep_y, dropped = [], [], 0
    for r in rows:
        pr = res.get(r["jid"])
        if pr is None or pr.verdict.startswith("out-of-domain"):
            dropped += 1
            continue
        keep_x.append(pr.gap)
        keep_y.append(r["hse"])
    x, y = np.array(keep_x), np.array(keep_y)
    print(f"  dropped as out-of-domain: {dropped}  -> fitting on {len(x)}")

    a, b = np.polyfit(x, y, 1)
    out = {
        "a": float(a), "b": float(b), "n": int(len(x)),
        "mae_in_sample": float(np.mean(np.abs(a * x + b - y))),
        "mae_loo": loo_mae(x, y),
        "mae_uncorrected": float(np.mean(np.abs(x - y))),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "source": "JARVIS dft_2d hse_gap",
    }
    print(f"  raw prediction vs HSE: MAE {out['mae_uncorrected']:.3f} eV, r = {out['pearson']:.3f}")
    print(f"  fit: HSE ~= {a:.3f}*pred {b:+.3f}")
    print(f"  MAE in-sample {out['mae_in_sample']:.3f}, leave-one-out {out['mae_loo']:.3f} eV")
    return out


def fit_optical(P: Predictor) -> tuple[dict, list[dict]]:
    rows = []
    for name, exp in OPTICAL.items():
        path = os.path.join(ROOT, "examples", f"{name}.vasp")
        if not os.path.exists(path):
            continue
        r = P.run(read_structure(path))
        rows.append({"material": name, "pred": r.gap, "exp": exp,
                     "ood": r.verdict.startswith("out-of-domain")})
    usable = [r for r in rows if r["exp"] > 0.1 and not r["ood"]]
    x = np.array([r["pred"] for r in usable])
    y = np.array([r["exp"] for r in usable])
    a, b = np.polyfit(x, y, 1)
    out = {
        "a": float(a), "b": float(b), "n": int(len(x)),
        "mae_in_sample": float(np.mean(np.abs(a * x + b - y))),
        "mae_loo": loo_mae(x, y),
        "mae_uncorrected": float(np.mean(np.abs(x - y))),
        "fitted_on": [r["material"] for r in usable],
        "source": "measured optical gaps of reference monolayers",
    }
    print(f"\noptical fit on {out['n']} reference monolayers "
          f"({', '.join(out['fitted_on'])})")
    print(f"  fit: optical ~= {a:.3f}*pred {b:+.3f}")
    print(f"  MAE in-sample {out['mae_in_sample']:.3f}, leave-one-out {out['mae_loo']:.3f} eV")
    return out, rows


def report_exciton(qp: dict, opt: dict, rows: list[dict]) -> dict:
    """The gap between the two corrections should be the exciton binding energy."""
    print("\nThe two corrections do not compete — they target different quantities.")
    print(f"{'material':13s} {'pred':>6s} {'quasipart.':>11s} {'optical':>9s} "
          f"{'measured':>9s} {'difference':>11s}")
    diffs = []
    for r in rows:
        if r["ood"] or r["exp"] <= 0.1:
            continue
        q = qp["a"] * r["pred"] + qp["b"]
        o = opt["a"] * r["pred"] + opt["b"]
        diffs.append({"material": r["material"], "binding_eV": round(float(q - o), 3)})
        print(f"{r['material']:13s} {r['pred']:6.2f} {q:11.2f} {o:9.2f} {r['exp']:9.2f} "
              f"{q - o:11.2f}")
    tmd = [d["binding_eV"] for d in diffs if d["material"] in ("MoS2", "MoSe2", "WS2", "WSe2")]
    vals = [d["binding_eV"] for d in diffs]
    print(f"\nThat difference is the exciton binding energy. On the four TMDs it averages "
          f"{np.mean(tmd):.2f} eV\n(range {min(tmd):.2f}-{max(tmd):.2f}), which is the "
          "published order of magnitude for a monolayer — so the two\nfits are physically "
          "consistent rather than contradictory, and reporting only one would be the\nactual "
          "error.")
    print("h-BN sits apart at "
          f"{[d['binding_eV'] for d in diffs if d['material'] == 'hBN']}: its 6.0 eV reference is "
          "itself an excitonic\npeak, not a quasiparticle gap, so the comparison does not hold "
          "there.")
    return {"mean_tmd_eV": float(np.mean(tmd)), "mean_all_eV": float(np.mean(vals)),
            "per_material": diffs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="store the fits in the checkpoint")
    args = ap.parse_args()

    P = Predictor(verbose=False)
    qp = fit_quasiparticle(P)
    opt, rows = fit_optical(P)
    exciton = report_exciton(qp, opt, rows)

    payload = {"quasiparticle": qp, "optical": opt, "exciton_binding": exciton}
    if not args.write:
        print("\n--write not given: checkpoint untouched")
        return
    ck = torch.load(ENSEMBLE, map_location="cpu")
    ck["gap_corrections"] = payload
    torch.save(ck, ENSEMBLE)
    print(f"\nwritten gap_corrections into {ENSEMBLE}")


if __name__ == "__main__":
    main()
