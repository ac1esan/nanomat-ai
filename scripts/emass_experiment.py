#!/usr/bin/env python
"""Band-edge effective masses: can structure predict them, and how?

For a transistor channel the gap decides whether the device switches off; the
effective mass decides how fast carriers move once it is on. The JARVIS dft_2d masses
proved unusable (values in the millions). The full C2DB release stores emass_cbm and
emass_vbm for ~2 950 non-magnetic semiconductors each, inside 0.01-20 m_e (outside it
are flat bands and failed fits). Masses span two decades, so the target is log10 |m|
and errors are in dex: 0.1 dex is a factor 1.26.

Three ways to predict one carrier's mass, all scored on the composition-disjoint test
split that train_cgcnn.py writes for it:
  CGCNN    a dedicated 5-member ensemble with angular features (the recipe of the
           gap and work-function models), with its own calibration and verdict
  latent   ridge on the gap ensemble's embedding + predicted gap, fitted on the same
           split's train+val; answers only where the gap model does not disown the
           structure, and inherits the gap model's verdict
  comp     composition-only random forest (scripts/composition_baseline.py), the bar

Decision rule, written before any model was trained (16 October 2026). Electrons and
holes are decided separately.
  1. CGCNN qualifies if its log10 MAE on all test structures is at least 10% below
     the composition baseline's on the same structures.
  2. The latent head qualifies if its log10 MAE on the test structures the gap model
     answers is at least 10% below the composition baseline's on the same structures.
  3. A qualifier's verdict must rank its error on the test split: for CGCNN its own
     tiers (thresholds from validation), reliable < check < out-of-domain; for the
     latent head the gap model's tiers, reliable < check.
  4. Of those passing, the lower log10 MAE on the test structures the gap model
     answers ships; within 0.01 dex the latent head ships (no new checkpoint). If
     none passes, nothing ships for that carrier.
Reported, not voting: paired bootstrap 95% intervals of each difference to the
composition baseline, the four 1H TMDs (with their role in each split), and
literature masses for orientation.

    python scripts/emass_experiment.py prepare      # export both carriers from c2db.db
    # train (see runs/emass/run.sh), calibrate, run the composition baseline, then:
    python scripts/emass_experiment.py evaluate --device cuda

Data: C2DB, CC BY-NC 4.0; Haastrup et al., 2D Mater. 5, 042002 (2018); Gjerding et
al., 2D Mater. 8, 044002 (2021).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

CARRIERS = {"cbm": "electrons", "vbm": "holes"}
DATA = {c: os.path.join(ROOT, f"data_c2db_emass_{c}") for c in CARRIERS}
RUNS = os.path.join(ROOT, "runs", "emass")
TMD_FILES = {"MoS2": "1MoS2-1.vasp", "MoSe2": "1MoSe2-1.vasp",
             "WS2": "1WS2-1.vasp", "WSe2": "1WSe2-1.vasp"}

RUN_SH = """#!/bin/sh
# effective-mass experiment: two ensembles in parallel, then calibration (sequential:
# calibrate_uncertainty.py stages through one shared folder), baselines, evaluation
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python}
W=${WORKERS:-8}
# one ensemble per GPU when there are two (GPUS="0 1"), both on one otherwise
set -- ${GPUS:-0 0}
for c in cbm vbm; do
  CUDA_VISIBLE_DEVICES=$1 $PY train_cgcnn.py --data data_c2db_emass_$c --task gap --split group \\
      --ensemble 5 --angles 9 --cache --workers $W --log-every 20 --out runs/emass/$c.pt \\
      > runs/emass/train_$c.log 2>&1 &
  shift
done
wait
for c in cbm vbm; do
  $PY scripts/calibrate_uncertainty.py --weights runs/emass/$c.pt \\
      --data data_c2db_emass_$c --split runs/emass/$c.split.json > runs/emass/calib_$c.log 2>&1
  $PY scripts/composition_baseline.py --data data_c2db_emass_$c --split runs/emass/$c.split.json \\
      --jobs $W --json runs/emass/$c.baseline.json --preds runs/emass/$c.baseline.csv \\
      > runs/emass/baseline_$c.log 2>&1
done
$PY scripts/emass_experiment.py evaluate --device ${DEVICE:-cuda} > runs/emass/evaluate.log 2>&1
"""


def prepare():
    from c2db_full import export
    os.makedirs(RUNS, exist_ok=True)
    for c in CARRIERS:
        export(DATA[c], f"emass_{c}", log10=True)
    with open(os.path.join(RUNS, "run.sh"), "w") as f:
        f.write(RUN_SH)
    os.chmod(os.path.join(RUNS, "run.sh"), 0o755)
    print(f"written {RUNS}/run.sh")


def boot_ci(d: np.ndarray, n: int = 5000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def tiers_mae(err: np.ndarray, verdicts: list[str]) -> dict:
    from nanomat.predict import tier_key
    keys = np.array([tier_key(v) for v in verdicts])
    return {k: {"n": int((keys == k).sum()),
                "mae": float(err[keys == k].mean()) if (keys == k).any() else None}
            for k in ("reliable", "check", "out_of_domain")}


def ranks(t: dict, keys: tuple) -> bool:
    vals = [t[k]["mae"] for k in keys]
    return all(v is not None for v in vals) and all(a < b for a, b in zip(vals, vals[1:]))


def evaluate_carrier(c: str, P, device: str) -> dict:
    import pandas as pd
    from nanomat.predict import PropertyEnsemble, read_structure
    from c2db_full import embed
    from fit_exciton import ridge

    name = CARRIERS[c]
    split = json.load(open(os.path.join(RUNS, f"{c}.split.json")))
    truth = {l.split(",")[0]: float(l.split(",")[1]) for l in open(os.path.join(DATA[c], "id_prop.csv"))}
    files = split["train"] + split["val"] + split["test"]
    sts = {f: read_structure(os.path.join(DATA[c], f)) for f in files}

    # the gap model on everything: its verdict says where the latent head may answer
    res = dict(P.run_many([(f, sts[f]) for f in files], batch_size=256, device=device))
    ok = {f: res.get(f) is not None and not res[f].verdict.startswith("out-of-domain") for f in files}
    fit = [f for f in split["train"] + split["val"] if ok[f]]
    test = [f for f in split["test"] if res.get(f) is not None]
    test_in = [f for f in test if ok[f]]

    def feats(fs):
        return np.column_stack([embed(P, [sts[f] for f in fs]), [res[f].gap for f in fs]])
    Xf, Xt = feats(fit), feats(test_in)
    sc, m = ridge(Xf, np.array([truth[f] for f in fit]), np.arange(len(fit)))
    lat = dict(zip(test_in, m.predict(sc.transform(Xt))))

    # the dedicated ensemble, on graphs built the way the tool builds them
    E = PropertyEnsemble(os.path.join(RUNS, f"{c}.pt"), log=lambda *_: None,
                         high_unc_reason="unusual structure for this property")
    for mm in E.models:
        mm.to(next(P.models[0].parameters()).device)
    graphs = [P.graph(sts[f]) for f in test]
    vals, spreads, dists = [], [], []
    for i in range(0, len(graphs), 256):
        v, s = E.predict(graphs[i:i + 256])
        vals.append(v), spreads.append(s), dists.append(E.latent_distance(graphs[i:i + 256]))
    cg = dict(zip(test, np.concatenate(vals)))
    cg_v = {f: E.judge(float(u), float(d)) for f, u, d in
            zip(test, np.concatenate(spreads), np.concatenate(dists))}

    comp = pd.read_csv(os.path.join(RUNS, f"{c}.baseline.csv")).set_index("file")["pred"].to_dict()
    test = [f for f in test if f in comp]
    test_in = [f for f in test_in if f in comp]
    y = lambda fs: np.array([truth[f] for f in fs])
    err = lambda pred, fs: np.abs(np.array([pred[f] for f in fs]) - y(fs))

    e_cg_all, e_comp_all = err(cg, test), err(comp, test)
    e_cg_in, e_lat_in, e_comp_in = err(cg, test_in), err(lat, test_in), err(comp, test_in)
    q1 = e_cg_all.mean() <= 0.9 * e_comp_all.mean()
    q2 = e_lat_in.mean() <= 0.9 * e_comp_in.mean()
    t_cg = tiers_mae(e_cg_all, [cg_v[f] for f in test])
    t_lat = tiers_mae(e_lat_in, [res[f].verdict for f in test_in])
    r_cg, r_lat = ranks(t_cg, ("reliable", "check", "out_of_domain")), ranks(t_lat, ("reliable", "check"))
    passing = [k for k, q, r in (("cgcnn", q1, r_cg), ("latent", q2, r_lat)) if q and r]
    mae_in = {"cgcnn": float(e_cg_in.mean()), "latent": float(e_lat_in.mean()), "comp": float(e_comp_in.mean())}
    ship = None
    if passing:
        ship = min(passing, key=lambda k: mae_in[k])
        if ship == "cgcnn" and "latent" in passing and mae_in["latent"] - mae_in["cgcnn"] <= 0.01:
            ship = "latent"

    print(f"\n=== {name} ({c}): test {len(test)}, answered by the gap model {len(test_in)}; "
          f"latent head fitted on {len(fit)}")
    print(f"  log10 MAE, all test:      CGCNN {e_cg_all.mean():.3f}   comp {e_comp_all.mean():.3f}"
          f"   (diff {np.mean(e_cg_all - e_comp_all):+.3f}, CI {boot_ci(e_cg_all - e_comp_all)})")
    print(f"  log10 MAE, answered test: CGCNN {mae_in['cgcnn']:.3f}   latent {mae_in['latent']:.3f}"
          f"   comp {mae_in['comp']:.3f}   (latent - comp CI {boot_ci(e_lat_in - e_comp_in)})")
    print(f"  as a factor: CGCNN x{10 ** e_cg_all.mean():.2f}, latent x{10 ** mae_in['latent']:.2f}, "
          f"comp x{10 ** e_comp_all.mean():.2f}")
    print(f"  CGCNN own tiers: " + ", ".join(f"{k} {v['mae']:.3f} (n={v['n']})" if v["mae"] is not None
                                            else f"{k} - " for k, v in t_cg.items()))
    print(f"  latent, gap tiers: " + ", ".join(f"{k} {v['mae']:.3f} (n={v['n']})" if v["mae"] is not None
                                              else f"{k} - " for k, v in t_lat.items()))
    print(f"  rule: 1. CGCNN >=10% below comp: {q1}   2. latent >=10% below comp: {q2}"
          f"\n        3. tiers rank: CGCNN {r_cg}, latent {r_lat}\n        4. ships: {ship}")

    tmd = {}
    for nm, f in TMD_FILES.items():
        role = next((k for k in ("train", "val", "test") if f in split[k]), None)
        if role is None:
            continue
        g = np.array([[res[f].gap]]) if res.get(f) is not None else None
        row = {"role": role, "c2db_m": float(10 ** truth[f])}
        if res.get(f) is not None:
            v, s = E.predict([P.graph(sts[f])])
            row["cgcnn_m"] = float(10 ** v[0])
            if ok[f]:
                row["latent_m"] = float(10 ** m.predict(sc.transform(
                    np.column_stack([embed(P, [sts[f]]), g])))[0])
        tmd[nm] = row
        print(f"  {nm:6s} ({role:5s}) C2DB {row['c2db_m']:.3f}  CGCNN {row.get('cgcnn_m', float('nan')):.3f}"
              f"  latent {row.get('latent_m', float('nan')):.3f}  m_e")
    return {"n_test": len(test), "n_answered": len(test_in), "n_latent_fit": len(fit),
            "mae_all": {"cgcnn": float(e_cg_all.mean()), "comp": float(e_comp_all.mean())},
            "mae_answered": mae_in,
            "ci_cgcnn_minus_comp": boot_ci(e_cg_all - e_comp_all),
            "ci_latent_minus_comp": boot_ci(e_lat_in - e_comp_in),
            "tiers_cgcnn": t_cg, "tiers_latent": t_lat,
            "rule": {"1": bool(q1), "2": bool(q2), "3_cgcnn": bool(r_cg), "3_latent": bool(r_lat)},
            "ships": ship, "tmd": tmd}


def evaluate(device: str):
    from nanomat import Predictor
    P = Predictor(verbose=False)
    out = {c: evaluate_carrier(c, P, device) for c in CARRIERS}
    out["written"] = "2026-10-16, rule in this file's docstring"
    json.dump(out, open(os.path.join(RUNS, "evaluation.json"), "w"), indent=1)
    print(f"\nwritten {RUNS}/evaluation.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "evaluate"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    prepare() if args.step == "prepare" else evaluate(args.device)


if __name__ == "__main__":
    main()
