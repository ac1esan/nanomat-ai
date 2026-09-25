#!/usr/bin/env python
"""Extend the calibration to the population the validation split does not contain.

calibrate_uncertainty.py fits the interval scale and the verdict thresholds on the
validation split, which is near-hull (e_above_hull <= 0.1 eV/atom), and checks them
on the test split: 90% coverage. The ensemble has since been trained on metastable
monolayers too (0.1-0.2 eV/atom, stability_experiment.py). On the held-out ones the
same interval covers about 85%, and inside the reliable tier about 74%, because a
confident prediction there errs twice as much (MAE 0.20 against 0.10).

Neither trust signal sees it. At equal spread or equal latent distance a metastable
structure errs more, and each signal alone separates the two populations barely
better than chance (ROC-AUC 0.59 and 0.60). The latent space itself does: a linear
head on the ensemble's embeddings separates held-out metastable structures from
held-out near-hull ones at ROC-AUC ~0.85. So this script:

  1. fits that head - logistic regression on the concatenated member embeddings,
     near-hull against metastable TRAINING structures (Alexandria only: 2DMatPedia
     records no hull distance) - and scores it on validation + test against T_meta;
  2. fits the 90% interval scale per cell of (head > 0.5) x (spread quartile), on
     the validation split plus the held-out metastable set T_meta;
  3. measures, and does NOT write, a verdict rule: reliable becomes check when the
     head says metastable. It sharpens the reliable tier on near-hull structures
     (0.104 -> 0.094 eV) and inverts the tier order on held-out metastable ones
     (reliable 0.25, check 0.21): the structures it leaves in the reliable tier are
     the metastable ones the head mistakes for near-hull, which is exactly where the
     model is confidently wrong. The screening table showed the same inversion on
     C2DB and JARVIS rows, so the head sets the interval and nothing else;
and reports what that buys with 20 repeated halvings of T_meta by composition: fit
on the validation split and one half, report on the other half and on the test
split, which no fit here touches. The tier errors written into the calibration are
re-measured with the full verdict (spread and latent distance) on the test split.

    python scripts/calibrate_population.py --weights weights/cgcnn_2d_ensemble.pt --data alignn_data_exp

--dry-run prints everything without writing. Re-run after calibrate_uncertainty.py
whenever the ensemble is retrained: the head reads that ensemble's latent space.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat import Predictor  # noqa: E402
from nanomat.predict import read_structure  # noqa: E402

P_THRESHOLD = 0.5
MIN_CELL = 30          # fewer points than this and a cell borrows its spread column's scale


def fit_table(ratio, pm, unc, edges, fallback: float) -> list[list[float]]:
    row, col = (pm > P_THRESHOLD).astype(int), np.digitize(unc, edges)
    table = []
    for r in (0, 1):
        cells = []
        for c in range(len(edges) + 1):
            sel = (row == r) & (col == c)
            if sel.sum() < MIN_CELL:
                sel = col == c
            cells.append(float(np.quantile(ratio[sel], 0.9)) if sel.sum() >= MIN_CELL else fallback)
        table.append(cells)
    return table


def apply_table(table, pm, unc, edges) -> np.ndarray:
    t = np.asarray(table)
    return t[(pm > P_THRESHOLD).astype(int), np.digitize(unc, edges)]


def tiers(unc, lat, pm, cal, rule: bool) -> np.ndarray:
    """0 reliable, 1 check, 2 out-of-domain - nanomat.predict.verdict, vectorised."""
    t = np.where(unc <= cal["unc_median"], 0, np.where(unc <= cal["unc_q75"], 1, 2))
    t = np.where(lat > cal["latent_q90"], 2, t)
    t = np.where((lat > cal["latent_q75"]) & (t == 0), 1, t)
    if rule:
        t = np.where((pm > P_THRESHOLD) & (t == 0), 1, t)
    return t


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="calibrated ensemble checkpoint (.pt)")
    ap.add_argument("--data", required=True, help="alignn_data_exp: splits/A0.json, A2.json, experiment.json")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    a0 = json.load(open(os.path.join(args.data, "splits", "A0.json")))
    a2 = json.load(open(os.path.join(args.data, "splits", "A2.json")))
    exp = json.load(open(os.path.join(args.data, "experiment.json")))
    stable_tr, meta_tr = list(a0["train"]), sorted(set(a2["train"]) - set(a0["train"]))
    t_meta = exp["tests"]["T_meta"]
    ens_split = os.path.splitext(args.weights)[0] + ".split.json"
    if os.path.exists(ens_split):
        trained = set(json.load(open(ens_split))["train"])
        if not set(stable_tr + meta_tr) <= trained:
            raise SystemExit("these weights were not trained on both populations; the head "
                             "would be fitted on structures the ensemble never saw")
    truth = {}
    for line in open(os.path.join(args.data, "id_prop.csv")):
        f, y = line.strip().split(",")[:2]
        truth[f] = float(y)
    roles = ([("train_stable", f) for f in stable_tr] + [("train_meta", f) for f in meta_tr]
             + [("val", f) for f in a0["val"]] + [("test", f) for f in a0["test"]]
             + [("T_meta", f) for f in t_meta] + [("T_2dmp", f) for f in exp["tests"]["T_2dmp"]])

    stage = os.path.join(ROOT, ".calib_pop_tmp")
    os.makedirs(stage, exist_ok=True)
    shutil.copy(args.weights, os.path.join(stage, "cgcnn_2d_ensemble.pt"))
    P = Predictor(stage, verbose=False)
    shutil.rmtree(stage, ignore_errors=True)
    if len(P.models) < 2 or "unc_q75" not in P.cal or "latent_q90" not in P.cal:
        raise SystemExit("needs an ensemble already calibrated by calibrate_uncertainty.py")
    cal = P.cal

    t0 = time.time()
    graphs, keep = [], []
    for i, (_, f) in enumerate(roles):
        g = P.graph(read_structure(os.path.join(args.data, f)))
        if g is not None:
            graphs.append(g)
            keep.append(i)
    roles = [roles[i] for i in keep]
    P.type_model = P.metal_model = None      # not needed here, and would sit on the CPU
    for m in P.models:
        m.to(device)
    with torch.no_grad():
        gap, unc, _, _, emb = P._forward_many(graphs, args.batch, device, 0)
        lat = np.concatenate([P.latent_distance(graphs[i:i + args.batch])
                              for i in range(0, len(graphs), args.batch)])
    print(f"{len(graphs)} structures embedded in {time.time() - t0:.0f}s on {device}")
    R = np.array([r for r, _ in roles])
    files = [f for _, f in roles]
    err = np.abs(gap - np.array([truth[f] for f in files]))
    ratio = err / np.maximum(unc, 1e-6)

    # 1. the head
    tr = np.isin(R, ["train_stable", "train_meta"])
    sc = StandardScaler().fit(emb[tr])
    lr = LogisticRegression(C=0.001, max_iter=5000).fit(sc.transform(emb[tr]), R[tr] == "train_meta")
    w = lr.coef_[0] / sc.scale_
    b = float(lr.intercept_[0] - np.dot(lr.coef_[0], sc.mean_ / sc.scale_))
    pm = 1 / (1 + np.exp(-(emb.astype(np.float64) @ w + b)))
    held = np.isin(R, ["val", "test", "T_meta"])
    auc = float(roc_auc_score(R[held] == "T_meta", pm[held]))
    auc_unc = float(roc_auc_score(R[held] == "T_meta", unc[held]))
    auc_lat = float(roc_auc_score(R[held] == "T_meta", lat[held]))
    print(f"\n1. metastability head, held-out ROC-AUC (validation + test against T_meta): {auc:.3f}"
          f"   - spread alone {auc_unc:.3f}, latent distance alone {auc_lat:.3f}")
    print(f"   median P: near-hull test {np.median(pm[R == 'test']):.2f}, T_meta {np.median(pm[R == 'T_meta']):.2f}")

    # 2. + 3. repeated halvings of T_meta by composition
    edges = [cal["unc_q25"], cal["unc_median"], cal["unc_q75"]]
    s90 = cal["scale90"]
    iv, it, im = np.where(R == "val")[0], np.where(R == "test")[0], np.where(R == "T_meta")[0]
    groups = np.array([exp["formula"][files[i]] for i in im])
    rec = {k: [] for k in ("cov_test_old", "cov_test_new", "cov_meta_old", "cov_meta_new",
                           "w_test_old", "w_test_new", "w_meta_old", "w_meta_new")}
    quart = {"old": [], "new": []}          # coverage per spread quartile on the test split
    for rep in range(args.repeats):
        a, bb = next(GroupShuffleSplit(1, test_size=0.5, random_state=rep).split(im, groups=groups))
        fit = np.concatenate([iv, im[a]])
        table = fit_table(ratio[fit], pm[fit], unc[fit], edges, s90)
        for name, idx in (("test", it), ("meta", im[bb])):
            s = apply_table(table, pm[idx], unc[idx], edges)
            rec[f"cov_{name}_old"].append(np.mean(ratio[idx] <= s90))
            rec[f"cov_{name}_new"].append(np.mean(ratio[idx] <= s))
            rec[f"w_{name}_old"].append(np.median(s90 * unc[idx]))
            rec[f"w_{name}_new"].append(np.median(s * unc[idx]))
            if name == "test":
                q = np.digitize(unc[idx], edges)
                quart["old"].append([np.mean(ratio[idx][q == k] <= s90) for k in range(4)])
                quart["new"].append([np.mean(ratio[idx][q == k] <= s[q == k]) for k in range(4)])
    ms = lambda k: (float(np.mean(rec[k])), float(np.std(rec[k])))
    print(f"\n2. 90% interval, {args.repeats} halvings of T_meta by composition "
          f"(fit on validation + one half, report on the other half and on the test split)")
    print(f"   {'':26s} {'one scale x%.2f' % s90:>22s} {'per cell':>22s}")
    for name, label in (("test", "near-hull test split"), ("meta", "held-out metastable half")):
        o, n = ms(f"cov_{name}_old"), ms(f"cov_{name}_new")
        wo, wn = ms(f"w_{name}_old"), ms(f"w_{name}_new")
        print(f"   {label:26s} {100 * o[0]:5.1f}% +-{100 * o[1]:.1f}  ({wo[0]:.2f} eV) "
              f"{100 * n[0]:5.1f}% +-{100 * n[1]:.1f}  ({wn[0]:.2f} eV)")
    for k in ("old", "new"):
        print(f"   test split by spread quartile, {k:3s}: " +
              " / ".join(f"{100 * v:.0f}%" for v in np.mean(quart[k], axis=0)))
    table = fit_table(ratio[np.concatenate([iv, im])], pm[np.concatenate([iv, im])],
                      unc[np.concatenate([iv, im])], edges, s90)
    i2 = np.where(R == "T_2dmp")[0]
    s2 = apply_table(table, pm[i2], unc[i2], edges)
    cov2 = (float(np.mean(ratio[i2] <= s90)), float(np.mean(ratio[i2] <= s2)))
    print(f"   2DMatPedia test set (n={len(i2)}, in no fit here, another database): "
          f"{100 * cov2[0]:.1f}% -> {100 * cov2[1]:.1f}%")
    print("   final table (validation + all of T_meta), rows P<=0.5 / P>0.5, columns spread quartiles:")
    for r, cells in zip(("near-hull-like", "metastable-like"), table):
        print(f"     {r:16s} " + "  ".join(f"x{c:.2f}" for c in cells))

    print("\n3. verdict tiers, MAE (share), without and with a demotion rule (the rule is NOT written)")
    tier_rep = {}
    for name, idx in (("test", it), ("T_meta", im), ("T_2dmp", i2)):
        for rule in (False, True):
            t = tiers(unc[idx], lat[idx], pm[idx], cal, rule)
            cells = {k: (float(err[idx][t == j].mean()) if (t == j).any() else None, float(np.mean(t == j)))
                     for j, k in enumerate(("reliable", "check", "out_of_domain"))}
            tier_rep[f"{name}_{'rule' if rule else 'before'}"] = cells
            print(f"   {name:7s} {'rule  ' if rule else 'before'}  " + "  ".join(
                f"{k} {v[0]:.3f} ({100 * v[1]:.0f}%)" for k, v in cells.items() if v[0] is not None))

    # the typical error of each tier as the full verdict assigns it, on the test split
    new_tier_mae = {k: v[0] for k, v in tier_rep["test_before"].items()}
    population = {
        "w": torch.tensor(w, dtype=torch.float32), "b": b, "threshold": P_THRESHOLD,
        "fitted_on": f"logistic regression on the concatenated member embeddings, "
                     f"{len(stable_tr)} near-hull against {len(meta_tr)} metastable training structures",
        "auc_heldout": auc, "auc_spread": auc_unc, "auc_latent": auc_lat,
        "scale90_table": {"p_threshold": P_THRESHOLD, "unc_edges": [float(e) for e in edges], "scale": table,
                          "fitted_on": "validation split + T_meta (held-out metastable)"},
        "report": {"repeats": args.repeats, **{k: ms(k) for k in rec}, "tiers": tier_rep,
                   "test_coverage_by_spread_quartile": {k: [float(x) for x in np.mean(v, axis=0)]
                                                        for k, v in quart.items()},
                   "t_2dmp_coverage": {"one_scale": cov2[0], "per_cell": cov2[1], "n": int(len(i2))},
                   "tier_mae_spread_only": dict(cal["tier_mae"])},
    }
    if args.dry_run:
        print("\n--dry-run: checkpoint not modified")
        return
    ck = torch.load(args.weights, map_location="cpu")
    ck["population"] = population
    ck["calibration"].pop("population_threshold", None)
    ck["calibration"]["tier_mae"] = new_tier_mae
    torch.save(ck, args.weights)
    print(f"\nwritten the head and the interval table into {args.weights}; "
          f"tier errors {', '.join(f'{k} {v:.3f}' for k, v in new_tier_mae.items())}")


if __name__ == "__main__":
    main()
