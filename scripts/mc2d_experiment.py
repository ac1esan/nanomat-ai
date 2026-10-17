#!/usr/bin/env python
"""Does training on MC2D help the model on layers of crystals that exist?

scripts/audit_mc2d.py found MC2D's labels agree with Alexandria's to 0.070 eV on the
structures both hold - 0.057 eV once compounds of the magnetic 3d metals are left out,
against 0.374 for those - and that the shipped model does much worse on MC2D's layers
(MAE 0.51, 71% disowned) than on Alexandria's (0.19), worst on carbon, nitrogen and
boron, which it puts too low. MC2D holds 1 103 semiconductors with no Alexandria twin;
1 015 of them contain none of V, Cr, Mn, Fe, Co, Ni.

Two arms, one environment, the shipped validation and test in both:
  E0  the shipped training set (hull_1ev_experiment.py arm C1)          control
  E1  + MC2D semiconductors with no Alexandria twin and no magnetic 3d metal,
        minus the held-out formulas below

Test sets: the five of the last experiment (T_alex, T_meta, T_2dmp, T_far, T_vfar) and
T_mc2d - a fifth of those MC2D formulas, held out by formula, plus any whose formula
an earlier experiment already held out. None of T_mc2d's formulas is in E1's additions.

Decision rule, written before either arm was trained (11 October 2026). Against E0
from the same session, paired on the same structures, 95% bootstrap intervals:
  1. The near-hull population is not worse: on T_alex the MAE difference is at most
     +0.005 eV and the upper end of its interval is below +0.015.
  2. MC2D is better: on T_mc2d the difference is negative with the whole interval
     below zero. Unlike T_vfar last time, this set votes: layers of real crystals are
     the population the tool is for, and it is held out by formula.
  3. No other held-out set gets worse: none of T_meta, T_2dmp, T_far, T_vfar has its
     whole interval above zero.
If E1 fails, the shipped ensemble stays and MC2D remains an external check.

    python scripts/hull_1ev_experiment.py prepare   # rebuilds alignn_data_hull1 (C1)
    python scripts/audit_mc2d.py                    # runs/audit_mc2d_entries.csv
    python scripts/mc2d_experiment.py prepare
    python scripts/mc2d_experiment.py cache
    ... two train_cgcnn.py runs, printed by prepare ...
    python scripts/mc2d_experiment.py evaluate --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import hull_experiment as H  # noqa: E402

PREV = os.path.join(ROOT, "alignn_data_hull1")
OUT = os.path.join(ROOT, "alignn_data_mc2dexp")
RUNS = os.path.join(ROOT, "runs", "mc2dexp")
ENTRIES = os.path.join(ROOT, "runs", "audit_mc2d_entries.csv")
ARMS = {"E0": (), "E1": ("mc2d",)}
MAGNETIC_3D = {"V", "Cr", "Mn", "Fe", "Co", "Ni"}
HOLD_OUT = 0.2
GUARDS = ("T_meta", "T_2dmp", "T_far", "T_vfar")


def prepare(args):
    from sklearn.model_selection import GroupShuffleSplit
    from fetch_mc2d import load as load_mc2d, to_structure
    from nanomat.graph import ensure_vacuum, to_graph

    prev = json.load(open(os.path.join(PREV, "experiment.json")))
    c1 = json.load(open(os.path.join(PREV, "splits", "C1.json")))
    shipped = json.load(open(os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.split.json")))
    # C1 was shipped when this experiment ran; E1 (C1 + MC2D) is shipped since. Either
    # way the rebuilt arm must reproduce the shipped split exactly - checked below,
    # once E1 exists
    labels = H.id_prop(PREV)
    formula, elements = dict(prev["formula"]), dict(prev["elements"])
    held = ({formula[f] for f in c1["val"] + c1["test"]}
            | {formula[f] for v in prev["tests"].values() for f in v})
    trained = {formula[f] for f in c1["train"]}

    mc = {r["id"]: r for r in load_mc2d()}
    cands, skip = [], Counter()
    for r in csv.DictReader(open(ENTRIES)):
        if r["status"] == "matched":
            skip["has an Alexandria twin"] += 1
            continue
        if float(r["gap"]) <= H.METAL_GAP:
            skip["metal"] += 1
            continue
        els = set(r["elements"].split())
        if els & MAGNETIC_3D:
            skip["magnetic 3d metal"] += 1
            continue
        st = to_structure(mc[r["id"]])
        if to_graph(ensure_vacuum(st)[0]) is None:
            skip["no graph"] += 1
            continue
        f = f"mc2d-{r['id']}.vasp"
        formula[f], elements[f], labels[f] = r["formula"], sorted(els), float(r["gap"])
        cands.append((f, st))
    print(f"MC2D candidates: {len(cands)}; skipped {dict(skip)}")

    # T_mc2d: formulas an earlier experiment held out, plus a fifth of the rest
    free = [f for f, _ in cands if formula[f] not in held]
    gss = GroupShuffleSplit(n_splits=1, test_size=HOLD_OUT, random_state=0)
    _, te = next(gss.split(np.arange(len(free)), groups=[formula[f] for f in free]))
    test_formulas = {formula[free[i]] for i in te} | {formula[f] for f, _ in cands
                                                      if formula[f] in held}
    t_mc2d = sorted(f for f, _ in cands if formula[f] in test_formulas)
    adds = [f for f, _ in cands if formula[f] not in test_formulas]
    assert not ({formula[f] for f in adds} & held), "an addition shares a held-out formula"

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "splits"))
    for f in os.listdir(PREV):
        if f.endswith(".vasp"):
            os.link(os.path.join(PREV, f), os.path.join(OUT, f))
    for f, st in cands:
        st.to(filename=os.path.join(OUT, f), fmt="poscar")
    tests = dict(prev["tests"])
    tests["T_alex"] = c1["test"]
    tests["T_mc2d"] = t_mc2d
    used = list(dict.fromkeys(c1["train"] + c1["val"] + c1["test"]
                              + [f for v in tests.values() for f in v] + adds))
    with open(os.path.join(OUT, "id_prop.csv"), "w") as fh:
        for f in used:
            fh.write(f"{f},{labels[f]}\n")
    counts = {}
    for arm, extra in ARMS.items():
        train = list(c1["train"]) + (adds if extra else [])
        json.dump({"train": train, "val": c1["val"], "test": c1["test"]},
                  open(os.path.join(OUT, "splits", f"{arm}.json"), "w"))
        c = Counter(el for f in train for el in elements[f] if el in H.LIGHT)
        counts[arm] = {"n_train": len(train), **{el: c[el] for el in H.LIGHT}}
    rebuilt = {arm: sorted(json.load(open(os.path.join(OUT, "splits", f"{arm}.json")))["train"])
               for arm in ARMS}
    if sorted(shipped["train"]) not in rebuilt.values():
        raise SystemExit("neither rebuilt arm reproduces the shipped training split; "
                         "check alignn_data_hull1 and runs/audit_mc2d_entries.csv")
    seen = {f: formula[f] in trained for v in tests.values() for f in v}
    json.dump({"tests": tests, "formula": {f: formula[f] for f in used},
               "elements": {f: elements[f] for f in used}, "band": {},
               "seen_in_E0": seen, "train_counts": counts},
              open(os.path.join(OUT, "experiment.json"), "w"))

    print(f"\n{'arm':4s} {'train':>7s}   " + "  ".join(f"{el:>5s}" for el in H.LIGHT))
    for arm, c in counts.items():
        print(f"{arm:4s} {c['n_train']:7d}   " + "  ".join(f"{c[el]:5d}" for el in H.LIGHT))
    print("\ntest sets: " + ", ".join(f"{k} {len(v)}" for k, v in tests.items()))
    print(f"\nwritten {OUT} ({len(used)} structures)\n")
    for arm in ARMS:
        print(f"python train_cgcnn.py --data {os.path.relpath(OUT, ROOT)} --split-file "
              f"{os.path.relpath(OUT, ROOT)}/splits/{arm}.json --epochs 200 --batch 64 "
              f"--ensemble 5 --angles {H.N_ANG} --cache --workers 4 --out runs/mc2dexp/{arm}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "cache", "evaluate"])
    ap.add_argument("--data", default=OUT)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "cache":
        H.cache(args)
    else:
        H.evaluate(args, arms_def=ARMS, control="E0", gain_tests=("T_mc2d",),
                   seen_key="seen_in_E0", prefix="mc2dexp", guard_tests=GUARDS)


if __name__ == "__main__":
    main()
