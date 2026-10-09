#!/usr/bin/env python
"""One more step up the hull: Alexandria semiconductors at 0.5-1.0 eV/atom.

hull_experiment.py took the training data from 0.2 to 0.5 eV/atom above the hull and
every test population improved, the near-hull one included (0.230 -> 0.197 against
its own control). That is the shipped ensemble (B2). The gain per added structure
was shrinking (B0 -> B1 -0.020 on the near-hull test, B1 -> B2 -0.013), and the next
band is small: about 6k semiconductors on top of 40.5k. This asks whether it still
pays, and where.

Two arms, one environment, the shipped validation and test in both:
  C0  the shipped training set (B2)                               control
  C1  + Alexandria semiconductors at 0.5-1.0 eV/atom

Test sets: the four of the last experiment (T_alex, T_meta, T_2dmp, T_far: their
formulas stay out of the additions) and a new one, T_vfar, cut by formula from the
additions among formulas no arm has trained on.

    python scripts/hull_1ev_experiment.py prepare
    python scripts/hull_1ev_experiment.py cache        # once
    ... two train_cgcnn.py runs, printed by prepare ...
    python scripts/hull_1ev_experiment.py evaluate --device cuda

Decision rule, written before either arm was trained (9 October 2026). Against C0
from the same session, paired on the same structures, 95% bootstrap intervals:
  1. The near-hull population is not worse: on T_alex the MAE difference is at most
     +0.005 eV and the upper end of its interval is below +0.015.
  2. A population that existed before this step is better: on T_meta, T_2dmp or
     T_far the difference is negative with the whole interval below zero.
T_vfar is reported but cannot decide: an arm trained on its band will improve on it
almost by construction, and structures 0.5-1.0 eV/atom above the hull are rarely the
ones anybody screens. If C1 fails, the shipped ensemble stays and 0.5 eV/atom is
written up as where the added data stops paying for itself.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import hull_experiment as H  # noqa: E402

PREV = H.OUT                                              # alignn_data_hull
OUT = os.path.join(ROOT, "alignn_data_hull1")
RUNS = os.path.join(ROOT, "runs", "hull1")
BANDS = {"b10": (0.5, 1.0)}                               # (low, high], eV/atom
ARMS = {"C0": (), "C1": ("b10",)}
GAIN_TESTS = ("T_meta", "T_2dmp", "T_far")                # T_vfar reported, not deciding


def prepare(args):
    from jarvis.core.atoms import Atoms
    from jarvis.db.figshare import data
    from pymatgen.core import Composition
    from sklearn.model_selection import GroupShuffleSplit
    from nanomat.graph import ensure_vacuum, to_graph

    prev = json.load(open(os.path.join(PREV, "experiment.json")))
    b2 = json.load(open(os.path.join(PREV, "splits", "B2.json")))
    labels = H.id_prop(PREV)
    formula, elements = dict(prev["formula"]), dict(prev["elements"])
    held = ({formula[f] for f in b2["val"] + b2["test"]}
            | {formula[f] for v in prev["tests"].values() for f in v})
    trained = {formula[f] for f in b2["train"]}

    alex = data("alex_pbe_2d_all")
    band_of, add = dict(prev["band"]), []
    n_skip = Counter()
    for e in alex:
        eh, gap = e.get("e_above_hull"), e.get("band_gap_ind")
        if eh is None or gap is None:
            continue
        band = next((b for b, (lo, hi) in BANDS.items() if lo < eh <= hi), None)
        if band is None:
            continue
        if gap <= H.METAL_GAP:
            n_skip["metal"] += 1
            continue
        fm = Composition(e["formula"]).reduced_formula
        if fm in held:
            n_skip["formula held out elsewhere"] += 1
            continue
        f = e["mat_id"] + ".vasp"
        if f in labels:
            n_skip["already in the folder"] += 1
            continue
        band_of[f] = band
        formula[f], elements[f] = fm, sorted(set(e["atoms"]["elements"]))
        labels[f] = float(gap)
        add.append((f, e["atoms"]))
    print(f"additions: {len(add)}; skipped {dict(n_skip)}")

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for f in labels:
        if os.path.exists(os.path.join(PREV, f)):
            os.link(os.path.join(PREV, f), os.path.join(OUT, f))
    kept = []
    for f, atoms in add:
        a = Atoms.from_dict(atoms)
        if to_graph(ensure_vacuum(a.pymatgen_converter())[0]) is None:
            n_skip["no graph"] += 1
            continue
        with open(os.path.join(OUT, f), "w") as fh:
            fh.write(a.get_string())
        kept.append(f)
    if n_skip["no graph"]:
        print(f"  {n_skip['no graph']} additions have no graph and are left out")

    # T_vfar: whole formulas, drawn only from formulas no arm has trained on
    fresh = [f for f in kept if formula[f] not in trained]
    gss = GroupShuffleSplit(n_splits=1, test_size=H.EXTRA_TEST, random_state=0)
    _, te = next(gss.split(np.arange(len(fresh)), groups=[formula[f] for f in fresh]))
    t_vfar = sorted(fresh[i] for i in te)
    vfar_formulas = {formula[f] for f in t_vfar}
    part = [f for f in kept if formula[f] not in vfar_formulas]
    tests = dict(prev["tests"])
    tests["T_vfar"] = t_vfar
    added = {formula[f] for f in part}
    for k in ("T_meta", "T_2dmp", "T_far"):
        assert not ({formula[f] for f in tests[k]} & added), k
    assert not (vfar_formulas & (trained | added)), "T_vfar"
    seen = {f: formula[f] in trained for k in tests for f in tests[k]}

    used = list(dict.fromkeys(b2["train"] + b2["val"] + b2["test"]
                              + [f for k in tests for f in tests[k]] + part))
    with open(os.path.join(OUT, "id_prop.csv"), "w") as fh:
        for f in used:
            fh.write(f"{f},{labels[f]}\n")
    os.makedirs(os.path.join(OUT, "splits"), exist_ok=True)
    counts = {}
    for arm, adds in ARMS.items():
        train = list(b2["train"]) + (part if adds else [])
        json.dump({"train": train, "val": b2["val"], "test": b2["test"]},
                  open(os.path.join(OUT, "splits", f"{arm}.json"), "w"))
        c = Counter(el for f in train for el in elements[f] if el in H.LIGHT)
        counts[arm] = {"n_train": len(train), **{el: c[el] for el in H.LIGHT}}
    json.dump({"tests": tests, "formula": {f: formula[f] for f in used},
               "elements": {f: elements[f] for f in used},
               "band": {f: band_of[f] for f in used if f in band_of},
               "seen_in_C0": seen, "train_counts": counts},
              open(os.path.join(OUT, "experiment.json"), "w"))

    print(f"\n{'arm':4s} {'train':>7s}   " + "  ".join(f"{el:>5s}" for el in H.LIGHT))
    for arm, c in counts.items():
        print(f"{arm:4s} {c['n_train']:7d}   " + "  ".join(f"{c[el]:5d}" for el in H.LIGHT))
    print("\ntest sets: " + ", ".join(f"{k} {len(v)}" for k, v in tests.items()))
    print(f"\nwritten {OUT} ({len(used)} structures)\n")
    for arm in ARMS:
        print(f"python train_cgcnn.py --data {os.path.relpath(OUT, ROOT)} --split-file "
              f"{os.path.relpath(OUT, ROOT)}/splits/{arm}.json --epochs 200 --batch 64 --ensemble 5 "
              f"--angles {H.N_ANG} --cache --workers 4 --out runs/hull1/{arm}.pt")


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
        H.evaluate(args, arms_def=ARMS, control="C0", gain_tests=GAIN_TESTS,
                   seen_key="seen_in_C0", prefix="hull1")


if __name__ == "__main__":
    main()
