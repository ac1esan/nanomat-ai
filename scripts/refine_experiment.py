#!/usr/bin/env python
"""Three changes to the shipped training recipe, each tested alone against one control.

  C  control: the shipped recipe (mc2d_experiment.py arm E1) - its data, its split,
     five seeds, the per-atom angular descriptor
  F  C without the training labels data/reference_flags.csv marks: four known wrong
     (graphene, planar silicene, BP) and the "disputed" ones, where Alexandria labels
     one structure differently in different cells. Hygiene: the question is whether
     it costs anything
  L  C plus layers from the full C2DB release with no element heavier than Sb
     (Z < 52), non-magnetic semiconductors, outside the JARVIS mirror of C2DB (which
     the project keeps as an external check), with no structure twin in the training
     data and no formula any test set holds. Label: C2DB's PBE gap minus its own
     spin-orbit shift of the direct gap (median -0.016 eV for these). A fifth of
     their formulas is held out by formula as T_c2db
  T  C with the three-body edge update (train_cgcnn.py --tri 9)

Test sets: T_alex (the shipped test), T_meta, T_2dmp, T_far, T_vfar, T_mc2d, T_c2db.
Labels data/reference_flags.csv marks are left out of every score, in every arm.

Decision rule, written before any arm was trained (17 October 2026). Against C from
the same session, paired on the same structures, 95% bootstrap intervals:
  1. every arm: on T_alex the MAE difference is at most +0.005 eV and the upper end
     of its interval below +0.015;
  2. L: T_c2db is better, the whole interval below zero;
     T: at least one of T_alex, T_meta, T_2dmp, T_far, T_vfar, T_mc2d is better, the
        whole interval below zero;
     F: nothing - it passes on 1 and 3;
  3. every arm: no other test set has its whole interval above zero.
Reported, not voting: light-element slices; the polymorph contrast on C2DB
(scripts/polymorph_sensitivity.py) for T; seconds per epoch.
If more than one arm passes, what ships is their combination, trained in a follow-up
run and held to 1 and 3 against C plus each passing arm's own condition 2.

    python scripts/hull_1ev_experiment.py prepare   # alignn_data_hull1 (C1)
    python scripts/mc2d_experiment.py prepare       # alignn_data_mc2dexp (E1 = shipped)
    python scripts/refine_experiment.py prepare     # alignn_data_refine + splits
    python scripts/refine_experiment.py cache
    ... four train_cgcnn.py runs, printed by prepare ...
    python scripts/refine_experiment.py evaluate --device cuda

C2DB: CC BY-NC 4.0; Haastrup et al., 2D Mater. 5, 042002 (2018); Gjerding et al.,
2D Mater. 8, 044002 (2021).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import hull_experiment as H  # noqa: E402

PREV = os.path.join(ROOT, "alignn_data_mc2dexp")
OUT = os.path.join(ROOT, "alignn_data_refine")
RUNS = os.path.join(ROOT, "runs", "refine")
FLAGS = os.path.join(ROOT, "data", "reference_flags.csv")
MIRROR = (os.path.join(ROOT, "alignn_data_c2db", "id_prop.csv"),
          os.path.join(ROOT, "data_c2db_wf", "id_prop.csv"))
HEAVY_Z = 52
HOLD_OUT = 0.2
N_TRI = 9
ARMS = {"C": {}, "F": {"split": "F"}, "L": {"split": "L"}, "T": {"tri": N_TRI}}
GAIN = {"L": ("T_c2db",), "T": ("T_alex", "T_meta", "T_2dmp", "T_far", "T_vfar", "T_mc2d"),
        "F": ()}
TESTS = ("T_alex", "T_meta", "T_2dmp", "T_far", "T_vfar", "T_mc2d", "T_c2db")


def _standardize(item):
    from audit_2dmatpedia import standardize
    key, path = item
    from nanomat.predict import read_structure
    try:
        return key, standardize(read_structure(path))
    except Exception:
        return key, None


def _std_struct(item):
    from audit_2dmatpedia import standardize
    key, st = item
    try:
        return key, standardize(st)
    except Exception:
        return key, None


def mirror_ids() -> set[str]:
    ids = set()
    for p in MIRROR:
        if os.path.exists(p):
            ids |= {l.split(",")[0].rsplit(".", 1)[0] for l in open(p)}
    try:
        from jarvis.db.figshare import data as jarvis_data
        ids |= {str(e.get("id") or e.get("jid")) for e in jarvis_data("c2db")}
    except Exception as exc:
        print(f"  (JARVIS c2db mirror list not loaded: {exc}; using local folders only)")
    return ids


def prepare(args):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from sklearn.model_selection import GroupShuffleSplit
    from audit_2dmatpedia import MATCHER, same_structure
    from c2db_full import records, to_structure
    from nanomat.graph import ensure_vacuum, to_graph

    prev = json.load(open(os.path.join(PREV, "experiment.json")))
    e1 = json.load(open(os.path.join(PREV, "splits", "E1.json")))
    shipped = json.load(open(os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.split.json")))
    if sorted(e1["train"]) != sorted(shipped["train"]):
        raise SystemExit("alignn_data_mc2dexp/splits/E1.json is not the shipped split; rebuild it")
    labels = H.id_prop(PREV)
    formula, elements = dict(prev["formula"]), dict(prev["elements"])
    tests = dict(prev["tests"])
    held = ({formula[f] for f in e1["val"] + e1["test"]}
            | {formula[f] for v in tests.values() for f in v})
    trained = {formula[f] for f in e1["train"]}

    # F: flagged training labels
    flagged = {r["id"] + ".vasp": r["flag"] for r in csv.DictReader(open(FLAGS))}
    drop = [f for f in e1["train"] if f in flagged]
    print(f"flags: {len(flagged)} labels flagged, {len(drop)} of them in training "
          f"({Counter(flagged[f] for f in drop)})")

    # L: C2DB additions
    mirror = mirror_ids()
    cands, skip = [], Counter()
    for r in records():
        st = to_structure(r)
        zmax = max(site.specie.Z for site in st)
        if zmax >= HEAVY_Z:
            skip["element with Z >= 52"] += 1
            continue
        if r["olduid"] in mirror:
            skip["in the JARVIS mirror"] += 1
            continue
        kv = r["kv"]
        y = kv["gap"] - (kv["gap_dir"] - kv["gap_dir_nosoc"])
        if y <= H.METAL_GAP:
            skip["not a semiconductor without SOC"] += 1
            continue
        if to_graph(ensure_vacuum(st)[0]) is None:
            skip["no graph"] += 1
            continue
        f = f"c2db-{r['uid']}.vasp"
        fm = st.composition.reduced_formula
        if fm in held:
            skip["formula held out by a test set"] += 1
            continue
        formula[f], elements[f], labels[f] = fm, sorted({s.symbol for s in st.composition}), float(y)
        cands.append((f, st))
    print(f"C2DB candidates before the twin check: {len(cands)}; skipped {dict(skip)}")

    # twins: a candidate whose structure the training set already holds adds nothing
    by_formula = {}
    for f in e1["train"]:
        by_formula.setdefault(formula[f], []).append(f)
    need = list(dict.fromkeys((g, os.path.join(PREV, g)) for c, _ in cands
                              for g in by_formula.get(formula[c], [])))
    with Pool(args.workers) as pool:
        ref = dict(pool.imap_unordered(_standardize, need, chunksize=8))
        mine = dict(pool.imap_unordered(_std_struct, cands, chunksize=8))
    sm = StructureMatcher(**MATCHER)
    twins = set()
    for f, _ in cands:
        s = mine.get(f)
        if s is None:
            continue
        for g in by_formula.get(formula[f], []):
            t = ref.get(g)
            if t is not None and same_structure(sm, s, t):
                twins.add(f)
                break
    cands = [(f, st) for f, st in cands if f not in twins]
    print(f"  structure twins of training entries: {len(twins)} -> {len(cands)} candidates")

    gss = GroupShuffleSplit(n_splits=1, test_size=HOLD_OUT, random_state=0)
    _, te = next(gss.split(np.arange(len(cands)), groups=[formula[f] for f, _ in cands]))
    test_formulas = {formula[cands[i][0]] for i in te}
    t_c2db = sorted(f for f, _ in cands if formula[f] in test_formulas)
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
    tests["T_c2db"] = t_c2db
    used = list(dict.fromkeys(e1["train"] + e1["val"] + e1["test"]
                              + [f for v in tests.values() for f in v] + adds))
    with open(os.path.join(OUT, "id_prop.csv"), "w") as fh:
        for f in used:
            fh.write(f"{f},{labels[f]}\n")
    splits = {"C": list(e1["train"]), "F": [f for f in e1["train"] if f not in flagged],
              "L": list(e1["train"]) + adds}
    counts = {}
    for arm, train in splits.items():
        json.dump({"train": train, "val": e1["val"], "test": e1["test"]},
                  open(os.path.join(OUT, "splits", f"{arm}.json"), "w"))
        c = Counter(el for f in train for el in elements[f] if el in H.LIGHT)
        counts[arm] = {"n_train": len(train), **{el: c[el] for el in H.LIGHT}}
    seen = {f: formula[f] in trained for v in tests.values() for f in v}
    json.dump({"tests": tests, "formula": {f: formula[f] for f in used},
               "elements": {f: elements[f] for f in used}, "seen_in_C": seen,
               "flagged": sorted(f for f in used if f in flagged), "train_counts": counts},
              open(os.path.join(OUT, "experiment.json"), "w"))

    print(f"\n{'arm':4s} {'train':>7s}   " + "  ".join(f"{el:>5s}" for el in H.LIGHT))
    for arm, c in counts.items():
        print(f"{arm:4s} {c['n_train']:7d}   " + "  ".join(f"{c[el]:5d}" for el in H.LIGHT))
    print("\ntest sets: " + ", ".join(f"{k} {len(v)}" for k, v in tests.items()))
    print(f"written {OUT} ({len(used)} structures)\n")
    rel = os.path.relpath(OUT, ROOT)
    for arm, spec in ARMS.items():
        print(f"python train_cgcnn.py --data {rel} --split-file {rel}/splits/{spec.get('split', 'C')}.json "
              f"--epochs 200 --batch 64 --ensemble 5 --angles {H.N_ANG}"
              + (f" --tri {spec['tri']}" if spec.get("tri") else "")
              + f" --cache --workers 4 --out runs/refine/{arm}.pt")


def cache(args):
    import pandas as pd
    from train_cgcnn import build_graphs
    files = pd.read_csv(os.path.join(args.data, "id_prop.csv"), header=None)[0].tolist()
    for n_tri in (0, N_TRI):
        build_graphs(args.data, files, 8.0, True, H.N_ANG, n_tri)
    print(f"caches ready: {len(files)} graphs, with and without triplets")


def evaluate(args):
    from nanomat import Predictor
    from nanomat.predict import read_structure

    exp = json.load(open(os.path.join(args.data, "experiment.json")))
    truth = H.id_prop(args.data)
    flagged = set(exp["flagged"])
    arms = [a for a in ARMS if os.path.exists(os.path.join(args.runs, f"{a}.pt"))]
    if "C" not in arms:
        raise SystemExit("the control C is required: every comparison is paired against it")
    tests = {k: [f for f in exp["tests"][k] if f not in flagged] for k in TESTS}
    structures = {k: [(f, read_structure(os.path.join(args.data, f))) for f in v]
                  for k, v in tests.items()}
    pred = {}
    for arm in arms:
        stage = os.path.join(ROOT, f".refine_{arm}")
        os.makedirs(stage, exist_ok=True)
        shutil.copy(os.path.join(args.runs, f"{arm}.pt"), os.path.join(stage, "cgcnn_2d_ensemble.pt"))
        P = Predictor(stage, verbose=False)
        pred[arm] = {k: {key: r.gap for key, r in P.run_many(v, batch_size=256, device=args.device)
                         if r is not None} for k, v in structures.items()}
        shutil.rmtree(stage)
        print(f"{arm}: predicted " + ", ".join(f"{k} {len(v)}" for k, v in pred[arm].items()))

    rng = np.random.default_rng(0)
    report = {"excluded_flagged": len(flagged)}
    for k in TESTS:
        files = [f for f in tests[k] if all(f in pred[a][k] for a in arms)]
        y = np.array([truth[f] for f in files])
        subsets = {"all": np.ones(len(files), bool),
                   "formula unseen": np.array([not exp["seen_in_C"][f] for f in files])}
        for el in H.LIGHT:
            subsets[f"has {el}"] = np.array([el in exp["elements"][f] for f in files])
        report[k] = {}
        print(f"\n{k} (n={len(files)})   MAE, and the paired difference to C")
        for name, sel in subsets.items():
            if sel.sum() < 10:
                continue
            e0 = np.abs(np.array([pred["C"][k][f] for f in files]) - y)[sel]
            row, cells = {"n": int(sel.sum()), "C": {"mae": float(e0.mean())}}, [f"C {e0.mean():.3f}"]
            for a in arms:
                if a == "C":
                    continue
                e = np.abs(np.array([pred[a][k][f] for f in files]) - y)[sel]
                d = e - e0
                boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                row[a] = {"mae": float(e.mean()), "diff": float(d.mean()), "ci95": [float(lo), float(hi)]}
                cells.append(f"{a} {e.mean():.3f} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
            report[k][name] = row
            print(f"  {name:14s} {int(sel.sum()):5d}  " + "   ".join(cells))

    print("\ndecision rule (written 17 October 2026, before training):")
    passed = []
    for a in arms:
        if a == "C":
            continue
        t = report["T_alex"]["all"][a]
        ok1 = t["diff"] <= 0.005 and t["ci95"][1] < 0.015
        ok2 = (not GAIN[a]) or any(report[k]["all"][a]["ci95"][1] < 0 for k in GAIN[a])
        ok3 = all(report[k]["all"][a]["ci95"][0] <= 0 for k in TESTS if k != "T_alex")
        print(f"  {a}: 1. T_alex {t['diff']:+.3f} [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] -> {ok1}   "
              f"2. {'gain on ' + ', '.join(GAIN[a]) if GAIN[a] else 'not required'} -> {ok2}   "
              f"3. no other set worse -> {ok3}")
        if ok1 and ok2 and ok3:
            passed.append(a)
    report["passed"] = passed
    print(f"  passed: {passed or 'none'}"
          + ("  -> the combination is trained next and held to the same rule" if len(passed) > 1 else ""))
    json.dump(report, open(os.path.join(args.runs, "evaluation.json"), "w"), indent=1)
    print(f"\nwritten {os.path.join(args.runs, 'evaluation.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "cache", "evaluate"])
    ap.add_argument("--data", default=OUT)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    {"prepare": prepare, "cache": cache, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
