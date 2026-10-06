#!/usr/bin/env python
"""Past 0.2 eV/atom: does more metastable Alexandria data still help?

stability_experiment.py showed that adding Alexandria monolayers at 0.1-0.2 eV/atom
above the hull (with non-magnetic 2DMatPedia) improved every test population, the
near-hull one included: 0.249 -> 0.225 on the shipped test. That is the shipped
ensemble (A3). Alexandria holds about 24k more semiconductors at 0.2-0.5 eV/atom,
with roughly as many C, B, N and H structures again as the whole current training
set. This asks whether the gain continues or turns.

Three arms, one environment, the shipped validation and test in every arm:
  B0  the shipped training set (A3)                               control
  B1  + Alexandria semiconductors at 0.2-0.3 eV/atom
  B2  + Alexandria semiconductors at 0.2-0.5 eV/atom

Test sets: the shipped test (T_alex), the two held-out sets of the last experiment
(T_meta, T_2dmp: their formulas stay out of every arm), and a new one, T_far, cut
from the additions by formula among formulas no arm has trained on.

    python scripts/hull_experiment.py prepare      # on the Mac
    python scripts/hull_experiment.py cache        # once on the GPU box
    ... three train_cgcnn.py runs, printed by prepare ...
    python scripts/hull_experiment.py evaluate

Decision rule, written before any arm was trained (6 October 2026). Against B0 from
the same session, paired on the same structures, 95% bootstrap intervals:
  1. The near-hull population is not worse: on T_alex the MAE difference is at most
     +0.005 eV and the upper end of its interval is below +0.015.
  2. Some population is better: on T_meta, T_2dmp or T_far the difference is
     negative with the whole interval below zero.
Of the arms that pass both, the one with the largest summed gain over T_meta, T_2dmp
and T_far; if the two arms' sums are within 0.01 eV of each other, the smaller arm.
If no arm passes, the shipped ensemble stays and the result is written up as the
point where metastable data stops paying.
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

PREV = os.path.join(ROOT, "alignn_data_exp")             # the stability experiment's folder
OUT = os.path.join(ROOT, "alignn_data_hull")
RUNS = os.path.join(ROOT, "runs", "hull")
BANDS = {"b03": (0.2, 0.3), "b05": (0.3, 0.5)}           # (low, high], eV/atom
ARMS = {"B0": (), "B1": ("b03",), "B2": ("b03", "b05")}
EXTRA_TEST = 0.2      # share of the new-to-every-arm formulas held out as T_far
METAL_GAP = 0.01      # the export's threshold: at or below is a metal
LIGHT = ("C", "N", "B", "H", "O")
N_ANG = 9


def id_prop(folder: str) -> dict[str, float]:
    out = {}
    for line in open(os.path.join(folder, "id_prop.csv")):
        f, y = line.strip().split(",")[:2]
        out[f] = float(y)
    return out


def prepare(args):
    from jarvis.core.atoms import Atoms
    from jarvis.db.figshare import data
    from pymatgen.core import Composition
    from sklearn.model_selection import GroupShuffleSplit
    from nanomat.graph import ensure_vacuum, to_graph

    prev = json.load(open(os.path.join(PREV, "experiment.json")))
    a3 = json.load(open(os.path.join(PREV, "splits", "A3.json")))
    labels = id_prop(PREV)
    formula, elements = dict(prev["formula"]), dict(prev["elements"])
    # formulas already held out somewhere: they may enter no training part
    held = {formula[f] for f in a3["val"] + a3["test"] + prev["tests"]["T_meta"]
            + prev["tests"]["T_2dmp"]}
    trained = {formula[f] for f in a3["train"]}

    alex = data("alex_pbe_2d_all")
    band_of, add = {}, []
    n_skip = Counter()
    for e in alex:
        eh, gap = e.get("e_above_hull"), e.get("band_gap_ind")
        if eh is None or gap is None:
            continue
        band = next((b for b, (lo, hi) in BANDS.items() if lo < eh <= hi), None)
        if band is None:
            continue
        if gap <= METAL_GAP:
            n_skip["metal"] += 1
            continue
        fm = Composition(e["formula"]).reduced_formula
        if fm in held:
            n_skip["formula held out elsewhere"] += 1
            continue
        f = e["mat_id"] + ".vasp"
        band_of[f] = band
        formula[f], elements[f] = fm, sorted(set(e["atoms"]["elements"]))
        labels[f] = float(gap)
        add.append((f, e["atoms"]))
    print(f"additions: {len(add)} ({Counter(band_of.values())}); skipped {dict(n_skip)}")

    # write the structures; one with no neighbour inside the cutoff cannot be named
    # in a split (the trainer refuses such a split rather than guess)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for f in labels:
        if os.path.exists(os.path.join(PREV, f)):
            os.link(os.path.join(PREV, f), os.path.join(OUT, f))
    kept = []
    t = time.time()
    for i, (f, atoms) in enumerate(add):
        a = Atoms.from_dict(atoms)
        if to_graph(ensure_vacuum(a.pymatgen_converter())[0]) is None:
            n_skip["no graph"] += 1
            continue
        with open(os.path.join(OUT, f), "w") as fh:
            fh.write(a.get_string())
        kept.append(f)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(add)} written, {time.time() - t:.0f}s")
    if n_skip["no graph"]:
        print(f"  {n_skip['no graph']} additions have no graph and are left out")

    # T_far: whole formulas, drawn only from formulas no arm has trained on
    fresh = [f for f in kept if formula[f] not in trained]
    gss = GroupShuffleSplit(n_splits=1, test_size=EXTRA_TEST, random_state=0)
    _, te = next(gss.split(np.arange(len(fresh)), groups=[formula[f] for f in fresh]))
    t_far = sorted(fresh[i] for i in te)
    far_formulas = {formula[f] for f in t_far}
    parts = {b: [f for f in kept if band_of[f] == b and formula[f] not in far_formulas]
             for b in BANDS}
    tests = {"T_alex": a3["test"], "T_meta": prev["tests"]["T_meta"],
             "T_2dmp": prev["tests"]["T_2dmp"], "T_far": t_far}
    # T_meta and T_2dmp keep the last experiment's guarantee - no addition shares their
    # formulas - but about 40% of T_meta are polymorphs of formulas in the base training
    # set (that guarantee never covered the base set). T_far is disjoint from everything.
    added = {formula[f] for p in parts.values() for f in p}
    for k in ("T_meta", "T_2dmp"):
        assert not ({formula[f] for f in tests[k]} & added), k
    assert not ({formula[f] for f in t_far} & (trained | added)), "T_far"
    seen = {f: formula[f] in trained for k in tests for f in tests[k]}

    used = list(dict.fromkeys(a3["train"] + a3["val"] + a3["test"] + tests["T_meta"]
                              + tests["T_2dmp"] + parts["b03"] + parts["b05"] + t_far))
    with open(os.path.join(OUT, "id_prop.csv"), "w") as fh:
        for f in used:
            fh.write(f"{f},{labels[f]}\n")
    os.makedirs(os.path.join(OUT, "splits"), exist_ok=True)
    counts = {}
    for arm, adds in ARMS.items():
        train = list(a3["train"]) + [f for b in adds for f in parts[b]]
        json.dump({"train": train, "val": a3["val"], "test": a3["test"]},
                  open(os.path.join(OUT, "splits", f"{arm}.json"), "w"))
        c = Counter(el for f in train for el in elements[f] if el in LIGHT)
        counts[arm] = {"n_train": len(train), **{el: c[el] for el in LIGHT}}
    source = {f: ("2dmatpedia" if f.startswith("2dm") else "alexandria") for f in used}
    json.dump({"tests": tests, "formula": {f: formula[f] for f in used},
               "elements": {f: elements[f] for f in used}, "source": source,
               "band": {f: band_of[f] for f in t_far}, "seen_in_B0": seen,
               "train_counts": counts},
              open(os.path.join(OUT, "experiment.json"), "w"))

    print(f"\n{'arm':4s} {'train':>7s}   " + "  ".join(f"{el:>5s}" for el in LIGHT))
    for arm, c in counts.items():
        print(f"{arm:4s} {c['n_train']:7d}   " + "  ".join(f"{c[el]:5d}" for el in LIGHT))
    print("\ntest sets: " + ", ".join(f"{k} {len(v)}" for k, v in tests.items()))
    print(f"  T_far by band: {dict(Counter(band_of[f] for f in t_far))}")
    print(f"\nwritten {OUT} ({len(used)} structures)")


def cache(args):
    """Build the graph cache once; three runs building it at once would race."""
    import pandas as pd
    from train_cgcnn import build_graphs
    files = pd.read_csv(os.path.join(args.data, "id_prop.csv"), header=None)[0].tolist()
    t = time.time()
    build_graphs(args.data, files, 8.0, True, N_ANG)
    print(f"cache ready: {len(files)} graphs in {time.time() - t:.0f}s")


def evaluate(args):
    from scipy.stats import wilcoxon
    from nanomat import Predictor
    from nanomat.predict import read_structure

    exp = json.load(open(os.path.join(args.data, "experiment.json")))
    truth = id_prop(args.data)
    arms = [a for a in ARMS if os.path.exists(os.path.join(args.runs, f"{a}.pt"))]
    if "B0" not in arms:
        raise SystemExit("the control B0 is required: every comparison is paired against it")
    structures = {k: [(f, read_structure(os.path.join(args.data, f))) for f in v]
                  for k, v in exp["tests"].items()}
    pred = {}
    for arm in arms:
        stage = os.path.join(ROOT, f".hull_{arm}")
        os.makedirs(stage, exist_ok=True)
        shutil.copy(os.path.join(args.runs, f"{arm}.pt"), os.path.join(stage, "cgcnn_2d_ensemble.pt"))
        P = Predictor(stage, verbose=False)
        pred[arm] = {k: {key: r.gap for key, r in P.run_many(v, batch_size=256, device=args.device)
                         if r is not None}
                     for k, v in structures.items()}
        shutil.rmtree(stage)
        print(f"{arm}: predicted " + ", ".join(f"{k} {len(v)}" for k, v in pred[arm].items()))

    rng = np.random.default_rng(0)
    report = {}
    for k, files in exp["tests"].items():
        files = [f for f in files if all(f in pred[a][k] for a in arms)]
        y = np.array([truth[f] for f in files])
        subsets = {"all": np.ones(len(files), bool)}
        if k == "T_far":
            for b in BANDS:
                subsets[b] = np.array([exp["band"][f] == b for f in files])
        subsets["formula unseen"] = np.array([not exp["seen_in_B0"][f] for f in files])
        for el in LIGHT:
            subsets[f"has {el}"] = np.array([el in exp["elements"][f] for f in files])
        report[k] = {}
        print(f"\n{k} (n={len(files)})   MAE, and the paired difference to B0 "
              f"(negative = better than the control)")
        for name, sel in subsets.items():
            if sel.sum() < 10:
                continue
            e0 = np.abs(np.array([pred["B0"][k][f] for f in files]) - y)[sel]
            cells, row = [], {"n": int(sel.sum())}
            for a in arms:
                e = np.abs(np.array([pred[a][k][f] for f in files]) - y)[sel]
                if a == "B0":
                    cells.append(f"B0 {e.mean():.3f}")
                    row[a] = {"mae": float(e.mean())}
                    continue
                d = e - e0
                boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                p = float(wilcoxon(e, e0).pvalue) if np.any(d != 0) else 1.0
                cells.append(f"{a} {e.mean():.3f} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
                row[a] = {"mae": float(e.mean()), "diff": float(d.mean()),
                          "ci95": [float(lo), float(hi)], "wilcoxon_p": p}
            report[k][name] = row
            print(f"  {name:14s} {int(sel.sum()):5d}  " + "   ".join(cells))

    # the rule, applied mechanically; the write-up still looks at the numbers
    print("\ndecision rule:")
    passed = {}
    for a in arms:
        if a == "B0":
            continue
        t = report["T_alex"]["all"][a]
        ok1 = t["diff"] <= 0.005 and t["ci95"][1] < 0.015
        gains = {k: report[k]["all"][a] for k in ("T_meta", "T_2dmp", "T_far")}
        ok2 = any(g["diff"] < 0 and g["ci95"][1] < 0 for g in gains.values())
        total = -sum(g["diff"] for g in gains.values())
        print(f"  {a}: near-hull not worse {ok1}, some population better {ok2}, "
              f"summed gain {total:+.3f}")
        if ok1 and ok2:
            passed[a] = total
    if not passed:
        choice = "B0 (no arm passed)"
    else:
        best = max(passed, key=passed.get)
        within = [a for a in passed if passed[best] - passed[a] <= 0.01]
        choice = min(within, key=lambda a: len(ARMS[a]))
    print(f"  -> {choice}")
    report["decision"] = choice
    json.dump(report, open(os.path.join(args.runs, "evaluation.json"), "w"), indent=1)
    print(f"\nwritten {os.path.join(args.runs, 'evaluation.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "cache", "evaluate"])
    ap.add_argument("--data", default=OUT)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    {"prepare": prepare, "cache": cache, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
