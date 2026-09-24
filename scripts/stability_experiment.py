#!/usr/bin/env python
"""Does the stability filter keep out exactly the data the model is weakest on?

The training set is Alexandria monolayers within 0.1 eV/atom of the hull. The rule
was adopted after a 26k set (0.2 eV/atom) scored worse than the 13k one - 0.262
against 0.215 - but each was scored on its own test split, and the 26k split was
half metastable, which is harder for any model: the shipped ensemble errs by 0.51 eV
there against 0.26 on stable structures. That comparison changed the population it
measured on, not only the data it trained on. It never isolated the question.

It matters because the chemistry the model is weakest on (C, N, B, H) is mostly
metastable. Of the light-element 2DMatPedia structures whose stability can be
estimated, few are below 0.1 eV/atom; and on 2DMatPedia the shipped model errs by
0.43 eV on stable structures against 0.82-0.88 on the rest (audit_2dmatpedia.py).

Four arms, one split, one environment; only the training part differs:
  A0  the shipped training set                                   control
  A1  + 2DMatPedia monolayers Alexandria does not have           non-magnetic, any stability
  A2  + Alexandria monolayers at 0.1-0.2 eV/atom                 the rule itself, one database
  A3  + both

Validation and test are the shipped ones in every arm, so A0 against A2 on the
shipped test is exactly "did the metastable data hurt the stable population". Two
more test sets are cut from the additions by composition, so that no arm has trained
on any of their formulas:
  T_meta  Alexandria at 0.1-0.2 eV/atom
  T_2dmp  2DMatPedia, non-magnetic

    python scripts/audit_2dmatpedia.py            # once: decides which 2DMatPedia entries are new
    python scripts/stability_experiment.py prepare
    python scripts/stability_experiment.py cache  # once on the GPU box, before the arms start
    ... four train_cgcnn.py runs, printed by prepare ...
    python scripts/stability_experiment.py evaluate
"""

from __future__ import annotations

import argparse
import csv
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

BASE = os.path.join(ROOT, "alignn_data_alex_2d")          # e_above_hull <= 0.1
WIDE = os.path.join(ROOT, "alignn_data_alex_2d_eh02")     # e_above_hull <= 0.2
SPLIT = os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.split.json")
AUDIT = os.path.join(ROOT, "runs", "audit_2dmatpedia_entries.csv")
OUT = os.path.join(ROOT, "alignn_data_exp")
RUNS = os.path.join(ROOT, "runs", "exp")
ARMS = {"A0": (), "A1": ("2dmp",), "A2": ("meta",), "A3": ("2dmp", "meta")}
EXTRA_TEST = 0.2      # share of the additions' compositions held out as test sets
LIGHT = ("C", "N", "B", "H", "O")
N_ANG = 9             # the shipped ensemble's angular width


def id_prop(folder: str) -> dict[str, float]:
    out = {}
    for line in open(os.path.join(folder, "id_prop.csv")):
        f, y = line.strip().split(",")[:2]
        out[f] = float(y)
    return out


def prepare(args):
    from jarvis.db.figshare import data
    from pymatgen.core import Composition
    from pymatgen.io.vasp import Poscar
    from sklearn.model_selection import GroupShuffleSplit
    from audit_2dmatpedia import to_pymatgen

    if not os.path.exists(AUDIT):
        raise SystemExit(f"{AUDIT} missing: run scripts/audit_2dmatpedia.py first")
    split = json.load(open(SPLIT))
    base, wide = id_prop(BASE), id_prop(WIDE)
    assert set(base) <= set(wide), "the 0.1 export should be a subset of the 0.2 one"

    alex = {x["mat_id"]: x for x in data("alex_pbe_2d_all")}
    formula = {f: Composition(alex[f[:-5]]["formula"]).reduced_formula for f in wide}
    elements = {f: sorted(set(alex[f[:-5]]["atoms"]["elements"])) for f in wide}
    held = {formula[f] for k in ("val", "test") for f in split[k]}

    # additions: formulas of the shipped val/test never enter any training part
    meta = [f for f in wide if f not in base and formula[f] not in held]
    mpd = {x["material_id"]: x for x in data("twod_matpd")}
    rows = list(csv.DictReader(open(AUDIT)))
    new = [r for r in rows if r["status"] in ("new_polymorph", "formula_absent")
           and float(r["gap"]) > 0.01 and float(r["mag"]) <= 0.1 and r["formula"] not in held]
    # a structure train_cgcnn would drop (no neighbour within the cutoff) must not be
    # named in a split; the trainer refuses such a split rather than guess
    from nanomat.graph import ensure_vacuum, to_graph
    ok = [r for r in new if to_graph(ensure_vacuum(to_pymatgen(mpd[r["id"]]["atoms"]))[0]) is not None]
    if len(ok) < len(new):
        print(f"  {len(new) - len(ok)} 2DMatPedia structures have no graph and are left out")
    new = ok
    for r in new:
        f = r["id"] + ".vasp"
        formula[f], elements[f] = r["formula"], r["elements"].split()
    d_files = [r["id"] + ".vasp" for r in new]
    print(f"additions after removing the shipped val/test compositions: "
          f"{len(meta)} metastable Alexandria, {len(d_files)} 2DMatPedia")

    # one group split over both additions: a formula is test everywhere or nowhere,
    # so no arm can meet a test formula through the other database
    extra = meta + d_files
    groups = np.array([formula[f] for f in extra])
    gss = GroupShuffleSplit(n_splits=1, test_size=EXTRA_TEST, random_state=0)
    tr, te = next(gss.split(np.arange(len(extra)), groups=groups))
    is_test = np.zeros(len(extra), bool)
    is_test[te] = True
    is_meta = set(meta)
    parts = {"meta": [f for f, t in zip(extra, is_test) if not t and f in is_meta],
             "2dmp": [f for f, t in zip(extra, is_test) if not t and f not in is_meta]}
    tests = {"T_alex": split["test"],
             "T_meta": [f for f, t in zip(extra, is_test) if t and f in is_meta],
             "T_2dmp": [f for f, t in zip(extra, is_test) if t and f not in is_meta]}
    assert not ({formula[f] for f in tests["T_meta"] + tests["T_2dmp"]}
                & {formula[f] for p in parts.values() for f in p})

    # the folder: hard links where the files already exist, 2DMatPedia written once
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    labels = {}
    for f in base:
        os.link(os.path.join(BASE, f), os.path.join(OUT, f))
        labels[f] = base[f]
    for f in meta:
        os.link(os.path.join(WIDE, f), os.path.join(OUT, f))
        labels[f] = wide[f]
    for r in new:
        f = r["id"] + ".vasp"
        Poscar(to_pymatgen(mpd[r["id"]]["atoms"])).write_file(os.path.join(OUT, f))
        labels[f] = float(r["gap"])
    # only structures some arm or test set uses are listed: the graph cache covers
    # exactly these, and every run builds the same list in the same order
    used = list(base) + parts["meta"] + tests["T_meta"] + parts["2dmp"] + tests["T_2dmp"]
    with open(os.path.join(OUT, "id_prop.csv"), "w") as fh:
        for f in used:
            fh.write(f"{f},{labels[f]}\n")

    os.makedirs(os.path.join(OUT, "splits"), exist_ok=True)
    counts = {}
    for arm, adds in ARMS.items():
        train = list(split["train"]) + [f for a in adds for f in parts[a]]
        json.dump({"train": train, "val": split["val"], "test": split["test"]},
                  open(os.path.join(OUT, "splits", f"{arm}.json"), "w"))
        c = Counter(el for f in train for el in elements[f] if el in LIGHT)
        counts[arm] = {"n_train": len(train), **{el: c[el] for el in LIGHT}}
    json.dump({"tests": tests, "formula": {f: formula[f] for f in used},
               "elements": {f: elements[f] for f in used},
               "source": {f: ("alexandria" if f.startswith("agm") else "2dmatpedia") for f in used},
               "train_counts": counts},
              open(os.path.join(OUT, "experiment.json"), "w"))

    print(f"\n{'arm':4s} {'train':>7s}   " + "  ".join(f"{el:>5s}" for el in LIGHT))
    for arm, c in counts.items():
        print(f"{arm:4s} {c['n_train']:7d}   " + "  ".join(f"{c[el]:5d}" for el in LIGHT))
    print("\ntest sets: " + ", ".join(f"{k} {len(v)}" for k, v in tests.items()))
    for k in ("T_meta", "T_2dmp"):
        c = Counter(el for f in tests[k] for el in elements[f] if el in LIGHT)
        print(f"  {k}: " + "  ".join(f"{el} {c[el]}" for el in LIGHT))
    print(f"\nwritten {OUT} ({len(used)} structures) - zip it for the GPU box:")
    print(f"  cd {ROOT} && zip -qr alignn_data_exp.zip alignn_data_exp")
    print("\non the GPU box, once, then all four arms side by side:")
    print("  python scripts/stability_experiment.py cache")
    for arm in ARMS:
        print(f"  nohup python train_cgcnn.py --data alignn_data_exp --split-file "
              f"alignn_data_exp/splits/{arm}.json --ensemble 5 --angles {N_ANG} --cache "
              f"--workers 5 --out runs/exp/{arm}.pt > runs/exp/{arm}.log 2>&1 &")


def cache(args):
    """Build the graph cache once; four runs building it at once would race."""
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
    if "A0" not in arms:
        raise SystemExit("the control A0 is required: every comparison is paired against it")
    structures = {k: [(f, read_structure(os.path.join(args.data, f))) for f in v]
                  for k, v in exp["tests"].items()}
    pred = {}
    for arm in arms:
        stage = os.path.join(ROOT, f".exp_{arm}")
        os.makedirs(stage, exist_ok=True)
        shutil.copy(os.path.join(args.runs, f"{arm}.pt"), os.path.join(stage, "cgcnn_2d_ensemble.pt"))
        P = Predictor(stage, verbose=False)
        pred[arm] = {k: {key: r.gap for key, r in P.run_many(v, batch_size=256) if r is not None}
                     for k, v in structures.items()}
        shutil.rmtree(stage)
        print(f"{arm}: predicted " + ", ".join(f"{k} {len(v)}" for k, v in pred[arm].items()))

    rng = np.random.default_rng(0)
    report = {}
    for k, files in exp["tests"].items():
        files = [f for f in files if all(f in pred[a][k] for a in arms)]
        y = np.array([truth[f] for f in files])
        subsets = {"all": np.ones(len(files), bool)}
        for el in LIGHT:
            subsets[f"has {el}"] = np.array([el in exp["elements"][f] for f in files])
        report[k] = {}
        print(f"\n{k} (n={len(files)})   MAE, and the paired difference to A0 "
              f"(negative = better than the control)")
        print(f"  {'subset':10s} {'n':>5s}  " + "  ".join(f"{a:>26s}" for a in arms))
        for name, sel in subsets.items():
            if sel.sum() < 10:
                continue
            e0 = np.abs(np.array([pred["A0"][k][f] for f in files]) - y)[sel]
            cells, row = [], {"n": int(sel.sum())}
            for a in arms:
                e = np.abs(np.array([pred[a][k][f] for f in files]) - y)[sel]
                if a == "A0":
                    cells.append(f"{e.mean():26.3f}")
                    row[a] = {"mae": float(e.mean())}
                    continue
                d = e - e0
                boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                p = float(wilcoxon(e, e0).pvalue) if np.any(d != 0) else 1.0
                cells.append(f"{e.mean():.3f} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
                row[a] = {"mae": float(e.mean()), "diff": float(d.mean()),
                          "ci95": [float(lo), float(hi)], "wilcoxon_p": p}
            report[k][name] = row
            print(f"  {name:10s} {int(sel.sum()):5d}  " + "  ".join(cells))
    json.dump(report, open(os.path.join(args.runs, "evaluation.json"), "w"), indent=1)
    print(f"\nwritten {os.path.join(args.runs, 'evaluation.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "cache", "evaluate"])
    ap.add_argument("--data", default=OUT)
    ap.add_argument("--runs", default=RUNS)
    args = ap.parse_args()
    {"prepare": prepare, "cache": cache, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
