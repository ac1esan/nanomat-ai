#!/usr/bin/env python
"""Do the metal gate and the gap-type classifier gain from the data the gap model did?

The gap ensemble improved on every population once its training set took in the
metastable Alexandria monolayers (0.1-0.2 eV/atom) and 2DMatPedia's non-magnetic
ones (stability_experiment.py). The two classifiers next to it were never
retrained. The metal gate is the first stage of the trust layer, and on 2DMatPedia
it catches 71% of the non-magnetic metals while flagging 17% of the semiconductors
(audit_2dmatpedia.py --predict).

The stability experiment's design, applied to both. One environment; the shipped
validation and test sets in every arm, so only the training part differs; extra
test sets cut from the additions by composition, so that no arm trained on their
formulas. Formulas the gap experiment held out stay held out here, so both models
can be scored end to end on the same unseen compositions.

  metal gate  Alexandria <= 0.1 eV/atom + C2DB + JARVIS dft_2d, metals kept
    M0   the shipped recipe and split                                  control
    M0s  M0 with seed 1                                                seed noise
    M1   + Alexandria 0.1-0.2 and non-magnetic 2DMatPedia, metals included
    M2   M1 with the angular descriptor the gap ensemble uses
  gap type    Alexandria semiconductors, the gap ensemble's split
    Y0   the shipped recipe and split                                  control
    Y0s  Y0 with seed 1                                                seed noise
    Y1   + Alexandria 0.1-0.2 (2DMatPedia records no direct gap)
    Y2   Y1 with angles

Decision rule, written before any arm had trained. Threshold-free ROC-AUC, paired
against the control on the same structures, 95% bootstrap interval:
  1. not worse where the shipped model was measured - metal gate on T_old:
     delta >= -0.005 and the interval's lower end above -0.015; gap type on T_alex
     (279 positives, so noisier): delta >= -0.01 and the lower end above -0.03;
  2. better where it was weak - on T_meta or T_2dmp the whole interval above zero,
     and the gain larger than the seed-noise arm's distance from the control;
  3. metal gate only: graphene still flagged, and none of the six reference
     semiconductors in examples/ flagged.
Among accepted arms the largest summed gain on T_meta + T_2dmp; within seed noise,
the arm without angles (the smaller change to what ships). The operating point at
the validation threshold is reported beside it - that is the number people quote -
but it is not what decides: the shipped validation set has none of the new
populations, so a threshold fitted there is not tuned for them in any arm.

    python scripts/classifier_experiment.py prepare      # needs the JARVIS downloads
    python scripts/classifier_experiment.py cache --folder metal --angles 0   # x4, see prepare
    ... train_cgcnn.py runs, printed by prepare ...
    python scripts/classifier_experiment.py evaluate
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

EXP = os.path.join(ROOT, "alignn_data_exp")                  # the stability experiment
SHIPPED_METAL_SPLIT = os.path.join(ROOT, "runs", "metal_merged.split.json")
AUDIT = os.path.join(ROOT, "runs", "audit_2dmatpedia_entries.csv")
FOLDERS = {"metal": os.path.join(ROOT, "alignn_data_metal"),
           "type": os.path.join(ROOT, "alignn_data_typed")}
RUNS = os.path.join(ROOT, "runs", "cls")
EXTRA_TEST = 0.2       # as in the stability experiment
N_ANG = 9              # the shipped gap ensemble's angular width
METAL_GAP = 0.05       # train_cgcnn --metal-threshold: the shipped gate's label
TYPE_GAP = 0.1         # train_cgcnn --type-threshold
# the shipped models' own hyper-parameters (read from their checkpoints' meta)
HYPER = {"metal": "--epochs 120 --batch 128", "type": "--epochs 150 --batch 64"}
ARMS = {  # arm: (task, split file, angles, seed)
    "M0": ("metal", "M0", 0, 0), "M1": ("metal", "M1", 0, 0), "M2": ("metal", "M1", N_ANG, 0),
    "Y0": ("type", "Y0", 0, 0), "Y1": ("type", "Y1", 0, 0), "Y2": ("type", "Y1", N_ANG, 0),
    "M0s": ("metal", "M0", 0, 1), "Y0s": ("type", "Y0", 0, 1),
}
CONTROL = {"metal": "M0", "type": "Y0"}
NOISE = {"metal": "M0s", "type": "Y0s"}
SHIPPED = {"metal": "cgcnn_2d_metal.pt", "type": "cgcnn_2d_typed.pt"}
LIGHT = ("C", "N", "B", "H", "O")


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def reduced(elements) -> str:
    from pymatgen.core import Composition
    return Composition(dict(Counter(elements))).reduced_formula


def id_prop_files(folder: str) -> list[str]:
    return [line.split(",")[0] for line in open(os.path.join(folder, "id_prop.csv"))]


def poscar(atoms: dict) -> str:
    from jarvis.core.atoms import Atoms
    return Atoms.from_dict(atoms).get_string()   # what export_structures_for_alignn.py writes


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def prepare_metal(alex: dict, c2db: list, dft: list, mpd: dict, gap_exp: dict):
    from sklearn.model_selection import GroupShuffleSplit
    from audit_2dmatpedia import to_pymatgen
    from nanomat.graph import ensure_vacuum, to_graph

    out = FOLDERS["metal"]
    shipped = json.load(open(SHIPPED_METAL_SPLIT))
    entries = {}                                   # file -> (atoms, gap, source)
    for e in alex.values():
        eh, g = num(e.get("e_above_hull")), num(e.get("band_gap_ind"))
        if eh is not None and g is not None and eh <= 0.2:
            entries[e["mat_id"] + ".vasp"] = (e["atoms"], g, "alexandria" if eh <= 0.1 else "meta")
    for e in c2db:
        if (g := num(e.get("gap"))) is not None:
            entries[e["id"] + ".vasp"] = (e["atoms"], g, "c2db")
    for e in dft:
        if (g := num(e.get("optb88vdw_bandgap"))) is not None:
            entries[e["jid"] + ".vasp"] = (e["atoms"], g, "dft_2d")
    for r in csv.DictReader(open(AUDIT)):
        # structure twins of an Alexandria entry are left out (same structure, same
        # scheme); magnetic entries too - the audit found 2DMatPedia often in a
        # different magnetic state there, which flips metal against semiconductor
        if r["status"] in ("new_polymorph", "formula_absent") and float(r["mag"]) <= 0.1:
            entries[r["id"] + ".vasp"] = (mpd[r["id"]]["atoms"], float(r["gap"]), "2dmp")

    shipped_all = [f for k in ("train", "val", "test") for f in shipped[k]]
    missing = [f for f in shipped_all if f not in entries]
    if missing:
        raise SystemExit(f"{len(missing)} files of the shipped split are not rebuilt, "
                         f"first {missing[:3]}: the control would not be the shipped recipe")
    formula = {f: reduced(v[0]["elements"]) for f, v in entries.items()}
    held = {formula[f] for k in ("val", "test") for f in shipped[k]}
    old_train = {formula[f] for f in shipped["train"]}
    gap_test = {gap_exp["formula"][f] for k in ("T_meta", "T_2dmp") for f in gap_exp["tests"][k]}
    a0 = set(json.load(open(os.path.join(EXP, "splits", "A0.json")))["train"])
    a3 = json.load(open(os.path.join(EXP, "splits", "A3.json")))["train"]
    gap_add_train = {gap_exp["formula"][f] for f in a3 if f not in a0}

    adds = [f for f, v in entries.items() if v[2] in ("meta", "2dmp") and formula[f] not in held]
    t = time.time()
    ok = []
    for f in adds:
        st = to_pymatgen(entries[f][0])
        if to_graph(ensure_vacuum(st)[0]) is not None:
            ok.append(f)
    print(f"additions with a graph: {len(ok)} of {len(adds)} ({time.time() - t:.0f}s); "
          f"left out as val/test compositions of the shipped gate: "
          f"{sum(v[2] in ('meta', '2dmp') for v in entries.values()) - len(adds)}")
    fixed_train, fixed_test, free = [], [], []
    for f in ok:
        fo = formula[f]
        if fo in gap_test:
            (fixed_train if fo in old_train else fixed_test).append(f)
        elif fo in gap_add_train or fo in old_train:
            fixed_train.append(f)
        else:
            free.append(f)
    # formulas the gap experiment never saw - mostly compositions with no
    # semiconducting polymorph - are split the way it split its additions
    gss = GroupShuffleSplit(n_splits=1, test_size=EXTRA_TEST, random_state=0)
    tr, te = next(gss.split(np.arange(len(free)), groups=[formula[f] for f in free]))
    add_train = fixed_train + [free[i] for i in tr]
    add_test = fixed_test + [free[i] for i in te]
    tests = {"T_old": list(shipped["test"]),
             "T_meta": [f for f in add_test if entries[f][2] == "meta"],
             "T_2dmp": [f for f in add_test if entries[f][2] == "2dmp"]}
    seen = {formula[f] for f in list(shipped["train"]) + add_train}
    assert not ({formula[f] for f in add_test} & seen), "a test formula leaked into training"

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "splits"))
    used = shipped_all + add_train + add_test
    with open(os.path.join(out, "id_prop.csv"), "w") as fh:
        for f in used:
            atoms, gap, _ = entries[f]
            with open(os.path.join(out, f), "w") as p:
                p.write(poscar(atoms))
            fh.write(f"{f},{gap}\n")
    json.dump(shipped, open(os.path.join(out, "splits", "M0.json"), "w"))
    json.dump({"train": list(shipped["train"]) + add_train, "val": shipped["val"],
               "test": shipped["test"]}, open(os.path.join(out, "splits", "M1.json"), "w"))
    json.dump({"tests": tests, "formula": {f: formula[f] for f in used},
               "elements": {f: sorted(set(entries[f][0]["elements"])) for f in used},
               "source": {f: entries[f][2] for f in used}},
              open(os.path.join(out, "experiment.json"), "w"))

    def share(files):
        return f"{len(files):6d} ({100 * np.mean([entries[f][1] <= METAL_GAP for f in files]):.0f}% metals)"
    print(f"metal gate: M0 train {share(shipped['train'])}, additions {share(add_train)}")
    by = Counter(entries[f][2] for f in add_train)
    print(f"  additions by source: {dict(by)}")
    for k, v in tests.items():
        print(f"  {k}: {share(v)}")


def prepare_type(alex: dict, gap_exp: dict):
    out = FOLDERS["type"]
    a0 = json.load(open(os.path.join(EXP, "splits", "A0.json")))
    a2 = json.load(open(os.path.join(EXP, "splits", "A2.json")))
    t_meta = gap_exp["tests"]["T_meta"]
    labels = {}
    for line in open(os.path.join(EXP, "id_prop.csv")):
        f, y = line.strip().split(",")[:2]
        labels[f] = float(y)
    files = list(dict.fromkeys(a0["train"] + a0["val"] + a0["test"] + a2["train"] + t_meta))
    rows, dropped = [], []
    for f in files:
        e = alex[f[:-5]]
        ind, dr = num(e.get("band_gap_ind")), num(e.get("band_gap_dir"))
        if ind is None or dr is None:
            dropped.append(f)
            continue
        assert abs(ind - labels[f]) < 1e-6, f"{f}: label differs from the gap experiment's"
        rows.append((f, ind, dr))
    if dropped:
        print(f"gap type: {len(dropped)} structures without a direct gap are left out everywhere")
    gone = set(dropped)

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "splits"))
    with open(os.path.join(out, "id_prop.csv"), "w") as fh:
        for f, ind, dr in rows:
            os.link(os.path.join(EXP, f), os.path.join(out, f))
            fh.write(f"{f},{ind},{dr}\n")
    for name, sp in (("Y0", a0), ("Y1", a2)):
        json.dump({k: [f for f in sp[k] if f not in gone] for k in ("train", "val", "test")},
                  open(os.path.join(out, "splits", f"{name}.json"), "w"))
    kept = {f: (ind, dr) for f, ind, dr in rows}
    tests = {"T_alex": [f for f in a0["test"] if f not in gone],
             "T_meta": [f for f in t_meta if f not in gone]}
    json.dump({"tests": tests,
               "elements": {f: gap_exp["elements"][f] for f in kept}},
              open(os.path.join(out, "experiment.json"), "w"))

    def share(fs):
        return f"{len(fs):6d} ({100 * np.mean([kept[f][1] - kept[f][0] >= TYPE_GAP for f in fs]):.0f}% indirect)"
    print(f"gap type: Y0 train {share([f for f in a0['train'] if f not in gone])}, "
          f"Y1 train {share([f for f in a2['train'] if f not in gone])}")
    for k, v in tests.items():
        print(f"  {k}: {share(v)}")


def prepare(args):
    from jarvis.db.figshare import data
    gap_exp = json.load(open(os.path.join(EXP, "experiment.json")))
    alex = {x["mat_id"]: x for x in data("alex_pbe_2d_all")}
    prepare_metal(alex, data("c2db"), data("dft_2d"),
                  {x["material_id"]: x for x in data("twod_matpd")}, gap_exp)
    prepare_type(alex, gap_exp)
    print("\ncaches first (one per folder and angular width, they may run side by side):")
    for folder in FOLDERS:
        for ang in (0, N_ANG):
            print(f"  python scripts/classifier_experiment.py cache --folder {folder} --angles {ang}")
    print("\nthen the arms, one after another:")
    for arm, (task, split, ang, seed) in ARMS.items():
        print(f"  python train_cgcnn.py --data {os.path.relpath(FOLDERS[task], ROOT)} --task {task} "
              f"--split-file {os.path.relpath(FOLDERS[task], ROOT)}/splits/{split}.json {HYPER[task]} "
              f"--seed {seed} --angles {ang} --cache --workers 2 --out runs/cls/{arm}.pt")


def cache(args):
    """Build a graph cache once, so the arms read it instead of racing to write it."""
    from train_cgcnn import build_graphs, read_id_prop
    folder = FOLDERS[args.folder]
    files = read_id_prop(folder)["file"].tolist()     # the list train_cgcnn will ask for
    t = time.time()
    build_graphs(folder, files, 8.0, True, args.angles)
    print(f"cache ready: {args.folder}, angles {args.angles}, {len(files)} graphs in {time.time() - t:.0f}s")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
_GRAPHS: dict = {}


def load_probs(folder: str, ck_path: str, files: list[str], device: str):
    import torch
    from torch_geometric.loader import DataLoader
    from train_cgcnn import build_graphs, read_id_prop
    from nanomat.model import CGCNNcls

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    n_ang = int(ck.get("n_ang") or 0)
    key = (folder, n_ang)
    if key not in _GRAPHS:                      # the caches are large: read each once
        all_files = read_id_prop(folder)["file"].tolist()
        graphs, _ = build_graphs(folder, all_files, float(ck["cutoff"]), True, n_ang)
        _GRAPHS[key] = (graphs, {f: i for i, f in enumerate(all_files)})
    graphs, pos = _GRAPHS[key]
    m = CGCNNcls(h=int(ck.get("h", 128)), n_conv=int(ck.get("n_conv", 4)),
                 cutoff=float(ck["cutoff"]), n_rbf=int(ck["n_rbf"]), ang_dim=n_ang)
    m.load_state_dict(ck["state_dict"])
    m.eval().to(device)
    out = []
    with torch.no_grad():
        for b in DataLoader([graphs[pos[f]] for f in files], batch_size=512):
            out.append(torch.sigmoid(m(b.to(device))[1]).cpu().numpy())
    return np.concatenate(out), float(ck["threshold"]), n_ang


def canaries(ck_path: str) -> dict[str, float]:
    import torch
    from pymatgen.core import Structure
    from torch_geometric.data import Batch
    from nanomat.graph import ensure_vacuum, to_graph
    from nanomat.model import CGCNNcls
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    n_ang = int(ck.get("n_ang") or 0)
    m = CGCNNcls(h=int(ck.get("h", 128)), n_conv=int(ck.get("n_conv", 4)),
                 cutoff=float(ck["cutoff"]), n_rbf=int(ck["n_rbf"]), ang_dim=n_ang)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    out = {}
    for name in ("graphene", "MoS2", "MoSe2", "WS2", "WSe2", "hBN", "phosphorene"):
        st = Structure.from_file(os.path.join(ROOT, "examples", f"{name}.vasp"))
        g = to_graph(ensure_vacuum(st, float(ck["cutoff"]))[0], float(ck["cutoff"]), n_ang=n_ang)
        with torch.no_grad():
            out[name] = float(torch.sigmoid(m(Batch.from_data_list([g]))[1]))
    return out


def evaluate(args):
    import torch
    from sklearn.metrics import roc_auc_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    report = {}
    for task, folder in FOLDERS.items():
        exp = json.load(open(os.path.join(folder, "experiment.json")))
        truth = {}
        for line in open(os.path.join(folder, "id_prop.csv")):
            parts = line.strip().split(",")
            y = float(parts[1])
            truth[parts[0]] = (y <= METAL_GAP) if task == "metal" else (float(parts[2]) - y >= TYPE_GAP)
        arms = [a for a, v in ARMS.items() if v[0] == task and os.path.exists(os.path.join(args.runs, f"{a}.pt"))]
        if CONTROL[task] not in arms:
            print(f"{task}: control {CONTROL[task]} not trained yet, skipped")
            continue
        paths = {a: os.path.join(args.runs, f"{a}.pt") for a in arms}
        paths["shipped"] = os.path.join(ROOT, "weights", SHIPPED[task])
        report[task] = {"tests": {}}
        for k, files in exp["tests"].items():
            y = np.array([truth[f] for f in files])
            probs, thr = {}, {}
            for a, p in paths.items():
                probs[a], thr[a], _ = load_probs(folder, p, files, device)
            c = CONTROL[task]
            print(f"\n{task} / {k}: n={len(files)}, {100 * y.mean():.0f}% positive "
                  f"({'metal' if task == 'metal' else 'indirect'})")
            print(f"  {'arm':8s} {'AUC':>6s} {'vs ' + c:>24s}   {'caught':>6s} {'false':>6s} {'bal.acc':>7s}  thr")
            rows = {}
            for a in paths:
                auc = float(roc_auc_score(y, probs[a]))
                row = {"auc": auc}
                cell = ""
                if a != c:
                    d, n = [], len(y)
                    for _ in range(2000):
                        i = rng.integers(0, n, n)
                        if y[i].all() or not y[i].any():
                            continue
                        d.append(roc_auc_score(y[i], probs[a][i]) - roc_auc_score(y[i], probs[c][i]))
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    row.update(diff=auc - float(roc_auc_score(y, probs[c])), ci95=[float(lo), float(hi)])
                    cell = f"{row['diff']:+.4f} [{lo:+.4f},{hi:+.4f}]"
                flag = probs[a] >= thr[a]
                tpr, fpr = float(flag[y].mean()), float(flag[~y].mean())
                row.update(caught=tpr, false_flags=fpr, balanced_accuracy=(tpr + 1 - fpr) / 2,
                           threshold=thr[a])
                rows[a] = row
                print(f"  {a:8s} {auc:6.4f} {cell:>24s}   {100 * tpr:5.1f}% {100 * fpr:5.1f}% "
                      f"{row['balanced_accuracy']:7.3f}  {thr[a]:.2f}")
            light = np.array([bool(set(exp["elements"][f]) & set(LIGHT)) for f in files])
            if light.sum() >= 50 and 0 < y[light].mean() < 1:
                print("  with C/N/B/H/O (n=%d): " % light.sum() + "  ".join(
                    f"{a} {roc_auc_score(y[light], probs[a][light]):.3f}" for a in paths))
            report[task]["tests"][k] = {"n": len(files), "positive_rate": float(y.mean()), "arms": rows}
        if task == "metal":
            print("\nmetal gate on the reference structures (flagged = p >= own threshold):")
            report[task]["canaries"] = {}
            for a, p in paths.items():
                cn = canaries(p)
                thr_a = float(torch.load(p, map_location="cpu", weights_only=False)["threshold"])
                report[task]["canaries"][a] = cn
                print(f"  {a:8s} " + "  ".join(f"{n} {v:.2f}{'*' if v >= thr_a else ''}" for n, v in cn.items()))
    json.dump(report, open(os.path.join(args.runs, "evaluation.json"), "w"), indent=1)
    print(f"\nwritten {os.path.join(args.runs, 'evaluation.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "cache", "evaluate"])
    ap.add_argument("--folder", choices=list(FOLDERS), default="metal")
    ap.add_argument("--angles", type=int, default=0)
    ap.add_argument("--runs", default=RUNS)
    args = ap.parse_args()
    {"prepare": prepare, "cache": cache, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
