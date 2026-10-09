#!/usr/bin/env python
"""The two classifiers, brought up to the gap ensemble: five seeds, data to 0.5 eV/atom.

The gap ensemble now trains on Alexandria monolayers up to 0.5 eV/atom above the hull
(hull_experiment.py). The metal gate stops at 0.2 and the gap-type classifier at 0.1,
and both are single models. Two misses on reference materials point at the second
part: with another seed the type classifier calls monolayer MoS2 indirect, and the
gate rejects an Alexandria MoS2 polymorph with a 0.83 eV gap (p 0.54 against 0.52).
A single classifier's call near its threshold is partly seed luck; five seeds
averaged is what the gap model already does.

classifier_experiment.py's design, one step further. The same folders, splits and
held-out sets are rebuilt by its prepare(), then Alexandria at 0.2-0.5 eV/atom is
added: the gap experiment's held-out 0.2-0.5 formulas (T_far) stay held out here,
and formulas it never saw - mostly compositions with no semiconducting polymorph -
are split 80/20 by formula the same way.

  metal gate (angles, as shipped)
    G0  the shipped recipe: <= 0.2 eV/atom + 2DMatPedia, one model       control
    G1  the same data, five seeds averaged
    G2  + Alexandria 0.2-0.5 eV/atom, metals included, five seeds
  gap type (no angles, as shipped)
    T0  the shipped recipe: <= 0.1 eV/atom, one model                    control
    T1  <= 0.2 eV/atom, five seeds averaged
    T2  + Alexandria semiconductors at 0.2-0.5 eV/atom, five seeds

Decision rule, written before any arm trained (8 October 2026). Threshold-free
ROC-AUC, paired against the control on the same structures, 95% bootstrap interval.
An arm is accepted when all three hold:
  1. not worse where the shipped model was measured - metal gate on T_old: delta
     >= -0.005 and the interval's lower end above -0.015; gap type on T_alex: delta
     >= -0.01 and the lower end above -0.03;
  2. better on at least one held-out population - T_meta, T_2dmp (gate only) or
     T_far - with the whole interval above zero;
  3. the reference structures in examples/: for the gate, graphene flagged and none
     of the six semiconductors flagged at -1%, 0 or +1% in-plane strain; for the
     type classifier, MoS2, MoSe2, WS2, WSe2 and phosphorene called direct and h-BN
     indirect, as in validate_experiment.py.
Among accepted arms, the one with the largest summed AUC gain over the held-out
populations; if the two are within 0.01 of each other, the one with less data (G1,
T1). If no arm is accepted the shipped classifier stays, and the result is written
up. The MoS2 polymorph the gate rejected (agm2000041223) is reported, not decided on:
a rule tuned to the one case it was built from would prove nothing.

    python scripts/classifier_hull_experiment.py prepare     # downloads through JARVIS
    python scripts/classifier_hull_experiment.py cache --folder metal --angles 9
    python scripts/classifier_hull_experiment.py cache --folder type --angles 0
    ... six train_cgcnn.py runs, printed by prepare ...
    python scripts/classifier_hull_experiment.py evaluate
"""

from __future__ import annotations

import argparse
import json
import os
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

import classifier_experiment as CE  # noqa: E402

FOLDERS = CE.FOLDERS
HULL = os.path.join(ROOT, "alignn_data_hull")      # hull_experiment.py: experiment.json, splits/B2.json
RUNS = os.path.join(ROOT, "runs", "cls_hull")
BAND = (0.2, 0.5)                                  # (low, high], eV/atom
ARMS = {  # arm: (task, split file, angles, ensemble)
    "G0": ("metal", "M1", CE.N_ANG, 1), "G1": ("metal", "M1", CE.N_ANG, 5), "G2": ("metal", "G2", CE.N_ANG, 5),
    "T0": ("type", "Y0", 0, 1), "T1": ("type", "Y1", 0, 5), "T2": ("type", "T2", 0, 5),
}
CONTROL = {"metal": "G0", "type": "T0"}
SMALLER = {"metal": "G1", "type": "T1"}
SHIPPED = CE.SHIPPED
HELD_OUT = {"metal": ("T_meta", "T_2dmp", "T_far"), "type": ("T_meta", "T_far")}
NOT_WORSE = {"metal": ("T_old", -0.005, -0.015), "type": ("T_alex", -0.01, -0.03)}
TYPE_EXPECTED = {"MoS2": "direct", "MoSe2": "direct", "WS2": "direct", "WSe2": "direct",
                 "phosphorene": "direct", "hBN": "indirect"}
SEMIS = ("MoS2", "MoSe2", "WS2", "WSe2", "hBN", "phosphorene")
FALSE_ALARM = "agm2000041223.vasp"                 # the MoS2 polymorph the shipped gate rejects


def hull_sets():
    exp = json.load(open(os.path.join(HULL, "experiment.json")))
    b2 = set(json.load(open(os.path.join(HULL, "splits", "B2.json")))["train"])
    a3 = set(json.load(open(os.path.join(CE.EXP, "splits", "A3.json")))["train"])
    far = exp["tests"]["T_far"]
    return ({exp["formula"][f] for f in far}, {exp["formula"][f] for f in b2 - a3},
            sorted(b2 - a3), far)


def extend_metal(alex: dict):
    from sklearn.model_selection import GroupShuffleSplit
    from audit_2dmatpedia import to_pymatgen
    from nanomat.graph import ensure_vacuum, to_graph

    folder = FOLDERS["metal"]
    exp = json.load(open(os.path.join(folder, "experiment.json")))
    m1 = json.load(open(os.path.join(folder, "splits", "M1.json")))
    formula = exp["formula"]
    trained = {formula[f] for f in m1["train"]}
    held = {formula[f] for k in exp["tests"] for f in exp["tests"][k]} | {formula[f] for f in m1["val"]}
    far_f, hull_train_f, _, _ = hull_sets()

    cands, t = [], time.time()
    for e in alex.values():
        eh, g = CE.num(e.get("e_above_hull")), CE.num(e.get("band_gap_ind"))
        if eh is None or g is None or not BAND[0] < eh <= BAND[1]:
            continue
        fo = CE.reduced(e["atoms"]["elements"])
        if fo in held:
            continue
        if to_graph(ensure_vacuum(to_pymatgen(e["atoms"]))[0]) is None:
            continue
        cands.append((e["mat_id"] + ".vasp", e, fo))
    train, test, free = [], [], []
    for f, e, fo in cands:
        if fo in far_f and fo not in trained:
            test.append(f)
        elif fo in far_f or fo in trained or fo in hull_train_f:
            train.append(f)
        else:
            free.append((f, fo))
    gss = GroupShuffleSplit(n_splits=1, test_size=CE.EXTRA_TEST, random_state=0)
    tr, te = next(gss.split(np.arange(len(free)), groups=[fo for _, fo in free]))
    train += [free[i][0] for i in tr]
    test += [free[i][0] for i in te]
    by = {f: (e, fo) for f, e, fo in cands}
    seen = trained | {by[f][1] for f in train}
    assert not ({by[f][1] for f in test} & seen), "a T_far formula leaked into training"

    with open(os.path.join(folder, "id_prop.csv"), "a") as fh:
        for f in train + test:
            e = by[f][0]
            with open(os.path.join(folder, f), "w") as p:
                p.write(CE.poscar(e["atoms"]))
            fh.write(f"{f},{float(e['band_gap_ind'])}\n")
            formula[f] = by[f][1]
            exp["elements"][f] = sorted(set(e["atoms"]["elements"]))
            exp["source"][f] = "far"
    exp["tests"]["T_far"] = test
    json.dump(exp, open(os.path.join(folder, "experiment.json"), "w"))
    json.dump({"train": list(m1["train"]) + train, "val": m1["val"], "test": m1["test"]},
              open(os.path.join(folder, "splits", "G2.json"), "w"))
    metal = lambda fs: 100 * np.mean([float(by[f][0]["band_gap_ind"]) <= CE.METAL_GAP for f in fs])
    print(f"metal gate, 0.2-0.5 eV/atom: {len(train)} added to training ({metal(train):.0f}% metals), "
          f"T_far {len(test)} ({metal(test):.0f}% metals); {time.time() - t:.0f}s")


def extend_type(alex: dict):
    folder = FOLDERS["type"]
    exp = json.load(open(os.path.join(folder, "experiment.json")))
    y1 = json.load(open(os.path.join(folder, "splits", "Y1.json")))
    _, _, hull_add, far = hull_sets()
    rows, gone = {}, 0
    for f in hull_add + far:
        e = alex[f[:-5]]
        ind, dr = CE.num(e.get("band_gap_ind")), CE.num(e.get("band_gap_dir"))
        if ind is None or dr is None:
            gone += 1
            continue
        rows[f] = (e, ind, dr)
    with open(os.path.join(folder, "id_prop.csv"), "a") as fh:
        for f, (e, ind, dr) in rows.items():
            with open(os.path.join(folder, f), "w") as p:
                p.write(CE.poscar(e["atoms"]))
            fh.write(f"{f},{ind},{dr}\n")
            exp["elements"][f] = sorted(set(e["atoms"]["elements"]))
    add = [f for f in hull_add if f in rows]
    exp["tests"]["T_far"] = [f for f in far if f in rows]
    json.dump(exp, open(os.path.join(folder, "experiment.json"), "w"))
    json.dump({"train": list(y1["train"]) + add, "val": y1["val"], "test": y1["test"]},
              open(os.path.join(folder, "splits", "T2.json"), "w"))
    ind = lambda fs: 100 * np.mean([rows[f][2] - rows[f][1] >= CE.TYPE_GAP for f in fs])
    print(f"gap type, 0.2-0.5 eV/atom: {len(add)} added ({ind(add):.0f}% indirect), "
          f"T_far {len(exp['tests']['T_far'])} ({ind(exp['tests']['T_far']):.0f}% indirect); "
          f"{gone} without a direct gap left out")


def prepare(args):
    from jarvis.db.figshare import data
    CE.prepare(args)                                  # the previous experiment's folders, rebuilt
    alex = {x["mat_id"]: x for x in data("alex_pbe_2d_all")}
    extend_metal(alex)
    extend_type(alex)
    print("\ncaches (one per folder, they may run side by side):")
    print(f"  python scripts/classifier_hull_experiment.py cache --folder metal --angles {CE.N_ANG}")
    print("  python scripts/classifier_hull_experiment.py cache --folder type --angles 0")
    print("\nthe arms:")
    for arm, (task, split, ang, ens) in ARMS.items():
        d = os.path.relpath(FOLDERS[task], ROOT)
        print(f"  python train_cgcnn.py --data {d} --task {task} --split-file {d}/splits/{split}.json "
              f"{CE.HYPER[task]} --seed 0 --ensemble {ens} --angles {ang} --cache --workers 5 "
              f"--out runs/cls_hull/{arm}.pt")


def cache(args):
    from train_cgcnn import build_graphs, read_id_prop
    folder = FOLDERS[args.folder]
    files = read_id_prop(folder)["file"].tolist()
    t = time.time()
    build_graphs(folder, files, 8.0, True, args.angles)
    print(f"cache ready: {args.folder}, angles {args.angles}, {len(files)} graphs in {time.time() - t:.0f}s")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
_GRAPHS: dict = {}


def members(ck):
    from nanomat.model import CGCNNcls
    out = []
    for sd in ck.get("state_dicts") or [ck["state_dict"]]:
        m = CGCNNcls(h=int(ck.get("h", 128)), n_conv=int(ck.get("n_conv", 4)),
                     cutoff=float(ck["cutoff"]), n_rbf=int(ck["n_rbf"]), ang_dim=int(ck.get("n_ang") or 0))
        m.load_state_dict(sd)
        out.append(m.eval())
    return out


def probs_for(folder, ck_path, files, device):
    """Mean of the members' probabilities, the way nanomat.predict.ClsEnsemble reads them."""
    import torch
    from torch_geometric.loader import DataLoader
    from train_cgcnn import build_graphs, read_id_prop
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    n_ang = int(ck.get("n_ang") or 0)
    key = (folder, n_ang)
    if key not in _GRAPHS:
        all_files = read_id_prop(folder)["file"].tolist()
        graphs, _ = build_graphs(folder, all_files, float(ck["cutoff"]), True, n_ang)
        _GRAPHS[key] = (graphs, {f: i for i, f in enumerate(all_files)})
    graphs, pos = _GRAPHS[key]
    ms = [m.to(device) for m in members(ck)]
    out = []
    with torch.no_grad():
        for b in DataLoader([graphs[pos[f]] for f in files], batch_size=512):
            b = b.to(device)
            out.append(torch.stack([torch.sigmoid(m(b)[1]) for m in ms]).mean(0).cpu().numpy())
    return np.concatenate(out), float(ck["threshold"])


def reference_probs(ck_path, eps):
    """Probabilities on examples/ at an in-plane strain eps, as tests/test_smoke.py does it."""
    import torch
    from pymatgen.core import Structure
    from torch_geometric.data import Batch
    from nanomat.graph import ensure_vacuum, layer_info, to_graph
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ms, n_ang, cut = members(ck), int(ck.get("n_ang") or 0), float(ck["cutoff"])
    out = {}
    for name in ("graphene",) + SEMIS:
        st = ensure_vacuum(Structure.from_file(os.path.join(ROOT, "examples", f"{name}.vasp")), cut)[0]
        L, axis = st.lattice.matrix.copy(), layer_info(st)["axis"]
        for i in range(3):
            if i != axis:
                L[i] *= 1 + eps
        g = to_graph(Structure(L, st.species, st.frac_coords), cut, n_ang=n_ang)
        with torch.no_grad():
            out[name] = float(np.mean([torch.sigmoid(m(Batch.from_data_list([g]))[1]).item() for m in ms]))
    return out, float(ck["threshold"])


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
            truth[parts[0]] = (y <= CE.METAL_GAP) if task == "metal" else (float(parts[2]) - y >= CE.TYPE_GAP)
        arms = [a for a, v in ARMS.items() if v[0] == task and os.path.exists(os.path.join(args.runs, f"{a}.pt"))]
        c = CONTROL[task]
        if c not in arms:
            print(f"{task}: control {c} not trained yet, skipped")
            continue
        paths = {a: os.path.join(args.runs, f"{a}.pt") for a in arms}
        paths["shipped"] = os.path.join(ROOT, "weights", SHIPPED[task])
        rep = report[task] = {"tests": {}, "references": {}, "decision": {}}
        for k, files in exp["tests"].items():
            y = np.array([truth[f] for f in files])
            pr = {a: probs_for(folder, p, files, device) for a, p in paths.items()}
            print(f"\n{task} / {k}: n={len(files)}, {100 * y.mean():.0f}% "
                  f"{'metal' if task == 'metal' else 'indirect'}")
            rows = {}
            for a in paths:
                p, thr = pr[a]
                row = {"auc": float(roc_auc_score(y, p))}
                cell = ""
                if a != c:
                    d, n = [], len(y)
                    for _ in range(2000):
                        i = rng.integers(0, n, n)
                        if y[i].all() or not y[i].any():
                            continue
                        d.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], pr[c][0][i]))
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    row.update(diff=row["auc"] - float(roc_auc_score(y, pr[c][0])), ci95=[float(lo), float(hi)])
                    cell = f"{row['diff']:+.4f} [{lo:+.4f},{hi:+.4f}]"
                flag = p >= thr
                row.update(caught=float(flag[y].mean()), false_flags=float(flag[~y].mean()), threshold=thr)
                rows[a] = row
                print(f"  {a:8s} AUC {row['auc']:.4f} {cell:>26s}   caught {100 * row['caught']:5.1f}%  "
                      f"false {100 * row['false_flags']:5.1f}%  thr {thr:.2f}")
            rep["tests"][k] = {"n": len(files), "arms": rows}

        print(f"\n{task}: reference structures")
        for a, p in paths.items():
            if task == "metal":
                ok, cells = True, []
                for eps in (-0.01, 0.0, 0.01):
                    pr_, thr = reference_probs(p, eps)
                    bad = [n for n in SEMIS if pr_[n] >= thr]
                    ok &= not bad and (eps != 0.0 or pr_["graphene"] >= thr)
                    cells.append(f"{eps:+.0%}: " + ", ".join(f"{n} {pr_[n]:.2f}" for n in ("graphene",) + SEMIS))
                    rep["references"].setdefault(a, {})[f"{eps:+.2f}"] = pr_
                if FALSE_ALARM in truth:
                    fa = probs_for(folder, p, [FALSE_ALARM], device)[0][0]
                    rep["references"][a]["MoS2_polymorph_agm2000041223"] = float(fa)
                    cells.append(f"MoS2 polymorph agm2000041223 {fa:.2f} (thr {thr:.2f}, reported only)")
                print(f"  {a:8s} {'pass' if ok else 'FAIL'}\n           " + "\n           ".join(cells))
            else:
                pr_, thr = reference_probs(p, 0.0)
                calls = {n: ("indirect" if pr_[n] >= thr else "direct") for n in SEMIS}
                ok = all(calls[n] == TYPE_EXPECTED[n] for n in SEMIS)
                rep["references"][a] = {n: [pr_[n], calls[n]] for n in SEMIS}
                print(f"  {a:8s} {'pass' if ok else 'FAIL'}  " + "  ".join(
                    f"{n} {pr_[n]:.2f} {calls[n][:3]}{'' if calls[n] == TYPE_EXPECTED[n] else '!'}" for n in SEMIS))
            rep["references"].setdefault(a, {})["pass"] = bool(ok)

        # the rule, mechanically
        name, d_min, lo_min = NOT_WORSE[task]
        accepted = {}
        for a in arms:
            if a == c:
                continue
            t0 = rep["tests"][name]["arms"][a]
            ok1 = t0["diff"] >= d_min and t0["ci95"][0] > lo_min
            held = {k: rep["tests"][k]["arms"][a] for k in HELD_OUT[task] if k in rep["tests"]}
            ok2 = any(v["ci95"][0] > 0 for v in held.values())
            ok3 = rep["references"][a]["pass"]
            gain = sum(v["diff"] for v in held.values())
            rep["decision"][a] = {"not_worse": ok1, "better_somewhere": ok2, "references": ok3, "summed_gain": gain}
            print(f"  rule {a}: not worse {ok1}, better somewhere {ok2}, references {ok3}, summed gain {gain:+.4f}")
            if ok1 and ok2 and ok3:
                accepted[a] = gain
        if not accepted:
            choice = "shipped (no arm accepted)"
        else:
            best = max(accepted, key=accepted.get)
            choice = SMALLER[task] if SMALLER[task] in accepted and accepted[best] - accepted[SMALLER[task]] <= 0.01 else best
        rep["decision"]["choice"] = choice
        print(f"  -> {task}: {choice}")
    os.makedirs(args.runs, exist_ok=True)
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
