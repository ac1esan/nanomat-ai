#!/usr/bin/env python
"""Name the training structures behind the latent distance, and write the names into the checkpoint.

The ensemble checkpoint stores one embedding per training structure
(`reference_embeddings`, written by calibrate_uncertainty.py in the order of
split["train"]) so it can measure how far a new structure sits from what the model
has seen. This script attaches to each of those rows what a user needs in order to
read a neighbour: its id, source database, reduced formula, DFT label and, where the
label is known to be bad, the flag from data/reference_flags.csv. With it, a
prediction can say which known materials it resembles and what their gaps are.

    python scripts/reference_meta.py                    # shipped checkpoint and split
    python scripts/reference_meta.py --dry-run          # print the summary only

Labels come from the training folder's id_prop.csv where the file is present, from
the Alexandria 2D cache of jarvis-tools for Alexandria rows that are not, and from
data/mc2d.json.gz for MC2D rows. MC2D labels are not stored: the licence of the
OPTIMADE response has not been confirmed, so those rows carry the formula only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from glob import glob

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def source_of(name: str) -> str:
    if name.startswith("agm") or name == ".vasp":
        return "alexandria"
    if name.startswith("2dm-"):
        return "2dmatpedia"
    if name.startswith("mc2d-"):
        return "mc2d"
    return "other"


def poscar_composition(path: str) -> dict[str, float] | None:
    """Element counts from a VASP 5 POSCAR (symbols on line 6, counts on line 7)."""
    with open(path) as fh:
        lines = [fh.readline() for _ in range(7)]
    els, counts = lines[5].split(), lines[6].split()
    if not els or not all(c.isdigit() for c in counts) or len(els) != len(counts):
        return None
    comp: dict[str, float] = {}
    for e, c in zip(els, counts):
        comp[e] = comp.get(e, 0) + int(c)
    return comp


def alexandria_cache() -> dict[str, dict]:
    paths = glob(os.path.expanduser("~/.cache/atomgptlab/jarvis_data/alexandria_pbe_2d_*.json.zip"))
    if not paths:
        return {}
    with zipfile.ZipFile(sorted(paths)[-1]) as z:
        rows = json.loads(z.read(z.namelist()[0]))
    return {e["mat_id"] + ".vasp": e for e in rows}


def build(names: list[str], data_dirs: list[str]) -> dict:
    from pymatgen.core import Composition

    labels: dict[str, float] = {}
    folder_of: dict[str, str] = {}
    for d in data_dirs:
        for line in open(os.path.join(d, "id_prop.csv")):
            f, y = line.strip().split(",")[:2]
            if f not in labels:
                labels[f], folder_of[f] = float(y), d

    missing = [n for n in names if n not in labels]
    alex = alexandria_cache() if any(source_of(n) == "alexandria" for n in missing) else {}
    mc2d = {}
    if any(source_of(n) == "mc2d" for n in missing):
        from fetch_mc2d import load
        mc2d = {f"mc2d-{r['id']}.vasp": r for r in load()}

    flags = {r["id"]: r["flag"] for r in csv.DictReader(open(os.path.join(ROOT, "data", "reference_flags.csv")))}

    ids, formulas, sources, gaps, flag_list = [], [], [], [], []
    unresolved = []
    for n in names:
        src = source_of(n)
        comp, gap = None, None
        if n in labels:
            comp = poscar_composition(os.path.join(folder_of[n], n))
            gap = labels[n]
        elif src == "alexandria" and n in alex:
            comp = Composition(alex[n]["formula"]).as_dict()
            gap = float(alex[n]["band_gap_ind"])
        elif src == "mc2d" and n in mc2d:
            comp = Composition(mc2d[n]["formula"]).as_dict()
        if comp is None:
            unresolved.append(n)
        if src == "mc2d":
            gap = None                        # licence of the OPTIMADE response unconfirmed
        rid = n[:-5] if n.endswith(".vasp") else n
        ids.append(rid)
        sources.append(src)
        formulas.append(Composition(comp).reduced_formula if comp else "")
        gaps.append(float("nan") if gap is None else gap)
        flag_list.append(flags.get(rid, ""))
    return {"id": ids, "source": sources, "formula": formulas,
            "gap": torch.tensor(gaps, dtype=torch.float32), "flag": flag_list,
            "label": "PBE band gap of the training label (eV); NaN where not redistributed",
            "unresolved": unresolved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.pt"))
    ap.add_argument("--split", default=None, help="defaults to <weights>.split.json")
    ap.add_argument("--data", nargs="+", default=[os.path.join(ROOT, "alignn_data_hull")],
                    help="training folders with id_prop.csv, searched in order")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split = json.load(open(args.split or args.weights.replace(".pt", ".split.json")))
    ck = torch.load(args.weights, map_location="cpu")
    ref = ck.get("reference_embeddings")
    if ref is None:
        raise SystemExit("checkpoint has no reference_embeddings; run calibrate_uncertainty.py first")
    names = split["train"]
    if len(names) != ref.shape[0]:
        raise SystemExit(f"split has {len(names)} training structures but the checkpoint holds "
                         f"{ref.shape[0]} embeddings; they were not written from this split")

    meta = build(names, args.data)
    unresolved = meta.pop("unresolved")
    gaps = meta["gap"].numpy()
    by_src = {s: sum(x == s for x in meta["source"]) for s in sorted(set(meta["source"]))}
    print(f"{len(names)} reference rows: {by_src}")
    print(f"  formula resolved for {len(names) - len(unresolved)}, label for {int(np.isfinite(gaps).sum())}; "
          f"flagged labels {sum(bool(f) for f in meta['flag'])}")
    if unresolved:
        print(f"  unresolved ({len(unresolved)}): {unresolved[:5]}")
    if args.dry_run:
        return
    ck["reference_meta"] = meta
    torch.save(ck, args.weights)
    print(f"written reference_meta into {args.weights}")


if __name__ == "__main__":
    main()
