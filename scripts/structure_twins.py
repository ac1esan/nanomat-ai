#!/usr/bin/env python
"""Which rows of the screening table hold the same structure?

The table stacks three databases, and one structure can sit in more than one of them
- and in Alexandria more than once, in cells of different size. Listed row by row,
one material becomes two or three candidates. In the language-model test SbTeCl did
exactly that: one structure, twice in Alexandria (a 3-atom and a 6-atom cell) and once
in C2DB, predicted 1.51 eV each time, and offered as separate candidates in five of
ten answers to one question.

Every structure is put in one frame - layer normal along c, fixed vacuum, primitive
cell - and compared with the tight matcher of scripts/audit_2dmatpedia.py, within its
formula and primitive size only. A match also has to predict within MAX_PRED_DIFF of
its partner. The matcher alone joined 130 groups whose predictions differ by more
than 0.05 eV, up to 0.8: SbI3 and BiI3 cells whose own Alexandria references differ
by a factor of two to four (the primitive reduction smooths a distortion away), and
AuI relaxed by two functionals. For the tool "the same structure" has to mean the
same input to the model. Matches are joined transitively into groups.

    python scripts/structure_twins.py

Writes data/structure_twins.csv (id, source, group), listing only rows that have a
twin. nanomat/llm_tools.py reads it to return one row per structure, naming the
others and their references.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

TABLE = os.path.join(ROOT, "screening_table.csv")
OUT = os.path.join(ROOT, "data", "structure_twins.csv")
FOLDERS = {"alexandria": "alignn_data_alex_2d_eh02", "c2db": "alignn_data_c2db",
           "jarvis_dft2d": "alignn_data"}
MAX_PRED_DIFF = 0.05   # eV; the median inside a matched group is 0.003


def main():
    import pandas as pd
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from audit_2dmatpedia import MATCHER, standardize

    t = pd.read_csv(TABLE, usecols=["id", "source", "formula", "pred_gap_eV"])
    multi = t.groupby("formula")["id"].transform("size") > 1
    rows = t[multi].reset_index(drop=True)
    print(f"{len(rows)} rows share their formula with another row ({rows.formula.nunique()} formulas)")

    t0 = time.time()
    prim, skipped = {}, 0
    for r in rows.itertuples():
        path = os.path.join(ROOT, FOLDERS[r.source], f"{r.id}.vasp")
        s = standardize(Structure.from_file(path)) if os.path.exists(path) else None
        if s is None:
            skipped += 1
            continue
        prim[r.id] = s[0]
    print(f"standardised {len(prim)} in {time.time() - t0:.0f}s ({skipped} without a file or a vacuum gap)")

    parent = {i: i for i in prim}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sm = StructureMatcher(**MATCHER)
    pred = dict(zip(rows.id, rows.pred_gap_eV))
    pairs = vetoed = 0
    bucket = defaultdict(list)
    for r in rows.itertuples():
        if r.id in prim:
            bucket[(r.formula, len(prim[r.id]))].append(r.id)
    for ids in bucket.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if find(ids[a]) == find(ids[b]):
                    continue
                pairs += 1
                if sm.fit(prim[ids[a]], prim[ids[b]]):
                    if abs(pred[ids[a]] - pred[ids[b]]) > MAX_PRED_DIFF:
                        vetoed += 1
                        continue
                    parent[find(ids[b])] = find(ids[a])
    groups = defaultdict(list)
    for i in prim:
        groups[find(i)].append(i)
    groups = {g: m for g, m in groups.items() if len(m) > 1}
    print(f"{pairs} comparisons in {time.time() - t0:.0f}s: {len(groups)} structures appear more "
          f"than once, {sum(len(m) for m in groups.values())} rows; {vetoed} matches vetoed "
          f"for predicting more than {MAX_PRED_DIFF} eV apart")

    src = dict(zip(rows.id, rows.source))
    kinds = defaultdict(int)
    spread = []
    for m in groups.values():
        kinds[" + ".join(sorted({src[i] for i in m}))] += 1
        g = [pred[i] for i in m]
        spread.append(max(g) - min(g))
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:40s} {v}")
    spread.sort()
    print(f"prediction spread inside a group (same structure, other cell): median "
          f"{spread[len(spread) // 2]:.3f} eV, max {spread[-1]:.3f} eV")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "source", "group"])
        for g, m in sorted(groups.items()):
            for i in sorted(m):
                w.writerow([i, src[i], g])
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
