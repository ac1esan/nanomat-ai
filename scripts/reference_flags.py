#!/usr/bin/env python
"""Which DFT reference labels should not be trusted, and what that does to the numbers.

Two kinds, written to data/reference_flags.csv (id, flag, note):

  wrong     an Alexandria label checked against the same layer in other databases and
            found wrong - nanomat.llm_tools.KNOWN_BAD_REFERENCE: graphene in two cells,
            planar silicene, planar BP. All are planar honeycombs with their band edges
            at K.
  disputed  one structure, several Alexandria cells, labels that differ by more than
            SPREAD eV. The structure twins (scripts/structure_twins.py) are the same
            layer *and* get the same prediction to 0.05 eV, so a larger difference in
            the label is the label's, not the structure's. Which cell is right is not
            decided here; the row says what the other cells read.

The browser shows the note and leaves flagged rows out of every error it reports, the
language-model tools quote it, and this script prints what the shipped test MAE is
without them.

    python scripts/reference_flags.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat.llm_tools import KNOWN_BAD_REFERENCE  # noqa: E402

TABLE = os.path.join(ROOT, "screening_table.csv")
TWINS = os.path.join(ROOT, "data", "structure_twins.csv")
OUT = os.path.join(ROOT, "data", "reference_flags.csv")
SPREAD = 0.1   # eV; Alexandria against itself on one structure: median 0.004


def main():
    t = pd.read_csv(TABLE).set_index("id")
    tw = pd.read_csv(TWINS)
    a = tw[tw.source == "alexandria"].copy()
    a["y"] = a.id.map(t.dft_gap_eV)
    a["natoms"] = a.id.map(t.natoms)
    groups = {g: r for g, r in a.groupby("group") if len(r) >= 2}
    flags = {i: ("wrong", note) for i, note in KNOWN_BAD_REFERENCE.items()}

    disputed, small_high = [], 0
    for g, r in groups.items():
        if r.y.max() - r.y.min() <= SPREAD:
            continue
        disputed.append(r)
        # the K-point hypothesis: a small cell samples K or misses it; a larger cell of
        # the same layer folds K onto points its mesh does sample
        if r.sort_values("natoms").y.iloc[0] == r.y.max():
            small_high += 1
        for i, y in zip(r.id, r.y):
            if i in flags:
                continue
            others = ", ".join(f"{yy:.2f} eV ({n} atoms)" for ii, yy, n in zip(r.id, r.y, r.natoms)
                               if ii != i)
            flags[i] = ("disputed", f"the same structure is labelled {others} in other "
                                    f"Alexandria cells; this cell reads {y:.2f} eV")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "flag", "note"])
        for i in sorted(flags):
            w.writerow([i, *flags[i]])

    print(f"{len(groups)} structures held by Alexandria in more than one cell; "
          f"{len(disputed)} with labels more than {SPREAD} eV apart")
    print(f"  in {small_high} of {len(disputed)} the smallest cell has the highest label "
          "(what a mesh missing K would do)")
    for r in sorted(disputed, key=lambda r: -(r.y.max() - r.y.min()))[:25]:
        print(f"  {t.loc[r.id.iloc[0], 'formula']:12s} " + "  ".join(
            f"{n:>2d} at {y:5.2f}" for n, y in sorted(zip(r.natoms, r.y))) +
            f"   predicted {t.loc[r.id.iloc[0], 'pred_gap_eV']:.2f}")

    role = t.in_training_set
    fl = pd.Series({i: k for i, (k, _) in flags.items()})
    print(f"\nwritten {OUT}: {len(flags)} rows "
          f"({(fl == 'wrong').sum()} wrong, {(fl == 'disputed').sum()} disputed)")
    print("  by role in the gap model's training:",
          {r: int(((role.reindex(fl.index) == r)).sum()) for r in ("train", "val", "test", "unseen")})
    test = t[role == "test"]
    err = (test.pred_gap_eV - test.dft_gap_eV).abs()
    keep = ~test.index.isin(fl.index)
    print(f"\ntest split, every row: MAE {err.mean():.4f} eV (n = {len(err)}); without flagged "
          f"references: {err[keep].mean():.4f} (n = {keep.sum()})")
    unseen = t[(role == "unseen") & (t.source == "alexandria")]
    e2 = (unseen.pred_gap_eV - unseen.dft_gap_eV).abs()
    k2 = ~unseen.index.isin(fl.index)
    print(f"unseen Alexandria rows: MAE {e2.mean():.4f} -> {e2[k2].mean():.4f} without flagged")


if __name__ == "__main__":
    main()
