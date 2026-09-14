#!/usr/bin/env python
"""Run the model over whole databases once, so the common case needs no file upload.

Produces one table with a prediction per structure. Every row carries the two
out-of-domain signals, the calibrated interval, and - importantly - whether that
structure was part of the shipped ensemble's training set, so memorisation is never
mistaken for prediction.

    python scripts/precompute_screening.py \
        --data alignn_data_alex_2d_eh02:alexandria alignn_data_c2db:c2db alignn_data:jarvis_dft2d \
        --split weights/cgcnn_2d_ensemble.split.json --out screening_table.csv

Each --data entry is `folder` or `folder:source_label`. Runs on CPU; ~28 000
structures take a few minutes on a laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from pymatgen.core import Structure

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat import Predictor  # noqa: E402


def read_index(folder: str) -> list[tuple[str, float | None]]:
    """(filename, reference DFT gap) pairs from id_prop.csv, or every .vasp if absent."""
    idx = os.path.join(folder, "id_prop.csv")
    if not os.path.exists(idx):
        return [(f, None) for f in sorted(os.listdir(folder)) if f.endswith(".vasp")]
    rows = []
    for line in open(idx):
        parts = line.strip().split(",")
        if len(parts) >= 2:
            rows.append((parts[0], float(parts[1])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="folders, optionally as folder:source_label")
    ap.add_argument("--split", help="split.json of the shipped ensemble, to tag training rows")
    ap.add_argument("--out", default="screening_table.csv")
    ap.add_argument("--chunk", type=int, default=2000, help="structures held in RAM at once")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--weights", help="weights directory (default ./weights)")
    args = ap.parse_args()

    role = {}
    if args.split:
        sp = json.load(open(args.split))
        for name in ("train", "val", "test"):
            for f in sp[name]:
                role[f] = name

    P = Predictor(args.weights)
    rows, t0 = [], time.time()
    for entry in args.data:
        folder, _, label = entry.partition(":")
        label = label or os.path.basename(folder.rstrip("/"))
        index = read_index(folder)
        print(f"\n{label}: {len(index)} structures from {folder}")
        for start in range(0, len(index), args.chunk):
            chunk = index[start:start + args.chunk]
            items, refs = [], {}
            for fname, ref in chunk:
                try:
                    items.append((fname, Structure.from_file(os.path.join(folder, fname))))
                    refs[fname] = ref
                except Exception:
                    continue
            elements = {k: " ".join(sorted(e.symbol for e in st.composition.elements))
                        for k, st in items}
            for key, r in P.run_many(items, batch_size=args.batch):
                if r is None:
                    continue
                rows.append({
                    "id": os.path.splitext(key)[0], "source": label,
                    "formula": r.formula,
                    "elements": elements.get(key),
                    "natoms": r.natoms,
                    "dft_gap_eV": refs.get(key),
                    "pred_gap_eV": round(r.gap, 3),
                    "uncertainty_eV": round(r.unc, 3),
                    "interval90_eV": None if r.interval90 is None else round(r.interval90, 3),
                    "latent_distance": None if r.latent_distance is None else round(r.latent_distance, 3),
                    "p_metal": None if r.p_metal is None else round(r.p_metal, 2),
                    "gap_type": r.gap_type,
                    "p_indirect": None if r.p_indirect is None else round(r.p_indirect, 2),
                    "exp_gap_est_eV": None if r.exp_gap_est is None else round(r.exp_gap_est, 3),
                    "verdict": r.verdict,
                    "in_training_set": role.get(key, "unseen"),
                })
            done = len(rows)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {start + len(chunk)}/{len(index)}  ({done} rows, {rate:.0f}/s)", flush=True)

    df = pd.DataFrame(rows)
    df["error_eV"] = (df["pred_gap_eV"] - df["dft_gap_eV"]).abs().round(3)
    df = df.sort_values("pred_gap_eV").reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(f"\n{len(df)} rows -> {args.out}  ({time.time() - t0:.0f}s)")
    print("\nby verdict:")
    print(df["verdict"].str.split(" (", regex=False).str[0].value_counts().to_string())
    seen = df[df.in_training_set == "train"]
    unseen = df[df.in_training_set.isin(["test", "unseen"])].dropna(subset=["error_eV"])
    if len(seen) and len(unseen):
        print(f"\nMAE on rows the ensemble trained on : {seen['error_eV'].mean():.3f} eV "
              f"(n={len(seen)})")
        print(f"MAE on rows it never saw            : {unseen['error_eV'].mean():.3f} eV "
              f"(n={len(unseen)}) <- the honest one")
        print("  (mixing sources mixes DFT functionals; compare within one source)")

    if len(unseen):
        import numpy as np
        print("\ndoes the verdict rank the error?")
        tiers = unseen["verdict"].str.split(" (", regex=False).str[0]
        for t in ("reliable", "check", "out-of-domain"):
            g = unseen[tiers == t]
            if len(g):
                print(f"  {t:14s} n={len(g):6d}  MAE={g['error_eV'].mean():.3f}  "
                      f"p90={np.quantile(g['error_eV'], 0.9):.3f}")

        cov = unseen.dropna(subset=["interval90_eV"])
        if len(cov):
            hit = float(np.mean(cov["error_eV"] <= cov["interval90_eV"]))
            print(f"\ncalibration check: {100*hit:.1f}% of unseen rows fall inside the 90% "
                  "interval")
            if hit < 0.85:
                print("  WARNING: the shipped calibration was fitted on stable "
                      "(e_above_hull <= 0.1) Alexandria 2D semiconductors. On a population "
                      "outside that - metastable structures, other functionals - the "
                      "intervals and the per-tier error figures are optimistic. The verdict "
                      "still ranks correctly; only its absolute numbers do not transfer. "
                      "Re-run scripts/calibrate_uncertainty.py on the population you screen.")


if __name__ == "__main__":
    main()
