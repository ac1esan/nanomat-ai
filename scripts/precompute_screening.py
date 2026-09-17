#!/usr/bin/env python
"""Run the model over whole databases once, so the common case needs no file upload.

Produces one table with a prediction per structure. Every row carries the two
out-of-domain signals, the calibrated interval, and - importantly - whether that
structure was part of the shipped ensemble's training set, so memorisation is never
mistaken for prediction.

    python scripts/precompute_screening.py \
        --data alignn_data_alex_2d_eh02:alexandria alignn_data_c2db:c2db alignn_data:jarvis_dft2d \
        --split weights/cgcnn_2d_ensemble.split.json \
        --wf-ref data_c2db_wf:weights/cgcnn_2d_workfunction.split.json \
        --out screening_table.csv

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
from nanomat.families import classify  # noqa: E402


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
    ap.add_argument("--wf-ref", metavar="FOLDER:SPLIT",
                    help="the work-function training folder and its split.json, e.g. "
                         "data_c2db_wf:weights/cgcnn_2d_workfunction.split.json. Rows whose "
                         "file appears there get a reference work function and a tag saying "
                         "which of the two models had already seen them - the second model "
                         "has its own training set, so one tag cannot cover both")
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

    wf_ref, wf_role = {}, {}
    if args.wf_ref:
        folder, _, spath = args.wf_ref.partition(":")
        for fname, value in read_index(folder):
            if value is not None:
                wf_ref[fname] = value
        sp = json.load(open(spath))
        for name in ("train", "val", "test"):
            for f in sp[name]:
                wf_role[f] = name
        print(f"work-function reference: {len(wf_ref)} values, "
              f"{len(wf_role)} of them in that model's split")

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
            # structural prototype, so predictions can be pivoted into the families
            # people actually think in (MX2 across the chalcogens, and so on)
            fams = {k: classify(st) for k, st in items}
            for key, r in P.run_many(items, batch_size=args.batch):
                if r is None:
                    continue
                rows.append({
                    "id": os.path.splitext(key)[0], "source": label,
                    "formula": r.formula,
                    "elements": elements.get(key),
                    "family": fams.get(key, (None, None, None))[0],
                    "family_a": fams.get(key, (None, None, None))[1],
                    "family_b": fams.get(key, (None, None, None))[2],
                    "natoms": r.natoms,
                    "dft_gap_eV": refs.get(key),
                    "pred_gap_eV": round(r.gap, 3),
                    "uncertainty_eV": round(r.unc, 3),
                    "interval90_eV": None if r.interval90 is None else round(r.interval90, 3),
                    "latent_distance": None if r.latent_distance is None else round(r.latent_distance, 3),
                    "p_metal": None if r.p_metal is None else round(r.p_metal, 2),
                    "gap_type": r.gap_type,
                    "p_indirect": None if r.p_indirect is None else round(r.p_indirect, 2),
                    # the corrected gaps now come from linear heads on the model's
                    # own latent space, which the static browser cannot evaluate, so
                    # they travel in the data rather than being derived in the page
                    "gap_quasiparticle_eV": None if r.gap_quasiparticle is None else round(r.gap_quasiparticle, 3),
                    "exciton_binding_eV": None if r.exciton_binding is None else round(r.exciton_binding, 3),
                    "exp_gap_est_eV": None if r.exp_gap_est is None else round(r.exp_gap_est, 3),
                    "verdict": r.verdict,
                    "in_training_set": role.get(key, "unseen"),
                    # second property: its own number, its own spread, its own verdict
                    # against its own training set, and the band edges the pair unlocks
                    "work_function_eV": None if r.work_function is None else round(r.work_function, 3),
                    "wf_uncertainty_eV": None if r.work_function_unc is None else round(r.work_function_unc, 3),
                    "wf_latent_distance": None if r.work_function_latent is None else round(r.work_function_latent, 3),
                    "wf_verdict": r.work_function_verdict,
                    "electron_affinity_eV": None if r.electron_affinity is None else round(r.electron_affinity, 3),
                    "ionisation_potential_eV": None if r.ionisation_potential is None else round(r.ionisation_potential, 3),
                    "dft_wf_eV": wf_ref.get(key),
                    "in_wf_training_set": wf_role.get(key, "unseen") if wf_ref else None,
                })
            done = len(rows)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {start + len(chunk)}/{len(index)}  ({done} rows, {rate:.0f}/s)", flush=True)

    df = pd.DataFrame(rows)
    # to_numeric: a column of all-None (a source with no reference for that property)
    # arrives as object dtype and would not subtract
    for col in ("dft_gap_eV", "dft_wf_eV", "work_function_eV"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["error_eV"] = (df["pred_gap_eV"] - df["dft_gap_eV"]).abs().round(3)
    df["wf_error_eV"] = (df["work_function_eV"] - df["dft_wf_eV"]).abs().round(3)
    df = df.sort_values("pred_gap_eV").reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(f"\n{len(df)} rows -> {args.out}  ({time.time() - t0:.0f}s)")
    fam = df[df.family.notna()]
    if len(fam):
        print(f"\nstructural prototypes recognised in {len(fam)} rows:")
        for name, g in fam.groupby("family"):
            pairs = g.groupby(["family_a", "family_b"]).ngroups
            print(f"  {name:10s} {len(g):5d} rows, {pairs} distinct element pairs")

    if df["work_function_eV"].notna().any():
        edges = df["electron_affinity_eV"].notna().sum()
        print(f"\nwork function predicted for {df['work_function_eV'].notna().sum()} rows; "
              f"band edges shown for {edges} ({100 * edges / len(df):.0f}%)")
        wv = df["wf_verdict"].str.split(" (", regex=False).str[0]
        for src, g in df.groupby("source"):
            share = wv[g.index].value_counts(normalize=True)
            print(f"  {src:12s} " + "  ".join(f"{k} {100*v:.0f}%" for k, v in share.items()))
        print("  the second model was trained on C2DB alone, so on Alexandria it is "
              "frequently\n  outside its own training distribution and says so - the edges "
              "are withheld there.")

        ref = df.dropna(subset=["wf_error_eV"])
        if len(ref):
            import numpy as np
            print(f"\nwork function against its DFT reference, {len(ref)} rows with one:")
            for tag, g in ref.groupby("in_wf_training_set"):
                print(f"  {tag:7s} n={len(g):5d}  MAE={g['wf_error_eV'].mean():.3f} eV")
            held = ref[ref.in_wf_training_set.isin(["test", "unseen"])]
            if len(held):
                print("  does its own verdict rank its own error? (rows it never trained on)")
                tiers = held["wf_verdict"].str.split(" (", regex=False).str[0]
                for t in ("reliable", "check", "out-of-domain"):
                    g = held[tiers == t]
                    if len(g):
                        print(f"    {t:14s} n={len(g):5d}  MAE={g['wf_error_eV'].mean():.3f}  "
                              f"p90={np.quantile(g['wf_error_eV'], 0.9):.3f}")

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
