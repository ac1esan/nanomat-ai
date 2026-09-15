#!/usr/bin/env python
"""Turn screening_table.csv into the compact payload the static browser loads.

Writes two files under docs/data/:
  screening.csv  - one row per structure, short column names, categoricals encoded
                   as small integers, and everything derivable dropped (the 90%
                   interval, the experimental estimate and the absolute error are
                   recomputed in the browser from meta.json).
  meta.json      - calibration constants, verdict thresholds, source and tier
                   labels, dataset counts, and the per-element trust map.

    python scripts/build_site_data.py

The stability flag is not in screening_table.csv, so it is recovered here from the
`e_above_hull <= 0.1` export: those ids are the population the shipped calibration
was actually fitted on, and the browser defaults to them.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat.predict import A_CORR, B_CORR, METAL_GAP  # noqa: E402

TABLE = os.path.join(ROOT, "screening_table.csv")
STABLE_INDEX = os.path.join(ROOT, "alignn_data_alex_2d", "id_prop.csv")
ENSEMBLE = os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.pt")
OUT_DIR = os.path.join(ROOT, "docs", "data")

SOURCES = ["alexandria", "c2db", "jarvis_dft2d"]
TIERS = ["reliable", "check", "out-of-domain"]
ROLES = ["unseen", "train", "val", "test"]
TYPES = ["direct", "indirect"]


def tier_code(v: str) -> int:
    return 0 if v.startswith("reliable") else (1 if v.startswith("check") else 2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(TABLE)
    print(f"read {len(df)} rows from {os.path.basename(TABLE)}")

    # --- stability flag: which Alexandria ids passed e_above_hull <= 0.1 -------
    stable = set()
    if os.path.exists(STABLE_INDEX):
        for line in open(STABLE_INDEX):
            stable.add(os.path.splitext(line.split(",")[0])[0])
        print(f"stable (ehull<=0.1) reference ids: {len(stable)}")
    else:
        print("WARNING: stable index missing; every row will be marked non-stable")
    df["b"] = [1 if (s == "alexandria" and i in stable) else 0
               for s, i in zip(df["source"], df["id"])]

    # --- compact encoding -----------------------------------------------------
    out = pd.DataFrame({
        "i": df["id"],
        "s": df["source"].map({v: k for k, v in enumerate(SOURCES)}).fillna(0).astype(int),
        "f": df["formula"],
        "e": df["elements"],
        "n": df["natoms"].astype(int),
        "d": df["dft_gap_eV"].round(3),
        "g": df["pred_gap_eV"].round(3),
        "u": df["uncertainty_eV"].round(3),
        "l": df["latent_distance"].round(3),
        "m": df["p_metal"].round(2),
        "t": df["gap_type"].map({v: k for k, v in enumerate(TYPES)}),
        "p": df["p_indirect"].round(2),
        "w": df["verdict"].map(tier_code).astype(int),
        "r": df["in_training_set"].map({v: k for k, v in enumerate(ROLES)}).fillna(0).astype(int),
        "b": df["b"],
    })
    csv_path = os.path.join(OUT_DIR, "screening.csv")
    out.to_csv(csv_path, index=False)
    raw = os.path.getsize(csv_path)
    gz = len(gzip.compress(open(csv_path, "rb").read(), 6))
    print(f"wrote {csv_path}: {raw/1e6:.2f} MB, ~{gz/1e6:.2f} MB gzipped over the wire")

    # --- calibration straight from the shipped checkpoint ---------------------
    import torch
    ck = torch.load(ENSEMBLE, map_location="cpu")
    cal = dict(ck.get("calibration", {}))
    cal.pop("fitted_on", None)
    cal.pop("verified_on", None)
    for key, fname in (("type_threshold", "cgcnn_2d_typed.pt"),
                       ("metal_threshold", "cgcnn_2d_metal.pt")):
        fp = os.path.join(ROOT, "weights", fname)
        if os.path.exists(fp):
            cal[key] = float(torch.load(fp, map_location="cpu").get("threshold", 0.5))

    # --- trust map: mean error per element, Alexandria rows never trained on ---
    unseen = df[(df.source == "alexandria") & (df.in_training_set != "train")].dropna(
        subset=["error_eV"])
    per_el: dict[str, list[float]] = defaultdict(list)
    for els, err in zip(unseen["elements"], unseen["error_eV"]):
        for el in str(els).split():
            per_el[el].append(err)
    # 20 is the floor for reporting an element at all: below it a single bad
    # structure swings the mean and distorts the colour scale for everything else.
    # Elements between 20 and SPARSE are still shown but marked as thinly measured.
    MIN_N, SPARSE = 20, 60
    trust = {el: {"n": len(v), "mae": round(float(np.mean(v)), 3),
                  "sparse": 1 if len(v) < SPARSE else 0}
             for el, v in sorted(per_el.items()) if len(v) >= MIN_N}
    print(f"trust map: {len(trust)} elements over {len(unseen)} unseen Alexandria rows")

    # --- periodic table layout for the trust map (row, group), f-block on 8/9 ---
    from pymatgen.core.periodic_table import Element
    layout = {}
    for el in Element:
        if el.Z > 103:
            continue
        try:
            row, grp = el.row, el.group
        except Exception:
            continue
        if 57 <= el.Z <= 71:
            row, grp = 8, el.Z - 57 + 3
        elif 89 <= el.Z <= 103:
            row, grp = 9, el.Z - 89 + 3
        layout[el.symbol] = [row, grp]

    meta = {
        "generated_from": "screening_table.csv",
        "n_rows": int(len(out)),
        "sources": SOURCES,
        "tiers": TIERS,
        "roles": ROLES,
        "types": TYPES,
        "counts": {
            "by_source": {s: int((df.source == s).sum()) for s in SOURCES},
            "by_tier": {TIERS[k]: int((out.w == k).sum()) for k in range(3)},
            "stable": int(out.b.sum()),
            "default_view": int(((out.b == 1) & (out.r != 1)).sum()),
        },
        "calibration": cal,
        "pbe_correction": {"a": A_CORR, "b": B_CORR},
        "metal_gap_eV": METAL_GAP,
        "trust_map": trust,
        "ptable": layout,
        "trust_map_basis": {
            "n_structures": int(len(unseen)),
            "source": "alexandria",
            "min_n": MIN_N, "sparse_below": SPARSE,
            "note": "mean absolute error against the PBE reference, over Alexandria "
                    "structures the ensemble never trained on; elements measured on "
                    "fewer than 20 structures are omitted, and those under 60 are "
                    "marked as thinly measured",
        },
    }
    meta_path = os.path.join(OUT_DIR, "meta.json")
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"wrote {meta_path}: {os.path.getsize(meta_path)/1000:.0f} kB")
    print(f"\ndefault view (stable, not trained on): {meta['counts']['default_view']} rows")


if __name__ == "__main__":
    main()
