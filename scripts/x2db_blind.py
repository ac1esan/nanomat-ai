#!/usr/bin/env python
"""Predictions for the experimentally realised 2D materials in X2DB, archived before
any measured property is in it.

X2DB (Akhound et al., ACS Nano 20, 12008 (2026); https://x2db.fysik.dtu.dk/) lists 2D
materials that have been made as monolayers or few-layer films, and links them, where
it can, to a C2DB structure by uid. It holds no measured band gaps yet; its authors
are adding experimental property data. Predictions written now, with their verdicts,
and archived under a DOI, become a blind test once those data arrive: of the model,
and of whether its trust verdict ranks the error against experiment.

Structures come from the full C2DB release (c2db.db, an ASE database; see
https://2dhub.org/c2db/c2db.html), looked up by the uids X2DB links to. Neither the
structures nor X2DB's metadata are redistributed here: both are CC BY-NC 4.0, and the
output lists only uids, formulas and this model's predictions.

    python scripts/x2db_blind.py --c2db data/c2db.db --x2db runs/x2db/x2db_rows.json \
        --out runs/x2db/predictions --device cuda

Cite C2DB (Haastrup et al., 2D Mater. 5, 042002 (2018); Gjerding et al., 2D Mater. 8,
044002 (2021)) and X2DB (above; Akhound, Sauer & Thygesen, Nat. Rev. Mater. (2026)).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2db", required=True, help="c2db.db from https://wiki.fysik.dtu.dk/c2db.db")
    ap.add_argument("--x2db", required=True, help="X2DB table rows (JSON: header + rows of cells)")
    ap.add_argument("--x2db-date", default="2026-10-13", help="when the X2DB table was read")
    ap.add_argument("--out", required=True, help="output prefix: writes <out>.csv and <out>.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ase.db import connect
    from pymatgen.io.ase import AseAtomsAdaptor
    from nanomat import Predictor
    from nanomat.predict import tier_key

    rows = json.load(open(args.x2db))["rows"]
    entries = {}
    for r in rows:
        formula, year, methods, _, _, uid = r["cells"][:6]
        if uid and uid != "C2DB UID":   # the table repeats its header row
            e = entries.setdefault(uid, {"x2db_formula": formula, "years": set(), "methods": set()})
            e["years"].add(year)
            e["methods"].update(m.strip() for m in methods.split(",") if m.strip())
    print(f"X2DB: {len(rows)} rows, {len(entries)} distinct C2DB uids")

    db = connect(args.c2db)
    structures, missing = [], []
    for uid in sorted(entries):
        hits = list(db.select(uid=uid))
        if not hits:
            missing.append(uid)
            continue
        row = hits[0]
        entries[uid]["olduid"] = row.get("olduid")
        structures.append((uid, AseAtomsAdaptor.get_structure(row.toatoms())))
    print(f"found in c2db.db: {len(structures)}; missing: {missing}")

    t0 = time.time()
    P = Predictor(os.path.join(ROOT, "weights"), verbose=False)
    preds = P.run_many(structures, device=args.device)
    print(f"predicted {len(preds)} structures in {time.time() - t0:.1f}s")

    out_rows = []
    for uid, p in preds:
        e = entries[uid]
        base = {"c2db_uid": uid, "c2db_olduid": e.get("olduid"), "x2db_formula": e["x2db_formula"],
                "x2db_years": " ".join(sorted(e["years"])),
                "x2db_synthesis": "; ".join(sorted(e["methods"]))}
        if p is None:
            out_rows.append({**base, "verdict": "no graph (not a 2D layer?)"})
            continue
        row = p.as_row()
        row["tier"] = tier_key(p.verdict)
        row["gap_range90_low_eV"] = None if p.interval90 is None or row["tier"] == "out_of_domain" \
            else round(p.gap - p.interval90, 3)
        row["gap_range90_high_eV"] = None if p.interval90 is None or row["tier"] == "out_of_domain" \
            else round(p.gap + p.interval90, 3)
        out_rows.append({**base, **row})

    keys = list(dict.fromkeys(k for r in out_rows for k in r))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)

    tiers = {}
    for r in out_rows:
        tiers[r.get("tier", "none")] = tiers.get(r.get("tier", "none"), 0) + 1
    weights = sorted(f for f in os.listdir(os.path.join(ROOT, "weights")) if f.endswith(".pt") and not f.startswith("._"))
    manifest = {
        "purpose": "predictions archived before X2DB holds measured properties: a blind test",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "x2db_table_read": args.x2db_date, "x2db_rows": len(rows),
        "c2db_uids_linked": len(entries), "c2db_uids_found": len(structures),
        "c2db_uids_missing": missing, "c2db_db_sha256": sha256(args.c2db),
        "weights_sha256": {w: sha256(os.path.join(ROOT, "weights", w)) for w in weights},
        "tiers": tiers,
        "what_is_predicted": "PBE band gap without spin-orbit coupling of the C2DB monolayer "
                             "structure, plus G0W0-level and optical-gap estimates from latent "
                             "heads fitted on C2DB G0W0/BSE; the optical estimate is the number "
                             "to compare with optical measurements",
        "not_redistributed": "C2DB structures and X2DB metadata (both CC BY-NC 4.0)",
    }
    json.dump(manifest, open(args.out + ".json", "w"), indent=1)
    print(f"tiers: {tiers}")
    print(f"written {args.out}.csv and {args.out}.json")


if __name__ == "__main__":
    main()
