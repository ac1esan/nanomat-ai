#!/usr/bin/env python
"""The full C2DB release, read from its ASE database file.

The JARVIS mirror of C2DB holds 3 520 entries and one gap, the PBE gap with spin-orbit
coupling. The release itself (https://wiki.fysik.dtu.dk/c2db.db, see
https://2dhub.org/c2db/c2db.html) holds 16 905 entries and, for the 4 514
semiconductors among them, the direct gap with and without spin-orbit coupling
(gap_dir, gap_dir_nosoc), HSE06 gaps (gap_hse), and band-edge effective masses
(emass_cbm, emass_vbm). Entries are keyed by the new-style uid ("1MoS2-1"); "olduid"
is the one the JARVIS mirror uses.

    python scripts/c2db_full.py --summary
    python scripts/c2db_full.py --export data_c2db_emass_e --target emass_cbm --log10

The database is CC BY-NC 4.0 and is not redistributed here. Cite Haastrup et al.,
2D Materials 5, 042002 (2018) and Gjerding et al., 2D Materials 8, 044002 (2021).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "data", "c2db.db")
METAL_GAP = 0.01
# effective masses outside this window are flat bands or failed fits, not transport
# masses: C2DB stores inf for some, and values in the thousands for others
MASS_MIN, MASS_MAX = 0.01, 20.0
# the 1H reference monolayers, by new-style uid
TMD = {"1MoS2-1": "MoS2", "1MoSe2-1": "MoSe2", "1WS2-1": "WS2", "1WSe2-1": "WSe2"}


def records(db_path: str = DB, semiconductors: bool = True, nonmagnetic: bool = True):
    """Yield one dict per entry: uid, olduid, formula, magnetic, kv, row."""
    from ase.db import connect
    for row in connect(db_path).select():
        kv = row.key_value_pairs
        gap = kv.get("gap")
        if semiconductors and (gap is None or gap <= METAL_GAP):
            continue
        if nonmagnetic and kv.get("is_magnetic"):
            continue
        yield {"uid": kv.get("uid"), "olduid": kv.get("olduid"), "formula": row.formula,
               "magnetic": bool(kv.get("is_magnetic")), "kv": kv, "row": row}


def to_structure(rec):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(rec["row"].toatoms())


def embed(P, structures: list, batch: int = 128):
    """Concatenated member embeddings of the gap ensemble, on the models' own device.

    The same features the shipped latent heads read: [5 x 128 embedding, gap].
    """
    import numpy as np
    import torch
    from torch_geometric.data import Batch
    dev = next(P.models[0].parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(structures), batch):
            b = Batch.from_data_list([P.graph(s) for s in structures[i:i + batch]]).to(dev)
            out.append(torch.cat([m.encode(b) for m in P.models], dim=1).cpu().numpy())
    return np.concatenate(out)


def predict_in_domain(P, recs: list, device: str = "cpu"):
    """Gap predictions and embeddings for the records the gap model does not disown.

    Returns (kept records, {uid: Structure}, {uid: Prediction}, embeddings in the
    order of the kept records). Heads are fitted and applied only in-domain: off it
    they extrapolate badly, and the tool shows nothing there anyway.
    """
    sts = {r["uid"]: to_structure(r) for r in recs}
    res = dict(P.run_many(list(sts.items()), batch_size=256, device=device))
    keep = [r for r in recs if res.get(r["uid"]) is not None
            and not res[r["uid"]].verdict.startswith("out-of-domain")]
    return keep, sts, res, embed(P, [sts[r["uid"]] for r in keep])


def usable_mass(m) -> bool:
    return (isinstance(m, (int, float)) and math.isfinite(m)
            and MASS_MIN <= abs(m) <= MASS_MAX)


def export(out: str, target: str, log10: bool, db_path: str = DB) -> int:
    """POSCARs + id_prop.csv, the format train_cgcnn.py reads."""
    os.makedirs(out, exist_ok=True)
    n = skipped = 0
    with open(os.path.join(out, "id_prop.csv"), "w") as fh:
        for rec in records(db_path):
            v = rec["kv"].get(target)
            if target.startswith("emass"):
                if not usable_mass(v):
                    skipped += 1
                    continue
                v = abs(v)
            elif v is None:
                skipped += 1
                continue
            y = math.log10(v) if log10 else v
            to_structure(rec).to(filename=os.path.join(out, f"{rec['uid']}.vasp"), fmt="poscar")
            fh.write(f"{rec['uid']}.vasp,{y}\n")
            n += 1
    print(f"exported {n} structures to {out} ({target}{', log10' if log10 else ''}); "
          f"skipped {skipped} without a usable value")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--export")
    ap.add_argument("--target", default="emass_cbm")
    ap.add_argument("--log10", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.db):
        raise SystemExit(f"missing {args.db}: download https://wiki.fysik.dtu.dk/c2db.db")
    if args.summary:
        recs = list(records(args.db))
        print(f"{len(recs)} non-magnetic semiconductors")
        for k in ("gap_dir_nosoc", "gap_hse", "gap_gw", "emass_cbm", "emass_vbm"):
            ok = [r for r in recs if (usable_mass(r["kv"].get(k)) if k.startswith("emass")
                                      else r["kv"].get(k) is not None)]
            print(f"  {k:14s} {len(ok)}")
    if args.export:
        export(args.export, args.target, args.log10, args.db)


if __name__ == "__main__":
    sys.exit(main())
