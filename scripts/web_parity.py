#!/usr/bin/env python
"""Cases for the browser-vs-Python parity check.

Picks structures from the reference examples and from each database the project
uses, writes each as the text a user would upload - POSCAR, and for a share of them
a symmetrised CIF, which exercises the symmetry-operation expansion in the page -
and records what Predictor returns for it. scripts/web_parity.mjs then runs the
same texts through docs/js/nanomat.js and compares field by field.

    python scripts/web_parity.py --per-source 400 --out runs/web/parity.json
    python scripts/web_parity.py --examples-only --out /tmp/parity.json   # CI
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# (folder, file-name prefix); MC2D comes from its own cache, data/mc2d.json.gz
SOURCES = {"alexandria": ("alignn_data_hull", "agm"), "c2db": ("alignn_data_c2db", ""),
           "jarvis_dft2d": ("alignn_data", "")}
FIELDS = ("natoms", "gap", "unc", "verdict", "latent_distance", "p_metal", "p_metastable",
          "interval90", "typical_error", "gap_type", "p_indirect", "is_metal_like",
          "gap_quasiparticle", "gap_quasiparticle_direct", "exciton_binding", "exp_gap_est",
          "work_function", "work_function_unc", "work_function_latent", "work_function_verdict",
          "work_function_interval90", "electron_affinity", "ionisation_potential")


def cases(args):
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    from nanomat.predict import read_structure
    rng = random.Random(0)

    def as_cif(st):
        # CifWriter with symprec writes the refined cell and its symmetry operations,
        # so Python must predict the structure read back from that text - exactly
        # what a Python user uploading the same file would get
        text = str(CifWriter(st, symprec=0.01))
        return text, Structure.from_str(text, fmt="cif")

    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "examples", "*.vasp"))):
        st = read_structure(p)
        out.append({"name": "examples/" + os.path.basename(p), "fmt": "poscar",
                    "text": open(p).read(), "st": st})
        text, st_c = as_cif(st)
        out.append({"name": "examples/" + os.path.basename(p) + ".cif", "fmt": "cif",
                    "text": text, "st": st_c})
    if args.examples_only:
        return out
    for src, (folder, prefix) in SOURCES.items():
        path = os.path.join(ROOT, folder)
        if not os.path.isdir(path):
            print(f"  {src}: {folder} not found, skipped")
            continue
        files = sorted(f for f in os.listdir(path) if f.endswith(".vasp") and f.startswith(prefix))
        for f in rng.sample(files, min(args.per_source, len(files))):
            st = read_structure(os.path.join(path, f))
            name = f"{src}/{f}"
            if rng.random() < args.cif_share:
                # symmetrised: the page has to expand the operations back into the cell
                try:
                    text, st = as_cif(st)
                    fmt = "cif"
                except Exception:
                    text, fmt = open(os.path.join(path, f)).read(), "poscar"
            else:
                text, fmt = open(os.path.join(path, f)).read(), "poscar"
            out.append({"name": name, "fmt": fmt, "text": text, "st": st})
    try:
        sys.path.insert(0, HERE)
        from fetch_mc2d import load, to_structure
        from pymatgen.io.vasp import Poscar
        entries = load()
        for r in rng.sample(entries, min(args.per_source, len(entries))):
            st = to_structure(r)
            out.append({"name": f"mc2d/{r['id']}", "fmt": "poscar",
                        "text": str(Poscar(st)), "st": st})
    except FileNotFoundError:
        print("  mc2d: data/mc2d.json.gz not found, skipped")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=400)
    ap.add_argument("--cif-share", type=float, default=0.25)
    ap.add_argument("--examples-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "web", "parity.json"))
    args = ap.parse_args()
    from nanomat import Predictor
    P = Predictor(verbose=False)
    cs = cases(args)
    res = dict(P.run_many([(c["name"], c["st"]) for c in cs], device=args.device))
    rows = []
    for c in cs:
        p = res.get(c["name"])
        row = {"name": c["name"], "fmt": c["fmt"], "text": c["text"]}
        if p is None:
            row["python"] = None
        else:
            row["python"] = {k: getattr(p, k) for k in FIELDS}
            row["python"]["is_layer"] = bool(p.layer.get("is_layer"))
            row["python"]["padded"] = bool(p.layer.get("padded"))
            row["python"]["neighbours"] = [{"id": n["id"], "distance": n["distance"]}
                                           for n in p.neighbours]
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rows, open(args.out, "w"))
    print(f"wrote {len(rows)} cases ({sum(r['fmt'] == 'cif' for r in rows)} as CIF) to {args.out}")


if __name__ == "__main__":
    main()
