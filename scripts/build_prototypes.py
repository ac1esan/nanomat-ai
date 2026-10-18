#!/usr/bin/env python
"""Relaxed geometry of every prototype pair in the table, for the page's prototype builder.

The builder on the "Your structure" tab makes a 1H-MX2, 1T-MX2 or honeycomb cell from
a prototype, the elements and a lattice constant. The lattice constant is the number
that matters (scripts/geometry_sensitivity.py: an ideal chalcogen height costs 0.001 eV
on four TMDs, a lattice constant from covalent radii 0.4 eV), so whenever the table
already holds a DFT-relaxed cell of that pair the builder fills in its lattice constant
and layer thickness instead of guessing them.

For each (prototype, element A, element B) the cell is taken from Alexandria if it has
one, then C2DB, then JARVIS dft_2d: the first two are PBE, the level the model learned
from. Among several cells of one source, the one with the smallest ensemble spread,
the same choice the Families tab makes.

    python scripts/build_prototypes.py          # -> docs/data/prototypes.json
"""

from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FOLDERS = {"alexandria": "alignn_data_alex_2d_eh02", "c2db": "alignn_data_c2db",
           "jarvis_dft2d": "alignn_data"}
PREFERENCE = ["alexandria", "c2db", "jarvis_dft2d"]


def geometry(path: str, family: str) -> tuple[float, float]:
    """(lattice constant, layer thickness) in angstrom.

    Thickness is the vertical distance between the two outer atomic planes: the
    chalcogen-chalcogen distance for MX2, the buckling for a honeycomb (0 if planar).
    """
    from pymatgen.core import Structure
    st = Structure.from_file(path)
    lat = st.lattice.matrix
    # the vacuum axis is the lattice vector most nearly perpendicular to the other two
    normal = [abs(sum(x * y for x, y in zip(lat[i], _cross(lat[(i + 1) % 3], lat[(i + 2) % 3]))))
              / (_norm(lat[i]) * _norm(_cross(lat[(i + 1) % 3], lat[(i + 2) % 3]))) for i in range(3)]
    axis = max(range(3), key=lambda i: normal[i])
    n = _cross(lat[(axis + 1) % 3], lat[(axis + 2) % 3])
    n = [x / _norm(n) for x in n]
    # heights along the normal, unwrapped so a layer split across the cell edge stays whole
    c = _norm(lat[axis]) * normal[axis]
    h = sorted((sum(a * b for a, b in zip(s.coords, n))) % c for s in st)
    # gap above each atom; the last one wraps across the cell edge to the first atom
    gaps = [h[i + 1] - h[i] for i in range(len(h) - 1)] + [h[0] + c - h[-1]]
    k = max(range(len(h)), key=lambda i: gaps[i])
    thickness = c - gaps[k]
    a = _norm(lat[(axis + 1) % 3])
    return round(a, 4), round(thickness, 4)


def _cross(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def _norm(u):
    return sum(x * x for x in u) ** 0.5


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "screening_table.csv"))) if r["family"]]
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["family"], r["family_a"], r["family_b"])
        rank = (PREFERENCE.index(r["source"]), float(r["uncertainty_eV"]))
        if key not in best or rank < best[key]["rank"]:
            best[key] = {"rank": rank, "row": r}
    out: dict[str, dict] = {}
    for (fam, a_el, b_el), v in sorted(best.items()):
        r = v["row"]
        path = os.path.join(ROOT, FOLDERS[r["source"]], r["id"] + ".vasp")
        a, t = geometry(path, fam)
        out.setdefault(fam, {})[f"{a_el}|{b_el}"] = {"a": a, "t": t, "id": r["id"], "source": r["source"]}
    dest = os.path.join(ROOT, "docs", "data", "prototypes.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"{sum(len(v) for v in out.values())} pairs -> {dest} "
          f"({', '.join(f'{k} {len(v)}' for k, v in out.items())})")


if __name__ == "__main__":
    sys.exit(main())
