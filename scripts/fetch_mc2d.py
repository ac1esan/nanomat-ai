#!/usr/bin/env python
"""Fetch MC2D, the Materials Cloud database of exfoliable 2D crystals, over OPTIMADE.

MC2D (Mounet et al., Nature Nanotechnology 13, 246 (2018); Campi et al., ACS Nano 17,
11268 (2023)) holds monolayers obtained by computationally exfoliating layered bulk
crystals from experimental databases. That makes it a different population from
Alexandria 2D, most of which is hypothetical: these are layers of materials somebody
has made in bulk. About half the entries carry a band gap.

The official OPTIMADE endpoint is used - no scraping - and the result is cached in
data/mc2d.json.gz so nothing downstream depends on the service being up.

    python scripts/fetch_mc2d.py                 # refresh the cache
    python scripts/fetch_mc2d.py --export DIR    # also write DIR/<id>.vasp + id_prop.csv
                                                 # for the entries with a gap

Data: Materials Cloud MC2D; cite the two papers above and see its Materials Cloud page
for the licence before redistributing the cache.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "mc2d.json.gz")
BASE = "https://optimade.materialscloud.org/main/mc2d/v1/structures"
FIELDS = ("chemical_formula_reduced,nsites,lattice_vectors,cartesian_site_positions,"
          "species_at_sites,dimension_types,_mcloud_band_gap")
UA = "nanomat-ai (research use; https://github.com/ac1esan/nanomat-ai)"


def fetch() -> list[dict]:
    url = f"{BASE}?page_limit=200&response_fields={FIELDS}"
    rows = []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        d = json.load(urllib.request.urlopen(req, timeout=120))
        for x in d["data"]:
            a = x["attributes"]
            rows.append({"id": x["id"], "formula": a["chemical_formula_reduced"],
                         "gap": a.get("_mcloud_band_gap"), "lattice": a["lattice_vectors"],
                         "positions": a["cartesian_site_positions"],
                         "species": a["species_at_sites"], "dims": a["dimension_types"]})
        nxt = (d.get("links") or {}).get("next")
        url = nxt.get("href") if isinstance(nxt, dict) else nxt
        print(f"  {len(rows)} / {d['meta'].get('data_available')}", end="\r", flush=True)
        time.sleep(0.5)
    print()
    return rows


def to_structure(r):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice(r["lattice"]), r["species"], r["positions"],
                     coords_are_cartesian=True)


def load(path: str = OUT) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help="write POSCARs and id_prop.csv for entries with a gap")
    ap.add_argument("--cached", action="store_true", help="use the cache, do not download")
    args = ap.parse_args()
    if not args.cached:
        rows = fetch()
        with gzip.open(OUT, "wt") as f:
            json.dump(rows, f)
        print(f"written {OUT}: {len(rows)} entries")
    rows = load()
    gapped = [r for r in rows if r["gap"] is not None]
    print(f"{len(rows)} entries, {len(gapped)} with a band gap, "
          f"{sum(r['gap'] > 0.01 for r in gapped)} above 0.01 eV")
    if args.export:
        os.makedirs(args.export, exist_ok=True)
        with open(os.path.join(args.export, "id_prop.csv"), "w") as f:
            for r in gapped:
                st = to_structure(r)
                st.to(filename=os.path.join(args.export, f"{r['id']}.vasp"), fmt="poscar")
                f.write(f"{r['id']}.vasp,{r['gap']}\n")
        print(f"exported {len(gapped)} structures to {args.export}")


if __name__ == "__main__":
    main()
