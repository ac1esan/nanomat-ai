#!/usr/bin/env python
"""Band edges against C2DB's own: how far the edges the tool shows sit from computed ones.

The tool places the band edges at work function +- half the gap, from two models
trained on different data (gap: Alexandria, MC2D and 2DMatPedia, without spin-orbit
coupling; work function: the C2DB mirror). Until now they were checked against four
literature TMDs. The full C2DB release stores both edges relative to the vacuum level
(vbm, cbm; PBE with spin-orbit coupling) for 4 567 materials, so the rows of
screening_table.csv that come from C2DB can be scored against them directly, with no
new inference.

Declared before the first run (19 October 2026), so the numbers cannot pick the
story. Reported, on C2DB rows matched to the release by olduid:
- MAE and mean signed error of the electron affinity (-cbm) and the ionisation
  potential (-vbm), where the tool shows edges, split by the work-function model's
  training role (trained on / held out) and by heavy elements (Z >= 52), because the
  gap model carries no spin-orbit coupling and C2DB does;
- the floor of the construction itself: C2DB's own work function +- its own gap
  against its own edges (zero if C2DB's Fermi level sits mid-gap);
- whether the withholding rule earns its keep: the same error where the edges are
  withheld (either model disowns the structure) against where they are shown, and by
  the gap verdict's tier among the shown.
Nothing ships on these numbers; they decide only how the README describes the edges.

    python scripts/edges_vs_c2db.py
    python scripts/edges_vs_c2db.py --json runs/edges_vs_c2db.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HEAVY_Z = 52
# atomic numbers up to Z = 52 are light; the set of heavier symbols is what matters
LIGHT = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
            "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb".split())


def c2db_edges(db_path: str) -> dict[str, dict]:
    """olduid -> vbm, cbm, efermi, gap from the release (read as SQLite; no ASE needed)."""
    out = {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    for (kvp,) in con.execute("select key_value_pairs from systems"):
        kv = json.loads(kvp)
        if kv.get("olduid") and kv.get("vbm") is not None and kv.get("cbm") is not None:
            out[kv["olduid"]] = {k: kv.get(k) for k in ("uid", "vbm", "cbm", "efermi", "gap",
                                                        "evacdiff", "is_magnetic")}
    return out


def heavy(elements: str) -> bool:
    return any(e not in LIGHT for e in elements.split())


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def stats(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0}
    n = len(errs)
    return {"n": n, "mae": sum(abs(e) for e in errs) / n, "bias": sum(errs) / n}


def fmt(s: dict) -> str:
    return "n=0" if not s["n"] else f"n={s['n']:4d}  MAE {s['mae']:.3f}  bias {s['bias']:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(ROOT, "screening_table.csv"))
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "c2db.db"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ref = c2db_edges(args.db)
    rows = [r for r in csv.DictReader(open(args.table)) if r["source"] == "c2db"]
    matched = [r for r in rows if r["id"] in ref]
    print(f"C2DB rows in the table: {len(rows)}; matched by olduid with vbm/cbm: {len(matched)}")

    # the construction's own floor: C2DB's work function (-efermi) +- its gap/2
    floor_ea, floor_ip = [], []
    for r in matched:
        c = ref[r["id"]]
        wf, g = -c["efermi"], c["gap"]
        floor_ea.append((wf - g / 2) - (-c["cbm"]))
        floor_ip.append((wf + g / 2) - (-c["vbm"]))
    print("\nFloor of wf +- gap/2 with C2DB's own numbers (0 if its Fermi level is mid-gap):")
    print(f"  affinity   {fmt(stats(floor_ea))}")
    print(f"  ionisation {fmt(stats(floor_ip))}")

    res = {"n_rows": len(rows), "n_matched": len(matched),
           "floor": {"ea": stats(floor_ea), "ip": stats(floor_ip)}}
    groups: dict[str, dict[str, list]] = {}

    def add(key, ea, ip, dwf, dgap):
        g = groups.setdefault(key, {"ea": [], "ip": [], "wf": [], "half_gap": []})
        g["ea"].append(ea), g["ip"].append(ip), g["wf"].append(dwf), g["half_gap"].append(dgap / 2)

    for r in matched:
        c = ref[r["id"]]
        wf, gap = fnum(r["work_function_eV"]), fnum(r["pred_gap_eV"])
        if wf is None or gap is None:
            continue
        ea_ref, ip_ref = -c["cbm"], -c["vbm"]
        shown = fnum(r["electron_affinity_eV"]) is not None
        ea, ip = wf - gap / 2, wf + gap / 2      # what the tool would show
        e_ea, e_ip = ea - ea_ref, ip - ip_ref
        dwf, dgap = wf - (-c["efermi"]), gap - c["gap"]
        role = "trained" if r["in_wf_training_set"] == "train" else "held out"
        add("withheld" if not shown else "shown", e_ea, e_ip, dwf, dgap)
        if shown:
            add(f"shown, wf model {role}", e_ea, e_ip, dwf, dgap)
            add("shown, heavy (Z>=52)" if heavy(r["elements"]) else "shown, light", e_ea, e_ip, dwf, dgap)
            tier = r["verdict"].split(" ")[0]
            add(f"shown, gap verdict {tier}", e_ea, e_ip, dwf, dgap)
            if role == "held out":
                add("shown, held out, heavy" if heavy(r["elements"]) else "shown, held out, light",
                    e_ea, e_ip, dwf, dgap)

    order = ["shown", "withheld", "shown, wf model trained", "shown, wf model held out",
             "shown, light", "shown, heavy (Z>=52)", "shown, held out, light", "shown, held out, heavy",
             "shown, gap verdict reliable", "shown, gap verdict check"]
    print("\nError of the edges the tool shows (prediction - C2DB), eV:")
    res["groups"] = {}
    for k in order + sorted(set(groups) - set(order)):
        if k not in groups:
            continue
        g = groups[k]
        s = {q: stats(v) for q, v in g.items()}
        res["groups"][k] = s
        print(f"  {k}")
        print(f"     affinity   {fmt(s['ea'])}")
        print(f"     ionisation {fmt(s['ip'])}")
        print(f"     parts: work function {fmt(s['wf'])}; half the gap {fmt(s['half_gap'])}")

    # the four reference TMDs, by name, for the README's existing claim
    tmd = {r["formula"]: r for r in matched if r["formula"] in ("MoS2", "MoSe2", "WS2", "WSe2")
           and fnum(r["electron_affinity_eV"]) is not None and r.get("family", "").startswith("1H")}
    if tmd:
        print("\n1H TMDs (shown edges):")
        for f, r in sorted(tmd.items()):
            c = ref[r["id"]]
            print(f"  {f:6s} affinity {float(r['electron_affinity_eV']):.2f} vs {-c['cbm']:.2f}   "
                  f"ionisation {float(r['ionisation_potential_eV']):.2f} vs {-c['vbm']:.2f}   "
                  f"(wf model: {r['in_wf_training_set']})")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
