#!/usr/bin/env python
"""Fetch the GW + BSE reference table from C2DB, so the optical correction has data.

The optical gap is the absorption onset: the quasiparticle gap minus the exciton
binding energy. Measuring it takes an experiment, and the number of 2D monolayers
with a well-established measured value is small — which is why the correction in
this repository was fitted on five of them and its leave-one-out error was 0.71 eV,
four times its in-sample error.

C2DB computes both halves from first principles for a subset of its materials:
G0W0 for the quasiparticle gaps and the Bethe-Salpeter equation for the exciton.
That gives 283 optical gaps instead of 5. They are theory, not measurement, but
they reproduce the five measured monolayers to within 0.02-0.26 eV, which is what
earns them the job — see scripts/fit_gap_corrections.py, which prints that check.

    python scripts/fetch_c2db_optical.py            # refresh data/c2db_optical.csv
    python scripts/fetch_c2db_optical.py --check    # report on the cached copy only

The result is cached in the repository, so nothing downstream depends on the site
being reachable or on its internals staying the same. Re-run this only to refresh.

Data: C2DB, https://c2db.fysik.dtu.dk/, CC-BY-SA 4.0. Cite Haastrup et al.,
2D Materials 5, 042002 (2018) and Gjerding et al., 2D Materials 8, 044002 (2021).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "c2db_optical.csv")

BASE = "https://c2db.fysik.dtu.dk"
UA = "nanomat-ai/1.0 (research use; https://github.com/ac1esan/nanomat-ai)"
PAUSE = 0.5   # between requests; the whole pull is ~20 of them

# the column keys the site's table endpoint understands, and what we call them
COLUMNS = [
    ("olduid", "uid"),            # matches the id in the JARVIS c2db mirror
    ("gap_gw", "gap_gw"),         # fundamental quasiparticle gap, G0W0
    ("gap_dir_gw", "gap_dir_gw"), # direct quasiparticle gap - what absorption sees
    ("E_B", "E_B"),               # exciton binding energy, BSE
    ("gap_dir", "gap_dir_pbe"),
]
HEADER_TO_KEY = {
    "Formula": "formula",
    "Energy above hull [eV/atom]": "ehull",
    "Heat of formation [eV/atom]": "hform",
    "Band gap (PBE) [eV]": "gap_pbe",
    "Magnetic": "magnetic",
    "Layer group (not Space group)": "layergroup",
    "Old uid": "uid",
    "Band gap (G₀W₀) [eV]": "gap_gw",
    "Direct band gap (G₀W₀) [eV]": "gap_dir_gw",
    "Exciton binding energy (BSE) [eV]": "E_B",
    "Direct band gap (PBE) [eV]": "gap_dir_pbe",
}


def get(url: str, jar: dict) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if jar:
        req.add_header("Cookie", "; ".join(f"{k}={v}" for k, v in jar.items()))
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r.headers.get_all("Set-Cookie") or []:
            k, _, v = raw.split(";")[0].partition("=")
            jar[k.strip()] = v.strip()
        return r.read().decode("utf-8", "replace")


def cells(html: str) -> tuple[list[str], list[list[str]]]:
    """Pull the header and body of the one table in a fragment, without a parser.

    Body cells are <th scope="row">, not <td>, so both have to be matched. Tags are
    dropped to nothing rather than to a space, because a formula is marked up as
    Tl<sub>2</sub>Br<sub>2</sub> and a space there would split it.
    """
    strip = lambda s: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()
    cell = r"<t[dh][ >].*?</t[dh]>"
    head = re.search(r"<thead.*?</thead>", html, re.S)
    headers = [strip(c) for c in re.findall(cell, head.group(0), re.S)] if head else []
    body = re.search(r"<tbody.*?</tbody>", html, re.S)
    rows = []
    for tr in re.findall(r"<tr.*?</tr>", body.group(0) if body else "", re.S):
        rows.append([strip(c) for c in re.findall(cell, tr, re.S)])
    return headers, [r for r in rows if r]


def fetch() -> tuple[list[str], list[list[str]]]:
    jar: dict[str, str] = {}
    # filter=E_B selects the materials that have a BSE exciton at all; clearing the
    # stability window keeps every one of them rather than the default ehull <= 0.2
    page = get(f"{BASE}/?filter=E_B&from_ehull=&to_ehull=", jar)
    m = re.search(r'name="sid"\s+value="(\d+)"', page)
    if not m:
        raise SystemExit("could not find the session id; the site's markup has changed")
    sid = m.group(1)
    total = re.search(r"Found ([\d ]+) rows", page)
    print(f"session {sid}: {total.group(1).strip() if total else '?'} materials carry an exciton")

    for key, _ in COLUMNS:
        time.sleep(PAUSE)
        get(f"{BASE}/table?sid={sid}&toggle={key}", jar)

    headers, rows = [], []
    for p in range(0, 40):
        time.sleep(PAUSE)
        h, r = cells(get(f"{BASE}/table?sid={sid}&page={p}", jar))
        if not r:
            break
        headers = headers or h
        rows += r
        print(f"  page {p}: {len(rows)} rows", end="\r", flush=True)
    print(f"\ncollected {len(rows)} rows")
    keys = [HEADER_TO_KEY.get(h, h) for h in headers]
    missing = [k for _, k in COLUMNS if k not in keys]
    if missing:
        raise SystemExit(f"columns missing from the table: {missing}; keys were {keys}")
    return keys, rows


def report(path: str) -> None:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    num = lambda v: float(v) if v not in ("", None) else None
    both = [r for r in rows if num(r["gap_dir_gw"]) is not None and num(r["E_B"]) is not None]
    print(f"\n{path}: {len(rows)} materials, {len(both)} with both a direct G0W0 gap "
          "and a BSE exciton")
    print("optical gap = direct G0W0 gap - exciton binding energy:")
    for uid_prefix in ("MoS2-", "MoSe2-", "WS2-", "WSe2-", "BN-"):
        for r in both:
            if r["uid"].startswith(uid_prefix):
                o = num(r["gap_dir_gw"]) - num(r["E_B"])
                print(f"  {r['formula']:8s} {num(r['gap_dir_gw']):5.2f} - "
                      f"{num(r['E_B']):4.2f} = {o:5.2f} eV")
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report on the cached file")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if not args.check:
        keys, rows = fetch()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(keys)
        w.writerows(rows)
        with open(args.out, "w") as f:
            f.write(buf.getvalue())
        print(f"written {args.out}")
    if not os.path.exists(args.out):
        raise SystemExit(f"{args.out} does not exist; run without --check to fetch it")
    report(args.out)


if __name__ == "__main__":
    main()
