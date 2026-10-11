#!/usr/bin/env python
"""Direct, indirect - or too close to call.

The gap-type classifier answers direct or indirect at one threshold (0.33). Its
indirect calls are right about a third of the time, and its retrained successors
called MoS2 indirect: Alexandria labels MoS2 direct by a margin of 0.07 eV against a
boundary of 0.1 (indirect when band_gap_dir - band_gap_ind >= 0.1). A binary answer
near that boundary is a coin the tool should not be flipping.

So three answers instead of two, with both edges chosen on the validation split:
  direct     p_indirect < lo, lo the largest value at which direct calls are right
             at least 90% of the time
  indirect   p_indirect > hi, hi the smallest value at which indirect calls are right
             at least 75% of the time (none, if no value gets there)
  borderline in between - "too close to call", said in so many words

Decision rule, written before the test split was scored (11 October 2026). The
three-way answer ships if, on the test split:
  1. direct calls are right at least 85% of the time (the base rate is 79%),
  2. indirect calls, if there are any, are right at least 65% of the time,
  3. at least half of the structures get a direct or indirect answer, and
  4. the borderline structures sit closer to the physical boundary than the rest:
     their median |dir - ind - 0.1| is smaller.
And on the reference monolayers: MoS2, MoSe2, WS2, WSe2 and phosphorene are not
called indirect (borderline is allowed - MoS2's margin is 0.07 eV).

    python scripts/gap_type_borderline.py            # report
    python scripts/gap_type_borderline.py --write    # store lo/hi in the type checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

TYPED = os.path.join(ROOT, "weights", "cgcnn_2d_typed.pt")
BOUNDARY = 0.1
REFS = ("MoS2", "MoSe2", "WS2", "WSe2", "phosphorene", "hBN")


def edges(p, y, prec_direct=0.90, prec_indirect=0.75):
    grid = np.unique(np.round(p, 3))
    lo = max((g for g in grid if (p < g).sum() >= 20 and (~y[p < g]).mean() >= prec_direct),
             default=None)
    hi = min((g for g in grid if (p > g).sum() >= 10 and y[p > g].mean() >= prec_indirect),
             default=None)
    return lo, hi


def score(p, y, margin, lo, hi):
    d = p < lo if lo is not None else np.zeros(len(p), bool)
    i = p > hi if hi is not None else np.zeros(len(p), bool)
    b = ~(d | i)
    return {"n": int(len(p)), "direct_calls": int(d.sum()),
            "direct_precision": float((~y[d]).mean()) if d.any() else None,
            "indirect_calls": int(i.sum()),
            "indirect_precision": float(y[i].mean()) if i.any() else None,
            "borderline": int(b.sum()), "answered": float((d | i).mean()),
            "margin_borderline": float(np.median(margin[b])) if b.any() else None,
            "margin_answered": float(np.median(margin[d | i])) if (d | i).any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    import pandas as pd
    import torch
    from jarvis.db.figshare import data
    from nanomat import Predictor
    from nanomat.predict import read_structure

    alex = {x["mat_id"]: x for x in data("alex_pbe_2d_all")}
    split = json.load(open(os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.split.json")))
    t = pd.read_csv(os.path.join(ROOT, "screening_table.csv")).set_index("id")
    thr = float(torch.load(TYPED, map_location="cpu", weights_only=False)["threshold"])

    def part(name):
        ids = [f.rsplit(".", 1)[0] for f in split[name]]
        ids = [i for i in ids if i in t.index and i in alex and not pd.isna(t.loc[i, "p_indirect"])]
        diff = np.array([alex[i]["band_gap_dir"] - alex[i]["band_gap_ind"] for i in ids])
        return t.loc[ids, "p_indirect"].to_numpy(), diff >= BOUNDARY, np.abs(diff - BOUNDARY)

    pv, yv, mv = part("val")
    pt, yt, mt = part("test")
    lo, hi = edges(pv, yv)
    print(f"threshold now {thr:.3f}; on validation: lo {lo}, hi {hi}")
    binary_val = score(pv, yv, mv, thr, thr)
    print(f"  binary, validation: direct {binary_val['direct_precision']:.3f}, "
          f"indirect {binary_val['indirect_precision']:.3f}")
    s = score(pt, yt, mt, lo, hi)
    b = score(pt, yt, mt, thr, thr)
    print(f"\ntest split, n = {s['n']} ({100 * yt.mean():.0f}% indirect)")
    print(f"  binary at {thr:.2f}: direct calls right {100 * b['direct_precision']:.1f}% "
          f"({b['direct_calls']}), indirect {100 * b['indirect_precision']:.1f}% ({b['indirect_calls']})")
    ip = f"{100 * s['indirect_precision']:.1f}%" if s["indirect_precision"] is not None else "-"
    print(f"  three-way:     direct calls right {100 * s['direct_precision']:.1f}% "
          f"({s['direct_calls']}), indirect {ip} ({s['indirect_calls']}), "
          f"borderline {s['borderline']}, answered {100 * s['answered']:.0f}%")
    print(f"  median |dir - ind - 0.1|: borderline {s['margin_borderline']:.3f} eV, "
          f"answered {s['margin_answered']:.3f} eV")

    P = Predictor(os.path.join(ROOT, "weights"), verbose=False)
    calls = {}
    for n in REFS:
        r = P.run(read_structure(os.path.join(ROOT, "examples", f"{n}.vasp")))
        p = r.p_indirect
        c = ("direct" if lo is not None and p < lo else
             "indirect" if hi is not None and p > hi else "borderline")
        calls[n] = (round(float(p), 3), c)
    print("\nreferences:", calls)

    ok = [s["direct_precision"] is not None and s["direct_precision"] >= 0.85,
          s["indirect_precision"] is None or s["indirect_precision"] >= 0.65,
          s["answered"] >= 0.5,
          s["margin_borderline"] is not None and s["margin_borderline"] < s["margin_answered"],
          all(calls[n][1] != "indirect" for n in REFS if n != "hBN")]
    print(f"\ndecision rule: {ok} -> {'SHIP' if all(ok) else 'do not ship'}")
    if all(ok) and args.write:
        ck = torch.load(TYPED, map_location="cpu", weights_only=False)
        ck["three_way"] = {"lo": float(lo), "hi": None if hi is None else float(hi),
                           "test": s, "boundary_eV": BOUNDARY,
                           "rule": "direct if p < lo, indirect if p > hi, else borderline"}
        torch.save(ck, TYPED)
        print(f"written three_way into {TYPED}")


if __name__ == "__main__":
    main()
