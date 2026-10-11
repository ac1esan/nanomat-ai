#!/usr/bin/env python
"""MC2D against Alexandria, structure by structure, and the shipped model on MC2D.

Two questions before any MC2D label goes near training:

1. Do the labels agree with the model's target where the structure is the same?
   MC2D is computed with Quantum ESPRESSO on structures relaxed with a van der Waals
   functional; Alexandria with VASP and PBE. Every layer is put in one frame and
   matched with the audit's matcher plus its layer-thickness check
   (scripts/audit_2dmatpedia.py). On matched pairs: bias, MAE, metal agreement, and
   the same split by heavy elements (Z >= 52) that exposed spin-orbit coupling in
   C2DB.

2. How does the shipped model do on these layers? MC2D is exfoliated from bulk
   crystals in experimental databases, so it is closer to the population people
   screen than Alexandria's mostly hypothetical one. Scored on entries whose
   Alexandria twin, if any, the model never trained on: MAE against the MC2D label,
   by verdict tier, by element, and the coverage of the 90% interval.

    python scripts/fetch_mc2d.py          # once: data/mc2d.json.gz
    python scripts/audit_mc2d.py          # writes runs/audit_mc2d.json and _entries.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from audit_2dmatpedia import (MATCHER, THICK_TOL, same_structure, standardize,  # noqa: E402
                              to_pymatgen)
from fetch_mc2d import load as load_mc2d, to_structure  # noqa: E402

METAL_GAP = 0.01
HEAVY_Z = 52
LIGHT = ("C", "N", "B", "H", "O")
_SM = None


def _prep(item):
    key, st = item
    try:
        return key, standardize(st)
    except Exception:
        return key, None


def _init():
    global _SM
    from pymatgen.analysis.structure_matcher import StructureMatcher
    _SM = StructureMatcher(**MATCHER)


def _match(task):
    key, s, cands = task
    hits = []
    for other, t in cands:
        try:
            if same_structure(_SM, s, t):
                rms = _SM.get_rms_dist(s[0], t[0])
                hits.append((other, float(rms[0]) if rms else 0.0))
        except Exception:
            pass
    return key, hits


def stats(a, b) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    both = (a > METAL_GAP) & (b > METAL_GAP)
    d = a[both] - b[both]
    return {"n": int(len(a)), "metal_agree": float(((a > METAL_GAP) == (b > METAL_GAP)).mean()),
            "n_gapped": int(both.sum()), "bias": float(d.mean()) if both.any() else None,
            "mae": float(np.abs(d).mean()) if both.any() else None,
            "median": float(np.median(np.abs(d))) if both.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "audit_mc2d"))
    args = ap.parse_args()
    from jarvis.db.figshare import data
    from pymatgen.core import Composition, Element

    t0 = time.time()
    mc = [r for r in load_mc2d() if r["gap"] is not None]
    alex = data("alex_pbe_2d_all")
    for x in alex:
        x["_f"] = Composition(x["formula"]).reduced_formula
    by_f = defaultdict(list)
    for j, x in enumerate(alex):
        by_f[x["_f"]].append(j)
    split = json.load(open(os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.split.json")))
    role = {f.rsplit(".", 1)[0]: k for k in ("train", "val", "test") for f in split[k]}
    sts = {r["id"]: to_structure(r) for r in mc}
    for r in mc:
        r["_f"] = sts[r["id"]].composition.reduced_formula
    print(f"MC2D {len(mc)} entries with a gap; Alexandria {len(alex)}  [{time.time() - t0:.0f}s]")

    # --- one frame, then match within formula and primitive size ------------------
    need = sorted({j for r in mc for j in by_f.get(r["_f"], [])})
    items = [(("m", r["id"]), sts[r["id"]]) for r in mc]
    items += [(("a", j), to_pymatgen(alex[j]["atoms"])) for j in need]
    with Pool(args.workers) as pool:
        std = dict(pool.imap_unordered(_prep, items, chunksize=32))
    tasks = []
    for r in mc:
        s = std[("m", r["id"])]
        if s is None:
            continue
        cands = [(j, std[("a", j)]) for j in by_f.get(r["_f"], [])
                 if std[("a", j)] is not None and len(std[("a", j)][0]) == len(s[0])]
        if cands:
            tasks.append((r["id"], s, cands))
    with Pool(args.workers, initializer=_init) as pool:
        hits = dict(pool.imap_unordered(_match, tasks, chunksize=4))
    print(f"matched {len(tasks)} entries with candidates  [{time.time() - t0:.0f}s]")

    rows = []
    for r in mc:
        els = {e.symbol for e in sts[r["id"]].composition.elements}
        heavy = max(Element(e).Z for e in els) >= HEAVY_Z
        row = {"id": r["id"], "formula": r["_f"], "gap": r["gap"], "heavy": heavy,
               "elements": " ".join(sorted(els)),
               "status": ("no_vacuum_gap" if std[("m", r["id"])] is None else
                          "formula_absent" if not by_f.get(r["_f"]) else
                          "matched" if hits.get(r["id"]) else "new_polymorph")}
        if row["status"] == "matched":
            j, rms = min(hits[r["id"]], key=lambda h: h[1])
            row.update(alex_id=alex[j]["mat_id"], alex_gap=float(alex[j]["band_gap_ind"]),
                       alex_ehull=alex[j]["e_above_hull"], rms=rms,
                       alex_role=role.get(alex[j]["mat_id"], "not_in_training"))
        rows.append(row)
    status = Counter(r["status"] for r in rows)
    print(f"\nMC2D entries by status: {dict(status)}")

    M = [r for r in rows if r["status"] == "matched"]
    rep = {"status": dict(status), "thick_tol": THICK_TOL}
    print("\nlabels on the same structure (MC2D - Alexandria):")
    for lab, sel in (("all", M), ("heavy (Z>=52)", [r for r in M if r["heavy"]]),
                     ("light", [r for r in M if not r["heavy"]])):
        s = stats([r["gap"] for r in sel], [r["alex_gap"] for r in sel])
        rep[f"labels_{lab}"] = s
        print(f"  {lab:14s} n={s['n']:4d}  metal agree {100 * s['metal_agree']:.1f}%  "
              f"gapped {s['n_gapped']}: bias {s['bias']:+.3f}  MAE {s['mae']:.3f}  "
              f"median {s['median']:.3f}" if s["n_gapped"] else f"  {lab}: none")
    big = sorted((r for r in M if r["gap"] > METAL_GAP and r["alex_gap"] > METAL_GAP
                  and abs(r["gap"] - r["alex_gap"]) > 0.3),
                 key=lambda r: -abs(r["gap"] - r["alex_gap"]))
    print(f"  disagreements over 0.3 eV: {len(big)}")
    for r in big[:15]:
        print(f"    {r['formula']:12s} MC2D {r['gap']:.2f}  Alexandria {r['alex_gap']:.2f}  "
              f"rms {r['rms']:.3f}")

    # --- the shipped model on MC2D -----------------------------------------------
    from nanomat import Predictor
    from nanomat.predict import tier_key
    P = Predictor(os.path.join(ROOT, "weights"), verbose=False)
    preds = dict(P.run_many([(r["id"], sts[r["id"]]) for r in rows], batch_size=256,
                            device=args.device))
    for r in rows:
        p = preds.get(r["id"])
        if p is not None:
            r.update(pred=p.gap, unc=p.unc, iv=p.interval90, tier=tier_key(p.verdict),
                     verdict=p.verdict, p_metal=p.p_metal)
    # memorised: the entry itself is a training structure (an ensemble trained with
    # MC2D, scripts/mc2d_experiment.py, holds them as mc2d-<id>.vasp), or its
    # Alexandria twin is
    ev = [r for r in rows if "pred" in r and r["gap"] > METAL_GAP
          and r.get("alex_role") != "train" and role.get(f"mc2d-{r['id']}") != "train"]
    err = lambda rs: np.array([abs(r["pred"] - r["gap"]) for r in rs])
    n_own = sum(role.get(f"mc2d-{r['id']}") == "train" for r in rows)
    print(f"\nshipped model on {len(ev)} MC2D semiconductors it never trained on "
          f"(entries in training, {n_own}, and twins of training structures excluded):")
    out = {"n": len(ev), "mae": float(err(ev).mean())}
    print(f"  all            MAE {out['mae']:.3f}  median {np.median(err(ev)):.3f}")
    out["by_tier"] = {}
    for k in ("reliable", "check", "out_of_domain"):
        sel = [r for r in ev if r["tier"] == k]
        if sel:
            out["by_tier"][k] = {"n": len(sel), "mae": float(err(sel).mean())}
            print(f"  {k:14s} n={len(sel):4d}  MAE {err(sel).mean():.3f}")
    shown = [r for r in ev if r["tier"] != "out_of_domain" and r["iv"] is not None]
    cov = float(np.mean([abs(r["pred"] - r["gap"]) <= r["iv"] for r in shown])) if shown else None
    out["coverage90"] = cov
    print(f"  90% interval covers {100 * cov:.1f}% of the {len(shown)} rows where it is shown")
    for lab, f in (("heavy (Z>=52)", lambda r: r["heavy"]), ("light", lambda r: not r["heavy"]),
                   *[(f"contains {el}", (lambda e: lambda r: e in r["elements"].split())(el))
                     for el in LIGHT]):
        sel = [r for r in ev if f(r)]
        if len(sel) >= 10:
            e = np.array([r["pred"] - r["gap"] for r in sel])
            out[lab] = {"n": len(sel), "mae": float(np.abs(e).mean()), "bias": float(e.mean())}
            print(f"  {lab:14s} n={len(sel):4d}  MAE {np.abs(e).mean():.3f}  bias {e.mean():+.3f}")
    metals = [r for r in rows if "pred" in r and r["gap"] <= METAL_GAP]
    if metals:
        caught = np.mean([r["tier"] == "out_of_domain" and r["p_metal"] is not None
                          and r["p_metal"] >= P.metal_thr for r in metals])
        out["metal_gate_catches"] = float(caught)
        print(f"  metal gate catches {100 * caught:.0f}% of the {len(metals)} MC2D metals")
    rep["model"] = out

    # --- what MC2D could add to training ------------------------------------------
    new = [r for r in rows if r["status"] != "matched" and r["gap"] > METAL_GAP]
    light = Counter(el for r in new for el in r["elements"].split() if el in LIGHT)
    rep["new_semiconductors"] = {"n": len(new), **{el: light[el] for el in LIGHT}}
    print(f"\nsemiconductors with no Alexandria twin: {len(new)}  light elements {dict(light)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out + ".json", "w"), indent=1)
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(args.out + "_entries.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten {args.out}.json and _entries.csv  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
