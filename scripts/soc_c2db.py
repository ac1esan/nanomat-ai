#!/usr/bin/env python
"""The spin-orbit shift of the gap, learned from C2DB's own with/without pairs.

The model predicts a PBE gap without spin-orbit coupling (SOC); SOC lowers gaps, by
tenths of an eV in compounds of heavy elements. The first attempt (scripts/fit_soc.py)
had to infer the shift from 576 structures held by both Alexandria (no SOC) and the
JARVIS mirror of C2DB (with SOC), and failed its rule: it invented a shift on light
compounds, where the two codes' own difference was all there was to learn. The full
C2DB release (scripts/c2db_full.py) stores, for each of its 3 878 non-magnetic
semiconductors, the direct gap computed both ways. The shift is then a property of
one calculation, with no code difference mixed in:

    delta = gap_dir - gap_dir_nosoc        (C2DB, PBE, direct gap)

On the reference monolayers it is the literature value: MoS2 -0.08, MoSe2 -0.11,
WS2 -0.28, WSe2 -0.32 eV. It is positive for about 9% of entries: SOC can also open
a gap where bands cross. The fundamental gap's SOC-free value is not stored; the
direct-gap shift stands in for it, and the use check below measures how well.

Candidates, on identical composition-disjoint folds (8 folds x 6 shuffles):
  A  one constant shift for compounds of an element with Z >= 52, another for the rest
  B  ridge on composition-only SOC features (Z^2, Z^4 moments, heavy fraction) + gap
  C  ridge on the gap ensemble's latent space + the predicted gap
  D  C plus B's features

Decision rule, written before any candidate was scored (16 October 2026). The
candidate among B, C, D with the lowest cross-validated MAE ships, as a "PBE + SOC"
estimate next to the PBE gap, only if
  1. its MAE beats A's by at least 0.02 eV overall;
  2. on compounds without a heavy element it is no worse than A's by more than
     0.01 eV: it must not invent a shift where the physics has none;
  3. on compounds with one it beats A's;
  4. its out-of-fold shift for each of MoS2, MoSe2, WS2 and WSe2 (1H) is within
     0.08 eV of C2DB's.
Fitted on non-magnetic semiconductors the gap model does not call out-of-domain.

Use check, reported, not voting: what a user would read is the model's gap plus the
predicted shift, as an estimate of C2DB's FUNDAMENTAL PBE+SOC gap ("gap"). With the
out-of-fold shifts, is that closer than the model's gap alone, and than gap + A?
This is where the direct-gap proxy could fail.

    python scripts/soc_c2db.py --device cuda            # fit, cross-validate, report
    python scripts/soc_c2db.py --device cuda --write    # also store the head if it passed

Data: C2DB, CC BY-NC 4.0; Haastrup et al., 2D Mater. 5, 042002 (2018); Gjerding et
al., 2D Mater. 8, 044002 (2021).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from nanomat import Predictor  # noqa: E402
from c2db_full import DB, TMD, predict_in_domain, records  # noqa: E402
from fit_exciton import collapse, make_folds, ridge, FOLDS, SHUFFLES  # noqa: E402
from fit_soc import soc_features, HEAVY_Z  # noqa: E402

ENSEMBLE = os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.pt")


def load(P: Predictor, device: str):
    recs = list(records(DB))
    keep, sts, res, lat = predict_in_domain(P, recs, device)
    print(f"{len(recs)} non-magnetic C2DB semiconductors; {len(recs) - len(keep)} "
          f"out-of-domain for the gap model -> {len(keep)} kept")
    uid = np.array([r["uid"] for r in keep])
    gap = np.array([res[u].gap for u in uid])
    comp_f = np.array([soc_features(sts[u], g) for u, g in zip(uid, gap)])
    hv = comp_f[:, 0] >= HEAVY_Z
    comp = np.array([sts[u].composition.reduced_formula for u in uid])
    delta = np.array([r["kv"]["gap_dir"] - r["kv"]["gap_dir_nosoc"] for r in keep])
    fund = np.array([r["kv"]["gap"] for r in keep])
    return uid, gap, lat, comp_f, hv, comp, delta, fund


def cross_validate(feats, hv, comp, delta):
    preds = {k: np.zeros((SHUFFLES, len(delta))) for k in ("A", *feats)}
    for s in range(SHUFFLES):
        for tr, te in make_folds(comp, s):
            for h in (True, False):
                preds["A"][s, te[hv[te] == h]] = delta[tr][hv[tr] == h].mean()
            for k, X in feats.items():
                sc, m = ridge(X, delta, tr)
                preds[k][s, te] = m.predict(sc.transform(X[te]))
    out = {}
    print(f"\ncomposition-disjoint, {FOLDS} folds x {SHUFFLES} shuffles, MAE of the SOC shift (eV)")
    print(f"  {'':4s} {'all':>7s} {'heavy':>7s} {'light':>7s}")
    for k, p in preds.items():
        e = np.abs(p - delta).mean(0)
        out[k] = {"all": float(e.mean()), "heavy": float(e[hv].mean()), "light": float(e[~hv].mean())}
        print(f"  {k:4s} {out[k]['all']:7.3f} {out[k]['heavy']:7.3f} {out[k]['light']:7.3f}")
    return out, {k: p.mean(0) for k, p in preds.items()}


def use_check(gap, fund, hv, cvp, best):
    """Model gap (+ out-of-fold shift) against C2DB's fundamental PBE+SOC gap."""
    out = {}
    print("\nuse check (not voting): MAE against C2DB's fundamental PBE+SOC gap, out-of-fold")
    for lab, sel in (("all", np.ones(len(gap), bool)), ("heavy", hv), ("light", ~hv)):
        e0 = np.abs(gap - fund)[sel].mean()
        eA = np.abs(gap + cvp["A"] - fund)[sel].mean()
        eb = np.abs(gap + cvp[best] - fund)[sel].mean()
        out[lab] = {"n": int(sel.sum()), "model": float(e0), "plus_A": float(eA), "plus_best": float(eb)}
        print(f"  {lab:5s} n={sel.sum():4d}  model {e0:.3f}   + A {eA:.3f}   + {best} {eb:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--json", default=os.path.join(ROOT, "runs", "c2db_full", "soc.json"))
    args = ap.parse_args()
    P = Predictor(verbose=False)
    uid, gap, lat, comp_f, hv, comp, delta, fund = load(P, args.device)
    print(f"  heavy-element compounds {hv.sum()}, others {(~hv).sum()}; mean shift "
          f"{delta[hv].mean():+.3f} / {delta[~hv].mean():+.3f} eV; {len(set(comp))} compositions")
    feats = {"B": comp_f, "C": np.column_stack([lat, gap]),
             "D": np.column_stack([lat, gap, comp_f[:, :4]])}
    scores, cvp = cross_validate(feats, hv, comp, delta)

    best = min(("B", "C", "D"), key=lambda k: scores[k]["all"])
    ok1 = scores["A"]["all"] - scores[best]["all"] >= 0.02
    ok2 = scores[best]["light"] <= scores["A"]["light"] + 0.01
    ok3 = scores[best]["heavy"] < scores["A"]["heavy"]
    print("\nTMDs, out-of-fold (mean over shuffles):")
    ok4, tmd = True, {}
    for u, name in TMD.items():
        i = np.where(uid == u)[0]
        if not len(i):
            print(f"  {name}: not in the fitted set - condition 4 fails")
            ok4 = False
            continue
        i = i[0]
        ok4 &= abs(cvp[best][i] - delta[i]) <= 0.08
        tmd[name] = {"c2db": float(delta[i]), "best": float(cvp[best][i]), "A": float(cvp["A"][i])}
        print(f"  {name:6s} C2DB {delta[i]:+.3f}  {best} {cvp[best][i]:+.3f}  A {cvp['A'][i]:+.3f}")
    print(f"\ndecision rule, best candidate {best}:"
          f"\n  1. beats A by {scores['A']['all'] - scores[best]['all']:+.3f} eV (>= 0.02): {ok1}"
          f"\n  2. light {scores[best]['light']:.3f} vs A {scores['A']['light']:.3f} (+0.01 allowed): {ok2}"
          f"\n  3. heavy {scores[best]['heavy']:.3f} vs A {scores['A']['heavy']:.3f}: {ok3}"
          f"\n  4. TMDs within 0.08 eV of C2DB: {ok4}")
    passed = bool(ok1 and ok2 and ok3 and ok4)
    use = use_check(gap, fund, hv, cvp, best)

    report = {"n": int(len(delta)), "n_heavy": int(hv.sum()), "n_compositions": int(len(set(comp))),
              "mae_cv": scores, "best": best, "rule": [bool(ok1), bool(ok2), bool(ok3), bool(ok4)],
              "passed": passed, "tmd": tmd, "use_check": use,
              "written": "2026-10-16, rule in this file's docstring"}
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(report, open(args.json, "w"), indent=1)
    print(f"\n{'PASSED' if passed else 'NOT PASSED'}; written {args.json}")
    if not (passed and args.write):
        print("checkpoint untouched")
        return
    sc, m = ridge(feats[best], delta, np.arange(len(delta)))
    ck = torch.load(ENSEMBLE, map_location="cpu")
    ck["soc_head"] = {**collapse(sc, m), "features": best, "heavy_z": HEAVY_Z,
                      "n": int(len(delta)), "mae_cv": scores[best],
                      "target": "C2DB gap_dir - gap_dir_nosoc (PBE, direct gap)",
                      "source": "C2DB (c2db.db, 2025-07-08), CC BY-NC 4.0"}
    torch.save(ck, ENSEMBLE)
    print(f"written soc_head into {ENSEMBLE}")


if __name__ == "__main__":
    main()
