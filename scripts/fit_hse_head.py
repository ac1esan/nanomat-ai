#!/usr/bin/env python
"""An HSE06 gap from the latent space, fitted on the full C2DB release.

The quasiparticle-level fallback the tool carries is a straight line from the model's
gap to HSE06, fitted on 36 JARVIS dft_2d entries (leave-one-out MAE ~0.2 eV). The
full C2DB release holds an HSE06 gap for 2 897 of its non-magnetic semiconductors.
That is enough to ask the question the exciton heads answered for G0W0: does the
structure, through the gap ensemble's embedding, say more about the HSE correction
than the PBE gap alone does?

C2DB's HSE06 gaps include spin-orbit coupling; so will this head's output.

Candidates, on identical composition-disjoint folds (8 folds x 6 shuffles):
  L   a straight line in the model's gap, fitted in the fold
  Q   a quadratic in the model's gap
  H   ridge on [latent, gap] -> HSE gap
  Hr  ridge on [latent, gap] -> the residual of L, fitted in the fold
Reference, not a candidate: a straight line in C2DB's own PBE gap, the floor any
function of the gap alone could reach. Also reported: the shipped fallback line,
applied as it is.

Decision rule, written before any candidate was scored (16 October 2026). The better
of H and Hr by cross-validated MAE ships, as an HSE06+SOC estimate, only if
  1. its MAE is at least 20% below the better of L and Q;
  2. in every band of the model's gap (0-1, 1-2, 2-3, 3-5, >= 5 eV) holding at least
     50 materials, it is no worse than the better of L and Q in that band by more
     than 0.02 eV;
  3. its out-of-fold HSE gap for each of MoS2, MoSe2, WS2 and WSe2 (1H) is within
     0.2 eV of C2DB's.
Fitted on non-magnetic semiconductors the gap model does not call out-of-domain.

External check, reported, not voting: JARVIS dft_2d entries with an HSE gap (VASP,
no spin-orbit coupling), the set the shipped line was fitted on. The head is refitted
without any C2DB material sharing their compositions before it is applied, and is
compared with the line's leave-one-out error.

    python scripts/fit_hse_head.py --device cuda
    python scripts/fit_hse_head.py --device cuda --write

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
from nanomat.predict import correct  # noqa: E402
from c2db_full import DB, TMD, embed, predict_in_domain, records  # noqa: E402
from fit_exciton import collapse, make_folds, ridge, FOLDS, SHUFFLES  # noqa: E402

ENSEMBLE = os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.pt")
BANDS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]


def cross_validate(lat, gap, y, comp, pbe_c2db):
    X = np.column_stack([lat, gap])
    names = ("L", "Q", "H", "Hr", "ref")
    preds = {k: np.zeros((SHUFFLES, len(y))) for k in names}
    for s in range(SHUFFLES):
        for tr, te in make_folds(comp, s):
            lin = np.polyfit(gap[tr], y[tr], 1)
            preds["L"][s, te] = np.polyval(lin, gap[te])
            preds["Q"][s, te] = np.polyval(np.polyfit(gap[tr], y[tr], 2), gap[te])
            sc, m = ridge(X, y, tr)
            preds["H"][s, te] = m.predict(sc.transform(X[te]))
            res = y - np.polyval(lin, gap)
            sc, m = ridge(X, res, tr)
            preds["Hr"][s, te] = np.polyval(lin, gap[te]) + m.predict(sc.transform(X[te]))
            preds["ref"][s, te] = np.polyval(np.polyfit(pbe_c2db[tr], y[tr], 1), pbe_c2db[te])
    mean = {k: p.mean(0) for k, p in preds.items()}
    scores = {k: float(np.abs(p - y).mean(1).mean()) for k, p in preds.items()}
    return scores, mean


def external(P, lat_all, gap_all, y_all, comp_all, best, device):
    """JARVIS dft_2d HSE gaps (VASP, no SOC); C2DB compositions in it removed first."""
    try:
        from jarvis.core.atoms import Atoms
        from jarvis.db.figshare import data as jarvis_data
    except ImportError:
        print("\nexternal check skipped: jarvis-tools not installed")
        return None
    import pandas as pd
    num = lambda v: pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    rows = [(e["jid"], float(num(e.get("hse_gap"))), Atoms.from_dict(e["atoms"]).pymatgen_converter())
            for e in jarvis_data("dft_2d") if not pd.isna(num(e.get("hse_gap")))
            and num(e.get("hse_gap")) > 0.01]
    res = dict(P.run_many([(j, st) for j, _, st in rows], batch_size=128, device=device))
    rows = [r for r in rows if res.get(r[0]) is not None
            and not res[r[0]].verdict.startswith("out-of-domain")]
    jcomp = {st.composition.reduced_formula for _, _, st in rows}
    tr = np.array([c not in jcomp for c in comp_all])
    X = np.column_stack([lat_all, gap_all])
    lin = np.polyfit(gap_all[tr], y_all[tr], 1)
    g = np.array([res[j].gap for j, _, _ in rows])
    Xj = np.column_stack([embed(P, [st for _, _, st in rows]), g])
    if best == "H":
        sc, m = ridge(X, y_all, np.where(tr)[0])
        pj = m.predict(sc.transform(Xj))
    else:
        sc, m = ridge(X, y_all - np.polyval(lin, gap_all), np.where(tr)[0])
        pj = np.polyval(lin, g) + m.predict(sc.transform(Xj))
    yj = np.array([h for _, h, _ in rows])
    ck = torch.load(ENSEMBLE, map_location="cpu")
    qp = ck.get("gap_corrections", {}).get("quasiparticle", {})
    shipped = np.array([correct(x, "quasiparticle", ck.get("gap_corrections")) for x in g])
    out = {"n": len(rows), "n_c2db_fitted": int(tr.sum()),
           "head": float(np.abs(pj - yj).mean()),
           "c2db_line": float(np.abs(np.polyval(lin, g) - yj).mean()),
           "shipped_line_in_sample": float(np.abs(shipped - yj).mean()),
           "shipped_line_loo": qp.get("mae_loo")}
    print(f"\nexternal (not voting): {len(rows)} JARVIS dft_2d HSE gaps (VASP, no SOC), "
          f"head refitted on {tr.sum()} C2DB materials of other compositions")
    print(f"  head {out['head']:.3f}   C2DB line {out['c2db_line']:.3f}   shipped line "
          f"{out['shipped_line_in_sample']:.3f} in sample (LOO {out['shipped_line_loo']})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--json", default=os.path.join(ROOT, "runs", "c2db_full", "hse.json"))
    args = ap.parse_args()
    P = Predictor(verbose=False)
    recs = [r for r in records(DB) if r["kv"].get("gap_hse") is not None]
    keep, sts, res, lat = predict_in_domain(P, recs, args.device)
    print(f"{len(recs)} non-magnetic C2DB semiconductors with an HSE gap; "
          f"{len(recs) - len(keep)} out-of-domain -> {len(keep)} kept")
    uid = np.array([r["uid"] for r in keep])
    gap = np.array([res[u].gap for u in uid])
    y = np.array([r["kv"]["gap_hse"] for r in keep])
    pbe = np.array([r["kv"]["gap"] for r in keep])
    comp = np.array([sts[u].composition.reduced_formula for u in uid])
    print(f"  {len(set(comp))} compositions; HSE - model gap: median {np.median(y - gap):+.3f} eV")

    scores, cvp = cross_validate(lat, gap, y, comp, pbe)
    print(f"\ncomposition-disjoint, {FOLDS} folds x {SHUFFLES} shuffles, MAE against C2DB HSE06 (eV)")
    for k, v in scores.items():
        print(f"  {k:4s} {v:.3f}")
    head = min(("H", "Hr"), key=lambda k: scores[k])
    base = min(("L", "Q"), key=lambda k: scores[k])
    ok1 = scores[head] <= 0.8 * scores[base]

    ok2, bands = True, {}
    print(f"\nby band of the model's gap: {head} vs {base}")
    for lo, hi in BANDS:
        sel = (gap >= lo) & (gap < hi)
        if not sel.any():
            continue
        eh, eb = np.abs(cvp[head] - y)[sel].mean(), np.abs(cvp[base] - y)[sel].mean()
        voting = sel.sum() >= 50
        if voting:
            ok2 &= eh <= eb + 0.02
        bands[f"{lo}-{hi}"] = {"n": int(sel.sum()), head: float(eh), base: float(eb)}
        print(f"  {lo}-{hi} eV  n={sel.sum():4d}  {eh:.3f} vs {eb:.3f}{'' if voting else '  (n<50, not voting)'}")

    ok3, tmd = True, {}
    print("\nTMDs, out-of-fold:")
    for u, name in TMD.items():
        i = np.where(uid == u)[0]
        if not len(i):
            print(f"  {name}: not in the fitted set - condition 3 fails")
            ok3 = False
            continue
        i = i[0]
        ok3 &= abs(cvp[head][i] - y[i]) <= 0.2
        tmd[name] = {"c2db_hse": float(y[i]), head: float(cvp[head][i]), base: float(cvp[base][i]),
                     "model_pbe": float(gap[i])}
        print(f"  {name:6s} C2DB {y[i]:.3f}  {head} {cvp[head][i]:.3f}  {base} {cvp[base][i]:.3f}  "
              f"(model PBE {gap[i]:.3f})")
    print(f"\ndecision rule, head {head} against {base}:"
          f"\n  1. {scores[head]:.3f} <= 0.8 x {scores[base]:.3f} = {0.8 * scores[base]:.3f}: {ok1}"
          f"\n  2. no band with n >= 50 worse by more than 0.02 eV: {ok2}"
          f"\n  3. TMDs within 0.2 eV of C2DB: {ok3}")
    passed = bool(ok1 and ok2 and ok3)
    ext = external(P, lat, gap, y, comp, head, args.device)

    report = {"n": int(len(y)), "n_compositions": int(len(set(comp))), "mae_cv": scores,
              "head": head, "baseline": base, "bands": bands, "tmd": tmd,
              "rule": [bool(ok1), bool(ok2), bool(ok3)], "passed": passed, "external": ext,
              "written": "2026-10-16, rule in this file's docstring"}
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(report, open(args.json, "w"), indent=1)
    print(f"\n{'PASSED' if passed else 'NOT PASSED'}; written {args.json}")
    if not (passed and args.write):
        print("checkpoint untouched")
        return
    X = np.column_stack([lat, gap])
    lin = np.polyfit(gap, y, 1)
    if head == "H":
        sc, m = ridge(X, y, np.arange(len(y)))
        entry = collapse(sc, m)
    else:
        sc, m = ridge(X, y - np.polyval(lin, gap), np.arange(len(y)))
        entry = collapse(sc, m)
        # the line is folded in: w.x + b + a*gap + c, gap being the last feature
        entry["w"] = entry["w"].clone()
        entry["w"][-1] += float(lin[0])
        entry["b"] += float(lin[1])
    ck = torch.load(ENSEMBLE, map_location="cpu")
    ck["hse_head"] = {**entry, "n": int(len(y)), "mae_cv": scores[head],
                      "target": "C2DB gap_hse (HSE06 with spin-orbit coupling)",
                      "source": "C2DB (c2db.db, 2025-07-08), CC BY-NC 4.0"}
    torch.save(ck, ENSEMBLE)
    print(f"written hse_head into {ENSEMBLE}")


if __name__ == "__main__":
    main()
