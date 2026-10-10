#!/usr/bin/env python
"""A spin-orbit estimate of the gap, learned from structures two databases share.

The model predicts Alexandria's PBE gap, which is computed without spin-orbit
coupling. SOC lowers a gap - in heavy-element compounds by tenths of an eV - and
C2DB, which includes it, sits 0.24 eV below the model on compounds with an element of
Z >= 52 and 0.02 eV on the rest. C2DB's table has no SOC-free gap to learn the
difference from, but about 700 structures sit in both databases
(data/structure_twins.csv): Alexandria without SOC, C2DB with it. On the light ones
the two agree to 0.02 eV, so the code difference (VASP against GPAW) is small and
what is left on the heavy ones is mostly the coupling.

Target: delta = C2DB gap - Alexandria gap, per structure. Candidates, scored on the
same composition-disjoint folds:
  A  a constant shift for heavy-element compounds and another for the rest (baseline)
  B  ridge on composition-only SOC features (Z^2, Z^4 moments, heavy fraction) + gap
  C  ridge on the ensemble's latent space + the predicted gap (as the exciton heads)
  D  C plus B's features

Decision rule, written before any candidate was scored (10 October 2026). The
candidate among B, C, D with the lowest cross-validated MAE ships, as a second gap
"with spin-orbit coupling" next to the PBE one, only if
  1. its MAE beats A's by at least 0.02 eV overall, and
  2. on compounds without a heavy element it is no worse than A's by more than 0.01
     eV - it must not invent a shift where the physics has none.
C2DB structures with no Alexandria twin never enter the fit; they are reported as
an external check, as are the four TMDs against literature SOC shifts. If nothing
passes, nothing ships and the result is written up.

    python scripts/fit_soc.py            # fit, cross-validate, report
    python scripts/fit_soc.py --write    # also store the head in the checkpoint
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from nanomat import Predictor  # noqa: E402
from nanomat.llm_tools import KNOWN_BAD_REFERENCE  # noqa: E402
from nanomat.predict import read_structure  # noqa: E402
from fit_exciton import collapse, embed, make_folds, ridge, FOLDS, SHUFFLES  # noqa: E402

ENSEMBLE = os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.pt")
TWINS = os.path.join(ROOT, "data", "structure_twins.csv")
ALEX = os.path.join(ROOT, "alignn_data_alex_2d_eh02")
C2DB = os.path.join(ROOT, "alignn_data_c2db")
TABLE = os.path.join(ROOT, "screening_table.csv")
HEAVY_Z = 52
ALEX_SPREAD_MAX = 0.1   # eV: one structure's Alexandria labels in different cells must agree
# published SOC shifts of the monolayer gap at the PBE level (gap with SOC - without),
# for a sanity check only; the conduction-band splitting partly offsets the valence one
LITERATURE = {"MoS2": -0.08, "MoSe2": -0.10, "WS2": -0.27, "WSe2": -0.30}


def id_prop(folder):
    return {os.path.splitext(l.split(",")[0])[0]: float(l.split(",")[1])
            for l in open(os.path.join(folder, "id_prop.csv"))}


def soc_features(st, gap: float) -> list[float]:
    """Composition-only proxies for how strongly SOC can act: it grows roughly as Z^4."""
    zs = np.array([site.specie.Z for site in st])
    return [zs.max(), (zs ** 2).mean() / 1e3, (zs ** 4).mean() / 1e7,
            (zs >= HEAVY_Z).mean(), gap]


def heavy(st) -> bool:
    return max(site.specie.Z for site in st) >= HEAVY_Z


def load(P):
    alex, c2 = id_prop(ALEX), id_prop(C2DB)
    groups = defaultdict(lambda: {"alexandria": [], "c2db": []})
    for r in csv.DictReader(open(TWINS)):
        if r["source"] in ("alexandria", "c2db"):
            groups[r["group"]][r["source"]].append(r["id"])
    rows, dropped = [], defaultdict(int)
    for g, m in groups.items():
        a = [i for i in m["alexandria"] if i in alex and i not in KNOWN_BAD_REFERENCE]
        c = [i for i in m["c2db"] if i in c2]
        if not a or not c:
            continue
        ya = [alex[i] for i in a]
        if max(ya) - min(ya) > ALEX_SPREAD_MAX:
            dropped["Alexandria labels disagree across cells"] += 1
            continue
        rows.append({"alex": a[0], "c2db": c[0], "y_alex": float(np.median(ya)),
                     "y_c2db": float(np.median([c2[i] for i in c]))})
    print(f"{len(rows)} structures in both databases; dropped {dict(dropped)}")

    sts = {r["alex"]: read_structure(os.path.join(ALEX, r["alex"] + ".vasp")) for r in rows}
    res = dict(P.run_many(list(sts.items()), batch_size=128))
    keep = [r for r in rows if res.get(r["alex"]) is not None
            and not res[r["alex"]].verdict.startswith("out-of-domain")]
    print(f"  {len(rows) - len(keep)} out-of-domain for the gap model -> {len(keep)} kept")
    gap = np.array([res[r["alex"]].gap for r in keep])
    lat = embed(P, [sts[r["alex"]] for r in keep])
    comp_f = np.array([soc_features(sts[r["alex"]], g) for r, g in zip(keep, gap)])
    hv = np.array([heavy(sts[r["alex"]]) for r in keep])
    comp = np.array([sts[r["alex"]].composition.reduced_formula for r in keep])
    delta = np.array([r["y_c2db"] - r["y_alex"] for r in keep])
    return keep, gap, lat, comp_f, hv, comp, delta


def cross_validate(feats, hv, comp, delta):
    preds = {k: np.zeros((SHUFFLES, len(delta))) for k in ("A", *feats)}
    for s in range(SHUFFLES):
        for tr, te in make_folds(comp, s):
            for h in (True, False):
                sel = te[hv[te] == h]
                preds["A"][s, sel] = delta[tr][hv[tr] == h].mean()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    P = Predictor(verbose=False)
    keep, gap, lat, comp_f, hv, comp, delta = load(P)
    print(f"  heavy-element compounds {hv.sum()}, others {(~hv).sum()}; "
          f"mean shift {delta[hv].mean():+.3f} / {delta[~hv].mean():+.3f} eV, "
          f"{len(set(comp))} compositions")
    feats = {"B": comp_f, "C": np.column_stack([lat, gap]),
             "D": np.column_stack([lat, gap, comp_f[:, :4]])}
    scores, cvp = cross_validate(feats, hv, comp, delta)

    best = min(("B", "C", "D"), key=lambda k: scores[k]["all"])
    ok1 = scores["A"]["all"] - scores[best]["all"] >= 0.02
    ok2 = scores[best]["light"] <= scores["A"]["light"] + 0.01
    print(f"\ndecision rule: best {best}; beats A by {scores['A']['all'] - scores[best]['all']:+.3f} "
          f"(need >= 0.02) {ok1}; light {scores[best]['light']:.3f} vs A "
          f"{scores['A']['light']:.3f} (+0.01 allowed) {ok2}")
    passed = ok1 and ok2

    # the four TMDs: cross-validated shift against C2DB's own and the literature
    print("\nTMDs (cross-validated, never in their own fold):")
    for name, lit in LITERATURE.items():
        idx = np.where(comp == name)[0]
        for i in idx[:1]:
            print(f"  {name:6s} C2DB-Alexandria {delta[i]:+.3f}  {best} {cvp[best][i]:+.3f}  "
                  f"A {cvp['A'][i]:+.3f}  literature {lit:+.2f}")

    # external check: C2DB rows with no Alexandria twin never entered the fit
    sc, m = ridge(feats[best], delta, np.arange(len(delta)))
    twinned = {r["c2db"] for r in keep}
    import pandas as pd
    t = pd.read_csv(TABLE)
    t = t[(t.source == "c2db") & ~t.id.isin(twinned) & ~t.verdict.str.startswith("out-of-domain")]
    sts = [read_structure(os.path.join(C2DB, i + ".vasp")) for i in t.id]
    g = t.pred_gap_eV.to_numpy()
    if best == "B":
        Xe = np.array([soc_features(s, x) for s, x in zip(sts, g)])
    else:
        Xe = np.column_stack([embed(P, sts), g])
        if best == "D":
            Xe = np.column_stack([Xe, np.array([soc_features(s, x) for s, x in zip(sts, g)])[:, :4]])
    shift = m.predict(sc.transform(Xe))
    hvx = np.array([heavy(s) for s in sts])
    y, iv = t.dft_gap_eV.to_numpy(), t.interval90_eV.to_numpy()
    a_shift = np.where(hvx, delta[hv].mean(), delta[~hv].mean())
    print(f"\nexternal: {len(t)} C2DB rows with no Alexandria twin, in-domain, never fitted on")
    for lab, sel in (("all", np.ones(len(t), bool)), ("heavy", hvx), ("light", ~hvx)):
        e0 = np.abs(g - y)[sel]
        eA = np.abs(g + a_shift - y)[sel]
        eh = np.abs(g + shift - y)[sel]
        c0 = (np.abs(g - y) <= iv)[sel].mean()
        ch = (np.abs(g + shift - y) <= iv)[sel].mean()
        print(f"  {lab:5s} n={sel.sum():4d}  MAE vs C2DB: PBE {e0.mean():.3f}  +A {eA.mean():.3f}  "
              f"+{best} {eh.mean():.3f}   interval covers C2DB {100 * c0:.0f}% -> {100 * ch:.0f}%")

    if not passed:
        print("\nno candidate passed the rule: nothing written")
        return
    payload = {"head": collapse(sc, m), "features": best, "n": int(len(delta)),
               "n_compositions": int(len(set(comp))), "mae_cv": scores,
               "heavy_z": HEAVY_Z,
               "source": "C2DB (PBE+SOC) minus Alexandria (PBE) on shared structures"}
    if not args.write:
        print("\n--write not given: checkpoint untouched")
        return
    ck = torch.load(ENSEMBLE, map_location="cpu")
    ck["soc_head"] = payload
    torch.save(ck, ENSEMBLE)
    print(f"\nwritten soc_head into {ENSEMBLE}")


if __name__ == "__main__":
    main()
