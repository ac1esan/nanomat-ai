#!/usr/bin/env python
"""Fit the two corrections that turn a PBE-level prediction into something measurable.

The model is trained on PBE gaps, which are not what anyone measures. There are two
different physical targets, and conflating them is the mistake this script exists to
avoid:

  quasiparticle gap  - what you would measure in photoemission, and what a transport
                       or device calculation wants. Approximated here by HSE06.
  optical gap        - the absorption onset, which is the quasiparticle gap minus the
                       exciton binding energy. In a 2D monolayer that binding energy
                       is several tenths of an eV, so the two differ a lot.

Both are fitted against the model's own output rather than against PBE, so each
absorbs the functional difference and the model's own bias in one step.

  quasiparticle: 32 structures from JARVIS dft_2d that carry `hse_gap`, restricted
                 to ones the tool does not flag as out-of-domain. Validated
                 leave-one-out.
  optical:       184 non-magnetic materials from C2DB that carry both a G0W0 gap and
                 a BSE exciton binding energy, so the target is G0W0 minus exciton.
                 Validated by 10-fold cross-validation.

WHAT CHANGED, AND WHY IT MATTERS. The optical correction used to be fitted on the
five reference monolayers in examples/. Five points spanning two clusters gave a line
with a negative intercept, which put a third of the screening table's optical gaps
BELOW their own raw PBE gap and 1 870 of them at or below zero - backwards, since PBE
underestimates every measurement. Its leave-one-out error was 0.71 eV against an
in-sample 0.17. Refitting on C2DB's many-body data fixes the shape and the sample
size at once, and it is checked against those same five measured monolayers, which it
never sees: 0.30 eV against the old fit's 0.71.

THE TWO CORRECTIONS DO NOT DIFFER BY THE EXCITON BINDING ENERGY. An earlier version
of this file said they did, on the strength of four TMDs that both fits had been
trained on. They are referenced to different quantities - HSE06 against G0W0 minus an
exciton - and HSE06 runs about 0.4 eV below G0W0 at a 1.7 eV gap, so their difference
mixes the exciton with that discrepancy. The exciton binding energy is reported here
directly from BSE instead.

    python scripts/fit_gap_corrections.py              # fit and report
    python scripts/fit_gap_corrections.py --write      # also store in the checkpoint

Needs data/c2db_optical.csv - run scripts/fetch_c2db_optical.py once to create it.
Run without a proxy if one intercepts figshare:
    env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy python scripts/...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat import Predictor  # noqa: E402
from nanomat.predict import correct, read_structure  # noqa: E402

ENSEMBLE = os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.pt")
C2DB = os.path.join(ROOT, "data", "c2db_optical.csv")
C2DB_STRUCTURES = os.path.join(ROOT, "data_c2db_wf")
# measured optical gaps, same references validate_experiment.py uses. These are the
# HELD-OUT check now: nothing is fitted on them, so the error against them is honest.
OPTICAL = {"MoS2": 1.88, "MoSe2": 1.55, "WS2": 2.00, "WSe2": 1.65,
           "hBN": 6.00, "phosphorene": 2.00, "graphene": 0.00}
MAX_DEGREE = 3


def loo_mae(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out error of a linear fit: refit without each point, predict it."""
    errs = []
    for i in range(len(x)):
        k = np.arange(len(x)) != i
        a, b = np.polyfit(x[k], y[k], 1)
        errs.append(abs(a * x[i] + b - y[i]))
    return float(np.mean(errs))


def cv_mae(x: np.ndarray, y: np.ndarray, degree: int, folds: int = 10,
           seeds: int = 12) -> tuple[float, float]:
    """(mean, sd) k-fold error over several shuffles.

    Several seeds because the choice between a line and a curve here comes down to
    about 0.006 eV, and one shuffle cannot tell that from noise.
    """
    per_seed = []
    for seed in range(seeds):
        idx = np.random.default_rng(seed).permutation(len(x))
        errs = []
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            p = np.polyfit(x[tr], y[tr], degree)
            errs += list(np.abs(np.polyval(p, x[te]) - y[te]))
        per_seed.append(np.mean(errs))
    return float(np.mean(per_seed)), float(np.std(per_seed))


def fit_quasiparticle(P: Predictor) -> dict:
    from jarvis.core.atoms import Atoms
    from jarvis.db.figshare import data as jarvis_data

    num = lambda v: pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    raw = jarvis_data("dft_2d")
    rows = []
    for e in raw:
        hse = num(e.get("hse_gap"))
        if pd.isna(hse) or hse <= 0.01:
            continue
        rows.append({"jid": e.get("jid"), "hse": float(hse), "atoms": e["atoms"]})
    print(f"dft_2d entries carrying an HSE gap: {len(rows)}")

    items = [(r["jid"], Atoms.from_dict(r["atoms"]).pymatgen_converter()) for r in rows]
    res = dict(P.run_many(items, batch_size=128))
    keep_x, keep_y, dropped = [], [], 0
    for r in rows:
        pr = res.get(r["jid"])
        if pr is None or pr.verdict.startswith("out-of-domain"):
            dropped += 1
            continue
        keep_x.append(pr.gap)
        keep_y.append(r["hse"])
    x, y = np.array(keep_x), np.array(keep_y)
    print(f"  dropped as out-of-domain: {dropped}  -> fitting on {len(x)}")

    a, b = np.polyfit(x, y, 1)
    out = {
        "a": float(a), "b": float(b), "n": int(len(x)),
        "mae_in_sample": float(np.mean(np.abs(a * x + b - y))),
        "mae_loo": loo_mae(x, y),
        "mae_uncorrected": float(np.mean(np.abs(x - y))),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "source": "JARVIS dft_2d hse_gap",
    }
    print(f"  raw prediction vs HSE: MAE {out['mae_uncorrected']:.3f} eV, r = {out['pearson']:.3f}")
    print(f"  fit: HSE ~= {a:.3f}*pred {b:+.3f}")
    print(f"  MAE in-sample {out['mae_in_sample']:.3f}, leave-one-out {out['mae_loo']:.3f} eV")
    return out


def load_c2db(P: Predictor) -> list[dict]:
    """C2DB materials with both halves of an optical gap, run through the model.

    Magnetic ones are dropped: their BSE binding energies are a different regime,
    localised d-states rather than a Wannier exciton. VBr2 comes out with a binding
    energy 0.94 times its own gap, against a median ratio of 0.29.
    """
    if not os.path.exists(C2DB):
        raise SystemExit(f"missing {C2DB} - run scripts/fetch_c2db_optical.py first")
    from pymatgen.core import Structure

    raw = [r for r in csv.DictReader(open(C2DB))
           if r["gap_dir_gw"] and r["E_B"] and r["magnetic"] == "No"]
    items = [(r["uid"], Structure.from_file(os.path.join(C2DB_STRUCTURES, r["uid"] + ".vasp")))
             for r in raw]
    res = dict(P.run_many(items, batch_size=128))
    out, ood, nonpos = [], 0, 0
    for r in raw:
        pr = res.get(r["uid"])
        if pr is None:
            continue
        if pr.verdict.startswith("out-of-domain"):
            ood += 1
            continue
        optical = float(r["gap_dir_gw"]) - float(r["E_B"])
        if optical <= 0.1:
            nonpos += 1
            continue
        out.append({"uid": r["uid"], "formula": r["formula"], "pred": pr.gap,
                    "optical": optical, "E_B": float(r["E_B"]),
                    "gw_dir": float(r["gap_dir_gw"])})
    print(f"C2DB: {len(raw)} non-magnetic materials with G0W0 + BSE; dropped {ood} the "
          f"tool calls out-of-domain and {nonpos} with no optical gap left")
    return out


def fit_optical(P: Predictor) -> tuple[dict, list[dict]]:
    rows = load_c2db(P)
    x = np.array([r["pred"] for r in rows])
    y = np.array([r["optical"] for r in rows])
    print(f"\noptical fit on {len(x)} materials, predicted gaps {x.min():.2f}-{x.max():.2f} eV, "
          f"targets {y.min():.2f}-{y.max():.2f} eV (pearson r = {np.corrcoef(x, y)[0, 1]:.3f})")

    # choose the shape by cross-validation, not by eye: a line is biased high through
    # the middle of the range, and the extra curvature has to earn its parameter
    print(f"  {'degree':>6s} {'in-sample':>10s} {'10-fold':>9s} {'sd':>7s}")
    scores = {}
    for d in range(1, MAX_DEGREE + 1):
        p = np.polyfit(x, y, d)
        m, sd = cv_mae(x, y, d)
        scores[d] = (m, sd, p)
        print(f"  {d:6d} {np.mean(np.abs(np.polyval(p, x) - y)):10.3f} {m:9.3f} {sd:7.4f}")
    degree = min(scores, key=lambda d: scores[d][0])
    mae_cv, sd_cv, coeffs = scores[degree]
    print(f"  -> degree {degree} wins on cross-validation")

    out = {
        "coeffs": [float(c) for c in coeffs], "degree": int(degree),
        "n": int(len(x)),
        "mae_in_sample": float(np.mean(np.abs(np.polyval(coeffs, x) - y))),
        "mae_cv": float(mae_cv), "mae_cv_sd": float(sd_cv),
        "mae_uncorrected": float(np.mean(np.abs(x - y))),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "reference": "G0W0 - BSE exciton",
        "source": "C2DB (Haastrup 2018, Gjerding 2021), CC-BY-SA 4.0",
    }
    print(f"  fit: optical ~= " + " + ".join(
        f"{c:.4f}" + ("" if i == len(coeffs) - 1 else f"*g^{len(coeffs)-1-i}".replace("^1", ""))
        for i, c in enumerate(coeffs)))
    print(f"  raw prediction vs target: MAE {out['mae_uncorrected']:.3f} eV")
    return out, rows


def check_against_measurement(P: Predictor, qp: dict, opt: dict) -> dict:
    """Score the fits on the measured monolayers. The optical fit never saw them."""
    from nanomat.predict import correct as apply_correction

    OLD = [1.390, -0.442]   # the five-point fit this replaces
    rows, errs_new, errs_old = [], [], []
    print(f"\nheld-out check on measured monolayers (the optical fit never saw these)")
    print(f"  {'material':13s} {'pred':>6s} {'new':>6s} {'old 5-pt':>9s} {'measured':>9s}")
    for name, exp in OPTICAL.items():
        path = os.path.join(ROOT, "examples", f"{name}.vasp")
        if not os.path.exists(path):
            continue
        r = P.run(read_structure(path))
        ood = r.verdict.startswith("out-of-domain")
        new = apply_correction(r.gap, "optical", {"optical": opt})
        old = OLD[0] * r.gap + OLD[1]
        rows.append({"material": name, "pred": r.gap, "new": new, "old": old,
                     "measured": exp, "ood": ood})
        if exp > 0.1 and not ood:
            errs_new.append(abs(new - exp))
            errs_old.append(abs(old - exp))
        flag = "  (out-of-domain, not scored)" if ood else ""
        print(f"  {name:13s} {r.gap:6.2f} {new:6.2f} {old:9.2f} {exp:9.2f}{flag}")
    out = {"n": len(errs_new), "mae_new": float(np.mean(errs_new)),
           "mae_old_in_sample": float(np.mean(errs_old)), "mae_old_loo": 0.705}
    print(f"\n  new fit, never trained on these : MAE {out['mae_new']:.3f} eV")
    print(f"  old fit, trained ON these       : MAE {out['mae_old_in_sample']:.3f} eV "
          "<- in-sample, not a fair number")
    print(f"  old fit, leave-one-out          : MAE 0.705 eV <- the comparable one")
    return out


def report_exciton(rows: list[dict], qp: dict, opt: dict) -> dict:
    """The exciton binding energy, reported from BSE rather than inferred.

    The previous version of this script inferred it from the gap between the two
    corrections and found 0.55 eV on the TMDs, matching the published value. That
    agreement was circular: both fits had been trained on those same four materials.
    With the optical fit moved onto 184 materials the inferred difference collapses
    to about 0.3 eV at a TMD, while BSE says 0.5 - which is the honest way to find
    out that the inference was never valid. The two corrections are referenced to
    different quantities, so their difference is not a physical energy.
    """
    eb = np.array([r["E_B"] for r in rows])
    ratio = eb / np.array([r["gw_dir"] for r in rows])
    print(f"\nexciton binding energy, straight from BSE on {len(rows)} materials:")
    print(f"  E_B spans {eb.min():.2f}-{eb.max():.2f} eV, median {np.median(eb):.2f}")
    print(f"  E_B / G0W0 gap: median {np.median(ratio):.2f}, sd {np.std(ratio):.2f} "
          "-- the published E_g/4 scaling for 2D sits inside that")
    print(f"  {'material':10s} {'BSE E_B':>8s} {'implied by the two fits':>24s}")
    tmd = []
    for r in rows:
        if r["uid"].split("-")[0] not in ("MoS2", "MoSe2", "WS2", "WSe2"):
            continue
        implied = (correct(r["pred"], "quasiparticle", {"quasiparticle": qp})
                   - correct(r["pred"], "optical", {"optical": opt}))
        tmd.append({"material": r["formula"], "bse_eV": round(r["E_B"], 3),
                    "implied_eV": round(float(implied), 3)})
        print(f"  {r['formula']:10s} {r['E_B']:8.2f} {implied:24.2f}")
    print("  the two disagree, and BSE is the one to believe: the corrections are")
    print("  referenced to HSE06 and to G0W0, which differ by ~0.4 eV themselves.")
    return {"median_ratio_to_gw": float(np.median(ratio)),
            "sd_ratio": float(np.std(ratio)),
            "median_eV": float(np.median(eb)), "n": len(rows), "tmd": tmd,
            "note": "measured by BSE in C2DB; NOT the difference between the two "
                    "corrections, which are referenced to different quantities"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="store the fits in the checkpoint")
    args = ap.parse_args()

    P = Predictor(verbose=False)
    qp = fit_quasiparticle(P)
    opt, rows = fit_optical(P)
    held_out = check_against_measurement(P, qp, opt)
    exciton = report_exciton(rows, qp, opt)
    opt["held_out_measured"] = held_out

    payload = {"quasiparticle": qp, "optical": opt, "exciton_binding": exciton}
    if not args.write:
        print("\n--write not given: checkpoint untouched")
        return
    ck = torch.load(ENSEMBLE, map_location="cpu")
    ck["gap_corrections"] = payload
    torch.save(ck, ENSEMBLE)
    print(f"\nwritten gap_corrections into {ENSEMBLE}")


if __name__ == "__main__":
    main()
