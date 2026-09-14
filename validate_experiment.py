"""External validation: model (PBE level) vs EXPERIMENT on canonical monolayers.

Why: the model is trained on PBE gaps (Alexandria). PBE systematically
UNDERESTIMATES the true electronic gap. We run reference monolayers with measured
gaps through the tool and quantify the PBE -> experiment shift, then fit a rough
linear correction that screen_bandgap.py applies.

Physics caveat (stated honestly): measured monolayer gaps are usually OPTICAL
(exciton binding of ~0.3-0.7 eV in 2D), while PBE gives the Kohn-Sham gap. PBE
underestimates the quasiparticle gap, but optical = quasiparticle - exciton, so
the two errors partially cancel. That is why TMDs come out close to experiment
while wide-gap h-BN is underestimated the most.

Structures: examples/*.vasp (JARVIS-DFT dft_2d relaxed monolayers, shipped with
the repo, so this runs offline). Experimental values: literature optical gaps.

    python validate_experiment.py            # table
    python validate_experiment.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from nanomat import Predictor
from nanomat.predict import read_structure

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")

# file -> (label, experimental gap eV, experimental type, note)
REF = {
    "MoS2.vasp":        ("MoS2",        1.88, "direct",   "optical, monolayer"),
    "MoSe2.vasp":       ("MoSe2",       1.55, "direct",   "optical, monolayer"),
    "WS2.vasp":         ("WS2",         2.00, "direct",   "optical, monolayer"),
    "WSe2.vasp":        ("WSe2",        1.65, "direct",   "optical, monolayer"),
    "hBN.vasp":         ("h-BN",        6.00, "indirect", "wide-gap insulator"),
    "phosphorene.vasp": ("phosphorene", 2.00, "direct",   "optical; transport gap lower"),
    "graphene.vasp":    ("graphene",    0.00, "—",        "semimetal, out-of-domain sanity check"),
}


def evaluate(P: Predictor | None = None) -> list[dict]:
    P = P or Predictor(verbose=False)
    rows = []
    for fname, (name, exp, etype, note) in REF.items():
        path = os.path.join(EXAMPLES, fname)
        if not os.path.exists(path):
            continue
        r = P.run(read_structure(path))
        rows.append({
            "material": name, "model": r.gap, "unc": r.unc, "exp": exp,
            "type_model": r.gap_type or "—", "type_exp": etype,
            "verdict": r.verdict, "note": note,
        })
    return rows


def summarize(rows: list[dict], trusted_only: bool = True) -> dict:
    """Systematics of the PBE -> experiment shift.

    The linear correction is fitted only on points the tool itself calls usable:
    a prediction the model flags as out-of-domain must not steer the calibration
    that every other prediction is corrected by.
    """
    semi = [r for r in rows if r["exp"] > 0.1]
    if trusted_only:
        kept = [r for r in semi if not r["verdict"].startswith("out-of-domain")]
        if len(kept) >= 3:
            semi = kept
    mod = np.array([r["model"] for r in semi])
    ex = np.array([r["exp"] for r in semi])
    a, b = np.polyfit(mod, ex, 1)
    return {
        "n_semiconductors": len(semi),
        "fitted_on": [r["material"] for r in semi],
        "mean_shift_eV": float(np.mean(ex - mod)),
        "mean_rel_underestimate_pct": float(100 * np.mean((ex - mod) / ex)),
        "corr_a": float(a), "corr_b": float(b),
        "mae_raw_eV": float(np.mean(np.abs(mod - ex))),
        "mae_corrected_eV": float(np.mean(np.abs(a * mod + b - ex))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write rows + summary to this JSON file")
    args = ap.parse_args()

    rows = evaluate()
    print(f"\n{'material':12s} {'model(PBE)':>12s} {'exp':>6s} {'Δ(exp-mod)':>11s} "
          f"{'type-model':>11s} {'type-exp':>9s}  verdict")
    print("-" * 90)
    for r in rows:
        print(f"{r['material']:12s} {r['model']:7.2f}±{r['unc']:.2f} {r['exp']:6.2f} "
              f"{r['exp'] - r['model']:11.2f} {r['type_model']:>11s} {r['type_exp']:>9s}  {r['verdict']}")

    s = summarize(rows)
    print("\n--- systematics (semiconductors only) ---")
    print(f"mean shift (exp - model): {s['mean_shift_eV']:+.2f} eV "
          f"(PBE {'under' if s['mean_shift_eV'] > 0 else 'over'}estimates)")
    print(f"mean relative underestimate: {s['mean_rel_underestimate_pct']:.0f}%")
    print(f"\nfitted on (out-of-domain points excluded): {', '.join(s['fitted_on'])}")
    print(f"linear correction PBE -> exp:  exp ≈ {s['corr_a']:.2f}·gap_PBE + {s['corr_b']:+.2f}")
    print(f"MAE before correction: {s['mae_raw_eV']:.2f} eV")
    print(f"MAE after correction:  {s['mae_corrected_eV']:.2f} eV")
    print("\nReading: TMDs land close to the optical gap (exciton binding cancels part of "
          "the PBE underestimate); h-BN is underestimated the most (wide gap). Graphene is "
          "a semimetal outside the training domain: the value is meaningless, but the "
          "ensemble uncertainty is several times higher, which is exactly the signal the "
          "verdict uses.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "summary": s}, f, indent=2)
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
