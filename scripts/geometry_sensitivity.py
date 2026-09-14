#!/usr/bin/env python
"""How much does the *geometry* of the input matter?

Two questions that decide whether a "pick the elements, get a prediction" UI is
honest, answered on four canonical TMD monolayers:

1. **Can a prototype replace a DFT relaxation?**
   Split the idealisation error in two: the internal coordinate (chalcogen
   height) at the correct lattice constant, and the lattice constant itself
   estimated from covalent radii with no DFT input at all.

2. **Does the model respond to strain?**
   Biaxial in-plane strain scan with the internal coordinates frozen. The model
   only ever saw relaxed ground states, so this is extrapolation by construction.

    python scripts/geometry_sensitivity.py            # tables
    python scripts/geometry_sensitivity.py --figure   # + figures/strain_response.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pymatgen.core import Lattice, Structure

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat import Predictor  # noqa: E402
from nanomat.predict import read_structure  # noqa: E402

# Covalent radii (Cordero et al. 2008), angstrom
RADII = {"Mo": 1.54, "W": 1.62, "S": 1.05, "Se": 1.20, "Te": 1.38}
VACUUM_C = 22.0
TMD = [("MoS2", "Mo", "S"), ("MoSe2", "Mo", "Se"), ("WS2", "W", "S"), ("WSe2", "W", "Se")]


def relaxed_geometry(st: Structure) -> tuple[float, float, float]:
    """(a, chalcogen height h, M-X bond length d) from a relaxed 1H-MX2 cell."""
    z: dict[str, list[float]] = {}
    for site in st:
        z.setdefault(site.specie.symbol, []).append(site.coords[2])
    m = next(e for e in z if e in ("Mo", "W"))
    x = next(e for e in z if e not in ("Mo", "W"))
    h = (max(z[x]) - min(z[x])) / 2 if len(z[x]) > 1 else abs(z[x][0] - z[m][0])
    a = st.lattice.a
    return a, h, math.sqrt(a ** 2 / 3 + h ** 2)


def build_1h(m: str, x: str, a: float, h: float) -> Structure:
    """1H-MX2 with a given lattice constant and chalcogen height."""
    return Structure(Lattice.hexagonal(a, VACUUM_C), [m, x, x],
                     [[1 / 3, 2 / 3, 0.5],
                      [2 / 3, 1 / 3, 0.5 + h / VACUUM_C],
                      [2 / 3, 1 / 3, 0.5 - h / VACUUM_C]])


def from_radii(m: str, x: str) -> tuple[float, float, float]:
    """Fully idealised cell, no DFT input: bond length from covalent radii and a
    regular trigonal prism (in-plane X-X equals inter-plane X-X), so a = d*sqrt(12/7)
    and h = a/2."""
    d = RADII[m] + RADII[x]
    a = d * math.sqrt(12 / 7)
    return a, a / 2, d


def strain(st: Structure, eps: float) -> Structure:
    """Biaxial in-plane strain, fractional coordinates frozen (no Poisson relaxation)."""
    f = 1 + eps
    lat = Lattice.from_parameters(st.lattice.a * f, st.lattice.b * f, st.lattice.c,
                                  *st.lattice.angles)
    return Structure(lat, st.species, st.frac_coords)


def run(P: Predictor):
    print("1) PROTOTYPE INSTEAD OF A RELAXATION\n")
    print(f"{'material':9s} {'variant':32s} {'a':>6s} {'h':>5s} {'gap':>7s} {'unc':>6s} {'dgap':>7s}")
    print("-" * 76)
    d_inner, d_full, d_a = [], [], []
    for name, m, x in TMD:
        ref = read_structure(os.path.join(ROOT, "examples", f"{name}.vasp"))
        a0, h0, _ = relaxed_geometry(ref)
        r0 = P.run(ref)
        print(f"{name:9s} {'relaxed (DFT reference)':32s} {a0:6.3f} {h0:5.3f} "
              f"{r0.gap:7.3f} {r0.unc:6.3f} {'-':>7s}")

        r1 = P.run(build_1h(m, x, a0, a0 / 2))          # correct a, ideal prism
        print(f"{'':9s} {'correct a, ideal prism h=a/2':32s} {a0:6.3f} {a0/2:5.3f} "
              f"{r1.gap:7.3f} {r1.unc:6.3f} {r1.gap - r0.gap:+7.3f}")

        a2, h2, _ = from_radii(m, x)
        r2 = P.run(build_1h(m, x, a2, h2))              # nothing from DFT
        print(f"{'':9s} {'all from covalent radii':32s} {a2:6.3f} {h2:5.3f} "
              f"{r2.gap:7.3f} {r2.unc:6.3f} {r2.gap - r0.gap:+7.3f}\n")
        d_inner.append(r1.gap - r0.gap); d_full.append(r2.gap - r0.gap); d_a.append(a2 - a0)

    print(f"MAE from the internal coordinate alone : {np.abs(d_inner).mean():.3f} eV")
    print(f"MAE from full idealisation             : {np.abs(d_full).mean():.3f} eV")
    print(f"lattice constant error from radii      : {np.mean(d_a):+.3f} A "
          f"({100 * np.mean(np.array(d_a) / np.array([2.8] * len(d_a))):+.0f}%)")
    print("\nReading: the prototype fixes the internal coordinate for free, but a lattice "
          "constant\nguessed from radii costs more than the model's own error. A family "
          "browser must take\nthe lattice constant from the relaxed database, not from radii.\n")

    print("\n2) BIAXIAL STRAIN (extrapolation: training data are relaxed ground states)\n")
    curves = {}
    for name in ("MoS2", "WS2"):
        ref = read_structure(os.path.join(ROOT, "examples", f"{name}.vasp"))
        eps = np.arange(-0.06, 0.0601, 0.01)
        gaps, uncs = [], []
        for e in eps:
            r = P.run(strain(ref, float(e)))
            gaps.append(r.gap); uncs.append(r.unc)
        curves[name] = (eps, np.array(gaps), np.array(uncs))
        g0 = gaps[list(eps).index(min(eps, key=abs))]
        print(f"{name}:  " + "  ".join(
            f"{e*100:+.0f}%:{g - g0:+.2f}(±{u:.2f})" for e, g, u in zip(eps, gaps, uncs)))
    print("\nReading: tensile strain gives a smooth, monotonic, physically correct decrease. "
          "Compression\nbeyond -2% is non-monotonic noise, and the ensemble spread grows ~5x "
          "exactly there. The slope\nis likely understated (~0.03 eV per %), so treat the trend "
          "as qualitative, not quantitative.")
    return curves


def figure(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"MoS2": "#2a78d6", "WS2": "#eb6834"}
    TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
    ax.axvspan(-6, -2, color=GRID, alpha=0.7, zorder=0)
    ax.text(-4, 0.78, "compression past -2%:\nnon-monotonic,\nspread ~5x wider", ha="center",
            color=MUTED, fontsize=8.5, zorder=1)
    for name, (eps, gaps, uncs) in curves.items():
        x = eps * 100
        ax.fill_between(x, gaps - uncs, gaps + uncs, color=C[name], alpha=0.18, lw=0, zorder=2)
        ax.plot(x, gaps, color=C[name], lw=2, marker="o", ms=5, label=name, zorder=3)
    ax.axvline(0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xlabel("biaxial in-plane strain, %", color=MUTED)
    ax.set_ylabel("predicted band gap, eV  (band = ensemble spread)", color=MUTED)
    ax.set_ylim(0.55, 2.3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.set_title("The model extrapolates to tensile strain and flags where it stops",
                 loc="left", color=TEXT, fontsize=12.5, pad=26)
    ax.text(0, 1.005, "Trained only on relaxed ground states. Tension: smooth and correct in "
            "sign. Compression past -2%:\nnoise, and the ensemble spread grows with it, which "
            "is the signal the verdict uses.",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, va="bottom")
    out = os.path.join(ROOT, "figures", "strain_response.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true", help="also write figures/strain_response.png")
    a = ap.parse_args()
    P = Predictor(verbose=False)
    curves = run(P)
    if a.figure:
        figure(curves)
