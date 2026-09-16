"""Recognise the handful of 2D structural prototypes that people think in terms of.

A materials person does not ask "what is the gap of agm2000111654", they ask "what
happens across the MX2 family when I swap the chalcogen". This module tags a
structure with its prototype so predictions can be pivoted into that shape.

Only prototypes with an unambiguous signature are recognised; anything else is
left untagged rather than guessed at. The distinction that matters most here is
1H against 1T, because they are the same composition with different coordination
and genuinely different gaps — calling them one family would average two different
materials into one cell.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure

# (code, human label, axis labels) for everything this module can emit
FAMILIES = {
    "1H-MX2": ("1H-MX₂", "trigonal prismatic, the MoS₂ prototype"),
    "1T-MX2": ("1T-MX₂", "octahedral, the CdI₂ prototype"),
    "hc-AB": ("honeycomb AB", "two elements, the h-BN prototype"),
    "hc-A": ("honeycomb A", "one element, the graphene prototype"),
}

_TOL_ANGLE = 2.0    # degrees off 120 still counts as hexagonal
_TOL_FRAC = 0.02    # fractional in-plane distance for "stacked directly above"


def _hexagonal(st: Structure) -> bool:
    a, b, _ = st.lattice.abc
    return abs(st.lattice.angles[2] - 120) <= _TOL_ANGLE and abs(a - b) / max(a, b) < 0.02


def _inplane_same(f1: np.ndarray, f2: np.ndarray) -> bool:
    """Do two sites sit on the same in-plane position, across periodic images?"""
    d = np.abs(f1[:2] - f2[:2]) % 1.0
    d = np.minimum(d, 1.0 - d)
    return bool(np.all(d < _TOL_FRAC))


def classify(st: Structure) -> tuple[str | None, str | None, str | None]:
    """Return (family code, element A, element B), or (None, None, None).

    For MX2 families A is the metal (the single site) and B the chalcogen. For the
    honeycombs A and B are the two sublattice elements, ordered alphabetically so
    the same pair always lands in the same cell.
    """
    if not _hexagonal(st):
        return None, None, None
    symbols = [s.specie.symbol for s in st]
    counts = {e: symbols.count(e) for e in set(symbols)}

    if len(st) == 3 and sorted(counts.values()) == [1, 2]:
        m = next(e for e, c in counts.items() if c == 1)
        x = next(e for e, c in counts.items() if c == 2)
        xs = [s.frac_coords for s in st if s.specie.symbol == x]
        # 1H stacks the two chalcogens directly above one another (prismatic);
        # 1T staggers them (octahedral)
        code = "1H-MX2" if _inplane_same(xs[0], xs[1]) else "1T-MX2"
        return code, m, x

    if len(st) == 2:
        if len(counts) == 2:
            a, b = sorted(counts)
            return "hc-AB", a, b
        a = next(iter(counts))
        return "hc-A", a, a

    return None, None, None


def classify_many(structures) -> list[tuple[str | None, str | None, str | None]]:
    return [classify(st) for st in structures]
