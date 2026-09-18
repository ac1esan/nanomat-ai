"""Crystal structure -> graph, plus sanity checks specific to 2D slabs.

Graph definition (must be identical at train and inference time):
  * nodes  = atoms, feature = atomic number Z (embedded inside the model);
  * edges  = all periodic neighbour pairs within `cutoff` angstrom;
  * edge feature = distance only (radial basis expansion happens in the model,
    so 40 floats per edge are never materialised on disk / in RAM).

2D-specific checks:
  * `layer_info`   - detect the vacuum axis, vacuum thickness and slab thickness;
  * `ensure_vacuum`- pad vacuum so no atom sees its own periodic image across the
                     vacuum within the cutoff. With periodic boundary conditions a
                     vacuum thinner than the cutoff creates spurious inter-layer
                     edges and silently corrupts the prediction.
"""

from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Data

DEFAULT_CUTOFF = 8.0   # angstrom, neighbour cutoff used for the shipped weights
DEFAULT_N_RBF = 40     # radial basis functions on [0, cutoff]
MIN_LAYER_VACUUM = 5.0  # below this the cell does not look like an isolated layer

# Angular features live on a much shorter cutoff than the edges do. At 8 A an atom in
# a 2D slab has ~60 neighbours and so ~1 700 triplets, which is both expensive and
# meaningless: a bond angle is a statement about the coordination shell, not about
# atoms 8 A away. 4 A covers the first shell and, in a TMD, the metal-metal ring too.
DEFAULT_ANG_CUTOFF = 4.0
DEFAULT_N_ANG = 9      # soft bins over cos(theta) in [-1, 1]


def _axis_gap(st: Structure, axis: int) -> tuple[float, float, float]:
    """Largest empty gap along a lattice axis (with wrap-around).

    Returns (gap_fraction, start_fraction_of_layer, perpendicular_height).
    `perpendicular_height` is the cell extent perpendicular to the other two
    vectors, so gap_fraction * perpendicular_height is the real vacuum in angstrom
    even for non-orthogonal cells.
    """
    frac = np.mod(st.frac_coords[:, axis], 1.0)
    order = np.argsort(frac)
    s = frac[order]
    if len(s) == 1:
        return 1.0, float(s[0]), _perp_height(st.lattice, axis)
    gaps = np.diff(np.concatenate([s, [s[0] + 1.0]]))
    k = int(np.argmax(gaps))
    start = s[(k + 1) % len(s)]
    return float(gaps[k]), float(start), _perp_height(st.lattice, axis)


def _perp_height(lat: Lattice, axis: int) -> float:
    m = lat.matrix
    j, k = [i for i in range(3) if i != axis]
    area = np.linalg.norm(np.cross(m[j], m[k]))
    return float(lat.volume / area)


def layer_info(st: Structure) -> dict:
    """Detect the vacuum axis of a slab.

    Returns dict(axis, vacuum, thickness, is_layer). `vacuum` and `thickness`
    are in angstrom, measured perpendicular to the layer. `is_layer` is False
    when no axis has a gap of at least MIN_LAYER_VACUUM (looks like bulk).
    """
    best = None
    for axis in range(3):
        gap, start, h = _axis_gap(st, axis)
        vac = gap * h
        if best is None or vac > best["vacuum"]:
            best = {"axis": axis, "vacuum": vac, "thickness": (1.0 - gap) * h,
                    "_start": start, "_gap": gap, "_h": h}
    best["is_layer"] = best["vacuum"] >= MIN_LAYER_VACUUM
    return best


def ensure_vacuum(st: Structure, cutoff: float = DEFAULT_CUTOFF,
                  target_vacuum: float = 20.0) -> tuple[Structure, dict]:
    """Return a structure whose vacuum is safely larger than the cutoff.

    If the detected vacuum already exceeds `cutoff` + 1 A the structure is
    returned unchanged. Otherwise the layer is made contiguous, the vacuum axis
    is stretched so that the vacuum equals `target_vacuum`, and the layer is
    centred. Because the graph only depends on neighbours within the cutoff,
    the resulting graph is identical for any vacuum larger than the cutoff.

    Returns (structure, info) where info = layer_info(...) plus `padded: bool`.
    """
    info = layer_info(st)
    info["padded"] = False
    if not info["is_layer"] or info["vacuum"] > cutoff + 1.0:
        return st, _public(info)

    axis, start, gap, h = info["axis"], info["_start"], info["_gap"], info["_h"]
    # 1) make the layer contiguous starting at fractional 0 along the axis
    shift = np.zeros(3)
    shift[axis] = -start
    s2 = st.copy()
    s2.translate_sites(list(range(len(s2))), shift, frac_coords=True, to_unit_cell=True)
    # 2) stretch the vacuum axis (cartesian coordinates are preserved)
    thickness = (1.0 - gap) * h
    factor = (thickness + target_vacuum) / h
    matrix = s2.lattice.matrix.copy()
    matrix[axis] = matrix[axis] * factor
    s3 = Structure(Lattice(matrix), s2.species, s2.cart_coords, coords_are_cartesian=True)
    # 3) centre the layer in the new cell (cosmetic, graph-invariant)
    new_thick_frac = thickness / (thickness + target_vacuum)
    recentre = np.zeros(3)
    recentre[axis] = 0.5 - new_thick_frac / 2.0
    s3.translate_sites(list(range(len(s3))), recentre, frac_coords=True, to_unit_cell=True)

    info["padded"] = True
    info["vacuum_after"] = target_vacuum
    return s3, _public(info)


def _public(info: dict) -> dict:
    return {k: v for k, v in info.items() if not k.startswith("_")}


def angle_features(st: Structure, cutoff: float = DEFAULT_ANG_CUTOFF,
                   n_bins: int = DEFAULT_N_ANG) -> np.ndarray:
    """Per-atom distribution of bond angles, as a soft histogram over cos(theta).

    Why this exists. The edge feature is a distance and nothing else, so two
    polymorphs with the same composition and the same bond lengths look nearly
    alike to the model. 1H-MX2 and 1T-MX2 are exactly that case - a trigonal prism
    against an octahedron, median X-M-X angle 85.0 against 91.7 degrees - and the
    shipped ensemble compresses their measured gap difference by a factor of four
    (0.44 eV in Alexandria becomes 0.12 eV predicted). An angle is the missing
    coordinate.

    Each triplet (j, i, k) at atom i contributes a Gaussian bump at cos(theta_jik),
    weighted by a smooth cutoff on both bond lengths so the descriptor has no
    discontinuity when a neighbour crosses the radius. The histogram is normalised
    to sum to one: it describes the SHAPE of the coordination, and how many
    neighbours there are is something the atom graph already carries.
    """
    out = np.zeros((len(st), n_bins), dtype=np.float64)
    c, n, img, d = st.get_neighbor_list(r=cutoff)
    if len(c) == 0:
        return out.astype(np.float32)
    vec = st.cart_coords[n] + img @ st.lattice.matrix - st.cart_coords[c]
    envelope = 0.5 * (np.cos(np.pi * np.clip(d / cutoff, 0.0, 1.0)) + 1.0)
    centres = np.linspace(-1.0, 1.0, n_bins)
    width = float(centres[1] - centres[0]) if n_bins > 1 else 1.0

    order = np.argsort(c, kind="stable")
    c_s, vec_s, env_s, d_s = c[order], vec[order], envelope[order], d[order]
    lo = np.searchsorted(c_s, np.arange(len(st)), side="left")
    hi = np.searchsorted(c_s, np.arange(len(st)), side="right")
    for atom in range(len(st)):
        a, b = lo[atom], hi[atom]
        if b - a < 2:
            continue                      # no angle exists with fewer than two bonds
        unit = vec_s[a:b] / np.maximum(d_s[a:b], 1e-9)[:, None]
        w = env_s[a:b]
        iu = np.triu_indices(b - a, k=1)
        cos = np.clip((unit @ unit.T)[iu], -1.0, 1.0)
        pair = np.outer(w, w)[iu]
        bump = np.exp(-0.5 * ((cos[:, None] - centres[None, :]) / width) ** 2)
        out[atom] = (bump * pair[:, None]).sum(0)
    total = out.sum(1, keepdims=True)
    return np.divide(out, total, out=np.zeros_like(out), where=total > 1e-9).astype(np.float32)


def to_graph(st: Structure, cutoff: float = DEFAULT_CUTOFF, n_ang: int = 0,
             ang_cutoff: float = DEFAULT_ANG_CUTOFF) -> Data | None:
    """Periodic neighbour graph with distances as edge weights. None if no edges.

    `n_ang > 0` additionally attaches per-atom angular features. Left at 0 the graph
    is byte-identical to the one the shipped weights were trained on.
    """
    c, n, _img, d = st.get_neighbor_list(r=cutoff)
    if len(c) == 0:
        return None
    z = torch.tensor([s.specie.Z for s in st], dtype=torch.long)
    ei = torch.tensor(np.vstack([c, n]), dtype=torch.long)
    ew = torch.tensor(d, dtype=torch.float)
    data = Data(z=z, edge_index=ei, edge_weight=ew, num_nodes=len(z))
    if n_ang:
        data.ang = torch.tensor(angle_features(st, ang_cutoff, n_ang), dtype=torch.float)
    return data
