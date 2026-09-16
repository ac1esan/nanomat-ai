"""Inference: load shipped weights, run the two-stage pipeline on a structure.

Pipeline for one structure
  1. `ensure_vacuum`     - 2D sanity check + vacuum padding (graph-invariant);
  2. metal gate (opt.)   - CGCNNcls with cls="metal" if weights/cgcnn_2d_metal.pt exists;
  3. gap ensemble        - mean of 5 CGCNN, uncertainty = std between members;
  4. gap type (opt.)     - CGCNNcls with cls="type" (direct / indirect);
  5. PBE -> experiment   - linear correction fitted on reference monolayers;
  6. verdict             - reliable / check / out-of-domain, from uncertainty.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from pymatgen.core import Structure
from torch_geometric.data import Batch

from .graph import DEFAULT_CUTOFF, DEFAULT_N_RBF, ensure_vacuum, to_graph
from .model import CGCNN, CGCNNcls

# --- calibration constants --------------------------------------------------
# Two corrections turn a PBE-level prediction into something measurable, and they
# target DIFFERENT quantities - see scripts/fit_gap_corrections.py:
#   quasiparticle: what photoemission or a transport calculation wants. Fitted against
#     HSE06 on 32 structures; leave-one-out MAE 0.19 eV, so it is properly validated.
#   optical: the absorption onset, which is the quasiparticle gap minus the exciton
#     binding energy. Fitted on only five reference monolayers; in-sample MAE is 0.17 eV
#     but leave-one-out is 0.71 eV, so this one is a rough indication, not a measurement.
# The difference between them is the exciton binding energy, ~0.55 eV on the TMDs,
# which is the published order for a monolayer.
DEFAULT_CORRECTIONS = {
    "quasiparticle": {"a": 1.179, "b": 0.450, "n": 32, "mae_loo": 0.193},
    "optical": {"a": 1.390, "b": -0.442, "n": 5, "mae_loo": 0.705},
}
A_CORR, B_CORR = 1.39, -0.44  # kept for backwards compatibility: the optical fit
METAL_GAP = 0.1  # eV, below this we call it metal / semimetal

# Fallback calibration, used only when the checkpoint carries none. Real numbers are
# written into the checkpoint by scripts/calibrate_uncertainty.py: the raw ensemble
# spread ranks errors well but is overconfident (+-1 sigma covers ~37%, not 68%), so
# intervals are scaled and the verdict tiers sit at uncertainty quartiles.
DEFAULT_CAL = {
    "scale68": 2.37, "scale90": 5.09,
    "unc_median": 0.092, "unc_q75": 0.142,
    "tier_mae": {"reliable": 0.14, "check": 0.25, "out_of_domain": 0.46},
}
MED_UNC = DEFAULT_CAL["unc_median"]  # kept for backwards compatibility

WEIGHT_FILES = {
    "ensemble": "cgcnn_2d_ensemble.pt",
    "single": "cgcnn_2d_bandgap.pt",
    "type": "cgcnn_2d_typed.pt",
    "metal": "cgcnn_2d_metal.pt",
}


def default_weights_dir() -> str:
    """Resolve the weights directory: $NANOMAT_WEIGHTS, then <repo>/weights."""
    env = os.environ.get("NANOMAT_WEIGHTS")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "weights")


def corrected_gap(gap: float, corrections: dict | None = None) -> float:
    """PBE -> optical gap. Kept for backwards compatibility; prefer `correct()`."""
    c = (corrections or DEFAULT_CORRECTIONS)["optical"]
    return c["a"] * gap + c["b"]


def correct(gap: float, kind: str, corrections: dict | None = None) -> float:
    """Apply one of the two corrections. `kind` is "quasiparticle" or "optical"."""
    c = (corrections or DEFAULT_CORRECTIONS)[kind]
    return c["a"] * gap + c["b"]


def verdict(unc: float, cal: dict | None = None, latent: float | None = None) -> str:
    """Trust tier from two independent signals.

    `unc` is the ensemble spread: it ranks errors well, but every member shares one
    training set, so a chemistry none of them saw yields confident agreement
    (phosphorene is the worked example - spread 0.035 eV and an error of 1.2 eV).
    `latent` is the distance to the nearest training structures in the model's own
    embedding space, which catches exactly that case. Thresholds are quartiles
    measured on a held-out split by scripts/calibrate_uncertainty.py.
    """
    cal = cal or DEFAULT_CAL
    if unc <= cal["unc_median"]:
        tier = "reliable"
    elif unc <= cal["unc_q75"]:
        tier = "check (elevated uncertainty)"
    else:
        tier = "out-of-domain (possibly metal / unusual structure)"
    if latent is not None:
        if latent > cal.get("latent_q90", float("inf")):
            return "out-of-domain (chemistry unlike anything in training)"
        if latent > cal.get("latent_q75", float("inf")) and tier == "reliable":
            return "check (chemistry sparsely covered by training data)"
    return tier


def tier_key(v: str) -> str:
    return "reliable" if v.startswith("reliable") else (
        "check" if v.startswith("check") else "out_of_domain")


@dataclass
class Prediction:
    formula: str
    natoms: int
    gap: float                 # eV, PBE level
    unc: float                 # eV, ensemble std (or MC-dropout std)
    verdict: str
    exp_gap_est: float | None  # eV, after PBE->exp correction (None for metals)
    is_metal_like: bool        # gap < METAL_GAP
    gap_quasiparticle: float | None = None  # eV, HSE-level estimate (photoemission / transport)
    latent_distance: float | None = None  # 1 - mean cosine sim to 10 nearest training structures
    interval90: float | None = None  # eV, half-width of the calibrated 90% interval
    typical_error: float | None = None  # eV, measured MAE of this trust tier
    gap_type: str | None = None      # "direct" / "indirect"
    p_indirect: float | None = None
    p_metal: float | None = None     # from the metal gate, if available
    layer: dict = field(default_factory=dict)  # vacuum axis / thickness / padded
    warnings: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        row = {
            "formula": self.formula,
            "natoms": self.natoms,
            "band_gap_PBE_eV": round(self.gap, 3),
            "gap_quasiparticle_eV": None if self.gap_quasiparticle is None else round(self.gap_quasiparticle, 3),
            "exp_gap_est_eV": None if self.exp_gap_est is None else round(self.exp_gap_est, 3),
            "uncertainty_eV": round(self.unc, 3),
            "interval90_eV": None if self.interval90 is None else round(self.interval90, 3),
            "latent_distance": None if self.latent_distance is None else round(self.latent_distance, 3),
            "typical_error_eV": None if self.typical_error is None else round(self.typical_error, 3),
            "verdict": self.verdict,
        }
        if self.p_metal is not None:
            row["p_metal"] = round(self.p_metal, 2)
        if self.gap_type is not None:
            row["gap_type"] = self.gap_type
            row["p_indirect"] = round(self.p_indirect, 2)
        row["vacuum_A"] = round(self.layer.get("vacuum", float("nan")), 1)
        row["warnings"] = "; ".join(self.warnings)
        return row


class Predictor:
    """Loads weights once, predicts many structures on CPU."""

    def __init__(self, weights_dir: str | None = None, verbose: bool = True):
        self.weights_dir = weights_dir or default_weights_dir()
        self.cutoff, self.n_rbf = DEFAULT_CUTOFF, DEFAULT_N_RBF
        self.cal = dict(DEFAULT_CAL)
        self.corr = {k: dict(v) for k, v in DEFAULT_CORRECTIONS.items()}
        self.ref_emb: torch.Tensor | None = None  # normalised training embeddings
        self.models: list[nn.Module] = []
        self.mean = self.std = 0.0
        self.type_model: CGCNNcls | None = None
        self.metal_model: CGCNNcls | None = None
        # decision thresholds: chosen on validation at training time, because
        # pos_weight on the rare class shifts probabilities away from 0.5
        self.type_thr = 0.5
        self.metal_thr = 0.5
        self._log = print if verbose else (lambda *a, **k: None)
        self._load_gap_models()
        self.type_model, self.type_thr = self._load_cls("type")
        self.metal_model, self.metal_thr = self._load_cls("metal")

    # --- loading -------------------------------------------------------------
    def _path(self, key: str) -> str:
        return os.path.join(self.weights_dir, WEIGHT_FILES[key])

    def _load_gap_models(self) -> None:
        ens, one = self._path("ensemble"), self._path("single")
        if os.path.exists(ens):
            ck = torch.load(ens, map_location="cpu")
            self.cutoff, self.n_rbf = float(ck.get("cutoff", self.cutoff)), int(ck.get("n_rbf", self.n_rbf))
            for sd in ck["state_dicts"]:
                m = CGCNN(cutoff=self.cutoff, n_rbf=self.n_rbf)
                m.load_state_dict(sd)
                m.eval()
                self.models.append(m)
            self.mean, self.std = float(ck["mean"]), float(ck["std"])
            if ck.get("reference_embeddings") is not None:
                self.ref_emb = ck["reference_embeddings"].float()
            gc = ck.get("gap_corrections")
            if isinstance(gc, dict):
                for kind in ("quasiparticle", "optical"):
                    if isinstance(gc.get(kind), dict):
                        self.corr[kind].update(gc[kind])
            if isinstance(ck.get("calibration"), dict):
                self.cal.update(ck["calibration"])
                self._log(f"Loaded gap ensemble: {len(self.models)} models, calibrated "
                          f"(90% interval = {self.cal['scale90']:.2f}x spread, "
                          f"test MAE {self.cal.get('test_mae', float('nan')):.3f} eV).")
            else:
                self._log(f"Loaded gap ensemble: {len(self.models)} models "
                          f"(no calibration in checkpoint, using defaults).")
            return
        if os.path.exists(one):
            ck = torch.load(one, map_location="cpu")
            self.cutoff, self.n_rbf = float(ck.get("cutoff", self.cutoff)), int(ck.get("n_rbf", self.n_rbf))
            m = CGCNN(cutoff=self.cutoff, n_rbf=self.n_rbf)
            m.load_state_dict(ck["state_dict"])
            m.eval()
            self.models = [m]
            self.mean, self.std = float(ck["mean"]), float(ck["std"])
            self._log("Loaded single gap model (uncertainty via MC-dropout).")
            return
        raise FileNotFoundError(
            f"No weights found in {self.weights_dir}. Expected "
            f"{WEIGHT_FILES['ensemble']} (preferred) or {WEIGHT_FILES['single']}. "
            "Set NANOMAT_WEIGHTS to point at the directory with the .pt files."
        )

    def _load_cls(self, key: str) -> tuple[CGCNNcls | None, float]:
        p = self._path(key)
        if not os.path.exists(p):
            return None, 0.5
        ck = torch.load(p, map_location="cpu")
        m = CGCNNcls(cutoff=float(ck.get("cutoff", self.cutoff)), n_rbf=int(ck.get("n_rbf", self.n_rbf)))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        thr = float(ck.get("threshold", 0.5))
        self._log(f"Loaded {key} classifier (threshold {thr:.2f}).")
        return m, thr

    # --- inference -----------------------------------------------------------
    def prepare(self, st: Structure) -> tuple[Structure, dict, list[str]]:
        """Vacuum padding + warnings. Returns (structure, layer_info, warnings)."""
        warnings = []
        st2, info = ensure_vacuum(st, cutoff=self.cutoff)
        if not info["is_layer"]:
            warnings.append(
                f"no vacuum gap >= 5 A found (max {info['vacuum']:.1f} A): "
                "does not look like an isolated 2D layer; model trained on monolayers only"
            )
        elif info["padded"]:
            warnings.append(
                f"vacuum was {info['vacuum']:.1f} A < cutoff {self.cutoff:.0f} A; "
                f"padded to {info['vacuum_after']:.0f} A to avoid spurious inter-layer edges"
            )
        return st2, info, warnings

    @torch.no_grad()
    def predict_gap(self, st: Structure, mc: int = 30) -> tuple[float, float] | tuple[None, None]:
        """(gap, uncertainty) in eV. Ensemble: mean/std across members.
        Single model: point estimate + MC-dropout std."""
        g = to_graph(st, self.cutoff)
        if g is None:
            return None, None
        batch = Batch.from_data_list([g])
        if len(self.models) > 1:
            preds = [(m(batch) * self.std + self.mean).item() for m in self.models]
            return float(np.mean(preds)), float(np.std(preds))
        model = self.models[0]
        gap = (model(batch) * self.std + self.mean).item()
        model.train()  # dropout on, BatchNorm kept in eval mode
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.eval()
        preds = [(model(batch) * self.std + self.mean).item() for _ in range(mc)]
        model.eval()
        return gap, float(np.std(preds))

    @torch.no_grad()
    def latent_distance(self, graphs: list, k: int = 10) -> np.ndarray | None:
        """1 - mean cosine similarity to the k nearest training structures.
        None when the checkpoint carries no reference embeddings."""
        if self.ref_emb is None or not graphs:
            return None
        e = self.models[0].encode(Batch.from_data_list(graphs))
        e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-9)
        sims = e @ self.ref_emb.T
        k = min(k, sims.shape[1])
        return (1 - sims.topk(k, dim=1).values.mean(1)).cpu().numpy()

    @torch.no_grad()
    def _cls_prob(self, model: CGCNNcls | None, st: Structure) -> float | None:
        if model is None:
            return None
        g = to_graph(st, self.cutoff)
        if g is None:
            return None
        _, logit = model(Batch.from_data_list([g]))
        return float(torch.sigmoid(logit).item())

    def predict_type(self, st: Structure) -> tuple[str | None, float | None]:
        p = self._cls_prob(self.type_model, st)
        if p is None:
            return None, None
        return ("indirect" if p >= self.type_thr else "direct"), p

    def predict_metal(self, st: Structure) -> float | None:
        return self._cls_prob(self.metal_model, st)

    def run(self, st: Structure, pad_vacuum: bool = True) -> Prediction | None:
        """Full pipeline on one pymatgen Structure. None if no graph could be built."""
        if pad_vacuum:
            st, info, warns = self.prepare(st)
        else:
            from .graph import layer_info
            info, warns = layer_info(st), []
            info = {k: v for k, v in info.items() if not k.startswith("_")}
        gap, unc = self.predict_gap(st)
        if gap is None:
            return None
        g = to_graph(st, self.cutoff)
        ld = self.latent_distance([g])
        ld = None if ld is None else float(ld[0])
        p_metal = self.predict_metal(st)
        gap_type, p_ind = self.predict_type(st)
        v = verdict(unc, self.cal, ld)
        if p_metal is not None and p_metal >= self.metal_thr:
            v = "out-of-domain (metal gate: p_metal=%.2f)" % p_metal
            warns.append("metal gate says metal; gap regressor is not trained on metals")
        if not info["is_layer"]:
            v = "out-of-domain (not a 2D layer)"
        metal_like = gap < METAL_GAP
        return Prediction(
            formula=st.composition.reduced_formula, natoms=len(st),
            gap=gap, unc=unc, verdict=v,
            exp_gap_est=None if metal_like else correct(gap, "optical", self.corr),
            gap_quasiparticle=None if metal_like else correct(gap, "quasiparticle", self.corr),
            is_metal_like=metal_like, latent_distance=ld,
            interval90=self.cal["scale90"] * unc,
            typical_error=self.cal["tier_mae"].get(tier_key(v)),
            gap_type=gap_type, p_indirect=p_ind,
            p_metal=p_metal, layer=info, warnings=warns,
        )


    # --- batched inference (large screens) --------------------------------------
    @torch.no_grad()
    def _forward_many(self, graphs: list, batch_size: int, device: str, mc: int):
        """Run every head over a list of graphs. Returns (gap, unc, p_type, p_metal)
        as numpy arrays aligned with `graphs`."""
        gaps, uncs, p_type, p_metal = [], [], [], []
        for i in range(0, len(graphs), batch_size):
            chunk = graphs[i:i + batch_size]
            batch = Batch.from_data_list(chunk).to(device)
            if len(self.models) > 1:
                preds = torch.stack([m(batch) * self.std + self.mean for m in self.models])
                gaps.append(preds.mean(0).cpu().numpy())
                # unbiased=False to match np.std used by the single-structure path and
                # by the calibration in train_cgcnn.py; with 5 members the n-1
                # convention would inflate every uncertainty by 12% and shift verdicts
                uncs.append(preds.std(0, unbiased=False).cpu().numpy())
            else:
                model = self.models[0]
                gaps.append((model(batch) * self.std + self.mean).cpu().numpy())
                model.train()
                for mod in model.modules():
                    if isinstance(mod, nn.BatchNorm1d):
                        mod.eval()
                mc_preds = torch.stack([model(batch) * self.std + self.mean for _ in range(mc)])
                uncs.append(mc_preds.std(0, unbiased=False).cpu().numpy())
                model.eval()
            for model, sink in ((self.type_model, p_type), (self.metal_model, p_metal)):
                if model is None:
                    sink.append(np.full(len(chunk), np.nan))
                else:
                    sink.append(torch.sigmoid(model(batch)[1]).cpu().numpy())
        cat = lambda xs: np.concatenate(xs) if xs else np.array([])
        return cat(gaps), cat(uncs), cat(p_type), cat(p_metal)

    def run_many(self, items, batch_size: int = 256, pad_vacuum: bool = True,
                 device: str = "cpu", mc: int = 30, progress_every: int = 0):
        """Batched equivalent of `run()` for large screens.

        `items` is an iterable of (key, Structure). Returns a list of
        (key, Prediction | None), preserving order; None means no graph could be
        built.

        Measured ~2x faster than calling `run()` per structure on CPU (400 C2DB
        structures: 1.5 s vs 3.1 s), because graph construction, not the forward
        pass, dominates for models this small. The larger win is `device="cuda"`
        and not paying model-load overhead per call. Numerically identical to
        `run()` - guarded by tests/test_smoke.py.
        """
        items = list(items)
        keys, graphs, metas, order = [], [], [], []
        for n, (key, st) in enumerate(items):
            if pad_vacuum:
                st2, info, warns = self.prepare(st)
            else:
                from .graph import layer_info
                info = {k: v for k, v in layer_info(st).items() if not k.startswith("_")}
                st2, warns = st, []
            g = to_graph(st2, self.cutoff)
            if g is None:
                continue
            keys.append(key)
            graphs.append(g)
            metas.append((st2, info, warns))
            order.append(n)
            if progress_every and len(graphs) % progress_every == 0:
                print(f"  graphs {len(graphs)}/{len(items)}", flush=True)

        for m in self.models:
            m.to(device)
        for m in (self.type_model, self.metal_model):
            if m is not None:
                m.to(device)
        gaps, uncs, p_type, p_metal = self._forward_many(graphs, batch_size, device, mc)
        lat = None
        if self.ref_emb is not None and graphs:
            lat = np.concatenate([self.latent_distance(graphs[i:i + batch_size])
                                  for i in range(0, len(graphs), batch_size)])

        out: list = [(k, None) for k, _ in items]
        for j, key in enumerate(keys):
            st2, info, warns = metas[j]
            gap, unc = float(gaps[j]), float(uncs[j])
            pm = None if np.isnan(p_metal[j]) else float(p_metal[j])
            pt = None if np.isnan(p_type[j]) else float(p_type[j])
            ld = None if lat is None else float(lat[j])
            v = verdict(unc, self.cal, ld)
            w = list(warns)
            if pm is not None and pm >= self.metal_thr:
                v = "out-of-domain (metal gate: p_metal=%.2f)" % pm
                w.append("metal gate says metal; gap regressor is not trained on metals")
            if not info["is_layer"]:
                v = "out-of-domain (not a 2D layer)"
            metal_like = gap < METAL_GAP
            out[order[j]] = (key, Prediction(
                formula=st2.composition.reduced_formula, natoms=len(st2),
                gap=gap, unc=unc, verdict=v,
                exp_gap_est=None if metal_like else correct(gap, "optical", self.corr),
                gap_quasiparticle=None if metal_like else correct(gap, "quasiparticle", self.corr),
                is_metal_like=metal_like, latent_distance=ld,
                interval90=self.cal["scale90"] * unc,
                typical_error=self.cal["tier_mae"].get(tier_key(v)),
                gap_type=None if pt is None else ("indirect" if pt >= self.type_thr else "direct"),
                p_indirect=pt, p_metal=pm, layer=info, warnings=w))
        return out


def read_structure(path: str) -> Structure:
    """CIF / POSCAR / .vasp / extensionless POSCAR."""
    try:
        return Structure.from_file(path)
    except Exception:
        return Structure.from_file(path, fmt="poscar")
