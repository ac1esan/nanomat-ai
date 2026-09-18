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
#     HSE06 is itself below G0W0 by roughly 0.4 eV at a 1.7 eV gap, so read this as an
#     HSE-level estimate rather than as the last word on a quasiparticle gap.
#   optical: the absorption onset. Fitted against C2DB's G0W0 gap minus its BSE exciton
#     binding energy on 184 non-magnetic 2D materials; 10-fold MAE 0.38 eV.
# Their DIFFERENCE IS NOT THE EXCITON BINDING ENERGY. They are fitted against different
# references - HSE06 on one side, G0W0 minus an exciton on the other - so the gap
# between them mixes the exciton with the HSE-to-GW discrepancy. An earlier version of
# this file claimed otherwise; that claim came from fitting both on the same four TMDs.
DEFAULT_CORRECTIONS = {
    "quasiparticle": {"coeffs": [1.179, 0.450], "a": 1.179, "b": 0.450,
                      "n": 32, "mae_loo": 0.193, "reference": "HSE06"},
    "optical": {"coeffs": [0.0745, 0.7234, 0.6757],
                "n": 184, "mae_cv": 0.383, "reference": "G0W0 - BSE exciton"},
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
    "workfunction": "cgcnn_2d_workfunction.pt",
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
    return correct(gap, "optical", corrections)


def band_edges(work_function: float, gap: float) -> tuple[float, float]:
    """(electron affinity, ionisation potential) relative to vacuum, in eV.

    The trained target is what C2DB calls the work function: the vacuum level minus
    the Fermi level. For a metal that is the work function proper. For an undoped
    semiconductor DFT puts the Fermi level mid-gap, so the number on its own is a
    reference level rather than anything a probe measures — but together with the
    gap it places both band edges, which is the pair a contact is chosen on.

    Checked against published monolayer values: ionisation potentials land within
    0.15 eV for MoS2, MoSe2, WS2 and WSe2, affinities within 0.4 eV. The mid-gap
    assumption is a convention, not a law, so treat these as placements rather than
    measurements.
    """
    return work_function - gap / 2, work_function + gap / 2


def correct(gap: float, kind: str, corrections: dict | None = None) -> float:
    """Apply one of the two corrections. `kind` is "quasiparticle" or "optical".

    Stored as polynomial coefficients, highest power first, because the optical fit
    needed a quadratic: over 184 materials a straight line is biased high in the
    middle of the range and low at the top, and the curve beats it on every
    cross-validation seed. `a`/`b` are still read for older checkpoints.
    """
    c = (corrections or DEFAULT_CORRECTIONS)[kind]
    coeffs = c.get("coeffs") or [c["a"], c["b"]]
    out = 0.0
    for k in coeffs:
        out = out * gap + k
    return out


def verdict(unc: float, cal: dict | None = None, latent: float | None = None,
            high_unc_reason: str = "possibly metal / unusual structure") -> str:
    """Trust tier from two independent signals.

    `unc` is the ensemble spread: it ranks errors well, but every member shares one
    training set, so a chemistry none of them saw yields confident agreement
    (phosphorene is the worked example - spread 0.035 eV and an error of 1.2 eV).
    `latent` is the distance to the nearest training structures in the model's own
    embedding space, which catches exactly that case. Thresholds are quartiles
    measured on a held-out split by scripts/calibrate_uncertainty.py.

    `high_unc_reason` is the wording for the top tier, because the reason differs per
    property: a metal has no gap to predict, but it does have a work function, and
    C2DB's metals are in that model's training set.
    """
    cal = cal or DEFAULT_CAL
    if unc <= cal["unc_median"]:
        tier = "reliable"
    elif unc <= cal["unc_q75"]:
        tier = "check (elevated uncertainty)"
    else:
        tier = f"out-of-domain ({high_unc_reason})"
    if latent is not None:
        if latent > cal.get("latent_q90", float("inf")):
            return "out-of-domain (chemistry unlike anything in training)"
        if latent > cal.get("latent_q75", float("inf")) and tier == "reliable":
            return "check (chemistry sparsely covered by training data)"
    return tier


def tier_key(v: str) -> str:
    return "reliable" if v.startswith("reliable") else (
        "check" if v.startswith("check") else "out_of_domain")


class PropertyEnsemble:
    """A second regression head on the same graphs, loaded from its own checkpoint.

    The band gap is not the only thing a 2D material is chosen for. The work
    function decides which metal makes an ohmic contact and where the Schottky
    barrier sits, and it is a separate model with its own normalisation and its own
    calibration — sharing a trunk would tie their accuracies together for no reason.
    It carries its own trust layer for the same reason the gap has one: the spread
    and the latent distance are measured against ITS training set, which is a
    different one (C2DB, 2823 structures). A structure can be ordinary for the gap
    model and unseen for this one. Graphene is the worked example - 3.18 eV against
    4.25 in C2DB, spread 0.32 against 0.03 on the TMDs.

    Optional: absent weights simply mean the property is not reported.
    """

    def __init__(self, path: str, log=print, high_unc_reason: str = "unusual structure"):
        ck = torch.load(path, map_location="cpu")
        self.cutoff = float(ck.get("cutoff", DEFAULT_CUTOFF))
        self.n_rbf = int(ck.get("n_rbf", DEFAULT_N_RBF))
        self.mean, self.std = float(ck["mean"]), float(ck["std"])
        self.cal = dict(ck.get("calibration", {}))
        self.high_unc_reason = high_unc_reason
        self.ref_emb: torch.Tensor | None = None
        if ck.get("reference_embeddings") is not None:
            self.ref_emb = ck["reference_embeddings"].float()
        # angular width comes from this checkpoint, but the graphs are built by the
        # Predictor for the gap model. Training this one with angles therefore
        # requires the gap model to carry them too, or its graphs arrive without.
        self.n_ang = int(ck.get("n_ang", 0))
        states = ck.get("state_dicts") or [ck["state_dict"]]
        self.models = []
        for sd in states:
            m = CGCNN(cutoff=self.cutoff, n_rbf=self.n_rbf, ang_dim=self.n_ang)
            m.load_state_dict(sd)
            m.eval()
            self.models.append(m)
        log(f"Loaded {os.path.basename(path)}: {len(self.models)} models"
            + (f", test MAE {self.cal['test_mae']:.3f}" if "test_mae" in self.cal else ""))

    @torch.no_grad()
    def predict(self, graphs: list) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """(values, spreads) for a list of graphs; spread is 0 for a single model."""
        if not graphs:
            return None, None
        batch = Batch.from_data_list(graphs)
        preds = torch.stack([m(batch) * self.std + self.mean for m in self.models])
        spread = (preds.std(0, unbiased=False) if len(self.models) > 1
                  else torch.zeros(preds.shape[1]))
        return preds.mean(0).cpu().numpy(), spread.cpu().numpy()

    @torch.no_grad()
    def latent_distance(self, graphs: list, k: int = 10) -> np.ndarray | None:
        """Distance to this model's own training set, in this model's own space."""
        if self.ref_emb is None or not graphs:
            return None
        e = self.models[0].encode(Batch.from_data_list(graphs))
        e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-9)
        sims = e @ self.ref_emb.T
        k = min(k, sims.shape[1])
        return (1 - sims.topk(k, dim=1).values.mean(1)).cpu().numpy()

    def judge(self, unc: float, latent: float | None) -> str | None:
        """This property's own verdict, from its own calibrated thresholds.

        None when the checkpoint carries no calibration - falling back to the gap's
        thresholds would put one model's quartiles on another model's spread.
        """
        if "unc_median" not in self.cal:
            return None
        return verdict(unc, self.cal, latent, self.high_unc_reason)

    def interval90(self, unc: float) -> float | None:
        s = self.cal.get("scale90")
        return None if s is None else float(s) * unc


def _apply_heads(heads: dict | None, emb, gap: float, corr: dict,
                 metal_like: bool, gap_verdict: str) -> dict:
    """The three many-body numbers, from the latent heads when they are present.

    optical = direct G0W0 gap - exciton binding energy, by construction, so the
    three numbers a user reads cannot contradict each other. The polynomial
    corrections remain the fallback: they are a function of the gap alone, which is
    why they cannot see the exciton and why these heads exist.

    Nothing is returned for a structure the tool has disowned. The heads were fitted
    only on in-domain materials and they extrapolate badly off it - graphene comes
    out with a 3.4 eV binding energy and a negative optical gap - so a disowned
    prediction must not reappear here in three new forms.
    """
    if metal_like or gap_verdict.startswith("out-of-domain"):
        return {}
    if heads is None or emb is None:
        return {"gap_quasiparticle": correct(gap, "quasiparticle", corr),
                "exp_gap_est": correct(gap, "optical", corr)}
    x = np.append(np.asarray(emb, dtype=np.float64), gap)
    val = lambda k: float(np.dot(heads[k]["w"].numpy().astype(np.float64), x) + heads[k]["b"])
    eb = val("exciton_binding")
    direct = val("gap_dir_gw")
    return {"gap_quasiparticle": val("gap_gw"), "gap_quasiparticle_direct": direct,
            "exciton_binding": eb, "exp_gap_est": direct - eb}


def _wf_fields(wf: "PropertyEnsemble | None", value, spread, latent,
               gap: float, gap_verdict: str, metal_like: bool) -> dict:
    """Prediction fields for the second property, plus the band edges it unlocks.

    The edges are a subtraction between two models, so they are only shown when
    neither model has disowned its half: a gap the tool has just called
    out-of-domain must not reappear as two band positions, and neither must a work
    function from a chemistry the second model has never seen.
    """
    if value is None:
        return {}
    v, u = float(value), float(spread)
    ld = None if latent is None else float(latent)
    wf_verdict = wf.judge(u, ld) if wf is not None else None
    out = {"work_function": v, "work_function_unc": u, "work_function_latent": ld,
           "work_function_verdict": wf_verdict,
           "work_function_interval90": wf.interval90(u) if wf is not None else None}
    both_hold = (not metal_like
                 and not gap_verdict.startswith("out-of-domain")
                 and not (wf_verdict or "").startswith("out-of-domain"))
    if both_hold:
        out["electron_affinity"], out["ionisation_potential"] = band_edges(v, gap)
    return out


@dataclass
class Prediction:
    formula: str
    natoms: int
    gap: float                 # eV, PBE level
    unc: float                 # eV, ensemble std (or MC-dropout std)
    verdict: str
    # None whenever the tool will not stand behind a corrected value: a metal-like
    # gap, or a verdict of out-of-domain
    exp_gap_est: float | None = None   # eV, the absorption onset
    is_metal_like: bool = False        # gap < METAL_GAP
    work_function: float | None = None      # eV, only when the second model is present
    work_function_unc: float | None = None
    # the second model's own trust signals, against its own training set
    work_function_verdict: str | None = None
    work_function_latent: float | None = None
    work_function_interval90: float | None = None
    # Band edges relative to the vacuum level, the pair a contact is actually chosen
    # on. Derived from the two models together rather than predicted directly, so
    # they are withheld unless BOTH models stand behind their half.
    electron_affinity: float | None = None   # vacuum -> conduction band minimum
    ionisation_potential: float | None = None  # vacuum -> valence band maximum
    gap_quasiparticle: float | None = None  # eV, fundamental quasiparticle gap (G0W0)
    gap_quasiparticle_direct: float | None = None  # eV, the vertical one absorption sees
    exciton_binding: float | None = None    # eV, BSE-level; optical = direct - this
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
            "work_function_eV": None if self.work_function is None else round(self.work_function, 3),
            "electron_affinity_eV": None if self.electron_affinity is None else round(self.electron_affinity, 3),
            "ionisation_potential_eV": None if self.ionisation_potential is None else round(self.ionisation_potential, 3),
            "work_function_unc_eV": None if self.work_function_unc is None else round(self.work_function_unc, 3),
            "work_function_verdict": self.work_function_verdict,
            "gap_quasiparticle_eV": None if self.gap_quasiparticle is None else round(self.gap_quasiparticle, 3),
            "gap_quasiparticle_direct_eV": None if self.gap_quasiparticle_direct is None else round(self.gap_quasiparticle_direct, 3),
            "exciton_binding_eV": None if self.exciton_binding is None else round(self.exciton_binding, 3),
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
        self.n_ang = 0          # angular descriptor width, read from the checkpoint
        self.cal = dict(DEFAULT_CAL)
        self.corr = {k: dict(v) for k, v in DEFAULT_CORRECTIONS.items()}
        self.ref_emb: torch.Tensor | None = None  # normalised training embeddings
        # linear heads on the concatenated member embeddings: G0W0 gaps and the
        # exciton binding energy (scripts/fit_exciton.py). Absent = fall back to the
        # polynomial corrections above.
        self.heads: dict | None = None
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
        wf_path = self._path("workfunction")
        self.wf = PropertyEnsemble(wf_path, self._log) if os.path.exists(wf_path) else None

    # --- loading -------------------------------------------------------------
    def _path(self, key: str) -> str:
        return os.path.join(self.weights_dir, WEIGHT_FILES[key])

    def _load_gap_models(self) -> None:
        ens, one = self._path("ensemble"), self._path("single")
        if os.path.exists(ens):
            ck = torch.load(ens, map_location="cpu")
            self.cutoff, self.n_rbf = float(ck.get("cutoff", self.cutoff)), int(ck.get("n_rbf", self.n_rbf))
            self.n_ang = int(ck.get("n_ang", 0))
            for sd in ck["state_dicts"]:
                m = CGCNN(cutoff=self.cutoff, n_rbf=self.n_rbf, ang_dim=self.n_ang)
                m.load_state_dict(sd)
                m.eval()
                self.models.append(m)
            self.mean, self.std = float(ck["mean"]), float(ck["std"])
            if ck.get("reference_embeddings") is not None:
                self.ref_emb = ck["reference_embeddings"].float()
            oh = ck.get("optical_heads")
            if isinstance(oh, dict) and "exciton_binding" in oh:
                self.heads = oh
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
            self.n_ang = int(ck.get("n_ang", 0))
            m = CGCNN(cutoff=self.cutoff, n_rbf=self.n_rbf, ang_dim=self.n_ang)
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
        m = CGCNNcls(cutoff=float(ck.get("cutoff", self.cutoff)),
                     n_rbf=int(ck.get("n_rbf", self.n_rbf)),
                     ang_dim=int(ck.get("n_ang", 0)))
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
        g = to_graph(st, self.cutoff, n_ang=self.n_ang)
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
        g = to_graph(st, self.cutoff, n_ang=self.n_ang)
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
        g = to_graph(st, self.cutoff, n_ang=self.n_ang)
        emb = None
        if self.heads is not None and len(self.models) > 1:
            with torch.no_grad():
                b_ = Batch.from_data_list([g])
                emb = torch.cat([m.encode(b_) for m in self.models], dim=1).cpu().numpy()[0]
        ld = self.latent_distance([g])
        ld = None if ld is None else float(ld[0])
        wf_v, wf_u = (self.wf.predict([g]) if self.wf else (None, None))
        wf_l = self.wf.latent_distance([g]) if self.wf else None
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
            **_apply_heads(self.heads, emb, gap, self.corr, metal_like, v),
            is_metal_like=metal_like, latent_distance=ld,
            **_wf_fields(self.wf, None if wf_v is None else wf_v[0],
                         None if wf_u is None else wf_u[0],
                         None if wf_l is None else wf_l[0], gap, v, metal_like),
            interval90=self.cal["scale90"] * unc,
            typical_error=self.cal["tier_mae"].get(tier_key(v)),
            gap_type=gap_type, p_indirect=p_ind,
            p_metal=p_metal, layer=info, warnings=warns,
        )


    # --- batched inference (large screens) --------------------------------------
    @torch.no_grad()
    def _forward_many(self, graphs: list, batch_size: int, device: str, mc: int):
        """Run every head over a list of graphs. Returns (gap, unc, p_type, p_metal,
        embedding) as numpy arrays aligned with `graphs`.

        The embedding is the members' encoder outputs concatenated. It falls out of
        the same pass that produces the gap - in eval mode dropout is the identity,
        so head(encode(x)) is the forward - which is why the optical heads cost
        nothing on top of a prediction that was happening anyway.
        """
        gaps, uncs, p_type, p_metal, embs = [], [], [], [], []
        for i in range(0, len(graphs), batch_size):
            chunk = graphs[i:i + batch_size]
            batch = Batch.from_data_list(chunk).to(device)
            if len(self.models) > 1:
                hs = [m.encode(batch) for m in self.models]
                preds = torch.stack([m.head(h).squeeze(-1) * self.std + self.mean
                                     for m, h in zip(self.models, hs)])
                embs.append(torch.cat(hs, dim=1).cpu().numpy())
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
        return cat(gaps), cat(uncs), cat(p_type), cat(p_metal), (cat(embs) if embs else None)

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
            g = to_graph(st2, self.cutoff, n_ang=self.n_ang)
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
        gaps, uncs, p_type, p_metal, embs = self._forward_many(graphs, batch_size, device, mc)
        wfv = wfu = wfl = None
        if self.wf is not None and graphs:
            parts = [self.wf.predict(graphs[i:i + batch_size])
                     for i in range(0, len(graphs), batch_size)]
            wfv = np.concatenate([a for a, _ in parts])
            wfu = np.concatenate([b for _, b in parts])
            if self.wf.ref_emb is not None:
                wfl = np.concatenate([self.wf.latent_distance(graphs[i:i + batch_size])
                                      for i in range(0, len(graphs), batch_size)])
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
                **_apply_heads(self.heads, None if embs is None else embs[j],
                               gap, self.corr, metal_like, v),
                is_metal_like=metal_like, latent_distance=ld,
                **_wf_fields(self.wf, None if wfv is None else wfv[j],
                             None if wfu is None else wfu[j],
                             None if wfl is None else wfl[j], gap, v, metal_like),
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
