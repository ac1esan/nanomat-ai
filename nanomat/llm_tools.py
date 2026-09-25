"""The tools NanoMatAI offers a language model, as plain functions.

`nanomat/mcp_server.py` registers them with an MCP server; keeping them free of any
protocol lets the tests call them directly. Every answer leads with the trust
information, in words: the verdict, what it means, the typical error of that tier,
and whether the structure was in the training set. The raw gap is still returned
under an out-of-domain verdict, as the browser shows it.

Shaped by a test of ten language models (scripts/llm_probe.py). The first version
made them invent numbers it had hinted at but not returned, and read a 90%
half-width as a full width; each of those is fixed here, and the notes below say
which failure each field exists to prevent.
"""

from __future__ import annotations

import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "screening_table.csv")
MAX_ROWS = 25
# a reference further from the prediction than this many typical errors of its tier
# contradicts the verdict (the 1T'-MoS2 case: "reliable", 0.96 eV, DFT 0.05)
DISAGREE_TIERS = 3.0

_P = None
_T = None


def _predictor():
    global _P
    if _P is None:
        from nanomat import Predictor
        _P = Predictor(os.path.join(ROOT, "weights"), verbose=False)
    return _P


def _table():
    global _T
    if _T is None:
        import pandas as pd
        if not os.path.exists(TABLE):
            raise RuntimeError("screening_table.csv is missing; run scripts/precompute_screening.py")
        _T = pd.read_csv(TABLE)
        _T["tier"] = _T["verdict"].map(_tier)
    return _T


def _tier(verdict: str) -> str:
    v = str(verdict)
    return "reliable" if v.startswith("reliable") else "check" if v.startswith("check") else "out_of_domain"


def _num(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, nd)


def _tier_mae() -> dict:
    cal = _predictor().cal or {}
    return {"reliable": 0.12, "check": 0.21, "out_of_domain": 0.41, **cal.get("tier_mae", {})}


def _meaning(verdict: str) -> str:
    mae = _tier_mae()
    tier = _tier(verdict)
    if tier == "reliable":
        return (f"The model stands behind this number: its members agree and the structure is "
                f"well covered by training data. Typical error in this tier: "
                f"{mae['reliable']:.2f} eV against the PBE reference.")
    if tier == "check":
        return (f"Usable with care: elevated uncertainty or sparse training coverage. Typical "
                f"error in this tier: {mae['check']:.2f} eV. Verify before relying on it.")
    return (f"The model does NOT stand behind this number ({verdict}). It is not a prediction "
            f"and should not be reported as one; typical error in this tier is "
            f"{mae['out_of_domain']:.2f} eV and can be far larger.")


ROLE = {
    "train": "the model was trained on this exact structure, so its number is a memory of the "
             "DFT label rather than a prediction",
    "val": "held out from training (validation split)",
    "test": "held out from training (test split)",
    "unseen": "never used in training",
}

# the model's target is Alexandria's PBE gap, computed without spin-orbit coupling
REFERENCE = {
    "alexandria": "PBE without spin-orbit coupling (Alexandria) - the model's own target",
    "c2db": "PBE with spin-orbit coupling (C2DB); for compounds of heavy elements this sits "
            "0.2-0.3 eV below the model's target on average",
    "jarvis_dft2d": "OptB88vdW (JARVIS dft_2d), a different functional: no systematic offset from "
                    "the model's target (median -0.02 eV) but a wider scatter",
}

# reference labels the 2DMatPedia audit (scripts/audit_2dmatpedia.py) found wrong, with
# C2DB or JARVIS dft_2d as arbiter: Alexandria's k-mesh misses these Dirac points
KNOWN_BAD_REFERENCE = {
    "agm2000000082": "graphene: semimetal, 0.00 eV in C2DB and JARVIS dft_2d",
    "agm2000000375": "silicene: 0.00 eV in C2DB",
    "agm2000044138": "GaAs honeycomb: 0.00 eV in C2DB and 2DMatPedia",
    "agm2000002555": "AlAs honeycomb: 1.24 eV in C2DB and 2DMatPedia",
    "agm2000002852": "BP honeycomb: 0.90 eV in C2DB, 0.87 in 2DMatPedia",
}

# names people use for 2D materials that are not chemical formulas
NAMES = {
    "graphene": "C", "phosphorene": "P", "black phosphorus": "P", "blue phosphorene": "P",
    "silicene": "Si", "germanene": "Ge", "stanene": "Sn", "borophene": "B",
    "arsenene": "As", "antimonene": "Sb", "bismuthene": "Bi", "tellurene": "Te",
    "hbn": "BN", "h-bn": "BN", "hexagonal boron nitride": "BN",
    "g-c3n4": "C3N4", "graphitic carbon nitride": "C3N4",
}
PHASE = re.compile(r"^\s*(?:monolayer\s+|single-layer\s+)?(1T'|1T″|1T\"|1T|2H|1H|3R|2M|1T-d)\s*[-‐ ]\s*",
                   re.IGNORECASE)

LEVELS = ("PBE level: the model predicts the PBE band gap without spin-orbit coupling, which "
          "underestimates measured gaps. Each row also carries optical_gap_estimate_eV (compare "
          "with absorption or photoluminescence) and quasiparticle_gap_eV (compare with "
          "photoemission or transport); both are withheld when the verdict is out-of-domain.")


def _range(gap, half):
    gap, half = _num(gap), _num(half)
    if gap is None or half is None:
        return None
    return [round(gap - half, 3), round(gap + half, 3)]


def _row(r, full: bool = False) -> dict:
    ood = r["tier"] == "out_of_domain"
    out = {
        "id": r["id"], "formula": r["formula"], "source_database": r["source"],
        "prototype": r["family"] if isinstance(r["family"], str) else None,
        "verdict": r["verdict"], "verdict_meaning": _meaning(r["verdict"]),
        "band_gap_pbe_eV": _num(r["pred_gap_eV"]),
        # an explicit range: the bare half-width was read as a full width
        "gap_range90_eV": None if ood else _range(r["pred_gap_eV"], r["interval90_eV"]),
        "interval90_halfwidth_eV": None if ood else _num(r["interval90_eV"]),
        # returned rather than mentioned: a hint without the value was filled in by guesswork
        "optical_gap_estimate_eV": None if ood else _num(r["exp_gap_est_eV"]),
        "quasiparticle_gap_eV": None if ood else _num(r["gap_quasiparticle_eV"]),
        "gap_type": r["gap_type"] if isinstance(r["gap_type"], str) else None,
        "p_metal": _num(r["p_metal"], 2),
        "p_metastable": _num(r.get("p_metastable"), 2),
        "reference_dft_gap_eV": _num(r["dft_gap_eV"]),
        "reference_method": REFERENCE.get(r["source"], r["source"]),
        "training_role": ROLE.get(str(r["in_training_set"]), str(r["in_training_set"])),
    }
    ref, pred = _num(r["dft_gap_eV"]), _num(r["pred_gap_eV"])
    if r["id"] in KNOWN_BAD_REFERENCE:
        out["reference_note"] = ("this reference label is known to be wrong - "
                                 + KNOWN_BAD_REFERENCE[r["id"]])
    elif not ood and ref is not None and pred is not None:
        mae = _tier_mae()[r["tier"]]
        if abs(pred - ref) > DISAGREE_TIERS * mae:
            head = (f"the DFT reference ({ref:.2f} eV) is {abs(pred - ref):.2f} eV from the "
                    f"prediction, more than {DISAGREE_TIERS:.0f} times this tier's typical error of "
                    f"{mae:.2f} eV")
            if r["source"] == "alexandria":   # the model's own target: nothing else to blame
                out["reference_disagreement"] = (head + ". The reference uses the model's own "
                                                 "target method, so the verdict is contradicted "
                                                 "here: trust the reference, not the prediction.")
            else:
                out["reference_disagreement"] = (head + f". The reference method differs "
                                                 f"({REFERENCE[r['source']]}), which explains part of "
                                                 "a gap this size but rarely all of it: the verdict "
                                                 "is unconfirmed here - check before relying on it.")
    if full:
        out.update({
            "uncertainty_eV": _num(r["uncertainty_eV"]),
            "latent_distance": _num(r["latent_distance"]),
            "exciton_binding_eV": None if ood else _num(r["exciton_binding_eV"]),
            "work_function_eV": _num(r["work_function_eV"]),
            "work_function_verdict": r["wf_verdict"] if isinstance(r["wf_verdict"], str) else None,
            "electron_affinity_eV": _num(r["electron_affinity_eV"]),
            "ionisation_potential_eV": _num(r["ionisation_potential_eV"]),
            "note": LEVELS,
        })
    return out


def _parse(structure: str, fmt: str):
    """A POSCAR or CIF from text or a path; forgiving of a missing POSCAR comment line."""
    from pymatgen.core import Structure
    from nanomat.predict import read_structure
    text = structure.strip()
    if "\n" not in text and os.path.exists(text):
        return read_structure(text), None
    tries = [fmt] if fmt else ["poscar", "cif"]
    first = text.splitlines()[0].split() if text else []
    # a POSCAR whose comment line was dropped starts with the scale factor
    if (not fmt or fmt == "poscar") and len(first) == 1 and re.fullmatch(r"[-+.\deE]+", first[0]):
        tries.append("poscar-with-comment")
    last = None
    for f in tries:
        try:
            if f == "poscar-with-comment":
                return Structure.from_str("structure\n" + text, fmt="poscar"), \
                    "the first line was read as the POSCAR scale factor, so a comment line was added"
            return Structure.from_str(text, fmt=f), None
        except Exception as e:  # noqa: BLE001 - reported below
            last = e
    raise ValueError(f"could not parse the structure as {fmt or 'POSCAR or CIF'} ({last}). Pass "
                     "the whole file, including the POSCAR comment line.")


def predict_structure(structure: str, fmt: str = "") -> dict:
    """Predict the band gap of a 2D monolayer from its crystal structure.

    `structure` is the text of a POSCAR or CIF file, or a path to one. Returns the
    PBE-level gap with the model's own verdict on whether to trust it, a calibrated
    90% range, quasiparticle and optical estimates, gap type, the metal-gate
    probability, and the work function with the band edges it implies.
    """
    from nanomat.predict import tier_key
    try:
        st, fixed = _parse(structure, fmt)
    except ValueError as e:
        return {"error": str(e)}
    r = _predictor().run(st)
    if r is None:
        return {"error": "no graph could be built: no two atoms within the neighbour cutoff"}
    ood = tier_key(r.verdict) == "out_of_domain"
    out = {
        "formula": r.formula, "verdict": r.verdict, "verdict_meaning": _meaning(r.verdict),
        "band_gap_pbe_eV": _num(r.gap),
        "gap_range90_eV": None if ood else _range(r.gap, r.interval90),
        "interval90_halfwidth_eV": None if ood else _num(r.interval90),
        "uncertainty_eV": _num(r.unc), "latent_distance": _num(r.latent_distance),
        "p_metal": _num(r.p_metal, 2), "p_metastable": _num(r.p_metastable, 2),
        "gap_type": r.gap_type,
        "quasiparticle_gap_eV": None if ood else _num(r.gap_quasiparticle),
        "optical_gap_estimate_eV": None if ood else _num(r.exp_gap_est),
        "exciton_binding_eV": None if ood else _num(r.exciton_binding),
        "work_function_eV": _num(r.work_function),
        "work_function_verdict": r.work_function_verdict,
        "electron_affinity_eV": _num(r.electron_affinity),
        "ionisation_potential_eV": _num(r.ionisation_potential),
        "warnings": list(r.warnings), "note": LEVELS,
    }
    if fixed:
        out["input_note"] = fixed
    return out


def _formula(name: str):
    """(reduced formula, how the name was read) or (None, error)."""
    from pymatgen.core import Composition
    raw = name.strip()
    phase = None
    m = PHASE.match(raw)
    if m:
        phase, raw = m.group(1), raw[m.end():]
    key = raw.lower().replace("monolayer", "").strip()
    note = None
    if key in NAMES:
        note = f"read {raw!r} as the formula {NAMES[key]}"
        raw = NAMES[key]
    try:
        f = Composition(raw).reduced_formula
    except Exception:  # noqa: BLE001
        return None, (f"not a chemical formula: {name!r}. Give a formula such as 'MoS2' or 'C' "
                      "(graphene); common names are also understood.")
    if phase:
        extra = (f"the table has no phase field: polymorphs are told apart by the prototype tag "
                 f"(1H-MX2, 1T-MX2, hc-AB, hc-A); rows without a tag can be other or distorted "
                 f"phases, so {phase} is not filtered for")
        note = f"{note}; {extra}" if note else extra
    return f, note


def find_structures(formula: str, limit: int = 10) -> dict:
    """Every precomputed structure of one composition, e.g. "MoS2", "C3N4" or "graphene".

    One formula usually has several polymorphs (1H and 1T MoS2 differ in gap and
    even in being a semiconductor), so this returns them all, most trustworthy
    first, each with its verdict, its DFT reference and its role in training.
    """
    f, note = _formula(formula)
    if f is None:
        return {"error": note}
    t = _table()
    hit = t[t["formula"] == f].copy()
    order = {"reliable": 0, "check": 1, "out_of_domain": 2}
    hit["_o"] = hit["tier"].map(order)
    hit = hit.sort_values(["_o", "uncertainty_eV"])
    rows = [_row(r) for _, r in hit.head(max(1, min(limit, MAX_ROWS))).iterrows()]
    out = {"formula": f, "n_found": int(len(hit)), "structures": rows,
           "note": LEVELS if rows else "No precomputed structure; predict_structure takes a file."}
    if note:
        out["input_note"] = note
    return out


def search_materials(gap_min: float | None = None, gap_max: float | None = None,
                     include_elements: list[str] | None = None,
                     exclude_elements: list[str] | None = None,
                     verdicts: list[str] | None = None, gap_type: str | None = None,
                     prototype: str | None = None, source: str | None = None,
                     exclude_training: bool = False, sort_by: str = "uncertainty",
                     limit: int = 10) -> dict:
    """Screen 28 372 precomputed 2D structures by predicted PBE band gap and chemistry.

    verdicts: any of "reliable", "check", "out_of_domain"; default reliable and check.
    gap_type: "direct" or "indirect". prototype: "1H-MX2", "1T-MX2", "hc-AB", "hc-A".
    source: "alexandria", "c2db" or "jarvis_dft2d". exclude_training drops structures
    the model was trained on. sort_by: "uncertainty" (most confident first), "gap",
    or "gap_desc". gap_min and gap_max apply to the PBE gap; every row also carries
    the optical and quasiparticle estimates.
    """
    t = _table()
    keep = set(verdicts or ["reliable", "check"])
    m = t["tier"].isin(keep)
    if gap_min is not None:
        m &= t["pred_gap_eV"] >= gap_min
    if gap_max is not None:
        m &= t["pred_gap_eV"] <= gap_max
    els = t["elements"].fillna("").str.split()
    for el in include_elements or []:
        m &= els.map(lambda e, el=el: el in e)
    for el in exclude_elements or []:
        m &= els.map(lambda e, el=el: el not in e)
    if gap_type:
        m &= t["gap_type"] == gap_type
    if prototype:
        m &= t["family"] == prototype
    if source:
        m &= t["source"] == source
    if exclude_training:
        m &= t["in_training_set"] != "train"
    hit = t[m]
    key = {"gap": ("pred_gap_eV", True), "gap_desc": ("pred_gap_eV", False)}.get(
        sort_by, ("uncertainty_eV", True))
    hit = hit.sort_values(key[0], ascending=key[1])
    rows = [_row(r) for _, r in hit.head(max(1, min(limit, MAX_ROWS))).iterrows()]
    return {"n_matching": int(len(hit)), "returned": len(rows), "rows": rows, "note": LEVELS}


def get_material(material_id: str) -> dict:
    """Everything the table holds for one structure id (from find_structures or search_materials)."""
    t = _table()
    hit = t[t["id"] == material_id]
    if hit.empty:
        return {"error": f"no structure with id {material_id!r}"}
    return _row(hit.iloc[0], full=True)


def model_card() -> dict:
    """What the model predicts, how accurate it is, and where it should not be trusted."""
    cal = _predictor().cal or {}
    return {
        "predicts": "PBE band gap (without spin-orbit coupling) of an isolated 2D monolayer from "
                    "its crystal structure, plus quasiparticle and optical estimates, gap type "
                    "and work function",
        "accuracy": f"MAE {cal.get('test_mae', 0.225):.3f} eV on 1 326 held-out stable monolayers "
                    "whose compositions never appear in training; 0.43 for composition alone",
        "verdict_tiers": {k: f"typical error {v:.2f} eV" for k, v in _tier_mae().items()},
        "interval": "gap_range90_eV is the prediction plus or minus the ensemble spread times a "
                    "calibrated scale, read per cell of (does the structure resemble the metastable "
                    "population?, spread quartile). It covers about 92% of held-out near-hull "
                    "monolayers and 88% of held-out metastable ones; against references computed "
                    "with other methods (C2DB, JARVIS) it covers less",
        "training_data": "22 103 PBE monolayers: Alexandria 2D (stable and metastable) and "
                         "non-magnetic 2DMatPedia",
        "limits": [
            "Semiconductors only: a metal gate rejects metals first; its gap number means nothing.",
            "PBE underestimates measured gaps; use the optical or quasiparticle estimate for experiment.",
            "No spin-orbit coupling (SOC) in the target, and SOC LOWERS a band gap. So for compounds "
            "of heavy elements (W, Bi, Pb, Tl, I, Te, ...) the SOC-inclusive gap is lower than this "
            "number: against C2DB, which includes SOC, the model is 0.27 eV higher on average there "
            "and 0.03 eV higher without heavy elements.",
            "Structures that resemble the metastable population (p_metastable > 0.5) err about twice "
            "as much at the same verdict; their gap_range90_eV is widened for it.",
            "Weakest on light main-group chemistry (C, B, N, H), strongest on transition-metal compounds.",
            "Labels carry database errors: Alexandria misses the Dirac point of honeycombs (graphene "
            "is labelled 1.23 eV).",
            "A row whose DFT reference contradicts the prediction carries reference_disagreement; "
            "trust the reference there, not the verdict.",
            "Out-of-domain results carry no range and no optical estimate on purpose.",
        ],
    }
