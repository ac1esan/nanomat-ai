"""The same prediction, in words a non-specialist can act on.

A band gap in eV at the PBE level means little outside the field, and in the
language-model test it was misread in a specific way: models turned energies into
the wrong colours (1.93 eV called green, 1.7-2.4 eV placed at 700-1200 nm). So the
conversion is done here, once, from the absorption-onset estimate rather than the PBE
number, and every front end - the command line, the notebook, the tools a language
model gets - says the same thing. docs/index.html carries the same bands in
JavaScript; keep the two in step.
"""

from __future__ import annotations

from .predict import METAL_GAP

HC_EV_NM = 1239.84          # photon energy (eV) x wavelength (nm)

# (upper edge of the band in nm, name). Ordered by wavelength.
LIGHT = [(280, "deep ultraviolet"), (400, "ultraviolet"), (450, "violet"), (495, "blue"),
         (570, "green"), (590, "yellow"), (620, "orange"), (750, "red"),
         (1400, "near infrared"), (float("inf"), "infrared")]

# (upper edge of the optical gap in eV, what a gap like this is used for)
USES = [(0.5, "a very narrow gap - close to a metal, it conducts readily when warm"),
        (1.1, "a narrow, infrared gap, like germanium's (0.7 eV): infrared detectors"),
        (1.8, "in the range solar cells use, like silicon (1.1 eV) or GaAs (1.4 eV)"),
        (3.1, "a gap that absorbs visible light: LEDs, lasers, visible-light detectors"),
        (4.5, "a wide gap in the ultraviolet, like GaN (3.4 eV): UV emitters and detectors"),
        (float("inf"), "an insulator: the gap of dielectric and encapsulation layers, "
                       "of which hexagonal boron nitride is the classic one")]

VERDICT = {
    "reliable": "The model is confident here; predictions like this are typically off by "
                "about {err:.2f} eV.",
    "check": "Worth confirming with a DFT calculation before relying on it; predictions "
             "like this are typically off by about {err:.2f} eV.",
    "out-of-domain": "Do not use the number. {reason}",
}


# cosine distance in the ensemble's latent space below which the nearest training
# structure is practically the same layer: a training structure finds itself at
# ~1e-4 (float16 rounding), phosphorene relaxed in JARVIS finds its Alexandria twin
# at ~0.005. docs/index.html uses the same threshold.
SAME_LAYER = 0.005


def neighbour_detail(n: dict) -> str:
    """'DFT gap 1.76 eV', with the label's flag when it has one."""
    text = "from MC2D" if n.get("dft_gap") is None else f"DFT gap {n['dft_gap']:.2f} eV"
    if n.get("flag") == "wrong":
        text += ", a label known to be wrong"
    elif n.get("flag") == "disputed":
        text += ", a disputed label"
    return text


def nearest_words(neighbours: list[dict] | None) -> str:
    """One sentence on the closest training structure, or "" without one."""
    if not neighbours:
        return ""
    n = neighbours[0]
    if n["distance"] < SAME_LAYER:
        return (f"The model was trained on practically this same layer, {n['formula']} "
                f"({neighbour_detail(n)}), so the number largely recalls that result rather "
                "than predicting a new one.")
    return f"The closest material it was trained on is {n['formula']} ({neighbour_detail(n)})."


def wavelength_nm(gap_ev: float) -> float | None:
    return HC_EV_NM / gap_ev if gap_ev and gap_ev > 0 else None


def light_name(nm: float) -> str:
    return next(name for edge, name in LIGHT if nm < edge)


def use_of(gap_ev: float) -> str:
    return next(text for edge, text in USES if gap_ev < edge)


def describe(pred, typical_error: dict | None = None) -> str:
    """A few sentences for `pred`, a nanomat.predict.Prediction.

    The typical error quoted is the prediction's own (its tier's, measured on held-out
    data); `typical_error` maps tier to eV only as a fallback for a prediction without
    one.
    """
    err = {"reliable": 0.10, "check": 0.17, **(typical_error or {})}
    v = pred.verdict
    if v.startswith("out-of-domain"):
        reason = v[v.find("(") + 1:v.rfind(")")] if "(" in v else ""
        why = {"metal gate": "The structure looks metallic, and the gap model was only "
                             "ever trained on semiconductors.",
               "chemistry unlike anything in training": "Its chemistry is unlike anything "
                                                         "the model was trained on."}
        text = next((w for k, w in why.items() if reason.startswith(k)),
                    "The five models disagree too much to trust any of them; the structure "
                    "may be metallic or unusual.")
        out = VERDICT["out-of-domain"].format(reason=text)
        # what it resembles explains "unlike anything"; next to a metal or a non-layer
        # verdict it would only distract
        if reason.startswith("chemistry"):
            out = " ".join(x for x in (out, nearest_words(getattr(pred, "neighbours", None))) if x)
        return out
    tier = "reliable" if v.startswith("reliable") else "check"
    parts = []
    onset = pred.exp_gap_est
    if pred.gap is not None and pred.gap < METAL_GAP:
        parts.append("The model sees this as a metal or very nearly one.")
    elif onset:
        nm = wavelength_nm(onset)
        parts.append(f"It should start absorbing light at roughly {nm:.0f} nm "
                     f"({light_name(nm)}); light of longer wavelength passes through.")
        parts.append(f"That is {use_of(onset)}.")
    typical = getattr(pred, "typical_error", None) or err[tier]
    parts.append(VERDICT[tier].format(err=typical))
    parts.append(nearest_words(getattr(pred, "neighbours", None)))
    return " ".join(p for p in parts if p)
