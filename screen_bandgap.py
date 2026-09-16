"""NanoMatAI screening tool: crystal structure -> band gap of a 2D semiconductor.

Two modes
  # interactive web UI (drag & drop a CIF / POSCAR, or paste it as text)
  python screen_bandgap.py --app

  # batch screening: folder (or one file) -> CSV ranked by band gap
  python screen_bandgap.py --in examples/ --out results.csv

Output per structure: PBE band gap + ensemble uncertainty, estimated experimental
gap (linear PBE correction), direct/indirect type, and a verdict
(reliable / check / out-of-domain). Runs on CPU in well under a second per structure.

Weights are read from ./weights (or $NANOMAT_WEIGHTS). The model definition and
the graph construction live in the `nanomat` package and are shared with
train_cgcnn.py, so they cannot drift apart.
"""

from __future__ import annotations

import argparse
import glob
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

from nanomat import Predictor
from nanomat.predict import METAL_GAP, read_structure

BROWSER_URL = "https://ac1esan.github.io/nanomat-ai/"
REPO_URL = "https://github.com/ac1esan/nanomat-ai"
# Row count of the published browser; printed by scripts/build_site_data.py.
BROWSE_N = 28372

# Validated strain window (scripts/geometry_sensitivity.py): the response is smooth
# and correct in sign under tension, but breaks down below -2% compression.
STRAIN_MIN, STRAIN_MAX = -0.02, 0.06

STRUCTURE_GLOBS = ("*.cif", "*.vasp", "*.poscar", "POSCAR*", "CONTCAR*")


def find_structures(path: str) -> list[str]:
    if os.path.isdir(path):
        files: list[str] = []
        for pat in STRUCTURE_GLOBS:
            files += glob.glob(os.path.join(path, "**", pat), recursive=True)
        return sorted(set(files))
    return [path]


# ---------------------------------------------------------------------------
# Mode 1: batch screening -> CSV
# ---------------------------------------------------------------------------
def run_batch(in_path: str, out_path: str | None, weights_dir: str | None = None,
              pad_vacuum: bool = True):
    import pandas as pd

    files = find_structures(in_path)
    if not files:
        raise SystemExit(f"No structures found in {in_path}")
    P = Predictor(weights_dir)

    rows = []
    for i, f in enumerate(files, 1):
        try:
            st = read_structure(f)
            r = P.run(st, pad_vacuum=pad_vacuum)
            if r is None:
                print(f"[{i}/{len(files)}] SKIP {os.path.basename(f)}: no neighbours within cutoff")
                continue
            row = {"file": os.path.basename(f), **r.as_row()}
            rows.append(row)
            extra = f"  [{r.gap_type}]" if r.gap_type else ""
            flag = "  !" if r.warnings else ""
            print(f"[{i}/{len(files)}] {r.formula:12s} gap={r.gap:.3f} ± {r.unc:.3f} eV{extra}"
                  f"  {r.verdict}{flag}")
        except Exception as e:  # keep going on a bad file
            print(f"[{i}/{len(files)}] SKIP {os.path.basename(f)}: {e}")

    if not rows:
        raise SystemExit("Nothing predicted.")
    df = pd.DataFrame(rows).sort_values("band_gap_PBE_eV").reset_index(drop=True)
    out_path = out_path or "results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDone: {len(df)} structures -> {out_path} (sorted by band gap)")
    return df


# ---------------------------------------------------------------------------
# Mode 2: Gradio UI
# ---------------------------------------------------------------------------
def format_markdown(r, cal: dict | None = None) -> str:
    cal = cal or {}
    ood = r.verdict.startswith("out-of-domain")
    kind = "metal / semimetal" if r.is_metal_like else "semiconductor / insulator"
    headline = f"**{r.gap:.2f} eV**"
    if ood:
        # The interval is conditioned on the ensemble spread, and out-of-domain is
        # exactly where that spread is untrustworthy. Showing a tight interval next
        # to a number we have just disowned would be the worst of both worlds.
        headline += " &nbsp; — **do not use this number**"
    elif r.interval90 is not None:
        headline += f" &nbsp; (90% interval ±{r.interval90:.2f})"
    lines = [
        f"### {r.formula} &nbsp;·&nbsp; {r.natoms} atoms &nbsp;·&nbsp; "
        f"vacuum {r.layer.get('vacuum', float('nan')):.1f} Å",
        "",
        "| | |",
        "|---|---|",
        f"| **Band gap (PBE)** | {headline} &nbsp; ({kind}) |",
    ]
    if r.exp_gap_est is not None and not ood:
        lines.append(f"| Estimated experimental gap | ≈ {r.exp_gap_est:.2f} eV (linear PBE correction) |")
    if r.gap_type is not None:
        conf = r.p_indirect if r.gap_type == "indirect" else 1 - r.p_indirect
        lines.append(f"| Gap type | {r.gap_type} (p = {conf:.2f}) |")
    if r.p_metal is not None:
        lines.append(f"| Metal gate | p(metal) = {r.p_metal:.2f} |")
    lines.append(f"| Ensemble spread | {r.unc:.3f} eV |")
    if r.latent_distance is not None and cal.get("latent_q75"):
        near = "inside" if r.latent_distance <= cal["latent_q75"] else "OUTSIDE"
        lines.append(f"| Distance to training data | {r.latent_distance:.3f} — {near} the "
                     f"familiar region (q75 = {cal['latent_q75']:.3f}) |")
    lines.append(f"| **Verdict** | **{r.verdict}** |")
    if r.typical_error is not None:
        lines.append(f"| Typical error of this tier | {r.typical_error:.2f} eV (measured on "
                     "held-out data) |")
    if ood:
        lines += ["", "> **Out of the model's domain.** The prediction above is reported for "
                  "transparency, not for use: the calibrated interval assumes the ensemble "
                  "spread is meaningful, and on out-of-domain inputs it is not. Phosphorene is "
                  "the documented example — spread 0.03 eV, actual error 1.2 eV."]
    if r.warnings:
        lines += ["", "⚠️ " + "  \n⚠️ ".join(r.warnings)]
    mae = cal.get("test_mae")
    lines += [
        "",
        "<sub>CGCNN ensemble (5 models) on 13 349 stable 2D semiconductors (Alexandria, PBE), "
        f"evaluated on a composition-disjoint split: test MAE {mae:.2f} eV. "
        "Two independent out-of-domain signals are used, because neither alone is enough: "
        "the spread between ensemble members, and the distance to the training set in the "
        "model's own latent space. The second one exists because all members share a training "
        "set, so a chemistry none of them saw produces confident agreement. "
        f"Gaps below {METAL_GAP} eV are reported as metal-like; PBE underestimates real gaps."
        "</sub>" if mae else
        "<sub>CGCNN ensemble on 2D semiconductors (Alexandria, PBE).</sub>",
    ]
    return "\n".join(lines)


def strain_curve(P, st, eps_min: float = STRAIN_MIN, eps_max: float = STRAIN_MAX):
    """Band gap under biaxial in-plane strain, over the validated window only.

    Internal coordinates are frozen (no Poisson relaxation), and the model was
    trained on relaxed ground states only, so this is a qualitative trend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from pymatgen.core import Lattice, Structure

    eps = np.arange(eps_min, eps_max + 1e-9, 0.01)
    gaps, uncs = [], []
    for e in eps:
        f = 1 + float(e)
        lat = Lattice.from_parameters(st.lattice.a * f, st.lattice.b * f, st.lattice.c,
                                      *st.lattice.angles)
        r = P.run(Structure(lat, st.species, st.frac_coords))
        gaps.append(r.gap)
        uncs.append(r.unc)
    gaps, uncs, x = np.array(gaps), np.array(uncs), eps * 100

    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=140)
    ax.fill_between(x, gaps - uncs, gaps + uncs, color="#2a78d6", alpha=0.18, lw=0)
    ax.plot(x, gaps, color="#2a78d6", lw=2, marker="o", ms=4)
    ax.axvline(0, color="#52514e", lw=1, ls=":")
    ax.set_xlabel("biaxial strain, %", fontsize=9)
    ax.set_ylabel("band gap, eV", fontsize=9)
    ax.set_title("Trend under strain (qualitative)", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def build_demo(weights_dir: str | None = None):
    import gradio as gr

    P = Predictor(weights_dir)

    def infer(file, text, want_strain):
        path = file
        if text and text.strip():
            suffix = ".cif" if "_cell_length_a" in text or "loop_" in text else ".vasp"
            tmp = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
            tmp.write(text)
            tmp.close()
            path = tmp.name
        if not path:
            return "Upload a structure file (CIF / POSCAR / .vasp) or paste its contents.", None
        try:
            st = read_structure(path)
        except Exception as e:
            return f"Could not parse the structure: {e}", None
        r = P.run(st)
        if r is None:
            return "Could not build a graph (no neighbours within the cutoff radius).", None
        fig = None
        if want_strain and r.layer.get("is_layer"):
            try:
                fig = strain_curve(P, st)
            except Exception:
                fig = None
        return format_markdown(r, P.cal), fig

    here = os.path.dirname(os.path.abspath(__file__))
    ex_dir = os.path.join(here, "examples")
    examples = [[os.path.join(ex_dir, f), ""] for f in
                ("MoS2.vasp", "WS2.vasp", "WSe2.vasp", "hBN.vasp", "phosphorene.vasp",
                 "MoS2_thin_vacuum.vasp", "graphene.vasp")
                if os.path.exists(os.path.join(ex_dir, f))]

    with gr.Blocks(title="NanoMatAI · 2D band gap") as demo:
        gr.Markdown(
            "# NanoMatAI — band gap of 2D materials from structure\n"
            "Upload a monolayer structure (CIF / POSCAR / .vasp) **or** paste it as text. "
            "A graph neural network predicts the PBE band gap with an uncertainty estimate "
            "in well under a second on CPU, instead of hours of DFT.\n\n"
            f"No structure file to hand? [Browse {BROWSE_N:,} precomputed predictions]"
            f"({BROWSER_URL}) instead — filter by element, gap range and trust verdict, and "
            f"see a periodic-table map of where the model actually works. "
            f"[Source and method]({REPO_URL})."
        )
        with gr.Row():
            with gr.Column(scale=1):
                f_in = gr.File(label="Structure file", type="filepath",
                               file_types=[".cif", ".vasp", ".poscar", ""])
                t_in = gr.Textbox(label="…or paste POSCAR / CIF here", lines=8,
                                  placeholder="MoS2\n1.0\n 3.19 0 0\n ...")
                strain_cb = gr.Checkbox(
                    value=False,
                    label="also show the trend under biaxial strain (-2%…+6%)",
                    info="Adds ~9 extra predictions. Qualitative only: the model was "
                         "trained on relaxed ground states, and the response breaks "
                         "down below -2% compression.")
                btn = gr.Button("Predict band gap", variant="primary")
            with gr.Column(scale=1):
                out = gr.Markdown(label="Prediction")
                plot = gr.Plot(label="Strain response")
        if examples:
            gr.Examples(examples=examples, inputs=[f_in, t_in], label="Try an example",
                        examples_per_page=8)
        btn.click(infer, inputs=[f_in, t_in, strain_cb], outputs=[out, plot])
        f_in.change(infer, inputs=[f_in, t_in, strain_cb], outputs=[out, plot])
    return demo


def run_app(weights_dir: str | None = None, share: bool = False):
    build_demo(weights_dir).launch(share=share)


def main():
    ap = argparse.ArgumentParser(description="Band-gap screening of 2D materials from structure")
    ap.add_argument("--app", action="store_true", help="launch the Gradio web UI")
    ap.add_argument("--in", dest="inp", help="structure file or folder (batch mode)")
    ap.add_argument("--out", help="output CSV (default results.csv)")
    ap.add_argument("--weights", help="directory with .pt weights (default ./weights)")
    ap.add_argument("--no-pad", action="store_true",
                    help="do not auto-pad thin vacuum (debug only)")
    ap.add_argument("--share", action="store_true", help="Gradio public link")
    args = ap.parse_args()

    if args.app:
        run_app(args.weights, share=args.share)
    elif args.inp:
        run_batch(args.inp, args.out, args.weights, pad_vacuum=not args.no_pad)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
