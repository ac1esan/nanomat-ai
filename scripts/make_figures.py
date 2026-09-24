"""Regenerate the README figures from the project's recorded results.

    python scripts/make_figures.py          # writes figures/*.png

Numbers come from the project log (CLAUDE.md): composition baseline (Magpie +
RF/XGBoost, 5-fold CV MAE) vs CGCNN (val/test MAE) on the same data at
increasing N, plus the experimental validation from validate_experiment.py.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "figures")

# validated categorical palette (dataviz skill, light surface): blue, orange
C_COMP, C_GNN = "#2a78d6", "#eb6834"
C_TEXT, C_MUTED, C_GRID, C_OOD = "#0b0b0b", "#52514e", "#e6e5e1", "#9a9892"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_TEXT,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

# All four points use a RANDOM split, so the series is methodologically consistent.
# The composition-disjoint ("grouped") re-measurement of the 13 349 point is marked
# separately: same code, same data, only the split differs.
GROUPED_13K = (13349, 0.2614)
# The shipped ensemble: angles, plus metastable Alexandria and 2DMatPedia in training,
# scored on the same composition-disjoint stable test set as the diamond above.
SHIPPED = (22103, 0.2246)
# (dataset, N, composition CV MAE, CGCNN MAE)
SCALING = [
    ("JARVIS dft_2d", 696, 0.583, 0.585),
    ("C2DB", 1115, 0.445, 0.42),
    ("Alexandria 2D\nehull ≤ 0.1", 13349, 0.360, 0.215),
    ("Alexandria 2D\nehull ≤ 0.2", 26561, 0.419, 0.262),
]


def fig_scaling():
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    n = [r[1] for r in SCALING]
    comp = [r[2] for r in SCALING]
    gnn = [r[3] for r in SCALING]
    ax.plot(n, comp, color=C_COMP, lw=2, marker="o", ms=7, label="Composition only (Magpie + RF/XGBoost)")
    ax.plot(n, gnn, color=C_GNN, lw=2, marker="o", ms=7, label="Structure (CGCNN, PyTorch Geometric)")
    gx, gy = GROUPED_13K
    ax.plot([gx], [gy], marker="D", ms=8, mfc="white", mec=C_GNN, mew=2, ls="none",
            label="First ensemble, composition-disjoint split")
    ax.annotate(f"{gy:.2f}", (gx, gy), textcoords="offset points", xytext=(-32, -3),
                color=C_TEXT, fontsize=9.5)
    ax.annotate("", xy=(gx, gy - 0.004), xytext=(gx, 0.2214),
                arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.8, ls=":"))
    sx, sy = SHIPPED
    ax.plot([sx], [sy], marker="D", ms=8, color=C_GNN, ls="none",
            label="Shipped: + metastable + 2DMatPedia, same test")
    ax.annotate(f"{sy:.3f}", (sx, sy), textcoords="offset points", xytext=(0, -16),
                ha="center", color=C_TEXT, fontsize=9.5)
    for x, yc, yg in zip(n, comp, gnn):
        ax.annotate(f"{yc:.2f}", (x, yc), textcoords="offset points", xytext=(0, 9),
                    ha="center", color=C_TEXT, fontsize=9.5)
        ax.annotate(f"{yg:.2f}", (x, yg), textcoords="offset points", xytext=(0, -16),
                    ha="center", color=C_TEXT, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{v:,}".replace(",", " ") + "\n" + r[0] for v, r in zip(n, SCALING)],
                       fontsize=8.5)
    ax.minorticks_off()
    ax.set_xlim(520, 40000)
    ax.set_ylim(0.15, 0.70)
    ax.set_xlabel("training structures, N (2D semiconductors)", color=C_MUTED)
    ax.set_ylabel("MAE of band gap, eV  (lower is better)", color=C_MUTED)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.set_title("Structure beats composition only once there is enough data",
                 loc="left", color=C_TEXT, fontsize=12.5, pad=26)
    ax.text(0, 1.005, "Solid lines: random split. Diamonds: no composition shared between "
            "train and test, one fixed stable test set -\nhollow for the first ensemble, "
            "filled for the shipped one, which also trains on metastable and 2DMatPedia data",
            transform=ax.transAxes, color=C_MUTED, fontsize=8.5, va="bottom")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mae_vs_n.png"))
    plt.close(fig)


def fig_experiment():
    """Model against measured optical gaps on the reference monolayers.

    Two markers per material: the PBE-level prediction (hollow), below the
    measurement by design, and the optical-gap estimate from the latent heads
    (filled) - direct G0W0 gap minus exciton binding - which is the number the tool
    offers for comparison with absorption. No fitted line: an earlier version drew a
    descriptive fit through these seven points, which read as the shipped correction.
    """
    import numpy as np
    from validate_experiment import evaluate
    rows = evaluate()
    fig, ax = plt.subplots(figsize=(6.8, 6.4), dpi=160)
    ax.set_aspect("equal")
    lim = (-0.3, 6.6)
    ax.plot(lim, lim, color=C_GRID, lw=1.2, ls="--", zorder=1)
    ax.text(4.9, 5.2, "measured = predicted", color=C_MUTED, fontsize=8.5, rotation=45,
            rotation_mode="anchor")
    # label positions (data coords) chosen by hand to keep the TMD cluster legible
    LBL = {"WS2": (2.75, 2.75), "MoS2": (2.75, 2.2), "phosphorene": (2.75, 0.75),
           "MoSe2": (0.05, 3.15), "WSe2": (0.05, 2.55), "h-BN": (5.9, 3.9),
           "graphene": (0.75, 5.35)}
    errs = []
    for r in rows:
        ood = r["verdict"].startswith("out-of-domain")
        c = C_OOD if ood else C_COMP
        ax.errorbar(r["exp"], r["model"], yerr=r["unc"], fmt="o", mfc="white", mec=c,
                    color=c, ms=7, capsize=3, lw=1.2, zorder=3)
        opt = r.get("optical")
        if opt is not None and not ood:
            ax.plot([r["exp"], r["exp"]], [r["model"], opt], color=C_MUTED, lw=0.7,
                    ls=":", zorder=2)
            ax.plot(r["exp"], opt, "o", color=C_GNN, ms=7, zorder=4)
            errs.append(abs(opt - r["exp"]))
            label, at = f"{r['material']}  {r['model']:.2f} → {opt:.2f}", opt
        else:
            label, at = f"{r['material']}\nrejected by the metal gate", r["model"]
        left_side = r["material"] in ("MoSe2", "WSe2")   # label sits left of the point
        ax.annotate(label, (r["exp"], at), xytext=LBL[r["material"]], color=C_TEXT,
                    fontsize=8.5, va="center",
                    arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.7, shrinkA=0,
                                    shrinkB=5, relpos=(1, 0.5) if left_side else (0, 0.5)))
    ax.plot([], [], "o", mfc="white", mec=C_COMP, ms=7, ls="none",
            label="PBE-level prediction ± ensemble spread")
    ax.plot([], [], "o", color=C_GNN, ms=7, ls="none",
            label="optical estimate (G₀W₀ − exciton heads)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("measured optical gap, eV", color=C_MUTED)
    ax.set_ylabel("model, eV", color=C_MUTED)
    ax.grid(color=C_GRID, lw=0.8)
    ax.set_title("Model against experiment on reference monolayers",
                 loc="left", color=C_TEXT, fontsize=12.5, pad=26)
    ax.text(0, 1.005, "The PBE-level number sits below the measurement by design; the "
            f"optical estimate is what compares with\nabsorption (MAE {np.mean(errs):.2f} eV "
            "over the six semiconductors). Grey: graphene, rejected by the metal gate",
            transform=ax.transAxes, color=C_MUTED, fontsize=8.5, va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "experiment_validation.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_scaling()
    print("figures/mae_vs_n.png")
    fig_experiment()
    print("figures/experiment_validation.png")
