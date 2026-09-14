"""Smoke / contract tests. Run: pytest -q  (needs weights/ and examples/)."""

import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanomat import CGCNN, Predictor, to_graph  # noqa: E402
from nanomat.predict import MED_UNC, read_structure  # noqa: E402

EX = os.path.join(ROOT, "examples")
WEIGHTS = os.path.join(ROOT, "weights")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(WEIGHTS, "cgcnn_2d_ensemble.pt")),
    reason="weights/cgcnn_2d_ensemble.pt missing",
)


@pytest.fixture(scope="module")
def P():
    return Predictor(WEIGHTS, verbose=False)


def test_checkpoint_matches_model_contract():
    ck = torch.load(os.path.join(WEIGHTS, "cgcnn_2d_ensemble.pt"), map_location="cpu")
    model_keys = set(CGCNN(cutoff=ck["cutoff"], n_rbf=ck["n_rbf"]).state_dict())
    assert set(ck["state_dicts"][0]) == model_keys
    assert len(ck["state_dicts"]) == 5


def test_mos2_reference_prediction(P):
    """Pinned regression values for the shipped ensemble (MoS2 from JARVIS dft_2d)."""
    r = P.run(read_structure(os.path.join(EX, "MoS2.vasp")))
    assert r.formula == "MoS2"
    assert abs(r.gap - 1.685) < 0.02
    assert abs(r.unc - 0.021) < 0.01
    assert r.verdict == "reliable"
    assert r.gap_type == "direct"
    assert 1.7 < r.exp_gap_est < 2.1          # experiment: 1.88 eV
    assert r.interval90 is not None and 0.05 < r.interval90 < 0.25
    assert r.latent_distance is not None and r.latent_distance < P.cal["latent_q75"]


def test_checkpoint_carries_calibration(P):
    """The shipped ensemble must bring its own calibration, not fall back to defaults."""
    for key in ("scale90", "unc_median", "unc_q75", "latent_q75", "latent_q90", "test_mae"):
        assert key in P.cal, key
    assert P.ref_emb is not None and P.ref_emb.shape[0] > 1000
    # +-1 sigma of the raw spread is known to be overconfident; the scale corrects it
    assert 3.0 < P.cal["scale90"] < 8.0


def test_phosphorene_caught_by_latent_distance(P):
    """The failure mode that the ensemble spread alone misses.

    Alexandria contains exactly one elemental-phosphorus 2D structure, so every
    ensemble member learned the same thing from the same single example and they
    agree confidently (spread ~0.04 eV) while being ~1.2 eV wrong. Distance to the
    training set in latent space is what flags it.
    """
    r = P.run(read_structure(os.path.join(EX, "phosphorene.vasp")))
    assert r.unc < P.cal["unc_median"], "premise: the ensemble is confident here"
    assert r.latent_distance > P.cal["latent_q90"], "latent distance must flag it"
    assert r.verdict.startswith("out-of-domain")


def test_ensemble_is_deterministic(P):
    st = read_structure(os.path.join(EX, "WS2.vasp"))
    a, b = P.run(st), P.run(st)
    assert a.gap == b.gap and a.unc == b.unc


def test_vacuum_padding_is_graph_invariant(P):
    """A 7 A vacuum (< 8 A cutoff) creates spurious edges; padding must remove them
    and reproduce the graph of the properly padded cell exactly."""
    orig = read_structure(os.path.join(EX, "MoS2.vasp"))
    thin = read_structure(os.path.join(EX, "MoS2_thin_vacuum.vasp"))
    g_orig = to_graph(P.prepare(orig)[0], P.cutoff)
    g_pad = to_graph(P.prepare(thin)[0], P.cutoff)
    g_raw = to_graph(thin, P.cutoff)
    assert g_raw.num_edges > g_orig.num_edges          # the bug we are guarding against
    assert g_pad.num_edges == g_orig.num_edges
    assert torch.allclose(g_pad.edge_weight.sort().values,
                          g_orig.edge_weight.sort().values, atol=1e-4)
    r_thin, r_orig = P.run(thin), P.run(orig)
    assert abs(r_thin.gap - r_orig.gap) < 1e-3
    assert r_thin.layer["padded"] and any("padded" in w for w in r_thin.warnings)


def test_bulk_cell_is_flagged_out_of_domain(P):
    from pymatgen.core import Lattice, Structure
    bulk = Structure(Lattice.hexagonal(3.19, 6.2), ["Mo", "S", "S"],
                     [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.0], [2 / 3, 1 / 3, 0.5]])
    r = P.run(bulk)
    assert not r.layer["is_layer"]
    assert r.verdict.startswith("out-of-domain")


def test_graphene_has_high_uncertainty(P):
    """Semimetal is outside the training domain; the ensemble spread must flag it."""
    r = P.run(read_structure(os.path.join(EX, "graphene.vasp")))
    assert r.unc > 3 * MED_UNC
    assert r.verdict.startswith("out-of-domain")


def test_batch_cli_writes_csv(tmp_path):
    out = tmp_path / "results.csv"
    cmd = [sys.executable, os.path.join(ROOT, "screen_bandgap.py"), "--in", EX, "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert res.returncode == 0, res.stderr
    import pandas as pd
    df = pd.read_csv(out)
    assert len(df) == 8
    assert {"file", "formula", "band_gap_PBE_eV", "uncertainty_eV", "verdict"} <= set(df.columns)
    assert df["band_gap_PBE_eV"].is_monotonic_increasing


def test_batched_matches_single_structure(P):
    """run_many must be numerically identical to run(), including the uncertainty.

    Regression guard: torch's std defaults to the unbiased (n-1) estimator while
    numpy uses the biased one. With a 5-member ensemble that inflates every
    uncertainty by 12% and silently changes verdicts near a threshold.
    """
    import glob
    files = sorted(glob.glob(os.path.join(EX, "*.vasp")))
    items = [(os.path.basename(f), read_structure(f)) for f in files]
    batched = P.run_many(items)
    for (key, a), (_, st) in zip(batched, items):
        b = P.run(st)
        assert a is not None and b is not None, key
        assert abs(a.gap - b.gap) < 1e-4, key
        assert abs(a.unc - b.unc) < 1e-4, (key, a.unc, b.unc)
        assert a.verdict == b.verdict, key
        assert a.gap_type == b.gap_type, key
