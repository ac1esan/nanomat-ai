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
    # n_ang is part of the contract now: a checkpoint trained with angular features
    # carries ang.* and a model built without them cannot load it
    model_keys = set(CGCNN(cutoff=ck["cutoff"], n_rbf=ck["n_rbf"],
                           ang_dim=ck.get("n_ang", 0)).state_dict())
    assert set(ck["state_dicts"][0]) == model_keys
    assert len(ck["state_dicts"]) == 5


def test_mos2_reference_prediction(P):
    """Pinned regression values for the shipped ensemble (MoS2 from JARVIS dft_2d)."""
    r = P.run(read_structure(os.path.join(EX, "MoS2.vasp")))
    assert r.formula == "MoS2"
    assert abs(r.gap - 1.700) < 0.02
    assert abs(r.unc - 0.034) < 0.015
    assert r.verdict == "reliable"
    assert r.gap_type == "direct"
    # experiment says 1.88 eV. The correction was refitted from five measured
    # monolayers onto 184 C2DB materials with G0W0 + BSE, which moved this from 1.90
    # to 2.11: the old value was memorised, since MoS2 was one of the five.
    assert 1.8 < r.exp_gap_est < 2.3
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
    from nanomat.predict import verdict

    r = P.run(read_structure(os.path.join(EX, "phosphorene.vasp")))
    assert r.unc < P.cal["unc_median"], "premise: the ensemble is confident here"
    assert r.latent_distance > P.cal["latent_q75"], "latent distance must notice it"
    # The mechanism, not which side of a threshold it lands on. Phosphorene sits at
    # 0.282 against a q90 of 0.290 in the angular ensemble and sat at 0.325 against
    # 0.314 in the one before - a knife edge both times, so pinning the crossing
    # tests luck. What must hold is that the spread alone would endorse this
    # prediction and the latent distance takes that endorsement away.
    assert verdict(r.unc, P.cal, None).startswith("reliable"), \
        "premise: without the latent check this would be endorsed"
    assert not r.verdict.startswith("reliable"), "the latent check must downgrade it"


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


def test_prototype_classification():
    """The family view is only as good as the prototype tags behind it.

    1H against 1T is the distinction that matters: same composition, different
    coordination, genuinely different gaps. Collapsing them into one MX2 family
    would average two different materials into a single cell.
    """
    from pymatgen.core import Lattice, Structure
    from nanomat.families import classify

    for name, expected in [("MoS2", ("1H-MX2", "Mo", "S")),
                           ("WSe2", ("1H-MX2", "W", "Se")),
                           ("hBN", ("hc-AB", "B", "N")),
                           ("graphene", ("hc-A", "C", "C"))]:
        assert classify(read_structure(os.path.join(EX, f"{name}.vasp"))) == expected, name

    # phosphorene is none of these prototypes and must not be forced into one
    assert classify(read_structure(os.path.join(EX, "phosphorene.vasp")))[0] is None

    # same composition and lattice as 1H, chalcogens staggered instead of eclipsed
    lat = Lattice.hexagonal(3.19, 22.0)
    one_t = Structure(lat, ["Mo", "S", "S"],
                      [[0, 0, 0.5], [1 / 3, 2 / 3, 0.57], [2 / 3, 1 / 3, 0.43]])
    assert classify(one_t)[0] == "1T-MX2"
    one_h = Structure(lat, ["Mo", "S", "S"],
                      [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.57], [2 / 3, 1 / 3, 0.43]])
    assert classify(one_h)[0] == "1H-MX2"

    # a cubic cell is not a 2D hexagonal prototype whatever its composition
    cubic = Structure(Lattice.cubic(5.0), ["Mo", "S", "S"],
                      [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])
    assert classify(cubic)[0] is None


def test_work_function_has_its_own_verdict(P):
    """The second model is trained on a different set, so it judges independently.

    Graphene is the worked case: the gap model rejects it through the metal gate,
    but even on its own terms the work-function model has seen one carbon-only
    structure and disagrees with itself about it. Band edges must be withheld
    whenever either model disowns its half, because the edges are a subtraction
    between the two.
    """
    if P.wf is None:
        pytest.skip("work-function weights not present")
    for name in ("MoS2", "WS2"):
        r = P.run(read_structure(os.path.join(EX, f"{name}.vasp")))
        assert r.work_function_verdict.startswith("reliable"), name
        assert r.electron_affinity is not None, name
        # the edges are the work function plus/minus half the gap, nothing else
        assert abs(r.ionisation_potential - r.electron_affinity - r.gap) < 1e-6
        assert abs((r.ionisation_potential + r.electron_affinity) / 2 - r.work_function) < 1e-6

    for name in ("graphene", "phosphorene"):
        r = P.run(read_structure(os.path.join(EX, f"{name}.vasp")))
        assert r.work_function is not None, name
        assert r.work_function_verdict.startswith("out-of-domain"), name
        assert r.electron_affinity is None and r.ionisation_potential is None, name


def test_work_function_thresholds_are_its_own(P):
    """Its quartiles must come from its checkpoint, never from the gap model's.

    Applying one model's uncertainty quartiles to another model's spread would put
    every structure in the wrong tier while looking entirely plausible.
    """
    if P.wf is None:
        pytest.skip("work-function weights not present")
    assert P.wf.cal["unc_median"] != P.cal["unc_median"]
    assert P.wf.ref_emb is not None, "latent distance needs the reference embeddings"
    assert P.wf.ref_emb.shape[0] != (0 if P.ref_emb is None else P.ref_emb.shape[0]), \
        "the two models were trained on different sets; the embedding counts differ"


def test_checkpoint_opens_under_the_default_torch_load(P):
    """torch.load defaults to weights_only=True since 2.6.

    A numpy array anywhere in the checkpoint makes it refuse to open the file at
    all, which breaks the shipped weights for everyone rather than just for the
    script that put it there. This caught exactly that while adding the latent
    heads, so it guards every .pt we ship.
    """
    import glob
    for path in sorted(glob.glob(os.path.join(ROOT, "weights", "*.pt"))):
        torch.load(path, map_location="cpu")   # weights_only defaults to True


def test_latent_heads_are_self_consistent(P):
    """optical = direct quasiparticle gap - exciton binding energy, by construction.

    Three independently fitted numbers used to imply a binding energy that was not
    the real one; the identity is what retires that, so it is worth pinning. The
    binding energy of a TMD monolayer is a few tenths of an eV in the literature and
    C2DB's BSE puts MoS2 at 0.55.
    """
    if P.heads is None:
        pytest.skip("latent heads not in the checkpoint")
    r = P.run(read_structure(os.path.join(EX, "MoS2.vasp")))
    assert abs(r.gap_quasiparticle_direct - r.exciton_binding - r.exp_gap_est) < 1e-6
    assert 0.3 < r.exciton_binding < 0.9, r.exciton_binding
    assert r.exp_gap_est > r.gap, "the absorption onset sits above the PBE gap"

    # a disowned prediction must not come back as three new numbers
    bad = P.run(read_structure(os.path.join(EX, "graphene.vasp")))
    assert bad.verdict.startswith("out-of-domain")
    assert bad.exciton_binding is None and bad.exp_gap_est is None
    assert bad.gap_quasiparticle is None


def test_optical_correction_never_goes_backwards(P):
    """The corrected optical gap must stay above the raw PBE gap it comes from.

    PBE underestimates every measurement, so an optical estimate BELOW the raw PBE
    number is wrong by construction. The five-point fit this replaces had a negative
    intercept and broke that for a third of the screening table, and returned zero or
    less below 0.32 eV. Guarding the property, not the coefficients, so a future
    refit cannot quietly reintroduce it.
    """
    from nanomat.predict import correct
    for gap in (0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0):
        optical = correct(gap, "optical", P.corr)
        assert optical > gap, f"optical {optical:.3f} below the raw gap {gap}"
        quasi = correct(gap, "quasiparticle", P.corr)
        assert quasi > gap, f"quasiparticle {quasi:.3f} below the raw gap {gap}"


def test_corrections_carry_their_own_provenance(P):
    """Each correction must say what it was fitted against and how well it held up.

    Two corrections fitted against different references cannot be compared, and the
    project has already been burnt once by treating their difference as the exciton
    binding energy. The fields are what stops that being invisible.
    """
    qp, opt = P.corr["quasiparticle"], P.corr["optical"]
    assert qp["reference"] != opt["reference"], "different targets, or the pair is pointless"
    assert qp["n"] >= 32 and opt["n"] >= 100
    assert 0 < opt["mae_cv"] < 0.5, "a cross-validated error, not an in-sample one"
    assert len(opt["coeffs"]) >= 2


def test_angle_features_separate_the_polymorphs():
    """1H and 1T-MX2 differ by an angle, and the edge feature is a distance.

    The shipped ensemble compresses their measured gap difference by a factor of
    four (0.44 eV in Alexandria arrives as 0.12 eV) because a trigonal prism and an
    octahedron have nearly the same bond lengths. The descriptor exists to carry
    what the distance drops, so it has to tell them apart on structures that differ
    in nothing else.
    """
    import numpy as np
    from pymatgen.core import Lattice, Structure

    from nanomat.graph import DEFAULT_N_ANG, angle_features, to_graph

    a, h, vac = 3.19, 1.57, 25.0
    lat = Lattice.from_parameters(a, a, vac, 90, 90, 120)
    # same metal, same bond lengths; the chalcogens are eclipsed in 1H and
    # staggered in 1T, which is purely an angular statement
    one_h = Structure(lat, ["Mo", "S", "S"],
                      [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5 + h / vac], [1 / 3, 2 / 3, 0.5 - h / vac]])
    one_t = Structure(lat, ["Mo", "S", "S"],
                      [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5 + h / vac], [2 / 3, 1 / 3, 0.5 - h / vac]])

    fh, ft = angle_features(one_h), angle_features(one_t)
    assert fh.shape == (3, DEFAULT_N_ANG)
    assert np.abs(fh[0] - ft[0]).sum() > 0.1, "the metal must see a different geometry"
    assert np.allclose(fh.sum(1), 1.0) and np.allclose(ft.sum(1), 1.0)

    # and the distances alone really are near-identical, which is the whole point
    gh, gt = to_graph(one_h), to_graph(one_t)
    assert abs(float(gh.edge_weight.mean()) - float(gt.edge_weight.mean())) < 0.05
    assert to_graph(one_h, n_ang=DEFAULT_N_ANG).ang.shape == (3, DEFAULT_N_ANG)


def test_angles_are_off_by_default_and_do_not_touch_the_contract():
    """ang_dim=0 must leave the state_dict byte-identical to the shipped weights."""
    from nanomat.model import CGCNN
    plain, with_ang = set(CGCNN().state_dict()), set(CGCNN(ang_dim=9).state_dict())
    assert with_ang - plain == {"ang.weight", "ang.bias"}
    # the shipped ensemble is trained WITH angles, so it must carry them and must
    # declare the width that reproduces it
    ck = torch.load(os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.pt"),
                    map_location="cpu")
    shipped = set(ck["state_dicts"][0])
    assert ck["n_ang"] > 0 and shipped == set(CGCNN(ang_dim=ck["n_ang"]).state_dict())
    assert shipped - plain == {"ang.weight", "ang.bias"}
