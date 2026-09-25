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
    # The ensemble trained with metastable Alexandria and 2DMatPedia monolayers:
    # 1.676 against a PBE literature value near 1.67. Its spread on MoS2 doubled
    # (0.034 -> 0.063) and is still well inside the reliable tier.
    assert abs(r.gap - 1.676) < 0.02
    assert abs(r.unc - 0.063) < 0.015
    assert r.verdict == "reliable"
    assert r.gap_type == "direct"
    # experiment says 1.88 eV; the latent optical head gives 2.03
    assert 1.8 < r.exp_gap_est < 2.3
    assert r.interval90 is not None and 0.1 < r.interval90 < 0.4
    assert r.latent_distance is not None and r.latent_distance < P.cal["latent_q75"]


def test_checkpoint_carries_calibration(P):
    """The shipped ensemble must bring its own calibration, not fall back to defaults."""
    for key in ("scale90", "unc_median", "unc_q75", "latent_q75", "latent_q90", "test_mae"):
        assert key in P.cal, key
    assert P.ref_emb is not None and P.ref_emb.shape[0] > 1000
    # +-1 sigma of the raw spread is known to be overconfident; the scale corrects it
    assert 3.0 < P.cal["scale90"] < 8.0


def test_phosphorene_caught_by_latent_distance(P):
    """A sparse neighbourhood, which the ensemble spread alone does not see.

    Phosphorene is the only elemental-phosphorus layer in training (it is itself a
    training structure, agm2000000335), so every member learns it identically and
    the first ensembles agreed on it with a spread of 0.035 eV. This project once
    called that a confident failure; it was not - the PBE gap is 0.85-0.90 eV in
    all four databases and the prediction matched it. The 2.0 eV it was compared
    with is an optical measurement. What is real is the sparse coverage, and that is
    what the latent distance measures. The current ensemble shows some doubt of its
    own (0.096 against a median of 0.092), a knife edge, so the test pins the
    outcome - not endorsed, and noticed by the latent distance - and the mechanism
    separately, on the verdict function.
    """
    from nanomat.predict import verdict

    r = P.run(read_structure(os.path.join(EX, "phosphorene.vasp")))
    assert not r.verdict.startswith("reliable"), "phosphorene must not be endorsed"
    assert r.latent_distance > P.cal["latent_q75"], "latent distance must notice it"

    # the mechanism: a spread the tiers would endorse, far from the training set
    confident = 0.5 * P.cal["unc_median"]
    assert verdict(confident, P.cal, None).startswith("reliable")
    between = 0.5 * (P.cal["latent_q75"] + P.cal["latent_q90"])
    assert verdict(confident, P.cal, between).startswith("check")
    assert verdict(confident, P.cal, 1.1 * P.cal["latent_q90"]).startswith("out-of-domain")


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


def test_training_graph_is_the_inference_graph(P, tmp_path):
    """The predictor pads a thin vacuum, so the trainer has to as well. Otherwise a
    layer with less vacuum than the cutoff trains on edges to its own periodic image
    and is later predicted without them - two Alexandria cells (elemental Br and
    Cl2) were, until build_graphs learned to pad."""
    import shutil
    from train_cgcnn import build_graphs
    shutil.copy(os.path.join(EX, "MoS2_thin_vacuum.vasp"), tmp_path / "thin.vasp")
    graphs, _ = build_graphs(str(tmp_path), ["thin.vasp"], P.cutoff, cache=False,
                             n_ang=P.n_ang)
    g_inf = P.graph(read_structure(os.path.join(EX, "MoS2_thin_vacuum.vasp")))
    assert graphs[0].num_edges == g_inf.num_edges
    assert torch.allclose(graphs[0].edge_weight.sort().values,
                          g_inf.edge_weight.sort().values, atol=1e-4)


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
        assert (a.p_metastable is None) == (b.p_metastable is None), key
        if a.p_metastable is not None:
            assert abs(a.p_metastable - b.p_metastable) < 1e-4, key
            assert abs(a.interval90 - b.interval90) < 1e-4, key


def test_interval_reads_the_population_the_structure_resembles(P):
    """One scale fitted on near-hull structures covers them, but covers ~84% of
    held-out metastable ones and 82% of the most confident spread quartile.
    scripts/calibrate_population.py fits a metastability head on the latent space
    and a scale per (resemblance, spread quartile) cell; the shipped ensemble must
    carry both, and near-hull reference monolayers must not look metastable."""
    import numpy as np
    assert P.pop is not None and "scale90_table" in P.pop
    assert P.pop["auc_heldout"] > 0.8        # the head separates the populations
    t = P.pop["scale90_table"]
    assert len(t["scale"]) == 2 and all(len(r) == len(t["unc_edges"]) + 1 for r in t["scale"])
    assert "population_threshold" not in P.cal   # it sets the interval, never the verdict
    for name in ("MoS2", "WS2", "WSe2", "MoSe2"):
        r = P.run(read_structure(os.path.join(EX, f"{name}.vasp")))
        assert r.p_metastable is not None and r.p_metastable < 0.5, (name, r.p_metastable)
        assert r.verdict == "reliable", name
        cell = t["scale"][int(r.p_metastable > t["p_threshold"])][int(np.digitize(r.unc, t["unc_edges"]))]
        assert abs(r.interval90 - cell * r.unc) < 1e-9, name


def test_metal_gate_rejects_graphene_and_spares_the_reference_semiconductors(P):
    """The gate is the first stage of the verdict, and its threshold is fitted on a
    validation split: a retrain can move it past a canonical semiconductor. In the
    retraining experiment the recipe of the shipped gate, rerun in a new environment,
    flagged WS2 (0.80 against a threshold of 0.36), and the arm the written
    tie-breaker preferred flagged WS2 at +1% in-plane strain - inside the spread of
    lattice constants between functionals. So the check runs at -1%, 0 and +1%."""
    from pymatgen.core import Structure
    from nanomat.graph import layer_info
    if P.metal_model is None:
        pytest.skip("no metal gate in weights/")

    def gate(name, eps=0.0):
        st = P.prepare(read_structure(os.path.join(EX, f"{name}.vasp")))[0]
        L, axis = st.lattice.matrix.copy(), layer_info(st)["axis"]
        for i in range(3):
            if i != axis:          # stretch in the plane of the layer only
                L[i] *= 1 + eps
        return P.predict_metal(Structure(L, st.species, st.frac_coords))

    for eps in (-0.01, 0.0, 0.01):
        assert gate("graphene", eps) >= P.metal_thr, eps
        for name in ("MoS2", "MoSe2", "WS2", "WSe2", "hBN", "phosphorene"):
            p = gate(name, eps)
            assert p < P.metal_thr, (name, eps, p, P.metal_thr)


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
    # JARVIS dft_2d has 54 HSE gaps and only the in-domain ones are fitted: 32 for the
    # angular ensemble, 28 for the one trained with metastable data. 25 is the floor
    # below which a two-parameter leave-one-out stops meaning much.
    assert qp["n"] >= 25 and opt["n"] >= 100
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


def test_llm_tools_forgive_what_models_get_wrong():
    """The language-model tools, called directly: no MCP needed.

    Two failures from scripts/llm_probe.py are pinned here. A model that dropped the
    POSCAR comment line got three parse errors and then invented a "reliable"
    answer, so the parser forgives it. Two models read the bare 90% half-width as a
    full width, so the range is now explicit.
    """
    from nanomat import llm_tools as L
    text = open(os.path.join(EX, "phosphorene.vasp")).read()
    r = L.predict_structure(text.split("\n", 1)[1])          # comment line dropped
    assert r.get("formula") == "P" and "input_note" in r, r
    lo, hi = r["gap_range90_eV"]
    half = r["interval90_halfwidth_eV"]
    assert abs((hi - lo) - 2 * half) < 2e-3 and lo < r["band_gap_pbe_eV"] < hi
    assert "error" in L.predict_structure("not a structure")
    assert "mcp" not in L.__dict__, "the tools must not depend on the protocol"


def test_llm_tools_flag_what_the_verdict_misses():
    """The rows a model needs to be warned about carry the warning themselves.

    Needs screening_table.csv (not in git; scripts/precompute_screening.py writes it).
    """
    from nanomat import llm_tools as L
    if not os.path.exists(L.TABLE):
        pytest.skip("screening_table.csv not built")
    g = L.find_structures("graphene")
    assert g["formula"] == "C" and "input_note" in g
    assert any("reference_note" in s for s in g["structures"]), "graphene's 1.23 eV label is wrong"
    rows = {s["id"]: s for s in L.find_structures("1T-MoS2", limit=25)["structures"]}
    # 1T'-MoS2, 0.96 eV against a DFT 0.05, was the model's own "reliable" miss and
    # the reference flag had to catch it. The retrained metal gate rejects it
    # outright; either way the row must not reach a reader unwarned
    r = rows["Mo2S4-b06001f565b8"]
    assert r["verdict"].startswith("out-of-domain") or "reference_disagreement" in r
    assert all("optical_gap_estimate_eV" in s for s in rows.values())
