#!/usr/bin/env python
"""Can 2DMatPedia's band gaps join Alexandria's in one training set?

2DMatPedia (6 351 monolayers, PBE, derived from Materials Project) is the one open
source that lands on this model's weak chemistry: five times the carbon structures
the model trained on, three times the nitrogen and oxygen ones. A first comparison
by composition put the two databases 0.44 eV apart on the same formula - but a
formula is not a structure. One composition carries several polymorphs, the two
databases need not have picked the same one, and 1H against 1T of one formula
already differs by 0.4 eV. That number mixed a disagreement between databases with
a difference between structures.

This script separates the two. Every monolayer is put in a common frame - vacuum
axis turned to the layer normal, vacuum set to one thickness, cell reduced to
primitive - because the databases pad their layers differently, and a matcher fed
raw cells would call MoS2 two different materials. pymatgen's StructureMatcher then
decides, pair by pair within a formula, which structures are the same. Only on
those pairs is a gap difference a statement about the data.

Measured on the matched pairs:
  energy   PBE total energy per atom. Two setups that agree here to a few meV share
           pseudopotentials, +U and cutoffs, so a gap difference is about how the
           band structure was sampled, not about the physics.
  gap      bias, spread and metal/semiconductor agreement, split by magnetism and by
           the +U chemistry (Co Cr Fe Mn Mo Ni V W with O or F), where two setups
           that disagree on +U would show it first.
  floor    Alexandria against itself: where two of its own entries are the same
           structure, their gap difference is noise the training labels already carry.
  arbiter  two more databases on the same structure: C2DB (PBE, but GPAW rather than
           VASP, gap with spin-orbit coupling) and JARVIS dft_2d (VASP, OptB88vdW -
           a different functional). Neither decides a disagreement of a tenth; both
           decide one of an electronvolt, which is what they are asked.

And on the unmatched ones: how many are structures the model has never seen, in
which chemistry, and how far above the Alexandria convex hull they sit. The hull
estimate needs no hull: for any Alexandria entry of the same composition,
E_hull = E - e_above_hull, and a composition-only energy correction cancels.

With --predict the shipped model is run on every entry: an external validation on
a database it never saw, with labels nobody fitted it to, and the harder test of
the trust layer - whether the verdict still ranks the error there.

    python scripts/audit_2dmatpedia.py --workers 8 --predict

Writes runs/audit_2dmatpedia.json and runs/audit_2dmatpedia_entries.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MATCH_VACUUM = 10.0   # angstrom, identical on both sides; only the matcher sees it
METAL_GAP = 0.01      # eV, the export's threshold: at or below is a metal
MAGNETIC = 0.1        # mu_B per cell
TRAIN_EHULL = 0.1     # eV/atom, the stability filter of the shipped training set
U_METALS = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}
U_ANIONS = {"O", "F"}
LIGHT = {"H", "B", "C", "N", "O"}
MATCHER = dict(ltol=0.1, stol=0.1, angle_tol=5, primitive_cell=False, scale=False,
               attempt_supercell=False)
TM = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
         "Hf Ta W Re Os Ir Pt Au Hg".split())


def num(v):
    """JARVIS writes "na" where a number is missing."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_pymatgen(atoms: dict):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice(atoms["lattice_mat"]), atoms["elements"], atoms["coords"],
                     coords_are_cartesian=atoms["cartesian"])


def standardize(st, vacuum: float = MATCH_VACUUM):
    """Layer normal along c, fixed vacuum, primitive cell.

    Returns (primitive, area_per_atom, thickness) or None when the cell has no
    vacuum gap. In-plane positions are kept exactly; a tilted vacuum vector is
    replaced by the layer normal, which is the same slab in a different cell.
    """
    from pymatgen.core import Lattice, Structure
    from nanomat.graph import layer_info

    info = layer_info(st)
    if not info["is_layer"]:
        return None
    ax = info["axis"]
    shift = np.zeros(3)
    shift[ax] = -info["_start"]
    s = st.copy()
    s.translate_sites(list(range(len(s))), shift, frac_coords=True, to_unit_cell=True)

    m = s.lattice.matrix
    j, k = [i for i in range(3) if i != ax]
    a1, a2 = m[j], m[k]
    normal = np.cross(a1, a2)
    area = float(np.linalg.norm(normal))
    normal /= area
    cart = s.cart_coords
    z = cart @ normal
    z -= z.min()
    thickness = float(z.max())
    inplane = cart - np.outer(cart @ normal, normal)
    uv = np.linalg.lstsq(np.stack([a1, a2], axis=1), inplane.T, rcond=None)[0].T
    c = thickness + vacuum
    frac = np.column_stack([uv, (z + vacuum / 2.0) / c])
    out = Structure(Lattice(np.stack([a1, a2, normal * c])), s.species, frac)
    try:
        out = out.get_primitive_structure()
    except Exception:
        pass
    return out, area / len(s), thickness


def _prep(item):
    key, atoms = item
    try:
        res = standardize(to_pymatgen(atoms))
    except Exception:
        res = None
    return key, res


_SM = None


def _init_matcher():
    global _SM
    from pymatgen.analysis.structure_matcher import StructureMatcher
    # Already primitive and in one frame, so no primitive search and no supercells.
    # Not the defaults: stol is a fraction of (V/N)^(1/3), and in a slab V is mostly
    # vacuum, so the default 0.3 with volume scaling accepted as MoS2 a structure
    # 1 eV/atom higher in energy (measured). Two PBE relaxations of one structure
    # differ by ~1% in lattice and hundredths of an angstrom in positions.
    _SM = StructureMatcher(**MATCHER)


def _match(task):
    key, s, cands = task
    hits = []
    for other, t in cands:
        try:
            if _SM.fit(s, t):
                rms = _SM.get_rms_dist(s, t)
                hits.append((other, float(rms[0]) if rms else 0.0))
        except Exception:
            pass
    return key, hits


def load():
    from jarvis.db.figshare import data
    from pymatgen.core import Composition
    m = data("twod_matpd")
    a = data("alex_pbe_2d_all")
    c = data("c2db")
    d = data("dft_2d")
    for x in m + c + d:
        x["_f"] = Composition(dict(Counter(x["atoms"]["elements"]))).reduced_formula
    for x in a:
        x["_f"] = Composition(x["formula"]).reduced_formula
    return m, a, c, d


def gap_stats(gm: np.ndarray, ga: np.ndarray) -> dict:
    metal_m, metal_a = gm <= METAL_GAP, ga <= METAL_GAP
    both = ~metal_m & ~metal_a
    out = {"n": int(len(gm)),
           "metal_agreement": float(np.mean(metal_m == metal_a)) if len(gm) else None,
           "metal_in_2dmatpedia_only": int(np.sum(metal_m & ~metal_a)),
           "metal_in_alexandria_only": int(np.sum(~metal_m & metal_a)),
           "n_both_semiconductor": int(both.sum())}
    if both.sum() >= 3:
        d = gm[both] - ga[both]
        out.update({
            "bias_eV": float(d.mean()), "median_diff_eV": float(np.median(d)),
            "mae_eV": float(np.abs(d).mean()), "median_abs_eV": float(np.median(np.abs(d))),
            "p90_abs_eV": float(np.percentile(np.abs(d), 90)),
            "within_0.05": float(np.mean(np.abs(d) <= 0.05)),
            "within_0.1": float(np.mean(np.abs(d) <= 0.1)),
            "within_0.2": float(np.mean(np.abs(d) <= 0.2)),
            "pearson": float(np.corrcoef(gm[both], ga[both])[0, 1]),
            # a sampling artefact is one-sided: a mesh that misses the extremum can
            # only overestimate the gap, never underestimate it
            "alexandria_higher": float(np.mean(d < -0.02)),
            "alexandria_lower": float(np.mean(d > 0.02)),
        })
    return out


def show(label: str, s: dict):
    if not s.get("n"):
        print(f"  {label:34s} n=0")
        return
    line = (f"  {label:34s} n={s['n']:<5d} metal agree {100 * s['metal_agreement']:5.1f}%"
            f"  (M only in 2DMP {s['metal_in_2dmatpedia_only']}, only in Alex "
            f"{s['metal_in_alexandria_only']})")
    print(line)
    if "mae_eV" in s:
        print(f"  {'':34s} both gapped {s['n_both_semiconductor']}: bias {s['bias_eV']:+.3f}"
              f"  MAE {s['mae_eV']:.3f}  median|d| {s['median_abs_eV']:.3f}"
              f"  p90 {s['p90_abs_eV']:.3f}  within 0.1: {100 * s['within_0.1']:.0f}%"
              f"  r {s['pearson']:.3f}  Alex higher {100 * s['alexandria_higher']:.0f}%"
              f" / lower {100 * s['alexandria_lower']:.0f}%")


def validate(m, rows, part_f) -> dict:
    """The shipped model on a database it never saw.

    Excluded: entries whose Alexandria twin is in the training split (a memory, not
    a prediction). Kept and tagged: whether the composition itself was trained on,
    since a new polymorph of a known formula is an easier question than a formula
    the model never met. Labels carry the inter-database noise measured above
    (~0.08 eV MAE on non-magnetic structures), so these errors are an upper bound.
    """
    from nanomat import Predictor
    from nanomat.predict import tier_key

    P = Predictor(os.path.join(ROOT, "weights"), verbose=False)
    t = time.time()
    items = [(i, to_pymatgen(x["atoms"])) for i, x in enumerate(m)]
    for i, pred in P.run_many(items, batch_size=256):
        if pred is None:
            continue
        rows[i].update({"pred_gap": pred.gap, "pred_unc": pred.unc,
                        "pred_tier": tier_key(pred.verdict), "pred_verdict": pred.verdict,
                        "pred_latent": pred.latent_distance, "p_metal": pred.p_metal})
    print(f"\nExternal validation: shipped model on {len(items)} entries  [{time.time() - t:.0f}s]")

    seen_f = part_f["train"]
    base = [r for r in rows if "pred_gap" in r and r.get("alex_role") != "train"]
    sc = [r for r in base if r["gap"] > METAL_GAP and r["mag"] <= MAGNETIC]
    out = {"note": "2DMatPedia labels; non-magnetic semiconductors; twins of training "
                   "structures excluded"}

    def line(label, rs):
        if len(rs) < 5:
            print(f"  {label:40s} n={len(rs)}")
            return None
        e = np.array([abs(r["pred_gap"] - r["gap"]) for r in rs])
        print(f"  {label:40s} n={len(rs):<5d} MAE {e.mean():.3f}  median {np.median(e):.3f}")
        return {"n": len(rs), "mae": float(e.mean()), "median": float(np.median(e))}

    print(" non-magnetic semiconductors, excluding twins of training structures:")
    out["all"] = line("all", sc)
    out["composition_trained_on"] = line("composition in the training split", [r for r in sc if r["formula"] in seen_f])
    out["composition_new"] = line("composition never trained on", [r for r in sc if r["formula"] not in seen_f])
    print(" by the model's own verdict (does the trust layer transfer?):")
    out["by_verdict"] = {k: line(k, [r for r in sc if r["pred_tier"] == k])
                         for k in ("reliable", "check", "out_of_domain")}
    print(" by stability on Alexandria's hull:")
    stab = {"ehull <= 0.1": lambda r: r.get("ehull_est") is not None and r["ehull_est"] <= TRAIN_EHULL,
            "ehull > 0.1": lambda r: r.get("ehull_est") is not None and r["ehull_est"] > TRAIN_EHULL,
            "unknown (formula absent)": lambda r: r.get("ehull_est") is None}
    out["by_stability"] = {k: line(k, [r for r in sc if f(r)]) for k, f in stab.items()}
    print(" by chemistry:")
    out["by_element"] = {el: line(f"contains {el}", [r for r in sc if el in r["elements"].split()])
                         for el in ("C", "N", "B", "H", "O", "Si", "P")}
    out["light_no_tm"] = line("light, no TM", [r for r in sc if r["light"]])
    out["has_tm"] = line("has TM", [r for r in sc if r["has_tm"]])

    # the metal gate on metals it never saw
    metals = [r for r in base if r["gap"] <= METAL_GAP and r["mag"] <= MAGNETIC
              and r.get("p_metal") is not None]
    semis = [r for r in sc if r.get("p_metal") is not None]
    if metals and semis:
        thr = P.metal_thr
        caught = float(np.mean([r["p_metal"] >= thr for r in metals]))
        false = float(np.mean([r["p_metal"] >= thr for r in semis]))
        out["metal_gate"] = {"threshold": thr, "n_metals": len(metals), "recall": caught,
                             "false_alarm_on_semiconductors": false}
        print(f" metal gate (threshold {thr:.2f}): catches {100 * caught:.0f}% of {len(metals)} "
              f"non-magnetic metals, flags {100 * false:.0f}% of semiconductors")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--split", default=os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.split.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "audit_2dmatpedia"))
    ap.add_argument("--predict", action="store_true",
                    help="run the shipped model on every entry: external validation")
    args = ap.parse_args()
    t0 = time.time()

    m, a, c, jd = load()
    role = {}
    if os.path.exists(args.split):
        sp = json.load(open(args.split))
        for name in ("train", "val", "test"):
            for f in sp[name]:
                role[f.rsplit(".", 1)[0]] = name
    by_f_a = defaultdict(list)
    part_f = defaultdict(set)
    for j, x in enumerate(a):
        by_f_a[x["_f"]].append(j)
        if x["mat_id"] in role:
            part_f[role[x["mat_id"]]].add(x["_f"])
    print(f"2DMatPedia {len(m)}, Alexandria {len(a)}; {len(role)} Alexandria ids in the "
          f"training split  [{time.time() - t0:.0f}s]")

    # --- 1) one frame for every structure the comparison can touch -----------------
    need_a = sorted({j for x in m for j in by_f_a.get(x["_f"], [])})
    items = [(("m", i), x["atoms"]) for i, x in enumerate(m)]
    items += [(("a", j), a[j]["atoms"]) for j in need_a]
    items += [(("c", q), x["atoms"]) for q, x in enumerate(c)]
    items += [(("d", q), x["atoms"]) for q, x in enumerate(jd)]
    std = {}
    with Pool(args.workers) as pool:
        for n, (key, res) in enumerate(pool.imap_unordered(_prep, items, chunksize=64), 1):
            std[key] = res
            if n % 4000 == 0:
                print(f"  standardised {n}/{len(items)}", end="\r", flush=True)
    bad_m = sum(std[("m", i)] is None for i in range(len(m)))
    bad_a = sum(std[("a", j)] is None for j in need_a)
    print(f"  standardised {len(items)} structures; no vacuum gap: 2DMP {bad_m}, "
          f"Alex {bad_a}  [{time.time() - t0:.0f}s]")

    # --- 2) match within a formula and a primitive cell size -------------------------
    tasks, n_pairs = [], 0
    for i, x in enumerate(m):
        s = std[("m", i)]
        if s is None:
            continue
        cands = [(j, std[("a", j)][0]) for j in by_f_a.get(x["_f"], [])
                 if std[("a", j)] is not None and len(std[("a", j)][0]) == len(s[0])]
        if cands:
            tasks.append((i, s[0], cands))
            n_pairs += len(cands)
    print(f"  {n_pairs} candidate pairs over {len(tasks)} 2DMatPedia entries")
    hits = {}
    with Pool(args.workers, initializer=_init_matcher) as pool:
        for n, (i, h) in enumerate(pool.imap_unordered(_match, tasks, chunksize=4), 1):
            hits[i] = h
            if n % 500 == 0:
                print(f"  matched {n}/{len(tasks)}", end="\r", flush=True)
    print(f"  matching done  [{time.time() - t0:.0f}s]                ")

    # --- 3) one row per 2DMatPedia entry ---------------------------------------------
    from nanomat.families import classify
    def e_atom(x):
        return x["energy_total"] / x["nsites"]

    rows = []
    for i, x in enumerate(m):
        els = set(x["atoms"]["elements"])
        r = {"id": x["material_id"], "source": x["source_id"], "formula": x["_f"],
             "gap": num(x["bandgap"]), "e_atom": num(x["energy_per_atom"]),
             "mag": abs(num(x["total_magnetization"]) or 0.0),
             "n_atoms": len(x["atoms"]["elements"]),
             "light": bool(els & LIGHT) and not (els & TM),
             "has_tm": bool(els & TM),
             "u_chem": bool(els & U_METALS) and bool(els & U_ANIONS),
             "elements": " ".join(sorted(els))}
        same_f = by_f_a.get(x["_f"], [])
        if std[("m", i)] is None:
            r["status"] = "no_vacuum_gap"
        elif not same_f:
            r["status"] = "formula_absent"
        elif not hits.get(i):
            r["status"] = "new_polymorph"
        else:
            r["status"] = "matched"
        if same_f:
            # e_above_hull on Alexandria's hull, through any entry of the formula
            est = [a[j]["e_above_hull"] + r["e_atom"] - e_atom(a[j]) for j in same_f
                   if r["e_atom"] is not None]
            r["ehull_est"] = float(np.median(est)) if est else None
            # the pairing a formula-level comparison makes: the lowest-hull polymorph
            jl = min(same_f, key=lambda j: a[j]["e_above_hull"])
            r["gap_lowest_polymorph"] = float(a[jl]["band_gap_ind"])
            r["n_alex_polymorphs"] = len(same_f)
        if r["status"] == "matched":
            best, rms = min(hits[i], key=lambda h: h[1])
            y = a[best]
            r.update({"alex_id": y["mat_id"], "rms": rms, "n_alex_matches": len(hits[i]),
                      "alex_gap": float(y["band_gap_ind"]),
                      "alex_gap_dir": float(y["band_gap_dir"]),
                      "alex_e_atom": e_atom(y), "alex_ehull": y["e_above_hull"],
                      "alex_mag": abs(float(y["total_mag"] or 0.0)),
                      "alex_spg": int(y["spg"]),
                      # 143-194: trigonal/hexagonal, i.e. a hexagonal 2D lattice with K
                      "hexagonal": 143 <= int(y["spg"]) <= 194,
                      "alex_role": role.get(y["mat_id"], "not_in_training_data"),
                      "area_ratio": std[("m", i)][1] / std[("a", best)][1] - 1.0,
                      "alex_match_ids": " ".join(a[j]["mat_id"] for j, _ in hits[i])})
            r["d_e_atom"] = (r["e_atom"] - r["alex_e_atom"]) if r["e_atom"] is not None else None
            r["family"] = classify(to_pymatgen(x["atoms"]))[0] or ""
        rows.append(r)

    status = Counter(r["status"] for r in rows)
    print(f"\n2DMatPedia entries by status: {dict(status)}")
    M = [r for r in rows if r["status"] == "matched"]

    # --- 4) is it one computational setup? ----------------------------------------
    ME = [r for r in M if r["d_e_atom"] is not None]
    dE = np.array([r["d_e_atom"] for r in ME])
    print(f"\nPBE energy, same structure (2DMP - Alex, eV/atom), n={len(ME)}:")
    print(f"  median {np.median(dE):+.4f}  median|d| {np.median(np.abs(dE)):.4f}  "
          f"|d|<0.01: {100 * np.mean(np.abs(dE) < 0.01):.0f}%  <0.05: "
          f"{100 * np.mean(np.abs(dE) < 0.05):.0f}%  >0.2: {int(np.sum(np.abs(dE) > 0.2))}")
    energy = {"n": len(ME), "median": float(np.median(dE)),
              "median_abs": float(np.median(np.abs(dE))),
              "frac_below_0.01": float(np.mean(np.abs(dE) < 0.01)),
              "frac_below_0.05": float(np.mean(np.abs(dE) < 0.05)), "by_group": {}}
    for name, sel in (("+U chemistry", lambda r: r["u_chem"]),
                      ("magnetic in either", lambda r: r["mag"] > MAGNETIC or r["alex_mag"] > MAGNETIC),
                      ("light, no TM", lambda r: r["light"])):
        d = np.array([r["d_e_atom"] for r in ME if sel(r)])
        if len(d):
            energy["by_group"][name] = {"n": len(d), "median_abs": float(np.median(np.abs(d)))}
            print(f"  {name:22s} n={len(d):<5d} median|d| {np.median(np.abs(d)):.4f}")

    # --- 5) the gaps ---------------------------------------------------------------
    def arr(sel, key="alex_gap"):
        use = [r for r in M if sel(r)]
        return np.array([r["gap"] for r in use]), np.array([r[key] for r in use])

    groups = {
        "all matched": lambda r: True,
        "non-magnetic in both": lambda r: r["mag"] <= MAGNETIC and r["alex_mag"] <= MAGNETIC,
        "magnetic in either": lambda r: r["mag"] > MAGNETIC or r["alex_mag"] > MAGNETIC,
        "magnetic state disagrees": lambda r: (r["mag"] > MAGNETIC) != (r["alex_mag"] > MAGNETIC),
        "+U chemistry": lambda r: r["u_chem"],
        "light, no TM": lambda r: r["light"],
        "light, no TM, non-magnetic": lambda r: r["light"] and r["mag"] <= MAGNETIC
                                                 and r["alex_mag"] <= MAGNETIC,
        "has TM": lambda r: r["has_tm"],
        "same energy (|dE|<0.01)": lambda r: r["d_e_atom"] is not None
                                             and abs(r["d_e_atom"]) < 0.01,
        "honeycomb (hc-A, hc-AB)": lambda r: r["family"] in ("hc-A", "hc-AB"),
        "MX2 (1H, 1T)": lambda r: r["family"] in ("1H-MX2", "1T-MX2"),
        "hexagonal lattice": lambda r: r["hexagonal"],
        "other lattices": lambda r: not r["hexagonal"],
        "hexagonal, non-magnetic": lambda r: r["hexagonal"] and r["mag"] <= MAGNETIC
                                            and r["alex_mag"] <= MAGNETIC,
        "other, non-magnetic": lambda r: not r["hexagonal"] and r["mag"] <= MAGNETIC
                                        and r["alex_mag"] <= MAGNETIC,
        "Alexandria stable (<=0.1)": lambda r: r["alex_ehull"] is not None
                                               and r["alex_ehull"] <= TRAIN_EHULL,
    }
    print("\nBand gap, same structure (2DMP - Alex):")
    gaps = {}
    for name, sel in groups.items():
        gaps[name] = gap_stats(*arr(sel))
        show(name, gaps[name])

    worst = sorted(M, key=lambda r: -abs(r["gap"] - r["alex_gap"]))[:20]
    print("\nLargest disagreements on the same structure:")
    for r in worst:
        de = f"{r['d_e_atom']:+.3f}" if r["d_e_atom"] is not None else "  n/a "
        print(f"  {r['formula']:12s} {r['id']:9s} = {r['alex_id']}  2DMP {r['gap']:.2f}"
              f"  Alex {r['alex_gap']:.2f}  spg {r['alex_spg']:3d}  dE {de}  rms {r['rms']:.3f}"
              f"  mag {r['mag']:.1f}/{r['alex_mag']:.1f}")

    # the confound, measured: same entries, the pairing a formula comparison makes
    confound = gap_stats(*arr(lambda r: True, "gap_lowest_polymorph"))
    print("\nSame 2DMP entries, paired by formula (lowest-hull Alexandria polymorph):")
    show("formula pairing", confound)

    # --- the arbiters: the same structure in two more databases ------------------
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(**MATCHER)
    index_m = {x["material_id"]: i for i, x in enumerate(m)}
    arbiter = {}
    for tag, key, db, gap_key, id_key in (("C2DB", "c", c, "gap", "id"),
                                          ("dft_2d", "d", jd, "optb88vdw_bandgap", "jid")):
        by_f = defaultdict(list)
        for q, x in enumerate(db):
            if std[(key, q)] is not None and num(x.get(gap_key)) is not None:
                by_f[x["_f"]].append(q)
        for r in M:
            s0 = std[("m", index_m[r["id"]])][0]
            for q in by_f.get(r["formula"], []):
                t = std[(key, q)][0]
                if len(t) == len(s0) and sm.fit(s0, t):
                    r[tag + "_gap"], r[tag + "_id"] = num(db[q][gap_key]), db[q][id_key]
                    break
        T = [r for r in M if r.get(tag + "_gap") is not None]
        big = [r for r in T if abs(r["gap"] - r["alex_gap"]) > 0.3]
        res = {"n_triples": len(T), "n_disagreements": len(big)}
        if T:
            dm = np.array([abs(r["gap"] - r[tag + "_gap"]) for r in T])
            da = np.array([abs(r["alex_gap"] - r[tag + "_gap"]) for r in T])
            res.update({"mae_2dmatpedia": float(dm.mean()), "mae_alexandria": float(da.mean())})
        if big:
            res["sides_with"] = dict(Counter(
                "2DMatPedia" if abs(r["gap"] - r[tag + "_gap"]) < abs(r["alex_gap"] - r[tag + "_gap"])
                else "Alexandria" for r in big))
        arbiter[tag] = res
        print(f"\nArbiter {tag}: {len(T)} matched structures found there"
              + (f"; mean |2DMP - {tag}| {res['mae_2dmatpedia']:.3f}, |Alex - {tag}| "
                 f"{res['mae_alexandria']:.3f}" if T else "")
              + (f"; on {len(big)} disagreements over 0.3 eV it sides with {res['sides_with']}"
                 if big else ""))
    print("\nEvery disagreement over 0.3 eV, with whatever arbiter exists:")
    for r in sorted((r for r in M if abs(r["gap"] - r["alex_gap"]) > 0.3),
                    key=lambda r: (r["mag"] > MAGNETIC or r["alex_mag"] > MAGNETIC, r["formula"])):
        arb = "  ".join(f"{t} {r[t + '_gap']:.2f}" for t in ("C2DB", "dft_2d")
                        if r.get(t + "_gap") is not None) or "no arbiter"
        print(f"  {r['formula']:10s} {r['family'] or '-':7s} 2DMP {r['gap']:.2f}  Alex "
              f"{r['alex_gap']:.2f}  | {arb:28s} mag {r['mag']:.1f}/{r['alex_mag']:.1f}")

    # Alexandria against itself: entries that match the same 2DMP structure
    floor = []
    idx = {x["mat_id"]: j for j, x in enumerate(a)}
    for r in M:
        ids = r["alex_match_ids"].split()
        if len(ids) > 1:
            g = [a[idx[k]]["band_gap_ind"] for k in ids]
            floor.append(max(g) - min(g))
    if floor:
        floor = np.array(floor)
        print(f"\nAlexandria duplicates of one structure: {len(floor)} groups, gap range "
              f"median {np.median(floor):.3f}, mean {floor.mean():.3f}, "
              f">0.1 eV in {100 * np.mean(floor > 0.1):.0f}%")

    # --- 6) what would be new to the model ------------------------------------------
    roles = Counter(r["alex_role"] for r in M)
    print(f"\nMatched entries by the Alexandria twin's role: {dict(roles)}")
    new = [r for r in rows if r["status"] in ("new_polymorph", "formula_absent")]
    new_sc = [r for r in new if r["gap"] > METAL_GAP]
    print(f"Structures Alexandria does not have: {len(new)}, semiconductors {len(new_sc)}")
    est_ok = [r for r in new_sc if r.get("ehull_est") is not None]
    if est_ok:
        e = np.array([r["ehull_est"] for r in est_ok])
        print(f"  with a hull estimate: {len(est_ok)}; e_above_hull median {np.median(e):.3f}, "
              f"<= {TRAIN_EHULL}: {int(np.sum(e <= TRAIN_EHULL))}")
    check = np.array([(r["ehull_est"], r["alex_ehull"]) for r in M
                      if r.get("ehull_est") is not None and r["alex_ehull"] is not None])
    if len(check):
        err = np.abs(check[:, 0] - check[:, 1])
        print(f"  hull estimate checked on matched entries: median error {np.median(err):.4f}"
              f" eV/atom, 90th pct {np.percentile(err, 90):.4f}")

    counts = Counter()
    for r in new_sc:
        for el in r["elements"].split():
            if el in LIGHT | {"Si", "P"}:
                counts[el] += 1
    print(f"  light elements among the new semiconductors: {dict(counts.most_common())}")

    # --- 7) the pool a merge could draw from ----------------------------------------
    held = part_f["val"] | part_f["test"]
    funnel = [("not in Alexandria", lambda r: r["status"] in ("new_polymorph", "formula_absent")),
              ("semiconductor", lambda r: r["gap"] > METAL_GAP),
              ("non-magnetic", lambda r: r["mag"] <= MAGNETIC),
              ("hull estimable", lambda r: r.get("ehull_est") is not None),
              (f"e_above_hull <= {TRAIN_EHULL}", lambda r: r["ehull_est"] <= TRAIN_EHULL),
              ("composition not in val/test", lambda r: r["formula"] not in held)]
    print("\nWhat a merge under the training set's own rules could add:")
    pool, pool_counts = rows, []
    for name, keep in funnel:
        pool = [r for r in pool if keep(r)]
        c = Counter(el for r in pool for el in r["elements"].split() if el in LIGHT)
        pool_counts.append({"step": name, "n": len(pool), **{el: c[el] for el in sorted(LIGHT)}})
        print(f"  {name:30s} {len(pool):5d}   " + "  ".join(f"{el} {c[el]:4d}" for el in "CNBHO"))
    have = Counter()
    for j, x in enumerate(a):
        if role.get(x["mat_id"]) == "train":
            have.update(set(x["atoms"]["elements"]) & LIGHT)
    print(f"  {'in the training set today':30s} {sum(v == 'train' for v in role.values()):5d}   "
          + "  ".join(f"{el} {have[el]:4d}" for el in "CNBHO"))

    # --- 8) external validation --------------------------------------------------
    external = None
    if args.predict:
        external = validate(m, rows, part_f)

    # --- 9) write ----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary = {"n_2dmatpedia": len(m), "status": dict(status), "energy": energy, "gap": gaps,
               "formula_pairing": confound, "arbiter": arbiter,
               "alexandria_floor": {"n": int(len(floor)), "median_range": float(np.median(floor)),
                                    "mean_range": float(np.mean(floor))} if len(floor) else None,
               "matched_roles": dict(roles), "merge_funnel": pool_counts,
               "external_validation": external,
               "new_structures": {"n": len(new), "semiconductors": len(new_sc),
                                  "light_element_counts": dict(counts)},
               "settings": {"match_vacuum": MATCH_VACUUM, "metal_gap": METAL_GAP,
                            "magnetic": MAGNETIC, "ltol": 0.2, "stol": 0.3, "angle_tol": 5}}
    json.dump(summary, open(args.out + ".json", "w"), indent=1)
    keys = sorted({k for r in rows for k in r})
    with open(args.out + "_entries.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten {args.out}.json and {args.out}_entries.csv  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
