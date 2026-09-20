#!/usr/bin/env python
"""Does the model tell two polymorphs apart, or only two compositions?

The headline MAE cannot answer this. A model can reproduce the global spread of
band gaps almost perfectly and still be nearly blind to the difference between two
structures of the same composition - and for 2D electronics that difference is the
whole point, since 1H-MoS2 is a semiconductor and 1T-MoS2 is metallic.

Three measurements, from the least to the most specific:

  global        sd(predicted) / sd(reference) over every row. Near 1.0 means the
                model is not simply regressing to the mean.
  within        the same ratio computed inside each composition and averaged. Here
                the composition is constant, so everything that varies is structure.
  1H vs 1T      the difference between the two prototypes of one element pair,
                predicted against reference. They share a composition, a
                coordination number and nearly a bond length; what separates them is
                an angle. This is the hardest of the three and the reason
                nanomat.graph.angle_features exists.

Measured on the shipped ensemble: 0.96, 0.57 and 0.18-0.27. So the model is healthy
at the level of composition and progressively worse the more the answer depends on
geometry alone.

    python scripts/polymorph_sensitivity.py --a weights/cgcnn_2d_ensemble.pt \
        --b runs/ens_ang.pt --data alignn_data_alex_2d

ONE data folder at a time, always. Mixing sources mixes DFT functionals, and then
the difference between two polymorphs is partly a difference between two codes -
a mistake this file exists partly to prevent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat import Predictor  # noqa: E402
from nanomat.families import classify  # noqa: E402

PAIR_PROTOTYPES = ("1H-MX2", "1T-MX2")


def load(path: str) -> Predictor:
    """Predictor over one checkpoint, whatever its filename."""
    tmp = tempfile.mkdtemp(prefix="poly_")
    shutil.copy(path, os.path.join(tmp, "cgcnn_2d_ensemble.pt"))
    for extra in ("cgcnn_2d_metal.pt", "cgcnn_2d_typed.pt"):
        src = os.path.join(ROOT, "weights", extra)
        if os.path.exists(src):
            shutil.copy(src, tmp)
    P = Predictor(tmp, verbose=False)
    print(f"{os.path.basename(path)}: {len(P.models)} members, angular width {P.n_ang}")
    return P


def collect(P: Predictor, data: str, role: dict) -> list[dict]:
    from pymatgen.core import Structure
    truth = {}
    for line in open(os.path.join(data, "id_prop.csv")):
        parts = line.strip().split(",")
        truth[parts[0]] = float(parts[1])
    names = sorted(truth)
    rows = []
    for start in range(0, len(names), 2000):
        chunk = names[start:start + 2000]
        items, fam = [], {}
        for f in chunk:
            st = Structure.from_file(os.path.join(data, f))
            items.append((f, st))
            fam[f] = classify(st)
        for key, pred in P.run_many(items, batch_size=256):
            if pred is None:
                continue
            rows.append({"file": key, "formula": pred.formula, "pred": pred.gap,
                         "dft": truth[key], "family": fam[key][0],
                         "a": fam[key][1], "b": fam[key][2],
                         "role": role.get(key, "unseen")})
        print(f"  {min(start + 2000, len(names))}/{len(names)}", end="\r", flush=True)
    print(f"  {len(rows)} structures predicted        ")
    return rows


def compression(rows: list[dict], label: str) -> dict:
    p = np.array([r["pred"] for r in rows])
    d = np.array([r["dft"] for r in rows])
    out = {"n": len(rows), "global": float(p.std() / d.std())}
    print(f"\n{label}  (n={len(rows)})")
    print(f"  global spread          sd(pred)/sd(DFT) = {out['global']:.2f}")

    by_comp: dict[str, list[dict]] = {}
    for r in rows:
        by_comp.setdefault(r["formula"], []).append(r)
    pairs = [(np.std([x["pred"] for x in g]), np.std([x["dft"] for x in g]))
             for g in by_comp.values() if len(g) >= 2]
    pairs = [(a, b) for a, b in pairs if b > 0.05]
    if pairs:
        out["within_composition"] = float(np.median([a for a, _ in pairs]) /
                                          np.median([b for _, b in pairs]))
        out["n_compositions"] = len(pairs)
        print(f"  within a composition   {out['within_composition']:.2f}"
              f"   ({len(pairs)} compositions with a real spread)")
    return out


def polymorph_pairs(rows: list[dict], label: str, out: dict) -> dict:
    """1H against 1T of the same element pair: composition constant, angle different."""
    keyed: dict[tuple, dict] = {}
    for r in rows:
        if r["family"] in PAIR_PROTOTYPES:
            keyed.setdefault((r["a"], r["b"]), {})[r["family"]] = r
    both = [v for v in keyed.values() if len(v) == 2]
    if not both:
        print("  no element pair carries both prototypes in this folder")
        return out

    def score(sel, tag):
        use = [v for v in both if sel(v)]
        if len(use) < 5:
            print(f"  {tag}: {len(use)} pairs, too few")
            return None
        dd = np.array([v["1H-MX2"]["dft"] - v["1T-MX2"]["dft"] for v in use])
        dp = np.array([v["1H-MX2"]["pred"] - v["1T-MX2"]["pred"] for v in use])
        c = float(np.median(np.abs(dp)) / np.median(np.abs(dd)))
        sign = float(np.mean(np.sign(dd) == np.sign(dp)))
        r = float(np.corrcoef(dd, dp)[0, 1])
        print(f"  {tag:22s} |d|DFT {np.median(np.abs(dd)):.3f}  "
              f"|d|pred {np.median(np.abs(dp)):.3f}  compression {c:.2f}"
              f"   sign {100 * sign:.0f}%  r {r:+.2f}   (n={len(use)})")
        return {"n": len(use), "compression": c, "sign_accuracy": sign, "pearson": r,
                "median_abs_dft": float(np.median(np.abs(dd))),
                "median_abs_pred": float(np.median(np.abs(dp)))}

    print(f"  1H against 1T, {len(both)} element pairs carry both:")
    out["polymorph_all"] = score(lambda v: True, "all pairs")
    # the honest subset: neither entry was in the training set, so neither number
    # can be a memory rather than a prediction
    out["polymorph_held_out"] = score(
        lambda v: all(x["role"] != "train" for x in v.values()), "neither one trained on")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint to measure")
    ap.add_argument("--b", help="second checkpoint, measured on the identical rows")
    ap.add_argument("--data", required=True, help="ONE data folder (one DFT functional)")
    ap.add_argument("--split", help="split.json, so training rows can be excluded")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json", help="write the numbers here")
    args = ap.parse_args()

    role = {}
    if args.split:
        sp = json.load(open(args.split))
        for name in ("train", "val", "test"):
            for f in sp[name]:
                role[f] = name
        print(f"split: {len(role)} files tagged")

    results = {}
    for tag, path, label in (("a", args.a, args.label_a), ("b", args.b, args.label_b)):
        if not path:
            continue
        P = load(path)
        rows = collect(P, args.data, role)
        res = compression(rows, label)
        res = polymorph_pairs(rows, label, res)
        results[label] = res

    if len(results) == 2:
        (la, ra), (lb, rb) = results.items()
        print(f"\n{'measurement':26s} {la:>10s} {lb:>10s}   closer to 1.0 is better")
        for key in ("global", "within_composition"):
            if key in ra and key in rb:
                print(f"  {key:24s} {ra[key]:10.2f} {rb[key]:10.2f}")
        for key in ("polymorph_all", "polymorph_held_out"):
            if ra.get(key) and rb.get(key):
                print(f"  {key + ' compression':24s} "
                      f"{ra[key]['compression']:10.2f} {rb[key]['compression']:10.2f}")
                print(f"  {key + ' sign':24s} "
                      f"{100 * ra[key]['sign_accuracy']:9.0f}% {100 * rb[key]['sign_accuracy']:9.0f}%")

    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
