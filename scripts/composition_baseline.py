#!/usr/bin/env python
"""Composition-only baseline on exactly the split a structure model was trained with.

"Does structure beat composition" is only answerable if both are scored on the same
test set. Earlier in this project a CGCNN and a Magpie baseline were compared across
different random splits and the conclusion had to be withdrawn; this script exists so
that cannot happen again. It takes the `<run>.split.json` the trainer wrote and uses
exactly those file lists.

    python scripts/composition_baseline.py --data data_c2db_wf --split runs/wf_ensemble.split.json

Works for any target, not just the band gap: the value comes from the second column
of the data folder's id_prop.csv, whatever that column means.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with .vasp files and id_prop.csv")
    ap.add_argument("--split", required=True, help="<run>.split.json written by train_cgcnn.py")
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=1,
                    help="matminer workers. Keep 1 on macOS: its spawn start method "
                         "deadlocks a script without a __main__ guard. On Linux raise it.")
    ap.add_argument("--json", help="write the metrics to this file")
    args = ap.parse_args()

    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (ElementProperty, Stoichiometry,
                                                  ValenceOrbital)
    from pymatgen.core import Composition, Structure
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    split = json.load(open(args.split))
    truth, formula = {}, {}
    for line in open(os.path.join(args.data, "id_prop.csv")):
        parts = line.strip().split(",")
        truth[parts[0]] = float(parts[1])
    names = split["train"] + split["val"] + split["test"]
    for f in names:
        formula[f] = Structure.from_file(os.path.join(args.data, f)).composition.reduced_formula
    print(f"split: train {len(split['train'])}  val {len(split['val'])}  test {len(split['test'])}")

    feat = MultipleFeaturizer([ElementProperty.from_preset("magpie"),
                               Stoichiometry(), ValenceOrbital()])
    feat.set_n_jobs(args.jobs)
    X = feat.featurize_many([Composition(formula[f]) for f in names],
                            ignore_errors=True, pbar=False)
    X = pd.DataFrame(X, columns=feat.feature_labels()).apply(pd.to_numeric, errors="coerce")
    ok = X.notna().all(1).values
    y = np.array([truth[f] for f in names])
    idx = {f: i for i, f in enumerate(names)}

    pick = lambda group: [idx[f] for f in split[group] if ok[idx[f]]]
    tr = pick("train") + pick("val")     # the structure model gets val for early stopping;
    te = pick("test")                    # giving it to the baseline too keeps this fair
    print(f"usable after featurisation: train+val {len(tr)}, test {len(te)}")

    model = RandomForestRegressor(n_estimators=args.trees, n_jobs=-1, random_state=0)
    model.fit(X.values[tr], y[tr])
    pred = model.predict(X.values[te])
    out = {
        "n_test": len(te),
        "mae": float(mean_absolute_error(y[te], pred)),
        "rmse": float(np.sqrt(np.mean((pred - y[te]) ** 2))),
        "r2": float(r2_score(y[te], pred)),
        "mae_predict_median": float(np.mean(np.abs(y[te] - np.median(y[tr])))),
        "split": os.path.basename(args.split),
        "data": os.path.basename(args.data.rstrip("/")),
    }
    print(f"\ncomposition baseline on the SAME test set as the structure model:")
    print(f"  MAE  {out['mae']:.3f}   RMSE {out['rmse']:.3f}   R2 {out['r2']:.3f}   n={out['n_test']}")
    print(f"  predicting the median would give MAE {out['mae_predict_median']:.3f}")
    top = sorted(zip(feat.feature_labels(), model.feature_importances_), key=lambda t: -t[1])[:6]
    print("  top features: " + ", ".join(k for k, _ in top))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
