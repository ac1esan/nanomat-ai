#!/usr/bin/env python
"""Can the trust layer see more than the spread and the latent distance?

Two questions, both asked of the shipped ensemble on structures it never trained on.

1. A better out-of-domain signal. Both signals the verdict uses lost ground as the
   training set grew (Spearman against |error| on the test split: spread 0.45 ->
   0.40, latent distance 0.47 -> 0.39, the two together 0.50 -> 0.44). Candidates:
   Mahalanobis distance in the first member's embedding space (the space the
   reference embeddings live in), k-nearest-neighbour cosine distance at other k,
   and an error head - a ridge regression of |error| on the signals and the
   concatenated member embeddings, fitted on the validation split.

2. The interval on other databases. The 90% interval covers 90% of unseen
   Alexandria rows but 82% against C2DB and 79% against JARVIS dft_2d, whose labels
   come from other methods (GPAW with spin-orbit coupling; the OptB88vdW
   functional). This measures how much wider an interval quoted *against those
   references* would have to be, by composition-grouped cross-validation.

    python scripts/trust_signals.py dump        # predictions, signals, embeddings
    python scripts/trust_signals.py signals     # question 1
    python scripts/trust_signals.py sources     # question 2

Decision rule for question 1, written before any candidate was scored (10 October
2026). A candidate enters the verdict only if, on sets no fit here touched:
  a. combined with the spread the way the latent distance is now (product of
     ranks), its Spearman against |error| on the test split beats the current
     combination (spread x latent distance) by more than 0.02, and
  b. it does not do worse than the current combination on any of the held-out
     metastable Alexandria, far (0.2-0.5 eV/atom) and 2DMatPedia sets.
The error head is fitted on the validation split only, so the test split and the
three held-out sets stay untouched by it. If nothing passes, the verdict stays as
it is and the result is written up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

HULL = os.path.join(ROOT, "alignn_data_hull")
OUT = os.path.join(ROOT, "runs", "trust", "signals.npz")
SOURCES = {"c2db": "alignn_data_c2db", "jarvis_dft2d": "alignn_data"}
HELD_OUT = ("T_meta", "T_far", "T_2dmp")


def id_prop(folder: str) -> dict[str, float]:
    out = {}
    for line in open(os.path.join(folder, "id_prop.csv")):
        f, y = line.strip().split(",")[:2]
        out[f] = float(y)
    return out


def dump(args):
    import torch
    from nanomat import Predictor
    from nanomat.predict import read_structure, tier_key, verdict

    split = json.load(open(os.path.join(ROOT, "weights", "cgcnn_2d_ensemble.split.json")))
    exp = json.load(open(os.path.join(HULL, "experiment.json")))
    hull_y = id_prop(HULL)
    sets = {"val": [(os.path.join(HULL, f), hull_y[f]) for f in split["val"]],
            "test": [(os.path.join(HULL, f), hull_y[f]) for f in split["test"]]}
    for k in HELD_OUT:
        sets[k] = [(os.path.join(HULL, f), hull_y[f]) for f in exp["tests"][k]]
    for name, folder in SOURCES.items():
        y = id_prop(os.path.join(ROOT, folder))
        sets[name] = [(os.path.join(ROOT, folder, f), v) for f, v in y.items()]

    P = Predictor(os.path.join(ROOT, "weights"), verbose=False)
    dev = args.device
    for m in P.models:
        m.to(dev)
    for m in (P.type_model, P.metal_model):
        if m is not None:
            m.to(dev)
    out = {}
    t0 = time.time()
    for name, items in sets.items():
        graphs, ys, files, formulas = [], [], [], []
        for path, y in items:
            st = read_structure(path)
            g = P.graph(st)
            if g is not None:
                graphs.append(g)
                ys.append(y)
                files.append(os.path.basename(path))
                formulas.append(st.composition.reduced_formula)
        with torch.no_grad():
            gap, unc, _, p_metal, emb = P._forward_many(graphs, 256, dev, 0)
            lat = np.concatenate([P.latent_distance(graphs[i:i + 256])
                                  for i in range(0, len(graphs), 256)])
            e0 = []
            for i in range(0, len(graphs), 256):
                from torch_geometric.data import Batch
                b = Batch.from_data_list(graphs[i:i + 256]).to(dev)
                e0.append(P.models[0].encode(b).cpu().numpy())
            e0 = np.concatenate(e0)
        pm = np.array([P.p_metastable(e) for e in emb])
        iv = np.array([P.interval90(u, p) for u, p in zip(unc, pm)])
        tiers = []
        for u, l, g in zip(unc, lat, p_metal):
            v = verdict(float(u), P.cal, float(l))
            if P.metal_thr is not None and g >= P.metal_thr:
                v = "out-of-domain (metal gate)"
            tiers.append(tier_key(v))
        out[name] = dict(gap=gap, unc=unc, lat=lat, p_metal=p_metal, p_meta=pm, iv=iv,
                         y=np.array(ys), tier=np.array(tiers), emb=emb.astype(np.float16),
                         emb0=e0.astype(np.float16), files=np.array(files),
                         formula=np.array(formulas))
        print(f"{name:13s} {len(graphs):5d} structures  MAE {np.abs(gap - np.array(ys)).mean():.3f}"
              f"  [{time.time() - t0:.0f}s]")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    flat = {f"{s}__{k}": v for s, d in out.items() for k, v in d.items()}
    flat["ref_emb0"] = P.ref_emb.numpy().astype(np.float16)
    np.savez_compressed(OUT, **flat)
    print(f"written {OUT}")


def load():
    z = np.load(OUT, allow_pickle=True)
    sets = {}
    for key in z.files:
        if "__" not in key:
            continue
        s, k = key.split("__", 1)
        sets.setdefault(s, {})[k] = z[key]
    return sets, z["ref_emb0"].astype(np.float32)


def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def _knn(e0, ref, k):
    """1 - mean cosine similarity to the k nearest reference embeddings."""
    import torch
    e = torch.tensor(_norm(e0))
    r = torch.tensor(ref)
    out = []
    for i in range(0, len(e), 1024):
        s = e[i:i + 1024] @ r.T
        out.append((1 - s.topk(k, dim=1).values.mean(1)).numpy())
    return np.concatenate(out)


def signals(args):
    from scipy.stats import spearmanr
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    sets, ref = load()
    names = ["val", "test", *HELD_OUT, "c2db", "jarvis_dft2d"]
    lw = LedoitWolf().fit(ref)
    mu, prec = lw.location_, lw.precision_
    for s in names:
        d = sets[s]
        d["err"] = np.abs(d["gap"] - d["y"])
        e0 = _norm(d["emb0"].astype(np.float32))
        diff = e0 - mu
        d["maha"] = np.sqrt(np.einsum("ij,jk,ik->i", diff, prec, diff))
        for k in (1, 5, 25, 50):
            d[f"knn{k}"] = _knn(d["emb0"].astype(np.float32), ref, k)

    # the error head: ridge on the signals and a PCA of the concatenated embeddings,
    # fitted on the validation split alone
    pca = PCA(n_components=32, random_state=0).fit(sets["val"]["emb"].astype(np.float32))

    def feats(d):
        base = np.column_stack([np.log(np.maximum(d["unc"], 1e-4)), d["lat"], d["maha"],
                                d["p_meta"], d["p_metal"], d["gap"], d["knn1"], d["knn50"]])
        return np.hstack([base, pca.transform(d["emb"].astype(np.float32))])

    sc = StandardScaler().fit(feats(sets["val"]))
    head = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(feats(sets["val"])),
                                                      np.log(sets["val"]["err"] + 0.02))
    print(f"error head: ridge alpha {head.alpha_:.3g}, fitted on {len(sets['val']['err'])} "
          "validation structures")

    cands = {"spread": lambda d: d["unc"], "latent k10 (now)": lambda d: d["lat"],
             "spread x latent (now)": lambda d: d["unc"] * d["lat"]}
    for k in ("maha", "knn1", "knn5", "knn25", "knn50"):
        cands[k] = (lambda kk: (lambda d: d[kk]))(k)
        cands[f"spread x {k}"] = (lambda kk: (lambda d: d["unc"] * d[kk]))(k)
    cands["error head"] = lambda d: head.predict(sc.transform(feats(d)))

    report = {}
    print(f"\nSpearman against |error| (higher is better); val is the head's fit set")
    print(f"{'signal':24s}" + "".join(f"{n:>10s}" for n in names))
    for c, f in cands.items():
        row = {n: float(spearmanr(f(sets[n]), sets[n]["err"]).correlation) for n in names}
        report[c] = row
        print(f"{c:24s}" + "".join(f"{row[n]:10.3f}" for n in names))

    # the rule, mechanically
    now = report["spread x latent (now)"]
    print("\ndecision rule (test beats now by > 0.02, no held-out set worse):")
    passed = []
    for c in cands:
        if not (c.startswith("spread x ") and "(now)" not in c) and c != "error head":
            continue
        a = report[c]["test"] - now["test"] > 0.02
        b = all(report[c][n] >= now[n] for n in HELD_OUT)
        print(f"  {c:22s} test {report[c]['test'] - now['test']:+.3f}  "
              + "  ".join(f"{n} {report[c][n] - now[n]:+.3f}" for n in HELD_OUT)
              + f"  -> {'PASS' if a and b else 'no'}")
        if a and b:
            passed.append(c)
    report["passed"] = passed

    # what a tier built from each signal would do: MAE in the quartiles of the signal
    print("\nMAE by quartile of the signal, test / T_meta / T_far / T_2dmp:")
    for c in ("spread x latent (now)", *passed):
        cells = []
        for n in ("test", *HELD_OUT):
            v, e = cands[c](sets[n]), sets[n]["err"]
            q = np.quantile(v, [0.25, 0.5, 0.75])
            b = np.digitize(v, q)
            cells.append("/".join(f"{e[b == i].mean():.2f}" for i in range(4)))
        print(f"  {c:22s} " + "   ".join(cells))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(os.path.join(os.path.dirname(OUT), "signals.json"), "w"), indent=1)


def sources(args):
    """How wide the interval would have to be to cover each database's own label."""
    from sklearn.model_selection import GroupKFold
    sets, _ = load()
    print("90% interval against each set's own reference, with the shipped half-widths")
    print(f"{'set':13s} {'n':>5s} {'covered':>8s} {'scale for 90%':>14s} {'CV coverage':>12s}")
    rep = {}
    for n in ("test", *HELD_OUT, "c2db", "jarvis_dft2d"):
        d = sets[n]
        err, iv = np.abs(d["gap"] - d["y"]), d["iv"]
        ok = (d["tier"] != "out_of_domain")      # the tool shows no interval out-of-domain
        r = (err / np.maximum(iv, 1e-6))[ok]
        cov = float((r <= 1).mean())
        need = float(np.quantile(r, 0.9))
        # composition-grouped 5-fold: scale fitted on four folds, coverage on the fifth
        groups = d["formula"][ok]
        hits = []
        for tr, te in GroupKFold(n_splits=5).split(r, groups=groups):
            s = max(1.0, np.quantile(r[tr], 0.9))
            hits += list(r[te] <= s)
        rep[n] = {"n": int(ok.sum()), "coverage": cov, "scale_needed": need,
                  "cv_coverage": float(np.mean(hits))}
        print(f"{n:13s} {ok.sum():5d} {100 * cov:7.1f}% {need:14.2f} {100 * np.mean(hits):11.1f}%")
    # C2DB: is the shortfall the spin-orbit offset of heavy-element compounds?
    d = sets["c2db"]
    from pymatgen.core import Composition
    heavy = np.array([max(el.Z for el in Composition(f).elements) >= 52 for f in d["formula"]])
    ok = d["tier"] != "out_of_domain"
    for lab, sel in (("C2DB heavy (Z>=52)", heavy & ok), ("C2DB other", ~heavy & ok)):
        e = d["gap"][sel] - d["y"][sel]
        cov = float((np.abs(e) <= d["iv"][sel]).mean())
        print(f"  {lab:20s} n={sel.sum():4d}  mean (model - C2DB) {e.mean():+.3f}  covered {100 * cov:.1f}%")
        rep[lab] = {"n": int(sel.sum()), "bias": float(e.mean()), "coverage": cov}
    json.dump(rep, open(os.path.join(os.path.dirname(OUT), "sources.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dump", "signals", "sources"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.stage == "dump":
        dump(args)
    elif args.stage == "signals":
        signals(args)
    else:
        sources(args)


if __name__ == "__main__":
    main()
