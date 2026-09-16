#!/usr/bin/env python
"""Train CGCNN on exported 2D structures (format of export_structures_for_alignn.py).

Reproduces the shipped weights and supports the three tasks the tool uses:

  # band-gap regressor, 5-model ensemble, split grouped by composition (honest)
  python train_cgcnn.py --data alignn_data_alex_2d --task gap --split group \
      --ensemble 5 --epochs 200 --batch 64 --out weights/cgcnn_2d_ensemble.pt

  # direct/indirect classifier (id_prop.csv needs a 3rd column: direct gap)
  python train_cgcnn.py --data alignn_data_alex_2d_typed --task type --out weights/cgcnn_2d_typed.pt

  # metal/semiconductor gate (export with --keep-metals)
  python train_cgcnn.py --data alignn_data_alex_2d_all --task metal --out weights/cgcnn_2d_metal.pt

  # transfer learning: initialise the trunk from a 3D pre-trained checkpoint
  python train_cgcnn.py ... --pretrained weights/cgcnn_3d_pretrain.pt

Input folder: <id>.vasp files + id_prop.csv (no header): file,target[,target2].
Outputs next to --out: <out>.metrics.json (test metrics, ensemble calibration) and
<out>.split.json (exact file lists of train/val/test, for reproducibility).

Hardware notes (from the project log): build graphs serially (joblib + PyG
tensors hangs on Kaggle); on Kaggle pick a T4 (P100 is not supported by recent
torch). CPU is fine for a smoke run with --limit 200 --epochs 2.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core import Structure
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nanomat.graph import to_graph  # noqa: E402
from nanomat.model import CGCNN, CGCNNcls, load_trunk_from  # noqa: E402


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def read_id_prop(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, "id_prop.csv"), header=None)
    df = df.rename(columns={0: "file", 1: "target", 2: "target2"})
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if "target2" in df:
        df["target2"] = pd.to_numeric(df["target2"], errors="coerce")
    return df.dropna(subset=["target"]).reset_index(drop=True)


def build_graphs(data_dir: str, files: list[str], cutoff: float, cache: bool):
    cache_path = os.path.join(data_dir, f"graphs_cache_c{cutoff:g}.pt")
    if cache and os.path.exists(cache_path):
        ck = torch.load(cache_path, weights_only=False)
        if ck["files"] == files:
            print(f"graphs: loaded {len(files)} from cache {cache_path}")
            return ck["graphs"], ck["formulas"]
    graphs, formulas, t0 = [], [], time.time()
    for i, f in enumerate(files):
        st = Structure.from_file(os.path.join(data_dir, f))
        g = to_graph(st, cutoff)
        graphs.append(g)
        formulas.append(st.composition.reduced_formula)
        if (i + 1) % 500 == 0 or i + 1 == len(files):
            print(f"graphs: {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    if cache:
        torch.save({"files": files, "graphs": graphs, "formulas": formulas}, cache_path)
    return graphs, formulas


def make_split(n: int, formulas: list[str], mode: str, val_frac: float, test_frac: float, seed: int):
    """Return index arrays (train, val, test). `group` keeps every composition
    in exactly one subset, so the test set contains unseen chemistries."""
    idx = np.arange(n)
    if mode == "random":
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(idx, test_size=test_frac, random_state=seed)
        tr, va = train_test_split(tr, test_size=val_frac / (1 - test_frac), random_state=seed)
        return tr, va, te
    from sklearn.model_selection import GroupShuffleSplit
    groups = np.array(formulas)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr, te = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
    tr2, va = next(gss2.split(tr, groups=groups[tr]))
    return tr[tr2], tr[va], te


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, task, opt, mean, std, loss_fn, pos_weight, aux_w, device, train: bool):
    model.train(train)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for b in loader:
            b = b.to(device)
            y = (b.y - mean) / std
            if task == "gap":
                loss = loss_fn(model(b), y)
            else:
                reg, logit = model(b)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, b.lbl, pos_weight=pos_weight)
                loss = loss + aux_w * nn.functional.mse_loss(reg, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * b.num_graphs
            n += b.num_graphs
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, loader, task, mean, std, device):
    model.eval()
    ys, ps, lbls, probs = [], [], [], []
    for b in loader:
        b = b.to(device)
        if task == "gap":
            p = model(b)
        else:
            p, logit = model(b)
            lbls.append(b.lbl.cpu())
            probs.append(torch.sigmoid(logit).cpu())
        ps.append((p * std + mean).cpu())
        ys.append(b.y.cpu())
    out = {"y": torch.cat(ys).numpy(), "pred": torch.cat(ps).numpy()}
    if task != "gap":
        out["lbl"] = torch.cat(lbls).numpy()
        out["prob"] = torch.cat(probs).numpy()
    return out


def reg_metrics(y, p, n_boot: int = 1000, seed: int = 0) -> dict:
    err = p - y
    rng = np.random.default_rng(seed)
    boots = [np.mean(np.abs(err[rng.integers(0, len(err), len(err))])) for _ in range(n_boot)]
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "mae_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def pick_threshold(lbl, prob) -> float:
    """Decision threshold maximising balanced accuracy on the VALIDATION set.

    Training uses `pos_weight` on the rare class, which shifts the predicted
    probabilities away from 0.5. Keeping a hard 0.5 cut would systematically
    over-predict the positive class (e.g. flag semiconductors as metals), so the
    operating point is chosen from data and stored in the checkpoint.
    """
    from sklearn.metrics import balanced_accuracy_score
    grid = np.unique(np.quantile(prob, np.linspace(0.02, 0.98, 97)))
    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = balanced_accuracy_score(lbl, (prob >= t).astype(int))
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t


def cls_metrics(lbl, prob, threshold: float = 0.5) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    pred = (prob >= threshold).astype(int)
    return {
        "n": int(len(lbl)),
        "positive_rate": float(lbl.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(lbl, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(lbl, pred)),
        "f1_positive": float(f1_score(lbl, pred)),
        "accuracy": float((pred == lbl).mean()),
        "precision_positive": float((lbl[pred == 1].mean()) if (pred == 1).any() else 0.0),
    }


def ensemble_calibration(y, preds: np.ndarray) -> dict:
    """preds: (n_models, n). Uncertainty = std across members."""
    from scipy.stats import spearmanr
    mu, sd = preds.mean(0), preds.std(0)
    err = np.abs(mu - y)
    q = np.quantile(sd, [0.25, 0.5, 0.75])
    bins = np.digitize(sd, q)
    return {
        "median_unc": float(np.median(sd)),
        "spearman_unc_vs_abs_err": float(spearmanr(sd, err).correlation),
        "coverage_1sigma": float(np.mean(err <= sd)),
        "coverage_2sigma": float(np.mean(err <= 2 * sd)),
        "mae_by_unc_quartile": [float(err[bins == k].mean()) for k in range(4)],
        "unc_quartile_edges": [float(v) for v in q],
    }


def train_one(args, graphs, tr, va, te, mean, std, pos_weight, device, seed: int):
    set_seed(seed)
    if args.task == "gap":
        model = CGCNN(args.h, args.n_conv, args.dropout, args.cutoff, args.n_rbf)
    else:
        model = CGCNNcls(args.h, args.n_conv, args.cutoff, args.n_rbf)
    if args.pretrained:
        ck = torch.load(args.pretrained, map_location="cpu")
        sd = ck["state_dicts"][0] if "state_dicts" in ck else ck["state_dict"]
        print(f"  pretrained: copied {load_trunk_from(model, sd)} tensors from {args.pretrained}")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()

    # num_workers matters more than the GPU here: batching PyG graphs is pure
    # Python and saturates exactly one core, leaving the GPU ~40-60% idle.
    mk = lambda idx, shuffle: DataLoader(
        [graphs[i] for i in idx], batch_size=args.batch, shuffle=shuffle,
        num_workers=args.workers, persistent_workers=args.workers > 0,
        pin_memory=(str(device) != "cpu"))
    L_tr, L_va, L_te = mk(tr, True), mk(va, False), mk(te, False)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl = run_epoch(model, L_tr, args.task, opt, mean, std, loss_fn, pos_weight, args.aux_weight, device, True)
        vl = run_epoch(model, L_va, args.task, opt, mean, std, loss_fn, pos_weight, args.aux_weight, device, False)
        out = predict(model, L_va, args.task, mean, std, device)
        score = np.mean(np.abs(out["pred"] - out["y"])) if args.task == "gap" else vl
        sched.step(score)
        if score < best - 1e-5:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % args.log_every == 0 or ep == 1 or ep == args.epochs:
            tag = "val MAE" if args.task == "gap" else "val loss"
            print(f"  ep {ep:4d}  train {tl:.4f}  {tag} {score:.4f}  best {best:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.1e}  {time.time() - t0:.1f}s", flush=True)
        if bad >= args.patience:
            print(f"  early stop at epoch {ep} (no val improvement for {args.patience} epochs)")
            break
    model.load_state_dict(best_state)
    test = predict(model, L_te, args.task, mean, std, device)
    val = predict(model, L_va, args.task, mean, std, device)
    return best_state, test, val


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with id_prop.csv and .vasp files")
    ap.add_argument("--task", choices=["gap", "type", "metal"], default="gap")
    ap.add_argument("--out", required=True, help="checkpoint path (.pt)")
    ap.add_argument("--split", choices=["random", "group"], default="group",
                    help="group = no composition shared between train/val/test (default)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--loss", choices=["mse", "l1"], default="mse")
    ap.add_argument("--patience", type=int, default=40, help="early-stopping patience (epochs)")
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--n-conv", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--n-rbf", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0, help="split seed and first model seed")
    ap.add_argument("--ensemble", type=int, default=1, help="train N members with seeds seed..seed+N-1")
    ap.add_argument("--bootstrap", action="store_true",
                    help="bagging: resample the training set per ensemble member instead of "
                         "showing every member identical data. Without it the spread between "
                         "members only reflects initialisation, so a chemistry with one "
                         "training example produces confident agreement; with it roughly a "
                         "third of members never see that example and disagree.")
    ap.add_argument("--pretrained", help="checkpoint to initialise the trunk from (transfer learning)")
    ap.add_argument("--type-threshold", type=float, default=0.1,
                    help="task=type: direct if gap_dir - gap_ind < threshold (eV)")
    ap.add_argument("--metal-threshold", type=float, default=0.05,
                    help="task=metal: metal if gap <= threshold (eV)")
    ap.add_argument("--aux-weight", type=float, default=1.0, help="weight of the auxiliary gap loss in cls tasks")
    ap.add_argument("--limit", type=int, help="debug: use only the first N structures")
    ap.add_argument("--cache", action="store_true", help="cache built graphs inside --data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader worker processes (0 = main process). Raises GPU "
                         "utilisation; graph collation is the bottleneck, not the GPU.")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   torch {torch.__version__}")

    df = read_id_prop(args.data)
    if args.task == "type" and "target2" not in df:
        raise SystemExit("task=type needs a 3rd column (direct gap) in id_prop.csv; "
                         "export with --extra-target band_gap_dir")
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    files = df["file"].tolist()
    graphs, formulas = build_graphs(args.data, files, args.cutoff, args.cache)

    keep = [i for i, g in enumerate(graphs) if g is not None]
    if len(keep) < len(graphs):
        print(f"dropped {len(graphs) - len(keep)} structures without neighbours within cutoff")
    df, graphs, formulas = df.iloc[keep].reset_index(drop=True), [graphs[i] for i in keep], [formulas[i] for i in keep]

    y = df["target"].to_numpy(dtype=np.float32)
    for g, v in zip(graphs, y):
        g.y = torch.tensor([v], dtype=torch.float)
    if args.task == "type":
        lbl = ((df["target2"] - df["target"]) >= args.type_threshold).to_numpy(dtype=np.float32)
    elif args.task == "metal":
        lbl = (df["target"] <= args.metal_threshold).to_numpy(dtype=np.float32)
    else:
        lbl = None
    if lbl is not None:
        for g, v in zip(graphs, lbl):
            g.lbl = torch.tensor([v], dtype=torch.float)
        print(f"positive class rate ({'indirect' if args.task == 'type' else 'metal'}): {lbl.mean():.3f}")

    tr, va, te = make_split(len(graphs), formulas, args.split, args.val_frac, args.test_frac, args.seed)
    print(f"split={args.split}: train {len(tr)}  val {len(va)}  test {len(te)}  "
          f"(unique compositions: {len(set(formulas))})")
    mean, std = float(y[tr].mean()), float(y[tr].std() + 1e-8)
    pos_weight = None
    if lbl is not None:
        p = lbl[tr].mean()
        pos_weight = torch.tensor([(1 - p) / max(p, 1e-6)], device=device)
        print(f"pos_weight = {pos_weight.item():.2f}")

    states, tests, vals, member_metrics = [], [], [], []
    for k in range(args.ensemble):
        seed = args.seed + k
        print(f"\n=== model {k + 1}/{args.ensemble}  seed {seed} ===")
        tr_k = tr
        if args.bootstrap:
            rng = np.random.default_rng(seed)
            tr_k = rng.choice(tr, size=len(tr), replace=True)
            print(f"  bootstrap resample: {len(np.unique(tr_k))} unique of {len(tr)} "
                  f"({100 * len(np.unique(tr_k)) / len(tr):.0f}%)")
        state, test, val = train_one(args, graphs, tr_k, va, te, mean, std, pos_weight, device, seed)
        states.append(state)
        tests.append(test)
        vals.append(val)
        if args.task == "gap":
            m = reg_metrics(test["y"], test["pred"])
        else:
            thr = pick_threshold(val["lbl"], val["prob"])
            print(f"  threshold from validation: {thr:.3f} (0.5 would be wrong after pos_weight)")
            m = cls_metrics(test["lbl"], test["prob"], thr)
        member_metrics.append(m)
        print("  test:", json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in m.items()}))

    # --- aggregate + save --------------------------------------------------
    metrics = {"task": args.task, "split": args.split, "members": member_metrics,
               "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    if args.task == "gap":
        P = np.stack([t["pred"] for t in tests])
        metrics["ensemble"] = reg_metrics(tests[0]["y"], P.mean(0))
        if args.ensemble > 1:
            metrics["calibration"] = ensemble_calibration(tests[0]["y"], P)
        print("\nensemble test:", json.dumps(metrics["ensemble"]))
        if "calibration" in metrics:
            print("calibration:", json.dumps(metrics["calibration"]))
    else:
        prob = np.mean([t["prob"] for t in tests], 0)
        val_prob = np.mean([v["prob"] for v in vals], 0)
        threshold = pick_threshold(vals[0]["lbl"], val_prob)
        metrics["ensemble"] = cls_metrics(tests[0]["lbl"], prob, threshold)
        print("\nensemble test:", json.dumps(metrics["ensemble"]))

    meta = {"created": dt.datetime.now().isoformat(timespec="seconds"), "git": git_commit(),
            "torch": str(torch.__version__), "args": vars(args),
            "bagged": bool(args.bootstrap)}
    ck = {"mean": mean, "std": std, "cutoff": args.cutoff, "n_rbf": args.n_rbf,
          "h": args.h, "n_conv": args.n_conv, "meta": meta}
    if args.task == "gap" and args.ensemble > 1:
        ck["state_dicts"] = states
        ck["seeds"] = list(range(args.seed, args.seed + args.ensemble))
    else:
        ck["state_dict"] = states[0]
        if args.task != "gap":
            ck["cls"] = args.task
            ck["threshold"] = threshold
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ck, args.out)
    base = os.path.splitext(args.out)[0]
    with open(base + ".metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(base + ".split.json", "w") as f:
        json.dump({"train": [files[i] for i in tr], "val": [files[i] for i in va],
                   "test": [files[i] for i in te]}, f)
    print(f"\nsaved {args.out}, {base}.metrics.json, {base}.split.json")


if __name__ == "__main__":
    main()
