#!/usr/bin/env python
"""Export the shipped checkpoints for the in-browser predictor (docs/js/nanomat.js).

Writes three files:
  manifest.json  - architecture, normalisation, calibration, thresholds, the heads'
                   and the population's constants, and where every tensor lives
  weights.bin    - every member of the gap, metal-gate, gap-type and work-function
                   ensembles, plus the latent heads, one tensor after another
  refs.bin       - the training-set embeddings the latent distance is measured
                   against (gap model and work-function model), float16 as shipped

Everything the page computes comes from here, so the page cannot drift from the
checkpoints: re-export after every recalibration, the same rule as for
scripts/build_site_data.py.

    python scripts/export_web_model.py                         # -> docs/model/
    python scripts/export_web_model.py --weights-dtype float32 # exact, twice the size

Parity with Python is checked by scripts/web_parity.py + scripts/web_parity.mjs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanomat.predict import DEFAULT_CAL, METAL_GAP, WEIGHT_FILES  # noqa: E402

CAL_KEYS = ("unc_median", "unc_q75", "latent_q75", "latent_q90", "scale90", "tier_mae", "test_mae")


class Blob:
    """Tensors appended to one buffer, each 8-byte aligned so typed arrays can view it."""

    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, arr: np.ndarray, dtype: str) -> dict:
        np_dtype = {"float16": np.float16, "float32": np.float32, "int8": np.int8}[dtype]
        a = np.ascontiguousarray(arr, dtype=np_dtype).ravel()
        pad = (-self.size) % 8
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        spec = {"offset": self.size, "size": int(a.size), "dtype": dtype, "shape": list(arr.shape)}
        self.parts.append(a.tobytes())
        self.size += a.nbytes
        return spec

    def write(self, path: str):
        with open(path, "wb") as f:
            for p in self.parts:
                f.write(p)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refs_spec(blob: Blob, emb: np.ndarray, dtype: str) -> dict:
    """Unit-length training embeddings. int8 keeps one float32 scale per row: the
    latent distance is a mean of cosine similarities, which tolerates the rounding."""
    if dtype != "int8":
        return blob.add(emb, dtype)
    e = emb.astype(np.float32)
    scale = np.maximum(np.abs(e).max(1), 1e-12) / 127.0
    q = np.clip(np.rint(e / scale[:, None]), -127, 127)
    spec = blob.add(q, "int8")
    spec["scale"] = blob.add(scale, "float32")
    return spec


def members(ck: dict, blob: Blob, dtype: str, keep) -> list[dict]:
    out = []
    for sd in ck.get("state_dicts") or [ck["state_dict"]]:
        # float16 only for weight matrices: batch-norm statistics overflow it (a running
        # variance above 65 504 exists), and biases and buffers are too small to matter
        tensors = {k: blob.add(v.float().numpy(), dtype if (k.endswith(".weight") and v.dim() == 2) else "float32")
                   for k, v in sd.items() if keep(k)}
        out.append({"h": int(ck.get("h", 128)), "n_conv": int(ck.get("n_conv", 4)),
                    "n_rbf": int(ck.get("n_rbf", 40)), "n_ang": int(ck.get("n_ang") or 0),
                    "mean": float(ck["mean"]), "std": float(ck["std"]), "tensors": tensors})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "model"))
    ap.add_argument("--weights-dtype", choices=["float16", "float32"], default="float32")
    ap.add_argument("--refs-dtype", choices=["float16", "int8"], default="float16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    load = lambda k: torch.load(os.path.join(args.weights, WEIGHT_FILES[k]), map_location="cpu")
    W, R = Blob(), Blob()
    gap = load("ensemble")
    if int(gap.get("n_tri") or 0):
        raise SystemExit("the three-body update is not ported to the browser")
    no_reg_tail = lambda k: not k.startswith("reg.")      # classifiers: cls head only
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights_dtype": args.weights_dtype,
        "refs_dtype": args.refs_dtype,
        "metal_gap_eV": METAL_GAP,
        "models": {"gap": {"members": members(gap, W, args.weights_dtype, lambda k: True)}},
        "calibration": {k: gap.get("calibration", DEFAULT_CAL).get(k, DEFAULT_CAL.get(k))
                        for k in CAL_KEYS},
        "sha256": {WEIGHT_FILES["ensemble"]: sha256(os.path.join(args.weights, WEIGHT_FILES["ensemble"]))},
    }
    manifest["calibration"] = {k: v for k, v in manifest["calibration"].items() if v is not None}

    for key, name in (("metal", "metal"), ("type", "type")):
        p = os.path.join(args.weights, WEIGHT_FILES[name])
        if os.path.exists(p):
            ck = load(name)
            manifest["models"][key] = {"members": members(ck, W, args.weights_dtype, no_reg_tail),
                                       "threshold": float(ck.get("threshold", 0.5))}
            manifest["sha256"][WEIGHT_FILES[name]] = sha256(p)
    p = os.path.join(args.weights, WEIGHT_FILES["workfunction"])
    if os.path.exists(p):
        wf = load("workfunction")
        cal = dict(wf.get("calibration", {}))
        manifest["models"]["wf"] = {
            "members": members(wf, W, args.weights_dtype, lambda k: True),
            "calibration": {k: cal[k] for k in CAL_KEYS if k in cal}}
        manifest["sha256"][WEIGHT_FILES["workfunction"]] = sha256(p)

    # latent heads and the population head stay float32: they are dot products over
    # 640 features, where float16 rounding would add up
    heads = gap.get("optical_heads")
    if isinstance(heads, dict) and "exciton_binding" in heads:
        manifest["heads"] = {"w": {k: W.add(heads[k]["w"].numpy(), "float32")
                                   for k in ("gap_gw", "gap_dir_gw", "exciton_binding")},
                             "b": {k: float(heads[k]["b"]) for k in ("gap_gw", "gap_dir_gw", "exciton_binding")},
                             "n": heads.get("n"), "mae_cv": heads.get("mae_cv")}
    pop = gap.get("population")
    if isinstance(pop, dict) and "w" in pop:
        t = pop.get("scale90_table")
        manifest["population"] = {"w": W.add(pop["w"].numpy(), "float32"), "b": float(pop["b"]),
                                  "table": None if t is None else {
                                      "p_threshold": float(t["p_threshold"]),
                                      "unc_edges": [float(e) for e in t["unc_edges"]],
                                      "scale": [[float(v) for v in row] for row in t["scale"]]}}

    manifest["refs"] = {"gap": refs_spec(R, gap["reference_embeddings"].float().numpy(), args.refs_dtype)}
    if "wf" in manifest["models"] and wf.get("reference_embeddings") is not None:
        manifest["refs"]["wf"] = refs_spec(R, wf["reference_embeddings"].float().numpy(), args.refs_dtype)

    W.write(os.path.join(args.out, "weights.bin"))
    R.write(os.path.join(args.out, "refs.bin"))
    manifest["bytes"] = {"weights.bin": W.size, "refs.bin": R.size}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    n = {k: len(v["members"]) for k, v in manifest["models"].items()}
    print(f"members {n}; weights.bin {W.size / 1e6:.1f} MB ({args.weights_dtype}), "
          f"refs.bin {R.size / 1e6:.1f} MB ({args.refs_dtype}) -> {args.out}")


if __name__ == "__main__":
    main()
