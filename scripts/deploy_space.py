#!/usr/bin/env python
"""Assemble (and optionally publish) the Hugging Face Space for the uploader.

The Space is the half of the interface that needs a runtime: it loads torch and
runs the ensemble on a structure you upload. The other half — browsing the
precomputed predictions — is a static page on GitHub Pages and needs no server.

    python scripts/deploy_space.py                      # build build/hf_space/ only
    python scripts/deploy_space.py --push               # build, then create + upload
    python scripts/deploy_space.py --push --repo you/name

Publishing needs `pip install huggingface_hub` and a token, either from
`huggingface-cli login` or in the HF_TOKEN environment variable. The token is
read by huggingface_hub itself; this script never handles or prints it.

Why a separate folder rather than pointing the Space at the GitHub repo: a Space
needs a README.md whose YAML front matter declares the SDK, and that front matter
would render as a stray table at the top of the project README on GitHub.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_REPO = "ac1esan/nanomat-ai"
DEFAULT_OUT = os.path.join(ROOT, "build", "hf_space")

# Only what the uploader actually imports at run time. The training scripts, the
# precomputed table and the browser stay out: the Space cold-starts faster and
# there is less to keep in sync.
FILES = ["app.py", "screen_bandgap.py"]
DIRS = ["nanomat", "examples"]
WEIGHTS = ["cgcnn_2d_ensemble.pt", "cgcnn_2d_metal.pt", "cgcnn_2d_typed.pt"]

SPACE_README = """---
title: NanoMatAI 2D Band Gap
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Band gap of a 2D monolayer from its structure, with a calibrated uncertainty
---

# NanoMatAI — band gap of 2D materials from structure

Upload a monolayer structure (CIF / POSCAR / .vasp) and get its band gap at the
PBE level in under a second on CPU, together with a calibrated interval and an
explicit verdict on whether the number is usable.

A CGCNN ensemble of five models trained on 13 349 stable 2D semiconductors from
Alexandria. Test MAE is **0.26 eV** on a split where no composition is shared
between training and test.

**The verdict is the point.** Three independent checks decide whether to trust a
prediction, and each exists because the previous one was caught failing on a real
case: a metal gate that rejects metals outright, the spread between ensemble
members, and the distance to the training set in the model's own latent space.
The last one exists because every ensemble member shares one training set, so a
chemistry none of them saw produces confident agreement — phosphorene came out at
0.82 eV against an experimental 2.0 with a spread of 0.035 eV.

An out-of-domain result deliberately hides its interval instead of showing a tight
number next to a prediction the tool has disowned.

- **[Browse 28 372 precomputed predictions]({browser})** — no upload needed, plus a
  periodic-table map of where the model actually works.
- **[Source, method and model card]({repo})**

Limits worth knowing: the target is the PBE gap, which underestimates real gaps;
the correction to experiment is fitted on five reference monolayers. Inputs must
be relaxed monolayers with a vacuum gap. The calibration holds for stable 2D
semiconductors and is optimistic outside that population.
"""

GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n"

REQUIREMENTS = """# CPU wheels keep the Space image small and the cold start short;
# the default PyPI torch wheel for Linux drags in CUDA and is several GB.
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.4
torch_geometric>=2.5
pymatgen>=2024.1
gradio>=5.0
numpy>=1.26
pandas>=2.0
matplotlib>=3.8
"""


def build(out_dir: str, browser: str, repo_url: str) -> None:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for name in FILES:
        shutil.copy(os.path.join(ROOT, name), out_dir)
    for name in DIRS:
        shutil.copytree(os.path.join(ROOT, name), os.path.join(out_dir, name),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.csv"))
    os.makedirs(os.path.join(out_dir, "weights"))
    for name in WEIGHTS:
        src = os.path.join(ROOT, "weights", name)
        if not os.path.exists(src):
            raise SystemExit(f"missing {src} — the Space needs the shipped weights")
        shutil.copy(src, os.path.join(out_dir, "weights", name))

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(SPACE_README.format(browser=browser, repo=repo_url))
    with open(os.path.join(out_dir, "requirements.txt"), "w") as f:
        f.write(REQUIREMENTS)
    with open(os.path.join(out_dir, ".gitattributes"), "w") as f:
        f.write(GITATTRIBUTES)

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(out_dir) for f in fs)
    print(f"built {out_dir}  ({total / 1e6:.1f} MB)")
    for dp, _, fs in sorted(os.walk(out_dir)):
        rel = os.path.relpath(dp, out_dir)
        for f in sorted(fs):
            path = os.path.join(rel, f) if rel != "." else f
            print(f"  {path:44s} {os.path.getsize(os.path.join(dp, f)) / 1000:8.0f} kB")


def push(out_dir: str, repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub, then `huggingface-cli login`")
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="space",
                      commit_message="Deploy NanoMatAI uploader")
    print(f"\npushed to https://huggingface.co/spaces/{repo_id}")
    print("First build takes a few minutes while torch installs. The free tier "
          "sleeps after inactivity and wakes in about 30 s.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face Space id, owner/name")
    ap.add_argument("--browser", default="https://ac1esan.github.io/nanomat-ai/")
    ap.add_argument("--repo-url", default="https://github.com/ac1esan/nanomat-ai")
    ap.add_argument("--push", action="store_true", help="create the Space and upload")
    args = ap.parse_args()

    build(args.out, args.browser, args.repo_url)
    if args.push:
        push(args.out, args.repo)
    else:
        print(f"\nnot pushed. To publish:  python {os.path.relpath(__file__, os.getcwd())} "
              f"--push --repo {args.repo}")


if __name__ == "__main__":
    main()
