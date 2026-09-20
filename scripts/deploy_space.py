#!/usr/bin/env python
"""Publish to Hugging Face Spaces.

Two things could go to a Space, and only one of them is free:

  static  (default)  The browser over the precomputed predictions. Pure HTML, CSS
                     and data, no Python at run time. Static Spaces are free, and
                     unlike GitHub Pages they sit in a browsable gallery, which is
                     the whole reason to publish a second copy of the same page.
  gradio             The uploader, which needs torch to run the ensemble on a
                     structure you supply. As of September 2026 Hugging Face
                     requires a PRO subscription to host a Gradio Space even on
                     free CPU hardware, so this mode exists for whoever has one or
                     wants to move the same folder to another host. Locally the
                     uploader needs nothing: `python screen_bandgap.py --app`.

    python scripts/deploy_space.py                          # build the static payload
    python scripts/deploy_space.py --push                   # create + upload it
    python scripts/deploy_space.py --kind gradio --push     # needs HF PRO

Publishing needs `pip install huggingface_hub` and a token, from `hf auth login`
or HF_TOKEN. The token is read by huggingface_hub itself; this script never
handles or prints it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_REPO = "ac1esan/nanomat-ai"
BROWSER_URL = "https://ac1esan.github.io/nanomat-ai/"
REPO_URL = "https://github.com/ac1esan/nanomat-ai"

# gradio payload: only what the uploader imports at run time
GRADIO_FILES = ["app.py", "screen_bandgap.py"]
GRADIO_DIRS = ["nanomat", "examples"]
WEIGHTS = ["cgcnn_2d_ensemble.pt", "cgcnn_2d_metal.pt", "cgcnn_2d_typed.pt"]

FRONT_MATTER = """---
title: NanoMatAI 2D Band Gap
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: {sdk}
{extra}pinned: false
license: mit
short_description: Band gaps of 28 372 2D structures, with a verdict
---

"""

STATIC_BODY = """# NanoMatAI — band gaps of 2D materials, with a verdict on each

Browse predicted band gaps for **28 372 two-dimensional structures**. Filter by
element, gap range and trust verdict, and see a periodic-table map of where the
model actually works.

A CGCNN ensemble of five models trained on 13 349 stable 2D semiconductors from
Alexandria, reading bond angles as well as bond lengths. Test MAE **0.25 eV** on a
split where no composition is shared between training and test. Nothing is computed
in this page: the predictions were made once and are served as a static table.

**The verdict is the point.** Three independent checks decide whether to trust a
prediction, and each exists because the previous one was caught failing on a real
case: a metal gate that rejects metals outright, the spread between ensemble
members, and the distance to the training set in the model's own latent space. The
last one exists because every ensemble member shares one training set, so a
chemistry none of them saw produces confident agreement — phosphorene came out at
0.82 eV against an experimental 2.0 with a spread of 0.035 eV.

Three many-body numbers are reported alongside the raw PBE gap: the **quasiparticle
gap** that photoemission and transport see, the **exciton binding energy**, and the
**optical gap** that absorption sees, which is the first minus the second. They come
from linear heads on the ensemble's own latent space, fitted against G₀W₀ and
Bethe–Salpeter results from C2DB and validated on composition-disjoint folds.

- **[Source, method and model card]({repo})**
- **[The same page on GitHub Pages]({browser})**

To run a structure of your own, clone the repository and use the uploader locally:
`python screen_bandgap.py --app`. It is not hosted here because a Gradio Space now
requires a paid subscription, and the model itself is free.
"""

GRADIO_BODY = """# NanoMatAI — band gap of 2D materials from structure

Upload a monolayer structure (CIF / POSCAR / .vasp) and get its band gap in under
a second on CPU, with a calibrated interval and an explicit verdict on whether the
number is usable.

- **[Browse 28 372 precomputed predictions]({browser})** — no upload needed.
- **[Source, method and model card]({repo})**
"""

GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n"

REQUIREMENTS = """# CPU wheels keep the image small and the cold start short; the default
# PyPI torch wheel for Linux drags in CUDA and is several GB.
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.4
torch_geometric>=2.5
pymatgen>=2024.1
gradio>=5.0
numpy>=1.26
pandas>=2.0
matplotlib>=3.8
"""


def build_static(out_dir: str) -> None:
    """The published browser: index.html plus its data, straight from docs/."""
    src = os.path.join(ROOT, "docs")
    index = os.path.join(src, "index.html")
    if not os.path.exists(index):
        raise SystemExit(f"missing {index} — run scripts/build_site_data.py first")
    shutil.copy(index, out_dir)
    shutil.copytree(os.path.join(src, "data"), os.path.join(out_dir, "data"))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(FRONT_MATTER.format(sdk="static", extra="app_file: index.html\n"))
        f.write(STATIC_BODY.format(repo=REPO_URL, browser=BROWSER_URL))


def build_gradio(out_dir: str) -> None:
    for name in GRADIO_FILES:
        shutil.copy(os.path.join(ROOT, name), out_dir)
    for name in GRADIO_DIRS:
        shutil.copytree(os.path.join(ROOT, name), os.path.join(out_dir, name),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.csv"))
    os.makedirs(os.path.join(out_dir, "weights"))
    for name in WEIGHTS:
        src = os.path.join(ROOT, "weights", name)
        if not os.path.exists(src):
            raise SystemExit(f"missing {src} — the uploader needs the shipped weights")
        shutil.copy(src, os.path.join(out_dir, "weights", name))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(FRONT_MATTER.format(sdk="gradio", extra="app_file: app.py\n"))
        f.write(GRADIO_BODY.format(repo=REPO_URL, browser=BROWSER_URL))
    with open(os.path.join(out_dir, "requirements.txt"), "w") as f:
        f.write(REQUIREMENTS)
    with open(os.path.join(out_dir, ".gitattributes"), "w") as f:
        f.write(GITATTRIBUTES)


def build(kind: str, out_dir: str) -> None:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    (build_static if kind == "static" else build_gradio)(out_dir)

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(out_dir) for f in fs)
    print(f"built {kind} payload in {out_dir}  ({total / 1e6:.1f} MB)")
    for dp, _, fs in sorted(os.walk(out_dir)):
        rel = os.path.relpath(dp, out_dir)
        for f in sorted(fs):
            path = os.path.join(rel, f) if rel != "." else f
            print(f"  {path:44s} {os.path.getsize(os.path.join(dp, f)) / 1000:8.0f} kB")


def push(kind: str, out_dir: str, repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub, then `hf auth login`")
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="space", space_sdk=kind, exist_ok=True)
    except Exception as e:
        if "402" in str(e):
            raise SystemExit(
                "Hugging Face refused: hosting a Gradio Space needs a PRO subscription.\n"
                "Static Spaces are free — run without --kind gradio to publish the browser,\n"
                "and run the uploader locally with `python screen_bandgap.py --app`.")
        raise
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="space",
                      commit_message=f"Deploy NanoMatAI ({kind})")
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\npushed to {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["static", "gradio"], default="static",
                    help="static = the browser (free); gradio = the uploader (needs HF PRO)")
    ap.add_argument("--out", help="build directory (default build/hf_<kind>)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face Space id, owner/name")
    ap.add_argument("--push", action="store_true", help="create the Space and upload")
    args = ap.parse_args()

    out = args.out or os.path.join(ROOT, "build", f"hf_{args.kind}")
    build(args.kind, out)
    if args.push:
        push(args.kind, out, args.repo)
    else:
        print(f"\nnot pushed. To publish:  python scripts/deploy_space.py "
              f"--kind {args.kind} --push --repo {args.repo}")


if __name__ == "__main__":
    main()
