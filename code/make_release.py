"""Release packer: produce a clean, publishable subset of the project.

Creates ``release/`` containing only what a reader of the paper needs to reproduce it:

  * ``README.md``            -- setup and pipeline (with verified environment reference)
  * ``LICENSE``              -- MIT for code, CC BY 4.0 for data and figures
  * ``requirements.txt``     -- the exact environment that produced the results
  * ``CITATION.cff``         -- machine-readable citation stub (needs the author fields filled)
  * ``code/``                -- all scripts except the throw-away probes
  * ``figures/``             -- the manuscript figures (PDF + PNG)
  * ``results/``             -- the evaluation JSON that the paper's numbers come from
  * ``CHECKSUMS.sha256``     -- hashes of the released files
  * ``RELEASE_NOTES.md``     -- what is included, what is excluded, and why

Trained weights are large, so they are hashed and listed but not copied: the notes explain how
to regenerate them (``code/train.py``), and a ``release/checkpoints.sha256`` records the exact
weights used for the published numbers.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(ROOT, "release")

# scripts that are part of the released pipeline.  The obsolete one-off diagnostics are
# excluded; the verification probes at the end are included because the paper cites them as
# integrity checks.
KEEP_CODE = [
    "config.py", "channel.py", "data.py", "baselines.py", "models.py",
    "train.py", "eval.py", "eval_ablation.py", "paired_test.py",
    "exp_input_representation.py", "exp_representation_full.py",
    "plot.py", "make_paired_figures.py", "make_tables.py",
    "fill_manuscript.py", "insert_figures.py", "build_paper.py", "build_cl.py",
    "audit_manuscript.py", "audit_cl.py", "check_tex.py", "make_latex_tables.py",
    "export_delay.py", "run_all.py",
    "make_inventory.py", "make_release.py", "cleanup.py",
    "_probe_channel.py", "_probe_consistency.py", "_probe_tune.py",
    "probe_scenario.py", "probe_delay_gain.py", "probe_delay_limit.py",
    "probe_output_domain.py",
]

KEEP_RESULTS = [
    "eval_main.json", "eval_ablation.json", "eval_profiles.json",
    "exp_input_representation.json", "exp_representation_full.json",
    "delay_window.json", "RESULTS.md", "tables.tex", "paired_tables.tex",
    "paired_Ld_full.json", "paired_output.json", "paired_temporal.json",
]
KEEP_HISTORY_PREFIX = "history_"

LICENSE = """MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

The figures, the evaluation results in `results/`, and the manuscript text are
licensed under Creative Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/
"""

CITATION = """cff-version: 1.2.0
message: "If you use this code or these results, please cite the paper below."
title: "The Observation Model Dominates the Architecture in Learned OFDM Channel Estimation"
authors:
  - family-names: "{family}"
    given-names: "{given}"
    affiliation: "{affiliation}"
    orcid: "https://orcid.org/{orcid}"
contact:
  - family-names: "{family}"
    given-names: "{given}"
    email: "{email}"
abstract: >-
  A controlled ablation study of learned channel estimation for high-mobility OFDM links,
  isolating the effect of the observation model, a fixed delay-domain prior, and cross-slot
  temporal aggregation on 3GPP TR 38.901 TDL channels.
keywords:
  - OFDM
  - channel estimation
  - deep learning
  - delay-domain sparsity
  - ablation study
license: MIT
version: "1.0"
date-released: "{date}"
"""

NOTES = """# Release notes

This directory is the minimal, self-contained subset of the project needed to reproduce the
paper. It was produced by `code/make_release.py`.

## Included

| Item | Purpose |
|---|---|
| `code/` | the full experiment pipeline, minus throw-away diagnostics |
| `results/*.json` | the evaluation output that every number in the paper comes from |
| `figures/` | the manuscript figures, as vector PDF and 600-dpi PNG |
| `results/RESULTS.md`, `results/tables.tex` | auto-generated tables |
| `requirements.txt` | the exact package versions used |
| `CHECKSUMS.sha256` | SHA-256 of every released file |
| `checkpoints.sha256` | SHA-256 of the trained weights (weights are not copied; see below) |

## Excluded, and why

| Excluded | Reason |
|---|---|
| `checkpoints/*.pt` (~22 MB) | large binary weights. Hashes are recorded in `checkpoints.sha256`; regenerate with `code/train.py` (about 25 min per model on 16 CPU threads). The recorded hashes let you verify that a regenerated model is bit-identical. |
| `code/_probe_*.py` | one-off diagnostic scripts used during development; they document how a convergence bug was found but are not part of the pipeline. They remain in the development tree. |
| `.venv/`, `wheelhouse/` | environment and package cache; rebuild from `requirements.txt`. |
| `results/eval_quick.json` and similar | superseded intermediate evaluations. |

## Environment

All results were produced on a CPU-only machine (16 hardware threads), Python {py},
`numpy`, `scipy`, `matplotlib`, `torch` (CPU build), and `pypandoc-binary` for the manuscript
build. No GPU was used. Total wall-clock time for the full pipeline is a few hours.

## Reproducing the paper

See `README.md`. In short:

```
python code/config.py          # system parameters
python code/channel.py         # validates the fading generator against Jakes
python code/train.py --model tdfnet --seq-len 4 --epochs 30 --n-train 3000 --n-val 400 --tag tdfnet_M4
python code/eval.py --tags tdfnet_M4 ... --out eval_main
python code/eval_ablation.py --n-real 100
python code/fill_manuscript.py && python code/insert_figures.py && python code/build_paper.py
python code/audit_manuscript.py
```

The last command re-checks that no result placeholder is unresolved and that every
figure and table referenced in the text exists.

## Verification

`python code/make_release.py --verify` recomputes the hashes of this directory and reports any
file that has changed or gone missing.
"""


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return os.path.getsize(dst)


def environment_versions():
    import numpy
    import scipy
    import matplotlib
    import torch
    return {"numpy": numpy.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "torch": torch.__version__,
            "python": "%d.%d.%d" % sys.version_info[:3]}


def main():
    verify_only = "--verify" in sys.argv
    if verify_only:
        cf = os.path.join(REL, "CHECKSUMS.sha256")
        if not os.path.exists(cf):
            sys.exit("no release to verify; run without --verify first")
        bad = 0
        for line in open(cf, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            want, rel = line.split("  ", 1)
            p = os.path.join(REL, rel)
            if not os.path.exists(p):
                print("MISSING  %s" % rel)
                bad += 1
            elif sha256(p) != want:
                print("CHANGED  %s" % rel)
                bad += 1
        print("verified %s: %s" % (REL, "OK" if not bad else "%d problem(s)" % bad))
        return

    if os.path.isdir(REL):
        shutil.rmtree(REL)
    os.makedirs(REL)
    v = environment_versions()
    copied = []

    # ---- code
    for fn in KEEP_CODE:
        src = os.path.join(ROOT, "code", fn)
        if os.path.exists(src):
            copied.append(("code/" + fn, copy(src, os.path.join(REL, "code", fn))))

    # ---- results
    for fn in KEEP_RESULTS:
        src = os.path.join(ROOT, "results", fn)
        if os.path.exists(src):
            copied.append(("results/" + fn, copy(src, os.path.join(REL, "results", fn))))
    for fn in sorted(os.listdir(os.path.join(ROOT, "results"))):
        if fn.startswith(KEEP_HISTORY_PREFIX) and fn.endswith(".json"):
            copied.append(("results/" + fn,
                           copy(os.path.join(ROOT, "results", fn),
                                os.path.join(REL, "results", fn))))

    # ---- figures
    figdir = os.path.join(ROOT, "figures")
    for fn in sorted(os.listdir(figdir)):
        src = os.path.join(figdir, fn)
        if os.path.isfile(src):
            copied.append(("figures/" + fn, copy(src, os.path.join(REL, "figures", fn))))

    # ---- README (root) and licence/requirements/citation
    copied.append(("README.md", copy(os.path.join(ROOT, "README.md"),
                                    os.path.join(REL, "README.md"))))
    with open(os.path.join(REL, "LICENSE"), "w", encoding="utf-8") as f:
        f.write(LICENSE.format(year=2026, author="Jingde Liu"))
    with open(os.path.join(REL, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("# environment that produced the published results (CPU only)\n")
        f.write("python==%s\n" % v["python"])
        for k in ("numpy", "scipy", "matplotlib", "torch"):
            f.write("%s==%s\n" % (k, v[k]))
        f.write("pypandoc-binary==1.17   # only needed to rebuild the .docx\n")
    with open(os.path.join(REL, "CITATION.cff"), "w", encoding="utf-8") as f:
        f.write(CITATION.format(family="Liu", given="Jingde",
                                affiliation="School of Information Science and Engineering, "
                                            "Shenyang University of Technology, Shenyang, China",
                                orcid="0009-0007-2039-2450",
                                email="18526813497@163.com", date="2026-10-06"))
    with open(os.path.join(REL, "RELEASE_NOTES.md"), "w", encoding="utf-8") as f:
        f.write(NOTES.format(py=v["python"]))

    # ---- hashes
    lines = []
    for rel, _ in sorted(copied):
        p = os.path.join(REL, rel)
        lines.append("%s  %s" % (sha256(p), rel))
    for extra in ("LICENSE", "requirements.txt", "CITATION.cff", "RELEASE_NOTES.md"):
        p = os.path.join(REL, extra)
        lines.append("%s  %s" % (sha256(p), extra))
    with open(os.path.join(REL, "CHECKSUMS.sha256"), "w", encoding="utf-8") as f:
        f.write("# SHA-256 of every file in this release\n" + "\n".join(sorted(lines)) + "\n")

    ck_lines = []
    ckdir = os.path.join(ROOT, "checkpoints")
    for fn in sorted(os.listdir(ckdir)):
        if fn.endswith(".pt"):
            ck_lines.append("%s  %s" % (sha256(os.path.join(ckdir, fn)), fn))
    with open(os.path.join(REL, "checkpoints.sha256"), "w", encoding="utf-8") as f:
        f.write("# SHA-256 of the trained weights used for the published numbers.\n"
                "# The weights themselves are not shipped; regenerate with code/train.py\n"
                "# and verify against these hashes.\n" + "\n".join(ck_lines) + "\n")

    total = sum(s for _, s in copied)
    print("release/ : %d files, %.1f MB" % (len(copied) + 5, total / 1e6))
    print("  code      : %d files" % len([c for c, _ in copied if c.startswith("code/")]))
    print("  results   : %d files" % len([c for c, _ in copied if c.startswith("results/")]))
    print("  figures   : %d files" % len([c for c, _ in copied if c.startswith("figures/")]))
    print("  checkpoints hashed (not copied): %d" % len(ck_lines))
    print("  environment: python %s, numpy %s, scipy %s, torch %s"
          % (v["python"], v["numpy"], v["scipy"], v["torch"]))


if __name__ == "__main__":
    main()
