# Release notes

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

All results were produced on a CPU-only machine (16 hardware threads), Python 3.11.9,
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
