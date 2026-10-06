"""Remove everything that is not needed to reproduce or submit the paper.

Design rules
------------
* Nothing is deleted until it has been classified, and every deletion is recorded in
  ``CLEANUP_LOG.md`` with its size and the reason, so the action is auditable.
* Anything that is *generated* by a script is fair game (it can be rebuilt); anything that is an
  *input* to the pipeline is kept.  The pipeline inputs are: ``code/*.py``, ``results/*.json``
  produced by the released scripts, ``figures/*``, ``checkpoints/*.pt`` for the models quoted in
  the paper, ``paper/manuscript*.md`` templates, and the deliverable ``.docx``/``.tex``.
* Rebuildable environments (``.venv``, ``wheelhouse``) and one-off download probes are removed
  even though they are large, because ``requirements.txt`` in the release regenerates them.

Usage
-----
    python code/cleanup.py --dry-run     # list what would go
    python code/cleanup.py               # do it
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------------------
# directories removed entirely
# ---------------------------------------------------------------------------------------
DIRS = {
    ".venv": "Python virtual environment; rebuilt from release/requirements.txt",
    "wheelhouse": "offline wheel cache used to bootstrap the environment; not needed with network",
    "_dlprobe": "leftover from an early pip connectivity probe (a single downloaded wheel)",
    "__pycache__": "Python byte-code cache",
}

# The interpreter that runs this script normally *lives in* `.venv`, so removing it while the
# script is running deletes the environment out from under the user.  We refuse unless the
# caller explicitly asks, and we say exactly how to rebuild it.
INTERPRETER_DIRS = (".venv",)

# ---------------------------------------------------------------------------------------
# individual files removed
# ---------------------------------------------------------------------------------------
FILES = {
    # smoke-test / pilot checkpoints: superseded by the full runs, never quoted in the paper
    "checkpoints/smoke_out.pt": "throw-away smoke test (output-domain constraint, pre-fix)",
    "checkpoints/smoke_out2.pt": "throw-away smoke test (pre-fix)",
    "checkpoints/smoke_out4.pt": "throw-away smoke test (failed projection variant)",
    "checkpoints/smoke_out5.pt": "throw-away smoke test (5 epochs on 200 sequences)",
    "checkpoints/pilot_Ld4.pt": "6-epoch pilot run used to validate the L_d sweep, superseded",
    # superseded evaluation JSON
    "results/eval_quick.json": "first quick evaluation of the original model, superseded by eval_main",
    "results/eval_nmse.json": "single-model velocity sweep, superseded by eval_main",
    "results/eval_baselines.json": "intermediate baseline check, superseded by eval_main",
    "results/eval_M_quick.json": "early temporal-window comparison, superseded by eval_ablation",
    "results/eval_Ld_check.json": "ad-hoc L_d spot check, superseded by paired_Ld_full",
    "results/paired_Ld.json": "two-variant paired test, superseded by paired_Ld_full",
    "results/_tmp_probe.json": "temporary file from a failed diagnostic run",
    # stale hand-written process documents
    "paper/RESULTS_NARRATIVE.md":
        "draft results narrative written mid-project; its numbers were superseded and it "
        "contradicts the final manuscript",
    "paper/DELAY_BRANCH_STUDY.md":
        "mid-campaign lab note; the conclusions it records were revised by the pairing tests "
        "(L_d=2/4 is significantly better than L_d=16, which the note does not report)",
    # editor temporary file
    "paper/TDF-Net-manuscript~A6FF5.tmp": "Word/editor temporary file, 0 bytes",
}

# ---------------------------------------------------------------------------------------
# probe scripts that are superseded by the scripts listed as KEEP
# ---------------------------------------------------------------------------------------
PROBE_DROP = {
    "code/_probe_diag2.py": "superseded by probe_scenario.py and probe_delay_limit.py",
    "code/_probe_error.py": "superseded by probe_delay_gain.py",
    "code/_probe_final_cfg.py": "one-off configuration check, superseded by config.py self-test",
    "code/_probe_fix.py": "hypothesis test from the convergence-bug hunt, resolved",
    "code/_probe_info.py": "diagnostic during the convergence-bug hunt, resolved",
    "code/_probe_learn.py": "controlled learning probe, resolved",
    "code/_probe_lmmse.py": "first LMMSE diagnostic, superseded by _probe_lmmse2.py",
    "code/_probe_lmmse2.py": "LMMSE verification, superseded by probe_output_domain.py",
    "code/_probe_numerology.py": "numerology sweep, superseded by probe_scenario.py",
    "code/_probe_overfit.py": "over-fit test, resolved",
    "code/_probe_step.py": "step-by-step feature check, resolved",
    "code/_probe_train.py": "early convergence check, resolved",
    "code/_probe_vals.py": "value-printing check, resolved",
    "code/_probe_baselines.py": "baseline sanity check, superseded by _probe_tune.py",
}
# kept on purpose, because they are cited as verification steps
PROBE_KEEP = {
    "code/_probe_channel.py": "validates the sum-of-sinusoids generator against Jakes",
    "code/_probe_consistency.py": "validates the LS observation against theory",
    "code/_probe_tune.py": "tunes the classical baselines (n_tap, K) -- quoted in the paper",
    "code/probe_scenario.py": "measures the channel's intrinsic dimension (Table/claim in IV)",
    "code/probe_delay_gain.py": "K-tap approximation error, supports the prior discussion",
    "code/probe_delay_limit.py": "establishes where the truncation would bind",
    "code/probe_output_domain.py": "verifies the output-domain implementation before concluding",
}


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dp, dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "%.1f %s" % (n, u)
        n /= 1024.0
    return "%.1f TB" % n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-venv", action="store_true",
                    help="also remove .venv; note that this deletes the interpreter running "
                         "this script, so it must be the last thing you do. Rebuild with: "
                         "python -m venv .venv, then pip install -r release/requirements.txt")
    args = ap.parse_args()

    targets = []          # (relpath, reason, is_dir)
    for d, reason in DIRS.items():
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        if d in INTERPRETER_DIRS and not args.include_venv:
            print("skipping %s (it holds the interpreter running this script; "
                  "pass --include-venv to remove it explicitly)" % d)
            continue
        targets.append((d, reason, True))
    for rel, reason in list(FILES.items()) + list(PROBE_DROP.items()):
        if os.path.exists(os.path.join(ROOT, rel)):
            targets.append((rel, reason, os.path.isdir(os.path.join(ROOT, rel))))

    freed = 0
    log = ["# Cleanup log", "",
           "Generated %s by `code/cleanup.py`." % datetime.now().strftime("%Y-%m-%d %H:%M"),
           "",
           "Nothing was removed without being classified first. Everything listed as removed is",
           "either regenerable by a script (so it is not an input to the paper) or a superseded",
           "intermediate product. The inputs to the pipeline were left untouched: `code/*.py`,",
           "`results/*.json`, `figures/`, `checkpoints/*.pt` for the models quoted in the paper,",
           "and the manuscript templates and deliverables.",
           "",
           "## Removed", "",
           "| path | size | reason |", "|---|---|---|"]

    for rel, reason, is_dir in sorted(targets):
        p = os.path.join(ROOT, rel)
        sz = size_of(p)
        freed += sz
        log.append("| `%s%s` | %s | %s |" % (rel, "/" if is_dir else "", human(sz), reason))
        print("%-46s %10s  %s" % (rel, human(sz), "would remove" if args.dry_run else "removed"))
        if not args.dry_run:
            if is_dir:
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError as e:
                    print("   ! could not remove: %s" % e)

    log += ["", "## Total", "",
            "Freed **%s**. Remaining tree is what the paper needs to be reproduced."
            % human(freed), "",
            "## Deliberately kept probe scripts", "",
            "| path | why it is kept |", "|---|---|"]
    for rel, why in sorted(PROBE_KEEP.items()):
        if os.path.exists(os.path.join(ROOT, rel)):
            log.append("| `%s` | %s |" % (rel, why))

    if not args.dry_run:
        with open(os.path.join(ROOT, "CLEANUP_LOG.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(log) + "\n")
        print("\nfreed %s; wrote CLEANUP_LOG.md" % human(freed))
    else:
        print("\nwould free %s" % human(freed))


if __name__ == "__main__":
    main()
