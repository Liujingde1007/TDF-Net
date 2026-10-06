"""End-to-end experiment runner: evaluation -> tables -> figures.

Runs the full final protocol in the order required by the paper:

 1. main NMSE sweep:  TDL-C at v = 3/30/100/200 km/h, SNR -5..25 dB, all estimators
 2. profile robustness: TDL-A and TDL-D at v = 100 km/h
 3. BER sweep at v = 30 and 100 km/h
 4. temporal-window ablation (M = 1,2,4,8) and the delay-branch ablation
 5. complexity table
 6. LaTeX/Markdown tables and all figures

Usage:  python code/run_all.py [--n-real 150] [--quick]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CODE = os.path.join(ROOT, "code")


def run(script, *args):
    cmd = [PY, os.path.join(CODE, script)] + [str(a) for a in args]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print("[%s] exit=%d  %.1f min" % (script, r.returncode, (time.time() - t0) / 60), flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=150)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    main_models = ["tdfnet_M4", "tdfnet_M2", "tdfnet_M1", "rescnn_M1", "rescnn_M4gru",
                   "channelnet"]
    names = ["TDF-Net", "TDF-Net M=2 (temporal ablation)", "TDF-Net M=1 (temporal ablation)",
             "ResCNN", "ResCNN+GRU", "ChannelNet"]
    # the headline comparison uses one row per distinct architecture
    head_tags = ["tdfnet_M4", "rescnn_M1", "rescnn_M4gru", "channelnet"]
    head_names = ["TDF-Net", "ResCNN", "ResCNN+GRU", "ChannelNet"]

    n = args.n_real
    if args.quick:
        n = min(n, 60)

    # 1. main sweep: TDL-C, four velocities
    run("eval.py", "--tags", *head_tags, "--names", *head_names,
        "--snr", -5, 0, 5, 10, 15, 20, 25,
        "--vel", 3, 30, 100, 200, "--profiles", "TDL-C",
        "--n-real", n, "--ber-vel", 30, 100,
        "--out", "eval_main")

    # 2. profile robustness at 100 km/h
    run("eval.py", "--tags", *head_tags, "--names", *head_names,
        "--snr", -5, 0, 5, 10, 15, 20, 25, "--vel", 100,
        "--profiles", "TDL-A", "TDL-D", "--n-real", max(50, n // 2),
        "--skip-complexity", "--out", "eval_profiles")

    # 3. temporal-window and delay-branch ablation
    run("eval_ablation.py", "--n-real", max(50, n // 2))

    # 4. tables and figures
    run("make_tables.py", "--in", "eval_main", "--out-tex", "tables.tex",
        "--out-md", "RESULTS.md")
    run("plot.py", "--in", "eval_main", "--ablation", "eval_ablation")
    print("\nAll experiments complete. See results/ and figures/.")


if __name__ == "__main__":
    main()
