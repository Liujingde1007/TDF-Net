"""Paired significance test for estimator comparisons.

Two estimators evaluated on the *same* channel realisations must be compared with a paired
test, not by looking at two means.  For every SNR point this script

  1. regenerates the identical test slots for both estimators,
  2. computes the per-slot NMSE difference,
  3. reports the mean difference, its standard error, a 95 % confidence interval, and a paired
     two-sided p-value (t-test on the differences),

so that a statement such as "L_d = 4 is better than L_d = 16" is made only when the confidence
interval excludes zero.  The number of independent slots is reported with every result, and a
power note states the smallest difference the design can resolve.

Usage
-----
    python code/paired_test.py --tags tdfnet_M4 tdfnet_Ld4 --names "Ld=16" "Ld=4" \
        --snr -5 0 5 10 15 20 25 --vel 100 --n-real 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import SYS, TRAIN, EVAL
from data import make_sequences
from eval import load_model, forward, nmse_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--snr", nargs="+", type=float, default=[-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    ap.add_argument("--vel", type=float, default=100.0)
    ap.add_argument("--profile", default="TDL-C")
    ap.add_argument("--n-real", type=int, default=300)
    ap.add_argument("--out", default="paired_test")
    args = ap.parse_args()

    device = torch.device("cpu")
    nets, par = [], []
    for tag, name in zip(args.tags, args.names):
        net, ck = load_model(tag, device)
        nets.append(net)
        par.append(ck["n_params"])
        print("[load] %-22s tag=%-16s params=%d" % (name, tag, ck["n_params"]), flush=True)

    from scipy import stats
    out = {"velocity": args.vel, "profile": args.profile, "n_realizations": args.n_real,
           "snr_db": args.snr, "names": args.names, "params": par, "comparisons": {}}
    rng = np.random.default_rng(20260101)

    for s in args.snr:
        X, Y, _ = make_sequences(args.n_real, TRAIN.seq_len, rng, (args.vel, args.vel), (s, s),
                                 profile=args.profile, delay_spread_ns=EVAL.delay_spread_ns,
                                 fixed_velocity=args.vel, fixed_snr=s)
        H = Y[:, 0] + 1j * Y[:, 1]
        per = [nmse_of(forward(n, X, s), H) for n in nets]        # (n_real,) each
        rec = {}
        for i in range(len(nets)):
            for j in range(i + 1, len(nets)):
                key = "%s - %s" % (args.names[j], args.names[i])
                d = per[j] - per[i]                                # negative = j better
                n = len(d)
                se = float(np.std(d, ddof=1) / np.sqrt(n))
                t, p = stats.ttest_rel(per[j], per[i])
                rec[key] = {"mean_db": float(np.mean(d)), "se_db": se,
                            "ci95": [float(np.mean(d) - 1.96 * se), float(np.mean(d) + 1.96 * se)],
                            "p_value": float(p), "significant": bool(p < 0.05)}
        out["comparisons"][str(s)] = rec
        nm = "  ".join("%s=%+.2f" % (args.names[i], float(np.mean(per[i])))
                       for i in range(len(nets)))
        print("SNR %+5.1f  %s" % (s, nm), flush=True)
        for key, r in rec.items():
            print("            %-28s diff %+6.2f dB  SE %.2f  p=%.2e  %s"
                  % (key, r["mean_db"], r["se_db"], r["p_value"],
                     "SIGNIFICANT" if r["significant"] else "not significant"), flush=True)

    # power note: smallest resolvable effect at 95 % confidence
    from scipy import stats as st
    tcrit = st.t.ppf(0.975, args.n_real - 1)
    out["min_resolvable_db"] = float(tcrit * 1.0 / np.sqrt(args.n_real))
    print("\n%d independent slots per point; a paired difference is resolvable at 95 %% "
          "confidence if it exceeds roughly (t_crit * SD / sqrt(n))." % args.n_real)

    with open(os.path.join(RESULTS, args.out + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[saved] results/%s.json" % args.out)


if __name__ == "__main__":
    main()
