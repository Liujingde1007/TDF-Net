"""Ablation experiments.

Produces ``results/eval_ablation.json`` with:

  * ``by_M``                    : NMSE vs SNR for temporal windows M = 1, 2, 4, 8
  * ``with_delay`` / ``no_delay``: the parameter-free delay-domain branch removed
  * ``rescnn`` / ``rescnn_gru`` : the residual-CNN references at the same operating point
  * ``channelnet``              : the SRCNN-style reference
  * ``lmmse``                   : the classical sample-statistics LMMSE

All curves are evaluated on the TDL-C profile at the reference velocity.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from config import SYS, TRAIN, EVAL, ChannelConfig
from data import make_sequences
import baselines as bl
from eval import load_model, forward, nmse_of, _ls_obs
from channel import TDLChannel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def curve(net, snrs, velocity, profile, n_real, rng):
    seq_len = getattr(net, "seq_len", 1)
    out = []
    for snr in snrs:
        X, Y, _ = make_sequences(n_real, seq_len, rng, (velocity, velocity), (snr, snr),
                                 profile=profile, delay_spread_ns=EVAL.delay_spread_ns,
                                 fixed_velocity=velocity, fixed_snr=snr)
        H = Y[:, 0] + 1j * Y[:, 1]
        out.append(float(np.mean(nmse_of(forward(net, X, snr), H))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velocity", type=float, default=100.0)
    ap.add_argument("--profile", default="TDL-C")
    ap.add_argument("--snr", nargs="+", type=float,
                    default=[-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    ap.add_argument("--n-real", type=int, default=100)
    ap.add_argument("--out", default="eval_ablation")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    device = torch.device("cpu")
    payload = {"velocity": args.velocity, "profile": args.profile, "snr_db": args.snr,
               "n_realizations": args.n_real, "by_M": {}, "params": {}}

    def try_model(key, tag, store=None):
        path = os.path.join(ROOT, "checkpoints", tag + ".pt")
        if not os.path.exists(path):
            print("[skip] %s (%s not trained)" % (key, tag), flush=True)
            return None
        net, ck = load_model(tag, device)
        rng = np.random.default_rng(hash(tag) % 2**31)
        c = curve(net, args.snr, args.velocity, args.profile, args.n_real, rng)
        payload["params"][key] = ck["n_params"]
        print("%-12s params=%7d  " % (key, ck["n_params"])
              + " ".join("%+7.2f" % v for v in c), flush=True)
        return c

    for m in (1, 2, 4, 8):
        c = try_model("M%d" % m, "tdfnet_M%d" % m)
        if c is not None:
            payload["by_M"][str(m)] = c

    c = try_model("with_delay", "tdfnet_M4")
    if c is not None:
        payload["with_delay"] = c
    c = try_model("no_delay", "tdfnet_nodelay")
    if c is not None:
        payload["no_delay"] = c
    c = try_model("rescnn", "rescnn_M1")
    if c is not None:
        payload["rescnn"] = c
    c = try_model("rescnn_gru", "rescnn_M4gru")
    if c is not None:
        payload["rescnn_gru"] = c
    c = try_model("lstm", "tdfnet_lstm")
    if c is not None:
        payload["lstm"] = c
    c = try_model("channelnet", "channelnet")
    if c is not None:
        payload["channelnet"] = c

    # classical LMMSE at the same operating point
    rng = np.random.default_rng(7)
    ch = TDLChannel(ChannelConfig(profile=args.profile, delay_spread_ns=EVAL.delay_spread_ns,
                                  velocity_kmh=args.velocity), rng)
    lm = bl.LMMSE().fit(ch.frequency_response(8000).reshape(-1, SYS.n_fft, SYS.n_symbols))
    mask = SYS.pilot_mask()
    curve_lm = []
    for snr in args.snr:
        X, Y, _ = make_sequences(args.n_real, 1, rng, (args.velocity, args.velocity), (snr, snr),
                                 profile=args.profile, delay_spread_ns=EVAL.delay_spread_ns,
                                 fixed_velocity=args.velocity, fixed_snr=snr)
        H = Y[:, 0] + 1j * Y[:, 1]
        sigma = np.sqrt(10 ** (-snr / 10.0))
        est = np.stack([lm.estimate(_ls_obs(X[i, -1], sigma), mask, snr)
                        for i in range(args.n_real)])
        curve_lm.append(float(np.mean(nmse_of(est, H))))
    payload["lmmse"] = curve_lm
    print("%-12s                    " % "LMMSE" + " ".join("%+7.2f" % v for v in curve_lm), flush=True)

    with open(os.path.join(RESULTS, args.out + ".json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("[saved] %s" % os.path.join(RESULTS, args.out + ".json"), flush=True)


if __name__ == "__main__":
    main()
