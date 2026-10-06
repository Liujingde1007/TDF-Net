"""Evaluation: NMSE / BER sweeps, mobility generalisation, and complexity accounting.

Every reported number is produced by this script from held-out test sequences; nothing in the
paper is hand-entered.  Results are written to ``results/*.json``.

Usage
-----
    python code/eval.py --n-real 100 --vel 30 100 200 --profiles TDL-C --out eval_main
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from config import SYS, TRAIN, EVAL
import baselines as bl
from data import make_sequences
from models import TDFNet, ChannelNet, CDRN, count_parameters
from train import build_model, nmse_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "checkpoints")

METHOD_ORDER = ["LS-linear", "DFT-denoise", "OMP", "LMMSE", "ChannelNet", "CDRN+GRU", "TDF-Net"]


# ---------------------------------------------------------------------------------------
def load_model(tag: str, device):
    path = os.path.join(CKPT, tag + ".pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build_model(ck["model"], ck["seq_len"], base_ch=ck.get("base_ch"),
                      n_taps=ck.get("n_taps"), use_delay=ck.get("use_delay", True),
                      rnn=ck.get("rnn", "gru"),
                      with_rnn=ck.get("with_rnn", ck["seq_len"] > 1 and ck["model"] == "rescnn"),
                      constraint=ck.get("constraint", "feature"))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def forward(net, X, snr_db):
    """X : (N, M, 4, N_f, N_t) float32 -> complex estimate (N, N_f, N_t)."""
    with torch.no_grad():
        x = torch.from_numpy(X)
        if isinstance(net, TDFNet):
            s = torch.full((x.shape[0],), float(snr_db))
            out = net(x, s)
        else:
            out = net(x)
    o = out.numpy()
    return o[:, 0] + 1j * o[:, 1]


def nmse_of(H_hat, H, data_only=False):
    err = H_hat - H
    if data_only:
        keep = np.ones(SYS.n_symbols, dtype=bool)
        for s in SYS.pilot_symbols:
            keep[s] = False
        err = err[:, :, keep]
        H = H[:, :, keep]
    num = np.mean(np.abs(err) ** 2, axis=(1, 2))
    den = np.mean(np.abs(H) ** 2, axis=(1, 2))
    return 10.0 * np.log10(np.maximum(num / den, 1e-12))


def _ls_obs(slot_feat: np.ndarray, sigma: float):
    """LS observation from the stored feature planes (planes 0/1 hold Re/Im(LS)/sigma)."""
    return (slot_feat[0] + 1j * slot_feat[1]) * sigma


# ---------------------------------------------------------------------------------------
def make_classical(profile: str, velocity: float, delay_spread_ns: float, rng):
    """Build the LMMSE filter and the classical estimators for one operating point."""
    from channel import TDLChannel
    from config import ChannelConfig
    ch = TDLChannel(ChannelConfig(profile=profile, delay_spread_ns=delay_spread_ns,
                                  velocity_kmh=velocity), rng)
    H_tr = ch.frequency_response(12000).reshape(-1, SYS.n_fft, SYS.n_symbols)
    lm = bl.LMMSE().fit(H_tr)
    mask = SYS.pilot_mask()
    return {
        "LS-linear": lambda ls, snr: bl.ls_linear(ls, mask, snr),
        "DFT-denoise": lambda ls, snr: bl.dft_denoise(ls, mask, snr),
        "OMP": lambda ls, snr: bl.omp(ls, mask, snr),
        "LMMSE": lambda ls, snr: lm.estimate(ls, mask, snr),
    }


def evaluate_point(models, profile, velocity, snr, n_real, rng, data_only=False, with_ber=False):
    """NMSE (and optionally BER) at a single (profile, velocity, SNR) operating point."""
    seq_len = max([getattr(net, "seq_len", 1) for net in models.values()] + [1])
    X, Y, _ = make_sequences(n_real, seq_len, rng, (velocity, velocity), (snr, snr),
                             profile=profile, delay_spread_ns=EVAL.delay_spread_ns,
                             fixed_velocity=velocity, fixed_snr=snr)
    H = Y[:, 0] + 1j * Y[:, 1]
    sigma = np.sqrt(10 ** (-snr / 10.0))
    sigma2 = sigma ** 2
    rec = {}
    cls = make_classical(profile, velocity, EVAL.delay_spread_ns, rng)
    for name, fn in cls.items():
        est = np.stack([fn(_ls_obs(X[i, -1], sigma), snr) for i in range(n_real)])
        rec[name] = float(np.mean(nmse_of(est, H, data_only)))
        if with_ber:
            rec[name + "_ber"] = float(np.mean(
                [bl.ber_mmse_1tap(est[i], H[i], sigma2, n_sym=2000, rng=rng)
                 for i in range(n_real)]))
    for name, net in models.items():
        est = forward(net, X, snr)
        rec[name] = float(np.mean(nmse_of(est, H, data_only)))
        if with_ber:
            rec[name + "_ber"] = float(np.mean(
                [bl.ber_mmse_1tap(est[i], H[i], sigma2, n_sym=2000, rng=rng)
                 for i in range(n_real)]))
    return rec


def complexity(models, seq_len=None, repeats=15):
    """Parameter count and measured CPU inference time per slot."""
    res = {}
    for name, net in models.items():
        m = net.seq_len if isinstance(net, TDFNet) else TRAIN.seq_len
        m = seq_len or m
        x = torch.randn(1, m, 4, SYS.n_fft, SYS.n_symbols)
        s = torch.tensor([10.0])
        with torch.no_grad():
            net(x, s) if isinstance(net, TDFNet) else net(x)
            t0 = time.time()
            for _ in range(repeats):
                net(x, s) if isinstance(net, TDFNet) else net(x)
            dt = (time.time() - t0) / repeats
        res[name] = {"params": count_parameters(net),
                     "inference_ms_per_window": dt * 1e3,
                     "inference_ms_per_slot": dt * 1e3 / m,
                     "window_M": m}
    # classical reference costs
    mask = SYS.pilot_mask()
    rng = np.random.default_rng(0)
    cls = make_classical("TDL-C", 100.0, EVAL.delay_spread_ns, rng)
    from channel import TDLChannel
    from config import ChannelConfig
    ch = TDLChannel(ChannelConfig(), rng)
    H = ch.frequency_response(1)[0]
    sigma = np.sqrt(10 ** (-10.0 / 10.0))
    ls = np.where(mask, H, 0j)
    for name, fn in cls.items():
        fn(ls, 10.0)
        t0 = time.time()
        for _ in range(repeats):
            fn(ls, 10.0)
        dt = (time.time() - t0) / repeats
        res[name] = {"params": 0, "inference_ms_per_window": dt * 1e3,
                     "inference_ms_per_slot": dt * 1e3, "window_M": 1}
    return res


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["tdfnet_M4", "cdrn_M4", "channelnet"])
    ap.add_argument("--names", nargs="+", default=["TDF-Net", "CDRN+GRU", "ChannelNet"])
    ap.add_argument("--snr", nargs="+", type=float, default=list(EVAL.snr_db))
    ap.add_argument("--vel", nargs="+", type=float, default=[30.0, 100.0, 200.0])
    ap.add_argument("--profiles", nargs="+", default=["TDL-C"])
    ap.add_argument("--n-real", type=int, default=EVAL.n_realizations)
    ap.add_argument("--ber-vel", nargs="+", type=float, default=None,
                    help="velocities for which BER is computed (default: none)")
    ap.add_argument("--ber-snr", nargs="+", type=float, default=None)
    ap.add_argument("--out", default="eval_main")
    ap.add_argument("--skip-complexity", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    device = torch.device("cpu")
    models, meta_info = {}, {}
    for tag, name in zip(args.tags, args.names):
        net, ck = load_model(tag, device)
        models[name] = net
        meta_info[name] = {"tag": tag, "params": ck["n_params"],
                           "val_nmse_db": 10 * np.log10(ck["val_loss"]), "epoch": ck["epoch"],
                           "window_M": net.seq_len if isinstance(net, TDFNet) else 1}
        print("[load] %-10s tag=%-12s params=%d  val=%.2f dB (ep %d)"
              % (name, tag, ck["n_params"], 10 * np.log10(ck["val_loss"]), ck["epoch"]), flush=True)

    payload = {"meta": meta_info, "snr_db": args.snr, "velocities": args.vel,
               "profiles": args.profiles, "n_realizations": args.n_real,
               "delay_spread_ns": EVAL.delay_spread_ns, "methods": [],
               "system": {"n_fft": SYS.n_fft, "n_symbols": SYS.n_symbols,
                          "scs_khz": SYS.scs_khz, "fc_ghz": SYS.fc_ghz,
                          "pilot_symbols": list(SYS.pilot_symbols),
                          "pilot_spacing": SYS.pilot_spacing,
                          "pilot_overhead": SYS.pilot_overhead(),
                          "n_pilots": int(SYS.pilot_mask().sum())}}

    print("\n=== NMSE sweep (all REs), %d realisations/point ===" % args.n_real, flush=True)
    rng = np.random.default_rng(777)
    nmse_all = {}
    for profile in args.profiles:
        nmse_all[profile] = {}
        for v in args.vel:
            nmse_all[profile][str(v)] = {}
            for snr in args.snr:
                t0 = time.time()
                rec = evaluate_point(models, profile, v, snr, args.n_real, rng)
                nmse_all[profile][str(v)][str(snr)] = rec
                print("  %-6s v=%6.1f snr=%+5.1f  " % (profile, v, snr)
                      + "  ".join("%s=%+.2f" % (k, val) for k, val in rec.items())
                      + "  (%.1fs)" % (time.time() - t0), flush=True)
    payload["nmse_all"] = nmse_all

    print("\n=== NMSE (data REs only) ===", flush=True)
    rng = np.random.default_rng(778)
    nmse_data = {}
    for profile in args.profiles:
        nmse_data[profile] = {}
        for v in args.vel:
            nmse_data[profile][str(v)] = {}
            for snr in args.snr:
                nmse_data[profile][str(v)][str(snr)] = evaluate_point(
                    models, profile, v, snr, max(50, args.n_real // 4), rng, data_only=True)
    payload["nmse_data"] = nmse_data

    if args.ber_vel:
        print("\n=== BER ===", flush=True)
        rng = np.random.default_rng(779)
        ber = {}
        for v in args.ber_vel:
            ber[str(v)] = {}
            for snr in (args.ber_snr or args.snr):
                rec = evaluate_point(models, "TDL-C", v, snr, max(60, args.n_real // 3), rng,
                                     with_ber=True)
                ber[str(v)][str(snr)] = {k[:-4]: val for k, val in rec.items() if k.endswith("_ber")}
                print("  v=%6.1f snr=%+5.1f  " % (v, snr)
                      + "  ".join("%s=%.2e" % (k, val) for k, val in ber[str(v)][str(snr)].items()),
                      flush=True)
        payload["ber"] = ber

    if not args.skip_complexity:
        print("\n=== complexity ===", flush=True)
        payload["complexity"] = complexity(models)
        for k, val in payload["complexity"].items():
            print("  %-10s params=%8d  %.2f ms/slot (M=%d)"
                  % (k, val["params"], val["inference_ms_per_slot"], val["window_M"]), flush=True)

    # ordered method list actually present
    present = set(payload["nmse_all"][args.profiles[0]][str(args.vel[0])][str(args.snr[0])])
    payload["methods"] = [m for m in METHOD_ORDER if m in present]

    with open(os.path.join(RESULTS, args.out + ".json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\n[saved] %s" % os.path.join(RESULTS, args.out + ".json"), flush=True)


if __name__ == "__main__":
    main()
