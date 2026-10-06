"""Full-protocol version of the input-representation experiment.

The quick version (`exp_input_representation.py`) uses a fixed SNR and velocity.  This one
reproduces the *training distribution of the paper* -- velocity log-uniform in [20,140] km/h,
SNR uniform in [-5,25] dB, 4 pilot symbols per slot (3.57 % pilot overhead), M = 2 -- and
trains the same architecture twice, differing only in whether the observation is de-rotated by
the known pilot sequence.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import SYS, ChannelConfig
from channel import TDLChannel, generate_slot
from models import TDFNet, count_parameters

M = 2
NTR, NVA = 1200, 300
VEL_RANGE, SNR_RANGE = (20.0, 140.0), (-5.0, 25.0)
mask = SYS.pilot_mask()
kw = dict(n_fft=SYS.n_fft, n_sym=SYS.n_symbols, n_pilot_sym=len(SYS.pilot_symbols),
          base_ch=16, seq_len=M)


def build(n_seq, seed):
    rng = np.random.default_rng(seed)
    Xls = np.empty((n_seq, M, 4, SYS.n_fft, SYS.n_symbols), dtype=np.float32)
    Xraw = np.empty_like(Xls)
    Y = np.empty((n_seq, 2, SYS.n_fft, SYS.n_symbols), dtype=np.float32)
    snrs = np.empty(n_seq, dtype=np.float32)
    lo, hi = np.log(VEL_RANGE[0]), np.log(VEL_RANGE[1])
    for i in range(n_seq):
        v = float(np.exp(rng.uniform(lo, hi)))
        snr = float(rng.uniform(*SNR_RANGE))
        snrs[i] = snr
        sigma = np.sqrt(10 ** (-snr / 10.0))
        ch = TDLChannel(ChannelConfig(profile="TDL-C", delay_spread_ns=100.0,
                                      velocity_kmh=v), rng)
        H = ch.frequency_response(M)
        for m in range(M):
            y, x, _ = generate_slot(H[m], rng, snr)
            ls = np.where(mask, y / x, 0j)
            plane = np.full(H[m].shape, snr / 20.0)
            Xls[i, m] = np.stack([ls.real / sigma, ls.imag / sigma, mask.astype(float), plane])
            Xraw[i, m] = np.stack([y.real / sigma, y.imag / sigma, mask.astype(float), plane])
        Y[i, 0] = H[-1].real
        Y[i, 1] = H[-1].imag
    return Xls, Xraw, Y, snrs


print("building %d train / %d val sequences (full training distribution) ..." % (NTR, NVA),
      flush=True)
Xls, Xraw, Ytr, snr_tr = build(NTR, 11)
Vls, Vraw, Yva, snr_va = build(NVA, 22)
print("done", flush=True)


def fit(Xtr, Xva, label, iters=1500, bs=32, lr=1e-3):
    Xt, Yt, st = torch.from_numpy(Xtr), torch.from_numpy(Ytr), torch.from_numpy(snr_tr)
    Xv, Yv, sv = torch.from_numpy(Xva), torch.from_numpy(Yva), torch.from_numpy(snr_va)
    torch.manual_seed(0)
    net = TDFNet(**kw)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = 1e9
    for it in range(iters):
        idx = torch.randint(0, len(Xt), (bs,))
        opt.zero_grad()
        out = net(Xt[idx], st[idx])
        loss = (((out - Yt[idx]) ** 2).sum(dim=(1, 2, 3))
                / ((Yt[idx] ** 2).sum(dim=(1, 2, 3)) + 1e-12)).mean()
        loss.backward()
        opt.step()
        if it % 250 == 0 or it == iters - 1:
            with torch.no_grad():
                o = net(Xv, sv)
                nm = ((o - Yv) ** 2).sum(dim=(1, 2, 3)) / ((Yv ** 2).sum(dim=(1, 2, 3)) + 1e-12)
                best = min(best, float(nm.mean()))
                print("   %-22s it%4d  val NMSE %+7.2f dB  E|out|^2 %.3f"
                      % (label, it, 10 * np.log10(nm.mean()), float((o ** 2).mean())), flush=True)
    print("  -> %-22s best val NMSE %+.2f dB" % (label, 10 * np.log10(best)), flush=True)
    return best


a = fit(Xls, Vls, "de-rotated (LS)")
b = fit(Xraw, Vraw, "raw y/sigma")
print("\nfull-protocol gap = %.2f dB" % (10 * np.log10(b) - 10 * np.log10(a)), flush=True)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
with open(os.path.join(RESULTS, "exp_representation_full.json"), "w") as f:
    json.dump({"derotated_nmse_db": 10 * np.log10(a), "raw_nmse_db": 10 * np.log10(b),
               "M": M, "n_train": NTR, "velocity_range": list(VEL_RANGE),
               "snr_range": list(SNR_RANGE), "params": count_parameters(TDFNet(**kw))}, f, indent=2)
print("[saved] results/exp_representation_full.json")
