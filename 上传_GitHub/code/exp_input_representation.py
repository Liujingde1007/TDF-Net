"""Quantify the effect of the input representation.

The paper claims that de-rotating the observation by the known pilot sequence is a *necessary*
condition for any estimator that processes more than one slot, because the pilot QPSK values
differ from slot to slot.  This script provides the empirical counterpart: it trains the same
architecture twice at a low velocity, once on the de-rotated LS observation (the proposed
representation) and once on the raw received sample ``y/sigma`` (the representation used when
the pilot modulation is ignored).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import SYS, ChannelConfig
from channel import TDLChannel, generate_slot
from models import TDFNet, count_parameters
import baselines as bl

VEL, SNR = 10.0, 15.0          # low velocity: the temporal prior is maximally useful here
M = 4
N = 2000
rng = np.random.default_rng(0)
mask = SYS.pilot_mask()
kw = dict(n_fft=SYS.n_fft, n_sym=SYS.n_symbols, n_pilot_sym=len(SYS.pilot_symbols),
          base_ch=16, seq_len=M)


def build(n_seq):
    """Return features for both representations, plus the target and a residual-SNR baseline."""
    Xls = np.empty((n_seq, M, 4, SYS.n_fft, SYS.n_symbols), dtype=np.float32)
    Xraw = np.empty_like(Xls)
    Y = np.empty((n_seq, 2, SYS.n_fft, SYS.n_symbols), dtype=np.float32)
    sigma = np.sqrt(10 ** (-SNR / 10.0))
    for i in range(n_seq):
        ch = TDLChannel(ChannelConfig(profile="TDL-C", delay_spread_ns=100.0,
                                      velocity_kmh=VEL), rng)
        H = ch.frequency_response(M)
        for m in range(M):
            y, x, _ = generate_slot(H[m], rng, SNR)
            ls = np.where(mask, y / x, 0j)
            snr_plane = np.full(H[m].shape, SNR / 20.0)
            Xls[i, m] = np.stack([ls.real / sigma, ls.imag / sigma, mask.astype(float), snr_plane])
            Xraw[i, m] = np.stack([y.real / sigma, y.imag / sigma, mask.astype(float), snr_plane])
        Y[i, 0] = H[-1].real
        Y[i, 1] = H[-1].imag
    return Xls, Xraw, Y


print("generating %d sequences at v=%g km/h, SNR=%g dB ..." % (N, VEL, SNR), flush=True)
t0 = time.time()
Xls, Xraw, Y = build(N)
print("  done in %.1fs" % (time.time() - t0), flush=True)


def train_eval(X, label, iters=1200, bs=32, lr=1e-3):
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    st = torch.full((len(Xt),), SNR)
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
        if it % 200 == 0 or it == iters - 1:
            with torch.no_grad():
                o = net(Xt, st)
                n = ((o - Yt) ** 2).sum(dim=(1, 2, 3))
                d = (Yt ** 2).sum(dim=(1, 2, 3)) + 1e-12
                best = min(best, float((n / d).mean()))
                print("   %-14s it%4d  NMSE %+7.2f dB" % (label, it, 10 * np.log10((n / d).mean())),
                      flush=True)
    print("  -> %-14s final NMSE %+.2f dB (params %d)" % (label, 10 * np.log10(best),
                                                         count_parameters(net)), flush=True)
    return best


a = train_eval(Xls, "de-rotated (LS)")
b = train_eval(Xraw, "raw y/sigma")
print("\nNMSE difference = %.2f dB (de-rotated minus raw: negative is better)"
      % (10 * np.log10(a) - 10 * np.log10(b)))

# persist the result so that it can be quoted in the manuscript
import json
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "exp_input_representation.json"), "w") as f:
    json.dump({
        "derotated_nmse_db": 10 * np.log10(a),
        "raw_nmse_db": 10 * np.log10(b),
        "velocity_kmh": VEL, "snr_db": SNR, "M": M, "n_train": N,
        "params": count_parameters(TDFNet(**kw)),
        "note": "identical architecture/budget/schedule; only the input representation differs",
    }, f, indent=2)
print("[saved] results/exp_input_representation.json")

# reference: the classical LS/DFT estimators on the same data
H = Y[:, 0] + 1j * Y[:, 1]
sigma = np.sqrt(10 ** (-SNR / 10.0))
ls_hat = np.stack([bl.ls_linear((Xls[i, -1, 0] + 1j * Xls[i, -1, 1]) * sigma, mask, SNR)
                   for i in range(20)])
nmse_of = lambda a, b: None  # placeholder removed
print("reference LS-linear NMSE on 20 of the same slots: %+.2f dB"
      % np.mean(bl.nmse_db(ls_hat, H[:20])))
