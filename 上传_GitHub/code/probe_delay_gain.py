"""Measure how much a delay-domain truncation can actually gain, per configuration.

The relevant quantity for the branch question is not the tap count but *how well the channel is
approximated by K delay-domain coefficients*: that is exactly what the fixed IDFT + truncation
branch makes available.  For each configuration we compute, from the exact channel realisations,

    NMSE_K = E[ ||H - H_K||^2 ] / E[ ||H||^2 ],

where H_K is the best (least-squares) K-tap band-limited approximation of H.  A steep decrease
in NMSE_K means a low-dimensional delay representation captures the channel, i.e. the prior is
worth exploiting.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from config import SYS, ChannelConfig
from channel import TDLChannel

CONFIGS = [
    ("TDL-C", 100, 256, 15.0, 8, "reference"),
    ("TDL-C", 300, 256, 15.0, 8, ""),
    ("TDL-C", 600, 256, 15.0, 8, ""),
    ("TDL-C", 1000, 256, 15.0, 8, ""),
    ("TDL-A", 1000, 256, 15.0, 8, ""),
    ("TDL-C", 600, 512, 15.0, 16, ""),
    ("TDL-C", 1000, 512, 15.0, 16, ""),
    ("TDL-C", 1000, 1024, 30.0, 32, "wideband"),
]
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

print("%-7s %6s %5s %5s | %s" % ("profile", "DS[ns]", "Nfft", "Sp",
                                 " ".join("K=%-5d" % k for k in KS)))
for prof, ds, nf, scs, sp, tag in CONFIGS:
    SYS.n_fft, SYS.scs_khz, SYS.pilot_spacing = nf, scs, sp
    ch = TDLChannel(ChannelConfig(profile=prof, delay_spread_ns=ds, velocity_kmh=3.0),
                    np.random.default_rng(11))
    H = ch.frequency_response(600)[:, :, 0]                   # (600, N_f) slow fading
    N = nf
    k = np.arange(N)
    # band-limited basis: the K-tap delay responses evaluated on the FFT grid
    A = np.exp(-2j * np.pi * np.outer(k, np.arange(max(KS))) / N) / np.sqrt(N)   # (N, Kmax)
    row = []
    for K in KS:
        Ak = A[:, :K]
        # least-squares projection of every realisation onto the K-tap subspace
        coef, *_ = np.linalg.lstsq(Ak, H.T, rcond=None)        # (K, 600)
        Hk = (Ak @ coef).T
        num = np.mean(np.abs(H - Hk) ** 2)
        den = np.mean(np.abs(H) ** 2)
        row.append(10 * np.log10(num / den))
    print("%-7s %6d %5d %5d | %s"
          % (prof, ds, nf, sp, " ".join("%+7.1f" % v for v in row)))

print()
print("NMSE of the best K-tap delay-domain approximation, in dB.")
print("A steeper curve means the delay-domain prior is more exploitable by a truncation branch.")
