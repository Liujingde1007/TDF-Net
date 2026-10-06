"""Push the delay spread until the truncation itself becomes the binding constraint.

The previous probe showed that a K-tap delay-domain model is essentially exact for
tau_rms <= 1 us at N = 256 or 512, because the OFDM symbol is 71 us long and the channel
therefore has few significant taps relative to the band.  For the fixed truncation branch to be
*decisive* rather than merely sufficient, the truncation length L_d = 16 must start to bite,
i.e. the number of significant taps must approach L_d.

This script searches the delay-spread range where that happens and reports, for each
configuration:
  * the number of discrete taps and the number carrying > 2 % of the power,
  * the best K-tap NMSE for K = 4, 8, 16, 24, 32 (dB), and
  * the coherence bandwidth.

A configuration where K = 16 is substantially worse than K = 32 is one where truncating to
L_d = 16 discards real channel energy, and only there can the branch be expected to matter.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from config import SYS, TDL_PROFILES, ChannelConfig
from channel import TDLChannel

CONFIGS = [
    ("TDL-C", 1000, 256, 15.0),
    ("TDL-C", 2000, 256, 15.0),
    ("TDL-C", 4000, 256, 15.0),
    ("TDL-C", 8000, 256, 15.0),
    ("TDL-C", 20000, 256, 15.0),
    ("TDL-A", 2000, 256, 15.0),
    ("TDL-A", 8000, 256, 15.0),
]
KS = [2, 4, 8, 16, 24, 32, 48, 64]

print("%-7s %7s %5s %6s %6s %6s | %s"
      % ("profile", "DS[ns]", "Nfft", "cp", "#taps", "#>2%",
         " ".join("K=%-6d" % k for k in KS)))
rows = []
for prof, ds, nf, scs in CONFIGS:
    SYS.n_fft, SYS.scs_khz = nf, scs
    cp = int(round(nf * 0.0703))
    ts = 1.0 / SYS.sample_rate
    delays, pdb = TDL_PROFILES[prof]
    p = 10 ** (pdb / 10.0)
    p /= p.sum()
    d_cont = delays * ds * 1e-9
    raw = np.clip(np.round(d_cont / ts).astype(int), 0, cp)
    frac = d_cont / ts - raw
    pr = np.zeros(cp + 2)
    for d, f, pi in zip(raw, frac, p):
        pr[d] += pi * (1 - f)
        pr[d + 1] += pi * f
    ids = np.flatnonzero(pr > 1e-12)
    n_taps, n_gt2 = len(ids), int((pr[ids] > 0.02).sum())

    ch = TDLChannel(ChannelConfig(profile=prof, delay_spread_ns=ds, velocity_kmh=3.0),
                    np.random.default_rng(11))
    H = ch.frequency_response(500)[:, :, 0]
    k = np.arange(nf)
    A = np.exp(-2j * np.pi * np.outer(k, np.arange(max(KS))) / nf) / np.sqrt(nf)
    vals = []
    for K in KS:
        if K > nf:
            vals.append(float("nan"))
            continue
        Ak = A[:, :K]
        coef, *_ = np.linalg.lstsq(Ak, H.T, rcond=None)
        num = np.mean(np.abs(H - (Ak @ coef).T) ** 2)
        vals.append(10 * np.log10(num / np.mean(np.abs(H) ** 2)))
    print("%-7s %7d %5d %6d %6d %6d | %s"
          % (prof, ds, nf, cp, n_taps, n_gt2, " ".join("%+7.1f" % v for v in vals)))
    rows.append((prof, ds, n_taps, vals))

print()
print("Interpretation:")
print("  if K=16 is close to K=32, truncating at L_d=16 loses almost nothing -> the branch")
print("  cannot help; if K=16 is much worse than K=32, the channel has real energy beyond the")
print("  truncation and the branch becomes decisive.")
