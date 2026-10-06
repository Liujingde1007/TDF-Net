"""Tune the DFT-denoise baseline over its truncation length so the comparison is fair."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from config import SYS, EVAL, ChannelConfig
from channel import TDLChannel, generate_slot
import baselines as bl

rng = np.random.default_rng(17)
mask = SYS.pilot_mask()
ch = TDLChannel(ChannelConfig(profile="TDL-C", delay_spread_ns=EVAL.delay_spread_ns,
                              velocity_kmh=100.0), rng)
print("comb period S=%d -> alias-free delay range N/S=%d taps; CP=%d samples; effective taps=%.2f"
      % (SYS.pilot_spacing, SYS.n_fft // SYS.pilot_spacing,
         int(round(SYS.n_fft * 0.0703)), ch.n_eff_taps))
print("\nDFT-denoise NMSE (dB) vs truncation length n_taps:")
print("%8s" % "SNR" + "".join("%9s" % ("n=%d" % n) for n in (2, 3, 4, 6, 8, 12, 16, 32)))
best = {}
for snr in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
    row = []
    for n in (2, 3, 4, 6, 8, 12, 16, 32):
        e = []
        for _ in range(60):
            H = ch.frequency_response(1)[0]
            y, x, _ = generate_slot(H, rng, snr)
            ls = np.where(mask, y / x, 0j)
            e.append(bl.nmse_db(bl.dft_denoise(ls, mask, snr, n_taps=n), H, mask))
        row.append(np.mean(e))
    best[snr] = int(np.argmin(row))
    print("%+8.1f" % snr + "".join("%9.2f" % v for v in row))

print("\nOMP NMSE (dB) vs sparsity level:")
print("%8s" % "SNR" + "".join("%9s" % ("K=%d" % k) for k in (1, 2, 3, 4, 5, 6)))
for snr in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
    row = []
    for k in (1, 2, 3, 4, 5, 6):
        e = []
        for _ in range(60):
            H = ch.frequency_response(1)[0]
            y, x, _ = generate_slot(H, rng, snr)
            ls = np.where(mask, y / x, 0j)
            e.append(bl.nmse_db(bl.omp(ls, mask, snr, n_taps=k), H, mask))
        row.append(np.mean(e))
    print("%+8.1f" % snr + "".join("%9.2f" % v for v in row))
