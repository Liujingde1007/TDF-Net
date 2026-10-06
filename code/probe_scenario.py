"""Scenario probe: quantify how frequency selective each candidate configuration is.

Purpose
-------
The paper reports that the fixed delay-domain branch is inert at tau_rms = 100 ns with a
3.84 MHz band.  A reviewer will ask what happens when the channel actually has many significant
delay taps.  Before training anything we measure, for each candidate configuration:

  * the number of discrete (sample-spaced) taps,
  * the number of taps carrying more than 2 % of the power,
  * the effective number of taps, 1 / sum(p_i^2), which is the quantity that controls how much
    a delay-domain truncation can gain,
  * the 3 dB coherence bandwidth, from the frequency correlation function,
  * the pilot overhead of the comb, and the alias-free delay range it supports.

A configuration is only interesting for the branch question if the effective tap count is
substantially larger than in the reference configuration, while the pilot comb still observes
the whole delay support unaliased.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from config import SYS, TDL_PROFILES, ChannelConfig
from channel import TDLChannel

rng_global = np.random.default_rng(0)


def analyse(profile, ds_ns, n_fft, scs_khz, pilot_spacing, pilot_symbols):
    SYS.n_fft = n_fft
    SYS.scs_khz = scs_khz
    SYS.pilot_spacing = pilot_spacing
    SYS.pilot_symbols = tuple(pilot_symbols)

    ts = 1.0 / SYS.sample_rate
    delays, powers_db = TDL_PROFILES[profile]
    p = 10 ** (powers_db / 10.0)
    p = p / p.sum()
    d_cont = delays * ds_ns * 1e-9
    # fractional-delay discretisation, exactly as in channel.py
    raw = np.clip(np.round(d_cont / ts).astype(int), 0, 200)
    frac = d_cont / ts - raw
    prof = np.zeros(raw.max() + 2)
    for d, f, pi in zip(raw, frac, p):
        prof[d] += pi * (1 - f)
        prof[d + 1] += pi * f
    ids = np.flatnonzero(prof > 1e-12)
    n_taps = len(ids)
    n_gt2 = int((prof[ids] > 0.02).sum())
    eff = float(1.0 / np.sum(prof[ids] ** 2))

    # frequency correlation -> coherence bandwidth
    ch = TDLChannel(ChannelConfig(profile=profile, delay_spread_ns=ds_ns, velocity_kmh=3.0),
                    np.random.default_rng(1))
    H = ch.frequency_response(400)[:, :, 0]                 # (400, N_f), slow fading
    Hf = H - H.mean(axis=1, keepdims=True)
    R = (Hf.conj().T @ Hf) / Hf.shape[0]                    # (N_f, N_f)
    r = np.abs(np.diag(R, k=0))
    lag = 0
    for k in range(1, n_fft // 2):
        c = np.mean(np.abs(np.diag(R, k=k))) / np.mean(r)
        if c < 0.707:
            break
        lag = k
    bcoh = lag * scs_khz * 1e-3                             # MHz

    S = pilot_spacing
    alias_free = n_fft // S
    overhead = 100.0 * (len(pilot_symbols) * (n_fft // S)) / (n_fft * SYS.n_symbols)
    ok = alias_free >= n_taps
    return dict(n_taps=n_taps, n_gt2=n_gt2, eff=eff, bcoh_mhz=bcoh,
                alias_free=alias_free, overhead=overhead, ok=ok,
                ts_ns=ts * 1e9, cp=int(round(n_fft * 0.0703)))


CANDIDATES = [
    # profile, ds_ns, n_fft, scs, pilot spacing, pilot symbols
    ("TDL-C", 100, 256, 15.0, 8, (2, 5, 8, 11)),      # reference configuration
    ("TDL-C", 300, 256, 15.0, 8, (2, 5, 8, 11)),
    ("TDL-C", 600, 256, 15.0, 8, (2, 5, 8, 11)),
    ("TDL-C", 1000, 256, 15.0, 8, (2, 5, 8, 11)),
    ("TDL-A", 600, 256, 15.0, 8, (2, 5, 8, 11)),
    ("TDL-A", 1000, 256, 15.0, 8, (2, 5, 8, 11)),
    ("TDL-C", 600, 512, 15.0, 16, (2, 5, 8, 11)),
    ("TDL-C", 1000, 512, 15.0, 16, (2, 5, 8, 11)),
]

print("%-7s %6s %5s %5s %5s %6s %8s %8s %9s %7s %5s"
      % ("profile", "DS[ns]", "Nfft", "SCS", "Sp", "Ts[ns]", "#taps", "#>2%", "eff.taps",
         "Bcoh[MHz]", "ok"))
for prof, ds, nf, scs, sp, ps in CANDIDATES:
    r = analyse(prof, ds, nf, scs, sp, ps)
    print("%-7s %6d %5d %5.0f %5d %6.1f %8d %8d %9.2f %7.2f %5s"
          % (prof, ds, nf, scs, sp, r["ts_ns"], r["n_taps"], r["n_gt2"], r["eff"],
             r["bcoh_mhz"], "yes" if r["ok"] else "ALIAS"))

print()
print("Reference configuration has eff.taps = 1.65.  A configuration is useful for the branch")
print("question when eff.taps is much larger AND the pilot comb is alias-free (ok = yes).")
print()
print("tap power profiles (top 8 taps):")
for prof, ds, nf, scs, sp, ps in CANDIDATES:
    SYS.n_fft, SYS.scs_khz = nf, scs
    ts = 1.0 / SYS.sample_rate
    delays, pdb = TDL_PROFILES[prof]
    p = 10 ** (pdb / 10.0)
    p /= p.sum()
    d_cont = delays * ds * 1e-9
    raw = np.clip(np.round(d_cont / ts).astype(int), 0, 200)
    frac = d_cont / ts - raw
    pr = np.zeros(raw.max() + 2)
    for d, f, pi in zip(raw, frac, p):
        pr[d] += pi * (1 - f)
        pr[d + 1] += pi * f
    ids = np.flatnonzero(pr > 1e-12)
    top = ids[np.argsort(-pr[ids])][:8]
    print("  %-6s DS=%4d N=%3d : %s"
          % (prof, ds, nf, np.round(pr[top], 4)))
