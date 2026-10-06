"""Analytic check of the sum-of-sinusoids tap generator (no per-burst renormalisation)."""
import numpy as np
from config import ChannelConfig, SYS
from channel import TDLChannel

rng = np.random.default_rng(1)
ch = TDLChannel(ChannelConfig(n_sinusoids=32), rng)
t = np.arange(200_000) * SYS.ofdm_symbol_duration
g = ch.tap_process(t)
print("shape", g.shape)
print("mean |g|^2 per tap:", np.round(g.real.var(axis=1) * 2, 4))
print("overall mean |g|^2: %.4f" % np.mean(np.abs(g) ** 2))
print("theoretical J0 check at lag = 1 symbol (fd=%.1f Hz)" % ch.fd)
from scipy.special import j0
lag = 1
emp = np.mean((g[:, :-lag] * np.conj(g[:, lag:])).real) / np.mean(np.abs(g) ** 2)
print("  empirical R(tau)/R(0) = %+.4f" % emp)
print("  J0(2 pi fd tau)       = %+.4f" % j0(2 * np.pi * ch.fd * SYS.ofdm_symbol_duration))
lag = 14 * 1  # 1 slot
emp = np.mean((g[:, :-lag] * np.conj(g[:, lag:])).real) / np.mean(np.abs(g) ** 2)
print("  1-slot lag: emp %+.4f vs J0 %+.4f" % (emp, j0(2 * np.pi * ch.fd * SYS.slot_duration)))
