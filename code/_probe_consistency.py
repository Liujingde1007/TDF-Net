"""Point-by-point consistency check between the generated features and the true channel."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from config import SYS
from data import make_sequences
from channel import TDLChannel, generate_slot
from config import ChannelConfig

mask = SYS.pilot_mask()
print("pilot mask: %d pilots, per-symbol counts %s"
      % (mask.sum(), np.unique(mask.sum(axis=0))))
rng = np.random.default_rng(3)
SNR = 10.0

# --- direct generation (reference path) ---
ch = TDLChannel(ChannelConfig(profile="TDL-C", delay_spread_ns=100.0, velocity_kmh=100.0), rng)
H = ch.frequency_response(1)[0]
print("E|H|^2 = %.4f" % np.mean(np.abs(H) ** 2))
y, x, sigma2 = generate_slot(H, rng, SNR)
print("sigma2 = %.5f" % sigma2)
print("E|x|^2 on pilots = %.4f   mean x on pilots = %s"
      % (np.mean(np.abs(x[mask]) ** 2), np.round(x[mask][:3], 4)))
ls_direct = y / x
_c = (np.abs(np.mean(ls_direct[mask] * np.conj(H[mask])))
      / np.sqrt(np.mean(np.abs(ls_direct[mask]) ** 2) * np.mean(np.abs(H[mask]) ** 2)))
print("direct corr(y/x, H) on pilots = %.4f" % _c)

# --- via the feature path used by the network ---
X, Y, _ = make_sequences(200, 1, rng, (100.0, 100.0), (SNR, SNR), profile="TDL-C",
                         fixed_velocity=100.0, fixed_snr=SNR)
H2 = Y[:, 0] + 1j * Y[:, 1]
sigma = np.sqrt(10 ** (-SNR / 10.0))
obs = X[:, 0, 0] + 1j * X[:, 0, 1]          # y / sigma
ls = X[:, 0, 3] + 1j * X[:, 0, 4]           # LS / sigma
print("\nfeature-path check over %d sequences:" % len(X))
print("  E|y/sigma|^2 on pilots = %.4f  (expect (E|H|^2+sigma^2)/sigma^2 = %.2f)"
      % (np.mean(np.abs(obs[:, mask]) ** 2),
         (np.mean(np.abs(H2) ** 2) + sigma ** 2) / sigma ** 2))
print("  E|H|^2 = %.4f" % np.mean(np.abs(H2) ** 2))
def corr(a, b):
    num = np.mean(a * np.conj(b))
    return np.abs(num) / np.sqrt(np.mean(np.abs(a) ** 2) * np.mean(np.abs(b) ** 2))
print("  corr(y/sigma on pilots, H) = %.4f" % corr(obs[:, mask], H2[:, mask]))
print("  corr(LS/sigma on pilots, H) = %.4f" % corr(ls[:, mask], H2[:, mask]))
print("  corr(LS/sigma, H) all REs   = %.4f" % corr(ls.ravel(), H2.ravel()))
print("  theoretical corr = sqrt(SNR/(SNR+1)) with SNR=%.1f -> %.4f"
      % (10 ** (SNR / 10), np.sqrt(10 ** (SNR / 10) / (10 ** (SNR / 10) + 1))))

# is the LS plane zero where it should not be?
print("\n  nonzero REs in LS plane   : %d (expect %d)" % (np.count_nonzero(ls), mask.sum()))
print("  LS plane power on pilots  : %.4f" % np.mean(np.abs(ls[:, mask]) ** 2))
print("  LS plane power off pilots : %.4f" % np.mean(np.abs(ls[:, ~mask]) ** 2))
