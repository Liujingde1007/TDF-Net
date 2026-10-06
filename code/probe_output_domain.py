"""Diagnose the output-domain constraint's classical base estimate h0.

Compares, on real data:
  * the amplitude of the true channel and of h0,
  * the NMSE of h0,
  * the NMSE of the same projection applied per pilot symbol instead of after averaging,
  * the NMSE of the reference DFT-denoise estimator on the identical observation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import SYS, TRAIN, ChannelConfig
from data import make_sequences
from models import TDFNet
from channel import TDLChannel, generate_slot
import baselines as bl

SNR = 15.0
rng = np.random.default_rng(5)
X, Y, _ = make_sequences(120, 4, rng, (100.0, 100.0), (SNR, SNR), profile="TDL-C",
                         fixed_velocity=100.0, fixed_snr=SNR)
H = Y[:, 0] + 1j * Y[:, 1]

net = TDFNet(n_fft=SYS.n_fft, n_sym=SYS.n_symbols,
             n_pilot_sym=len(SYS.pilot_symbols), n_taps=4, seq_len=4,
             constraint="output").eval()
torch.manual_seed(0)
x = torch.from_numpy(X)
nt = SYS.n_symbols
pk, ps = net.pilot_k, net.pilot_sym_idx

with torch.no_grad():
    zr = x[:, -1, 0] * x[:, -1, 2]
    zi = x[:, -1, 1] * x[:, -1, 2]

    def project_and_rebuild(zr_, zi_, average_over_pilots):
        a = zr_[:, pk, :]
        b = zi_[:, pk, :]
        if average_over_pilots:
            a = a[:, :, ps].mean(2)
            b = b[:, :, ps].mean(2)
        gr = torch.einsum("lp,bp->bl", net.proj_re, a) - torch.einsum("lp,bp->bl", net.proj_im, b)
        gi = torch.einsum("lp,bp->bl", net.proj_re, b) + torch.einsum("lp,bp->bl", net.proj_im, a)
        g = torch.stack([gr, gi], 1).unsqueeze(-1).expand(-1, -1, -1, nt)
        return net._reconstruct(g)

    h0_avg = project_and_rebuild(zr, zi, True).numpy()

h0c_avg = h0_avg[:, 0] + 1j * h0_avg[:, 1]


def nmse(a, b):
    return 10 * np.log10(np.mean(np.abs(a - b) ** 2) / np.mean(np.abs(b) ** 2))


print("SNR = %.0f dB, v = 100 km/h, TDL-C" % SNR)
print("E|H|^2               = %.4f" % np.mean(np.abs(H) ** 2))
print("E|h0 (avg pilots)|^2 = %.4f" % np.mean(np.abs(h0c_avg) ** 2))
print()
print("NMSE of h0 (pilot-averaged delay-domain LS), 4 taps : %+.2f dB" % nmse(h0c_avg, H))
print("NMSE of h0 per OFDM symbol (variation not modelled) : %+.2f dB"
      % nmse(np.repeat(h0c_avg.mean(axis=2, keepdims=True), SYS.n_symbols, axis=2), H))

# classical references on the identical observation
sigma = np.sqrt(10 ** (-SNR / 10.0))
ch = TDLChannel(ChannelConfig(profile="TDL-C", delay_spread_ns=100.0, velocity_kmh=100.0),
                np.random.default_rng(9))
mask = SYS.pilot_mask()
for n_taps in (2, 4, 8, 16):
    e = []
    for i in range(40):
        Ht = ch.frequency_response(1)[0]
        y, xs, _ = generate_slot(Ht, ch.rng, SNR)
        ls = np.where(mask, y / xs, 0j)
        e.append(bl.nmse_db(bl.dft_denoise(ls, mask, SNR, n_taps=n_taps), Ht, mask))
    print("DFT-denoise n_taps=%-3d           : %+.2f dB" % (n_taps, np.mean(e)))
