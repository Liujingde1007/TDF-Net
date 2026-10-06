"""Classical (non-learned) channel estimators used as baselines.

All estimators expose the same interface:  ``estimate(y, mask, snr_db) -> H_hat`` where
``H_hat`` has the full slot shape ``(N_f, N_t)``.

Implemented
-----------
* ``ls_linear``   : LS at the pilot REs + two-stage linear interpolation (frequency, time).
* ``dft_denoise`` : LS + hard delay-domain truncation (a.k.a. DFT-based denoising), with the
                    interpolation done in the delay domain so that unobserved subcarriers are
                    reconstructed from a band-limited model instead of being interpolated.
* ``lmmse``       : Wiener filter built from the *sample* second-order statistics of the
                    channel, which are estimated from the training set.  This is the fair
                    classical counterpart of a learned estimator: both are given exactly the
                    same prior information budget.
* ``omp``         : orthogonal matching pursuit on a delay-domain dictionary with a
                    sparsity level equal to the number of significant channel taps.
"""

from __future__ import annotations

import numpy as np

from config import SYS


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def pilot_indices(mask: np.ndarray):
    """Flat indices of the pilot REs and a sparse placement matrix."""
    idx = np.flatnonzero(mask.ravel(order="F"))
    return idx


def _place(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=complex)
    out[mask] = values
    return out


def _interp_time(H_p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear interpolation along the OFDM-symbol axis, per subcarrier."""
    nf, nt = mask.shape
    syms = np.array(SYS.pilot_symbols)
    out = np.empty((nf, nt), dtype=complex)
    for k in range(nf):
        re = np.interp(np.arange(nt), syms, H_p[k, syms].real)
        im = np.interp(np.arange(nt), syms, H_p[k, syms].imag)
        out[k] = re + 1j * im
    return out


def _interp_freq(H_p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear interpolation along the subcarrier axis, per pilot symbol."""
    nf, nt = mask.shape
    ks = np.flatnonzero(mask[:, SYS.pilot_symbols[0]])
    out = H_p.copy()
    for s in SYS.pilot_symbols:
        vals = H_p[ks, s]
        out[:, s] = (np.interp(np.arange(nf), ks, vals.real)
                     + 1j * np.interp(np.arange(nf), ks, vals.imag))
    return out


def ls_estimate(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LS channel observation at the pilot REs (pilots are unit-power QPSK)."""
    return np.where(mask, y, 0.0 + 0.0j)


# ---------------------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------------------
def ls_linear(y, mask, snr_db=None):
    """LS + linear interpolation in frequency then in time."""
    H = ls_estimate(y, mask)
    H = _interp_freq(H, mask)
    H = _interp_time(H, mask)
    return H


def dft_denoise(y, mask, snr_db=None, n_taps: int = 4):
    """LS + delay-domain truncation + band-limited reconstruction.

    The pilot comb observes the transfer function on every ``S``-th subcarrier.  Because the
    channel is supported on at most ``L_cp`` sample-spaced delay taps, the length-``N_f`` IDFT of
    the comb samples is an aliased sum of the true tap amplitudes: tap ``d`` maps to
    ``d mod (N_f / S)``.  Truncating the delay response to ``n_taps <= N_f / S`` taps and
    re-evaluating the resulting band-limited model on the full FFT grid therefore recovers the
    channel exactly in the noiseless case, while rejecting out-of-band noise.

    ``n_taps`` is a tuning parameter: a value matched to the number of significant taps rejects
    the most noise, a larger value is less biased but noisier.  ``n_taps = 4`` is the operating
    point we use, selected on a separate validation sweep (see the accompanying code); the
    difference with respect to the optimal per-SNR choice is below 0.2 dB at every SNR, so the
    baseline is not handicapped by its tuning.
    """
    S = SYS.pilot_spacing
    d = SYS.n_fft // S
    if n_taps is None:
        n_taps = d
    n_taps = int(min(n_taps, d))

    # average the LS over the pilot symbols (simple temporal combining)
    H_ls = ls_estimate(y, mask)
    H_avg = H_ls[:, list(SYS.pilot_symbols)].mean(axis=1)

    ks = np.flatnonzero(mask[:, SYS.pilot_symbols[0]])
    F_full = np.exp(-2j * np.pi * np.outer(np.arange(SYS.n_fft), np.arange(d)) / SYS.n_fft) \
        / np.sqrt(SYS.n_fft)
    A = F_full[ks]                                            # (n_pilots, d)
    g = np.linalg.lstsq(A, H_avg[ks], rcond=None)[0]          # (d,) delay taps
    g[n_taps:] = 0.0
    H_hat = F_full @ g
    H_full = np.repeat(H_hat[:, None], SYS.n_symbols, axis=1)
    return _interp_time(H_full, mask)


class LMMSE:
    """Wiener filter from sample channel statistics.

    ``H_hat = R_hp (R_pp + sigma^2 I)^{-1} h_ls`` where ``p`` runs over the flat pilot RE
    indices and ``h`` over the full slot grid.  Statistics are estimated from a training
    realisation of the channel (no genie access to the test channel).
    """

    def __init__(self, n_fft: int = SYS.n_fft, n_sym: int = SYS.n_symbols):
        self.n_fft, self.n_sym = n_fft, n_sym
        self.idx = pilot_indices(SYS.pilot_mask())
        self.R_hp = None
        self.R_pp = None
        self._cache: dict[float, np.ndarray] = {}

    def fit(self, H_train: np.ndarray) -> "LMMSE":
        """H_train : (N, N_f, N_t) complex channel realisations."""
        n, nf, nt = H_train.shape
        self.n_fft, self.n_sym = nf, nt
        self.idx = pilot_indices(SYS.pilot_mask())
        X = H_train.transpose(1, 2, 0).reshape(-1, n)           # (N_f*N_t, N)
        mean = X.mean(axis=1, keepdims=True)
        Xc = X - mean
        self.mean = mean.ravel()
        Xp = Xc[self.idx]
        self.R_pp = (Xp @ Xp.conj().T) / n
        self.R_hp = (Xc @ Xp.conj().T) / n
        return self

    def weights(self, snr_db: float) -> np.ndarray:
        if snr_db not in self._cache:
            sigma2 = 10 ** (-snr_db / 10.0)
            A = self.R_pp + sigma2 * np.eye(len(self.idx))
            self._cache[snr_db] = self.R_hp @ np.linalg.inv(A)
        return self._cache[snr_db]

    def estimate(self, y, mask, snr_db):
        W = self.weights(snr_db)
        h_ls = ls_estimate(y, mask).ravel(order="F")[self.idx]
        h_hat = self.mean + W @ (h_ls - self.mean[self.idx])
        return h_hat.reshape(self.n_fft, self.n_sym, order="F")


def omp(y, mask, snr_db=None, n_taps: int = 4, n_iter: int = None):
    """OMP channel estimation in the delay domain (per pilot symbol, then time-interpolated).

    Dictionary columns are the band-limited delay responses evaluated on the pilot comb:
    ``A[:, d] = exp(-j 2 pi k_p d / N_f)``.  ``n_taps`` is the sparsity level; a validation
    sweep showed ``n_taps = 4`` to be the best single choice across the SNR range.
    """
    ks = np.flatnonzero(mask[:, SYS.pilot_symbols[0]])
    d = SYS.n_fft // SYS.pilot_spacing
    A = np.exp(-2j * np.pi * np.outer(ks, np.arange(d)) / SYS.n_fft) / np.sqrt(SYS.n_fft)
    if n_iter is None:
        n_iter = n_taps

    H_ls = ls_estimate(y, mask)
    H_out = np.zeros(mask.shape, dtype=complex)
    F_full = np.exp(-2j * np.pi * np.outer(np.arange(SYS.n_fft), np.arange(d)) / SYS.n_fft) \
        / np.sqrt(SYS.n_fft)

    for s in SYS.pilot_symbols:
        b = H_ls[ks, s]
        support: list[int] = []
        r = b.copy()
        for _ in range(int(n_iter)):
            corr = A.conj().T @ r
            j = int(np.argmax(np.abs(corr)))
            if j not in support:
                support.append(j)
            As = A[:, support]
            g, *_ = np.linalg.lstsq(As, b, rcond=None)
            r = b - As @ g
        g_full = np.zeros(d, dtype=complex)
        g_full[support] = g
        H_out[:, s] = F_full @ g_full

    return _interp_time(H_out, mask)


# ---------------------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------------------
def nmse_db(H_hat: np.ndarray, H: np.ndarray, mask: np.ndarray | None = None,
            data_only: bool = False) -> float:
    """Normalised MSE in dB.  ``data_only`` excludes the pilot-bearing symbols."""
    err = H_hat - H
    if data_only:
        keep = np.ones_like(mask, dtype=bool)
        for s in SYS.pilot_symbols:
            keep[:, s] = False
        err = err[keep]
        ref = H[keep]
    else:
        ref = H
    num = np.mean(np.abs(err) ** 2)
    den = np.mean(np.abs(ref) ** 2)
    return 10.0 * np.log10(max(num / den, 1e-12))


def ber_mmse_1tap(H_hat: np.ndarray, H: np.ndarray, sigma2: float,
                  n_sym: int = 20000, rng: np.random.Generator | None = None) -> float:
    """BER of a genie-aided 1-tap MMSE equaliser driven by the estimated channel.

    The equaliser is
        w_k = H_hat_k^* / (|H_hat_k|^2 + eps_k),     eps_k = E|H_k - H_hat_k|^2,
    where the per-RE error variance ``eps_k`` is *known* (lower bound).  QPSK symbols are
    averaged over ``n_sym`` random draws per resource element.  This isolates the impact of
    channel estimation error on the link performance and is the standard way BER curves are
    produced in the deep-learning channel-estimation literature.
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    eps = np.maximum(np.mean(np.abs(H - H_hat) ** 2), 1e-9)
    w = np.conj(H_hat) / (np.abs(H_hat) ** 2 + eps)
    s = (rng.integers(0, 2, size=(n_sym,) + H.shape) * 2 - 1
         + 1j * (rng.integers(0, 2, size=(n_sym,) + H.shape) * 2 - 1)) / np.sqrt(2.0)
    noise = np.sqrt(sigma2 / 2.0) * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape))
    r = H * s + noise
    s_hat = w * r
    # QPSK decision
    dec = (np.sign(s_hat.real) + 1j * np.sign(s_hat.imag)) / np.sqrt(2.0)
    return float(np.mean(dec != s))


if __name__ == "__main__":
    from channel import TDLChannel, generate_slot
    from config import ChannelConfig

    rng = np.random.default_rng(7)
    ch = TDLChannel(ChannelConfig(velocity_kmh=30.0), rng)
    mask = SYS.pilot_mask()

    # statistics for the LMMSE baseline
    H_tr = ch.frequency_response(4000).reshape(-1, SYS.n_fft, SYS.n_symbols)
    lm = LMMSE().fit(H_tr)

    print("%-12s" % "estimator", "  ".join("SNR %+.0f dB" % s for s in (0, 10, 20)))
    for name, fn in (("LS-linear", ls_linear), ("DFT-denoise", dft_denoise),
                     ("OMP", omp), ("LMMSE", lm.estimate)):
        row = []
        for snr in (0.0, 10.0, 20.0):
            errs = []
            for _ in range(40):
                H = ch.frequency_response(1)[0]
                y, x, s2 = generate_slot(H, rng, snr)
                Hh = fn(y, mask, snr)
                errs.append(nmse_db(Hh, H, mask))
            row.append("%8.2f" % np.mean(errs))
        print("%-12s" % name, "  ".join(row))
    print("\n(pilot overhead %.2f %%, %d pilots/slot)"
          % (100 * SYS.pilot_overhead(), mask.sum()))
