"""3GPP TR 38.901 TDL channel model with sum-of-sinusoids time evolution.

The discrete-time channel impulse response of slot ``s``, symbol ``l``, sample ``d`` reads

    h[s, l, d] = sum_q  sqrt(P_q) * g_q(s, l) * delta(d - d_q)

where ``g_q`` is a unit-power time-correlated Rayleigh process whose autocorrelation follows
the classical Jakes model, ``R(tau) = J0(2*pi*f_d*tau)``.  Frequency selectivity is obtained
by rounding the tap delays to the nearest sample period ``T_s = 1/(N_fft * SCS)``, which is
the standard "sample-spaced channel model" used in link-level evaluations.

Compared with the more common approach of drawing i.i.d. Rayleigh taps per slot, the
sum-of-sinusoids construction makes the temporal correlation *physical*: the channel evolves
smoothly from slot to slot, which is exactly the structure that the proposed estimator
exploits.  Without it, any temporal-fusion gain would be fictitious.
"""

from __future__ import annotations

import numpy as np

from config import SYS, TDL_PROFILES, ChannelConfig

C_LIGHT = 299_792_458.0


def max_doppler_hz(velocity_kmh: float, fc_ghz: float = SYS.fc_ghz) -> float:
    """Maximum Doppler shift f_d = v * f_c / c."""
    return (velocity_kmh / 3.6) * (fc_ghz * 1e9) / C_LIGHT


class TDLChannel:
    """Sum-of-sinusoids (SoS) TDL fading generator.

    Parameters
    ----------
    cfg : ChannelConfig
    rng : np.random.Generator
    """

    def __init__(self, cfg: ChannelConfig, rng: np.random.Generator, fc_ghz: float = SYS.fc_ghz):
        self.cfg = cfg
        self.rng = rng
        self.fc_ghz = fc_ghz
        self.fd = max_doppler_hz(cfg.velocity_kmh, fc_ghz)

        delays_s = cfg.taps_seconds()
        self.tap_power = cfg.tap_powers_linear()
        ts = 1.0 / SYS.sample_rate
        cp_samples = max(1, int(round(SYS.n_fft * 0.0703)))
        # Fractional-delay discretisation.  Rounding the continuous tap delays to the sample
        # grid would make several independent taps collapse onto the same integer delay,
        # where they add *coherently* and inflate the average channel power (by up to 2x for
        # TDL-C at 100 ns and 3.84 MHz).  Splitting each tap linearly between the two
        # neighbouring integer delays preserves both the total power and the second-order
        # statistics of the continuous model.
        raw_d = np.clip(np.round(delays_s / ts).astype(int), 0, cp_samples - 1)
        frac = delays_s / ts - raw_d
        n_taps = cp_samples + 1
        self.tap_power_d = np.zeros(n_taps)
        for d, f, p in zip(raw_d, frac, self.tap_power):
            self.tap_power_d[d] += p * (1.0 - f)
            self.tap_power_d[d + 1] += p * f
        self.tap_ids = np.flatnonzero(self.tap_power_d > 1e-12)
        self.n_eff_taps = float(1.0 / np.sum(self.tap_power_d[self.tap_ids] ** 2))

        n_q = cfg.n_sinusoids
        # One independent fading process per *discretised* delay position.  This must be sized
        # by the length of the discretised profile, not by the number of continuous taps: with
        # a large delay spread the fractional-delay splitting creates more discrete positions
        # than there are continuous taps (e.g. 25 positions from 24 taps at tau_rms = 1 us), and
        # indexing with the smaller size would be wrong.
        q = n_taps
        # random phases / angles of arrival -> independent fading per tap
        self.alpha = self.rng.uniform(-np.pi, np.pi, size=(q, n_q))
        self.psi = self.rng.uniform(-np.pi, np.pi, size=(q, n_q))
        self.beta = self.rng.uniform(-np.pi, np.pi, size=(q, n_q))
        # symmetric Doppler spectrum (Jakes): omega = 2*pi*f_d*cos(theta)
        self.theta = self.rng.uniform(0.0, 2 * np.pi, size=(q, n_q))
    # ---------------------------------------------------------------- tap processes
    def tap_process(self, t: np.ndarray, tap_ids: np.ndarray | None = None) -> np.ndarray:
        """Unit-power time-correlated Rayleigh tap gains.

        Parameters
        ----------
        t : (T,) time instants in seconds, measured from an arbitrary epoch.
        tap_ids : which taps of the discrete profile to realise (default: all).

        Returns
        -------
        g : (len(tap_ids), T) complex tap gains, E|g|^2 = 1 per tap.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        ids = np.arange(len(self.tap_power)) if tap_ids is None else np.asarray(tap_ids)
        omega = 2 * np.pi * self.fd * np.cos(self.theta[ids])     # (Q, Nq)
        phase = omega[:, :, None] * t[None, None, :] + self.alpha[ids][:, :, None]
        # two independent quadrature components -> Rayleigh envelope
        re = np.cos(phase + self.psi[ids][:, :, None]).sum(axis=1)
        im = np.cos(phase + self.beta[ids][:, :, None]).sum(axis=1)
        # Analytic normalisation.  re and im are each a sum of n_q unit-amplitude cosines with
        # independent random phases, so Var(re) = Var(im) = n_q/2 and E|re + j im|^2 = n_q.
        # Dividing by sqrt(n_q/2) makes each quadrature unit-variance, i.e. a unit-power
        # complex process.
        return (re + 1j * im) / np.sqrt(self.cfg.n_sinusoids / 2.0)

    def slot_times(self, n_slots: int, start_slot: int = 0) -> np.ndarray:
        """(n_slots * n_symbols,) sampling instants for n_slots consecutive slots."""
        l = np.arange(SYS.n_symbols) * SYS.ofdm_symbol_duration
        s = np.arange(start_slot, start_slot + n_slots) * SYS.slot_duration
        return (s[:, None] + l[None, :]).ravel()

    # ---------------------------------------------------------------- frequency response
    def frequency_response(self, n_slots: int, start_slot: int = 0) -> np.ndarray:
        """Exact frequency-domain channel of ``n_slots`` consecutive slots.

        Returns
        -------
        H : (n_slots, N_f, N_t) complex
        """
        t = self.slot_times(n_slots, start_slot)                 # (n_slots*N_t,)
        g = self.tap_process(t, self.tap_ids)                    # (Q_act, n_slots*N_t)
        a = np.sqrt(self.tap_power_d[self.tap_ids])[:, None] * g  # (Q_act, n_slots*N_t)

        # sum over taps on the FFT grid:  H(f_k) = sum_q a_q exp(-j 2 pi k d_q / N)
        k = np.arange(SYS.n_fft)
        phase = np.exp(-2j * np.pi * np.outer(k, self.tap_ids) / SYS.n_fft)  # (N_f, Q)
        h = phase @ a                                            # (N_f, n_slots*N_t)
        h = h.reshape(SYS.n_fft, n_slots, SYS.n_symbols).transpose(1, 0, 2)
        return h


def generate_slot(H: np.ndarray, rng: np.random.Generator, snr_db: float,
                   n_fft: int = SYS.n_fft):
    """Build the received pilot grid for one slot.

    QPSK data symbols are placed on all non-pilot REs; pilots use unit-power QPSK as well.
    The returned ``y`` is the received grid **restricted to the pilot OFDM symbols**, with
    zeros elsewhere (this is the input the neural estimators see, so that no genie
    information about the data symbols leaks into the input).

    Parameters
    ----------
    H : (N_f, N_t) exact channel of one slot
    snr_db : receive SNR per resource element, defined as E|x|^2 / sigma_n^2
    """
    mask = SYS.pilot_mask()
    # unit-power QPSK payload / pilots
    bits_i = rng.integers(0, 2, size=H.shape) * 2 - 1
    bits_q = rng.integers(0, 2, size=H.shape) * 2 - 1
    x = (bits_i + 1j * bits_q) / np.sqrt(2.0)
    x = np.where(mask, x, x)  # pilots and data are statistically identical here

    sigma2 = 10 ** (-snr_db / 10.0)
    noise = np.sqrt(sigma2 / 2.0) * (rng.standard_normal(H.shape) +
                                     1j * rng.standard_normal(H.shape))
    y_full = H * x + noise
    y = np.where(mask, y_full, 0.0 + 0.0j)
    return y, x, sigma2


def pilot_ls_observation(y: np.ndarray, mask: np.ndarray, snr_db: float):
    """Least-squares channel observation at pilot REs and its noise variance.

    Because pilots are unit-power QPSK, the LS estimate is simply y / x_pilot with
    x_pilot in {+-1/sqrt(2) +- j/sqrt(2)}.  For the *input* of the neural estimators we use
    the raw received pilot values ``y`` (they carry the same information up to a known
    unit-modulus rotation), which avoids feeding the network a deterministic pilot sequence.
    """
    h_ls = np.where(mask, y, 0.0 + 0.0j)
    return h_ls


if __name__ == "__main__":
    """Self-check: channel power and the Jakes autocorrelation of the scalar process."""
    import math
    from scipy.special import j0

    rng = np.random.default_rng(0)
    ch = TDLChannel(ChannelConfig(), rng)
    print("profile      : %s, tau_rms = %.0f ns, v = %.0f km/h"
          % (ch.cfg.profile, ch.cfg.delay_spread_ns, ch.cfg.velocity_kmh))
    print("delay taps   : %d discrete taps at positions %s" % (len(ch.tap_ids), ch.tap_ids))
    print("tap power    : %s" % np.round(ch.tap_power_d[ch.tap_ids], 4))
    print("effective taps: %.2f" % ch.n_eff_taps)
    print("f_d          : %.1f Hz ; slot = %.3f ms" % (ch.fd, SYS.slot_duration * 1e3))

    H = ch.frequency_response(n_slots=400)
    print("E|H|^2       : %.4f" % np.mean(np.abs(H) ** 2))
    print()
    print("temporal correlation of the scalar channel (empirical vs Jakes):")
    h = H.mean(axis=1)                                  # (n_slots, N_t) scalar process
    for lag_slots in (1, 2, 3):
        a = h[:-lag_slots].ravel()
        b = h[lag_slots:].ravel()
        emp = abs(np.mean(a * np.conj(b)) / np.sqrt(np.mean(np.abs(a) ** 2)
                                                    * np.mean(np.abs(b) ** 2)))
        theo = abs(j0(2 * math.pi * ch.fd * lag_slots * SYS.slot_duration))
        print("  %d slot(s): %.3f (Jakes %.3f)" % (lag_slots, emp, theo))
