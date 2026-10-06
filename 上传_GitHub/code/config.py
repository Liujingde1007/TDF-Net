"""Global simulation configuration.

System model: pilot-symbol-aided OFDM downlink, 3GPP TR 38.901 TDL channel models.

All arrays use the following conventions
----------------------------------------
H        : (N_f, N_t) complex frequency-domain channel of one slot
           N_f = number of subcarriers, N_t = number of OFDM symbols
y        : (N_f, N_t) complex received grid
pilot    : (N_f, N_t) bool mask, True where a pilot resource element (RE) is present
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np


# --------------------------------------------------------------------------------------
# 3GPP TR 38.901 Table 7.7.5-1/-2/-3 : TDL-A / TDL-B / TDL-C normalized tap delays & powers
# --------------------------------------------------------------------------------------
# delays are normalized to the RMS delay spread; powers are in dB.
TDL_PROFILES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    # TDL-A, 23 taps, NLOS
    "TDL-A": (
        np.array([0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.5350, 0.7306, 0.7501,
                  0.7259, 1.0701, 0.9138, 0.8431, 0.7434, 1.1690, 0.9519, 1.1883, 0.8799,
                  0.8047, 0.8403, 0.7525, 0.7928, 0.7928]),
        np.array([-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6,
                  -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3, -18.9, -16.6,
                  -19.9, -19.9]),
    ),
    # TDL-B, 23 taps, NLOS
    "TDL-B": (
        np.array([0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3319, 0.4586, 0.4656,
                  0.6104, 0.6731, 0.7275, 0.8003, 0.8069, 0.8690, 0.9077, 0.9401, 0.9996,
                  1.0764, 1.0840, 1.1684, 1.1846, 1.2076]),
        np.array([0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3.0, -8.9,
                  -9.0, -4.8, -5.7, -7.5, -6.9, -6.2, -9.3, -10.7, -10.0, -12.0,
                  -11.8, -12.0]),
    ),
    # TDL-C, 24 taps, NLOS
    "TDL-C": (
        np.array([0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584,
                  0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105, 4.2589, 4.6003,
                  5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523]),
        np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9, -7.4, -7.1, -10.7,
                  -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8, -17.1, -16.0,
                  -15.7, -21.6, -22.8]),
    ),
    # TDL-D, 13 taps, LOS (first tap is the LOS ray)
    "TDL-D": (
        np.array([0.0000, 0.0000, 0.0359, 0.0719, 0.1249, 0.1790, 0.2653, 0.3408, 0.4162,
                  0.4964, 0.5850, 0.6894, 0.8062]),
        np.array([-0.2, -13.5, -18.8, -21.0, -22.8, -17.9, -20.1, -21.9, -23.7, -22.0,
                  -24.0, -25.3, -24.8]),
    ),
    # TDL-E, 14 taps, LOS
    "TDL-E": (
        np.array([0.0000, 0.0000, 0.0487, 0.1062, 0.1642, 0.2096, 0.2299, 0.3192, 0.4169,
                  0.4336, 0.5373, 0.6337, 0.7083, 0.8070]),
        np.array([-1.7, -13.0, -17.5, -19.8, -21.2, -18.7, -17.9, -19.9, -21.3, -24.0,
                  -23.4, -24.3, -25.2, -25.6]),
    ),
}


@dataclass
class SystemConfig:
    """OFDM numerology.

    Chosen so that the composite channel is genuinely frequency selective *and* the
    temporal decorrelation across a slot is significant, i.e. the two structures that the
    proposed estimator exploits are both physically present:

    * ``n_fft = 256`` with 15 kHz SCS -> 3.84 MHz occupied bandwidth, sample period 260 ns.
      With an RMS delay spread of 100 ns (TDL-C) the discrete channel has ~2.7 effective
      sample-spaced taps, so the delay-domain sparsity prior is meaningful.
    * ``pilot_symbols = (2, 5, 8, 11)`` is the standard 5G-NR tracking-RS density: a
      frequency comb of period 8 on four OFDM symbols, i.e. 4.69 % pilot overhead.
    * At 3.5 GHz and 100 km/h the Jakes correlation between adjacent slots is ~0.38, so a
      single-slot estimator is fundamentally limited and temporal fusion pays off.
    """

    n_fft: int = 256                # number of subcarriers
    n_symbols: int = 14             # OFDM symbols per slot (normal CP)
    scs_khz: float = 15.0           # subcarrier spacing [kHz]
    fc_ghz: float = 3.5             # carrier frequency [GHz]
    pilot_symbols: tuple = (2, 5, 8, 11)   # OFDM symbol indices carrying the pilot comb
    pilot_spacing: int = 8          # pilot subcarrier spacing (comb size)

    @property
    def scs_hz(self) -> float:
        return self.scs_khz * 1e3

    @property
    def sample_rate(self) -> float:
        return self.n_fft * self.scs_hz

    @property
    def ofdm_symbol_duration(self) -> float:
        """Total OFDM symbol duration including normal CP (7.29 % of 1/SCS for 15 kHz)."""
        cp_ratio = 0.0703  # 144/2048 for 15 kHz SCS
        return (1.0 + cp_ratio) / self.scs_hz

    @property
    def slot_duration(self) -> float:
        return self.n_symbols * self.ofdm_symbol_duration

    def pilot_mask(self) -> np.ndarray:
        """(N_f, N_t) boolean pilot mask."""
        mask = np.zeros((self.n_fft, self.n_symbols), dtype=bool)
        for s in self.pilot_symbols:
            mask[:: self.pilot_spacing, s] = True
        return mask

    def pilot_overhead(self) -> float:
        return float(self.pilot_mask().sum()) / (self.n_fft * self.n_symbols)


@dataclass
class ChannelConfig:
    """3GPP TDL channel parameters."""

    profile: str = "TDL-C"
    delay_spread_ns: float = 100.0     # RMS delay spread [ns]
    velocity_kmh: float = 100.0        # UE velocity [km/h]
    n_sinusoids: int = 32              # sum-of-sinusoids components per tap
    normalize_power: bool = True       # normalize total tap power to unity

    def taps_seconds(self) -> np.ndarray:
        delays, _ = TDL_PROFILES[self.profile]
        return delays * self.delay_spread_ns * 1e-9

    def tap_powers_linear(self) -> np.ndarray:
        _, powers_db = TDL_PROFILES[self.profile]
        p = 10 ** (powers_db / 10.0)
        return p / p.sum() if self.normalize_power else p


@dataclass
class TrainConfig:
    """Dataset generation and training hyper-parameters."""

    seq_len: int = 4                  # number of consecutive slots presented to the network
    velocity_group: tuple = (20.0, 140.0)  # training velocities sampled log-uniformly
    snr_train_db: tuple = (-5.0, 25.0)     # training SNR range
    n_train: int = 4000               # number of training *sequences*
    n_val: int = 600
    n_test: int = 600
    batch_size: int = 32
    epochs: int = 45
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 20240517
    n_delay_taps: int = 16            # delay-domain branch dimension
    base_ch: int = 32                 # base channel width of the U-Net encoder
    grad_clip: float = 1.0
    profile: str = "TDL-C"            # training channel profile
    delay_spread_ns: float = 100.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["velocity_group"] = list(self.velocity_group)
        d["snr_train_db"] = list(self.snr_train_db)
        return d


@dataclass
class EvalConfig:
    """Test-set definition for the main SNR sweep and the mobility sweep."""

    snr_db: tuple = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
    velocity_kmh: tuple = (3.0, 100.0, 200.0)
    n_realizations: int = 200         # slots evaluated per (snr, velocity) point
    profile: str = "TDL-C"
    delay_spread_ns: float = 100.0
    n_test_sequences: int = 400       # sequences in the held-out test set


SYS = SystemConfig()
CHAN = ChannelConfig()
TRAIN = TrainConfig()
EVAL = EvalConfig()


if __name__ == "__main__":
    print("pilot overhead : %.3f %%" % (100 * SYS.pilot_overhead()))
    print("N pilots/slot  : %d" % SYS.pilot_mask().sum())
    print("symbol duration: %.3f us" % (SYS.ofdm_symbol_duration * 1e6))
    print("slot duration  : %.3f ms" % (SYS.slot_duration * 1e3))
    print("sample rate    : %.2f MHz" % (SYS.sample_rate / 1e6))
    import math
    c = 299_792_458.0
    for v in EVAL.velocity_kmh:
        fd = (v / 3.6) * (SYS.fc_ghz * 1e9) / c
        print("v=%6.1f km/h -> f_d = %8.1f Hz, correlation over 1 slot = %.3f"
              % (v, fd, __import__("scipy.special").special.j0(2 * math.pi * fd * SYS.slot_duration)))
