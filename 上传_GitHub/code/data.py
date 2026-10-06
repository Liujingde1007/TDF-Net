"""Dataset generation for learned OFDM channel estimation.

Observation model
-----------------
For every sequence we draw

* a random UE velocity (log-uniform in the training range),
* a random SNR (uniform in the training range),
* an independent small-scale fading realisation of the 3GPP TDL profile,

and produce ``M`` consecutive slots.  The target is the exact frequency-domain channel of
the **last** slot.  Slot ``i`` of the sequence is presented to the network as four feature
planes after de-rotation by the known pilot symbols and scaling by the known noise standard
deviation ``sigma``:

    plane 0 : Re( H_LS / sigma )     where H_LS = y / x on the pilot REs, 0 elsewhere
    plane 1 : Im( H_LS / sigma )
    plane 2 : pilot mask             (1 on pilot REs)
    plane 3 : normalised SNR in dB   (constant plane, snr_db / 20)

Why de-rotation matters
-----------------------
The sufficient statistic for pilot-aided estimation is ``y / x``, not the raw received sample
``y``.  Reserving the raw sample would leave a random unit-modulus rotation between the input
and the target — the QPSK pilot value, which differs from slot to slot and from realisation to
realisation — so the same channel would map to differently rotated inputs and no estimator
could separate the channel from the pilot sequence.  De-rotating by the known pilot sequence
removes this ambiguity exactly and is what a real receiver does.

Scaling by ``1/sigma`` is the standard noise-whitening conditioning trick: it maps the
observation to a unit-noise scale for every SNR, which is what allows a single network to
operate over a 30 dB SNR range.  The SNR itself is passed separately and embedded by the
decoder (and also supplied as a constant plane) so that the estimator can adapt its effective
shrinkage to the noise level.

Note on the input: the network sees the **pilot LS observation on the pilot-bearing OFDM
symbols**, with no interpolation, thresholding or other preconditioning.  In particular it is
not first interpolated onto the full grid, so no hand-tuned pre-processing can limit the
high-SNR behaviour (the failure mode reported for SRCNN-style estimators).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config import SYS, TRAIN, ChannelConfig, SystemConfig
from channel import TDLChannel, generate_slot

# number of input planes produced by ``_features``: Re(LS/sigma), Im(LS/sigma), mask, SNR
N_FEAT = 4


def _features(H: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """(4, N_f, N_t) float32 features for one slot at a given SNR.

    The pilot symbols are known to the receiver, so the sufficient statistic is the
    least-squares channel observation ``y / x`` on the pilot REs, *not* the raw received
    sample.  Using the raw sample would leave a random unit-modulus rotation per realisation
    (the QPSK pilot value) between the input and the target, which no estimator can undo
    because the pilot values differ from slot to slot.  We therefore de-rotate exactly.

    Planes
    ------
    0, 1 : Re/Im of the delay-restricted LS estimate, scaled by 1/sigma
    2    : pilot mask
    3    : constant plane carrying the normalised SNR in dB (1 when SNR is unknown)
    """
    mask = SYS.pilot_mask()
    y, x, sigma2 = generate_slot(H, rng, snr_db)
    sigma = np.sqrt(sigma2)
    ls = np.where(mask, y / x, 0.0 + 0.0j)     # exact de-rotation by the known pilots
    f = np.stack([
        ls.real / sigma,
        ls.imag / sigma,
        mask.astype(np.float64),
        np.full(H.shape, snr_db / 20.0),
    ])
    return f.astype(np.float32)


def make_sequences(n_seq: int, seq_len: int, rng: np.random.Generator,
                   velocity_range: tuple, snr_range: tuple,
                   profile: str = TRAIN.profile,
                   delay_spread_ns: float = TRAIN.delay_spread_ns,
                   fixed_velocity: float | None = None,
                   fixed_snr: float | None = None,
                   progress: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``n_seq`` sequences.

    Returns
    -------
    X    : (n_seq, M, 4, N_f, N_t) float32
    Y    : (n_seq, 2, N_f, N_t)    float32   (Re/Im of the exact channel of the last slot)
    meta : (n_seq, 2)              float32   (velocity [km/h], SNR [dB])
    """
    nf, nt = SYS.n_fft, SYS.n_symbols
    X = np.empty((n_seq, seq_len, N_FEAT, nf, nt), dtype=np.float32)
    Y = np.empty((n_seq, 2, nf, nt), dtype=np.float32)
    meta = np.empty((n_seq, 2), dtype=np.float32)

    log_lo, log_hi = np.log(velocity_range[0]), np.log(velocity_range[1])
    for i in range(n_seq):
        v = fixed_velocity if fixed_velocity is not None else \
            float(np.exp(rng.uniform(log_lo, log_hi)))
        snr = fixed_snr if fixed_snr is not None else float(rng.uniform(*snr_range))
        ch = TDLChannel(ChannelConfig(profile=profile, delay_spread_ns=delay_spread_ns,
                                      velocity_kmh=v), rng)
        H = ch.frequency_response(seq_len)                    # (M, N_f, N_t)
        for m in range(seq_len):
            X[i, m] = _features(H[m], rng, snr)
        Y[i, 0] = H[-1].real
        Y[i, 1] = H[-1].imag
        meta[i] = (v, snr)
        if progress and (i + 1) % max(1, n_seq // 20) == 0:
            print("    generated %d/%d" % (i + 1, n_seq), flush=True)
    return X, Y, meta


class ChannelSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, meta: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.meta = torch.from_numpy(meta)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.meta[i, 1]      # features, target, SNR


def build_datasets(seed: int = TRAIN.seed, verbose: bool = True):
    """Training and validation sets (same distribution)."""
    rng = np.random.default_rng(seed)
    if verbose:
        print("generating training set (%d sequences, M=%d) ..." % (TRAIN.n_train, TRAIN.seq_len))
    Xtr, Ytr, Mtr = make_sequences(TRAIN.n_train, TRAIN.seq_len, rng,
                                   TRAIN.velocity_group, TRAIN.snr_train_db, progress=verbose)
    if verbose:
        print("generating validation set (%d sequences) ..." % TRAIN.n_val)
    Xva, Yva, Mva = make_sequences(TRAIN.n_val, TRAIN.seq_len, rng,
                                   TRAIN.velocity_group, TRAIN.snr_train_db)
    return (Xtr, Ytr, Mtr), (Xva, Yva, Mva)


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    t0 = time.time()
    X, Y, meta = make_sequences(20, TRAIN.seq_len, rng, TRAIN.velocity_group,
                                TRAIN.snr_train_db)
    dt = time.time() - t0
    print("X", X.shape, X.dtype, "Y", Y.shape, "meta", meta.shape)
    print("generation speed: %.1f sequences/s -> %.1f min for %d"
          % (20 / dt, 20 / dt and TRAIN.n_train / (20 / dt) / 60, TRAIN.n_train))
    print("feature means :", np.round(X.mean(axis=(0, 1, 3, 4)), 3))
    print("feature stds  :", np.round(X.std(axis=(0, 1, 3, 4)), 3))
    print("target power  : %.4f" % np.mean(Y ** 2))
    print("velocity range: %.1f - %.1f km/h" % (meta[:, 0].min(), meta[:, 0].max()))
    print("SNR range     : %.1f - %.1f dB" % (meta[:, 1].min(), meta[:, 1].max()))
