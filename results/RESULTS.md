# Experiment results

Auto-generated from `results/eval_main.json`. Every number below comes directly from the
evaluation script.

## Trained models

| Model | tag | params | best val NMSE (dB) | epoch |
|---|---|---|---|---|
| TDF-Net | tdfnet_M4 | 753298 | -13.37 | 29 |
| ResCNN | rescnn_M1 | 257762 | -13.28 | 30 |
| ResCNN+GRU | rescnn_M4gru | 373410 | -13.38 | 29 |
| ChannelNet | channelnet | 14114 | -6.12 | 26 |

## NMSE, TDL-C profile (mean over 120 realisations, dB)

| Estimator | v=3 | v=30 | v=100 | v=200 |
|---|---|---|---|---|
| LS-linear | -14.24 | -13.99 | -12.02 | -10.06 |
| DFT-denoise | -25.68 | -16.47 | -7.86 | -3.36 |
| OMP | -19.10 | -17.86 | -14.35 | -11.69 |
| LMMSE | -6.16 | -5.83 | -4.30 | -2.40 |
| ChannelNet | -7.09 | -7.18 | -7.46 | -6.24 |
| TDF-Net | -17.76 | -17.74 | -15.85 | -11.03 |

## BER (TDL-C), at the highest evaluated SNR

| Estimator | v=30 | v=100 |
|---|---|---|
| LS-linear | 2.05e-03 | 1.03e-02 |
| DFT-denoise | 1.61e-02 | 1.45e-01 |
| OMP | 1.66e-03 | 9.58e-03 |
| LMMSE | 2.30e-01 | 2.89e-01 |
| ChannelNet | 6.07e-03 | 2.06e-02 |
| TDF-Net | 1.91e-03 | 5.99e-03 |

## Complexity

| Estimator | params | ms/slot |
|---|---|---|
| LS-linear | -- | 1.759 |
| DFT-denoise | -- | 2.025 |
| OMP | -- | 2.452 |
| LMMSE | -- | 0.135 |
| ChannelNet | 14114 | 0.186 |
| TDF-Net | 753298 | 2.246 |
