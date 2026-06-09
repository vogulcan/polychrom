# Comparison: config2 vs config3

## 1D top-line
| metric | config2 | config3 | config3/config2 |
|---|---|---|---|
| mean loop | 141.0 | 141.7 | 1.00 |
| cohesin@CTCF | 3.27 | 4.30 | 1.31 |
| cohesin@genes | 25.85 | 25.40 | 0.98 |
| corner% | 0.1 | 0.1 | 1.39 |
| global anchored-stripe% | 16.3 | 21.3 | 1.31 |
| boundary cross | 0.329 | 0.215 | 0.65 |
| boundary-crossing stripe share | 2.7% | 2.5% | 0.91 |

## P(s) — 1D bridge contacts
| s (kb) | config2 | config3 | config3/config2 |
|---|---|---|---|
| 5 | 4.80e-05 | 4.64e-05 | 0.97 |
| 10 | 2.91e-06 | 5.17e-06 | 1.78 |
| 20 | 6.51e-06 | 7.33e-06 | 1.13 |
| 50 | 8.48e-06 | 8.28e-06 | 0.98 |
| 100 | 7.84e-06 | 8.61e-06 | 1.10 |
| 150 | 6.43e-06 | 6.04e-06 | 0.94 |
| 200 | 4.71e-06 | 4.83e-06 | 1.03 |
| 300 | 2.60e-06 | 3.87e-06 | 1.49 |
| 500 | 6.22e-07 | 7.33e-07 | 1.18 |

## Plots
- [loop_length_compare.png](plots/config3_loop_length_compare.png)
- [Ps_1d_compare.png](plots/config3_Ps_1d_compare.png)