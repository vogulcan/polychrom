# Comparison: config2 vs config3

## 1D top-line
| metric | config2 | config3 | config3/config2 |
|---|---|---|---|
| mean loop | 127.5 | 135.5 | 1.06 |
| cohesin@CTCF | 49.64 | 75.72 | 1.53 |
| cohesin@genes | 132.17 | 125.90 | 0.95 |
| corner% | 0.1 | 0.2 | 2.29 |
| global anchored-stripe% | 9.8 | 14.8 | 1.51 |
| boundary cross | 0.758 | 0.636 | 0.84 |
| boundary-crossing stripe share | 4.4% | 4.0% | 0.92 |

## P(s) — 1D bridge contacts
| s (kb) | config2 | config3 | config3/config2 |
|---|---|---|---|
| 5 | 5.17e-05 | 4.99e-05 | 0.96 |
| 10 | 4.02e-06 | 4.07e-06 | 1.01 |
| 20 | 6.62e-06 | 6.45e-06 | 0.97 |
| 50 | 1.07e-05 | 1.05e-05 | 0.98 |
| 100 | 1.00e-05 | 9.71e-06 | 0.97 |
| 150 | 7.64e-06 | 7.84e-06 | 1.03 |
| 200 | 5.79e-06 | 5.21e-06 | 0.90 |
| 300 | 2.49e-06 | 2.73e-06 | 1.10 |
| 500 | 6.72e-07 | 6.67e-07 | 0.99 |

## Plots
- [loop_length_compare.png](plots/config3_loop_length_compare.png)
- [Ps_1d_compare.png](plots/config3_Ps_1d_compare.png)