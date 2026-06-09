# Comparison: config2 vs config1

## 1D top-line
| metric | config2 | config1 | config1/config2 |
|---|---|---|---|
| mean loop | 127.5 | 119.0 | 0.93 |
| cohesin@CTCF | 49.64 | 43.17 | 0.87 |
| cohesin@genes | 132.17 | 544.03 | 4.12 |
| corner% | 0.1 | 0.1 | 0.67 |
| global anchored-stripe% | 9.8 | 8.5 | 0.87 |
| boundary cross | 0.758 | 0.727 | 0.96 |
| boundary-crossing stripe share | 4.4% | 4.3% | 0.99 |

## P(s) — 1D bridge contacts
| s (kb) | config2 | config1 | config1/config2 |
|---|---|---|---|
| 5 | 5.17e-05 | 5.31e-05 | 1.03 |
| 10 | 4.02e-06 | 8.35e-06 | 2.08 |
| 20 | 6.62e-06 | 1.26e-05 | 1.90 |
| 50 | 1.07e-05 | 1.71e-05 | 1.59 |
| 100 | 1.00e-05 | 1.40e-05 | 1.40 |
| 150 | 7.64e-06 | 9.68e-06 | 1.27 |
| 200 | 5.79e-06 | 6.72e-06 | 1.16 |
| 300 | 2.49e-06 | 2.49e-06 | 1.00 |
| 500 | 6.72e-07 | 4.24e-07 | 0.63 |

## Plots
- [loop_length_compare.png](plots/config1_loop_length_compare.png)
- [Ps_1d_compare.png](plots/config1_Ps_1d_compare.png)