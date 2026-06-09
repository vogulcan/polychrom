# Comparison: config2 vs config1

## 1D top-line
| metric | config2 | config1 | config1/config2 |
|---|---|---|---|
| mean loop | 141.0 | 126.4 | 0.90 |
| cohesin@CTCF | 3.27 | 2.64 | 0.80 |
| cohesin@genes | 25.85 | 27.02 | 1.05 |
| corner% | 0.1 | 0.0 | nan |
| global anchored-stripe% | 16.3 | 13.2 | 0.81 |
| boundary cross | 0.329 | 0.322 | 0.98 |
| boundary-crossing stripe share | 2.7% | 3.1% | 1.14 |

## P(s) — 1D bridge contacts
| s (kb) | config2 | config1 | config1/config2 |
|---|---|---|---|
| 5 | 4.80e-05 | 4.96e-05 | 1.03 |
| 10 | 2.91e-06 | 8.20e-06 | 2.82 |
| 20 | 6.51e-06 | 1.23e-05 | 1.90 |
| 50 | 8.48e-06 | 1.59e-05 | 1.87 |
| 100 | 7.84e-06 | 1.31e-05 | 1.67 |
| 150 | 6.43e-06 | 9.69e-06 | 1.51 |
| 200 | 4.71e-06 | 6.21e-06 | 1.32 |
| 300 | 2.60e-06 | 2.57e-06 | 0.99 |
| 500 | 6.22e-07 | 4.67e-07 | 0.75 |

## Plots
- [loop_length_compare.png](plots/config1_loop_length_compare.png)
- [Ps_1d_compare.png](plots/config1_Ps_1d_compare.png)