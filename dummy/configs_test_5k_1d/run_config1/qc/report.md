# Simulation QC report

- chain_length: 5000
- num_chains: 1

## Sanity (1D)
- n_frames: 10000
- n_lefs: 20
- any_nan: False
- any_out_of_range: False
- any_cross_chain_leg: False

## Loop length
- mean: 126.4  median: 87.0  p10: 17.0  p90: 294.0

## Cohesin classification
- corner: 0.0%   stripe: 13.2%   free: 86.8%
- asymmetry index: 0.041

## Boundary crossing
- mean: 0.322

## P(s) — 1D bridge contacts
- P(s) at separations: s=5:4.96e-05  s=10:8.20e-06  s=20:1.23e-05  s=50:1.59e-05  s=100:1.31e-05  s=150:9.69e-06  s=200:6.21e-06  s=300:2.57e-06  s=500:4.67e-07

## RNAPII
- present: 100.0%   mean#: 71.35   max#: 104
- realized elongation: 5.96 kb/min (1.985 sites/tick)
- state mix %: {'POISED': 0.8165187374652602, 'PAUSED': 28.538212824641878, 'ELONGATING': 26.39628768480639, 'TERMINATING': 26.211849192100537, 'STALLED': 18.037131560985934}

## Plots
- [loop_length.png](plots/loop_length.png)
- [Ps_1d.png](plots/Ps_1d.png) -- 1D bridge-contact P(s)