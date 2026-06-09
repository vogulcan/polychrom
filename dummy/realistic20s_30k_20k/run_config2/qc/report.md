# Simulation QC report

- chain_length: 30000
- num_chains: 4

## Sanity (1D)
- n_frames: 20000
- n_lefs: 500
- any_nan: False
- any_out_of_range: False
- any_cross_chain_leg: False

## Loop length
- mean: 127.5  median: 91.0  p10: 17.0  p90: 283.0

## Cohesin classification
- corner: 0.1%   stripe: 9.8%   free: 90.2%
- asymmetry index: 0.002

## Boundary crossing
- mean: 0.758

## P(s) — 1D bridge contacts
- P(s) at separations: s=5:5.17e-05  s=10:4.02e-06  s=20:6.62e-06  s=50:1.07e-05  s=100:1.00e-05  s=150:7.64e-06  s=200:5.79e-06  s=300:2.49e-06  s=500:6.72e-07

## Plots
- [loop_length.png](plots/loop_length.png)
- [Ps_1d.png](plots/Ps_1d.png) -- 1D bridge-contact P(s)