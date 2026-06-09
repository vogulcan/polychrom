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
- mean: 141.0  median: 97.0  p10: 19.0  p90: 312.0

## Cohesin classification
- corner: 0.1%   stripe: 16.3%   free: 83.7%
- asymmetry index: 0.042

## Boundary crossing
- mean: 0.329

## P(s) — 1D bridge contacts
- P(s) at separations: s=5:4.80e-05  s=10:2.91e-06  s=20:6.51e-06  s=50:8.48e-06  s=100:7.84e-06  s=150:6.43e-06  s=200:4.71e-06  s=300:2.60e-06  s=500:6.22e-07

## Plots
- [loop_length.png](plots/loop_length.png)
- [Ps_1d.png](plots/Ps_1d.png) -- 1D bridge-contact P(s)