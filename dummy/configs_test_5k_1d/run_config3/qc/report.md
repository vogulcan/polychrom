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
- mean: 141.7  median: 101.0  p10: 19.0  p90: 323.0

## Cohesin classification
- corner: 0.1%   stripe: 21.3%   free: 78.6%
- asymmetry index: 0.043

## Boundary crossing
- mean: 0.215

## P(s) — 1D bridge contacts
- P(s) at separations: s=5:4.64e-05  s=10:5.17e-06  s=20:7.33e-06  s=50:8.28e-06  s=100:8.61e-06  s=150:6.04e-06  s=200:4.83e-06  s=300:3.87e-06  s=500:7.33e-07

## Plots
- [loop_length.png](plots/loop_length.png)
- [Ps_1d.png](plots/Ps_1d.png) -- 1D bridge-contact P(s)