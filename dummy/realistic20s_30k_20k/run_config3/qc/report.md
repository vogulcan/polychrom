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
- mean: 135.5  median: 94.0  p10: 18.0  p90: 302.0

## Cohesin classification
- corner: 0.2%   stripe: 14.8%   free: 85.0%
- asymmetry index: 0.002

## Boundary crossing
- mean: 0.636

## P(s) — 1D bridge contacts
- P(s) at separations: s=5:4.99e-05  s=10:4.07e-06  s=20:6.45e-06  s=50:1.05e-05  s=100:9.71e-06  s=150:7.84e-06  s=200:5.21e-06  s=300:2.73e-06  s=500:6.67e-07

## Plots
- [loop_length.png](plots/loop_length.png)
- [Ps_1d.png](plots/Ps_1d.png) -- 1D bridge-contact P(s)