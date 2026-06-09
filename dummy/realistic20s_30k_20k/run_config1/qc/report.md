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
- mean: 119.0  median: 84.0  p10: 16.0  p90: 265.0

## Cohesin classification
- corner: 0.1%   stripe: 8.5%   free: 91.4%
- asymmetry index: 0.002

## Boundary crossing
- mean: 0.727

## P(s) — 1D bridge contacts
- P(s) at separations: s=5:5.31e-05  s=10:8.35e-06  s=20:1.26e-05  s=50:1.71e-05  s=100:1.40e-05  s=150:9.68e-06  s=200:6.72e-06  s=300:2.49e-06  s=500:4.24e-07

## RNAPII
- present: 100.0%   mean#: 1528.75   max#: 1666
- realized elongation: 5.97 kb/min (1.991 sites/tick)
- state mix %: {'POISED': 1.0591202107179132, 'PAUSED': 29.848438078391517, 'ELONGATING': 22.572898712919738, 'TERMINATING': 27.023707596777953, 'STALLED': 19.49583540119288}

## Plots
- [loop_length.png](plots/loop_length.png)
- [Ps_1d.png](plots/Ps_1d.png) -- 1D bridge-contact P(s)