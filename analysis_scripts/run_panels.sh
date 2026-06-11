suffix=5k
chainlength=5000
seed=42
trajectory_length=20000 # (200k steps with 8s/tick = 44.4 hours)
tickseconds=8
nreplicates=10
nproc=36

micromamba run -n polychrom python scripts/gen_realistic_configs_variable_tick.py --chain ${chainlength} --num-chains ${nreplicates} --suffix ${suffix} --out-dir configs/ --seed ${seed} --tick-seconds ${tickseconds} --trajectory-length ${trajectory_length}

# Run the two configs for comparison Exp.1 vs Baseline
micromamba run -n polychrom python scripts/compare_config_chain_metrics.py \
      --config1 configs/config1_${suffix}.yaml \
      --config2 configs/config2_${suffix}.yaml \
      --out-dir results/${suffix}_12 \
      --label1 Baseline \
      --label2 'Exp. 1'

# Boundary strength sweep (boundary strength multiplier) # ideal: 3.0
micromamba run -n polychrom python scripts/sweep_rnapoff_boundary_strength_1d.py \
  --config1 configs/config1_${suffix}.yaml \
  --h5-config1 results/${suffix}_12/Baseline/LEFPositions.h5 \
  --multipliers 1,1.5,2,2.5,3,3.5,4,4.5,5 \
  --out-dir results/boundary_sweep_${suffix} \
  --jobs $nproc

# Type A probability sweep (TYPE A probability vs lesion density) # ideal: 0.10
micromamba run -n polychrom python scripts/gen_typea_density_grid.py \
  --config configs/config1_${suffix}.yaml \
  --h5 results/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir results/typea_density_grid_${suffix} \
  --ta-step 0.05 --ta-lo 0.0 --ta-hi 0.5 \
  --spacings 10 11 13 14 17 20 25 33 50 100 \
  --bstr-mult 3 \
  --block-prob 0.975 \
  --repair-seconds 360 \
  --prerecognition-seconds 900 \
  --jobs ${nproc}

# Block probability sweep (block probability vs lesion density) # ideal: 0.975
micromamba run -n polychrom python scripts/gen_lesion_grid_and_heatmaps.py \
  --config configs/config1_${suffix}.yaml \
  --h5 results/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir results/block_prob_grid_typea01_${suffix} \
  --p-step 0.0125 --p-lo 0.85 --p-hi 1.0 \
  --spacings 10 11 13 14 17 20 25 33 50 100 \
  --repair-seconds 360 \
  --prerecognition-seconds 900 \
  --bstr-mult 3 \
  --type-a-prob 0.10 \
  --jobs ${nproc}

# recognition and repair search in seconds (seconds vs lesion density) # ideal: prerecognition = 900s, repair = 360s
micromamba run -n polychrom python scripts/gen_lesion_timing_grids.py \
  --config configs/config1_${suffix}.yaml \
  --h5 results/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir results/rec_rep_timing \
  --type-a-prob 0.1 \
  --prerecog-lo 300 --prerecog-hi 7200 --prerecog-step 300 \
  --repair-fixed 360 \
  --repair-lo 100 --repair-hi 1000 --repair-step 100 \
  --prerecog-fixed 900 \
  --spacings 10 11 13 14 17 20 25 33 50 100 \
  --bstr-mult 3 --block-prob 0.975 \
  --jobs ${nproc}

