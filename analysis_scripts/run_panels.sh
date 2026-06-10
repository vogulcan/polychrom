suffix=20k
chainlength=20000
seed=42
trajectory_length=20000
nreplicates=10

micromamba run -n polychrom python scripts/gen_realistic_configs_variable_tick.py --chain ${chainlength} --num-chains ${nreplicates} --suffix ${suffix} --out-dir configs/ --seed ${seed} --tick-seconds 8 --trajectory-length ${trajectory_length}

micromamba run -n polychrom python scripts/compare_config_chain_metrics.py \
      --config1 configs/config1_${suffix}.yaml \
      --config2 configs/config2_${suffix}.yaml \
      --out-dir results/${suffix}_12

micromamba run -n polychrom python scripts/sweep_rnapoff_boundary_strength_1d.py \
  --config1 configs/config1_${suffix}.yaml \
  --h5-config1 results/${suffix}_12/config1/LEFPositions.h5 \
  --out-dir results/boundary_sweep_${suffix}