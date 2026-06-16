suffix=10k
outFolder=results_${suffix}_3
chainlength=10000
seed=2026
trajectory_length=20000
tickseconds=8
nreplicates=20
nproc=36
warmup_steps=10000

####
bstr_mult=4.0
typea_prob=0.25
blockProb=0.995
prerecognition_seconds=2400
repair_seconds=360

mkdir -p ${outFolder}
mkdir -p configs

# micromamba run -n polychrom python scripts/gen_realistic_configs_variable_tick.py --chain ${chainlength} --num-chains ${nreplicates} --suffix ${suffix} --out-dir configs/ --seed ${seed} --tick-seconds ${tickseconds} --trajectory-length ${trajectory_length} --warmup-steps ${warmup_steps}

# # Run the two configs for comparison Exp.1 vs Baseline
# micromamba run -n polychrom python scripts/compare_config_chain_metrics.py \
#       --config1 configs/config1_${suffix}.yaml \
#       --config2 configs/config2_${suffix}.yaml \
#       --out-dir ${outFolder}/${suffix}_12 \
#       --label1 Baseline \
#       --label2 'Exp. 1'

# # Calculate transcription metrics
# micromamba run -n polychrom python scripts/gen_transcription_metrics.py \
#   --config configs/config1_${suffix}.yaml \
#   --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
#   --out-dir ${outFolder}/${suffix}_Baseline_TX

# # Cohesin moving-barrier evaluation (Banigan 2023): cohesin accumulation around
# # genes, transcription ON vs an auto-derived RNAPII-OFF control (ON-OFF difference)
# micromamba run -n polychrom python scripts/gen_cohesin_barrier_eval.py \
#   --config configs/config1_${suffix}.yaml \
#   --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
#   --out-dir ${outFolder}/${suffix}_Baseline_cohesin_barrier

# # Boundary strength sweep (boundary strength multiplier)
# micromamba run -n polychrom python scripts/sweep_rnapoff_boundary_strength_1d.py \
#   --config1 configs/config1_${suffix}.yaml \
#   --h5-config1 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
#   --multipliers 1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 \
#   --out-dir ${outFolder}/boundary_sweep_${suffix} \
#   --jobs $nproc

# Type A probability sweep (TYPE A probability vs lesion density)
micromamba run -n polychrom python scripts/gen_typea_density_grid.py \
  --config configs/config1_${suffix}.yaml \
  --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir ${outFolder}/typea_density_grid_${suffix} \
  --ta-step 0.05 --ta-lo 0.05 --ta-hi 0.5 \
  --spacings 7 8 10 11 13 14 17 20 25 33 50 100 \
  --bstr-mult ${bstr_mult} \
  --block-prob ${blockProb} \
  --repair-seconds ${repair_seconds} \
  --prerecognition-seconds ${prerecognition_seconds} \
  --jobs ${nproc}

# Block probability sweep (block probability vs lesion density)
micromamba run -n polychrom python scripts/gen_lesion_grid_and_heatmaps.py \
  --config configs/config1_${suffix}.yaml \
  --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir ${outFolder}/block_prob_grid_typea01_${suffix} \
  --p-step 0.005 --p-lo 0.98 --p-hi 1.0 \
  --spacings 7 8 10 11 13 14 17 20 25 33 50 100 \
  --repair-seconds ${repair_seconds} \
  --prerecognition-seconds ${prerecognition_seconds} \
  --bstr-mult ${bstr_mult} \
  --type-a-prob ${typea_prob} \
  --jobs ${nproc}

# recognition and repair search in seconds (seconds vs lesion density)
micromamba run -n polychrom python scripts/gen_lesion_timing_grids.py \
  --config configs/config1_${suffix}.yaml \
  --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir ${outFolder}/rec_rep_timing \
  --type-a-prob ${typea_prob} \
  --prerecog-lo 300 --prerecog-hi 7200 --prerecog-step 300 \
  --repair-fixed ${repair_seconds} \
  --repair-lo 100 --repair-hi 1000 --repair-step 50 \
  --prerecog-fixed ${prerecognition_seconds} \
  --spacings 7 8 10 11 13 14 17 20 25 33 50 100 \
  --bstr-mult ${bstr_mult} --block-prob ${blockProb} \
  --jobs ${nproc}

