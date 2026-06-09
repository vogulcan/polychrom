suffix=5k
outfolder=runs_test

configs_arr=(
  /home/ogulcan/polychrom/configs_test/config1_${suffix}.yaml
  /home/ogulcan/polychrom/configs_test/config2_${suffix}.yaml
  /home/ogulcan/polychrom/configs_test/config3_${suffix}.yaml
)

for config in "${configs_arr[@]}"; do
  run_name=$(basename "${config}" .yaml)
  output_path="${outfolder}/${run_name}"
  mkdir -p "${output_path}"
  PYTHONPATH=. micromamba run -n polychrom \
    python -m polychrom.pipelines.loop_extrusion.cli all ${config} "${output_path}" \
    > "${output_path}/pipeline_run.log" 2>&1
done

PYTHONPATH=. micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli compare --cutoffs 2 --out ${outfolder}/${suffix}_1_2 ${outfolder}/config1_${suffix} ${outfolder}/config2_${suffix}

PYTHONPATH=. micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli compare --cutoffs 2 --out ${outfolder}/${suffix}_1_3 ${outfolder}/config1_${suffix} ${outfolder}/config3_${suffix}

PYTHONPATH=. micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli compare --cutoffs 2 --out ${outfolder}/${suffix}_2_3 ${outfolder}/config2_${suffix} ${outfolder}/config3_${suffix}

