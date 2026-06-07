configs_arr=(
  /home/ogulcan/polychrom/configs_test/config1_5k.yaml
  /home/ogulcan/polychrom/configs_test/config2_5k.yaml
  # /home/ogulcan/polychrom/configs_test/config3_5k.yaml
)

for config in "${configs_arr[@]}"; do
  run_name=$(basename "${config}" .yaml)
  output_path="runs_test/${run_name}"
  mkdir -p "${output_path}"
  PYTHONPATH=. micromamba run -n polychrom \
    python -m polychrom.pipelines.loop_extrusion.cli all ${config} "${output_path}" \
    > "${output_path}/pipeline_run.log" 2>&1
done