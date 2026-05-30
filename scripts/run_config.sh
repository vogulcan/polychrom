
for config in config1.yaml config2.yaml; do
  run_name=$(basename "${config}" .yaml)
  output_path="runs/${run_name}"
  mkdir -p "${output_path}"
  PYTHONPATH=. micromamba run -n polychrom \
    python -m polychrom.pipelines.loop_extrusion.cli all "configs/${config}" "${output_path}" \
    > "${output_path}/pipeline_run.log" 2>&1
done