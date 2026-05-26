config=${1:-configs/degron/control.yaml}
run_name=${2:-$(basename "${config%.yaml}")}

mkdir -p "runs/${run_name}"
PYTHONPATH=. micromamba run -n openmm \
  python -m polychrom.pipelines.loop_extrusion.cli all "${config}" \
  > "runs/${run_name}/pipeline_run.log" 2>&1
