config=/home/ogulcan/polychrom/configs/degron/control.yaml
run_name=control_v2
mkdir -p "runs/${run_name}"
PYTHONPATH=. micromamba run -n polychrom \
  python -m polychrom.pipelines.loop_extrusion.cli all "${config}" \
  > "runs/${run_name}/pipeline_run.log" 2>&1

config=/home/ogulcan/polychrom/configs/degron/degron.yaml
run_name=degron_v2
mkdir -p "runs/${run_name}"
PYTHONPATH=. micromamba run -n polychrom \
  python -m polychrom.pipelines.loop_extrusion.cli all "${config}" \
  > "runs/${run_name}/pipeline_run.log" 2>&1
