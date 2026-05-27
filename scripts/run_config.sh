config=/home/ogulcan/polychrom/configs/postUV.yaml
run_name=postUV
mkdir -p "runs/${run_name}"
PYTHONPATH=. micromamba run -n polychrom \
  python -m polychrom.pipelines.loop_extrusion.cli all "${config}" \
  > "runs/${run_name}/pipeline_run.log" 2>&1

config=/home/ogulcan/polychrom/configs/preUV.yaml
run_name=preUV
mkdir -p "runs/${run_name}"
PYTHONPATH=. micromamba run -n polychrom \
  python -m polychrom.pipelines.loop_extrusion.cli all "${config}" \
  > "runs/${run_name}/pipeline_run.log" 2>&1
