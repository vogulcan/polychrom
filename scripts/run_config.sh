config=configs/degron/control.yaml
mkdir -p runs/control
micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli all $config > runs/control/pipeline_run.log 2>&1
