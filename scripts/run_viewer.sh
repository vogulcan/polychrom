PYTHONPATH=. MAMBA_ROOT_PREFIX=/home/carlos/micromamba micromamba run -n openmm \
  python -m polychrom.pipelines.loop_extrusion.cli viewer \
  configs/degron/degron.yaml

PYTHONPATH=. MAMBA_ROOT_PREFIX=/home/carlos/micromamba micromamba run -n openmm \
  python -m polychrom.pipelines.loop_extrusion.cli viewer \
  configs/biophysical_calibrated.yaml runs/biophysical_calibrated
