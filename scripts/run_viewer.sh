rm -r preUV
rm -r postUV
rm -r control
rm -r degron

# PYTHONPATH=. MAMBA_ROOT_PREFIX=/home/carlos/micromamba micromamba run -n polychrom \
#   python -m polychrom.pipelines.loop_extrusion.cli viewer \
#   /home/carlos/Clone/polychrom/uv_configs/postUV.yaml

# PYTHONPATH=. MAMBA_ROOT_PREFIX=/home/carlos/micromamba micromamba run -n polychrom \
#   python -m polychrom.pipelines.loop_extrusion.cli viewer \
#   /home/carlos/Clone/polychrom/uv_configs/preUV_1.yaml

# PYTHONPATH=. MAMBA_ROOT_PREFIX=/home/carlos/micromamba micromamba run -n polychrom \
#   python -m polychrom.pipelines.loop_extrusion.cli viewer \
#   /home/carlos/Clone/polychrom/degron_configs/degron.yaml

PYTHONPATH=. MAMBA_ROOT_PREFIX=/home/carlos/micromamba micromamba run -n polychrom \
  python -m polychrom.pipelines.loop_extrusion.cli viewer \
  /home/carlos/Clone/polychrom/degron_configs/control.yaml