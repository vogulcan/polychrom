
# PYTHONPATH=. micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli compare --cutoffs 2 --out runs_test/15k_1_2 runs_test/config1_15k runs_test/config2_15k

# PYTHONPATH=. micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli compare --cutoffs 2 --out runs_test/15k_1_3 runs_test/config1_15k runs_test/config3_15k

PYTHONPATH=. micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli compare --cutoffs 2 --out runs_test/15k_2_3 runs_test/config2_15k runs_test/config3_15k
