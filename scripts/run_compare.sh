PYTHONPATH=. micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli compare --cutoffs 2 --out runs_test/5k_1_2 runs_test/config1_5k runs_test/config2_5k

PYTHONPATH=. micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli compare --cutoffs 2 --out runs_test/5k_1_3 runs_test/config1_5k runs_test/config3_5k

PYTHONPATH=. micromamba run -n polychrom python -m polychrom.pipelines.loop_extrusion.cli compare --cutoffs 2 --out runs_test/5k_2_3 runs_test/config2_5k runs_test/config3_5k

