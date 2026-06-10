suffix=$1
OUT_DIR=/home/ogulcan/polychrom/dummy/1d_${suffix}

micromamba run -n polychrom python scripts/compare4_loops_1d.py \
    $OUT_DIR \
    /home/ogulcan/polychrom/configs_test/config1_${suffix}.yaml \
    /home/ogulcan/polychrom/configs_test/config2_${suffix}.yaml \
    /home/ogulcan/polychrom/configs_test/config3_${suffix}.yaml \
    /home/ogulcan/polychrom/configs_test/config4_${suffix}.yaml
