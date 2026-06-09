import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def test_gen_realistic_configs_20s_defaults(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "gen_realistic_configs_20s.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(tmp_path),
            "--chain",
            "5000",
            "--num-chains",
            "2",
            "--suffix",
            "smoke",
        ],
        check=True,
    )

    cfg1 = yaml.safe_load((tmp_path / "config1_smoke.yaml").read_text())
    cfg2 = yaml.safe_load((tmp_path / "config2_smoke.yaml").read_text())
    cfg3 = yaml.safe_load((tmp_path / "config3_smoke.yaml").read_text())

    assert cfg1["lef"]["tick_seconds"] == 20
    assert cfg1["lef"]["lifetime"] == 75
    assert cfg1["lef"]["lifetime_stalled"] == 75
    assert cfg1["lef"]["lifetime_ctcf"] == 300
    assert cfg1["lef"]["separation"] == 240
    assert cfg1["lef"]["trajectory_length"] == 12096
    assert cfg1["polymer"]["md_steps_per_block"] == 5000

    topo1 = cfg1["lef"]["topology_kwargs"]
    assert cfg1["lef"]["max_rnapii"] > 0
    assert cfg2["lef"]["max_rnapii"] == 0
    assert cfg3["lef"]["max_rnapii"] == 0
    assert cfg1["lef"]["plugins"]["rnapii_load"].endswith(":load_rnapii")
    assert cfg2["lef"]["plugins"]["rnapii_load"] is None
    assert topo1["default_boundary_strength"] == pytest.approx(0.13)
    assert topo1["release_prob"] == 0.0
    assert topo1["target_tss"] is False
    assert topo1["rnapii_stride"] == 2
    assert topo1["rnapii_stall_prob"] == 0.0
    assert topo1["rnapii_push_prob"] == 0.0
    assert topo1["rnapii_headon_push_prob"] == 1.0
    assert topo1["rnapii_poised_block_prob"] == pytest.approx(0.819)
    assert topo1["rnapii_paused_block_prob"] == pytest.approx(0.819)
    assert topo1["rnapii_elongating_block_prob"] == pytest.approx(0.819)
    assert topo1["rnapii_terminating_block_prob"] == pytest.approx(0.819)
    assert topo1["lifetime_rnapii_stalled"] == 75

    first_gene = topo1["genes"][0]
    assert first_gene["load_prob"] == pytest.approx(0.0198)
    assert first_gene["termination_prob"] == pytest.approx(0.0392)
    assert first_gene["elongation_step_prob"] == 1.0

    b2 = cfg2["lef"]["topology_kwargs"]["boundary_strength"]
    b3 = cfg3["lef"]["topology_kwargs"]["boundary_strength"]
    assert b2
    assert b2.keys() == b3.keys()
    for boundary, strength in b2.items():
        assert 0.04 <= strength <= 0.13
        assert b3[boundary] == pytest.approx(round(min(strength * 2.0, 1.0), 2))
