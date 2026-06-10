import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_realistic_configs_variable_tick.py"


def _generate(tmp_path, tick_seconds, suffix, extra=None):
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(tmp_path),
            "--tick-seconds",
            str(tick_seconds),
            "--chain",
            "5000",
            "--num-chains",
            "2",
            "--suffix",
            suffix,
            *(extra or []),
        ],
        check=True,
    )
    return yaml.safe_load((tmp_path / f"config1_{suffix}.yaml").read_text())


def _first_gene(cfg):
    return cfg["lef"]["topology_kwargs"]["genes"][0]


def _topology(cfg):
    return cfg["lef"]["topology_kwargs"]


def test_variable_tick_20s_reproduces_reference_defaults(tmp_path):
    cfg = _generate(tmp_path, 20, "t20")
    lef = cfg["lef"]
    topo = _topology(cfg)
    gene = _first_gene(cfg)

    assert lef["tick_seconds"] == 20
    assert lef["lifetime"] == 75
    assert lef["lifetime_stalled"] == 75
    assert lef["lifetime_ctcf"] == 300
    assert topo["lifetime_rnapii_stalled"] == 75
    # 12096*20/20 = 12096 snapped to the nearest multiple of restart_every_blocks
    assert lef["trajectory_length"] == 10000
    assert cfg["polymer"]["restart_every_blocks"] == 5000
    assert lef["trajectory_length"] % cfg["polymer"]["restart_every_blocks"] == 0
    assert lef["warmup_steps"] == 10000
    assert cfg["polymer"]["md_steps_per_block"] == 5000

    assert topo["rnapii_stride"] == 2
    assert gene["elongation_step_prob"] == pytest.approx(1.0)
    assert gene["load_prob"] == pytest.approx(0.0198)
    assert gene["termination_prob"] == pytest.approx(0.0392)

    expected_block = math.exp(-20 / 100)
    assert topo["rnapii_poised_block_prob"] == pytest.approx(expected_block, abs=5e-4)
    assert topo["rnapii_paused_block_prob"] == pytest.approx(expected_block, abs=5e-4)
    assert topo["rnapii_elongating_block_prob"] == pytest.approx(expected_block, abs=5e-4)
    assert topo["rnapii_terminating_block_prob"] == pytest.approx(expected_block, abs=5e-4)


def test_variable_tick_4s_preserves_real_durations_and_rates(tmp_path):
    cfg = _generate(tmp_path, 4, "t4")
    lef = cfg["lef"]
    topo = _topology(cfg)
    gene = _first_gene(cfg)

    assert lef["tick_seconds"] == 4
    assert lef["lifetime"] == 375
    assert lef["lifetime_stalled"] == 375
    assert lef["lifetime_ctcf"] == 1500
    assert topo["lifetime_rnapii_stalled"] == 375
    # 12096*20/4 = 60480 snapped to the nearest multiple of restart_every_blocks
    assert lef["trajectory_length"] == 60000
    assert lef["trajectory_length"] % cfg["polymer"]["restart_every_blocks"] == 0
    assert lef["warmup_steps"] == 50000
    assert cfg["polymer"]["md_steps_per_block"] == 1000

    assert topo["rnapii_stride"] == 1
    assert gene["elongation_step_prob"] == pytest.approx(0.4)
    assert gene["load_prob"] == pytest.approx(1 - math.exp(-0.001 * 4), abs=5e-5)
    assert gene["termination_prob"] == pytest.approx(1 - math.exp(-0.002 * 4), abs=5e-5)

    expected_block = math.exp(-4 / 100)
    assert topo["rnapii_poised_block_prob"] == pytest.approx(expected_block, abs=5e-4)
    assert topo["rnapii_paused_block_prob"] == pytest.approx(expected_block, abs=5e-4)
    assert topo["rnapii_elongating_block_prob"] == pytest.approx(expected_block, abs=5e-4)
    assert topo["rnapii_terminating_block_prob"] == pytest.approx(expected_block, abs=5e-4)


def test_variable_tick_25s_uses_fractional_multistep_rnapii(tmp_path):
    cfg = _generate(tmp_path, 25, "t25")
    topo = _topology(cfg)
    gene = _first_gene(cfg)

    assert topo["rnapii_stride"] == 3
    assert gene["elongation_step_prob"] == pytest.approx(2.5 / 3)


def test_variable_tick_config4_defaults_use_phenotype_lesion_regime(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(tmp_path),
            "--tick-seconds",
            "8",
            "--chain",
            "5000",
            "--num-chains",
            "2",
            "--suffix",
            "phenotype",
        ],
        check=True,
    )

    cfg2 = yaml.safe_load((tmp_path / "config2_phenotype.yaml").read_text())
    cfg3 = yaml.safe_load((tmp_path / "config3_phenotype.yaml").read_text())
    cfg4 = yaml.safe_load((tmp_path / "config4_phenotype.yaml").read_text())
    topo2 = _topology(cfg2)
    topo3 = _topology(cfg3)
    topo4 = _topology(cfg4)

    for site, strength in topo2["boundary_strength"].items():
        assert topo3["boundary_strength"][site] == pytest.approx(
            round(min(strength * 3.0, 1.0), 2)
        )

    assert cfg4["lef"]["plugins"]["lesion"].endswith(":update_lesions")
    assert topo4["lesion_spacing"] == 10
    assert topo4["lesion_type_a_prob"] == pytest.approx(0.25)
    assert topo4["lesion_prerecognition_ticks"] == 150
    assert topo4["lesion_repair_ticks"] == 38
    assert topo4["lesion_block_prob"] == pytest.approx(0.97)


def test_variable_tick_allows_duration_overrides(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(tmp_path),
            "--tick-seconds",
            "20",
            "--trajectory-length",
            "15000",
            "--warmup-steps",
            "45",
            "--chain",
            "5000",
            "--suffix",
            "override",
        ],
        check=True,
    )

    cfg = yaml.safe_load((tmp_path / "config1_override.yaml").read_text())
    # A multiple of restart_every_blocks is honored verbatim.
    assert cfg["lef"]["trajectory_length"] == 15000
    assert cfg["lef"]["trajectory_length"] % cfg["polymer"]["restart_every_blocks"] == 0
    assert cfg["lef"]["warmup_steps"] == 45


def test_variable_tick_snaps_non_multiple_override(tmp_path):
    # A user-supplied trajectory length that is not a multiple of
    # restart_every_blocks is snapped so the polymer stage can run it.
    cfg = _generate(tmp_path, 20, "snap", extra=["--trajectory-length", "123"])
    traj = cfg["lef"]["trajectory_length"]
    restart = cfg["polymer"]["restart_every_blocks"]
    assert traj % restart == 0
    assert traj == restart  # 123 rounds up to one full block
