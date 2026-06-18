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


def _interior_strengths(topo):
    """Map interior boundary position -> strength from the per-TAD ``tads`` block.

    Each non-last TAD's right_strength is the strength of the boundary just past
    its right anchor (``right + 1``); this reconstructs the legacy
    ``boundary_strength`` mapping for assertions.
    """
    tads = topo["tads"]
    return {t["right"] + 1: t["right_strength"] for t in tads[:-1]}


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
    assert topo["rnapii_pre_initiation_block_prob"] == pytest.approx(expected_block, abs=5e-4)
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
    assert topo["rnapii_pre_initiation_block_prob"] == pytest.approx(expected_block, abs=5e-4)
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

    # config3 scales every (interior) boundary strength by --bstr-mult (=3),
    # capped at 1.0. With the per-TAD schema both oriented sides carry the value,
    # so the reconstructed interior strengths scale identically.
    s2 = _interior_strengths(topo2)
    s3 = _interior_strengths(topo3)
    assert set(s2) == set(s3)
    for site, strength in s2.items():
        assert s3[site] == pytest.approx(round(min(strength * 3.0, 1.0), 2))

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


def test_variable_tick_emits_tads_schema(tmp_path):
    cfg = _generate(tmp_path, 20, "tads")
    topo = _topology(cfg)
    chain = cfg["lef"]["chain_length"]
    assert "tads" in topo
    assert "tad_positions" not in topo and "boundary_strength" not in topo
    tlist = topo["tads"]
    assert len(tlist) >= 2
    for t in tlist:
        assert {"left", "right", "left_strength", "right_strength"} <= set(t)
        assert 0 <= t["left"] <= t["right"] < chain
    # gap_frac defaults to 0 -> TADs abut and fully tile [0, chain).
    assert tlist[0]["left"] == 0
    assert tlist[-1]["right"] == chain - 1
    for a, b in zip(tlist[:-1], tlist[1:]):
        assert b["left"] == a["right"] + 1


def test_variable_tick_nogap_topology_matches_legacy(tmp_path):
    """The emitted gap-free tads reproduce the legacy tad_positions ctcfCapture."""
    from polychrom.pipelines.loop_extrusion.config import LEFConfig
    from polychrom.pipelines.loop_extrusion.plugins import topology as T

    cfg = _generate(tmp_path, 20, "eq")
    lef = cfg["lef"]
    topo = _topology(cfg)
    lc = LEFConfig(chain_length=lef["chain_length"], num_chains=lef["num_chains"],
                   separation=lef["separation"])
    default = topo["default_boundary_strength"]
    from_tads = T.convergent_tad_topology(lc, tads=topo["tads"], default_boundary_strength=default)
    strengths = _interior_strengths(topo)
    from_legacy = T.convergent_tad_topology(
        lc, tad_positions=sorted(strengths), boundary_strength=strengths,
        default_boundary_strength=default,
    )
    assert from_tads["ctcfCapture"] == from_legacy["ctcfCapture"]
    assert from_tads["ctcfRelease"] == from_legacy["ctcfRelease"]


def test_variable_tick_gap_frac_opens_spacers(tmp_path):
    cfg = _generate(tmp_path, 20, "gap", extra=["--gap-frac", "0.2"])
    tlist = _topology(cfg)["tads"]
    # At least one interior gap appears: a TAD's right + 1 < the next TAD's left.
    assert any(b["left"] > a["right"] + 1 for a, b in zip(tlist[:-1], tlist[1:]))
    # TADs stay ordered and non-overlapping.
    for a, b in zip(tlist[:-1], tlist[1:]):
        assert b["left"] > a["right"]
