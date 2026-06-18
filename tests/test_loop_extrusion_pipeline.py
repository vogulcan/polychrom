import hashlib

import h5py
import numpy as np
import pytest

from polychrom.pipelines.loop_extrusion import cli as cli_stage
from polychrom.pipelines.loop_extrusion import lef as lef_stage
from polychrom.pipelines.loop_extrusion import contacts as contacts_stage
from polychrom.pipelines.loop_extrusion import viewer as viewer_stage
from polychrom.pipelines.loop_extrusion.config import (
    ContactsConfig,
    LEFConfig,
    PluginSpec,
    ViewerConfig,
    load_config,
)
from polychrom.pipelines.loop_extrusion.plugins.sampling import (
    balanced_observed_over_expected,
    iterative_correction,
)
from polychrom.pipelines.loop_extrusion.plugins.lef_dynamics import (
    COHESIN,
    RNAPII_CELL,
    Cohesin,
    Leg,
    load_one,
    load_targeted,
    translocate,
    translocate_with_rnapii,
)
from polychrom.pipelines.loop_extrusion.plugins.rnapii import (
    Gene,
    RNAPII,
    STATE_ELONGATING,
    STATE_PAUSED,
    STATE_PRE_INITIATION,
    STATE_TERMINATING,
    build_genes,
    compute_ep_contacts,
    enhancer_factor,
    load_rnapii,
    stateful_translocate_rnapii,
    translocate_rnapii,
)
from polychrom.pipelines.loop_extrusion.qc import rnapii_metrics
from polychrom.pipelines.loop_extrusion.plugins import (
    forces as force_plugins,
    topology as topology_plugins,
)
from polychrom.pipelines.loop_extrusion.polymer import StickyUpdater, _gene_bodies
from polychrom.pipelines.loop_extrusion.viewer import (
    build_elements_export,
    build_payload,
    build_visited_heatmap,
    effective_distance,
)


def test_balanced_observed_over_expected_reduces_row_bias():
    base = np.ones((8, 8), dtype=float)
    np.fill_diagonal(base, 10.0)
    bias = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 2.5, 0.7])
    biased = base * np.outer(bias, bias)

    corrected = iterative_correction(biased, ignore_diagonals=1, max_iter=200, tol=1e-8)
    fit_mask = np.ones_like(corrected)
    for offset in (0, 1):
        rows = np.arange(corrected.shape[0] - offset)
        cols = rows + offset
        fit_mask[rows, cols] = 0
        fit_mask[cols, rows] = 0
    row_sums = np.nansum(corrected * fit_mask, axis=1)
    assert np.nanmax(row_sums) / np.nanmin(row_sums) < 1.01

    oe = balanced_observed_over_expected(biased, ignore_diagonals=1)
    assert oe.shape == biased.shape
    assert np.isfinite(oe).all()


def test_loop_extrusion_config_supports_warmup_and_seed(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
lef:
  warmup_steps: 123
  seed: 17
  tick_seconds: 20
polymer:
  seed: 19
"""
    )
    cfg = load_config(cfg_path)
    assert cfg.lef.warmup_steps == 123
    assert cfg.lef.seed == 17
    assert cfg.lef.tick_seconds == 20
    assert cfg.polymer.seed == 19


def test_loop_extrusion_config_derives_paths_from_runtime_output(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
lef:
  output_path: /stale/LEFPositions.h5
viewer:
  lef_positions_path: /stale/LEFPositions.h5
  output_path: /stale/bridging_viewer.html
  heatmap_output_path: /stale/heatmap.npy
  elements_output_path: /stale/elements.json
polymer:
  lef_positions_path: /stale/LEFPositions.h5
  output_folder: /stale
contacts:
  trajectory_folder: /stale
  raw_output_path: /stale/contact_map.npy
  oe_output_path: /stale/contact_map_oe.npy
  viz_output_path: /stale/contact_map_oe.png
"""
    )
    out_dir = tmp_path / "run"

    cfg = load_config(cfg_path, output_path=out_dir)

    assert cfg.lef.output_path == str(out_dir / "LEFPositions.h5")
    assert cfg.viewer.lef_positions_path == str(out_dir / "LEFPositions.h5")
    assert cfg.viewer.output_path == str(out_dir / "bridging_viewer.html")
    assert cfg.viewer.heatmap_output_path == str(
        out_dir / "bridging_viewer_visited_heatmap.npy"
    )
    assert cfg.viewer.elements_output_path == str(
        out_dir / "bridging_viewer_elements.json"
    )
    assert cfg.polymer.lef_positions_path == str(out_dir / "LEFPositions.h5")
    assert cfg.polymer.output_folder == str(out_dir)
    assert cfg.contacts.trajectory_folder == str(out_dir)
    assert cfg.contacts.raw_output_path == str(out_dir / "contact_map.npy")
    assert cfg.contacts.oe_output_path == str(out_dir / "contact_map_oe.npy")
    assert cfg.contacts.viz_output_path == str(out_dir / "contact_map_oe.png")


def test_loop_extrusion_cli_accepts_runtime_output_path(tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "config.yaml"
    config_text = "lef:\n  chain_length: 10\n"
    cfg_path.write_text(config_text)
    out_dir = tmp_path / "run"
    calls = {}

    def fake_lef_run(lef_cfg):
        calls["lef_output_path"] = lef_cfg.output_path
        return lef_cfg.output_path

    monkeypatch.setattr(cli_stage.lef_stage, "run", fake_lef_run)

    assert cli_stage.main(["lef", str(cfg_path), str(out_dir)]) == 0

    assert calls["lef_output_path"] == str(out_dir / "LEFPositions.h5")
    assert (out_dir / "config.yaml").read_text() == config_text
    stdout = capsys.readouterr().out
    assert f"output directory {out_dir}" in stdout
    assert f"saved {out_dir / 'config.yaml'}" in stdout
    assert str(out_dir / "LEFPositions.h5") in stdout


def test_lef_stage_writes_h5_trajectory(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    h5 = tmp_path / "LEFPositions.h5"
    cfg_path.write_text(
        f"""
lef:
  chain_length: 200
  num_chains: 1
  separation: 60
  trajectory_length: 60
  chunk_size: 10
  seed: 7
  output_path: {h5}
  topology_kwargs:
    tad_positions: [50, 150]
    capture_prob: 0.9
    release_prob: 0.01
    symmetric: true
"""
    )
    cfg = load_config(cfg_path)
    out = lef_stage.run(cfg.lef)
    assert out.exists()                       # H5 trajectory written


def _cohesin_hits_rnapii_args(
    state,
    *,
    pre_initiation_block=1.0,
    paused_block=1.0,
    elongating_block=1.0,
    block=None,
):
    occupied = np.zeros(10, dtype=np.int8)
    left = Leg(3)
    right = Leg(8)
    cohesins = [Cohesin(left, right)]
    rnap = RNAPII(pos=2, gene_id=0, direction=1)
    rnap.attrs["state"] = state
    occupied[2] = RNAPII_CELL
    occupied[3] = COHESIN
    occupied[8] = COHESIN
    args = {
        "N": 10,
        "LIFETIME": 10**9,
        "LIFETIME_STALLED": 10**9,
        "ctcfCapture": {-1: {}, 1: {}},
        "ctcfRelease": {-1: {}, 1: {}},
        "rnapii_by_pos": {2: rnap},
        "tes_by_pos": {},
        "cohesin_leg_by_pos": {3: left, 8: right},
        "rnapii_pre_initiation_block_prob": pre_initiation_block,
        "rnapii_paused_block_prob": paused_block,
        "rnapii_elongating_block_prob": elongating_block,
    }
    if block is not None:
        args["rnapii_block_prob"] = block
    return occupied, cohesins, args


def test_pre_initiation_rnapii_block_probability_controls_cohesin_bypass():
    occupied, cohesins, args = _cohesin_hits_rnapii_args(
        STATE_PRE_INITIATION,
        pre_initiation_block=0.0,
        paused_block=1.0,
        elongating_block=1.0,
    )

    translocate_with_rnapii(cohesins, occupied, args, unload_prob_fn=lambda *_: 0.0)

    assert cohesins[0].left.pos == 2
    assert not cohesins[0].left.attrs["stalled"]


def test_rnapii_loading_can_require_ep_contact():
    occupied = np.zeros(10, dtype=np.int8)
    gene = Gene(
        gene_id=0,
        tss=2,
        tes=6,
        direction=1,
        load_prob=1.0,
        enhancer_pos=5,
        requires_enhancer=True,
        load_requires_enhancer=True,
    )
    rnapiis = []
    args = {
        "genes": [gene],
        "rnapii_by_pos": {},
        "current_ep_contacts": set(),
    }

    load_rnapii(rnapiis, occupied, args)
    assert rnapiis == []
    assert occupied[2] == 0

    args["current_ep_contacts"] = {0}
    load_rnapii(rnapiis, occupied, args)
    assert len(rnapiis) == 1
    assert occupied[2] == RNAPII_CELL


def test_paused_rnapii_block_probability_controls_cohesin_bypass():
    occupied, cohesins, args = _cohesin_hits_rnapii_args(
        STATE_PAUSED,
        paused_block=0.0,
        elongating_block=1.0,
    )

    translocate_with_rnapii(cohesins, occupied, args, unload_prob_fn=lambda *_: 0.0)

    assert cohesins[0].left.pos == 2
    assert not cohesins[0].left.attrs["stalled"]


def test_elongating_rnapii_block_probability_controls_cohesin_bypass():
    occupied, cohesins, args = _cohesin_hits_rnapii_args(
        STATE_ELONGATING,
        paused_block=0.0,
        elongating_block=1.0,
    )

    translocate_with_rnapii(cohesins, occupied, args, unload_prob_fn=lambda *_: 0.0)

    assert cohesins[0].left.pos == 3
    assert cohesins[0].left.attrs["stalled"]


def test_legacy_rnapii_block_probability_is_state_fallback():
    occupied, cohesins, args = _cohesin_hits_rnapii_args(
        STATE_ELONGATING,
        block=0.0,
    )
    del args["rnapii_elongating_block_prob"]

    translocate_with_rnapii(cohesins, occupied, args, unload_prob_fn=lambda *_: 0.0)

    assert cohesins[0].left.pos == 2
    assert not cohesins[0].left.attrs["stalled"]


def _two_chain_args():
    return {
        "N": 6,
        "chain_length": 3,
        "num_chains": 2,
        "LIFETIME": 10**9,
        "LIFETIME_STALLED": 10**9,
        "ctcfCapture": {-1: {}, 1: {}},
        "ctcfRelease": {-1: {}, 1: {}},
    }


def test_load_one_rejects_cross_chain_adjacent_pair():
    occupied = np.ones(6, dtype=np.int8)
    occupied[2] = 0
    occupied[3] = 0

    with pytest.raises(RuntimeError):
        load_one([], occupied, _two_chain_args())


def test_translocate_respects_chain_boundary():
    occupied = np.zeros(6, dtype=np.int8)
    left = Leg(3)
    right = Leg(4)
    cohesins = [Cohesin(left, right)]
    occupied[3] = COHESIN
    occupied[4] = COHESIN

    translocate(cohesins, occupied, _two_chain_args(), unload_prob_fn=lambda *_: 0.0)

    assert left.pos == 3
    assert left.attrs["stalled"]


def test_translocate_with_rnapii_respects_chain_boundary():
    occupied = np.zeros(6, dtype=np.int8)
    left = Leg(3)
    right = Leg(4)
    cohesins = [Cohesin(left, right)]
    occupied[3] = COHESIN
    occupied[4] = COHESIN
    args = _two_chain_args()
    args.update({
        "rnapii_by_pos": {},
        "tes_by_pos": {},
        "cohesin_leg_by_pos": {3: left, 4: right},
    })

    translocate_with_rnapii(cohesins, occupied, args, unload_prob_fn=lambda *_: 0.0)

    assert left.pos == 3
    assert left.attrs["stalled"]


def test_rnapii_terminates_at_tes_then_unloads():
    # Pol II sitting on its TES enters TERMINATING and dwells (termination_prob
    # < 1), then unloads once the roll succeeds.
    gene = build_genes([{"tss": 10, "tes": 20, "termination_prob": 0.0}])[0]
    occupied = np.zeros(40, dtype=np.int8)
    r = RNAPII(pos=20, gene_id=0, direction=1)
    r.attrs["state"] = STATE_ELONGATING
    occupied[20] = RNAPII_CELL
    args = {"N": 40, "chain_length": 40, "num_chains": 1, "genes": [gene],
            "rnapii_by_pos": {20: r}, "cohesin_leg_by_pos": {}, "ep_contact_tolerance": 1}
    rnapiis = [r]

    # termination_prob 0 -> never unloads; dwells as TERMINATING block
    for _ in range(5):
        stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert len(rnapiis) == 1
    assert r.attrs["state"] == STATE_TERMINATING
    assert occupied[20] == RNAPII_CELL          # still blocking the TES site

    # flip termination_prob to 1 -> unloads next tick
    gene.termination_prob = 1.0
    stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert len(rnapiis) == 0
    assert occupied[20] == 0


def test_stateful_rnapii_stride_advances_elongating_pol_ii():
    gene = build_genes([{"tss": 10, "tes": 30, "elongation_step_prob": 1.0}])[0]
    occupied = np.zeros(40, dtype=np.int8)
    r = RNAPII(pos=20, gene_id=0, direction=1)
    r.attrs["state"] = STATE_ELONGATING
    occupied[20] = RNAPII_CELL
    args = {
        "N": 40,
        "chain_length": 40,
        "num_chains": 1,
        "genes": [gene],
        "rnapii_by_pos": {20: r},
        "cohesin_leg_by_pos": {},
        "ep_contact_tolerance": 1,
        "rnapii_stride": 2,
    }

    stateful_translocate_rnapii([r], [], occupied, args)

    assert r.pos == 22
    assert r.attrs["state"] == STATE_ELONGATING
    assert occupied[20] == 0
    assert occupied[22] == RNAPII_CELL
    assert args["rnapii_by_pos"] == {22: r}


def test_rnapii_metrics_counts_multi_site_elongation_steps():
    rnapii_positions = np.array([[[10, 0]], [[12, 0]], [[14, 0]]], dtype=np.int32)
    rnapii_states = np.array([[STATE_ELONGATING], [STATE_ELONGATING], [STATE_ELONGATING]], dtype=np.int8)
    rnapii_ids = np.array([[7], [7], [7]], dtype=np.int32)

    metrics = rnapii_metrics(
        rnapii_positions,
        rnapii_states,
        tick_seconds=20.0,
        rnapii_ids=rnapii_ids,
    )

    assert metrics["elongation_speed_sites_per_tick"] == pytest.approx(2.0)
    assert metrics["elongation_speed_kb_per_min"] == pytest.approx(6.0)


def test_terminating_pol_ii_blocks_cohesin():
    from polychrom.pipelines.loop_extrusion.plugins.lef_dynamics import _rnapii_blocks_cohesin
    r = RNAPII(pos=20, gene_id=0, direction=1)
    r.attrs["state"] = STATE_TERMINATING
    # explicit terminating block prob 1.0 -> always blocks
    assert _rnapii_blocks_cohesin(r, {"rnapii_terminating_block_prob": 1.0}) is True
    # falls back to paused block prob when terminating-specific absent
    assert _rnapii_blocks_cohesin(r, {"rnapii_paused_block_prob": 1.0}) is True


def test_lesion_population_holds_at_target():
    """lesion_spacing sets the steady-state count N // spacing; the refill keeps
    the population pinned at the target as repaired lesions are removed."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import (
        precompute_lesion_fields, update_lesions,
    )
    gene = build_genes([{"tss": 20, "tes": 60}])[0]
    args = {"N": 200, "chain_length": 200, "num_chains": 1, "genes": [gene],
            "lesion_spacing": 10, "lesion_type_a_prob": 0.5,
            "lesion_prerecognition_ticks": 5, "lesion_repair_ticks": 5}
    precompute_lesion_fields(args, tad_positions=[100], gene_objs=[gene],
                             tad_size_exponent=1.0, spacing=10)
    assert args["lesion_target"] == 20
    np.random.seed(0)
    counts = []
    for _ in range(60):
        update_lesions(args)
        counts.append(len(args["lesions"]))
    assert all(c == 20 for c in counts)              # filled on tick 1, held thereafter


def test_lesion_placement_weighted_by_tad_size():
    """Per-site spawn prob ~ L_TAD**(-alpha): alpha=0 -> count proportional to TAD
    length; alpha=1 -> ~equal expected count per TAD (shorter TADs denser)."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import (
        precompute_lesion_fields, refill_lesions,
    )
    # One chain, two TADs: [0,100) length 100, [100,400) length 300.
    def run(alpha):
        args = {"N": 400, "chain_length": 400, "num_chains": 1, "genes": [],
                "lesion_type_a_prob": 0.0}
        precompute_lesion_fields(args, tad_positions=[100], gene_objs=[],
                                 tad_size_exponent=alpha, spacing=8)   # target 50
        np.random.seed(3)
        small = big = 0
        for _ in range(40):                          # many independent fills
            args["lesions"] = {}
            refill_lesions(args)
            sites = np.array(sorted(args["lesions"]))
            small += int((sites < 100).sum())
            big += int((sites >= 100).sum())
        return small, big

    s0, b0 = run(0.0)
    assert b0 > 2.0 * s0                             # ~3x more in the 3x-longer TAD
    s1, b1 = run(1.0)
    assert 0.7 < (s1 / b1) < 1.4                     # ~balanced count per TAD


def test_lesion_type_assignment_gene_body_vs_outside():
    """Gene-body lesion -> Type A w.p. lesion_type_a_prob else B; off gene body -> B."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import (
        precompute_lesion_fields, _assign_type, LESION_TYPE_A, LESION_TYPE_B,
    )
    gene = build_genes([{"tss": 20, "tes": 40}])[0]   # gene body [20, 40]
    args = {"N": 100, "chain_length": 100, "num_chains": 1, "lesion_type_a_prob": 0.5}
    precompute_lesion_fields(args, tad_positions=[], gene_objs=[gene],
                             tad_size_exponent=1.0, spacing=10)
    np.random.seed(0)
    assert all(_assign_type(s, args) == LESION_TYPE_B for s in (0, 5, 50, 99))
    types = [_assign_type(30, args) for _ in range(400)]
    assert all(t in (LESION_TYPE_A, LESION_TYPE_B) for t in types)
    frac_a = sum(t == LESION_TYPE_A for t in types) / len(types)
    assert 0.4 < frac_a < 0.6


def test_lesion_state_machine_stochastic_transitions():
    """PRE->REPAIR and REPAIR->removed fire at the configured 1/ticks rates."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import (
        Lesion, update_lesions, LESION_TYPE_A, LESION_TYPE_B, LESION_PRE, LESION_REPAIR,
    )
    # Spawning off (target 0): observe transitions of a fixed cohort.
    args = {"N": 1000, "lesion_target": 0, "lesion_site_p": None,
            "lesion_prerecognition_ticks": 4, "lesion_repair_ticks": 1000,
            "lesions": {s: Lesion(s, LESION_TYPE_A, LESION_PRE) for s in range(1000)}}
    np.random.seed(0)
    update_lesions(args)
    n_repair = sum(l.state == LESION_REPAIR for l in args["lesions"].values())
    assert 200 < n_repair < 300                      # ~1/4 advanced PRE->REPAIR
    assert len(args["lesions"]) == 1000              # huge repair_ticks -> none removed

    args2 = {"N": 1000, "lesion_target": 0, "lesion_site_p": None,
             "lesion_prerecognition_ticks": 1000, "lesion_repair_ticks": 2,
             "lesions": {s: Lesion(s, LESION_TYPE_B, LESION_REPAIR) for s in range(1000)}}
    np.random.seed(1)
    update_lesions(args2)
    assert 400 < len(args2["lesions"]) < 600         # ~1/2 of REPAIR lesions repaired


def test_lesion_cohesin_stall_matrix_without_rnapii():
    """With no RNAPII in the model: Type-A pre stalls cohesin intrinsically (fast
    eviction); Type-B pre does not; repair stalls generically (either type)."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import (
        Lesion, LESION_TYPE_A, LESION_TYPE_B, LESION_PRE, LESION_REPAIR,
    )

    def step(les):
        occupied = np.zeros(40, dtype=np.int8)
        left = Leg(10); right = Leg(11)
        occupied[10] = COHESIN; occupied[11] = COHESIN
        args = {"N": 40, "chain_length": 40, "num_chains": 1,
                "ctcfCapture": {-1: {}, 1: {}}, "ctcfRelease": {-1: {}, 1: {}},
                "rnapii_by_pos": {}, "tes_by_pos": {},
                "cohesin_leg_by_pos": {10: left, 11: right},
                "lesions": {9: les}, "lesion_block_prob": 1.0, "rnapii_enabled": False}
        translocate_with_rnapii([Cohesin(left, right)], occupied, args,
                                unload_prob_fn=lambda *_: 0.0)
        return left, right

    left, right = step(Lesion(9, LESION_TYPE_A, LESION_PRE))
    assert left.pos == 10 and left.attrs["stalled"] and left.attrs["rnapii_stalled"]
    assert right.pos == 12                            # other leg extrudes -> asymmetric

    left, right = step(Lesion(9, LESION_TYPE_B, LESION_PRE))
    assert left.pos == 9 and not left.attrs.get("stalled")   # read through, no stall

    for t in (LESION_TYPE_A, LESION_TYPE_B):
        left, right = step(Lesion(9, t, LESION_REPAIR))
        assert left.pos == 10 and left.attrs["stalled"] and not left.attrs["rnapii_stalled"]


def test_lesion_blocks_rnapii_by_type_state_and_evicts_on_repair():
    """Type-A pre hard-blocks Pol II; Type-B pre is read through; a lesion-stalled
    Pol II is evicted (not resumed) once its lesion is repaired."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import (
        Lesion, LESION_TYPE_A, LESION_TYPE_B, LESION_PRE,
    )

    def setup(les):
        gene = build_genes([{"tss": 10, "tes": 20, "elongation_step_prob": 1.0,
                             "initiation_prob": 1.0, "pause_release_prob": 1.0}])[0]
        occupied = np.zeros(40, dtype=np.int8)
        r = RNAPII(pos=14, gene_id=0, direction=1)
        r.attrs["state"] = STATE_ELONGATING
        occupied[14] = RNAPII_CELL
        args = {"N": 40, "chain_length": 40, "num_chains": 1, "genes": [gene],
                "rnapii_by_pos": {14: r}, "cohesin_leg_by_pos": {},
                "lesions": {15: les}, "ep_contact_tolerance": 1}
        return r, occupied, args

    r, occupied, args = setup(Lesion(15, LESION_TYPE_A, LESION_PRE))
    rnapiis = [r]
    for _ in range(5):
        stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert r.pos == 14 and r.attrs.get("lesion_stalled")     # blocked just upstream

    r, occupied, args = setup(Lesion(15, LESION_TYPE_B, LESION_PRE))
    rnapiis = [r]
    stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert r.pos == 15 and not r.attrs.get("lesion_stalled")  # read through onto site 15

    r, occupied, args = setup(Lesion(15, LESION_TYPE_A, LESION_PRE))
    rnapiis = [r]
    stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert rnapiis and r.attrs.get("lesion_stalled")          # stalled at 14
    del args["lesions"][15]                                   # lesion repaired
    stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert rnapiis == [] and occupied[14] == 0                # Pol II evicted


def test_lesion_repair_faster_in_shorter_tads():
    """lesion_tad_repair_exponent > 0 makes recognition (PRE->REPAIR) faster for
    lesions in shorter TADs; exponent 0 recovers TAD-independent rates."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import (
        precompute_lesion_fields, update_lesions, Lesion, LESION_TYPE_B,
        LESION_PRE, LESION_REPAIR,
    )

    def mean_recognition_ticks(sites, beta, seed):
        # short TAD [0,40) len 40, long TAD [40,400) len 360 -> L_ref = 200
        args = {"N": 400, "chain_length": 400, "num_chains": 1,
                "lesion_prerecognition_ticks": 8, "lesion_repair_ticks": 10**9}
        precompute_lesion_fields(args, tad_positions=[40], gene_objs=[],
                                 tad_size_exponent=0.0, spacing=0,
                                 tad_repair_exponent=beta)
        args["lesion_target"] = 0                    # no refill: the cohort just decays
        args["lesions"] = {s: Lesion(s, LESION_TYPE_B, LESION_PRE) for s in sites}
        np.random.seed(seed)
        pending = set(sites)
        recog = []
        for t in range(1, 5000):
            update_lesions(args)
            for s in list(pending):
                if args["lesions"][s].state == LESION_REPAIR:
                    recog.append(t)
                    pending.discard(s)
            if not pending:
                break
        return float(np.mean(recog))

    short = list(range(0, 40))
    long = list(range(40, 400))
    # beta = 1: short TAD recognised much faster (~1.6 vs ~14 ticks).
    s1 = mean_recognition_ticks(short, 1.0, 0)
    l1 = mean_recognition_ticks(long, 1.0, 1)
    assert s1 * 2 < l1
    # beta = 0: no TAD dependence -> comparable mean dwell (~8 ticks each).
    s0 = mean_recognition_ticks(short, 0.0, 2)
    l0 = mean_recognition_ticks(long, 0.0, 3)
    assert 0.6 < (s0 / l0) < 1.6


def test_lesion_type_b_disable_preserves_type_a():
    """lesion_type_b_enabled=False keeps ONLY Type-A lesions at the same count
    they have when B is on (no Type-B, no Type-A backfill)."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import (
        precompute_lesion_fields, update_lesions, LESION_TYPE_A, LESION_TYPE_B,
    )
    genes = build_genes([{"tss": 20, "tes": 60}, {"tss": 120, "tes": 160}])

    def make_args(b_enabled):
        args = {"N": 200, "chain_length": 200, "num_chains": 1, "genes": genes,
                "lesion_spacing": 4, "lesion_type_a_prob": 0.5,
                "lesion_prerecognition_ticks": 5, "lesion_repair_ticks": 5,
                "lesion_type_b_enabled": b_enabled}
        precompute_lesion_fields(args, tad_positions=[100], gene_objs=genes,
                                 tad_size_exponent=1.0, spacing=4)
        return args

    # target_a follows the calculation: target * gene_body_mass * type_a_prob.
    a = make_args(True)
    gene_body_mass = float(a["lesion_site_p"][a["gene_body_mask"]].sum())
    assert a["lesion_target_a"] == round(a["lesion_target"] * gene_body_mass * 0.5)
    target_a = a["lesion_target_a"]
    assert 0 < target_a < a["lesion_target"]

    # B disabled: every lesion is Type A, count pinned at target_a, no Type B.
    d = make_args(False)
    np.random.seed(0)
    nB = 0
    for _ in range(80):
        update_lesions(d)
        vals = list(d["lesions"].values())
        assert len(vals) == target_a
        assert all(l.ltype == LESION_TYPE_A for l in vals)
        nB += sum(1 for l in vals if l.ltype == LESION_TYPE_B)
    assert nB == 0

    # B enabled: total holds at target; mean Type-A ~ target_a (matches disabled).
    e = make_args(True)
    np.random.seed(0)
    a_counts = []
    for _ in range(80):
        update_lesions(e)
        vals = list(e["lesions"].values())
        assert len(vals) == e["lesion_target"]
        a_counts.append(sum(1 for l in vals if l.ltype == LESION_TYPE_A))
    assert abs(float(np.mean(a_counts)) - target_a) <= 0.15 * target_a + 2


def test_load_targeted_places_near_loading_site():
    occupied = np.zeros(100, dtype=np.int8)
    args = {"N": 100, "chain_length": 100, "num_chains": 1,
            "loading_sites": [50], "targeted_load_prob": 1.0, "loading_window": 2}
    cohesins = []
    np.random.seed(0)
    load_targeted(cohesins, occupied, args)
    assert len(cohesins) == 1
    c = cohesins[0]
    assert 48 <= c.left.pos <= 51          # within +/-window of site 50
    assert c.right.pos == c.left.pos + 1


def test_load_targeted_falls_back_to_uniform_when_no_sites():
    occupied = np.zeros(50, dtype=np.int8)
    args = {"N": 50, "chain_length": 50, "num_chains": 1,
            "loading_sites": [], "targeted_load_prob": 1.0}
    cohesins = []
    load_targeted(cohesins, occupied, args)
    assert len(cohesins) == 1               # still loaded (uniform fallback)


def test_load_targeted_never_crosses_chain_boundary():
    # site 49 is the last monomer of chain 0 (chains length 50). Targeting it
    # must keep the cohesin entirely within chain 0, never spilling into chain 1.
    occupied = np.zeros(100, dtype=np.int8)
    args = {"N": 100, "chain_length": 50, "num_chains": 2,
            "loading_sites": [49], "targeted_load_prob": 1.0, "loading_window": 3}
    for seed in range(20):
        cohesins = []
        np.random.seed(seed)
        load_targeted(cohesins, np.zeros(100, dtype=np.int8), args)
        c = cohesins[0]
        assert c.left.pos // 50 == c.right.pos // 50      # same chain
        assert c.left.pos // 50 == 0                       # stayed in site's chain


def test_rnapii_push_respects_chain_boundary():
    occupied = np.zeros(6, dtype=np.int8)
    leg = Leg(2)
    rnap = RNAPII(pos=1, gene_id=0, direction=1)
    occupied[1] = RNAPII_CELL
    occupied[2] = COHESIN
    args = _two_chain_args()
    args.update({
        "genes": [Gene(gene_id=0, tss=1, tes=4, direction=1, load_prob=0.0)],
        "rnapii_by_pos": {1: rnap},
        "cohesin_leg_by_pos": {2: leg},
        "rnapii_stall_prob": 0.0,
        "rnapii_push_prob": 1.0,
    })

    translocate_rnapii([rnap], [], occupied, args)

    assert rnap.pos == 1
    assert leg.pos == 2


def _push_args(extra=None):
    # Single chain of length 8, RNAPII at 1 (dir +1), cohesin leg at 2,
    # site behind (3) free so a push can land.
    args = {
        "N": 8,
        "chain_length": 8,
        "num_chains": 1,
        "genes": [Gene(gene_id=0, tss=1, tes=7, direction=1, load_prob=0.0)],
        "rnapii_by_pos": {},
        "rnapii_stall_prob": 0.0,
        "rnapii_push_prob": 1.0,
    }
    if extra:
        args.update(extra)
    return args


def test_pre_initiation_rnapii_cannot_push_cohesin():
    # Non-elongating Pol II is a stationary block: it never displaces cohesin
    # even with push_prob = 1.0 (Fursova & Larson 2024, Fig 3a).
    occupied = np.zeros(8, dtype=np.int8)
    leg = Leg(2, {"stalled": False, "CTCF": False, "dir": 1})  # co-directional
    rnap = RNAPII(pos=1, gene_id=0, direction=1)
    rnap.attrs["state"] = STATE_PRE_INITIATION
    occupied[1] = RNAPII_CELL
    occupied[2] = COHESIN
    args = _push_args({"cohesin_leg_by_pos": {2: leg}})

    translocate_rnapii([rnap], [], occupied, args)

    assert rnap.pos == 1          # blocked
    assert leg.pos == 2


def test_elongating_rnapii_pushes_codirectional_but_stalls_headon():
    # Elongating Pol II pushes a co-directional (rear) leg; a head-on
    # (converging) leg is governed by the separate, low headon push prob.
    # Head-on case: headon_push_prob = 0 -> stall.
    occ_h = np.zeros(8, dtype=np.int8)
    leg_h = Leg(2, {"stalled": False, "CTCF": False, "dir": -1})  # head-on
    rnap_h = RNAPII(pos=1, gene_id=0, direction=1)
    rnap_h.attrs["state"] = STATE_ELONGATING
    occ_h[1] = RNAPII_CELL
    occ_h[2] = COHESIN
    args_h = _push_args({
        "cohesin_leg_by_pos": {2: leg_h},
        "rnapii_headon_push_prob": 0.0,
    })
    translocate_rnapii([rnap_h], [], occ_h, args_h)
    assert rnap_h.pos == 1        # head-on -> stall
    assert leg_h.pos == 2

    # Co-directional case: push_prob = 1.0 -> push (leg 2 -> 3, Pol II 1 -> 2).
    occ_c = np.zeros(8, dtype=np.int8)
    leg_c = Leg(2, {"stalled": False, "CTCF": False, "dir": 1})  # co-directional
    rnap_c = RNAPII(pos=1, gene_id=0, direction=1)
    rnap_c.attrs["state"] = STATE_ELONGATING
    occ_c[1] = RNAPII_CELL
    occ_c[2] = COHESIN
    args_c = _push_args({"cohesin_leg_by_pos": {2: leg_c}})
    translocate_rnapii([rnap_c], [], occ_c, args_c)
    assert rnap_c.pos == 2        # co-directional -> push
    assert leg_c.pos == 3
    assert leg_c.attrs["pushed"]


# --------------------------------------------------------------------------- #
# Viewer stage
# --------------------------------------------------------------------------- #
def test_effective_distance_drops_with_bridging_loop():
    cl = 200  # single chain
    assert effective_distance([], 10, 90, cl) == 80.0                 # backbone only
    # loop (12, 88) brackets the pair: 10->12 (2) + chord (1) + 88->90 (2) = 5
    assert effective_distance([(12, 88)], 10, 90, cl, bridge_cost=1.0) == 5.0
    # a loop that does not bracket the pair gives no shortcut
    assert effective_distance([(100, 150)], 10, 90, cl) == 80.0


def test_effective_distance_unreachable_across_chains():
    assert effective_distance([], 10, 250, 200) is None               # different chains


def _plain_lef_cfg():
    return LEFConfig(
        chain_length=200, num_chains=1, separation=60,
        topology_kwargs={"tad_positions": [50, 150], "capture_prob": 0.9,
                         "release_prob": 0.01, "symmetric": True},
    )


def test_build_payload_plain_has_no_ep_panel():
    lef_cfg = _plain_lef_cfg()                       # num_lefs = 200 // 60 = 3
    pos = np.array([[[10, 90], [30, 60], [120, 170]]] * 5, dtype=np.int32)
    payload = build_payload(pos, ViewerConfig(stride=1, max_frames=0), lef_cfg)
    assert payload["eps"] == []
    assert payload["latticeSize"] == 200
    assert sorted(payload["ctcfSites"]) == [50, 150]
    assert payload["genes"] == []
    assert [t["label"] for t in payload["tads"]] == ["T0", "T1", "T2"]
    ins = payload["insulation"]
    assert ins["window"] == 50
    assert ins["positions"][0] == 50
    assert ins["positions"][-1] == 150
    # log2(score/mean): boundary 50 has 2 crossings, boundary 150 has 1, so the
    # log2 difference is log2(2/1) == 1 (the mean cancels).
    assert round(ins["frame"][0][0] - ins["frame"][0][-1], 6) == 1.0
    assert round(ins["cumulative"][-1][0] - ins["cumulative"][-1][-1], 6) == 1.0
    # positions with no crossings are masked as gaps (None), not 0.
    assert ins["frame"][0][ins["positions"].index(100)] is None
    assert len(payload["frames"]) == 5
    assert all(len(f["c"]) == 3 for f in payload["frames"])
    assert all(f["s"] == [] for f in payload["frames"])
    assert all(f["r"] == [] for f in payload["frames"])


def test_build_payload_includes_dynamic_rnapii_entries():
    lef_cfg = _plain_lef_cfg()
    pos = np.array([[[10, 90], [30, 60], [120, 170]]] * 2, dtype=np.int32)
    rnapii_positions = np.array(
        [
            [[20, 0], [95, 1], [-1, -1]],
            [[25, 0], [205, 1], [60, 2]],
        ],
        dtype=np.int32,
    )
    rnapii_states = np.array([[0, 1, -1], [2, 1, 0]], dtype=np.int8)
    vcfg = ViewerConfig(stride=1, max_frames=0, site_start=10, site_end=100)

    payload = build_payload(pos, vcfg, lef_cfg, rnapii_positions, rnapii_states)

    assert payload["latticeSize"] == 90
    assert payload["frames"][0]["r"] == [[10, 0, 0], [85, 1, 1]]
    assert payload["frames"][1]["r"] == [[15, 0, 2], [50, 2, 0]]


def test_build_payload_gene_aware_ep_distance_shrinks():
    lef_cfg = LEFConfig(
        chain_length=200, num_chains=1, separation=200,   # single cohesin -> isolate effect
        topology_kwargs={"tad_positions": [],
                         "genes": [{"gene_id": 0, "tss": 90, "tes": 120, "enhancer_pos": 10}]},
    )
    lef_cfg.plugins.topology = PluginSpec(
        target="polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_topology"
    )
    # frame 0: loop far from E(10)-P(90), no shortcut; frame 1: loop (12, 88) brackets it.
    pos = np.array([[[150, 160]], [[12, 88]]], dtype=np.int32)
    payload = build_payload(pos, ViewerConfig(stride=1, max_frames=0), lef_cfg)
    assert len(payload["eps"]) == 1
    assert payload["genes"] == [
        {"geneId": 0, "tss": 90, "tes": 120, "label": "G0", "start": 90, "end": 120}
    ]
    assert payload["eps"][0]["e"] == 10 and payload["eps"][0]["p"] == 90
    assert payload["eps"][0]["genomic"] == 80
    assert payload["frames"][0]["s"] == [80.0]    # backbone distance, no bridging
    assert payload["frames"][1]["s"] == [5.0]     # 2 + 1 (chord) + 2


def test_build_payload_multiple_ep_pairs_each_individual():
    lef_cfg = LEFConfig(
        chain_length=400, num_chains=1, separation=400,   # single cohesin
        topology_kwargs={"tad_positions": [],
                         "genes": [
                             {"gene_id": 1, "tss": 90, "tes": 120, "enhancer_pos": 10},
                             {"gene_id": 2, "tss": 300, "tes": 330, "enhancer_pos": 380},
                         ]},
    )
    lef_cfg.plugins.topology = PluginSpec(
        target="polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_topology"
    )
    # frame 0: loop brackets pair 1 (10-90); frame 1: loop brackets pair 2 (300-380).
    pos = np.array([[[12, 88]], [[305, 378]]], dtype=np.int32)
    payload = build_payload(pos, ViewerConfig(stride=1, max_frames=0), lef_cfg)
    assert [p["label"] for p in payload["eps"]] == ["E0-P0", "E1-P1"]  # build_genes re-indexes
    assert all(len(f["s"]) == 2 for f in payload["frames"])   # one distance per pair
    # pair 1 close in frame 0, far in frame 1; pair 2 the reverse
    assert payload["frames"][0]["s"][0] < payload["frames"][1]["s"][0]
    assert payload["frames"][1]["s"][1] < payload["frames"][0]["s"][1]


def test_build_payload_shared_enhancer_labels_unique_enhancer_once():
    lef_cfg = LEFConfig(
        chain_length=450, num_chains=1, separation=450,
        topology_kwargs={"tad_positions": [150, 300],
                         "genes": [
                             {"tss": 180, "tes": 240, "enhancer_pos": 300},
                             {"tss": 380, "tes": 320, "enhancer_pos": 300},
                         ]},
    )
    lef_cfg.plugins.topology = PluginSpec(
        target="polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_topology"
    )
    pos = np.array([[[150, 390]]], dtype=np.int32)
    payload = build_payload(pos, ViewerConfig(stride=1, max_frames=0), lef_cfg)
    enhancers = [e for e in payload["elements"] if e["type"] == "enhancer"]
    assert enhancers == [{"position": 300, "type": "enhancer", "label": "E0"}]
    assert [p["label"] for p in payload["eps"]] == ["E0-P0", "E0-P1"]


def test_gene_aware_topology_can_replicate_genes_across_chains():
    cfg = LEFConfig(chain_length=450, num_chains=3, separation=450)
    args = topology_plugins.gene_aware_convergent_tad_topology(
        cfg,
        genes=[
            {"tss": 260, "tes": 290, "enhancer_pos": 155},
            {"tss": 220, "tes": 250, "enhancer_pos": 155},
        ],
        replicate_genes_across_chains=True,
    )

    sites = [(g.tss, g.tes, g.enhancer_pos) for g in args["genes"]]
    assert sites == [
        (260, 290, 155),
        (220, 250, 155),
        (710, 740, 605),
        (670, 700, 605),
        (1160, 1190, 1055),
        (1120, 1150, 1055),
    ]


def test_convergent_tad_boundary_strength_per_anchor():
    cfg = LEFConfig(chain_length=800, num_chains=1, separation=800)
    bs = {0: 0.1, 160: 0.2, 320: 0.3, 480: 0.4, 640: 0.5, 800: 0.6}
    args = topology_plugins.convergent_tad_topology(
        cfg,
        tad_positions=[160, 320, 480, 640],
        boundary_strength=bs,
        include_chromosome_ends=True,
    )
    # Left-facing anchor keyed by interval start position. The chromosome-start
    # anchor (site 0) is forced to a hard 1.0 wall, overriding any bs entry.
    assert args["ctcfCapture"][-1] == {0: 1.0, 160: 0.2, 320: 0.3, 480: 0.4, 640: 0.5}
    # Right-facing anchor sits at end - 1 but takes the end boundary's strength;
    # the chromosome-end anchor (site 799) is forced to a hard 1.0 wall.
    assert args["ctcfCapture"][1] == {159: 0.2, 319: 0.3, 479: 0.4, 639: 0.5, 799: 1.0}


def test_convergent_tad_boundary_strength_default_fallback():
    cfg = LEFConfig(chain_length=800, num_chains=1, separation=800)
    args = topology_plugins.convergent_tad_topology(
        cfg,
        tad_positions=[160, 320],
        boundary_strength={160: 0.9},  # only one anchor specified
        include_chromosome_ends=True,
        default_boundary_strength=0.05,
    )
    # Site 0 is the chromosome start -> forced 1.0; 160 from bs; 320 from default.
    assert args["ctcfCapture"][-1] == {0: 1.0, 160: 0.9, 320: 0.05}


def test_convergent_tad_boundary_strength_scalar_backward_compat():
    cfg = LEFConfig(chain_length=800, num_chains=1, separation=800)
    args = topology_plugins.convergent_tad_topology(
        cfg,
        tad_positions=[160, 320],
        boundary_strength=0.7,
        include_chromosome_ends=True,
    )
    # Interior anchors take the scalar 0.7; the two chromosome ends (sites 0 and
    # 799) are forced to hard 1.0 walls.
    assert args["ctcfCapture"][-1] == {0: 1.0, 160: 0.7, 320: 0.7}
    assert args["ctcfCapture"][1] == {159: 0.7, 319: 0.7, 799: 1.0}


# ---------------------------------------------------------------------------
# Golden regression guardrails for the per-TAD oriented-boundary refactor.
#
# These freeze the CURRENT behavior of the convergent topology + 1D dynamics so
# that the staged refactor (topology canonicaliser, tads: schema, /ctcf_anchors
# persistence) can be proven non-regressing. The convergent topology and the
# gene-aware-convergent + RNAPII path are the exact code paths being refactored;
# the lesion fields depend on the reconstructed tad_positions. If any of these
# hashes change, a refactor step altered behavior it must not.
# ---------------------------------------------------------------------------

def _positions_hash(h5_path) -> str:
    with h5py.File(h5_path, "r") as fh:
        pos = fh["positions"][:]
    return hashlib.sha256(np.ascontiguousarray(pos).tobytes()).hexdigest()[:16]


def _f64_hash(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def test_golden_convergent_cohesin_trajectory(tmp_path):
    """Plain cohesin (convergent topology, no RNAPII/lesions) trajectory is frozen."""
    h5 = tmp_path / "plain.h5"
    cfg_path = tmp_path / "plain.yaml"
    cfg_path.write_text(
        f"""
lef:
  chain_length: 300
  num_chains: 2
  separation: 50
  trajectory_length: 80
  warmup_steps: 20
  chunk_size: 10
  seed: 12345
  output_path: {h5}
  topology_kwargs:
    tad_positions: [100, 200]
    boundary_strength: {{100: 0.5, 200: 0.7}}
    default_boundary_strength: 0.3
    release_prob: 0.0
  plugins:
    topology: polychrom.pipelines.loop_extrusion.plugins.topology:convergent_tad_topology
"""
    )
    lef_stage.run(load_config(cfg_path).lef)
    assert _positions_hash(h5) == "c33a0bf3e6f40cd8"


def test_golden_gene_aware_convergent_rnapii_trajectory(tmp_path):
    """gene_aware_convergent topology + RNAPII trajectory is frozen (proves the
    boundary refactor does not perturb RNAPII dynamics)."""
    h5 = tmp_path / "rnapii.h5"
    cfg_path = tmp_path / "rnapii.yaml"
    cfg_path.write_text(
        f"""
lef:
  chain_length: 300
  num_chains: 1
  separation: 50
  trajectory_length: 80
  warmup_steps: 20
  chunk_size: 10
  seed: 999
  max_rnapii: 8
  output_path: {h5}
  topology_kwargs:
    tad_positions: [100, 200]
    boundary_strength: {{100: 0.5, 200: 0.7}}
    default_boundary_strength: 0.3
    release_prob: 0.0
    replicate_genes_across_chains: true
    rnapii_stride: 1
    genes:
      - {{tss: 40, tes: 90, load_prob: 0.05}}
      - {{tss: 250, tes: 160, load_prob: 0.05}}
  plugins:
    topology: polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_convergent_tad_topology
    load: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:load_targeted
    translocate: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:translocate_with_rnapii
    rnapii_load: polychrom.pipelines.loop_extrusion.plugins.rnapii:load_rnapii
    rnapii_translocate: polychrom.pipelines.loop_extrusion.plugins.rnapii:stateful_translocate_rnapii
"""
    )
    lef_stage.run(load_config(cfg_path).lef)
    assert _positions_hash(h5) == "8ad923dfa870611c"


def test_golden_lesion_fields_for_tad_positions():
    """Lesion placement/repair fields derived from tad_positions are frozen, so
    the tads: reconstruction can be proven to reproduce them for the no-gap case."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import precompute_lesion_fields

    gene = build_genes([{"tss": 120, "tes": 180}])[0]
    args = {"N": 400, "chain_length": 400, "num_chains": 1, "genes": [gene],
            "lesion_type_a_prob": 0.25}
    precompute_lesion_fields(args, tad_positions=[100, 250], gene_objs=[gene],
                             tad_size_exponent=1.0, tad_repair_exponent=1.0, spacing=8)
    assert args["lesion_target"] == 50
    assert _f64_hash(args["lesion_site_p"]) == "4b76fa9710050276"
    assert _f64_hash(args["lesion_rate_mult"]) == "4ff2ea7971b60f17"
    assert _f64_hash(args["gene_body_mask"]) == "6685f4dea786526a"


# ---------------------------------------------------------------------------
# Per-TAD oriented-boundary schema (tads:) — Phase 2.
# ---------------------------------------------------------------------------

def test_tads_schema_per_tad_anchor_placement():
    cfg = LEFConfig(chain_length=800, num_chains=1, separation=800)
    args = topology_plugins.convergent_tad_topology(
        cfg,
        tads=[
            {"left": 0, "right": 159, "left_strength": 0.2, "right_strength": 0.25},
            {"left": 200, "right": 399, "left_strength": 0.3, "right_strength": 0.35},
        ],
        include_chromosome_ends=True,
    )
    # Left anchors capture the -1 leg; site 0 is forced to a hard 1.0 wall.
    assert args["ctcfCapture"][-1] == {0: 1.0, 200: 0.3}
    # Right anchors capture the +1 leg; 399 is interior so keeps its strength.
    assert args["ctcfCapture"][1] == {159: 0.25, 399: 0.35}


def test_tads_schema_independent_side_strengths():
    cfg = LEFConfig(chain_length=500, num_chains=1, separation=500)
    args = topology_plugins.convergent_tad_topology(
        cfg,
        tads=[{"left": 100, "right": 300, "left_strength": 0.4, "right_strength": 0.9}],
    )
    assert args["ctcfCapture"][-1] == {100: 0.4}
    assert args["ctcfCapture"][1] == {300: 0.9}


def test_tads_schema_gap_has_no_anchors():
    cfg = LEFConfig(chain_length=800, num_chains=1, separation=800)
    args = topology_plugins.convergent_tad_topology(
        cfg,
        tads=[
            {"left": 0, "right": 159, "left_strength": 0.2, "right_strength": 0.25},
            {"left": 200, "right": 399, "left_strength": 0.3, "right_strength": 0.35},
        ],
    )
    occupied = set(args["ctcfCapture"][-1]) | set(args["ctcfCapture"][1])
    # Gap [160, 199] and trailing region [400, 799] carry no anchors.
    assert occupied.isdisjoint(range(160, 200))
    assert occupied.isdisjoint(range(400, 800))


def test_tads_default_strength_fallback():
    cfg = LEFConfig(chain_length=600, num_chains=1, separation=600)
    args = topology_plugins.convergent_tad_topology(
        cfg,
        tads=[{"left": 100, "right": 200}],  # strengths omitted
        default_boundary_strength=0.07,
    )
    assert args["ctcfCapture"][-1] == {100: 0.07}
    assert args["ctcfCapture"][1] == {200: 0.07}


def test_tads_equivalent_to_legacy_tad_positions():
    """A gap-free tads layout reproduces the legacy tad_positions output exactly."""
    cfg = LEFConfig(chain_length=800, num_chains=2, separation=400)
    legacy = topology_plugins.convergent_tad_topology(
        cfg,
        tad_positions=[160, 320],
        boundary_strength={160: 0.2, 320: 0.3},
        default_boundary_strength=0.05,
    )
    tads = topology_plugins.convergent_tad_topology(
        cfg,
        tads=[
            {"left": 0, "right": 159, "left_strength": 1.0, "right_strength": 0.2},
            {"left": 160, "right": 319, "left_strength": 0.2, "right_strength": 0.3},
            {"left": 320, "right": 799, "left_strength": 0.3, "right_strength": 1.0},
        ],
    )
    assert tads["ctcfCapture"] == legacy["ctcfCapture"]
    assert tads["ctcfRelease"] == legacy["ctcfRelease"]


def test_tads_reconstructs_boundaries_for_downstream():
    cfg = LEFConfig(chain_length=800, num_chains=1, separation=800)
    nogap = topology_plugins.convergent_tad_topology(
        cfg,
        tads=[
            {"left": 0, "right": 159},
            {"left": 160, "right": 319},
            {"left": 320, "right": 799},
        ],
    )
    # Abutting layout reconstructs exactly the legacy interior boundaries.
    assert nogap["tad_positions"] == [160, 320]

    gapped = topology_plugins.convergent_tad_topology(
        cfg,
        tads=[{"left": 0, "right": 159}, {"left": 200, "right": 399}],
    )
    # Both edges of the gap appear, keeping the chain fully tiled by segments.
    assert gapped["tad_positions"] == [160, 200, 400]


def test_tads_and_tad_positions_are_mutually_exclusive():
    cfg = LEFConfig(chain_length=400, num_chains=1, separation=400)
    with pytest.raises(ValueError):
        topology_plugins.convergent_tad_topology(
            cfg, tad_positions=[100], tads=[{"left": 0, "right": 99}]
        )


def test_tads_reject_overlap():
    cfg = LEFConfig(chain_length=400, num_chains=1, separation=400)
    with pytest.raises(ValueError):
        topology_plugins.convergent_tad_topology(
            cfg, tads=[{"left": 0, "right": 150}, {"left": 100, "right": 200}]
        )


def test_tads_reject_out_of_range():
    cfg = LEFConfig(chain_length=400, num_chains=1, separation=400)
    with pytest.raises(ValueError):
        topology_plugins.convergent_tad_topology(
            cfg, tads=[{"left": 300, "right": 400}]  # right == chain_length (out of range)
        )


def test_tads_lesion_fields_match_legacy_no_gap():
    """gene_aware_convergent with a gap-free tads layout reproduces the lesion
    placement/repair fields of the equivalent legacy tad_positions config."""
    cfg = LEFConfig(chain_length=400, num_chains=1, separation=400)
    common = dict(
        lesion_spacing=8, lesion_type_a_prob=0.25,
        lesion_prerecognition_ticks=50, lesion_repair_ticks=50,
        lesion_tad_size_exponent=1.0, lesion_tad_repair_exponent=1.0,
    )
    legacy = topology_plugins.gene_aware_convergent_tad_topology(
        cfg, tad_positions=[100, 250],
        boundary_strength={100: 0.2, 250: 0.3}, default_boundary_strength=0.1,
        **common,
    )
    tads = topology_plugins.gene_aware_convergent_tad_topology(
        cfg,
        tads=[
            {"left": 0, "right": 99, "left_strength": 1.0, "right_strength": 0.2},
            {"left": 100, "right": 249, "left_strength": 0.2, "right_strength": 0.3},
            {"left": 250, "right": 399, "left_strength": 0.3, "right_strength": 1.0},
        ],
        **common,
    )
    assert tads["tad_positions"] == [100, 250]
    assert np.array_equal(tads["lesion_site_p"], legacy["lesion_site_p"])
    assert np.array_equal(tads["lesion_rate_mult"], legacy["lesion_rate_mult"])


def test_tads_gap_lesion_fields_are_finite_and_normalised():
    """With a gap, the reconstructed full tiling keeps lesion fields well-defined
    (no uninitialised np.empty leaking into placement/repair)."""
    cfg = LEFConfig(chain_length=400, num_chains=1, separation=400)
    args = topology_plugins.gene_aware_convergent_tad_topology(
        cfg,
        tads=[{"left": 0, "right": 99}, {"left": 200, "right": 399}],
        lesion_spacing=8, lesion_tad_size_exponent=1.0, lesion_tad_repair_exponent=1.0,
    )
    site_p = np.asarray(args["lesion_site_p"], dtype=float)
    rate_mult = np.asarray(args["lesion_rate_mult"], dtype=float)
    assert np.all(np.isfinite(site_p)) and np.all(site_p >= 0)
    assert site_p.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(rate_mult)) and np.all(rate_mult > 0)


# ---------------------------------------------------------------------------
# /ctcf_anchors H5 persistence — Phase 3.
# ---------------------------------------------------------------------------

def test_h5_ctcf_anchor_roundtrip(tmp_path):
    h5 = tmp_path / "anchors.h5"
    cfg_path = tmp_path / "anchors.yaml"
    cfg_path.write_text(
        f"""
lef:
  chain_length: 200
  num_chains: 2
  separation: 50
  trajectory_length: 30
  chunk_size: 10
  seed: 3
  output_path: {h5}
  topology_kwargs:
    tads:
      - {{left: 0, right: 79, left_strength: 0.4, right_strength: 0.5}}
      - {{left: 100, right: 199, left_strength: 0.6, right_strength: 0.7}}
  plugins:
    topology: polychrom.pipelines.loop_extrusion.plugins.topology:convergent_tad_topology
"""
    )
    lef_stage.run(load_config(cfg_path).lef)

    with h5py.File(h5, "r") as fh:
        assert "ctcf_anchors" in fh
        a = fh["ctcf_anchors"][:]
        assert fh.attrs["boundary_model"] == "per_tad_oriented"
        assert int(fh.attrs["ctcf_anchor_schema_version"]) == 1
        assert int(fh.attrs["ctcf_anchors_per_chain"]) == 4

    assert set(a.dtype.names) == {
        "abs_position", "chain_position", "chain_index",
        "side", "strength", "tad_index", "edge",
    }
    assert len(a) == 8  # 4 anchors/chain x 2 chains
    # Absolute position indexes directly into the (folded) positions dataset.
    assert np.array_equal(a["abs_position"], a["chain_position"] + a["chain_index"] * 200)
    # Chain-0 ground truth: site 0 forced 1.0 (-1), 79 -> 0.5 (+1),
    # 100 -> 0.6 (-1), 199 forced 1.0 (+1).
    c0 = a[a["chain_index"] == 0]
    by_site = {int(r["chain_position"]): r for r in c0}

    def check(site, side, strength, tad_index, edge):
        r = by_site[site]
        assert (int(r["side"]), int(r["tad_index"]), int(r["edge"])) == (side, tad_index, edge)
        assert float(r["strength"]) == pytest.approx(strength, abs=1e-6)

    check(0, -1, 1.0, 0, 0)        # chromosome start forced to 1.0
    check(79, 1, 0.5, 0, 1)
    check(100, -1, 0.6, 1, 0)
    check(199, 1, 1.0, 1, 1)       # chromosome end forced to 1.0


def test_h5_no_ctcf_anchors_for_uniform_topology(tmp_path):
    """Uniform (symmetric) topology exposes no anchor table -> dataset omitted."""
    h5 = tmp_path / "uniform.h5"
    cfg_path = tmp_path / "uniform.yaml"
    cfg_path.write_text(
        f"""
lef:
  chain_length: 200
  num_chains: 1
  separation: 60
  trajectory_length: 30
  chunk_size: 10
  seed: 7
  output_path: {h5}
  topology_kwargs:
    tad_positions: [50, 150]
    capture_prob: 0.9
    release_prob: 0.01
    symmetric: true
"""
    )
    lef_stage.run(load_config(cfg_path).lef)
    with h5py.File(h5, "r") as fh:
        assert "ctcf_anchors" not in fh
        assert "boundary_model" not in fh.attrs


# ---------------------------------------------------------------------------
# Package consumers route boundaries through one helper — Phase 4.
# ---------------------------------------------------------------------------

def test_boundaries_from_topology_kwargs_both_schemas():
    from polychrom.pipelines.loop_extrusion import annotate

    f = annotate.boundaries_from_topology_kwargs
    assert f({"tad_positions": [160, 320]}, 800) == [160, 320]
    nogap = {"tads": [
        {"left": 0, "right": 159}, {"left": 160, "right": 319}, {"left": 320, "right": 799},
    ]}
    assert f(nogap, 800) == [160, 320]               # gap-free == legacy interior bounds
    gapped = {"tads": [{"left": 0, "right": 159}, {"left": 200, "right": 399}]}
    assert f(gapped, 800) == [160, 200, 400]         # both gap edges retained
    assert f({}, 800) == [] and f(None, 800) == []   # robust to missing/invalid


def test_viewer_derive_tads_handles_tads_schema():
    legacy = LEFConfig(chain_length=800, num_chains=1,
                       topology_kwargs={"tad_positions": [160, 320]})
    tads_cfg = LEFConfig(chain_length=800, num_chains=1, topology_kwargs={"tads": [
        {"left": 0, "right": 159}, {"left": 160, "right": 319}, {"left": 320, "right": 799},
    ]})
    legacy_iv = [(t["start"], t["end"]) for t in viewer_stage.derive_tads(legacy)]
    tads_iv = [(t["start"], t["end"]) for t in viewer_stage.derive_tads(tads_cfg)]
    assert legacy_iv == tads_iv == [(0, 160), (160, 320), (320, 800)]


def test_annotate_from_lef_cfg_reads_tads_boundaries():
    from polychrom.pipelines.loop_extrusion import annotate

    cfg = LEFConfig(chain_length=800, num_chains=1, topology_kwargs={"tads": [
        {"left": 0, "right": 159}, {"left": 160, "right": 319}, {"left": 320, "right": 799},
    ]})
    ann = annotate.from_lef_cfg(cfg)
    assert ann["boundaries"] == [160, 320]


def test_contacts_can_replicate_map_starts_across_chains():
    contacts_cfg = ContactsConfig(
        map_starts=[0],
        map_size=450,
        replicate_map_starts_across_chains=True,
    )
    lef_cfg = LEFConfig(chain_length=450, num_chains=3, separation=450)

    starts = contacts_stage._effective_map_starts(contacts_cfg, lef_cfg)

    assert starts == [0, 450, 900]


def test_paper_force_builder_ep_pairs_support_shared_enhancer(monkeypatch):
    captured = {}

    class DummySim:
        N = 500

        def add_force(self, force):
            pass

    def fake_polymer_chains(sim, **kwargs):
        nb_kwargs = kwargs["nonbonded_force_kwargs"]
        captured["monomer_types"] = nb_kwargs["monomerTypes"]
        captured["interaction_matrix"] = nb_kwargs["interactionMatrix"]
        return object()

    monkeypatch.setattr(force_plugins.forcekits, "polymer_chains", fake_polymer_chains)
    monkeypatch.setattr(force_plugins.forces, "spherical_confinement", lambda *args, **kwargs: object())

    force_plugins.paper_force_builder(
        DummySim(),
        num_chains=1,
        chain_length=500,
        ep_pairs=[[155, 260], [155, 220]],
    )

    monomer_types = captured["monomer_types"]
    interaction_matrix = captured["interaction_matrix"]
    enhancer_type = monomer_types[155]
    promoter_a_type = monomer_types[260]
    promoter_b_type = monomer_types[220]

    assert enhancer_type != 0
    assert promoter_a_type != 0
    assert promoter_b_type != 0
    assert interaction_matrix[enhancer_type, promoter_a_type] == 1.0
    assert interaction_matrix[enhancer_type, promoter_b_type] == 1.0
    assert interaction_matrix[promoter_a_type, promoter_b_type] == 0.0


def test_paper_force_builder_can_replicate_ep_pairs_across_chains(monkeypatch):
    captured = {}

    class DummySim:
        N = 1350

        def add_force(self, force):
            pass

    def fake_polymer_chains(sim, **kwargs):
        nb_kwargs = kwargs["nonbonded_force_kwargs"]
        captured["monomer_types"] = nb_kwargs["monomerTypes"]
        captured["interaction_matrix"] = nb_kwargs["interactionMatrix"]
        return object()

    monkeypatch.setattr(force_plugins.forcekits, "polymer_chains", fake_polymer_chains)
    monkeypatch.setattr(force_plugins.forces, "spherical_confinement", lambda *args, **kwargs: object())

    force_plugins.paper_force_builder(
        DummySim(),
        num_chains=3,
        chain_length=450,
        ep_pairs=[[155, 260], [155, 220]],
        replicate_ep_pairs_across_chains=True,
    )

    monomer_types = captured["monomer_types"]
    interaction_matrix = captured["interaction_matrix"]
    expected_pairs = [
        (155, 260),
        (155, 220),
        (605, 710),
        (605, 670),
        (1055, 1160),
        (1055, 1120),
    ]
    for enhancer, promoter in expected_pairs:
        assert interaction_matrix[monomer_types[enhancer], monomer_types[promoter]] == 1.0


def test_paper_force_builder_can_restrict_nonbonded_to_chains(monkeypatch):
    captured = {}

    class DummyForce:
        name = "dummy_nonbonded"

        def __init__(self):
            self.groups = []

        def addInteractionGroup(self, set1, set2):
            self.groups.append((tuple(sorted(set1)), tuple(sorted(set2))))
            return len(self.groups) - 1

    class DummySim:
        N = 12

        def add_force(self, force):
            pass

    def fake_heteropolymer_ssw(sim, **kwargs):
        force = DummyForce()
        captured["force"] = force
        return force

    def fake_polymer_chains(sim, **kwargs):
        nb_force = kwargs["nonbonded_force_func"](
            sim,
            **kwargs["nonbonded_force_kwargs"],
        )
        captured["groups"] = nb_force.groups
        return object()

    monkeypatch.setattr(force_plugins.forces, "heteropolymer_SSW", fake_heteropolymer_ssw)
    monkeypatch.setattr(force_plugins.forcekits, "polymer_chains", fake_polymer_chains)
    monkeypatch.setattr(force_plugins.forces, "spherical_confinement", lambda *args, **kwargs: object())

    force_plugins.paper_force_builder(
        DummySim(),
        num_chains=3,
        chain_length=4,
        ep_pairs=[[1, 2]],
        replicate_ep_pairs_across_chains=True,
        restrict_nonbonded_to_chains=True,
    )

    assert captured["groups"] == [
        ((0, 1, 2, 3), (0, 1, 2, 3)),
        ((4, 5, 6, 7), (4, 5, 6, 7)),
        ((8, 9, 10, 11), (8, 9, 10, 11)),
    ]


def test_paper_force_builder_can_constrain_each_replicate_chain(monkeypatch):
    captured = []

    class DummyForce:
        def __init__(self, name):
            self.name = name

    class DummySim:
        N = 12

        def add_force(self, force):
            pass

    def fake_polymer_chains(sim, **kwargs):
        return DummyForce("polymer")

    def fake_spherical_confinement(sim, **kwargs):
        captured.append(kwargs)
        return DummyForce(kwargs["name"])

    monkeypatch.setattr(force_plugins.forcekits, "polymer_chains", fake_polymer_chains)
    monkeypatch.setattr(force_plugins.forces, "spherical_confinement", fake_spherical_confinement)

    force_plugins.paper_force_builder(
        DummySim(),
        num_chains=3,
        chain_length=4,
        ep_pairs=[],
        confinement_density=0.2,
        confinement_per_chain=True,
    )

    expected_radius = (3 * 4 / (4 * np.pi * 0.2)) ** (1.0 / 3.0)
    assert [call["name"] for call in captured] == [
        "spherical_confinement_chain_0",
        "spherical_confinement_chain_1",
        "spherical_confinement_chain_2",
    ]
    assert [list(call["particles"]) for call in captured] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
    assert all(np.isclose(call["r"], expected_radius) for call in captured)


def test_build_payload_promoter_direction_follows_gene(tmp_path):
    lef_cfg = LEFConfig(
        chain_length=400, num_chains=1, separation=400,
        topology_kwargs={"tad_positions": [],
                         "genes": [
                             {"tss": 90, "tes": 120, "enhancer_pos": 10},   # forward
                             {"tss": 330, "tes": 300, "enhancer_pos": 380},  # reverse
                         ]},
    )
    lef_cfg.plugins.topology = PluginSpec(
        target="polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_topology"
    )
    pos = np.array([[[12, 88]]], dtype=np.int32)
    payload = build_payload(pos, ViewerConfig(stride=1, max_frames=0), lef_cfg)
    promoters = {p["position"]: p["direction"]
                 for p in payload["elements"] if p["type"] == "promoter"}
    assert promoters[90] == 1     # TSS<TES -> chevron points right
    assert promoters[330] == -1   # TSS>TES -> chevron points left


def test_build_visited_heatmap_is_symmetric_cumulative():
    lef_cfg = _plain_lef_cfg()
    pos = np.array([[[10, 90], [30, 60], [120, 170]]] * 4, dtype=np.int32)
    payload = build_payload(pos, ViewerConfig(stride=1, max_frames=0), lef_cfg)
    mat = build_visited_heatmap(payload)
    assert mat.shape == (200, 200)
    assert mat[10, 90] == 4 and mat[90, 10] == 4     # accumulated over 4 frames
    assert (mat == mat.T).all()                       # symmetric
    assert mat.sum() == 3 * 4 * 2                      # 3 cohesins x 4 frames x 2 (sym)


def test_build_elements_export_carries_annotations():
    lef_cfg = _plain_lef_cfg()
    pos = np.array([[[10, 90], [30, 60], [120, 170]]] * 2, dtype=np.int32)
    payload = build_payload(pos, ViewerConfig(stride=1, max_frames=0), lef_cfg)
    export = build_elements_export(payload)
    assert set(export) == {"title", "latticeSize", "siteOffset",
                           "ctcfSites", "elements", "genes", "tads", "eps"}
    assert "frames" not in export                      # trajectory excluded
    assert sorted(export["ctcfSites"]) == [50, 150]


def test_viewer_run_writes_companion_exports(tmp_path):
    h5 = tmp_path / "LEFPositions.h5"
    pos = np.array([[[10, 90], [30, 60], [120, 170]]] * 4, dtype=np.int32)
    with h5py.File(h5, "w") as fh:
        fh.create_dataset("positions", data=pos)
    out = tmp_path / "viewer.html"
    vcfg = ViewerConfig(lef_positions_path=str(h5), output_path=str(out), max_frames=0)
    viewer_stage.run(vcfg, _plain_lef_cfg())

    heatmap = tmp_path / "viewer_visited_heatmap.npy"
    elements = tmp_path / "viewer_elements.json"
    assert heatmap.exists() and elements.exists()
    mat = np.load(heatmap)
    assert mat.shape == (200, 200) and mat[10, 90] == 4
    import json as _json
    meta = _json.loads(elements.read_text())
    assert meta["latticeSize"] == 200 and "frames" not in meta


def test_viewer_run_emits_self_contained_html(tmp_path):
    h5 = tmp_path / "LEFPositions.h5"
    pos = np.array([[[10, 90], [30, 60], [120, 170]]] * 4, dtype=np.int32)
    with h5py.File(h5, "w") as fh:
        fh.create_dataset("positions", data=pos)
    out = tmp_path / "viewer.html"
    vcfg = ViewerConfig(lef_positions_path=str(h5), output_path=str(out), max_frames=0)
    returned = viewer_stage.run(vcfg, _plain_lef_cfg())
    assert returned == out and out.exists()
    html = out.read_text()
    assert "__VIEWER_DATA__" not in html               # placeholder substituted
    assert "Understanding Cohesin Bridging" in html
    assert "Bridge Map" in html and "Kymograph" in html
    assert "const DATA = {" in html


def test_viewer_run_builds_h5_when_missing(tmp_path):
    h5 = tmp_path / "LEFPositions.h5"
    out = tmp_path / "viewer.html"
    lef_cfg = LEFConfig(
        chain_length=200, num_chains=1, separation=60,
        trajectory_length=40, chunk_size=10, seed=3,
        output_path=str(h5),
        topology_kwargs={"tad_positions": [50, 150], "capture_prob": 0.9,
                         "release_prob": 0.01, "symmetric": True},
    )
    vcfg = ViewerConfig(lef_positions_path=str(h5), output_path=str(out), max_frames=0)
    assert not h5.exists()
    viewer_stage.run(vcfg, lef_cfg)
    assert h5.exists()                                   # built by the viewer stage
    assert out.exists()
    assert "Understanding Cohesin Bridging" in out.read_text()


# --------------------------------------------------------------------------- #
# Pol II transcription-driven compaction (3D self-affinity)
# --------------------------------------------------------------------------- #
class _RecordingNB:
    """Fake heteropolymer nonbonded force recording per-particle type writes."""

    name = "heteropolymer_SSW"

    def __init__(self):
        self.params: dict = {}
        self.context_updates = 0

    def setParticleParameters(self, idx, params):
        self.params[int(idx)] = tuple(params)

    def updateParametersInContext(self, _context):
        self.context_updates += 1


def test_gene_bodies_span_tss_to_tes_either_direction():
    genes = np.array(
        [(0, 250, 320, 1, 0.04), (1, 480, 400, -1, 0.02)],
        dtype=[("gene_id", "i4"), ("tss", "i4"), ("tes", "i4"),
               ("direction", "i4"), ("load_prob", "f4")],
    )
    bodies = _gene_bodies(genes)
    assert bodies[0] == set(range(250, 321))
    assert bodies[1] == set(range(400, 481))   # min/max regardless of direction


def test_paper_force_builder_adds_polii_self_affinity(monkeypatch):
    captured = {}
    fake_nb = _RecordingNB()

    class DummySim:
        N = 800

        def __init__(self):
            self.force_dict = {}

        def add_force(self, force):
            self.force_dict["heteropolymer_SSW"] = fake_nb

    def fake_polymer_chains(sim, **kwargs):
        nb_kwargs = kwargs["nonbonded_force_kwargs"]
        captured["monomer_types"] = nb_kwargs["monomerTypes"]
        captured["interaction_matrix"] = nb_kwargs["interactionMatrix"]
        return object()

    monkeypatch.setattr(force_plugins.forcekits, "polymer_chains", fake_polymer_chains)
    monkeypatch.setattr(force_plugins.forces, "spherical_confinement", lambda *a, **k: object())

    force_plugins.paper_force_builder(
        DummySim(),
        num_chains=1,
        chain_length=800,
        ep_pairs=[[500, 250]],
        transcribed_particles=list(range(250, 321)),  # gene0 body incl P=250
        polii_self_affinity=2.0,
    )

    mt = captured["monomer_types"]
    im = captured["interaction_matrix"]
    polii_type = mt[300]                       # interior gene-body monomer
    assert polii_type != 0
    assert im[polii_type, polii_type] == 2.0   # self-attraction baked in
    # P=250 coincides with an E/P sticky site -> excluded from Pol II candidates
    assert mt[250] != polii_type
    # candidates switched OFF (type 0) after the force is built; E/P site untouched
    assert fake_nb.params[300] == (0.0, 0.0)
    assert 250 not in fake_nb.params


def test_polii_self_affinity_forbids_cross_chain_replicates():
    class DummySim:
        N = 800

    with pytest.raises(ValueError):
        force_plugins.paper_force_builder(
            DummySim(),
            num_chains=2,
            chain_length=400,
            transcribed_particles=[10, 11],
            polii_self_affinity=1.0,
            restrict_nonbonded_to_chains=False,    # would model across chains
        )


def test_sticky_updater_activates_gene_body_only_when_elongating():
    gene_bodies = {0: {10, 11, 12, 13, 14}, 1: {30, 31, 32}}
    candidates = sorted(gene_bodies[0] | gene_bodies[1])
    rpos = np.array(
        [
            [[10, 0], [-1, -1]],   # frame 0: RNAPII on gene 0
            [[31, 1], [-1, -1]],   # frame 1: RNAPII on gene 1
        ],
        dtype=np.int32,
    )
    rstate = np.array(
        [
            [STATE_ELONGATING, -1],   # gene 0 elongating
            [STATE_PAUSED, -1],       # gene 1 paused -> not transcribing
        ],
        dtype=np.int8,
    )
    force = _RecordingNB()
    su = StickyUpdater(
        rpos, rstate, gene_bodies,
        polii_type=3, candidates=candidates, extra_hard=set(),
    )

    active0 = su.setup(force, blocks=2, context=object())
    assert active0 == {10, 11, 12, 13, 14}
    assert all(force.params[m][0] == 3.0 for m in gene_bodies[0])
    assert force.context_updates == 1

    active1 = su.step(object())
    assert active1 == set()                       # gene1 paused, gene0 gone
    assert all(force.params[m][0] == 0.0 for m in gene_bodies[0])
    assert force.context_updates == 2


# --------------------------------------------------------------------------- #
# Multi-enhancer genes (shadow / super-enhancer integration)
# --------------------------------------------------------------------------- #
def test_build_genes_accepts_enhancer_list_and_legacy_scalar():
    g_list = build_genes([{"tss": 10, "tes": 50, "enhancers": [5, 90, 120]}])[0]
    assert g_list.enhancers == (5, 90, 120)
    assert g_list.enhancer_pos == 5                  # legacy mirror = first
    assert g_list.enhancer_logic == "additive"       # empirical default
    g_scalar = build_genes([{"tss": 10, "tes": 50, "enhancer_pos": 7}])[0]
    assert g_scalar.enhancers == (7,)
    assert g_scalar.enhancer_pos == 7


def test_build_genes_rejects_requires_enhancer_without_enhancers():
    with pytest.raises(ValueError):
        build_genes([{"tss": 10, "tes": 50, "requires_enhancer": True}])
    with pytest.raises(ValueError):
        build_genes([{"tss": 10, "tes": 50, "enhancer_logic": "bogus", "enhancers": [5]}])


def test_gene_post_init_syncs_scalar_constructor():
    # Direct Gene(enhancer_pos=...) construction (used by older tests) still
    # populates the canonical enhancers tuple.
    g = Gene(gene_id=0, tss=2, tes=6, direction=1, load_prob=1.0, enhancer_pos=5)
    assert g.enhancers == (5,)


def test_compute_ep_contacts_counts_enhancers_in_contact():
    # TSS=100, enhancers at 50 (bracketed by 40-110) and 900 (not).
    gene = build_genes([{"tss": 100, "tes": 130, "enhancers": [50, 900]}])[0]
    cohesins = [Cohesin(Leg(40), Leg(110))]
    contacts = compute_ep_contacts(cohesins, [gene], tolerance=2)
    assert contacts == {0: 1}                         # one of two enhancers in loop
    # both enhancers bracketed -> count 2
    cohesins2 = [Cohesin(Leg(40), Leg(950))]
    assert compute_ep_contacts(cohesins2, [gene], tolerance=2) == {0: 2}
    # none bracketed -> gene absent (membership still tests "has contact")
    assert compute_ep_contacts([Cohesin(Leg(60), Leg(80))], [gene]) == {}


def test_enhancer_factor_logics():
    base = dict(gene_id=0, tss=10, tes=50, direction=1, load_prob=1.0)
    any_g = Gene(**base, enhancers=(1, 2, 3), enhancer_logic="any")
    all_g = Gene(**base, enhancers=(1, 2, 3), enhancer_logic="all")
    add_g = Gene(**base, enhancers=(1, 2, 3), enhancer_logic="additive")
    syn_g = Gene(**base, enhancers=(1, 2, 3), enhancer_logic="synergistic",
                 enhancer_synergy=2.0)
    # gate closed when nothing in contact
    for g in (any_g, all_g, add_g, syn_g):
        assert enhancer_factor(g, 0) == 0.0
    assert enhancer_factor(any_g, 1) == 1.0
    assert enhancer_factor(all_g, 2) == 0.0           # needs all 3
    assert enhancer_factor(all_g, 3) == 1.0
    assert enhancer_factor(add_g, 2) == 2.0           # linear dosage
    assert enhancer_factor(add_g, 5) == 3.0           # capped at k
    assert enhancer_factor(syn_g, 2) == 4.0           # 2 ** 2
    # single-enhancer gene == old boolean behaviour for every logic
    for logic in ("any", "all", "additive", "synergistic"):
        g1 = Gene(**base, enhancers=(1,), enhancer_logic=logic)
        assert enhancer_factor(g1, 0) == 0.0
        assert enhancer_factor(g1, 1) == 1.0


def test_load_rnapii_additive_dosage_scales_recruitment():
    # additive 2-in-contact doubles load_prob (0.3 -> 0.6); single -> 0.3.
    # seed(0) -> first np.random.random() == 0.5488: loads at 0.6, not at 0.3.
    gene = build_genes([{"tss": 2, "tes": 6, "load_prob": 0.3,
                         "enhancers": [1, 9], "load_requires_enhancer": True}])[0]
    for n_contact, should_load in [(2, True), (1, False)]:
        np.random.seed(0)
        occupied = np.zeros(10, dtype=np.int8)
        rnapiis = []
        args = {"genes": [gene], "rnapii_by_pos": {},
                "current_ep_contacts": {0: n_contact}}
        load_rnapii(rnapiis, occupied, args)
        assert (len(rnapiis) == 1) is should_load


def test_stateful_pause_release_or_vs_all_logic():
    # One enhancer in contact (50-TSS100); the other (900) is not.
    specs = {"tss": 100, "tes": 130, "enhancers": [50, 900],
             "requires_enhancer": True, "pause_release_prob": 1.0,
             "elongation_step_prob": 0.0}
    cohesins = [Cohesin(Leg(40), Leg(110))]          # brackets only enhancer 50
    occupied = np.zeros(1000, dtype=np.int8)

    def run(logic):
        gene = build_genes([{**specs, "enhancer_logic": logic}])[0]
        r = RNAPII(pos=100, gene_id=0, direction=1)
        r.attrs["state"] = STATE_PAUSED
        args = {"genes": [gene], "ep_contact_tolerance": 2,
                "rnapii_by_pos": {100: r}, "cohesin_leg_by_pos": {},
                "N": 1000, "chain_length": 1000}
        stateful_translocate_rnapii([r], cohesins, occupied, args)
        return r.attrs["state"]

    assert run("any") == STATE_ELONGATING            # redundant: 1 suffices
    assert run("all") == STATE_PAUSED                 # obligate: needs both


def test_build_payload_multi_enhancer_gene_emits_arc_per_enhancer():
    lef_cfg = LEFConfig(
        chain_length=400, num_chains=1, separation=400,
        topology_kwargs={"tad_positions": [],
                         "genes": [{"tss": 200, "tes": 240,
                                    "enhancers": [20, 360]}]},
    )
    lef_cfg.plugins.topology = PluginSpec(
        target="polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_topology"
    )
    pos = np.array([[[18, 202]]], dtype=np.int32)
    payload = build_payload(pos, ViewerConfig(stride=1, max_frames=0), lef_cfg)
    assert [p["label"] for p in payload["eps"]] == ["E0-P0", "E1-P0"]
    assert all(p["geneId"] == 0 for p in payload["eps"])
    enh = sorted(e["position"] for e in payload["elements"] if e["type"] == "enhancer")
    assert enh == [20, 360]


def test_expand_genes_offsets_enhancer_list_across_chains():
    cfg = LEFConfig(chain_length=500, num_chains=2, separation=500)
    args = topology_plugins.gene_aware_convergent_tad_topology(
        cfg,
        genes=[{"tss": 200, "tes": 250, "enhancers": [100, 300]}],
        replicate_genes_across_chains=True,
    )
    assert [g.enhancers for g in args["genes"]] == [(100, 300), (600, 800)]
