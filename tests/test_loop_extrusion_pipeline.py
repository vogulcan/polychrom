import h5py
import numpy as np
import pytest

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
    translocate,
    translocate_with_rnapii,
)
from polychrom.pipelines.loop_extrusion.plugins.rnapii import (
    Gene,
    RNAPII,
    STATE_ELONGATING,
    STATE_PAUSED,
    STATE_POISED,
    translocate_rnapii,
)
from polychrom.pipelines.loop_extrusion.plugins import (
    forces as force_plugins,
    topology as topology_plugins,
)
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
polymer:
  seed: 19
"""
    )
    cfg = load_config(cfg_path)
    assert cfg.lef.warmup_steps == 123
    assert cfg.lef.seed == 17
    assert cfg.polymer.seed == 19


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
    poised_block=1.0,
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
        "rnapii_poised_block_prob": poised_block,
        "rnapii_paused_block_prob": paused_block,
        "rnapii_elongating_block_prob": elongating_block,
    }
    if block is not None:
        args["rnapii_block_prob"] = block
    return occupied, cohesins, args


def test_poised_rnapii_block_probability_controls_cohesin_bypass():
    occupied, cohesins, args = _cohesin_hits_rnapii_args(
        STATE_POISED,
        poised_block=0.0,
        paused_block=1.0,
        elongating_block=1.0,
    )

    translocate_with_rnapii(cohesins, occupied, args, unload_prob_fn=lambda *_: 0.0)

    assert cohesins[0].left.pos == 2
    assert not cohesins[0].left.attrs["stalled"]


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
