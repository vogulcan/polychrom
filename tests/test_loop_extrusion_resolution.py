"""Coarsened-resolution (N-monomer) analysis in the qc + compare stages.

The default per-monomer (resolution 1) ICE'd-map analysis must be preserved
unchanged, while every configured resolution ``N > 1`` additionally bins the raw
contact map ``N x N``, ICE-balances it, and runs the full 3D analysis on the
coarsened map.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from polychrom.pipelines.loop_extrusion import compare, qc
from polychrom.pipelines.loop_extrusion.config import (
    ContactsConfig,
    LEFConfig,
    PipelineConfig,
    PolymerConfig,
)

N = 200
CHAIN = 200
BOUNDARIES = [70, 130]


def test_coarsen_contact_map_sums_blocks():
    m = np.arange(100, dtype=float).reshape(10, 10)
    c = qc.coarsen_contact_map(m, 5)
    assert c.shape == (2, 2)
    # each output bin is the sum of its 5x5 source block
    assert c[0, 0] == m[:5, :5].sum()
    assert c[1, 1] == m[5:, 5:].sum()
    # total contacts are conserved when the size divides evenly
    assert c.sum() == m.sum()
    # factor <= 1 is the identity
    assert np.array_equal(qc.coarsen_contact_map(m, 1), m)


def test_coarsen_contact_map_drops_incomplete_tail():
    m = np.ones((23, 23), dtype=float)
    c = qc.coarsen_contact_map(m, 10)  # 23 // 10 -> 2 bins, last 3 rows/cols dropped
    assert c.shape == (2, 2)
    assert np.all(c == 100.0)


def test_scale_resolution_coords_identity_at_factor_1():
    sc = qc.scale_resolution_coords([70, 130], [40], [55], [12], factor=1, n_bins=200)
    assert sc["boundaries"] == [70, 130]
    assert sc["tads"] == [(0, 70), (70, 130), (130, 200)]
    assert sc["gene_tss"] == [40] and sc["gene_tes"] == [55]
    assert sc["lesion_sites"] == [12]


def test_scale_resolution_coords_bins_and_dedups():
    # 71 and 75 collapse into the same bin (7) at factor 10
    sc = qc.scale_resolution_coords([71, 75, 130], [44], [58], None, factor=10, n_bins=20)
    assert sc["boundaries"] == [7, 13]
    assert sc["tads"] == [(0, 7), (7, 13), (13, 20)]
    assert sc["gene_tss"] == [4] and sc["gene_tes"] == [5]
    assert sc["lesion_sites"] is None


def test_config_resolution_list_always_includes_native():
    assert ContactsConfig().resolution_list == [1, 10]
    assert ContactsConfig(analysis_resolutions=[10]).resolution_list == [1, 10]
    assert ContactsConfig(analysis_resolutions=[1, 5, 10, 10]).resolution_list == [1, 5, 10]
    assert ContactsConfig(analysis_resolutions=[1]).resolution_list == [1]


def _synth_map(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i = np.arange(N)
    d = np.abs(i[:, None] - i[None, :])
    m = 100.0 / (1.0 + d)
    for a, b in [(0, 70), (70, 130), (130, 200)]:
        m[a:b, a:b] += 30.0 / (1.0 + d[a:b, a:b])
    m += rng.random((N, N)) * 0.5
    return (m + m.T) / 2


def _make_run(dirp: Path, seed: int) -> None:
    dirp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with h5py.File(dirp / "LEFPositions.h5", "w") as fh:
        fh.create_dataset("positions", data=rng.integers(0, CHAIN, size=(20, 5, 2)))
        fh.attrs["N"] = CHAIN
        fh.attrs["chain_length"] = CHAIN
        fh.attrs["num_chains"] = 1
        fh.attrs["rnapii_enabled"] = False
        fh.attrs["lesion_enabled"] = False
    np.save(dirp / "contact_map.npy", _synth_map(seed))


def _cfg_for(dirp: Path, resolutions) -> PipelineConfig:
    return PipelineConfig(
        lef=LEFConfig(
            chain_length=CHAIN, num_chains=1,
            topology_kwargs={
                "tad_positions": BOUNDARIES,
                "genes": [{"tss": 40, "tes": 55, "enhancers": [20]}],
            },
            output_path=str(dirp / "LEFPositions.h5"),
        ),
        polymer=PolymerConfig(md_steps_per_block=750),
        contacts=ContactsConfig(
            trajectory_folder=str(dirp),
            raw_output_path=str(dirp / "contact_map.npy"),
            oe_output_path=str(dirp / "contact_map_oe.npy"),
            map_size=N, map_starts=[0],
            analysis_resolutions=resolutions,
        ),
    )


def test_qc_produces_native_and_coarsened_3d(tmp_path):
    run_dir = tmp_path / "run"
    _make_run(run_dir, seed=0)
    out = Path(qc.run(_cfg_for(run_dir, [1, 10])))

    metrics = json.loads((out / "metrics.json").read_text())
    assert "3d" in metrics and "3d_res10" in metrics
    assert metrics["3d"]["shape"] == [N, N]
    assert metrics["3d_res10"]["shape"] == [N // 10, N // 10]
    assert metrics["3d_res10"]["resolution"] == 10

    # Insulation windows are physical (monomers): native keeps 5..120; the
    # coarsened map must never report a window finer than its 10-monomer bin.
    assert set(metrics["3d"]["insulation_aggregate"]) == {"5", "10", "20", "40", "80", "120"}
    res10_windows = {int(w) for w in metrics["3d_res10"]["insulation_aggregate"]}
    assert res10_windows == {20, 40, 80, 120}
    assert min(res10_windows) >= 10

    # native plot names unchanged; coarsened plots carry the _res10 suffix
    assert (out / "plots" / "Ps_3d.png").exists()
    for fn in ("Ps_3d_res10.png", "contact_map_res10.png",
               "insulation_windows_res10.png", "pileups_tad_rescaled_res10.png"):
        assert (out / "plots" / fn).exists(), fn

    # Insulation is cooltools-style (the only insulation now): boundary strength
    # is a log2 prominence dict (with a "mean"), aggregate carries dip_depth (not
    # the legacy dip_ratio). Large windows can exceed this small test map (->
    # all-NaN score, NaN prominence), so require at least one finite positive dip.
    for key in ("3d", "3d_res10"):
        bs = metrics[key]["insulation_boundary_strength"]
        assert bs, key
        means = [v["mean"] for v in bs.values()]
        assert any(np.isfinite(mn) and mn > 0 for mn in means), (key, means)
        agg = metrics[key]["insulation_aggregate"]
        # cooltools log2 aggregate uses dip_depth, never the legacy dip_ratio;
        # windows too large for the map yield an empty aggregate (no dip_depth).
        assert all("dip_ratio" not in v for v in agg.values()), key
        assert any("dip_depth" in v for v in agg.values()), key

    assert "3D @ 10-monomer resolution" in (out / "report.md").read_text()
    assert "cooltools" in (out / "report.md").read_text()


def test_qc_resolution_1_only_skips_coarsening(tmp_path):
    run_dir = tmp_path / "run"
    _make_run(run_dir, seed=1)
    out = Path(qc.run(_cfg_for(run_dir, [1])))

    metrics = json.loads((out / "metrics.json").read_text())
    assert "3d" in metrics
    assert not any(k.startswith("3d_res") for k in metrics)
    assert not list((out / "plots").glob("*_res[0-9]*.png"))


def test_compare_produces_coarsened_section(tmp_path):
    a = tmp_path / "A"
    b = tmp_path / "B"
    _make_run(a, seed=2)
    _make_run(b, seed=3)
    out_dir = tmp_path / "cmp"

    compare.run(_cfg_for(a, [1, 10]), _cfg_for(b, [1, 10]), out_dir, "A", "B")

    cj = json.loads((out_dir / "compare.json").read_text())
    assert "3d" in cj["A"] and "3d_res10" in cj["A"]
    assert "3d_res10" in cj["B"]
    assert "folds" in cj and "folds_res10" in cj
    assert "tad_pileup_compare_res10" in cj
    assert cj["A"]["3d_res10"]["shape"] == [N // 10, N // 10]

    # Insulation is cooltools-style: boundary strength is a log2 prominence dict
    # (with a "mean"), aggregate carries dip_depth, and folds compare via deltas.
    for key in ("3d", "3d_res10"):
        bs = cj["A"][key]["insulation_boundary_strength"]
        assert bs and all("mean" in v for v in bs.values()), key
        agg = cj["A"][key]["insulation_aggregate"]
        assert agg and any("dip_depth" in v for v in agg.values()), key
        # no legacy raw-insulation keys should leak through
        assert all("dip_ratio" not in v for v in agg.values()), key
    assert "insulation_boundary_strength_delta" in cj["folds"]
    assert "insulation_aggregate_dip_delta" in cj["folds"]
    assert "insulation_boundary_strength_delta" in cj["folds_res10"]

    assert (out_dir / "plots" / "contact_map_compare.png").exists()  # native
    assert (out_dir / "plots" / "insulation_compare.png").exists()
    for fn in ("res10_contact_map_compare.png", "res10_Ps_3d_compare.png",
               "res10_tad_pileup_compare.png", "res10_insulation_compare.png"):
        assert (out_dir / "plots" / fn).exists(), fn

    assert "3D @ 10-monomer resolution" in (out_dir / "compare.md").read_text()
    assert "cooltools" in (out_dir / "compare.md").read_text()
