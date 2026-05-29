from types import SimpleNamespace

import h5py
import numpy as np

from polychrom.pipelines.loop_extrusion.compare import _anchor_groups, _load, _write_report, run_many


def test_compare_load_uses_config_genes_when_h5_lacks_genes(tmp_path):
    lef_path = tmp_path / "LEFPositions.h5"
    with h5py.File(lef_path, "w") as fh:
        fh.create_dataset("positions", data=np.array([[[10, 30], [45, 55]]]))
        fh.attrs["N"] = 100
        fh.attrs["chain_length"] = 100
        fh.attrs["num_chains"] = 1
        fh.attrs["rnapii_enabled"] = False
        fh.attrs["lesion_enabled"] = False

    cfg = SimpleNamespace(
        lef=SimpleNamespace(
            output_path=str(lef_path),
            topology_kwargs={
                "tad_positions": [50],
                "genes": [
                    {"tss": 30, "tes": 10},
                    {"tss": 60, "tes": 80},
                ],
            },
        ),
        contacts=SimpleNamespace(
            raw_output_path=str(tmp_path / "missing_contact_map.npy"),
            oe_output_path=str(tmp_path / "missing_contact_map_oe.npy"),
        ),
    )

    run = _load(cfg, "test")

    assert run.gene_tss == [30, 60]
    assert run.gene_tes == [10, 80]
    assert run.gene_bodies == [(10, 30), (60, 80)]
    assert _anchor_groups(run)["gene TSS"] == [30, 60]
    assert _anchor_groups(run)["gene TES"] == [10, 80]


def test_compare_report_writes_stripe_metrics(tmp_path):
    metrics = {
        "A": {
            "1d": {
                "loop_length": {"mean": 10.0},
                "classification": {"corner_pct": 20.0, "stripe_pct": 30.0},
                "boundary_crossing": {"mean": 0.2},
                "boundary_crossing_stripes": {
                    "mean": {"stripe_share_of_crossing_frames": 0.25}
                },
            },
            "3d": {
                "corner_dot_intensities": [1.0],
                "tad_pileup_oe_crossing_stripes": {"mean_contrast": 0.1},
                "fountain_oe": {"fountain_score": 1.2, "symmetry": 0.8, "enhancer_tss_mean": 1.5},
                "stripe_enrichment_per_boundary": {
                    "50": {"left_enrichment_x": 0.5, "right_enrichment_x": 1.0}
                },
            },
        },
        "B": {
            "1d": {
                "loop_length": {"mean": 20.0},
                "classification": {"corner_pct": 40.0, "stripe_pct": 45.0},
                "boundary_crossing": {"mean": 0.4},
                "boundary_crossing_stripes": {
                    "mean": {"stripe_share_of_crossing_frames": 0.5}
                },
            },
            "3d": {
                "corner_dot_intensities": [2.0],
                "tad_pileup_oe_crossing_stripes": {"mean_contrast": 0.3},
                "fountain_oe": {"fountain_score": 1.5, "symmetry": 0.4, "enhancer_tss_mean": 3.0},
                "stripe_enrichment_per_boundary": {
                    "50": {"left_enrichment_x": 1.0, "right_enrichment_x": 1.5}
                },
            },
        },
        "folds": {
            "loop_mean": 2.0,
            "corner_pct": 2.0,
            "stripe_pct": 1.5,
            "boundary_crossing_mean": 2.0,
            "boundary_crossing_stripe_share": 2.0,
            "corner_dot_intensities_per_tad": [2.0],
            "tad_pileup_oe_crossing_stripe_contrast_delta": 0.2,
            "fountain_oe_score": 1.25,
            "fountain_oe_symmetry": 0.5,
            "fountain_oe_enhancer_tss_mean": 2.0,
            "stripe_enrichment_per_boundary": {
                "50": {"left_enrichment_x": 2.0, "right_enrichment_x": 1.5}
            },
        },
    }
    report = tmp_path / "compare.md"

    _write_report(metrics, report, "A", "B")

    text = report.read_text()
    assert "| global anchored-stripe% | 30.0 | 45.0 | 1.50 |" in text
    assert "| boundary-crossing stripe share | 25.0% | 50.0% | 2.00 |" in text
    assert "### 3D Rescaled-TAD Crossing Stripes" in text
    assert "| O/E log2 | 0.100 | 0.300 | 0.200 |" in text
    assert "### Gene Fountain Stats" in text
    assert "| O/E | score | 1.200 | 1.500 | 1.25 |" in text
    assert "| O/E | enhancer-TSS | 1.500 | 3.000 | 2.00 |" in text
    assert "### 3D Contact-Map Stripe Enrichment Per Boundary" in text
    assert "| 50 | 0.50 | 1.00 | 2.00 | 1.00 | 1.50 | 1.50 |" in text


def test_run_many_writes_baseline_summary(tmp_path):
    def cfg_for(name, positions):
        run_dir = tmp_path / name
        run_dir.mkdir()
        lef_path = run_dir / "LEFPositions.h5"
        with h5py.File(lef_path, "w") as fh:
            fh.create_dataset("positions", data=np.asarray(positions, dtype=int))
            fh.attrs["N"] = 800
            fh.attrs["chain_length"] = 800
            fh.attrs["num_chains"] = 1
            fh.attrs["rnapii_enabled"] = False
            fh.attrs["lesion_enabled"] = False
        return SimpleNamespace(
            lef=SimpleNamespace(
                output_path=str(lef_path),
                topology_kwargs={
                    "tad_positions": [50],
                    "genes": [{"tss": 20, "tes": 30}],
                },
            ),
            contacts=SimpleNamespace(
                raw_output_path=str(run_dir / "missing_contact_map.npy"),
                oe_output_path=str(run_dir / "missing_contact_map_oe.npy"),
            ),
        )

    baseline = cfg_for("baseline", [[[10, 20], [40, 60]], [[15, 25], [45, 55]]])
    config2 = cfg_for("config2", [[[10, 30], [35, 65]], [[15, 35], [40, 70]]])
    config3 = cfg_for("config3", [[[5, 45], [55, 80]], [[10, 40], [60, 90]]])

    out = run_many(baseline, [config2, config3], tmp_path / "compare_many", "config1", ["config2", "config3"])

    report = (out / "compare_many.md").read_text()
    assert "baseline config1" in report
    assert "[config2](compare_config2.md)" in report
    assert "[config3](compare_config3.md)" in report
    assert (out / "compare_many.json").exists()
    assert (out / "compare_config2.md").exists()
    assert (out / "compare_config3.md").exists()
    assert (out / "plots" / "config2_loop_length_compare.png").exists()
    assert (out / "plots" / "config3_loop_length_compare.png").exists()
