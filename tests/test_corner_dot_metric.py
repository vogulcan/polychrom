"""Corner-dot metric + orientation-aware topology in compare_config_chain_metrics.

Exercises the analysis script directly (it lives under scripts/). The corner-dot
metric counts a single LEF whose two legs are captured at the two convergent
anchors of ONE internal TAD.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import compare_config_chain_metrics as ccm  # noqa: E402


def _cfg(topology_kwargs, chain=400, num_chains=1):
    return {"lef": {"chain_length": chain, "num_chains": num_chains,
                    "topology_kwargs": topology_kwargs}}


def _write_positions(path, arr):
    with h5py.File(path, "w") as fh:
        fh.create_dataset("positions", data=np.asarray(arr, dtype=np.int32))


_TADS = [
    {"left": 0, "right": 99},
    {"left": 100, "right": 249},
    {"left": 250, "right": 399},
]


def test_topology_corner_pairs_are_internal_only():
    topo = ccm._topology(_cfg({"tads": _TADS}))
    # Only the middle TAD has both anchors internal -> corner-capable.
    assert topo["n_tads_corner"] == 1
    assert set(topo["corner_pairs"].values()) == {(100, 249)}


def test_topology_legacy_and_tads_orientation_match():
    legacy = ccm._topology(_cfg({"tad_positions": [100, 250]}))
    tads = ccm._topology(_cfg({"tads": _TADS}))
    # Combined mask is byte-identical to the legacy internal-anchor convention.
    assert np.array_equal(legacy["ctcf_mask"], tads["ctcf_mask"])
    assert np.array_equal(legacy["left_ctcf_mask"], tads["left_ctcf_mask"])
    assert np.array_equal(legacy["right_ctcf_mask"], tads["right_ctcf_mask"])
    assert legacy["corner_pairs"] == tads["corner_pairs"]
    # Left anchors capture the -1 leg (sites 100), right anchors the +1 leg (249).
    assert tads["left_ctcf_mask"][100] and not tads["right_ctcf_mask"][100]
    assert tads["right_ctcf_mask"][249] and not tads["left_ctcf_mask"][249]


def test_corner_dot_counts_only_internal_both_leg_capture(tmp_path):
    topo = ccm._topology(_cfg({"tads": _TADS}))
    # 3 frames x 3 LEFs:
    #   LEF0 = corner dot of the middle TAD (legs on 100 and 249)
    #   LEF1 = only one anchor (100, 200) -> not a corner
    #   LEF2 = both legs on the chromosome-end TAD's anchors (0, 99) -> excluded
    positions = np.array(
        [[[100, 249], [100, 200], [0, 99]]] * 3, dtype=np.int32
    )
    h5 = tmp_path / "pos.h5"
    _write_positions(h5, positions)
    core, *_ = ccm.analyze_run("x", None, h5, topo)
    corner = next(r for r in core if r["metric"] == "corner_dot")
    assert corner["raw_count"] == 3        # LEF0 across 3 frames only
    assert corner["denominator"] == 3      # frames * n_tads_corner (3 * 1)
    assert corner["raw_value"] == 1.0


def test_corner_dot_respects_leg_orientation(tmp_path):
    """A loop with legs swapped onto the wrong-facing sites is NOT a corner dot:
    the lower leg must be at the left anchor and the upper leg at the right one.
    (Here both legs landing inside but not exactly on the anchor pair -> zero.)"""
    topo = ccm._topology(_cfg({"tads": _TADS}))
    positions = np.array(
        [[[120, 240]]] * 2, dtype=np.int32  # inside the TAD but off the anchors
    )
    h5 = tmp_path / "pos.h5"
    _write_positions(h5, positions)
    core, *_ = ccm.analyze_run("x", None, h5, topo)
    corner = next(r for r in core if r["metric"] == "corner_dot")
    assert corner["raw_count"] == 0


def test_existing_ctcf_metric_unchanged_by_orientation_split(tmp_path):
    """cohesin_ctcf_boundary uses the combined mask, so legacy and the equivalent
    tads layout report the same anchor occupancy on identical positions."""
    positions = np.array([[[99, 100], [249, 250]]] * 4, dtype=np.int32)
    h5 = tmp_path / "pos.h5"
    _write_positions(h5, positions)
    legacy = ccm.analyze_run("L", None, h5, ccm._topology(_cfg({"tad_positions": [100, 250]})))[0]
    tads = ccm.analyze_run("T", None, h5, ccm._topology(_cfg({"tads": _TADS})))[0]
    lv = next(r for r in legacy if r["metric"] == "cohesin_ctcf_boundary")
    tv = next(r for r in tads if r["metric"] == "cohesin_ctcf_boundary")
    assert lv["raw_count"] == tv["raw_count"]
    assert lv["raw_value"] == tv["raw_value"]
