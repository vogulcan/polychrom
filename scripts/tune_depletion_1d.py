#!/usr/bin/env python
"""Fast 1D-only sweep to tune the RNAPII-depletion contrast.

Runs the ``lef`` stage twice per parameter set -- WT (RNAPII on) and
DEPLETION (RNAPII plugins nulled, boundary slightly weakened) -- on a SHORT
trajectory, then reports the 1D observables that drive the Micro-C phenotype:

  * mean loop length, short-loop fraction (<50 kb), TAD-loop fraction (150-300 kb)
  * cohesin occupancy AT boundaries (corner-dot / insulation proxy)
  * cohesin stalled fraction (moving-barrier eviction proxy)

DEPLETION-vs-WT folds for these are the 1D shadow of the 3D map changes, so we
can iterate here in seconds instead of running CUDA polymer + contact maps.

Biological sanity (uses the project scripts) is checked separately on the WT
run: scripts/cohesin_rnapii_coupling.py (interference / directionality) and
scripts/nascent_rna_abundance.py (Pol II output per gene).

Usage:  python scripts/tune_depletion_1d.py [traj_length]
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polychrom.pipelines.loop_extrusion import lef as lef_stage
from polychrom.pipelines.loop_extrusion.config import load_config

BASE = "runs/config1/config1.yaml"
KB_PER_SITE = 1  # 1 site = 1 kb


# --------------------------------------------------------------------------- #
# parameter sets to compare
# --------------------------------------------------------------------------- #
# Each entry overrides cfg.lef.* and cfg.lef.topology_kwargs[*]. ``dep_boundary``
# is the per-anchor capture strength used for the DEPLETION run only (models the
# measured CTCF-occupancy drop). WT always uses the base 0.6.
PARAM_SETS = {
    # Baseline: generic stalled-lifetime = lifetime, no Pol II-specific eviction.
    "A_current": dict(
        lifetime_rnapii_stalled=82,
        rnapii_push_prob=0.50, rnapii_headon_push_prob=0.05,
        rnapii_stall_prob=0.20, rnapii_elongating_block_prob=0.80,
        dep_boundary=0.60,
    ),
    # Pol II-specific eviction ON (lifetime_stalled stays 82 = Hansen baseline).
    # Boundary held EQUAL in depletion: isolate the RNAPII-eviction relief.
    "F_evict8_bnd60": dict(
        lifetime_rnapii_stalled=8,
        rnapii_push_prob=0.05, rnapii_headon_push_prob=0.01,
        rnapii_stall_prob=0.45, rnapii_elongating_block_prob=0.98,
        dep_boundary=0.60,
    ),
    "H_evict15_bnd60": dict(
        lifetime_rnapii_stalled=15,
        rnapii_push_prob=0.10, rnapii_headon_push_prob=0.02,
        rnapii_stall_prob=0.35, rnapii_elongating_block_prob=0.95,
        dep_boundary=0.60,
    ),
    # Barely-weaker boundary (0.55): keep corners up but leak a few stripes.
    "G_evict8_bnd55": dict(
        lifetime_rnapii_stalled=8,
        rnapii_push_prob=0.05, rnapii_headon_push_prob=0.01,
        rnapii_stall_prob=0.45, rnapii_elongating_block_prob=0.98,
        dep_boundary=0.55,
    ),
    "I_evict15_bnd55": dict(
        lifetime_rnapii_stalled=15,
        rnapii_push_prob=0.10, rnapii_headon_push_prob=0.02,
        rnapii_stall_prob=0.35, rnapii_elongating_block_prob=0.95,
        dep_boundary=0.55,
    ),
}


def _boundaries(cfg) -> np.ndarray:
    """Absolute boundary sites = {0, tad_positions, chain_length} per chain."""
    tk = cfg.lef.topology_kwargs
    inner = list(tk.get("tad_positions", []))
    cl = int(cfg.lef.chain_length)
    rel = sorted({0, *inner, cl})
    sites = []
    for ci in range(int(cfg.lef.num_chains)):
        off = ci * cl
        sites += [off + r for r in rel if r < cl]  # cl of chain i == 0 of i+1
    sites.append((int(cfg.lef.num_chains) - 1) * cl + cl - 1)
    return np.array(sorted(set(sites)), dtype=np.int64)


def _scale_boundary(tk: dict, target: float) -> None:
    """Set every per-anchor boundary capture (and default) to ``target``."""
    bs = tk.get("boundary_strength")
    if isinstance(bs, dict):
        for k in list(bs):
            bs[k] = target
    tk["boundary_strength"] = bs if isinstance(bs, dict) else target
    tk["default_boundary_strength"] = target


def _disable_rnapii(cfg) -> None:
    """Null the RNAPII plugin slots (LEFPlugins dataclass) == transcription off.

    lef.run() enables RNAPII only when BOTH slots are non-None, so setting them
    to None is exactly the YAML `rnapii_load: null / rnapii_translocate: null`.
    """
    pl = cfg.lef.plugins
    pl.rnapii_load = None
    pl.rnapii_translocate = None


def _apply(cfg, ov: dict, *, deplete: bool, traj: int, out: Path):
    lc = cfg.lef
    lc.output_path = str(out)
    lc.trajectory_length = traj
    lc.warmup_steps = min(int(lc.warmup_steps), max(2000, traj // 5))
    tk = lc.topology_kwargs
    tk["lifetime_rnapii_stalled"] = int(ov["lifetime_rnapii_stalled"])
    for k in ("rnapii_push_prob", "rnapii_headon_push_prob",
              "rnapii_stall_prob", "rnapii_elongating_block_prob"):
        tk[k] = ov[k]
    if deplete:
        _disable_rnapii(cfg)
        _scale_boundary(tk, float(ov["dep_boundary"]))
    return cfg


def metrics_1d(h5: Path, boundaries: np.ndarray) -> dict:
    with h5py.File(h5, "r") as f:
        pos = f["positions"][:]            # (T, L, 2)
    left = pos[..., 0].astype(np.int64)
    right = pos[..., 1].astype(np.int64)
    loops = np.abs(right - left) * KB_PER_SITE
    loops = loops[loops > 0]
    legs = np.concatenate([left.ravel(), right.ravel()])
    bset = set(int(b) for b in boundaries)
    bnear = bset | {b + 1 for b in bset} | {b - 1 for b in bset}
    near = np.isin(legs, list(bnear))
    # True corner-dot proxy: BOTH legs of a cohesin anchored at (distinct)
    # boundaries -> a loop that spans a TAD and lands a corner dot.
    lnear = np.isin(left, list(bnear))
    rnear = np.isin(right, list(bnear))
    both = float((lnear & rnear & (np.abs(right - left) > 10)).mean())
    # Boundary-crossing-stripe proxy: a cohesin loop whose interval brackets an
    # INNER boundary (left < b < right) = a loop reaching across a TAD edge.
    lo = np.minimum(left, right); hi = np.maximum(left, right)
    bmax = int(boundaries.max())
    inner = [int(b) for b in boundaries if 0 < int(b) < bmax]
    cross = np.zeros_like(lo, dtype=bool)
    for b in inner:
        cross |= (lo < b) & (hi > b)
    return dict(
        mean_loop=float(loops.mean()),
        frac_short=float((loops < 50).mean()),
        frac_tad=float(((loops >= 150) & (loops < 400)).mean()),
        boundary_occ_pct=float(near.mean()) * 100.0,
        corner_both_pct=both * 100.0,
        cross_boundary_pct=float(cross.mean()) * 100.0,
    )


def run_one(tag: str, ov: dict, traj: int, tmp: Path) -> dict:
    out = {}
    for deplete, label in ((False, "WT"), (True, "DEPL")):
        cfg = load_config(BASE)
        h5 = tmp / f"{tag}_{label}.h5"
        if not h5.exists():                       # resumable: skip done runs
            _apply(cfg, ov, deplete=deplete, traj=traj, out=h5)
            lef_stage.run(cfg.lef)
        bnd = _boundaries(cfg)
        out[label] = metrics_1d(h5, bnd)
        out[f"{label}_h5"] = str(h5)
    return out


def fold(d, w, k):
    return d[k] / w[k] if w[k] else float("nan")


def main(argv):
    traj = int(argv[1]) if len(argv) > 1 else 20000
    tmp = Path(argv[2]) if len(argv) > 2 else Path("runs/tune1d")  # resumable dir
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"# out={tmp}  traj={traj}\n")
    results = {}
    for tag, ov in PARAM_SETS.items():
        print(f"==== {tag} ====  {ov}")
        r = run_one(tag, ov, traj, tmp)
        results[tag] = r
        w, d = r["WT"], r["DEPL"]
        print(f"  {'metric':<20}{'WT':>10}{'DEPL':>10}{'DEPL/WT':>10}  want")
        spec = [
            ("mean_loop", "UP"),
            ("frac_short", "DOWN"),
            ("frac_tad", "UP"),
            ("boundary_occ_pct", "UP (insulation)"),
            ("corner_both_pct", "UP (corner dot)"),
            ("cross_boundary_pct", "UP (cross stripe)"),
        ]
        for k, want in spec:
            print(f"  {k:<20}{w[k]:>10.3f}{d[k]:>10.3f}{fold(d, w, k):>10.3f}  {want}")
        print(f"  WT h5:   {r['WT_h5']}")
        print(f"  DEPL h5: {r['DEPL_h5']}\n")
    (tmp / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"# wrote {tmp/'summary.json'}")
    print("# Use scripts/cohesin_rnapii_coupling.py and "
          "scripts/nascent_rna_abundance.py on the chosen WT h5 to confirm "
          "interference + Pol II output stay biological.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
