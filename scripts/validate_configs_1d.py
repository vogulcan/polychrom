#!/usr/bin/env python
"""Validate the edited config1 (WT) vs config2 (depletion) at the 1D stage.

Runs the real YAML files (no parameter overrides) through lef.run and reports the
DEPL/WT folds for the Micro-C phenotype, then the biology scripts on config1 WT.

Usage:  python scripts/validate_configs_1d.py [traj_length]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (polychrom pkg)
sys.path.insert(0, str(Path(__file__).resolve().parent))         # scripts/ (driver helpers)
from polychrom.pipelines.loop_extrusion import lef as lef_stage
from polychrom.pipelines.loop_extrusion.config import load_config

# reuse the metric helpers from the tuning driver
from tune_depletion_1d import _boundaries, metrics_1d  # type: ignore


def run(cfg_path: str, out: Path, traj: int):
    cfg = load_config(cfg_path)
    cfg.lef.output_path = str(out)
    cfg.lef.trajectory_length = traj
    cfg.lef.warmup_steps = min(int(cfg.lef.warmup_steps), max(2000, traj // 5))
    lef_stage.run(cfg.lef)
    return metrics_1d(out, _boundaries(cfg))


def main(argv):
    traj = int(argv[1]) if len(argv) > 1 else 20000
    d = Path("runs/val_cfg"); d.mkdir(parents=True, exist_ok=True)
    wt = run("configs/config1.yaml", d / "c1_WT.h5", traj)
    dep = run("configs/config2.yaml", d / "c2_DEPL.h5", traj)
    print(f"\n# edited configs, traj={traj}")
    print(f"  {'metric':<20}{'config1':>10}{'config2':>10}{'c2/c1':>9}  want")
    spec = [("mean_loop", "UP"), ("frac_short", "DOWN"), ("frac_tad", "UP"),
            ("boundary_occ_pct", "UP"), ("corner_both_pct", "UP"),
            ("cross_boundary_pct", "UP")]
    for k, want in spec:
        f = dep[k] / wt[k] if wt[k] else float("nan")
        print(f"  {k:<20}{wt[k]:>10.3f}{dep[k]:>10.3f}{f:>9.3f}  {want}")
    print(f"\n# config1 WT h5 = {d/'c1_WT.h5'} (run the two biology scripts on it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
