#!/usr/bin/env python
"""Generate 1D sims for two configs and compare cohesin loop lengths.

Uses the SAME metric as the compare pipeline: qc.loop_length_stats over the 1D
LEF positions, histogram edges [0,50,100,150,200,300,500,N]. Reports per-config
mean/median/p10/p90 + the bin fractions, and the config2/config1 fold (compare's
_fold = b/a convention).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from polychrom.pipelines.loop_extrusion import lef as lef_stage
from polychrom.pipelines.loop_extrusion.config import load_config
from polychrom.pipelines.loop_extrusion.qc import loop_length_stats

CFG_A = "configs_test/config1_10k.yaml"   # txn ON  (WT)
CFG_B = "configs_test/config2_10k.yaml"   # txn OFF (depletion)
WARMUP, TRAJ = 5000, 20000                # plenty for a steady-state loop dist


def run_1d(cfg_path, out_h5):
    cfg = load_config(cfg_path)
    lc = cfg.lef
    lc.output_path = str(out_h5)
    lc.warmup_steps = min(int(lc.warmup_steps), WARMUP)
    lc.trajectory_length = min(int(lc.trajectory_length), TRAJ)
    lef_stage.run(lc)
    N = lc.chain_length
    with h5py.File(out_h5, "r") as f:
        pos = f["positions"][:]
    edges = [e for e in [0, 50, 100, 150, 200, 300, 500] if e < N] + [N]
    return loop_length_stats(pos, edges), N, pos.shape


def main():
    out = ROOT / "runs/looplen_cmp"; out.mkdir(parents=True, exist_ok=True)
    sa, N, shA = run_1d(CFG_A, out / "config1_10k.h5")
    sb, _, shB = run_1d(CFG_B, out / "config2_10k.h5")

    print(f"\n# 1D loop-length comparison  (warmup={WARMUP}, traj={TRAJ}, N={N})")
    print(f"  config1 (txn ON)  positions {shA[0]}x{shA[1]} LEFs")
    print(f"  config2 (txn OFF) positions {shB[0]}x{shB[1]} LEFs\n")
    def row(name, va, vb, fold=True):
        f = (vb / va) if (fold and va) else float("nan")
        print(f"  {name:14s} {va:9.1f} {vb:9.1f}   {f:+.2f}x" if fold
              else f"  {name:14s} {va:9.1f} {vb:9.1f}")
    print(f"  {'metric (kb)':14s} {'config1':>9s} {'config2':>9s}   fold(c2/c1)")
    print("  " + "-" * 48)
    for k in ("mean", "median", "p10", "p90"):
        row(k, sa[k], sb[k])

    print(f"\n  loop-length histogram (fraction of LEF-frames per bin):")
    edges = sa["histogram_edges_kb"]
    bins = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
    print(f"  {'bin kb':14s} {'config1':>9s} {'config2':>9s}")
    for b, fa, fb in zip(bins, sa["histogram_fraction"], sb["histogram_fraction"]):
        print(f"  {b:14s} {fa:9.3f} {fb:9.3f}")

    # compare-style derived fractions
    def frac(stats, lo, hi):
        e = stats["histogram_edges_kb"]; fr = stats["histogram_fraction"]
        return sum(f for f, a, b in zip(fr, e[:-1], e[1:]) if a >= lo and b <= hi)
    print()
    sh_a, sh_b = frac(sa, 0, 50), frac(sb, 0, 50)
    tad_a, tad_b = frac(sa, 150, 300), frac(sb, 150, 300)
    print(f"  frac_short (<50kb)   c1={sh_a:.3f}  c2={sh_b:.3f}  fold={sh_b/sh_a:+.2f}x")
    print(f"  frac_tadloop(150-300) c1={tad_a:.3f}  c2={tad_b:.3f}  fold={tad_b/tad_a:+.2f}x")
    print(f"  mean loop fold (c2/c1) = {sb['mean']/sa['mean']:+.2f}x")


if __name__ == "__main__":
    main()
