#!/usr/bin/env python
"""1D test for activity-weighted cohesin loading.

Runs config1_real through the loop-extrusion (1D) stage and checks that cohesin
loading/occupancy now correlates POSITIVELY with per-TAD transcription, vs the
legacy unweighted path (which loaded preferentially in silent domains).

Usage: python scripts/test_activity_loading_1d.py <dir-with-config1_real.yaml>
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
from polychrom.pipelines.loop_extrusion.plugins.topology import (
    gene_aware_convergent_tad_topology as topo,
)


def per_tad(values_per_site, intervals):
    return np.array([values_per_site[lo:hi].mean() for lo, hi in intervals])


def main(argv):
    d = Path(argv[1] if len(argv) > 1 else "/tmp/gentest")
    cfg = load_config(str(d / "config1_real.yaml"))
    lc = cfg.lef
    N = lc.chain_length
    tk = dict(lc.topology_kwargs)
    bounds = [int(b) for b in tk["tad_positions"]]
    intervals = list(zip([0, *bounds], [*bounds, N]))
    txn = np.load(d / "monomer_ratios_real.npy")
    txn_tad = per_tad(txn, intervals)

    def loadmass_per_tad(weight, tss):
        """Per-TAD targeted-loading probability mass, straight from the plugin."""
        a = topo(lc, **{**tk, "weight_loading_by_activity": weight, "target_tss": tss})
        sites = np.array(a["loading_sites"]) % N           # collapse replicate chains
        probs = a["loading_probs"]
        probs = np.ones(len(sites)) / len(sites) if probs is None else np.array(probs)
        mass = np.zeros(len(intervals))
        idx = np.clip(np.searchsorted(bounds, sites, side="right"), 0, len(intervals) - 1)
        for i, p in zip(idx, probs):
            mass[i] += p
        return mass

    print("== A. targeted-loading distribution (through the real plugin) ==")
    for label, w, tss in [("legacy (unweighted, enh-only)", False, False),
                           ("FIX (activity-weighted, enh+TSS)", True, True)]:
        mass = loadmass_per_tad(w, tss)
        m = np.median(txn_tad)
        print(f"  {label:36s} corr(txn, load-mass)={np.corrcoef(txn_tad, mass)[0,1]:+.2f}"
              f"  active/silent={mass[txn_tad>=m].sum():.3f}/{mass[txn_tad<m].sum():.3f}")

    print("\n== B. realized cohesin occupancy from a 1D run ==")
    # Amplify targeted loading so the placement bias is visible in occupancy.
    runs = Path("runs/test_actload"); runs.mkdir(parents=True, exist_ok=True)
    for label, w, tss in [("legacy", False, False), ("FIX", True, True)]:
        c = load_config(str(d / "config1_real.yaml")).lef
        c.topology_kwargs = {**dict(c.topology_kwargs),
                             "weight_loading_by_activity": w, "target_tss": tss,
                             "targeted_load_prob": 0.8}
        c.output_path = str(runs / f"{label}.h5")
        c.trajectory_length = 6000
        c.warmup_steps = 3000
        lef_stage.run(c)
        with h5py.File(c.output_path, "r") as f:
            pos = f["positions"][:]                     # (T, L, 2)
        sites = (pos.reshape(-1) % N).astype(int)
        occ = np.bincount(sites, minlength=N).astype(float)
        occ_tad = per_tad(occ, intervals)
        m = np.median(txn_tad)
        print(f"  {label:6s} corr(txn, occupancy)={np.corrcoef(txn_tad, occ_tad)[0,1]:+.2f}"
              f"  mean occ active/silent={occ_tad[txn_tad>=m].mean():.1f}/{occ_tad[txn_tad<m].mean():.1f}")


if __name__ == "__main__":
    main(sys.argv)
