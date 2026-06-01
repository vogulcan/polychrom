#!/usr/bin/env python
"""Short-trajectory CUDA grid over POLYMER confinement params, to recover the
transcription-OFF/ON P(s) crossover in 3D.

Diagnosis (see cis-decay analysis): the 1D bridge P(s) already shows the
experimental crossover -- OFF/ON ~0.7-0.9x at short-loop scale, ~1.3x at
TAD scale -- but the 3D contact map dilutes it to ~0.99 / ~1.06, because
nonspecific bulk-globule contacts (set by confinement/density, identical in
both conditions) dominate. Lowering the confined volume fraction should cut
that background and amplify the 3D fold-change toward the experimental
0.9 / 1.15.

LOOP-EXTRUSION BIOLOGY IS NOT TOUCHED. The 1D LEF trajectory is run ONCE per
condition (ON = --c1-base, OFF = --c2-base) and reused for every polymer
combo, exactly as grid_ps_ep.py does for force params.

Per (combo, seed) we run MD for BOTH conditions and measure P(s) via
contact_scaling, then report the OFF/ON fold-change in two bands:
  * short-loop band  [--short-lo, --short-hi]  kb   (target ~0.90)
  * TAD band         [--tad-lo,   --tad-hi]    kb   (target ~1.15)
Score = crossover separation (fold_tad - fold_short); bigger = closer to data.

Example (coarse, 1 seed):
    micromamba run -n openmm python scripts/grid_confinement.py \
        --c1-base configs/config1.yaml --c2-base configs/config2.yaml \
        --out runs/grid_conf --density 0.05 0.1 0.2 --confk 1.0 5.0 \
        --seeds 1 --relax 40000 --burnin 2000 --traj 800 --save 5 --mdsteps 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polychrom.pipelines.loop_extrusion.config import load_config
from polychrom.pipelines.loop_extrusion import lef as lef_stage
from polychrom.pipelines.loop_extrusion import polymer as polymer_stage
from polychrom.hdf5_format import list_URIs, load_URI_pos
from polychrom.polymer_analyses import contact_scaling, generate_bins

MONOMER_KB = 1.0   # 1 monomer = 1 kb (separation 200 = 200 kb)


def ps_curve(folder, chain_length, num_chains, cutoff):
    """Per-chain-averaged P(s) over saved conformations -> (mids_kb, prob).

    contact_scaling counts pairs by index separation, so it must be run on a
    single chain at a time; otherwise separations spanning a chain boundary
    pollute the curve. We slice each frame into its chains and average.
    """
    uris = list_URIs(folder)
    bins = generate_bins(chain_length, start=4, bins_per_order_magn=10)
    cps, mids = [], None
    for u in uris:
        pos = load_URI_pos(u).reshape(num_chains, chain_length, 3)
        for c in range(num_chains):
            mids, cp = contact_scaling(pos[c], bins0=bins, cutoff=cutoff)
            cps.append(cp)
    cp = np.nanmean(cps, axis=0)
    mids = np.asarray(mids, float) * MONOMER_KB
    return mids, np.asarray(cp, float), len(uris)


def band_mean(mids, cp, lo, hi):
    m = (mids >= lo) & (mids < hi) & np.isfinite(cp)
    return float(cp[m].mean()) if m.any() else float("nan")


def run_condition(cfg, lef_path, combo_dir, density, confk):
    fb = cfg.polymer.plugins.force_builder.kwargs
    fb["confinement_density"] = float(density)
    fb["confinement_k"] = float(confk)
    cfg.polymer.lef_positions_path = str(lef_path)
    cfg.polymer.output_folder = str(combo_dir)
    polymer_stage.run(cfg.polymer)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-base", default="configs/config1.yaml", help="txn ON")
    ap.add_argument("--c2-base", default="configs/config2.yaml", help="txn OFF")
    ap.add_argument("--out", default="runs/grid_conf")
    ap.add_argument("--density", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--confk", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--num-chains", type=int, default=None,
                    help="override config num_chains (default: keep config value)")
    ap.add_argument("--relax", type=int, default=40000)
    ap.add_argument("--burnin", type=int, default=2000)
    ap.add_argument("--traj", type=int, default=800)
    ap.add_argument("--restart", type=int, default=100)
    ap.add_argument("--save", type=int, default=5)
    ap.add_argument("--mdsteps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--cutoff", type=float, default=2.0)
    ap.add_argument("--short-lo", type=float, default=20.0)
    ap.add_argument("--short-hi", type=float, default=100.0)
    ap.add_argument("--tad-lo", type=float, default=150.0)
    ap.add_argument("--tad-hi", type=float, default=500.0)
    ap.add_argument("--target-short", type=float, default=0.90)
    ap.add_argument("--target-tad", type=float, default=1.15)
    args = ap.parse_args(argv)

    root = Path(args.out)
    conds = {"ON": args.c1_base, "OFF": args.c2_base}
    raw = []
    t0 = time.time()

    for s in range(args.seeds):
        # --- 1D LEF once per condition (biology fixed, reused across grid) ---
        lef_paths, cfgs = {}, {}
        chain_length = num_chains = None
        restart = args.restart if args.traj % args.restart == 0 else args.traj
        for cond, base in conds.items():
            shared = root / f"seed{s}" / f"shared_{cond}"
            cfg = load_config(base, output_path=shared)
            if args.num_chains is not None:
                cfg.lef.num_chains = args.num_chains
            cfg.lef.trajectory_length = args.traj
            cfg.lef.warmup_steps = args.warmup
            cfg.lef.seed = 1001 + s
            cfg.polymer.initial_relaxation_steps = args.relax
            cfg.polymer.pre_recording_steps = args.burnin
            cfg.polymer.restart_every_blocks = restart
            cfg.polymer.save_every_blocks = args.save
            cfg.polymer.md_steps_per_block = args.mdsteps
            cfg.polymer.seed = 2001 + s
            lef_stage.run(cfg.lef)
            lef_paths[cond] = shared / "LEFPositions.h5"
            cfgs[cond] = cfg
            chain_length = cfg.lef.chain_length
            num_chains = cfg.lef.num_chains
            print(f"[grid] seed {s} {cond}: lef done "
                  f"({chain_length}x{num_chains} sites)")

        # --- polymer grid: density x confinement_k, both conditions ---
        for dens in args.density:
            for ck in args.confk:
                ps = {}
                t = time.time()
                for cond in conds:
                    cdir = root / f"seed{s}" / f"d{dens:g}_k{ck:g}_{cond}"
                    run_condition(cfgs[cond], lef_paths[cond], cdir, dens, ck)
                    mids, cp, nconf = ps_curve(str(cdir), chain_length, num_chains, args.cutoff)
                    ps[cond] = (mids, cp)
                mids = ps["ON"][0]
                on_s = band_mean(mids, ps["ON"][1], args.short_lo, args.short_hi)
                off_s = band_mean(mids, ps["OFF"][1], args.short_lo, args.short_hi)
                on_t = band_mean(mids, ps["ON"][1], args.tad_lo, args.tad_hi)
                off_t = band_mean(mids, ps["OFF"][1], args.tad_lo, args.tad_hi)
                fold_short = off_s / on_s
                fold_tad = off_t / on_t
                sep = fold_tad - fold_short
                err = abs(fold_short - args.target_short) + abs(fold_tad - args.target_tad)
                rec = {"density": dens, "confk": ck, "seed": s,
                       "fold_short": fold_short, "fold_tad": fold_tad,
                       "crossover_sep": sep, "target_err": err,
                       "seconds": round(time.time() - t, 1)}
                raw.append(rec)
                print(f"[grid] d={dens:g} k={ck:g}  {rec['seconds']:5.0f}s  "
                      f"fold_short={fold_short:.3f} fold_tad={fold_tad:.3f} "
                      f"sep={sep:+.3f} err={err:.3f}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "grid_results_raw.json").write_text(json.dumps(raw, indent=2))

    # aggregate over seeds
    def agg(d, k, metric):
        vals = [r[metric] for r in raw if r["density"] == d and r["confk"] == k]
        a = np.asarray(vals, float)
        return a.mean(), (a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0)

    print("\n=== OFF/ON fold-change  (target: short~0.90, TAD~1.15) ===")
    print(f"{'dens':>6} {'k':>5} {'fold_short':>11} {'fold_tad':>10} "
          f"{'crossover':>10} {'err':>7}")
    best = None
    for d in args.density:
        for k in args.confk:
            fs, _ = agg(d, k, "fold_short")
            ft, _ = agg(d, k, "fold_tad")
            sep, _ = agg(d, k, "crossover_sep")
            er, _ = agg(d, k, "target_err")
            print(f"{d:>6g} {k:>5g} {fs:>11.3f} {ft:>10.3f} {sep:>+10.3f} {er:>7.3f}")
            if best is None or er < best[-1]:
                best = (d, k, fs, ft, er)
    print(f"\n[grid] best by target_err: density={best[0]:g} k={best[1]:g} "
          f"-> short={best[2]:.3f} tad={best[3]:.3f} err={best[4]:.3f}")
    print(f"[grid] {len(raw)} runs in {(time.time()-t0)/60:.1f} min -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
