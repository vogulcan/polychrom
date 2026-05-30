#!/usr/bin/env python
"""Short-trajectory CUDA grid over (selective_attraction_energy, angle_k).

Both knobs are 3D-only FORCE params, so the 1D LEF trajectory is identical
across the force grid *within a seed* -> run `lef` ONCE per seed and reuse its
LEFPositions.h5 for every MD combo of that seed.

Per (combo, seed) we measure, on the saved 3D conformations:
  * P(s) log-log slope in a fit band            -> angle_k (chain stiffness)
  * E-P 3D distance, split short- vs long-range -> selective_attraction_energy
  * E-P distance CONDITIONAL on a cohesin loop actually bracketing the pair
    ("loop-conditioned": isolates where the short-range E-P well can act)
Across seeds we report mean +/- SEM.

Full quality run:
    micromamba run -n openmm python scripts/grid_ps_ep.py \
        --base configs/config1.yaml --out runs/grid_ps_ep_q \
        --sae 0 0.5 1.0 2.0 --angle 0 1.0 --seeds 3 \
        --relax 40000 --burnin 2000 --traj 800 --restart 100 --save 5 --mdsteps 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from polychrom.pipelines.loop_extrusion.config import load_config
from polychrom.pipelines.loop_extrusion import lef as lef_stage
from polychrom.pipelines.loop_extrusion import polymer as polymer_stage
from polychrom.hdf5_format import list_URIs, load_URI_pos
from polychrom.polymer_analyses import contact_scaling, slope_contact_scaling, generate_bins

SHORT_KB = 60     # pairs with |e-p| <= this are "short-range" (already proximal)
LONG_KB = 100     # pairs with |e-p| >= this are "long-range" (cohesin-dependent)


def _brackets(loops, e, p, tol):
    """True if any cohesin loop [la,lb] contains both e and p (with tol slack)."""
    lo, hi = min(e, p), max(e, p)
    for a, b in loops:
        if min(a, b) - tol <= lo and max(a, b) + tol >= hi:
            return True
    return False


def measure(folder, lef_path, ep_pairs, n_sites, cutoff, fit_lo, fit_hi,
            save_every, tol, burn=0.3, seed=0):
    uris = list_URIs(folder)
    if not uris:
        raise RuntimeError(f"no conformations in {folder}")
    start = int(len(uris) * burn)
    kept = list(range(start, len(uris)))
    bins = generate_bins(n_sites, start=4, bins_per_order_magn=10)
    rng = np.random.default_rng(seed)
    wide = 2.5 * cutoff

    with h5py.File(lef_path, "r") as f:
        lef_pos = f["positions"][:]           # (n_frames, n_lef, 2)
    n_frames = lef_pos.shape[0]

    seps = np.array([abs(int(a) - int(b)) for a, b in ep_pairs])
    short_idx = np.where(seps <= SHORT_KB)[0]
    long_idx = np.where(seps >= LONG_KB)[0]

    cps = []
    d_short, d_long, d_ctrl_long = [], [], []
    d_long_looped, hit_wide_long = [], []
    frac_looped = []
    for i, j in enumerate(kept):
        pos = load_URI_pos(uris[j])
        mids, cp = contact_scaling(pos, bins0=bins, cutoff=cutoff)
        cps.append(cp)
        frame = min(j * save_every, n_frames - 1)
        loops = [tuple(map(int, x)) for x in lef_pos[frame]]
        dists = np.array([np.linalg.norm(pos[int(a)] - pos[int(b)]) for a, b in ep_pairs])
        if len(short_idx):
            d_short.append(dists[short_idx].mean())
        for k in long_idx:
            e, p = ep_pairs[k]
            d_long.append(dists[k])
            hit_wide_long.append(float(dists[k] < wide))
            s = seps[k]
            ri = int(rng.integers(0, n_sites - s))
            d_ctrl_long.append(np.linalg.norm(pos[ri] - pos[ri + s]))
            if _brackets(loops, int(e), int(p), tol):
                d_long_looped.append(dists[k])
        frac_looped.append(np.mean([_brackets(loops, int(ep_pairs[k][0]), int(ep_pairs[k][1]), tol)
                                    for k in long_idx]) if len(long_idx) else 0.0)

    cp = np.nanmean(cps, axis=0)
    sl_mids, slope = slope_contact_scaling(mids, cp)
    sl_mids = np.asarray(sl_mids, float); slope = np.asarray(slope, float)
    band = (sl_mids >= fit_lo) & (sl_mids <= fit_hi)
    ps_slope = float(np.nanmean(slope[band])) if band.any() else float("nan")

    d_long = np.asarray(d_long, float)
    d_ctrl_long = np.asarray(d_ctrl_long, float)
    return {
        "n_conf": len(kept),
        "ps_slope": ps_slope,
        "ep_dist_short": float(np.mean(d_short)) if d_short else float("nan"),
        "ep_dist_long": float(d_long.mean()),
        "ep_ratio_long": float(d_long.mean() / d_ctrl_long.mean()),
        "ep_dist_long_looped": float(np.mean(d_long_looped)) if d_long_looped else float("nan"),
        "frac_looped_long": float(np.mean(frac_looped)),
        "ep_freq_wide_long": float(np.mean(hit_wide_long)),
    }


def agg(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/config1.yaml")
    ap.add_argument("--out", default="runs/grid_ps_ep_q")
    ap.add_argument("--sae", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--angle", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--relax", type=int, default=40000)
    ap.add_argument("--burnin", type=int, default=2000)
    ap.add_argument("--traj", type=int, default=800)
    ap.add_argument("--restart", type=int, default=100)
    ap.add_argument("--save", type=int, default=5)
    ap.add_argument("--mdsteps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--cutoff", type=float, default=2.0)
    ap.add_argument("--fit-lo", type=float, default=20.0)
    ap.add_argument("--fit-hi", type=float, default=200.0)
    args = ap.parse_args(argv)

    root = Path(args.out)
    raw = []   # per (sae, angle, seed)
    t_start = time.time()
    for s in range(args.seeds):
        shared = root / f"seed{s}" / "shared"
        cfg = load_config(args.base, output_path=shared)
        cfg.lef.trajectory_length = args.traj
        cfg.lef.warmup_steps = args.warmup
        cfg.lef.seed = 1001 + s
        n_sites = cfg.lef.chain_length * cfg.lef.num_chains
        tol = int(cfg.lef.topology_kwargs.get("ep_contact_tolerance", 1))
        cfg.polymer.initial_relaxation_steps = args.relax
        cfg.polymer.pre_recording_steps = args.burnin
        cfg.polymer.restart_every_blocks = args.restart
        cfg.polymer.save_every_blocks = args.save
        cfg.polymer.md_steps_per_block = args.mdsteps
        cfg.polymer.seed = 2001 + s
        ep_pairs = list(cfg.polymer.plugins.force_builder.kwargs["ep_pairs"])

        lef_stage.run(cfg.lef)
        shared_lef = str(shared / "LEFPositions.h5")
        print(f"[grid] seed {s}: lef done ({n_sites} sites, {len(ep_pairs)} ep_pairs)")

        for sae in args.sae:
            for ak in args.angle:
                name = f"seed{s}/sae{sae:g}_ak{ak:g}"
                combo_dir = root / name
                fb = cfg.polymer.plugins.force_builder.kwargs
                fb["selective_attraction_energy"] = float(sae)
                fb["angle_k"] = None if ak == 0 else float(ak)
                cfg.polymer.lef_positions_path = shared_lef
                cfg.polymer.output_folder = str(combo_dir)
                t = time.time()
                polymer_stage.run(cfg.polymer)
                m = measure(str(combo_dir), shared_lef, ep_pairs, n_sites,
                            args.cutoff, args.fit_lo, args.fit_hi,
                            args.save, tol, seed=1000 + s)
                raw.append({"sae": sae, "angle_k": ak, "seed": s,
                            "seconds": round(time.time() - t, 1), **m})
                print(f"[grid] {name:22s} {raw[-1]['seconds']:5.0f}s "
                      f"slope={m['ps_slope']:+.3f} dL={m['ep_dist_long']:.2f} "
                      f"dL|loop={m['ep_dist_long_looped']:.2f} "
                      f"ratioL={m['ep_ratio_long']:.2f} loopfrac={m['frac_looped_long']:.2f}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "grid_results_raw.json").write_text(json.dumps(raw, indent=2))
    print(f"\n[grid] {len(raw)} runs in {(time.time()-t_start)/60:.1f} min -> {root}/grid_results_raw.json")

    def cell(metric, sae, ak):
        vals = [r[metric] for r in raw if r["sae"] == sae and r["angle_k"] == ak]
        mu, se = agg(vals)
        return f"{mu:6.2f}±{se:.2f}"

    def table(metric, label):
        print(f"\n=== {label}  (mean±SEM, n={args.seeds} seeds) ===")
        print("sae\\ak   " + "      ".join(f"{ak:>10g}" for ak in args.angle))
        for sae in args.sae:
            print(f"{sae:>5g}   " + "   ".join(cell(metric, sae, ak) for ak in args.angle))

    table("ps_slope", "P(s) slope [angle_k: stiffer->flatter]")
    table("ep_dist_long", "long-range E-P distance [sae: lower=>more observable]")
    table("ep_dist_long_looped", "long-range E-P distance | cohesin loop brackets pair [clean sae signal]")
    table("ep_ratio_long", "long-range E-P / matched-control ratio [<1 => enriched]")
    table("frac_looped_long", "fraction of long-range pairs bracketed by a loop")
    table("ep_freq_wide_long", "long-range E-P wide-contact frequency")
    return 0


if __name__ == "__main__":
    sys.exit(main())
