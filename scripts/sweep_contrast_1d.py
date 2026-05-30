#!/usr/bin/env python
"""2D sweep to maximize config2/config1 (txn-OFF / txn-ON) 1D effect sizes.

Sweeps config1's transcription-coupled cohesin eviction strength
(lifetime_rnapii_stalled) x config2's CTCF boundary strength, holding everything
else at the live config1.yaml. Reuses metrics_1d from the tuning driver and
reports all six CONFIG_EXPLANATIONS folds so we can pick the Pareto-best point.

Usage:  python scripts/sweep_contrast_1d.py [traj] [out_dir]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polychrom.pipelines.loop_extrusion import lef as lef_stage
from polychrom.pipelines.loop_extrusion.config import load_config
from tune_depletion_1d import _boundaries, metrics_1d, _scale_boundary, _disable_rnapii

# config1 barrier params held fixed (the biological "firm semi-permeable" set)
STALL, BLOCK, HEADON = 0.35, 0.92, 0.05
C1_BND = 0.62                     # config1 boundary (held)
EVICT = [8, 12]                   # config1 lifetime_rnapii_stalled
PUSH = [0.20, 0.35, 0.50]         # rnapii_push_prob (all "cohesin follows txn")
C2_BND = [0.58, 0.60]             # config2 boundary (< config1 0.62)
SPEC = [("mean_loop", "UP"), ("frac_short", "DN"), ("frac_tad", "UP"),
        ("boundary_occ_pct", "UP"), ("corner_both_pct", "UP"),
        ("cross_boundary_pct", "UP")]


def _cfg(evict, push):
    cfg = load_config("configs/config1.yaml")
    tk = cfg.lef.topology_kwargs
    tk["lifetime_rnapii_stalled"] = int(evict)
    tk["rnapii_stall_prob"] = STALL
    tk["rnapii_elongating_block_prob"] = BLOCK
    tk["rnapii_push_prob"] = push
    tk["rnapii_headon_push_prob"] = HEADON
    _scale_boundary(tk, C1_BND)
    return cfg


def _run(cfg, out, traj):
    cfg.lef.output_path = str(out)
    cfg.lef.trajectory_length = traj
    cfg.lef.warmup_steps = min(int(cfg.lef.warmup_steps), max(2000, traj // 5))
    lef_stage.run(cfg.lef)
    return metrics_1d(out, _boundaries(cfg))


def main(argv):
    traj = int(argv[1]) if len(argv) > 1 else 40000
    tmp = Path(argv[2]) if len(argv) > 2 else Path("runs/sweep1d")
    tmp.mkdir(parents=True, exist_ok=True)
    # WT (config1, txn ON) depends on (evict, push); cache per pair.
    wt = {}
    for e in EVICT:
        for p in PUSH:
            wt[(e, p)] = _run(_cfg(e, p), tmp / f"wt_e{e}_p{p}.h5", traj)
    print(f"# traj={traj}  config1 stall={STALL} block={BLOCK} c1_bnd={C1_BND}\n")
    hdr = "evict push c2bnd " + " ".join(f"{k.split('_')[0][:5]:>7}" for k, _ in SPEC)
    print(hdr)
    rows = []
    for e in EVICT:
        for p in PUSH:
            for b in C2_BND:
                cfg = _cfg(e, p)        # config2 = config1 + txn off + lower bnd
                _disable_rnapii(cfg)
                _scale_boundary(cfg.lef.topology_kwargs, b)
                dep = _run(cfg, tmp / f"dep_e{e}_p{p}_b{b}.h5", traj)
                w = wt[(e, p)]
                folds = {k: (dep[k] / w[k] if w[k] else float("nan"))
                         for k, _ in SPEC}
                allok = all((folds[k] > 1) == (wnt == "UP") for k, wnt in SPEC)
                score = sum(abs(np.log(folds[k])) for k, _ in SPEC)
                rows.append((e, p, b, folds, allok, score))
                fline = " ".join(f"{folds[k]:>7.3f}" for k, _ in SPEC)
                print(f"{e:>5} {p:>4} {b:>5} {fline}  "
                      f"{'OK' if allok else '..'}  spread={score:.2f}")
    ok = [r for r in rows if r[4]]
    pool = ok if ok else rows
    best = max(pool, key=lambda r: r[5])
    print(f"\n# BEST (all-correct={'yes' if best[4] else 'NO'}): "
          f"evict={best[0]} push={best[1]} c2_boundary={best[2]} spread={best[5]:.2f}")
    print("  " + " ".join(f"{k}={best[3][k]:.3f}" for k, _ in SPEC))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
