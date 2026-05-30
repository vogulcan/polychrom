#!/usr/bin/env python
"""1D insulation comparison from cohesin loop coverage.

Insulation at a TAD border = how strongly it blocks loop extrusion. In 1D the
clean proxy is COHESIN LOOP COVERAGE C(i) = expected number of cohesin loops
spanning site i (a loop (lo,hi) covers every i in [lo,hi]). A boundary that
stalls extrusion terminates loops, so C(i) dips there; the depth of the dip
relative to the TAD interior is the insulation strength.

Per boundary b we report:
    C_bnd   = C at the boundary site
    C_flank = mean C in the two TAD interiors flanking b (window w)
    depth   = (C_flank - C_bnd) / C_flank   (0..1; higher = sharper boundary)
and the per-config mean depth (overall insulation).

Usage:  python scripts/insulation_1d.py CONFIG.yaml [RUN_DIR_OR_H5] [window]
        python scripts/insulation_1d.py --pair CFG1 H5_1 CFG2 H5_2 [window]
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polychrom.pipelines.loop_extrusion.config import load_config
from tune_depletion_1d import _boundaries


def coverage(h5_path: Path) -> np.ndarray:
    """C(i) = mean over ticks of the number of cohesin loops covering site i."""
    with h5py.File(h5_path, "r") as f:
        pos = f["positions"][:]                       # (T, L, 2)
        N = int(f.attrs["N"])
    T = pos.shape[0]
    left = pos[..., 0].ravel(); right = pos[..., 1].ravel()
    lo = np.minimum(left, right); hi = np.maximum(left, right)
    ok = (left >= 0) & (right >= 0) & (hi > lo)
    lo, hi = lo[ok], hi[ok]
    diff = np.zeros(N + 1, dtype=np.float64)          # difference array
    np.add.at(diff, lo, 1.0)
    np.add.at(diff, hi + 1, -1.0)
    return np.cumsum(diff)[:N] / T                    # expected loops covering site i


def insulation(cfg, h5_path: Path, w: int):
    C = coverage(h5_path)
    N = C.shape[0]
    bnds = [int(b) for b in _boundaries(cfg)]
    inner = [b for b in bnds if 0 < b < N - 1]        # skip the two chromosome ends
    rows = []
    for b in inner:
        lhs = C[max(0, b - w):b]
        rhs = C[b + 1:min(N, b + 1 + w)]
        flank = np.concatenate([lhs, rhs])
        c_bnd = float(C[b])
        c_flk = float(flank.mean()) if flank.size else float("nan")
        depth = (c_flk - c_bnd) / c_flk if c_flk else float("nan")
        rows.append((b, c_bnd, c_flk, depth))
    mean_depth = float(np.nanmean([r[3] for r in rows])) if rows else float("nan")
    return C, rows, mean_depth


def _report(tag, cfg, h5, w):
    C, rows, md = insulation(cfg, h5, w)
    print(f"\n## {tag}  (h5={h5}, window={w} kb)")
    print(f"   mean C (genome) = {C.mean():.3f} loops/site")
    print(f"   {'boundary':>9} {'C_bnd':>8} {'C_flank':>8} {'depth':>7}")
    for b, cb, cf, d in rows:
        print(f"   {b:>9} {cb:>8.3f} {cf:>8.3f} {d:>7.3f}")
    print(f"   --> mean insulation depth = {md:.3f}")
    return md


def main(argv):
    if len(argv) >= 2 and argv[1] == "--pair":
        cfg1 = load_config(argv[2]); h1 = Path(argv[3])
        cfg2 = load_config(argv[4]); h2 = Path(argv[5])
        w = int(argv[6]) if len(argv) > 6 else 100
        d1 = _report("config1 (txn ON)", cfg1, h1, w)
        d2 = _report("config2 (txn OFF)", cfg2, h2, w)
        print(f"\n# insulation depth: config1={d1:.3f}  config2={d2:.3f}  "
              f"config2/config1={d2/d1 if d1 else float('nan'):.3f}  "
              f"(>1 => config2 MORE insulated / sharper TADs)")
        return 0
    cfg = load_config(argv[1])
    h5 = Path(argv[2]) if len(argv) > 2 else Path("trajectory/LEFPositions.h5")
    if not h5.is_file():
        h5 = h5 / "LEFPositions.h5"
    w = int(argv[3]) if len(argv) > 3 else 100
    _report("run", cfg, h5, w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
