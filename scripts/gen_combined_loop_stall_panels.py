#!/usr/bin/env python
"""One combined SVG assembled from two existing grid output dirs (no sims run).

Panels (all read from the grids' TSVs):

  Row 1 -- block_prob x density grid (``block_prob_grid_*``), y = block probability:
    A  Mean loop length, fold vs baseline      (lesion_grid_folds.tsv)
    B  Type A pre-recognition stall, mean s     \
    C  Type A repair stall, mean s               > lesion_grid_measured_stall.tsv
    D  Type B repair stall, mean s              /
  Row 2 -- Type-A prob x density grid (``typea_density_grid_*``), y = Type-A prob:
    E  Mean loop length, fold vs baseline       (ta_density_folds.tsv)
    F  Type-A lesion spacing, 1 per X kb (log)   (ta_density_measured_stall.tsv)

Each panel keeps the colour scale appropriate to its quantity (fold -> diverging
centred at 1.0; seconds -> sequential; spacing -> log sequential), its own
colorbar, and shared cosmetics. X-tick labels are rotated so the 12 density
values no longer overlap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_lesion_grid_and_heatmaps import FOLD_CMAP, SEQ_CMAP

TICK_FS, LABEL_FS, TITLE_FS, CELL_FS = 8, 9, 10, 7.5
X_LABEL = "Lesion density (lesions/Mbp)"


def _pivot(df, value, ycol):
    piv = df.pivot_table(index=ycol, columns="lesion_density_per_mbp", values=value)
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    return piv.to_numpy(dtype=float), [float(v) for v in piv.index], [float(c) for c in piv.columns]


def _draw(ax, grid, *, y_values, density_axis, norm, cmap, title, value_label,
          cell_text, y_fmt, y_label):
    cmap_obj = cmap.copy()
    cmap_obj.set_bad("#f1f3f5")
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto",
                   cmap=cmap_obj, norm=norm)
    ax.set_xticks(range(len(density_axis)))
    ax.set_xticklabels([f"{d:.1f}" for d in density_axis], fontsize=TICK_FS,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([y_fmt.format(v) for v in y_values], fontsize=TICK_FS)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_FS)
    ax.set_ylabel(y_label, fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS, pad=6)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = grid[r, c]
            txt = "n/a" if not np.isfinite(v) else cell_text(v)
            ax.text(c, r, txt, ha="center", va="center", fontsize=CELL_FS, color="#111111")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(value_label, fontsize=TICK_FS)


def _fold_norm(grid):
    f = grid[np.isfinite(grid)]
    lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 2.0)
    return TwoSlopeNorm(vcenter=1.0, vmin=min(lo, 1.0 - 1e-6), vmax=max(hi, 1.0 + 1e-6))


def _seq_norm(grid):
    f = grid[np.isfinite(grid)]
    lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 1.0)
    return Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-6)


def _log_norm(grid):
    f = grid[np.isfinite(grid)]
    pos = f[f > 0]
    lo = float(pos.min()) if pos.size else 1.0
    hi = float(f.max()) if f.size else lo * 10.0
    return LogNorm(vmin=lo, vmax=hi if hi > lo else lo * 10.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--block-dir", type=Path,
                    default=ROOT / "results_10k/block_prob_grid_typea01_10k")
    ap.add_argument("--density-dir", type=Path,
                    default=ROOT / "results_10k/typea_density_grid_10k")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results_10k/combined_loop_and_stall_panels.svg")
    args = ap.parse_args()

    # --- load block_prob grid (row 1) ---
    bfolds = pd.read_csv(args.block_dir / "lesion_grid_folds.tsv", sep="\t")
    bmeas = pd.read_csv(args.block_dir / "lesion_grid_measured_stall.tsv", sep="\t")
    YB = "lesion_block_prob"
    bloop, by_b, dx_b = _pivot(bfolds[bfolds["metric"] == "mean_loop_length"],
                               "fold_vs_baseline", YB)
    a_pre, _, _ = _pivot(bmeas, "a_pre_dwell_mean_s", YB)
    a_rep, _, _ = _pivot(bmeas, "a_repair_dwell_mean_s", YB)
    b_rep, _, _ = _pivot(bmeas, "b_repair_dwell_mean_s", YB)

    # --- load type-A density grid (row 2) ---
    dfolds = pd.read_csv(args.density_dir / "ta_density_folds.tsv", sep="\t")
    dmeas = pd.read_csv(args.density_dir / "ta_density_measured_stall.tsv", sep="\t")
    YD = "lesion_type_a_prob"
    dloop, by_d, dx_d = _pivot(dfolds[dfolds["metric"] == "mean_loop_length"],
                               "fold_vs_baseline", YD)
    perkb, _, _ = _pivot(dmeas, "typea_per_kb_genebody", YD)
    with np.errstate(divide="ignore", invalid="ignore"):
        spacing = np.where(perkb > 0, 1.0 / perkb, np.nan)

    sec = lambda v: f"{v:.0f}"
    f2 = lambda v: f"{v:.2f}"
    yb_fmt, yd_fmt = "{:.3f}", "{:.2f}"
    YB_LABEL = "Lesion block probability p (per tick)"
    YD_LABEL = "Lesion Type A probability"

    # 3 rows x 2 cols, one heatmap per cell. Block grid (4 panels) fills rows 1-2,
    # density grid (2 panels) fills row 3. Explicit margins so the section headers
    # sit cleanly in the inter-row gaps (no bbox="tight", which would shift them).
    fig = plt.figure(figsize=(2 * 7.0, 3 * 4.8))
    gs = fig.add_gridspec(3, 2, left=0.07, right=0.96, top=0.90, bottom=0.05,
                          hspace=0.62, wspace=0.32)
    ax = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

    _draw(ax[0][0], bloop, y_values=by_b, density_axis=dx_b, norm=_fold_norm(bloop),
          cmap=FOLD_CMAP, title="Mean loop length", value_label="fold vs baseline mean",
          cell_text=f2, y_fmt=yb_fmt, y_label=YB_LABEL)
    _draw(ax[0][1], a_pre, y_values=by_b, density_axis=dx_b, norm=_seq_norm(a_pre),
          cmap=SEQ_CMAP, title="Type A pre-recognition stall\n(mean, s)",
          value_label="stall duration (s)", cell_text=sec, y_fmt=yb_fmt, y_label=YB_LABEL)
    _draw(ax[1][0], a_rep, y_values=by_b, density_axis=dx_b, norm=_seq_norm(a_rep),
          cmap=SEQ_CMAP, title="Type A repair stall\n(mean, s)",
          value_label="stall duration (s)", cell_text=sec, y_fmt=yb_fmt, y_label=YB_LABEL)
    _draw(ax[1][1], b_rep, y_values=by_b, density_axis=dx_b, norm=_seq_norm(b_rep),
          cmap=SEQ_CMAP, title="Type B repair stall\n(mean, s)",
          value_label="stall duration (s)", cell_text=sec, y_fmt=yb_fmt, y_label=YB_LABEL)

    _draw(ax[2][0], dloop, y_values=by_d, density_axis=dx_d, norm=_fold_norm(dloop),
          cmap=FOLD_CMAP, title="Mean loop length", value_label="fold vs baseline mean",
          cell_text=f2, y_fmt=yd_fmt, y_label=YD_LABEL)
    _draw(ax[2][1], spacing, y_values=by_d, density_axis=dx_d, norm=_log_norm(spacing),
          cmap=SEQ_CMAP, title="Type-A lesions: 1 per X kb of gene body",
          value_label="kb of gene body per Type-A lesion",
          cell_text=lambda v: f"{round(v):d}", y_fmt=yd_fmt, y_label=YD_LABEL)

    # Section headers above the block-grid block (rows 1-2) and the density-grid
    # row (row 3). Header 1 sits near the page top, clear of the 2-line panel
    # titles; header 2 tracks the gap above row 3.
    top3 = ax[2][0].get_position().y1
    fig.text(0.5, 0.965, "block_prob x density grid  (Type-A prob fixed)  -  "
             f"{args.block_dir.name}", ha="center", fontsize=13, weight="bold")
    fig.text(0.5, top3 + 0.035, "Type-A prob x density grid  -  "
             f"{args.density_dir.name}", ha="center", fontsize=13, weight="bold")

    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
