#!/usr/bin/env python
"""Two-panel comparison figure from an existing typea-density-grid output dir:

  LEFT  -- Mean loop length, fold vs baseline (diverging scale centred at 1.0),
           read from ``ta_density_folds.tsv`` (metric ``mean_loop_length``).
  RIGHT -- Type-A lesion spacing = kb of gene body per Type-A lesion (log scale),
           the reciprocal of ``typea_per_kb_genebody`` from
           ``ta_density_measured_stall.tsv``; cells annotate the integer kb spacing.

Both panels share identical cosmetics (subplot size, fonts, tick labels, cell text
style) so they line up side by side. No simulations are run -- everything is read
from the TSVs the grid already wrote.
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
from matplotlib.colors import LogNorm, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_lesion_grid_and_heatmaps import FOLD_CMAP, SEQ_CMAP

# Shared cosmetics (mirror plot_heatmaps).
PANEL_W, PANEL_H = 5.2, 4.4
TICK_FS, LABEL_FS, TITLE_FS, CELL_FS = 8, 9, 10, 6
X_LABEL = "Lesion density (lesions/Mbp)"
Y_LABEL = "Lesion Type A probability"


def _pivot(df: pd.DataFrame, value: str) -> tuple[np.ndarray, list[float], list[float]]:
    piv = df.pivot_table(index="lesion_type_a_prob",
                         columns="lesion_density_per_mbp", values=value)
    piv = piv.sort_index(axis=0).sort_index(axis=1)  # type_a asc, density asc
    return piv.to_numpy(dtype=float), [float(v) for v in piv.index], [float(c) for c in piv.columns]


def _draw(ax, grid, *, y_values, density_axis, norm, cmap, title, value_label, cell_text):
    cmap_obj = cmap.copy()
    cmap_obj.set_bad("#f1f3f5")
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto",
                   cmap=cmap_obj, norm=norm)
    ax.set_xticks(range(len(density_axis)))
    ax.set_xticklabels([f"{d:.1f}" for d in density_axis], fontsize=TICK_FS)
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([f"{v:.2f}" for v in y_values], fontsize=TICK_FS)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_FS)
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS, pad=6)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = grid[r, c]
            txt = "n/a" if not np.isfinite(v) else cell_text(v)
            ax.text(c, r, txt, ha="center", va="center", fontsize=CELL_FS, color="#111111")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(value_label, fontsize=TICK_FS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--grid-dir", type=Path, required=True,
                    help="typea-density-grid output dir (holds the two TSVs)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output SVG (default: <grid-dir>/ta_density_loop_vs_typea_spacing.svg)")
    ap.add_argument("--suptitle", type=str, default=None, help="optional figure suptitle")
    args = ap.parse_args()

    folds = pd.read_csv(args.grid_dir / "ta_density_folds.tsv", sep="\t")
    meas = pd.read_csv(args.grid_dir / "ta_density_measured_stall.tsv", sep="\t")

    loop = folds[folds["metric"] == "mean_loop_length"]
    if loop.empty:
        raise SystemExit("no mean_loop_length rows in ta_density_folds.tsv")
    loop_grid, ta_l, dens_l = _pivot(loop, "fold_vs_baseline")
    perkb_grid, ta_r, dens_r = _pivot(meas, "typea_per_kb_genebody")
    if ta_l != ta_r or [round(d, 6) for d in dens_l] != [round(d, 6) for d in dens_r]:
        raise SystemExit("axis mismatch between folds and measured TSVs")

    with np.errstate(divide="ignore", invalid="ignore"):
        spacing = np.where(perkb_grid > 0, 1.0 / perkb_grid, np.nan)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(PANEL_W * 2, PANEL_H), squeeze=True)

    # LEFT: mean loop length fold vs baseline (diverging, centred at 1.0).
    fl = loop_grid[np.isfinite(loop_grid)]
    lmin, lmax = (float(fl.min()), float(fl.max())) if fl.size else (0.0, 2.0)
    _draw(axL, loop_grid, y_values=ta_l, density_axis=dens_l,
          norm=TwoSlopeNorm(vcenter=1.0, vmin=min(lmin, 1.0 - 1e-6), vmax=max(lmax, 1.0 + 1e-6)),
          cmap=FOLD_CMAP, title="Mean loop length", value_label="fold vs baseline mean",
          cell_text=lambda v: f"{v:.2f}")

    # RIGHT: Type-A lesion spacing (log sequential); cells = integer kb.
    sp = spacing[np.isfinite(spacing)]
    spos = sp[sp > 0]
    smin = float(spos.min()) if spos.size else 1.0
    smax = float(sp.max()) if sp.size else smin * 10.0
    _draw(axR, spacing, y_values=ta_r, density_axis=dens_r,
          norm=LogNorm(vmin=smin, vmax=smax if smax > smin else smin * 10.0),
          cmap=SEQ_CMAP, title="Type-A lesions: 1 per X kb of gene body",
          value_label="kb of gene body per Type-A lesion", cell_text=lambda v: f"{round(v):d}")

    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if args.suptitle else 1.0))
    out = args.out or (args.grid_dir / "ta_density_loop_vs_typea_spacing.svg")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
