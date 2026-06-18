#!/usr/bin/env python
"""Re-plot ONLY the mean-loop-length panel from an existing lifetime sweep,
with axis ticks shown in seconds (steps x 8) instead of N steps.

Reads ``sweep_summary.tsv`` + ``method.json`` from the grid out-dir (no H5
reads) and reuses the same colormap / norm / annotations as the original
``sweep_lifetime_cohesin_vs_ctcf_2d.py`` heatmaps so colors match exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_rnapoff_boundary_strength_1d import (  # noqa: E402
    FOLD_CMAP,
    METRIC_LABELS,
    _norm_for_row,
    multiplier_label,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SECONDS_PER_STEP = 8
METRIC = "mean_loop_length"


def second_tick_labels(multipliers: list[float], base_value: float) -> list[str]:
    """Tick label = multiplier plus the resulting lifetime in SECONDS.

    Mirrors the original ``axis_tick_labels`` (multiplier + absolute lifetime in
    steps) but multiplies the step count by ``SECONDS_PER_STEP`` to report
    seconds, matching the user's request to convert the N-step tick to seconds.
    """
    return [
        f"{multiplier_label(m)}\n{int(round(base_value * m)) * SECONDS_PER_STEP}s"
        for m in multipliers
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument(
        "--value-col", default="fold_vs_config1_mean", help="fold-change value column"
    )
    parser.add_argument(
        "--spread-col", default="fold_vs_config1_sd", help="spread annotation column"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    grid_dir = args.grid_dir
    method = json.loads((grid_dir / "method.json").read_text())
    life_mults = method["axes"]["y_cohesin_lifetime"]["multipliers"]
    ctcf_mults = method["axes"]["x_ctcf_lifetime"]["multipliers"]
    life_base = float(method["axes"]["y_cohesin_lifetime"]["base_values"]["lifetime"])
    ctcf_base = float(method["axes"]["x_ctcf_lifetime"]["base_value"])

    summary = pd.read_csv(grid_dir / "sweep_summary.tsv", sep="\t")
    msum = summary[summary["metric"] == METRIC].set_index("config")

    n_life, n_ctcf = len(life_mults), len(ctcf_mults)
    values = np.full((n_life, n_ctcf), np.nan)
    spreads = np.full((n_life, n_ctcf), np.nan)
    for i, lm in enumerate(life_mults):
        for j, cm in enumerate(ctcf_mults):
            label = f"life{multiplier_label(lm)}_ctcf{multiplier_label(cm)}"
            if label in msum.index:
                values[i, j] = float(msum.loc[label, args.value_col])
                spreads[i, j] = float(msum.loc[label, args.spread_col])

    life_ticks = second_tick_labels(life_mults, life_base)
    ctcf_ticks = second_tick_labels(ctcf_mults, ctcf_base)

    fig, ax = plt.subplots(figsize=(0.95 * n_ctcf + 3.4, 0.62 * n_life + 1.7))
    cmap = FOLD_CMAP.copy()
    cmap.set_bad("#f1f3f5")
    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=_norm_for_row(values.ravel()),
    )
    for i in range(n_life):
        for j in range(n_ctcf):
            value = values[i, j]
            sd = spreads[i, j]
            if not np.isfinite(value):
                text = "NA"
            elif np.isfinite(sd):
                text = f"{value:.2f}\n±{sd:.2f}"
            else:
                text = f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="#111111")

    ax.set_xticks(range(n_ctcf))
    ax.set_xticklabels(ctcf_ticks, fontsize=9)
    ax.set_yticks(range(n_life))
    ax.set_yticklabels(life_ticks, fontsize=9)
    ax.set_xlabel("lifetime_ctcf (multiplier / seconds)", fontsize=9)
    ax.set_ylabel("cohesin lifetime (multiplier / seconds)", fontsize=9)
    ax.set_title(METRIC_LABELS[METRIC].replace("\n", " "), fontsize=11)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8, length=2)

    fig.suptitle(
        "Mean loop length: fold change vs config1 (axes in seconds)", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = args.out or (grid_dir / "lifetime_cohesin_vs_ctcf_mean_loop_length_seconds.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
