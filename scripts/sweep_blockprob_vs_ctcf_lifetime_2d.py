#!/usr/bin/env python
"""2D RNAPII-cohesin block-probability x CTCF-lifetime sweep at the 1D LEF stage.

Companion to ``sweep_lifetime_cohesin_vs_ctcf_2d.py``. Instead of sweeping the
cohesin lifetime, this script sweeps two axes and produces a 2D heatmap per
metric:

* **y axis - RNAPII-cohesin block probability (active trio).** A single absolute
  probability is applied jointly to ``lef.topology_kwargs.rnapii_paused_block_prob``,
  ``rnapii_elongating_block_prob`` and ``rnapii_terminating_block_prob`` (the
  "strong" barriers, baseline 0.923 each). The pre-initiation barrier
  (``rnapii_pre_initiation_block_prob``, baseline 0.513) is left fixed at its baseline
  value. Each value is the probability that cohesin is blocked when it meets a
  paused/elongating/terminating Pol II.
* **x axis - ``lef.lifetime_ctcf``** (the WAPL-protected lifetime of cohesin
  captured at a CTCF anchor), swept by its own multiplier.

Each grid cell (block prob, CTCF-lifetime multiplier) gets its own derivative
config, its 1D trajectory is run/reused, and metrics are reported as fold changes
versus the baseline config1 replicate-chain mean.

RNAP is **left on** (NOT forced off): the block-probability axis is only
meaningful with RNAPII present.

Note: the H5-staleness check inspects only topology fields (frames, LEFs, N,
chain_length, num_chains, separation) - NOT block probabilities or
``lifetime_ctcf``. A fresh ``--out-dir`` is therefore correct (each cell's H5 is
absent and gets run). If you reuse a populated out-dir after changing either
axis, pass ``--force-1d`` to regenerate trajectories.
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported so that running many 1D
# sims under --jobs does not oversubscribe cores (each grid point is numpy-light;
# parallelism comes from running whole sims concurrently, not threaded BLAS).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import logging
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_config_chain_metrics import (  # noqa: E402
    STAT_TEST_NAME,
    _topology,
    _validate_compatible,
    analyze_run,
)

# Reuse the boundary-strength companion's building blocks so the run/analyze/
# summarize/colormap logic stays single-sourced.
from sweep_rnapoff_boundary_strength_1d import (  # noqa: E402
    FOLD_CMAP,
    METRIC_LABELS,
    METRIC_ORDER,
    _norm_for_row,
    _run_sweep_point,
    multiplier_label,
    parse_multipliers,
    read_yaml,
    selected_metric_rows,
    summarize_sweep,
    validate_baseline_h5,
    write_yaml,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The three "active"/strong RNAPII barrier states swept jointly on the y axis.
TRIO_KEYS = (
    "rnapii_paused_block_prob",
    "rnapii_elongating_block_prob",
    "rnapii_terminating_block_prob",
)

# Default seconds per simulation step, used only to annotate the block-prob axis
# with an effective RNAPII dwell time (see ``block_tick_labels``).
SECONDS_PER_STEP = 8.0


def block_prob_values(lo: float, hi: float, step: float) -> list[float]:
    """Ascending block probabilities from ``lo`` to ``hi`` (inclusive) by ``step``.

    Values are clamped to the closed range [0, 1] (0.0 = "cohesin never blocked",
    1.0 = "always blocked"). Ascending order so that, with ``origin='lower'``
    heatmaps, the block probability increases upward. The lo/hi/step convention
    mirrors ``--p-lo/--p-hi/--p-step`` in ``gen_lesion_grid_and_heatmaps.py``.
    """
    if step <= 0:
        raise argparse.ArgumentTypeError("--block-prob-step must be positive")
    if not (0.0 <= lo <= hi <= 1.0):
        raise argparse.ArgumentTypeError(
            "require 0 <= --block-prob-lo <= --block-prob-hi <= 1"
        )
    n = int(round((hi - lo) / step))
    values = [round(lo + i * step, 6) for i in range(n + 1)]
    return [min(1.0, max(0.0, v)) for v in values]


def inject_baseline_prob(
    probs: list[float],
    baseline: float,
    step: float,
    eps: float = 1e-9,
) -> tuple[list[float], str]:
    """Guarantee the config1 baseline block prob appears exactly in the y axis.

    So that one row reproduces config1 (the fold≈1.0 sanity cell when combined
    with the 1x lifetime_ctcf column):

    * If a generated value already equals the baseline, nothing changes.
    * Else if the nearest generated value is within ``step/2`` of the baseline and
      is an interior point (not the lo/hi endpoint), it is *snapped* to the exact
      baseline (e.g. a swept 0.925 becomes the config's 0.923) - avoiding a
      near-duplicate row and preserving the requested endpoints.
    * Else the baseline is *inserted* as an extra row.

    Returns the adjusted, de-duplicated, ascending list and a short note for logs.
    """
    if baseline is None:
        return probs, "no baseline block prob resolved; left unchanged"
    values = list(probs)
    if any(abs(p - baseline) <= eps for p in values):
        return values, f"baseline {baseline:g} already on the grid"

    note = ""
    if values:
        lo_val, hi_val = values[0], values[-1]
        nearest_i = min(range(len(values)), key=lambda i: abs(values[i] - baseline))
        nearest = values[nearest_i]
        is_endpoint = abs(nearest - lo_val) <= eps or abs(nearest - hi_val) <= eps
        if abs(nearest - baseline) < step / 2.0 and not is_endpoint:
            note = f"snapped {nearest:g} -> baseline {baseline:g}"
            values[nearest_i] = baseline
    if not any(abs(p - baseline) <= eps for p in values):
        values.append(baseline)
        note = note or f"inserted baseline {baseline:g}"

    # De-dup/sort, but keep the baseline value byte-for-byte (un-rounded) so the
    # cell uses exactly the parameter from config1.
    kept: list[float] = []
    for v in sorted(values):
        v = baseline if abs(v - baseline) <= eps else round(min(1.0, max(0.0, v)), 6)
        if not any(abs(v - k) <= eps for k in kept):
            kept.append(v)
    return kept, note


def prob_label(prob: float) -> str:
    """Filesystem/display-safe label for an absolute probability.

    Mirrors ``multiplier_label`` styling: ``1.0 -> "1"``, ``0.0 -> "0"``,
    ``0.5 -> "0p5"``, ``0.923 -> "0p923"`` (dots replaced with ``p``).
    """
    if float(prob).is_integer():
        return str(int(prob))
    return f"{prob:g}".replace(".", "p")


def _resolve_block_prob(kwargs: dict, key: str) -> float:
    """Resolve a baseline trio block prob using the topology's fallback chain.

    paused/elongating fall back to ``rnapii_block_prob`` (default 1.0);
    terminating falls back to the resolved paused value. Used only for
    ``method.json`` provenance - the sweep itself sets the three keys explicitly.
    """
    generic = float(kwargs.get("rnapii_block_prob", 1.0))
    if key == "rnapii_terminating_block_prob":
        explicit = kwargs.get(key)
        if explicit is not None:
            return float(explicit)
        return _resolve_block_prob(kwargs, "rnapii_paused_block_prob")
    value = kwargs.get(key)
    return float(value) if value is not None else generic


def _base_block_probs(base_cfg: dict) -> dict:
    kwargs = base_cfg["lef"].get("topology_kwargs", {}) or {}
    return {key: _resolve_block_prob(kwargs, key) for key in TRIO_KEYS}


def _base_ctcf_lifetime(base_cfg: dict) -> float:
    """``lifetime_ctcf`` falls back to the base ``lifetime`` when unset."""
    lef = base_cfg["lef"]
    ctcf = lef.get("lifetime_ctcf")
    return float(ctcf) if ctcf is not None else float(lef.get("lifetime", 200))


def make_grid_config(
    base_cfg: dict,
    base_ctcf: float,
    block_prob: float,
    ctcf_mult: float,
) -> tuple[dict, dict]:
    """Build one grid-cell config.

    ``block_prob`` is assigned (as an absolute probability) to the three active
    RNAPII barrier states together; ``ctcf_mult`` scales the CTCF-capture
    lifetime. The pre-initiation block prob, the cohesin lifetimes and all RNAP settings
    are left exactly as in the baseline config.
    """
    cfg = copy.deepcopy(base_cfg)
    lef = cfg["lef"]
    kwargs = lef.setdefault("topology_kwargs", {})

    for key in TRIO_KEYS:
        kwargs[key] = block_prob

    lifetime_ctcf = int(round(base_ctcf * ctcf_mult))
    lef["lifetime_ctcf"] = lifetime_ctcf

    record = {
        "rnapii_block_prob": block_prob,
        "ctcf_lifetime_multiplier": ctcf_mult,
        "lifetime_ctcf": lifetime_ctcf,
        "rnapii_paused_block_prob": block_prob,
        "rnapii_elongating_block_prob": block_prob,
        "rnapii_terminating_block_prob": block_prob,
        "rnapii_pre_initiation_block_prob": kwargs.get("rnapii_pre_initiation_block_prob"),
    }
    return cfg, record


def cell_label(block_prob: float, ctcf_mult: float) -> str:
    return f"block{prob_label(block_prob)}_ctcf{multiplier_label(ctcf_mult)}"


def block_dwell_seconds(prob: float, seconds_per_step: float) -> float:
    """Effective RNAPII dwell time implied by a per-step block probability.

    At a barrier the cohesin is blocked with probability ``p`` each step and
    passes with probability ``1 - p``, so the mean number of blocked steps before
    it passes is the geometric mean ``p / (1 - p)``. Multiplying by the wall-clock
    seconds per step gives an effective dwell time (e.g. p=0.923 at 8 s/step ->
    ~96 s). ``p = 1`` (always blocked) is an infinite dwell.
    """
    if prob >= 1.0:
        return float("inf")
    return prob / (1.0 - prob) * seconds_per_step


def block_tick_labels(probs: list[float], seconds_per_step: float = SECONDS_PER_STEP) -> list[str]:
    """y-axis tick = absolute block probability plus its effective dwell (seconds).

    Second line is the RNAPII dwell time ``p/(1-p) * seconds_per_step`` (``inf``
    shown as ``∞``) so the probability is readable as a residence time.
    """
    labels = []
    for p in probs:
        dwell = block_dwell_seconds(p, seconds_per_step)
        dwell_text = "∞" if not math.isfinite(dwell) else f"{dwell:.0f}s"
        labels.append(f"{p:g}\n{dwell_text}")
    return labels


def ctcf_tick_labels(multipliers: list[float], base_value: float) -> list[str]:
    """x-axis tick = multiplier plus the resulting absolute lifetime (steps)."""
    return [f"{multiplier_label(m)}\n{int(round(base_value * m))}" for m in multipliers]


def plot_grid_heatmaps(
    summary: pd.DataFrame,
    cell_grid: list[list[str]],
    block_ticks: list[str],
    ctcf_ticks: list[str],
    out_path: Path,
    value_col: str = "fold_vs_config1_mean",
    spread_col: str = "fold_vs_config1_sd",
    title: str = "RNAPII block-prob x CTCF-lifetime sweep: fold change vs config1",
) -> None:
    """One 2D heatmap per metric. x = lifetime_ctcf, y = RNAPII block prob.

    ``cell_grid[i][j]`` is the config label for block-prob row ``i`` and
    CTCF-lifetime column ``j``. Rows are drawn with ``origin='lower'`` so the
    block probability increases upward.
    """
    n_block = len(cell_grid)
    n_ctcf = len(cell_grid[0]) if n_block else 0
    metrics = METRIC_ORDER
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * (0.95 * n_ctcf + 3.4), nrows * (0.62 * n_block + 1.7)),
        squeeze=False,
    )

    for m_idx, metric in enumerate(metrics):
        r, c = divmod(m_idx, ncols)
        ax = axes[r][c]
        msum = summary[summary["metric"] == metric].set_index("config")

        values = np.full((n_block, n_ctcf), np.nan)
        spreads = np.full((n_block, n_ctcf), np.nan)
        for i in range(n_block):
            for j in range(n_ctcf):
                label = cell_grid[i][j]
                if label in msum.index:
                    values[i, j] = float(msum.loc[label, value_col])
                    spreads[i, j] = float(msum.loc[label, spread_col])

        cmap = FOLD_CMAP.copy()
        cmap.set_bad("#f1f3f5")
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=_norm_for_row(values.ravel()),
        )
        for i in range(n_block):
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
        ax.set_yticks(range(n_block))
        ax.set_yticklabels(block_ticks, fontsize=9)
        ax.set_xlabel("lifetime_ctcf (multiplier / steps)", fontsize=9)
        ax.set_ylabel("RNAPII block prob (prob / dwell s, paused/elong/term)", fontsize=9)
        ax.set_title(METRIC_LABELS[metric].replace("\n", " "), fontsize=10)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8, length=2)

    # Hide any unused trailing axes (when len(metrics) is odd vs the 2-col grid).
    for extra in range(len(metrics), nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 2D RNAPII block-prob x CTCF-lifetime sweep, run 1D, and plot fold heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config1", type=Path, required=True, help="baseline config1 YAML")
    parser.add_argument("--h5-config1", type=Path, required=True, help="existing baseline LEFPositions.h5")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--block-prob-lo",
        type=float,
        default=0.0,
        help="y-axis lowest RNAPII block probability (bottom row)",
    )
    parser.add_argument(
        "--block-prob-hi",
        type=float,
        default=1.0,
        help="y-axis highest RNAPII block probability (top row)",
    )
    parser.add_argument(
        "--block-prob-step",
        type=float,
        default=0.2,
        help="y-axis RNAPII block-probability increment (lo->hi inclusive), "
             "applied jointly to rnapii_paused/elongating/terminating_block_prob",
    )
    parser.add_argument(
        "--ctcf-lifetime-multipliers",
        type=parse_multipliers,
        default=parse_multipliers("0.125,0.25,0.5,1,2,4,6"),
        help="x-axis multipliers applied to lifetime_ctcf",
    )
    parser.add_argument(
        "--seconds-per-step",
        type=float,
        default=SECONDS_PER_STEP,
        help="wall-clock seconds per simulation step; annotates the block-prob "
             "axis with the effective RNAPII dwell time p/(1-p)*seconds_per_step",
    )
    parser.add_argument("--force-1d", action="store_true", help="rerun generated grid H5s")
    parser.add_argument("--label1", default="config1", help="baseline label")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of grid-point 1D sims to run concurrently (process pool); "
             "1 = sequential. Each sim is an independent process writing its own "
             "H5, so results are identical to sequential.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)

    out_dir = args.out_dir
    configs_dir = out_dir / "configs"
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = read_yaml(args.config1)
    base_topology = _topology(base_cfg)
    base_ctcf = _base_ctcf_lifetime(base_cfg)
    base_blocks = _base_block_probs(base_cfg)
    validate_baseline_h5(args.config1, args.h5_config1, args.label1)

    print(
        "[base] RNAPII block probs (paused/elongating/terminating) = "
        f"{base_blocks['rnapii_paused_block_prob']:g}/"
        f"{base_blocks['rnapii_elongating_block_prob']:g}/"
        f"{base_blocks['rnapii_terminating_block_prob']:g}; "
        f"pre-initiation (fixed) = {base_cfg['lef'].get('topology_kwargs', {}).get('rnapii_pre_initiation_block_prob')}; "
        f"lifetime_ctcf = {int(base_ctcf)}"
    )

    baseline_core, baseline_boundary, _, _ = analyze_run(
        args.label1,
        args.config1,
        args.h5_config1,
        base_topology,
    )
    baseline_rows = selected_metric_rows(baseline_core, baseline_boundary)

    block_probs = block_prob_values(args.block_prob_lo, args.block_prob_hi, args.block_prob_step)
    # Always include the config1 baseline block prob exactly (snap a near grid
    # value to it, else insert it) so one row reproduces config1.
    baseline_block = base_blocks["rnapii_paused_block_prob"]
    if len({round(v, 9) for v in base_blocks.values()}) > 1:
        print(
            "[base] WARNING: baseline trio block probs differ "
            f"({base_blocks}); using paused={baseline_block:g} as the reproduced baseline"
        )
    block_probs, inject_note = inject_baseline_prob(block_probs, baseline_block, args.block_prob_step)
    print(f"[base] block-prob axis: {[f'{p:g}' for p in block_probs]} ({inject_note})")
    ctcf_mults = args.ctcf_lifetime_multipliers

    generated_config_rows: list[dict] = []
    all_metric_rows = [baseline_rows]
    grid_labels: list[str] = []
    # cell_grid[i][j] -> config label (i indexes block prob, j indexes ctcf)
    cell_grid: list[list[str]] = [["" for _ in ctcf_mults] for _ in block_probs]
    cell_meta: dict[str, dict] = {}
    generated_paths: dict[str, dict] = {}
    tasks: list[dict] = []

    # 1) generate every grid config (cheap; always sequential) ------------
    for i, block_prob in enumerate(block_probs):
        for j, ctcf_mult in enumerate(ctcf_mults):
            label = cell_label(block_prob, ctcf_mult)
            grid_labels.append(label)
            cell_grid[i][j] = label
            cell_meta[label] = {
                "rnapii_block_prob": block_prob,
                "ctcf_lifetime_multiplier": ctcf_mult,
            }

            cfg, record = make_grid_config(base_cfg, base_ctcf, block_prob, ctcf_mult)
            cfg_path = configs_dir / f"{label}.yaml"
            h5_path = runs_dir / label / "LEFPositions.h5"
            write_yaml(cfg_path, cfg)
            generated_paths[label] = {"config": cfg_path, "h5": h5_path}

            generated_config_rows.append(
                {
                    "config": label,
                    **record,
                    "config_path": str(cfg_path),
                    "h5_path": str(h5_path),
                    "max_rnapii": cfg["lef"].get("max_rnapii"),
                }
            )

            _validate_compatible(base_topology, _topology(cfg))
            tasks.append(
                {
                    "label": label,
                    "cfg_path": str(cfg_path),
                    "h5_path": str(h5_path),
                    "out_dir": str(out_dir),
                    "force_1d": args.force_1d,
                }
            )

    # 2) run each grid point's 1D LEF stage (sequential or process pool) ---
    total = len(tasks)
    jobs = max(1, int(args.jobs))
    resolved_h5: dict[str, Path] = {}
    if jobs == 1:
        for n, task in enumerate(tasks, 1):
            print(f"[{n}/{total}] {task['label']}")
            res = _run_sweep_point(task)
            resolved_h5[res["label"]] = Path(res["h5"])
    else:
        print(f"running {total} grid points with --jobs {jobs} (process pool, fork context)")
        ctx = mp.get_context("fork")
        n = 0
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for res in ex.map(_run_sweep_point, tasks):
                n += 1
                print(f"[{n}/{total}] done {res['label']}")
                resolved_h5[res["label"]] = Path(res["h5"])

    # 3) analyze each resolved H5 (in main process, grid order) -----------
    for task in tasks:
        label = task["label"]
        core_rows, boundary_metric_rows, _, _ = analyze_run(
            label, Path(task["cfg_path"]), resolved_h5[label], base_topology
        )
        all_metric_rows.append(selected_metric_rows(core_rows, boundary_metric_rows))

    per_chain = pd.concat(all_metric_rows, ignore_index=True)
    summary, stats = summarize_sweep(per_chain, args.label1, grid_labels)

    # Attach the two axis values to the per-cell summary/stats for clarity.
    for frame in (summary, stats):
        frame["rnapii_block_prob"] = frame["config"].map(
            lambda c: cell_meta.get(c, {}).get("rnapii_block_prob")
        )
        frame["ctcf_lifetime_multiplier"] = frame["config"].map(
            lambda c: cell_meta.get(c, {}).get("ctcf_lifetime_multiplier")
        )

    per_chain.to_csv(out_dir / "sweep_per_chain_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "sweep_summary.tsv", sep="\t", index=False)
    stats.to_csv(out_dir / "sweep_stats.tsv", sep="\t", index=False)
    pd.DataFrame(generated_config_rows).to_csv(out_dir / "generated_configs.tsv", sep="\t", index=False)

    block_ticks = block_tick_labels(block_probs, args.seconds_per_step)
    ctcf_ticks = ctcf_tick_labels(ctcf_mults, base_ctcf)

    svg_path = out_dir / "blockprob_vs_ctcf_lifetime_heatmaps.svg"
    plot_grid_heatmaps(summary, cell_grid, block_ticks, ctcf_ticks, svg_path)

    median_svg_path = out_dir / "blockprob_vs_ctcf_lifetime_heatmaps_median.svg"
    plot_grid_heatmaps(
        summary,
        cell_grid,
        block_ticks,
        ctcf_ticks,
        median_svg_path,
        value_col="fold_vs_config1_median",
        spread_col="fold_vs_config1_median_mad",
        title="RNAPII block-prob x CTCF-lifetime sweep: median fold change vs config1",
    )

    method = {
        "baseline_label": args.label1,
        "baseline_config": str(args.config1),
        "baseline_h5": str(args.h5_config1),
        "axes": {
            "y_rnapii_block_prob": {
                "probabilities": block_probs,
                "range": {
                    "lo": args.block_prob_lo,
                    "hi": args.block_prob_hi,
                    "step": args.block_prob_step,
                },
                "baseline_block_prob": baseline_block,
                "baseline_injection": inject_note,
                "applies_to": list(TRIO_KEYS),
                "base_values": base_blocks,
                "fixed": {
                    "rnapii_pre_initiation_block_prob": base_cfg["lef"]
                    .get("topology_kwargs", {})
                    .get("rnapii_pre_initiation_block_prob")
                },
                "seconds_per_step": args.seconds_per_step,
                "dwell_seconds_formula": "p/(1-p)*seconds_per_step (effective RNAPII dwell; p=1 -> inf)",
                "dwell_seconds": {
                    f"{p:g}": block_dwell_seconds(p, args.seconds_per_step)
                    for p in block_probs
                },
            },
            "x_ctcf_lifetime": {
                "multipliers": ctcf_mults,
                "applies_to": ["lifetime_ctcf"],
                "base_value": base_ctcf,
            },
        },
        "block_prob_policy": "absolute probability assigned jointly to the three active RNAPII barrier states; pre-initiation left at baseline",
        "ctcf_lifetime_policy": "lifetime_ctcf = round(base_lifetime_ctcf * multiplier) in int steps",
        "rnap_policy": "baseline RNAP settings left unchanged (NOT forced off) so the block-prob axis is meaningful",
        "normalization": "mean raw per-chain metric in grid config divided by mean raw per-chain metric in config1",
        "median_normalization": "median raw per-chain metric in grid config divided by median raw per-chain metric in config1 (blockprob_vs_ctcf_lifetime_heatmaps_median.svg); spread annotation is the comparison-chain median absolute deviation in baseline-median units",
        "statistical_test": STAT_TEST_NAME,
        "metric_order": METRIC_ORDER,
        "generated": {
            label: {key: str(value) for key, value in paths.items()}
            for label, paths in generated_paths.items()
        },
    }
    (out_dir / "method.json").write_text(json.dumps(method, indent=2))

    print(f"wrote {svg_path}")
    print(f"wrote {median_svg_path}")
    print(f"wrote {out_dir / 'sweep_summary.tsv'}")
    print(f"wrote {out_dir / 'sweep_per_chain_metrics.tsv'}")
    print(f"wrote {out_dir / 'sweep_stats.tsv'}")
    print(f"wrote {out_dir / 'generated_configs.tsv'}")
    print(f"wrote {out_dir / 'method.json'}")
    print()
    print(
        summary[
            ["config", "rnapii_block_prob", "ctcf_lifetime_multiplier", "metric", "fold_vs_config1_mean"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
