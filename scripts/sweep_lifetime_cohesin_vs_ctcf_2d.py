#!/usr/bin/env python
"""2D cohesin-lifetime x CTCF-lifetime sweep at the 1D LEF stage.

Companion to ``sweep_rnapoff_boundary_strength_1d.py``. Instead of sweeping CTCF
boundary strength, this script sweeps two lifetime axes and produces a 2D
heatmap per metric:

* **y axis - cohesin lifetime.** A single multiplier is applied identically to
  ``lef.lifetime``, ``lef.lifetime_stalled`` and
  ``lef.topology_kwargs.lifetime_rnapii_stalled`` (the RNAPII-stalled cohesin
  lifetime). These three move together as "the cohesin lifetime".
* **x axis - ``lef.lifetime_ctcf``** (the WAPL-protected lifetime of cohesin
  captured at a CTCF anchor), swept by its own multiplier.

Each grid cell (cohesin-lifetime multiplier, CTCF-lifetime multiplier) gets its
own derivative config, its 1D trajectory is run/reused, and metrics are reported
as fold changes versus the baseline config1 replicate-chain mean.

Unlike the boundary-strength companion, RNAP is **left as configured in the
baseline** (not forced off) so that the ``lifetime_rnapii_stalled`` axis is
biologically meaningful.

Note: the H5-staleness check does not inspect lifetime fields, so if you change
the baseline config's lifetimes and reuse an out-dir, pass ``--force-1d`` to
regenerate trajectories.
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
from polychrom.pipelines.loop_extrusion.config import load_config  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _base_lifetimes(base_cfg: dict) -> dict:
    """Resolve the baseline values the two axes scale from.

    ``lifetime_ctcf`` and ``lifetime_rnapii_stalled`` both fall back to the base
    ``lifetime`` when unset (mirroring the config/topology defaults), so the
    sweep multiplies an explicit, resolved baseline in every case.
    """
    lef = base_cfg["lef"]
    kwargs = lef.get("topology_kwargs", {}) or {}
    base_lifetime = float(lef.get("lifetime", 200))
    base_stalled = float(lef.get("lifetime_stalled", base_lifetime))
    rnapii = kwargs.get("lifetime_rnapii_stalled")
    base_rnapii = float(rnapii) if rnapii is not None else base_lifetime
    ctcf = lef.get("lifetime_ctcf")
    base_ctcf = float(ctcf) if ctcf is not None else base_lifetime
    return {
        "lifetime": base_lifetime,
        "lifetime_stalled": base_stalled,
        "lifetime_rnapii_stalled": base_rnapii,
        "lifetime_ctcf": base_ctcf,
    }


def make_grid_config(
    base_cfg: dict,
    base_lifetimes: dict,
    life_mult: float,
    ctcf_mult: float,
) -> tuple[dict, dict]:
    """Build one grid-cell config.

    ``life_mult`` scales the three cohesin-lifetime fields together;
    ``ctcf_mult`` scales the CTCF-capture lifetime. RNAP settings are left
    exactly as in the baseline config.
    """
    cfg = copy.deepcopy(base_cfg)
    lef = cfg["lef"]
    kwargs = lef.setdefault("topology_kwargs", {})

    lifetime = int(round(base_lifetimes["lifetime"] * life_mult))
    lifetime_stalled = int(round(base_lifetimes["lifetime_stalled"] * life_mult))
    lifetime_rnapii_stalled = int(round(base_lifetimes["lifetime_rnapii_stalled"] * life_mult))
    lifetime_ctcf = int(round(base_lifetimes["lifetime_ctcf"] * ctcf_mult))

    lef["lifetime"] = lifetime
    lef["lifetime_stalled"] = lifetime_stalled
    kwargs["lifetime_rnapii_stalled"] = lifetime_rnapii_stalled
    lef["lifetime_ctcf"] = lifetime_ctcf

    record = {
        "cohesin_lifetime_multiplier": life_mult,
        "ctcf_lifetime_multiplier": ctcf_mult,
        "lifetime": lifetime,
        "lifetime_stalled": lifetime_stalled,
        "lifetime_rnapii_stalled": lifetime_rnapii_stalled,
        "lifetime_ctcf": lifetime_ctcf,
    }
    return cfg, record


def cell_label(life_mult: float, ctcf_mult: float) -> str:
    return f"life{multiplier_label(life_mult)}_ctcf{multiplier_label(ctcf_mult)}"


def axis_tick_labels(multipliers: list[float], base_value: float) -> list[str]:
    """Tick label = multiplier plus the resulting absolute lifetime (steps)."""
    return [f"{multiplier_label(m)}\n{int(round(base_value * m))}" for m in multipliers]


def plot_grid_heatmaps(
    summary: pd.DataFrame,
    cell_grid: list[list[str]],
    life_ticks: list[str],
    ctcf_ticks: list[str],
    out_path: Path,
    value_col: str = "fold_vs_config1_mean",
    spread_col: str = "fold_vs_config1_sd",
    title: str = "Cohesin-lifetime x CTCF-lifetime sweep: fold change vs config1",
) -> None:
    """One 2D heatmap per metric. x = lifetime_ctcf, y = cohesin lifetime.

    ``cell_grid[i][j]`` is the config label for cohesin-lifetime row ``i`` and
    CTCF-lifetime column ``j``. Rows are drawn with ``origin='lower'`` so the
    cohesin-lifetime multiplier increases upward.
    """
    n_life = len(cell_grid)
    n_ctcf = len(cell_grid[0]) if n_life else 0
    metrics = METRIC_ORDER
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * (0.95 * n_ctcf + 3.4), nrows * (0.62 * n_life + 1.7)),
        squeeze=False,
    )

    for m_idx, metric in enumerate(metrics):
        r, c = divmod(m_idx, ncols)
        ax = axes[r][c]
        msum = summary[summary["metric"] == metric].set_index("config")

        values = np.full((n_life, n_ctcf), np.nan)
        spreads = np.full((n_life, n_ctcf), np.nan)
        for i in range(n_life):
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
        ax.set_xlabel("lifetime_ctcf (multiplier / steps)", fontsize=9)
        ax.set_ylabel("cohesin lifetime (multiplier / steps)", fontsize=9)
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
        description="Generate a 2D cohesin-lifetime x CTCF-lifetime sweep, run 1D, and plot fold heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config1", type=Path, required=True, help="baseline config1 YAML")
    parser.add_argument("--h5-config1", type=Path, required=True, help="existing baseline LEFPositions.h5")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--cohesin-lifetime-multipliers",
        type=parse_multipliers,
        default=parse_multipliers("1,2,4,8"),
        help="y-axis multipliers applied jointly to lifetime, lifetime_stalled, lifetime_rnapii_stalled",
    )
    parser.add_argument(
        "--ctcf-lifetime-multipliers",
        type=parse_multipliers,
        default=parse_multipliers("1,2,4,8"),
        help="x-axis multipliers applied to lifetime_ctcf",
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
    base_lifetimes = _base_lifetimes(base_cfg)
    validate_baseline_h5(args.config1, args.h5_config1, args.label1)

    print(
        "[base] cohesin lifetimes (lifetime/lifetime_stalled/lifetime_rnapii_stalled) = "
        f"{int(base_lifetimes['lifetime'])}/{int(base_lifetimes['lifetime_stalled'])}/"
        f"{int(base_lifetimes['lifetime_rnapii_stalled'])}; "
        f"lifetime_ctcf = {int(base_lifetimes['lifetime_ctcf'])}"
    )

    baseline_core, baseline_boundary, _, _ = analyze_run(
        args.label1,
        args.config1,
        args.h5_config1,
        base_topology,
    )
    baseline_rows = selected_metric_rows(baseline_core, baseline_boundary)

    life_mults = args.cohesin_lifetime_multipliers
    ctcf_mults = args.ctcf_lifetime_multipliers

    generated_config_rows: list[dict] = []
    all_metric_rows = [baseline_rows]
    grid_labels: list[str] = []
    # cell_grid[i][j] -> config label (i indexes cohesin lifetime, j indexes ctcf)
    cell_grid: list[list[str]] = [["" for _ in ctcf_mults] for _ in life_mults]
    cell_meta: dict[str, dict] = {}
    generated_paths: dict[str, dict] = {}
    tasks: list[dict] = []

    # 1) generate every grid config (cheap; always sequential) ------------
    for i, life_mult in enumerate(life_mults):
        for j, ctcf_mult in enumerate(ctcf_mults):
            label = cell_label(life_mult, ctcf_mult)
            grid_labels.append(label)
            cell_grid[i][j] = label
            cell_meta[label] = {
                "cohesin_lifetime_multiplier": life_mult,
                "ctcf_lifetime_multiplier": ctcf_mult,
            }

            cfg, record = make_grid_config(base_cfg, base_lifetimes, life_mult, ctcf_mult)
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

    # Attach the two axis multipliers to the per-cell summary/stats for clarity.
    for frame in (summary, stats):
        frame["cohesin_lifetime_multiplier"] = frame["config"].map(
            lambda c: cell_meta.get(c, {}).get("cohesin_lifetime_multiplier")
        )
        frame["ctcf_lifetime_multiplier"] = frame["config"].map(
            lambda c: cell_meta.get(c, {}).get("ctcf_lifetime_multiplier")
        )

    per_chain.to_csv(out_dir / "sweep_per_chain_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "sweep_summary.tsv", sep="\t", index=False)
    stats.to_csv(out_dir / "sweep_stats.tsv", sep="\t", index=False)
    pd.DataFrame(generated_config_rows).to_csv(out_dir / "generated_configs.tsv", sep="\t", index=False)

    life_ticks = axis_tick_labels(life_mults, base_lifetimes["lifetime"])
    ctcf_ticks = axis_tick_labels(ctcf_mults, base_lifetimes["lifetime_ctcf"])

    svg_path = out_dir / "lifetime_cohesin_vs_ctcf_heatmaps.svg"
    plot_grid_heatmaps(summary, cell_grid, life_ticks, ctcf_ticks, svg_path)

    median_svg_path = out_dir / "lifetime_cohesin_vs_ctcf_heatmaps_median.svg"
    plot_grid_heatmaps(
        summary,
        cell_grid,
        life_ticks,
        ctcf_ticks,
        median_svg_path,
        value_col="fold_vs_config1_median",
        spread_col="fold_vs_config1_median_mad",
        title="Cohesin-lifetime x CTCF-lifetime sweep: median fold change vs config1",
    )

    method = {
        "baseline_label": args.label1,
        "baseline_config": str(args.config1),
        "baseline_h5": str(args.h5_config1),
        "axes": {
            "y_cohesin_lifetime": {
                "multipliers": life_mults,
                "applies_to": ["lifetime", "lifetime_stalled", "lifetime_rnapii_stalled"],
                "base_values": {
                    "lifetime": base_lifetimes["lifetime"],
                    "lifetime_stalled": base_lifetimes["lifetime_stalled"],
                    "lifetime_rnapii_stalled": base_lifetimes["lifetime_rnapii_stalled"],
                },
            },
            "x_ctcf_lifetime": {
                "multipliers": ctcf_mults,
                "applies_to": ["lifetime_ctcf"],
                "base_value": base_lifetimes["lifetime_ctcf"],
            },
        },
        "lifetime_policy": "each axis multiplies its resolved baseline lifetime(s); values rounded to int steps",
        "rnap_policy": "baseline RNAP settings left unchanged (NOT forced off) so lifetime_rnapii_stalled is meaningful",
        "normalization": "mean raw per-chain metric in grid config divided by mean raw per-chain metric in config1",
        "median_normalization": "median raw per-chain metric in grid config divided by median raw per-chain metric in config1 (lifetime_cohesin_vs_ctcf_heatmaps_median.svg); spread annotation is the comparison-chain median absolute deviation in baseline-median units",
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
            ["config", "cohesin_lifetime_multiplier", "ctcf_lifetime_multiplier", "metric", "fold_vs_config1_mean"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
