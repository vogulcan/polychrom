#!/usr/bin/env python
"""2D cohesin-separation x CTCF-lifetime sweep at the 1D LEF stage.

Companion to ``sweep_lifetime_cohesin_vs_ctcf_2d.py`` and
``sweep_blockprob_vs_ctcf_lifetime_2d.py``. Both axes are multipliers and produce
a 2D heatmap per metric:

* **y axis - cohesin separation (density).** ``lef.separation`` is the average
  number of lattice sites between LEFs; the number of cohesins is the derived
  property ``num_lefs = num_sites // separation``. Scaling ``separation`` by a
  multiplier therefore scales the cohesin *density* inversely (a larger
  separation multiplier means fewer cohesins / lower density).
* **x axis - ``lef.lifetime_ctcf``** (the WAPL-protected lifetime of cohesin
  captured at a CTCF anchor), swept by its own multiplier.

Each grid cell (separation multiplier, CTCF-lifetime multiplier) gets its own
derivative config, its 1D trajectory is run/reused, and metrics are reported as
fold changes versus the baseline config1 replicate-chain mean.

RNAP is left exactly as configured in the baseline (not forced off).

Note: ``separation`` *is* part of the H5-staleness check (unlike the lifetime /
block-prob axes), so a changed separation is detected and reruns automatically.
A fresh ``--out-dir`` is correct; reusing a populated out-dir is also safe here.
``lifetime_ctcf`` is not in the staleness check, so if you reuse an out-dir after
changing only that axis pass ``--force-1d``.
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


def _base_separation(base_cfg: dict) -> float:
    """Baseline ``lef.separation`` (sites between LEFs) the y axis scales from."""
    return float(base_cfg["lef"].get("separation", 800))


def _base_ctcf_lifetime(base_cfg: dict) -> float:
    """``lifetime_ctcf`` falls back to the base ``lifetime`` when unset."""
    lef = base_cfg["lef"]
    ctcf = lef.get("lifetime_ctcf")
    return float(ctcf) if ctcf is not None else float(lef.get("lifetime", 200))


def _num_lefs(cfg_path: Path) -> int:
    """Derived cohesin count (num_sites // separation) for a written config."""
    return int(load_config(cfg_path).lef.num_lefs)


def make_grid_config(
    base_cfg: dict,
    base_separation: float,
    base_ctcf: float,
    sep_mult: float,
    ctcf_mult: float,
) -> tuple[dict, dict]:
    """Build one grid-cell config.

    ``sep_mult`` scales ``lef.separation`` (and hence inversely the cohesin
    density, since ``num_lefs`` is derived); ``ctcf_mult`` scales the CTCF-capture
    lifetime. All other settings (including RNAP) are left as in the baseline.
    """
    cfg = copy.deepcopy(base_cfg)
    lef = cfg["lef"]

    separation = max(1, int(round(base_separation * sep_mult)))
    lifetime_ctcf = int(round(base_ctcf * ctcf_mult))

    lef["separation"] = separation
    lef["lifetime_ctcf"] = lifetime_ctcf

    record = {
        "separation_multiplier": sep_mult,
        "ctcf_lifetime_multiplier": ctcf_mult,
        "separation": separation,
        "lifetime_ctcf": lifetime_ctcf,
    }
    return cfg, record


def cell_label(sep_mult: float, ctcf_mult: float) -> str:
    return f"sep{multiplier_label(sep_mult)}_ctcf{multiplier_label(ctcf_mult)}"


def axis_tick_labels(multipliers: list[float], base_value: float) -> list[str]:
    """Tick label = multiplier plus the resulting absolute value."""
    return [f"{multiplier_label(m)}\n{int(round(base_value * m))}" for m in multipliers]


def plot_grid_heatmaps(
    summary: pd.DataFrame,
    cell_grid: list[list[str]],
    sep_ticks: list[str],
    ctcf_ticks: list[str],
    out_path: Path,
    value_col: str = "fold_vs_config1_mean",
    spread_col: str = "fold_vs_config1_sd",
    title: str = "Cohesin-separation x CTCF-lifetime sweep: fold change vs config1",
) -> None:
    """One 2D heatmap per metric. x = lifetime_ctcf, y = cohesin separation.

    ``cell_grid[i][j]`` is the config label for separation row ``i`` and
    CTCF-lifetime column ``j``. Rows are drawn with ``origin='lower'`` so the
    separation multiplier increases upward (i.e. cohesin density decreases
    upward).
    """
    n_sep = len(cell_grid)
    n_ctcf = len(cell_grid[0]) if n_sep else 0
    metrics = METRIC_ORDER
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * (0.95 * n_ctcf + 3.4), nrows * (0.62 * n_sep + 1.7)),
        squeeze=False,
    )

    for m_idx, metric in enumerate(metrics):
        r, c = divmod(m_idx, ncols)
        ax = axes[r][c]
        msum = summary[summary["metric"] == metric].set_index("config")

        values = np.full((n_sep, n_ctcf), np.nan)
        spreads = np.full((n_sep, n_ctcf), np.nan)
        for i in range(n_sep):
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
        for i in range(n_sep):
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
        ax.set_yticks(range(n_sep))
        ax.set_yticklabels(sep_ticks, fontsize=9)
        ax.set_xlabel("lifetime_ctcf (multiplier / steps)", fontsize=9)
        ax.set_ylabel("cohesin separation (multiplier / sites; ↑sep = ↓density)", fontsize=9)
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
        description="Generate a 2D cohesin-separation x CTCF-lifetime sweep, run 1D, and plot fold heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config1", type=Path, required=True, help="baseline config1 YAML")
    parser.add_argument("--h5-config1", type=Path, required=True, help="existing baseline LEFPositions.h5")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--separation-multipliers",
        type=parse_multipliers,
        default=parse_multipliers("0.125,0.25,0.5,1,2,4,6"),
        help="y-axis multipliers applied to lef.separation (inverse cohesin density)",
    )
    parser.add_argument(
        "--ctcf-lifetime-multipliers",
        type=parse_multipliers,
        default=parse_multipliers("0.125,0.25,0.5,1,2,4,6"),
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
    base_separation = _base_separation(base_cfg)
    base_ctcf = _base_ctcf_lifetime(base_cfg)
    base_num_lefs = _num_lefs(args.config1)
    validate_baseline_h5(args.config1, args.h5_config1, args.label1)

    print(
        f"[base] separation = {int(base_separation)} sites "
        f"(num_lefs = {base_num_lefs}); lifetime_ctcf = {int(base_ctcf)}"
    )

    baseline_core, baseline_boundary, _, _ = analyze_run(
        args.label1,
        args.config1,
        args.h5_config1,
        base_topology,
    )
    baseline_rows = selected_metric_rows(baseline_core, baseline_boundary)

    sep_mults = args.separation_multipliers
    ctcf_mults = args.ctcf_lifetime_multipliers

    generated_config_rows: list[dict] = []
    all_metric_rows = [baseline_rows]
    grid_labels: list[str] = []
    # cell_grid[i][j] -> config label (i indexes separation, j indexes ctcf)
    cell_grid: list[list[str]] = [["" for _ in ctcf_mults] for _ in sep_mults]
    cell_meta: dict[str, dict] = {}
    generated_paths: dict[str, dict] = {}
    tasks: list[dict] = []

    # 1) generate every grid config (cheap; always sequential) ------------
    for i, sep_mult in enumerate(sep_mults):
        for j, ctcf_mult in enumerate(ctcf_mults):
            label = cell_label(sep_mult, ctcf_mult)
            grid_labels.append(label)
            cell_grid[i][j] = label
            cell_meta[label] = {
                "separation_multiplier": sep_mult,
                "ctcf_lifetime_multiplier": ctcf_mult,
            }

            cfg, record = make_grid_config(base_cfg, base_separation, base_ctcf, sep_mult, ctcf_mult)
            cfg_path = configs_dir / f"{label}.yaml"
            h5_path = runs_dir / label / "LEFPositions.h5"
            write_yaml(cfg_path, cfg)
            generated_paths[label] = {"config": cfg_path, "h5": h5_path}

            record["num_lefs"] = _num_lefs(cfg_path)
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
        frame["separation_multiplier"] = frame["config"].map(
            lambda c: cell_meta.get(c, {}).get("separation_multiplier")
        )
        frame["ctcf_lifetime_multiplier"] = frame["config"].map(
            lambda c: cell_meta.get(c, {}).get("ctcf_lifetime_multiplier")
        )

    per_chain.to_csv(out_dir / "sweep_per_chain_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "sweep_summary.tsv", sep="\t", index=False)
    stats.to_csv(out_dir / "sweep_stats.tsv", sep="\t", index=False)
    pd.DataFrame(generated_config_rows).to_csv(out_dir / "generated_configs.tsv", sep="\t", index=False)

    sep_ticks = axis_tick_labels(sep_mults, base_separation)
    ctcf_ticks = axis_tick_labels(ctcf_mults, base_ctcf)

    svg_path = out_dir / "separation_vs_ctcf_lifetime_heatmaps.svg"
    plot_grid_heatmaps(summary, cell_grid, sep_ticks, ctcf_ticks, svg_path)

    median_svg_path = out_dir / "separation_vs_ctcf_lifetime_heatmaps_median.svg"
    plot_grid_heatmaps(
        summary,
        cell_grid,
        sep_ticks,
        ctcf_ticks,
        median_svg_path,
        value_col="fold_vs_config1_median",
        spread_col="fold_vs_config1_median_mad",
        title="Cohesin-separation x CTCF-lifetime sweep: median fold change vs config1",
    )

    method = {
        "baseline_label": args.label1,
        "baseline_config": str(args.config1),
        "baseline_h5": str(args.h5_config1),
        "axes": {
            "y_cohesin_separation": {
                "multipliers": sep_mults,
                "applies_to": ["separation"],
                "base_value": base_separation,
                "base_num_lefs": base_num_lefs,
                "note": "num_lefs = num_sites // separation; larger separation multiplier => fewer cohesins (lower density)",
            },
            "x_ctcf_lifetime": {
                "multipliers": ctcf_mults,
                "applies_to": ["lifetime_ctcf"],
                "base_value": base_ctcf,
            },
        },
        "separation_policy": "separation = round(base_separation * multiplier), floored at 1; num_lefs derived",
        "ctcf_lifetime_policy": "lifetime_ctcf = round(base_lifetime_ctcf * multiplier) in int steps",
        "rnap_policy": "baseline RNAP settings left unchanged (NOT forced off)",
        "normalization": "mean raw per-chain metric in grid config divided by mean raw per-chain metric in config1",
        "median_normalization": "median raw per-chain metric in grid config divided by median raw per-chain metric in config1 (separation_vs_ctcf_lifetime_heatmaps_median.svg); spread annotation is the comparison-chain median absolute deviation in baseline-median units",
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
            ["config", "separation_multiplier", "ctcf_lifetime_multiplier", "metric", "fold_vs_config1_mean"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
