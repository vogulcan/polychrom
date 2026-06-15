#!/usr/bin/env python
"""RNAP-off boundary-strength sweep at the 1D LEF stage.

Given a baseline config1 YAML and its existing ``LEFPositions.h5``, this script
generates RNAP-off derivative configs with explicit CTCF boundary strengths
multiplied by requested factors, runs/reuses their 1D trajectories, and reports
fold changes versus the baseline replicate-chain mean.
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported so that running many 1D
# sims under --jobs does not oversubscribe cores (each sweep point is numpy-light;
# parallelism comes from running whole sims concurrently, not threaded BLAS).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import logging
import multiprocessing as mp
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_config_chain_metrics import (  # noqa: E402
    BOUNDARY_LABELS,
    BOUNDARY_ORDER,
    CORE_LABELS,
    STAT_TEST_NAME,
    _format_h5_info,
    _h5_info,
    _h5_mismatch_reasons,
    _resolve_h5,
    _safe_label,
    _significance,
    _topology,
    _two_sample_permutation_pvalue,
    _validate_compatible,
    analyze_run,
)
from polychrom.pipelines.loop_extrusion.config import load_config  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# Diverging fold-change colormap: slate-blue (below baseline) -> white (fold=1.0,
# the TwoSlopeNorm center) -> brick-red (above baseline).
FOLD_CMAP = LinearSegmentedColormap.from_list(
    "fold_slate_red", ["#465775", "#ffffff", "#A63446"]
)


METRIC_ORDER = [
    "cohesin_gene_body",
    "cohesin_ctcf_boundary",
    "mean_loop_length",
    "corner_dot",
    "boundary_pass_event",
    "cross_tad_occupancy",
    "crossed_boundaries",
    "multi_tad_crossing",
    "one_ctcf_leg_stripe",
    "boundary_crossing_stripe",
]

METRIC_LABELS = {
    "cohesin_gene_body": CORE_LABELS["cohesin_gene_body"],
    "cohesin_ctcf_boundary": CORE_LABELS["cohesin_ctcf_boundary"],
    "mean_loop_length": CORE_LABELS["mean_loop_length"],
    "corner_dot": CORE_LABELS["corner_dot"],
    **BOUNDARY_LABELS,
}


def read_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def parse_multipliers(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("multipliers must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one multiplier is required")
    return values


def multiplier_label(multiplier: float) -> str:
    if float(multiplier).is_integer():
        return f"{int(multiplier)}x"
    label = f"{multiplier:g}x"
    return re.sub(r"[^A-Za-z0-9_.-]+", "p", label)


def boundary_strength_summary(records: list[dict]) -> dict:
    values = np.array([float(record["scaled_boundary_strength"]) for record in records], dtype=float)
    mean = float(values.mean()) if values.size else float("nan")
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {
        "mean_boundary_strength": mean,
        "sd_boundary_strength": sd,
        "mean_boundary_strength_pct": mean * 100.0,
        "sd_boundary_strength_pct": sd * 100.0,
    }


def heatmap_tick_label(label: str, strength_summary: dict) -> str:
    mean = strength_summary["mean_boundary_strength_pct"]
    sd = strength_summary["sd_boundary_strength_pct"]
    if not np.isfinite(mean) or not np.isfinite(sd):
        return f"{label}\nNA"
    return f"{label}\n{mean:.1f} ± {sd:.1f}%"


def _uses_tads(base_cfg: dict) -> bool:
    return bool(base_cfg.get("lef", {}).get("topology_kwargs", {}).get("tads"))


def require_boundary_strength(base_cfg: dict) -> dict:
    try:
        boundary_strength = base_cfg["lef"]["topology_kwargs"]["boundary_strength"]
    except KeyError as exc:
        raise ValueError(
            "config1 must contain lef.topology_kwargs.boundary_strength for this sweep"
        ) from exc
    if not isinstance(boundary_strength, dict) or not boundary_strength:
        raise ValueError(
            "lef.topology_kwargs.boundary_strength must be a non-empty mapping"
        )
    return boundary_strength


def require_sweepable_strengths(base_cfg: dict) -> None:
    """Accept either the per-TAD ``tads`` schema or a legacy ``boundary_strength``."""
    if _uses_tads(base_cfg):
        return
    require_boundary_strength(base_cfg)


def _scale_strength(original: float, multiplier: float) -> float:
    return min(original * multiplier, 1.0)


def _record(boundary, original: float, multiplier: float, scaled: float) -> dict:
    return {
        "boundary": boundary,
        "original_boundary_strength": original,
        "multiplier": multiplier,
        "scaled_boundary_strength": scaled,
        "capped": bool(scaled < original * multiplier),
    }


def make_sweep_config(base_cfg: dict, multiplier: float) -> tuple[dict, list[dict]]:
    cfg = copy.deepcopy(base_cfg)
    lef = cfg["lef"]
    kwargs = lef["topology_kwargs"]

    lef["max_rnapii"] = 0
    plugins = lef.setdefault("plugins", {})
    plugins["rnapii_load"] = None
    plugins["rnapii_translocate"] = None

    records: list[dict] = []
    if _uses_tads(base_cfg):
        # Per-TAD schema: scale BOTH oriented anchors of every TAD independently.
        default = float(kwargs.get("default_boundary_strength", 0.5))
        new_tads = []
        for i, tad in enumerate(kwargs["tads"]):
            new_tad = dict(tad)
            for side in ("left_strength", "right_strength"):
                original = float(tad.get(side, default))
                scaled = _scale_strength(original, multiplier)
                new_tad[side] = round(scaled, 6)
                records.append(_record(f"tad{i}:{side}", original, multiplier, scaled))
            new_tads.append(new_tad)
        kwargs["tads"] = new_tads
    else:
        source_strengths = require_boundary_strength(base_cfg)
        new_strengths = {}
        for key, value in source_strengths.items():
            original = float(value)
            scaled = _scale_strength(original, multiplier)
            new_strengths[key] = round(scaled, 6)
            records.append(_record(key, original, multiplier, scaled))
        kwargs["boundary_strength"] = new_strengths
    return cfg, records


def _run_sweep_point(task: dict) -> dict:
    """Worker: run (or reuse) the 1D LEF stage for one sweep multiplier and return
    its resolved H5 path. Top-level + picklable so it can be dispatched to a
    process pool. Each process re-seeds numpy from the config seed inside
    ``lef.run``, so results are deterministic and isolated from sibling workers."""
    resolved = _resolve_h5(
        cfg_path=Path(task["cfg_path"]),
        requested_h5=Path(task["h5_path"]),
        out_dir=Path(task["out_dir"]),
        label=task["label"],
        force_1d=task["force_1d"],
    )
    return {"label": task["label"], "h5": str(resolved)}


def validate_baseline_h5(config1: Path, h5_config1: Path, label: str) -> None:
    if not h5_config1.exists():
        raise FileNotFoundError(
            f"Baseline H5 does not exist and will not be generated automatically: {h5_config1}"
        )
    cfg = load_config(config1)
    info = _h5_info(h5_config1)
    reasons = _h5_mismatch_reasons(info, cfg.lef)
    if reasons:
        joined = "\n".join(f"- {reason}" for reason in reasons)
        raise RuntimeError(
            f"Baseline H5 is incompatible with {config1} for {label}: {h5_config1}\n"
            f"{_format_h5_info(info)}\n{joined}"
        )
    print(f"[baseline] {label}: {h5_config1}")
    print(f"           {_format_h5_info(info)}")


def rows_to_metric_frame(rows: list[dict], metrics: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df[df["metric"].isin(metrics)].copy()


def selected_metric_rows(core_rows: list[dict], boundary_rows: list[dict]) -> pd.DataFrame:
    core_keep = {"cohesin_gene_body", "cohesin_ctcf_boundary", "mean_loop_length", "corner_dot"}
    rows = [row for row in core_rows if row["metric"] in core_keep]
    rows.extend(row for row in boundary_rows if row["metric"] in BOUNDARY_ORDER)
    return rows_to_metric_frame(rows, METRIC_ORDER)


def summarize_sweep(
    per_chain: pd.DataFrame,
    label1: str,
    sweep_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = per_chain[per_chain["config"] == label1]
    baseline_means = baseline.groupby("metric")["raw_value"].mean().to_dict()

    summary_rows = []
    stats_rows = []
    for label in sweep_labels:
        for metric in METRIC_ORDER:
            base_values = baseline[baseline["metric"] == metric]["raw_value"].to_numpy(dtype=float)
            comp_values = per_chain[
                (per_chain["config"] == label) & (per_chain["metric"] == metric)
            ]["raw_value"].to_numpy(dtype=float)
            baseline_mean = float(np.nanmean(base_values)) if base_values.size else float("nan")
            comp_mean = float(np.nanmean(comp_values)) if comp_values.size else float("nan")
            fold = comp_mean / baseline_mean if baseline_mean else float("nan")
            # SD of the per-chain fold (raw_value / baseline_mean). Since the
            # baseline mean is a per-metric constant, this is sd(comp)/baseline_mean,
            # i.e. consistent with the fold mean above.
            comp_sd = (
                float(np.nanstd(comp_values, ddof=1))
                if int(np.isfinite(comp_values).sum()) > 1
                else float("nan")
            )
            fold_sd = comp_sd / baseline_mean if baseline_mean else float("nan")
            p_value = _two_sample_permutation_pvalue(base_values, comp_values)

            summary_rows.append(
                {
                    "config": label,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric].replace("\n", " "),
                    "baseline_mean_raw": baseline_mean,
                    "comparison_mean_raw": comp_mean,
                    "fold_vs_config1_mean": fold,
                    "fold_vs_config1_sd": fold_sd,
                    "n_config1": int(np.isfinite(base_values).sum()),
                    "n_comparison": int(np.isfinite(comp_values).sum()),
                }
            )
            stats_rows.append(
                {
                    "config": label,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric].replace("\n", " "),
                    "test": STAT_TEST_NAME,
                    "baseline_mean_raw": baseline_mean,
                    "comparison_mean_raw": comp_mean,
                    "mean_difference_raw": comp_mean - baseline_mean,
                    "mean_fold_vs_baseline_mean": fold,
                    "p_value": p_value,
                    "significance": _significance(p_value),
                    "n_config1": int(np.isfinite(base_values).sum()),
                    "n_comparison": int(np.isfinite(comp_values).sum()),
                }
            )

    per_chain["baseline_config1_mean"] = per_chain["metric"].map(baseline_means)
    per_chain["fold_vs_config1_mean"] = (
        per_chain["raw_value"] / per_chain["baseline_config1_mean"]
    )
    return pd.DataFrame(summary_rows), pd.DataFrame(stats_rows)


def _norm_for_row(values: np.ndarray) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return TwoSlopeNorm(vmin=0.99, vcenter=1.0, vmax=1.01)
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    spread = max(abs(low - 1.0), abs(high - 1.0), 0.05)
    return TwoSlopeNorm(vmin=1.0 - spread, vcenter=1.0, vmax=1.0 + spread)


def plot_heatmaps(
    summary: pd.DataFrame,
    labels: list[str],
    display_labels: list[str],
    out_path: Path,
) -> None:
    nrows = len(METRIC_ORDER)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(max(9.0, 1.25 * len(labels) + 4.8), 0.62 * nrows + 1.4),
        squeeze=False,
    )

    for row_idx, metric in enumerate(METRIC_ORDER):
        ax = axes[row_idx, 0]
        metric_summary = (
            summary[summary["metric"] == metric]
            .set_index("config")
            .reindex(labels)
        )
        values = metric_summary["fold_vs_config1_mean"].to_numpy(dtype=float)[None, :]
        sds = metric_summary["fold_vs_config1_sd"].to_numpy(dtype=float)[None, :]
        cmap = FOLD_CMAP.copy()
        cmap.set_bad("#f1f3f5")
        image = ax.imshow(values, aspect="auto", cmap=cmap, norm=_norm_for_row(values.ravel()))
        for col_idx, (value, sd) in enumerate(zip(values.ravel(), sds.ravel())):
            if not np.isfinite(value):
                text = "NA"
            elif np.isfinite(sd):
                text = f"{value:.2f}\n±{sd:.2f}"
            else:
                text = f"{value:.2f}"
            ax.text(col_idx, 0, text, ha="center", va="center", fontsize=11, color="#111111")

        ax.set_yticks([0])
        ax.set_yticklabels([METRIC_LABELS[metric].replace("\n", " ")], fontsize=12)
        ax.set_xticks(range(len(labels)))
        if row_idx == nrows - 1:
            ax.set_xticklabels(display_labels, fontsize=12)
            ax.set_xlabel("Boundary-strength multiplier, RNAP off (mean ± SD explicit boundary strength)", fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01)
        cbar.ax.tick_params(labelsize=9, length=2)

    fig.suptitle("RNAP-off boundary-strength sweep: fold change vs config1", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" expands the saved canvas to include the now-larger axis
    # tick labels so none are clipped at the figure edge.
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RNAP-off boundary-strength sweep configs, run 1D, and plot fold heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config1", type=Path, required=True, help="baseline config1 YAML")
    parser.add_argument("--h5-config1", type=Path, required=True, help="existing baseline LEFPositions.h5")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--multipliers", type=parse_multipliers, default=parse_multipliers("1,2,3,4,5"))
    parser.add_argument("--force-1d", action="store_true", help="rerun generated sweep H5s")
    parser.add_argument("--label1", default="config1", help="baseline label")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of sweep-point 1D sims to run concurrently (process pool); "
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
    require_sweepable_strengths(base_cfg)
    base_topology = _topology(base_cfg)
    validate_baseline_h5(args.config1, args.h5_config1, args.label1)

    baseline_core, baseline_boundary, _, _ = analyze_run(
        args.label1,
        args.config1,
        args.h5_config1,
        base_topology,
    )
    baseline_rows = selected_metric_rows(baseline_core, baseline_boundary)

    generated_config_rows = []
    boundary_rows = []
    all_metric_rows = [baseline_rows]
    sweep_labels = []
    sweep_display_labels = []
    generated_paths = {}

    # 1) generate every sweep config (cheap; always sequential) -----------
    tasks = []
    for multiplier in args.multipliers:
        label = multiplier_label(multiplier)
        sweep_labels.append(label)
        cfg, strength_records = make_sweep_config(base_cfg, multiplier)
        cfg_path = configs_dir / f"rnapoff_boundary_{label}.yaml"
        h5_path = runs_dir / label / "LEFPositions.h5"
        write_yaml(cfg_path, cfg)
        generated_paths[label] = {"config": cfg_path, "h5": h5_path}

        for record in strength_records:
            boundary_rows.append({"config": label, **record})
        strength_summary = boundary_strength_summary(strength_records)
        sweep_display_labels.append(heatmap_tick_label(label, strength_summary))
        generated_config_rows.append(
            {
                "config": label,
                "multiplier": multiplier,
                **strength_summary,
                "config_path": str(cfg_path),
                "h5_path": str(h5_path),
                "max_rnapii": 0,
                "rnapii_load": "null",
                "rnapii_translocate": "null",
                "default_boundary_strength": cfg["lef"]["topology_kwargs"].get("default_boundary_strength"),
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

    # 2) run each sweep point's 1D LEF stage (sequential or process pool) --
    total = len(tasks)
    jobs = max(1, int(args.jobs))
    resolved_h5: dict[str, Path] = {}
    if jobs == 1:
        for n, task in enumerate(tasks, 1):
            print(f"[{n}/{total}] {task['label']}")
            res = _run_sweep_point(task)
            resolved_h5[res["label"]] = Path(res["h5"])
    else:
        # Process pool: each sweep point is an independent process that re-seeds
        # numpy from the config and writes its own H5 -> identical to sequential,
        # just concurrent. A fork context avoids re-importing this module (and
        # matplotlib) in every worker.
        print(f"running {total} sweep points with --jobs {jobs} (process pool, fork context)")
        ctx = mp.get_context("fork")
        n = 0
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for res in ex.map(_run_sweep_point, tasks):
                n += 1
                print(f"[{n}/{total}] done {res['label']}")
                resolved_h5[res["label"]] = Path(res["h5"])

    # 3) analyze each resolved H5 (in main process, multiplier order) ------
    for task in tasks:
        label = task["label"]
        core_rows, boundary_metric_rows, _, _ = analyze_run(
            label, Path(task["cfg_path"]), resolved_h5[label], base_topology
        )
        all_metric_rows.append(selected_metric_rows(core_rows, boundary_metric_rows))

    per_chain = pd.concat(all_metric_rows, ignore_index=True)
    summary, stats = summarize_sweep(per_chain, args.label1, sweep_labels)

    per_chain.to_csv(out_dir / "sweep_per_chain_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "sweep_summary.tsv", sep="\t", index=False)
    stats.to_csv(out_dir / "sweep_stats.tsv", sep="\t", index=False)
    pd.DataFrame(generated_config_rows).to_csv(out_dir / "generated_configs.tsv", sep="\t", index=False)
    pd.DataFrame(boundary_rows).to_csv(out_dir / "boundary_strength_values.tsv", sep="\t", index=False)

    svg_path = out_dir / "boundary_strength_sweep_heatmaps.svg"
    plot_heatmaps(summary, sweep_labels, sweep_display_labels, svg_path)

    method = {
        "baseline_label": args.label1,
        "baseline_config": str(args.config1),
        "baseline_h5": str(args.h5_config1),
        "multipliers": args.multipliers,
        "boundary_strength_policy": "explicit boundary_strength entries multiplied and capped at 1.0; default_boundary_strength unchanged",
        "x_axis_boundary_strength_label": "multiplier plus mean ± sample SD of explicit capped boundary strengths, shown as percent",
        "rnap_policy": "generated configs set lef.max_rnapii=0 and rnapii_load/rnapii_translocate=null",
        "normalization": "mean raw per-chain metric in generated config divided by mean raw per-chain metric in config1",
        "statistical_test": STAT_TEST_NAME,
        "metric_order": METRIC_ORDER,
        "generated": {
            label: {key: str(value) for key, value in paths.items()}
            for label, paths in generated_paths.items()
        },
    }
    (out_dir / "method.json").write_text(json.dumps(method, indent=2))

    print(f"wrote {svg_path}")
    print(f"wrote {out_dir / 'sweep_summary.tsv'}")
    print(f"wrote {out_dir / 'sweep_per_chain_metrics.tsv'}")
    print(f"wrote {out_dir / 'sweep_stats.tsv'}")
    print(f"wrote {out_dir / 'generated_configs.tsv'}")
    print(f"wrote {out_dir / 'method.json'}")
    print()
    print(summary[["config", "metric", "fold_vs_config1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
