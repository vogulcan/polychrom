#!/usr/bin/env python
"""RNAP-off boundary-strength sweep at the 1D LEF stage.

Given a baseline config1 YAML and its existing ``LEFPositions.h5``, this script
generates RNAP-off derivative configs with explicit CTCF boundary strengths
multiplied by requested factors, runs/reuses their 1D trajectories, and reports
fold changes versus the baseline replicate-chain mean.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
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
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


METRIC_ORDER = [
    "cohesin_gene_body",
    "cohesin_ctcf_boundary",
    "mean_loop_length",
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
        return f"x{int(multiplier)}"
    label = f"x{multiplier:g}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "p", label)


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


def make_sweep_config(base_cfg: dict, multiplier: float) -> tuple[dict, list[dict]]:
    cfg = copy.deepcopy(base_cfg)
    lef = cfg["lef"]
    kwargs = lef["topology_kwargs"]
    source_strengths = require_boundary_strength(base_cfg)

    lef["max_rnapii"] = 0
    plugins = lef.setdefault("plugins", {})
    plugins["rnapii_load"] = None
    plugins["rnapii_translocate"] = None

    new_strengths = {}
    records = []
    for key, value in source_strengths.items():
        original = float(value)
        scaled = min(original * multiplier, 1.0)
        new_strengths[key] = round(scaled, 6)
        records.append(
            {
                "boundary": key,
                "original_boundary_strength": original,
                "multiplier": multiplier,
                "scaled_boundary_strength": scaled,
                "capped": bool(scaled < original * multiplier),
            }
        )
    kwargs["boundary_strength"] = new_strengths
    return cfg, records


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
    core_keep = {"cohesin_gene_body", "cohesin_ctcf_boundary", "mean_loop_length"}
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
            p_value = _two_sample_permutation_pvalue(base_values, comp_values)

            summary_rows.append(
                {
                    "config": label,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric].replace("\n", " "),
                    "baseline_mean_raw": baseline_mean,
                    "comparison_mean_raw": comp_mean,
                    "fold_vs_config1_mean": fold,
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


def plot_heatmaps(summary: pd.DataFrame, labels: list[str], out_path: Path) -> None:
    nrows = len(METRIC_ORDER)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(max(7.0, 1.25 * len(labels) + 3.2), 0.62 * nrows + 1.4),
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
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#f1f3f5")
        image = ax.imshow(values, aspect="auto", cmap=cmap, norm=_norm_for_row(values.ravel()))
        for col_idx, value in enumerate(values.ravel()):
            text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col_idx, 0, text, ha="center", va="center", fontsize=8, color="#111111")

        ax.set_yticks([0])
        ax.set_yticklabels([METRIC_LABELS[metric].replace("\n", " ")], fontsize=9)
        ax.set_xticks(range(len(labels)))
        if row_idx == nrows - 1:
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_xlabel("Boundary-strength multiplier, RNAP off", fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01)
        cbar.ax.tick_params(labelsize=7, length=2)

    fig.suptitle("RNAP-off boundary-strength sweep: fold change vs config1", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)

    out_dir = args.out_dir
    configs_dir = out_dir / "configs"
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = read_yaml(args.config1)
    require_boundary_strength(base_cfg)
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
    generated_paths = {}

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
        generated_config_rows.append(
            {
                "config": label,
                "multiplier": multiplier,
                "config_path": str(cfg_path),
                "h5_path": str(h5_path),
                "max_rnapii": 0,
                "rnapii_load": "null",
                "rnapii_translocate": "null",
                "default_boundary_strength": cfg["lef"]["topology_kwargs"].get("default_boundary_strength"),
            }
        )

        _validate_compatible(base_topology, _topology(cfg))
        resolved_h5 = _resolve_h5(
            cfg_path=cfg_path,
            requested_h5=h5_path,
            out_dir=out_dir,
            label=label,
            force_1d=args.force_1d,
        )
        core_rows, boundary_metric_rows, _, _ = analyze_run(label, cfg_path, resolved_h5, base_topology)
        all_metric_rows.append(selected_metric_rows(core_rows, boundary_metric_rows))

    per_chain = pd.concat(all_metric_rows, ignore_index=True)
    summary, stats = summarize_sweep(per_chain, args.label1, sweep_labels)

    per_chain.to_csv(out_dir / "sweep_per_chain_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "sweep_summary.tsv", sep="\t", index=False)
    stats.to_csv(out_dir / "sweep_stats.tsv", sep="\t", index=False)
    pd.DataFrame(generated_config_rows).to_csv(out_dir / "generated_configs.tsv", sep="\t", index=False)
    pd.DataFrame(boundary_rows).to_csv(out_dir / "boundary_strength_values.tsv", sep="\t", index=False)

    svg_path = out_dir / "boundary_strength_sweep_heatmaps.svg"
    plot_heatmaps(summary, sweep_labels, svg_path)

    method = {
        "baseline_label": args.label1,
        "baseline_config": str(args.config1),
        "baseline_h5": str(args.h5_config1),
        "multipliers": args.multipliers,
        "boundary_strength_policy": "explicit boundary_strength entries multiplied and capped at 1.0; default_boundary_strength unchanged",
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
