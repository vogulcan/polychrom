#!/usr/bin/env python
"""Compare two 1D LEF runs per replicate chain and plot normalized folds.

The first config/H5 is the baseline. For each metric, every replicate-chain
value is normalized by the mean of baseline replicate chains. The script writes
one SVG with separate metric panels plus TSV summaries.

Example
-------
    micromamba run -n polychrom python scripts/compare_config_chain_metrics.py \
      --config1 configs_test/config1_5k.yaml \
      --config2 configs_test/config2_5k.yaml \
      --out-dir dummy/1d_5k/config2_vs_config1_chain_metrics

If an H5 path is omitted, the script runs/reuses the 1D LEF stage at
``OUT_DIR/LABEL/LEFPositions.h5``.

Statistics
----------
For each metric the script tests config1 chains against config2 chains with a
two-sided exact two-sample permutation test on raw per-chain values. This treats
replicate chains as independent samples and does not assume normality.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import logging
from math import comb
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from polychrom.pipelines.loop_extrusion import lef as lef_stage
from polychrom.pipelines.loop_extrusion.config import load_config
from polychrom.pipelines.loop_extrusion.progress import fmt_hms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CORE_LABELS = {
    "cohesin_gene_body": "Cohesin at\ngene bodies",
    "cohesin_ctcf_boundary": "Cohesin at\nCTCF boundaries",
    "boundary_pass_event": "Boundary pass\nevents",
    "mean_loop_length": "Mean loop\nlength",
    "boundary_crossing_stripe": "Boundary-crossing\nstripe",
}
CORE_ORDER = list(CORE_LABELS)

BOUNDARY_LABELS = {
    "boundary_pass_event": "Boundary pass\nevents",
    "cross_tad_occupancy": "Cross-boundary\noccupancy",
    "crossed_boundaries": "Boundary-counted\ncrossing",
    "multi_tad_crossing": "Multi-TAD\ncrossing",
    "one_ctcf_leg_stripe": "One CTCF-leg\nstripe",
    "boundary_crossing_stripe": "Boundary-crossing\nstripe",
}
BOUNDARY_ORDER = list(BOUNDARY_LABELS)

STAT_TEST_NAME = "two-sided exact two-sample permutation test on raw per-chain values"


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def _read_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "run"


def _topology(cfg: dict) -> dict:
    lef = cfg["lef"]
    kwargs = lef["topology_kwargs"]
    chain = int(lef["chain_length"])
    num_chains = int(lef["num_chains"])
    tad_positions = [int(x) for x in kwargs.get("tad_positions", [])]
    edges = np.array([0, *tad_positions, chain], dtype=int)

    gene_mask = np.zeros(chain, dtype=bool)
    for gene in kwargs.get("genes", []):
        lo, hi = sorted((int(gene["tss"]), int(gene["tes"])))
        lo = max(0, min(chain - 1, lo))
        hi = max(0, min(chain - 1, hi))
        gene_mask[lo : hi + 1] = True

    # Directional convergent TAD topology has internal anchors flanking a
    # boundary p: right-facing anchor at p - 1, left-facing anchor at p.
    ctcf_mask = np.zeros(chain, dtype=bool)
    for pos in tad_positions:
        if 0 <= pos - 1 < chain:
            ctcf_mask[pos - 1] = True
        if 0 <= pos < chain:
            ctcf_mask[pos] = True

    return {
        "chain": chain,
        "num_chains": num_chains,
        "edges": edges,
        "tad_positions": np.array(tad_positions, dtype=np.int32),
        "gene_mask": gene_mask,
        "ctcf_mask": ctcf_mask,
        "gene_sites": int(gene_mask.sum()),
        "ctcf_sites": int(ctcf_mask.sum()),
        "boundaries": len(tad_positions),
        "tads": len(edges) - 1,
    }


def _validate_compatible(top1: dict, top2: dict) -> None:
    for field in ("chain", "num_chains", "boundaries", "tads"):
        if top1[field] != top2[field]:
            raise ValueError(f"Topology mismatch for {field}: {top1[field]} != {top2[field]}")
    if not np.array_equal(top1["edges"], top2["edges"]):
        raise ValueError("TAD edges differ between configs; folds would not be topology-matched")
    if not np.array_equal(top1["gene_mask"], top2["gene_mask"]):
        raise ValueError("Gene-body masks differ between configs; folds would not be topology-matched")
    if not np.array_equal(top1["ctcf_mask"], top2["ctcf_mask"]):
        raise ValueError("CTCF anchor masks differ between configs; folds would not be topology-matched")


def _load_positions(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        if "positions" not in h5:
            raise ValueError(f"{h5_path} has no 'positions' dataset")
        return h5["positions"][:]


def _h5_info(h5_path: Path) -> dict:
    def attr_int(attrs, name: str) -> int | None:
        if name not in attrs:
            return None
        return int(attrs[name])

    try:
        with h5py.File(h5_path, "r") as h5:
            if "positions" not in h5:
                raise ValueError(f"{h5_path} has no 'positions' dataset")
            positions = h5["positions"]
            info = {
                "frames": int(positions.shape[0]),
                "lefs": int(positions.shape[1]),
                "size_mb": h5_path.stat().st_size / (1024.0 * 1024.0),
                "rnapii": "rnapii_positions" in h5,
                "lesions": "lesions" in h5,
                "N": attr_int(h5.attrs, "N"),
                "LEFNum": attr_int(h5.attrs, "LEFNum"),
                "chain_length": attr_int(h5.attrs, "chain_length"),
                "num_chains": attr_int(h5.attrs, "num_chains"),
                "separation": attr_int(h5.attrs, "separation"),
            }
            return info
    except BlockingIOError as exc:
        raise RuntimeError(
            f"H5 file is locked or still being written: {h5_path}\n"
            "Wait for the 1D process to finish, then rerun the comparison."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"H5 file is not readable: {h5_path}\n"
            f"Reason: {exc}"
        ) from exc


def _format_h5_info(info: dict) -> str:
    return (
        f"frames={info['frames']}, lefs/frame={info['lefs']}, "
        f"size={info['size_mb']:.1f} MB, rnapii={'yes' if info['rnapii'] else 'no'}, "
        f"lesions={'yes' if info['lesions'] else 'no'}"
    )


def _h5_mismatch_reasons(info: dict, lef_cfg) -> list[str]:
    checks = [
        ("frames", info["frames"], int(lef_cfg.trajectory_length)),
        ("lefs/frame", info["lefs"], int(lef_cfg.num_lefs)),
        ("N attr", info.get("N"), int(lef_cfg.num_sites)),
        ("LEFNum attr", info.get("LEFNum"), int(lef_cfg.num_lefs)),
        ("chain_length attr", info.get("chain_length"), int(lef_cfg.chain_length)),
        ("num_chains attr", info.get("num_chains"), int(lef_cfg.num_chains)),
        ("separation attr", info.get("separation"), int(lef_cfg.separation)),
    ]
    reasons = []
    for name, observed, expected in checks:
        if observed is not None and int(observed) != expected:
            reasons.append(f"{name}: h5={observed}, config={expected}")
    return reasons


def _run_1d_with_progress(label: str, cfg, h5_path: Path) -> None:
    warmup = max(0, int(cfg.lef.warmup_steps))
    trajectory = int(cfg.lef.trajectory_length)
    total_ticks = warmup + trajectory
    print(
        f"[1d]    {label}: starting LEF simulation "
        f"warmup={warmup} ticks, trajectory={trajectory} ticks, "
        f"total={total_ticks} ticks, N={cfg.lef.num_sites}, LEFs={cfg.lef.num_lefs}"
    )
    print(f"[1d]    {label}: ETA will update in [lef:warmup] and [lef:record] progress lines")
    start = time.time()
    cfg.lef.output_path = str(h5_path)
    lef_stage.run(cfg.lef)
    elapsed = time.time() - start
    print(f"[1d]    {label}: finished in {fmt_hms(elapsed)}")


def _default_h5_path(out_dir: Path, label: str) -> Path:
    return out_dir / _safe_label(label) / "LEFPositions.h5"


def _resolve_h5(
    *,
    cfg_path: Path,
    requested_h5: Path | None,
    out_dir: Path,
    label: str,
    force_1d: bool,
) -> Path:
    h5_path = requested_h5 if requested_h5 is not None else _default_h5_path(out_dir, label)
    h5_path = h5_path.resolve()
    cfg = load_config(cfg_path)
    rerun_stale = False

    if h5_path.exists() and not force_1d:
        info = _h5_info(h5_path)
        mismatch_reasons = _h5_mismatch_reasons(info, cfg.lef)
        if not mismatch_reasons:
            print(f"[reuse] {label}: {h5_path}")
            print(f"        {_format_h5_info(info)}")
            return h5_path
        print(f"[stale] {label}: {h5_path}")
        print(f"        {_format_h5_info(info)}")
        for reason in mismatch_reasons:
            print(f"        mismatch: {reason}")
        print(f"[rerun] {label}: existing H5 does not match config; overwriting {h5_path}")
        rerun_stale = True

    if h5_path.exists() and force_1d:
        print(f"[force] {label}: rerunning 1D and overwriting {h5_path}")
    elif rerun_stale:
        pass
    elif requested_h5 is None:
        print(f"[run]   {label}: no H5 provided; running 1D -> {h5_path}")
    else:
        print(f"[run]   {label}: H5 not found; running 1D -> {h5_path}")

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    _run_1d_with_progress(label, cfg, h5_path)
    info = _h5_info(h5_path)
    mismatch_reasons = _h5_mismatch_reasons(info, cfg.lef)
    if mismatch_reasons:
        raise RuntimeError(
            f"Generated H5 still does not match config for {label}: {h5_path}\n"
            + "\n".join(f"- {reason}" for reason in mismatch_reasons)
        )
    print(f"[done]  {label}: {h5_path}")
    print(f"        {_format_h5_info(info)}")
    return h5_path


def _row(
    *,
    label: str,
    chain_idx: int,
    metric: str,
    metric_label: str,
    raw_value: float,
    raw_count: int,
    denominator: float,
    frames: int,
    lefs: int,
    topo: dict,
) -> dict:
    return {
        "config": label,
        "chain": chain_idx,
        "metric": metric,
        "metric_label": metric_label.replace("\n", " "),
        "raw_value": raw_value,
        "raw_count": raw_count,
        "denominator": denominator,
        "frames": frames,
        "lefs_per_frame": lefs,
        "chain_length": topo["chain"],
        "num_chains": topo["num_chains"],
        "gene_body_sites_per_chain": topo["gene_sites"],
        "ctcf_boundary_sites_per_chain": topo["ctcf_sites"],
        "internal_boundaries_per_chain": topo["boundaries"],
    }


def analyze_run(
    label: str,
    cfg_path: Path,
    h5_path: Path,
    topo: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    del cfg_path
    chain = topo["chain"]
    num_chains = topo["num_chains"]
    edges = topo["edges"]
    boundaries = topo["tad_positions"]
    gene_mask = topo["gene_mask"]
    ctcf_mask = topo["ctcf_mask"]
    n_boundaries = max(topo["boundaries"], 1)

    pos = _load_positions(h5_path)
    frames, lefs, _ = pos.shape

    legs = pos.reshape(frames, lefs * 2)
    leg_chain = legs // chain
    leg_rel = legs % chain

    lo = np.minimum(pos[..., 0], pos[..., 1])
    hi = np.maximum(pos[..., 0], pos[..., 1])
    lo_chain = lo // chain
    hi_chain = hi // chain
    same_chain = lo_chain == hi_chain
    lo_rel = lo % chain
    hi_rel = hi % chain
    span = hi_rel - lo_rel
    lo_tad = np.searchsorted(edges, lo_rel, side="right") - 1
    hi_tad = np.searchsorted(edges, hi_rel, side="right") - 1
    cross_tad = same_chain & (lo_tad != hi_tad)
    multi_tad = same_chain & (hi_tad > lo_tad + 1)
    crossed_boundary_count = np.maximum(0, hi_tad - lo_tad) * same_chain

    left_rel = pos[..., 0] % chain
    right_rel = pos[..., 1] % chain
    left_ctcf = ctcf_mask[left_rel]
    right_ctcf = ctcf_mask[right_rel]
    exactly_one_ctcf = same_chain & (left_ctcf ^ right_ctcf)
    boundary_crossing_stripe = exactly_one_ctcf & cross_tad

    prev = pos[:-1]
    curr = pos[1:]

    core_rows: list[dict] = []
    boundary_rows: list[dict] = []
    aux_rows: list[dict] = []
    length_rows: list[dict] = []

    for chain_idx in range(num_chains):
        chain_loops = same_chain & (lo_chain == chain_idx)
        spans = span[chain_loops].astype(float)
        n_spans = int(spans.size)
        mean_span = _safe_mean(spans)

        gene_hits = int(((leg_chain == chain_idx) & gene_mask[leg_rel]).sum())
        ctcf_hits = int(((leg_chain == chain_idx) & ctcf_mask[leg_rel]).sum())
        one_ctcf_count = int((exactly_one_ctcf & (lo_chain == chain_idx)).sum())
        stripe_cross_count = int((boundary_crossing_stripe & (lo_chain == chain_idx)).sum())
        cross_tad_count = int((cross_tad & (lo_chain == chain_idx)).sum())
        multi_tad_count = int((multi_tad & (lo_chain == chain_idx)).sum())
        crossed_boundary_total = int((crossed_boundary_count * (lo_chain == chain_idx)).sum())

        pass_events = 0
        moving_steps = 0
        for leg in (0, 1):
            before = prev[..., leg]
            after = curr[..., leg]
            in_chain = (before // chain == chain_idx) & (after // chain == chain_idx)
            before_rel = before % chain
            after_rel = after % chain
            real_step = in_chain & (np.abs(after_rel - before_rel) == 1)
            moving_steps += int(real_step.sum())
            step_lo = np.minimum(before_rel, after_rel)
            step_hi = np.maximum(before_rel, after_rel)
            pass_events += int((np.isin(step_hi, boundaries) & (step_lo == step_hi - 1) & real_step).sum())

        core_metrics = {
            "cohesin_gene_body": (
                gene_hits / (frames * max(topo["gene_sites"], 1)),
                gene_hits,
                frames * max(topo["gene_sites"], 1),
            ),
            "cohesin_ctcf_boundary": (
                ctcf_hits / (frames * max(topo["ctcf_sites"], 1)),
                ctcf_hits,
                frames * max(topo["ctcf_sites"], 1),
            ),
            "boundary_pass_event": (
                pass_events / ((frames - 1) * n_boundaries),
                pass_events,
                (frames - 1) * n_boundaries,
            ),
            "mean_loop_length": (mean_span, n_spans, 1),
            "boundary_crossing_stripe": (
                stripe_cross_count / (frames * n_boundaries),
                stripe_cross_count,
                frames * n_boundaries,
            ),
        }
        for metric, (raw_value, raw_count, denominator) in core_metrics.items():
            core_rows.append(
                _row(
                    label=label,
                    chain_idx=chain_idx,
                    metric=metric,
                    metric_label=CORE_LABELS[metric],
                    raw_value=raw_value,
                    raw_count=raw_count,
                    denominator=denominator,
                    frames=frames,
                    lefs=lefs,
                    topo=topo,
                )
            )

        boundary_metrics = {
            "boundary_pass_event": (
                pass_events / ((frames - 1) * n_boundaries),
                pass_events,
                (frames - 1) * n_boundaries,
            ),
            "cross_tad_occupancy": (
                cross_tad_count / (frames * n_boundaries),
                cross_tad_count,
                frames * n_boundaries,
            ),
            "crossed_boundaries": (
                crossed_boundary_total / (frames * n_boundaries),
                crossed_boundary_total,
                frames * n_boundaries,
            ),
            "multi_tad_crossing": (
                multi_tad_count / (frames * n_boundaries),
                multi_tad_count,
                frames * n_boundaries,
            ),
            "one_ctcf_leg_stripe": (
                one_ctcf_count / (frames * n_boundaries),
                one_ctcf_count,
                frames * n_boundaries,
            ),
            "boundary_crossing_stripe": (
                stripe_cross_count / (frames * n_boundaries),
                stripe_cross_count,
                frames * n_boundaries,
            ),
        }
        for metric, (raw_value, raw_count, denominator) in boundary_metrics.items():
            boundary_rows.append(
                _row(
                    label=label,
                    chain_idx=chain_idx,
                    metric=metric,
                    metric_label=BOUNDARY_LABELS[metric],
                    raw_value=raw_value,
                    raw_count=raw_count,
                    denominator=denominator,
                    frames=frames,
                    lefs=lefs,
                    topo=topo,
                )
            )

        length_rows.append(
            {
                "config": label,
                "chain": chain_idx,
                "n_loops": n_spans,
                "mean_loop_kb": mean_span,
                "median_loop_kb": _safe_percentile(spans, 50),
                "p75_loop_kb": _safe_percentile(spans, 75),
                "p90_loop_kb": _safe_percentile(spans, 90),
                "p95_loop_kb": _safe_percentile(spans, 95),
                "p99_loop_kb": _safe_percentile(spans, 99),
            }
        )

        aux_rows.append(
            {
                "config": label,
                "chain": chain_idx,
                "one_ctcf_leg_any_per_frame": one_ctcf_count / frames,
                "one_ctcf_leg_any_per_boundary_frame": one_ctcf_count / (frames * n_boundaries),
                "boundary_crossing_stripe_per_frame": stripe_cross_count / frames,
                "boundary_crossing_stripe_per_boundary_frame": stripe_cross_count / (frames * n_boundaries),
                "boundary_crossing_stripe_frac_of_one_ctcf": stripe_cross_count / max(one_ctcf_count, 1),
                "any_cross_per_frame": cross_tad_count / frames,
                "any_cross_per_boundary_frame": cross_tad_count / (frames * n_boundaries),
                "multi_tad_cross_per_frame": multi_tad_count / frames,
                "multi_tad_cross_per_boundary_frame": multi_tad_count / (frames * n_boundaries),
                "crossed_boundaries_per_frame": crossed_boundary_total / frames,
                "crossed_boundaries_per_boundary_frame": crossed_boundary_total / (frames * n_boundaries),
                "boundary_pass_events_per_boundary_frame": pass_events / ((frames - 1) * n_boundaries),
                "boundary_pass_events_per_moving_leg_step": pass_events / max(moving_steps, 1),
            }
        )

    return core_rows, boundary_rows, aux_rows, length_rows


def _two_sample_permutation_pvalue(base_values: np.ndarray, comp_values: np.ndarray) -> float:
    base = np.asarray(base_values, dtype=float)
    comp = np.asarray(comp_values, dtype=float)
    base = base[np.isfinite(base)]
    comp = comp[np.isfinite(comp)]
    if base.size < 2 or comp.size < 2:
        return float("nan")

    observed = abs(float(comp.mean() - base.mean()))
    if np.isclose(observed, 0.0):
        return 1.0

    pooled = np.concatenate([base, comp])
    n_total = pooled.size
    n_comp = comp.size
    total_partitions = comb(n_total, n_comp)

    if total_partitions > 200_000:
        rng = np.random.default_rng(12345)
        sampled = np.empty(200_000, dtype=float)
        for idx in range(sampled.size):
            comp_idx = rng.choice(n_total, size=n_comp, replace=False)
            mask = np.zeros(n_total, dtype=bool)
            mask[comp_idx] = True
            sampled[idx] = abs(float(pooled[mask].mean() - pooled[~mask].mean()))
        return float(np.mean(sampled >= observed - 1e-15))

    extreme = 0
    for comp_idx_tuple in combinations(range(n_total), n_comp):
        mask = np.zeros(n_total, dtype=bool)
        mask[list(comp_idx_tuple)] = True
        permuted = abs(float(pooled[mask].mean() - pooled[~mask].mean()))
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / total_partitions


def _significance(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _folds_summary_and_stats(
    rows: list[dict],
    label1: str,
    label2: str,
    metric_order: list[str],
    metric_labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame(rows)
    baseline_means = metrics[metrics["config"] == label1].groupby("metric")["raw_value"].mean().to_dict()
    metrics["baseline_config1_mean"] = metrics["metric"].map(baseline_means)
    metrics["fold_vs_config1_mean"] = metrics["raw_value"] / metrics["baseline_config1_mean"]

    summary = (
        metrics.groupby(["config", "metric"], observed=True)
        .agg(
            mean_raw=("raw_value", "mean"),
            sd_raw=("raw_value", "std"),
            mean_fold=("fold_vs_config1_mean", "mean"),
            sd_fold=("fold_vs_config1_mean", "std"),
            n_chains=("chain", "count"),
        )
        .reset_index()
    )
    summary["metric_label"] = summary["metric"].map(lambda metric: metric_labels[metric].replace("\n", " "))

    stat_rows = []
    for metric in metric_order:
        metric_values = metrics[metrics["metric"] == metric]
        base = metric_values[metric_values["config"] == label1]["raw_value"].to_numpy(dtype=float)
        comp = metric_values[metric_values["config"] == label2]["raw_value"].to_numpy(dtype=float)
        p_value = _two_sample_permutation_pvalue(base, comp)
        stat_rows.append(
            {
                "metric": metric,
                "metric_label": metric_labels[metric].replace("\n", " "),
                "test": STAT_TEST_NAME,
                "n_config1": int(base.size),
                "n_config2": int(comp.size),
                "baseline_mean_raw": float(base.mean()) if base.size else float("nan"),
                "comparison_mean_raw": float(comp.mean()) if comp.size else float("nan"),
                "mean_difference_raw": float(comp.mean() - base.mean()) if base.size and comp.size else float("nan"),
                "mean_fold_vs_baseline_mean": float((comp / base.mean()).mean()) if base.size and comp.size and base.mean() else float("nan"),
                "p_value": p_value,
                "significance": _significance(p_value),
            }
        )
    stats = pd.DataFrame(stat_rows)
    summary = summary.merge(stats[["metric", "p_value", "significance", "test"]], on="metric", how="left")
    return metrics, summary, stats


def _format_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=NA"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3g}"


def _draw_metric_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    stats: pd.DataFrame,
    metric: str,
    metric_label: str,
    *,
    label1: str,
    label2: str,
    section_label: str,
    rng: np.random.Generator,
    colors: dict[str, str],
    show_ylabel: bool,
) -> None:
    sub_metric = metrics[metrics["metric"] == metric]
    values = sub_metric["fold_vs_config1_mean"].to_numpy(dtype=float)
    y_min = min(0.78, float(np.nanmin(values)) * 0.92)
    y_max_points = max(1.22, float(np.nanmax(values)) * 1.08)
    y_range = max(y_max_points - y_min, 0.25)
    annot_y = y_max_points + 0.04 * y_range
    y_top = annot_y + 0.13 * y_range

    for label, offset in [(label1, -0.11), (label2, 0.11)]:
        sub = sub_metric[sub_metric["config"] == label].sort_values("chain")
        x = np.full(len(sub), offset) + rng.uniform(-0.035, 0.035, len(sub))
        y = sub["fold_vs_config1_mean"].to_numpy()
        ax.scatter(
            x,
            y,
            s=48,
            color=colors[label],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )
        ax.hlines(
            float(y.mean()),
            offset - 0.075,
            offset + 0.075,
            color=colors[label],
            linewidth=2.2,
            zorder=4,
        )

    stat = stats[stats["metric"] == metric].iloc[0]
    ax.text(
        0.0,
        annot_y,
        f"{stat['significance']}\n{_format_p(float(stat['p_value']))}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#222222",
    )

    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.65, zorder=1)
    ax.set_xticks([-0.11, 0.11])
    ax.set_xticklabels([label1, label2], fontsize=9)
    ax.set_title(f"{section_label}\n{metric_label}", fontsize=10, pad=10)
    if show_ylabel:
        ax.set_ylabel("Fold change vs config1 mean", fontsize=9)
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(y_min, y_top)
    ax.grid(axis="y", color="#d4d8df", linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric_panels_svg(
    core_metrics: pd.DataFrame,
    core_stats: pd.DataFrame,
    boundary_metrics: pd.DataFrame,
    boundary_stats: pd.DataFrame,
    out_dir: Path,
    *,
    label1: str,
    label2: str,
    stem: str,
    title: str,
    ncols: int = 3,
) -> Path:
    colors = {label1: "#8a8f98", label2: "#2563eb"}
    rng = np.random.default_rng(29)
    panels = [
        ("Core", "cohesin_gene_body", CORE_LABELS["cohesin_gene_body"], core_metrics, core_stats),
        ("Core", "cohesin_ctcf_boundary", CORE_LABELS["cohesin_ctcf_boundary"], core_metrics, core_stats),
        ("Core", "mean_loop_length", CORE_LABELS["mean_loop_length"], core_metrics, core_stats),
        ("Boundary", "boundary_pass_event", BOUNDARY_LABELS["boundary_pass_event"], boundary_metrics, boundary_stats),
        ("Boundary", "cross_tad_occupancy", BOUNDARY_LABELS["cross_tad_occupancy"], boundary_metrics, boundary_stats),
        ("Boundary", "crossed_boundaries", BOUNDARY_LABELS["crossed_boundaries"], boundary_metrics, boundary_stats),
        ("Boundary", "multi_tad_crossing", BOUNDARY_LABELS["multi_tad_crossing"], boundary_metrics, boundary_stats),
        ("Boundary", "one_ctcf_leg_stripe", BOUNDARY_LABELS["one_ctcf_leg_stripe"], boundary_metrics, boundary_stats),
        ("Boundary", "boundary_crossing_stripe", BOUNDARY_LABELS["boundary_crossing_stripe"], boundary_metrics, boundary_stats),
    ]

    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.15 * ncols, 3.75 * nrows), squeeze=False)

    for idx, (section, metric, metric_label, metrics, stats) in enumerate(panels):
        ax = axes.flat[idx]
        _draw_metric_panel(
            ax,
            metrics,
            stats,
            metric,
            metric_label.replace("\n", " "),
            label1=label1,
            label2=label2,
            section_label=section,
            rng=rng,
            colors=colors,
            show_ylabel=idx % ncols == 0,
        )

    for ax in axes.flat[len(panels) :]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], marker="o", color="none", label=f"{label1} chains", markerfacecolor=colors[label1], markeredgecolor="white", markersize=8),
        Line2D([0], [0], marker="o", color="none", label=f"{label2} chains", markerfacecolor=colors[label2], markeredgecolor="white", markersize=8),
        Line2D([0], [0], color="#222222", linestyle="--", label=f"{label1} mean = 1"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    svg_path = out_dir / f"{stem}_all_metrics.svg"
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    return svg_path


def write_outputs(
    out_dir: Path,
    label1: str,
    label2: str,
    cfg1: Path,
    h5_1: Path,
    cfg2: Path,
    h5_2: Path,
    core_rows: list[dict],
    boundary_rows: list[dict],
    aux_rows: list[dict],
    length_rows: list[dict],
    title: str | None,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    core_metrics, core_summary, core_stats = _folds_summary_and_stats(
        core_rows, label1, label2, CORE_ORDER, CORE_LABELS
    )
    core_metrics.to_csv(out_dir / "chain_metric_folds_extended.tsv", sep="\t", index=False)
    core_summary.to_csv(out_dir / "chain_metric_summary_extended.tsv", sep="\t", index=False)
    core_stats.to_csv(out_dir / "chain_metric_stats_extended.tsv", sep="\t", index=False)

    boundary_metrics, boundary_summary, boundary_stats = _folds_summary_and_stats(
        boundary_rows, label1, label2, BOUNDARY_ORDER, BOUNDARY_LABELS
    )
    boundary_metrics.to_csv(out_dir / "boundary_metric_folds.tsv", sep="\t", index=False)
    boundary_summary.to_csv(out_dir / "boundary_metric_summary.tsv", sep="\t", index=False)
    boundary_stats.to_csv(out_dir / "boundary_metric_stats.tsv", sep="\t", index=False)

    aux = pd.DataFrame(aux_rows)
    aux.to_csv(out_dir / "stripe_and_boundary_auxiliary.tsv", sep="\t", index=False)

    lengths = pd.DataFrame(length_rows)
    lengths.to_csv(out_dir / "loop_length_by_chain.tsv", sep="\t", index=False)

    method = {
        "config1_label": label1,
        "config2_label": label2,
        "config1_h5": str(h5_1),
        "config2_h5": str(h5_2),
        "config1_yaml": str(cfg1),
        "config2_yaml": str(cfg2),
        "normalization": "fold_vs_mean_raw_value_of_config1_replicate_chains_per_metric",
        "statistical_test": STAT_TEST_NAME,
        "significance_codes": {"ns": "p >= 0.05", "*": "p < 0.05", "**": "p < 0.01", "***": "p < 0.001"},
        "stripe_definition": (
            "exactly one LEF leg at an internal convergent CTCF anchor site and "
            "the two legs occupy different TAD intervals; anchor status is "
            "inferred from position because H5 does not store CTCF flags"
        ),
        "boundary_metric_definitions": {
            "boundary_pass_event": "consecutive-frame leg step across an internal TAD boundary, p-1 <-> p, filtering unload/reload jumps",
            "cross_tad_occupancy": "same-chain loop spans one or more TAD boundaries; each loop-frame counts once",
            "crossed_boundaries": "same-chain loop spans one or more TAD boundaries; a loop spanning k boundaries counts k",
            "multi_tad_crossing": "same-chain loop spans at least two TAD boundaries",
            "one_ctcf_leg_stripe": "exactly one leg is at an inferred CTCF anchor, regardless of whether the loop crosses a boundary",
            "boundary_crossing_stripe": "exactly one leg is at an inferred CTCF anchor and the loop spans at least one TAD boundary",
        },
        "loop_length_definition": "same-chain LEF span in kb because 1 monomer = 1 kb",
        "core_baseline_means": core_metrics[core_metrics["config"] == label1].groupby("metric")["raw_value"].mean().to_dict(),
        "boundary_baseline_means": boundary_metrics[boundary_metrics["config"] == label1].groupby("metric")["raw_value"].mean().to_dict(),
    }
    (out_dir / "method_extended.json").write_text(json.dumps(method, indent=2))

    stem = f"{_safe_label(label2)}_vs_{_safe_label(label1)}"
    svg_path = plot_metric_panels_svg(
        core_metrics,
        core_stats,
        boundary_metrics,
        boundary_stats,
        out_dir,
        label1=label1,
        label2=label2,
        stem=stem,
        title=title or f"{label2} vs {label1} per replicate chain, 1D LEF metrics",
    )
    return svg_path, core_summary, boundary_summary, lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two 1D LEF H5s per replicate chain and plot config2/config1 folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config1", type=Path, required=True, help="baseline config YAML")
    parser.add_argument(
        "--h5-config1",
        type=Path,
        default=None,
        help="baseline LEFPositions.h5; if omitted, run/reuse OUT_DIR/LABEL1/LEFPositions.h5",
    )
    parser.add_argument("--config2", type=Path, required=True, help="comparison config YAML")
    parser.add_argument(
        "--h5-config2",
        type=Path,
        default=None,
        help="comparison LEFPositions.h5; if omitted, run/reuse OUT_DIR/LABEL2/LEFPositions.h5",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for SVG/TSV outputs")
    parser.add_argument("--label1", default="config1", help="baseline label used in plots/tables")
    parser.add_argument("--label2", default="config2", help="comparison label used in plots/tables")
    parser.add_argument("--title", default=None, help="optional core plot title")
    parser.add_argument(
        "--force-1d",
        action="store_true",
        help="rerun the 1D LEF stage even if the target LEFPositions.h5 already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    cfg1 = _read_yaml(args.config1)
    cfg2 = _read_yaml(args.config2)
    top1 = _topology(cfg1)
    top2 = _topology(cfg2)
    _validate_compatible(top1, top2)

    h5_config1 = _resolve_h5(
        cfg_path=args.config1,
        requested_h5=args.h5_config1,
        out_dir=args.out_dir,
        label=args.label1,
        force_1d=args.force_1d,
    )
    h5_config2 = _resolve_h5(
        cfg_path=args.config2,
        requested_h5=args.h5_config2,
        out_dir=args.out_dir,
        label=args.label2,
        force_1d=args.force_1d,
    )

    core1, boundary1, aux1, lengths1 = analyze_run(args.label1, args.config1, h5_config1, top1)
    core2, boundary2, aux2, lengths2 = analyze_run(args.label2, args.config2, h5_config2, top2)
    svg_path, core_summary, boundary_summary, lengths = write_outputs(
        args.out_dir,
        args.label1,
        args.label2,
        args.config1,
        h5_config1,
        args.config2,
        h5_config2,
        core1 + core2,
        boundary1 + boundary2,
        aux1 + aux2,
        lengths1 + lengths2,
        args.title,
    )

    print(f"wrote {svg_path}")
    print(f"wrote {args.out_dir / 'chain_metric_summary_extended.tsv'}")
    print(f"wrote {args.out_dir / 'chain_metric_stats_extended.tsv'}")
    print(f"wrote {args.out_dir / 'boundary_metric_summary.tsv'}")
    print(f"wrote {args.out_dir / 'boundary_metric_stats.tsv'}")
    print(f"wrote {args.out_dir / 'loop_length_by_chain.tsv'}")
    print(f"wrote {args.out_dir / 'stripe_and_boundary_auxiliary.tsv'}")
    print()
    print("core metrics:")
    print(core_summary[["config", "metric", "mean_fold", "sd_fold", "mean_raw", "p_value", "significance", "n_chains"]].to_string(index=False))
    print()
    print("boundary metrics:")
    print(boundary_summary[["config", "metric", "mean_fold", "sd_fold", "mean_raw", "p_value", "significance", "n_chains"]].to_string(index=False))
    print()
    print("loop length by config:")
    loop_summary = lengths.groupby("config").mean(numeric_only=True)
    print(loop_summary[["mean_loop_kb", "median_loop_kb", "p95_loop_kb", "p99_loop_kb"]].to_string())


if __name__ == "__main__":
    main()
