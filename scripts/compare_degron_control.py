#!/usr/bin/env python3
"""Compare degron vs control contact maps.

Outputs:
  - control_noice_oe.npy
  - degron_noice_oe.npy
  - noice_vs_ice_oe.png
  - observed_only.png
  - observed_cis_decay_fold_change_bp.csv
  - observed_cis_decay_fold_change_bp.png
  - observed_degron_over_control_log2fc.npy
  - degron_over_control_oe_fold_change.npy
  - degron_over_control_oe_log2fc.npy
  - degron_over_control_oe_log2fc.png
  - insulation_scores_window{window}.csv
  - insulation_boundary_summary_window{window}.csv
  - insulation_compare_window{window}.png
  - observed_insulation_scores_window{window}.csv
  - observed_insulation_boundary_summary_window{window}.csv
  - observed_insulation_compare_window{window}.png
  - tad_strengths.csv
  - tad_strength_differences.csv
  - tad_strengths.png
  - boundary_strengths_window{window}.csv
  - tad_boundary_strengths_window{window}.csv
  - tad_boundary_strengths_window{window}.png
  - summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs_dir",
        nargs="?",
        default="runs",
        type=Path,
        help="Folder containing control/ and degron/ run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to RUNS_DIR/degron_control_compare.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Diamond insulation half-window in bins.",
    )
    parser.add_argument(
        "--condition-a",
        default="control",
        help="Denominator condition folder.",
    )
    parser.add_argument(
        "--condition-b",
        default="degron",
        help="Numerator condition folder.",
    )
    parser.add_argument(
        "--mask-diagonals",
        type=int,
        default=1,
        help=(
            "Mask diagonal band |i-j| <= N before map comparisons and observed "
            "insulation. Default masks main and +/-1 diagonals."
        ),
    )
    parser.add_argument(
        "--bin-size-bp",
        type=int,
        default=1000,
        help="Genomic size per monomer/bin in bp for cis-decay plots.",
    )
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def finite_quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def mask_diagonals(matrix: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(matrix, dtype=float).copy()
    if radius < 0:
        return out
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError("matrix must be square")

    n = out.shape[0]
    idx = np.arange(n)
    for offset in range(int(radius) + 1):
        rows = idx[: n - offset]
        cols = rows + offset
        out[rows, cols] = np.nan
        if offset:
            out[cols, rows] = np.nan
    return out


def fold_change_oe(control_oe: np.ndarray, degron_oe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if control_oe.shape != degron_oe.shape:
        raise ValueError(f"shape mismatch: control {control_oe.shape}, degron {degron_oe.shape}")

    valid = (
        np.isfinite(control_oe)
        & np.isfinite(degron_oe)
        & (control_oe > 0)
        & (degron_oe > 0)
    )
    fc = np.full(control_oe.shape, np.nan, dtype=float)
    fc[valid] = degron_oe[valid] / control_oe[valid]

    log2fc = np.full(control_oe.shape, np.nan, dtype=float)
    log2fc[valid] = np.log2(fc[valid])
    return fc, log2fc


def observed_over_expected(contact_map: np.ndarray) -> np.ndarray:
    cm = np.asarray(contact_map, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("contact map must be square")

    n = cm.shape[0]
    result = np.full_like(cm, np.nan, dtype=float)
    for offset in range(n):
        diag = np.diagonal(cm, offset=offset)
        finite = diag[np.isfinite(diag)]
        if finite.size == 0:
            continue
        expected = finite.mean()
        if expected <= 0:
            continue
        normalized = diag / expected
        rows = np.arange(n - offset)
        cols = rows + offset
        result[rows, cols] = normalized
        if offset:
            result[cols, rows] = normalized
    return result


def log2_positive(matrix: np.ndarray) -> np.ndarray:
    data = np.asarray(matrix, dtype=float)
    out = np.full(data.shape, np.nan, dtype=float)
    valid = np.isfinite(data) & (data > 0)
    out[valid] = np.log2(data[valid])
    return out


def draw_masked_diagonal_band(ax, n: int, radius: int) -> None:
    if radius < 0:
        return
    for offset in range(-int(radius), int(radius) + 1):
        if offset >= 0:
            x0, x1 = offset, n - 1
            y0, y1 = 0, n - 1 - offset
        else:
            x0, x1 = 0, n - 1 + offset
            y0, y1 = -offset, n - 1
        ax.plot([x0, x1], [y0, y1], color="#d1d5db", lw=1.0, alpha=0.95, solid_capstyle="butt")


def plot_noice_vs_ice(
    conditions: list[tuple[str, np.ndarray, np.ndarray]],
    boundaries: Iterable[int],
    output_path: Path,
    mask_diagonal_radius: int,
) -> dict[str, dict[str, float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked_cmap = plt.get_cmap("coolwarm").copy()
    masked_cmap.set_bad("#e5e7eb")
    log_pairs = [(name, log2_positive(noice), log2_positive(ice)) for name, noice, ice in conditions]
    combined = np.concatenate([
        arr[np.isfinite(arr)]
        for _, noice_log, ice_log in log_pairs
        for arr in (noice_log, ice_log)
        if np.isfinite(arr).any()
    ])
    vmax = 1.0
    if combined.size:
        vmax = max(0.5, float(np.quantile(np.abs(combined), 0.995)))
        vmax = min(vmax, 3.0)

    diffs = [(name, ice_log - noice_log) for name, noice_log, ice_log in log_pairs]
    diff_values = np.concatenate([
        diff[np.isfinite(diff)]
        for _, diff in diffs
        if np.isfinite(diff).any()
    ])
    diff_vmax = 0.5
    if diff_values.size:
        diff_vmax = max(0.25, float(np.quantile(np.abs(diff_values), 0.995)))
        diff_vmax = min(diff_vmax, 2.0)

    fig, axes = plt.subplots(len(log_pairs), 3, figsize=(15, 9), squeeze=False)
    stats: dict[str, dict[str, float]] = {}
    for row_idx, ((name, noice_log, ice_log), (_, diff)) in enumerate(zip(log_pairs, diffs)):
        panels = [
            ("no-ICE log2(O/E)", noice_log, vmax),
            ("ICE log2(O/E)", ice_log, vmax),
            ("ICE - no-ICE", diff, diff_vmax),
        ]
        stats[name] = {
            "ice_minus_noice_mean": float(np.nanmean(diff)),
            "ice_minus_noice_median": float(np.nanmedian(diff)),
            "ice_minus_noice_p05": finite_quantile(diff, 0.05),
            "ice_minus_noice_p95": finite_quantile(diff, 0.95),
        }
        for col_idx, (title, data, panel_vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                data,
                cmap=masked_cmap,
                vmin=-panel_vmax,
                vmax=panel_vmax,
                interpolation="nearest",
                origin="upper",
            )
            for boundary in boundaries:
                ax.axhline(boundary - 0.5, color="black", lw=0.5, ls="--", alpha=0.30)
                ax.axvline(boundary - 0.5, color="black", lw=0.5, ls="--", alpha=0.30)
            draw_masked_diagonal_band(ax, data.shape[0], mask_diagonal_radius)
            ax.set_title(f"{name}: {title}")
            ax.set_xlabel("monomer index")
            if col_idx == 0:
                ax.set_ylabel("monomer index")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("log2 scale")

    fig.suptitle(f"Masked diagonal band |i-j| <= {mask_diagonal_radius}", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return stats


def plot_observed_only(
    control_raw: np.ndarray,
    degron_raw: np.ndarray,
    boundaries: Iterable[int],
    output_path: Path,
    mask_diagonal_radius: int,
) -> tuple[np.ndarray, dict[str, float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    magmap = plt.get_cmap("magma").copy()
    magmap.set_bad("#e5e7eb")
    diverging = plt.get_cmap("coolwarm").copy()
    diverging.set_bad("#e5e7eb")
    control = np.asarray(control_raw, dtype=float)
    degron = np.asarray(degron_raw, dtype=float)
    valid = np.isfinite(control) & np.isfinite(degron) & (control > 0) & (degron > 0)
    raw_log2fc = np.full(control.shape, np.nan, dtype=float)
    raw_log2fc[valid] = np.log2(degron[valid] / control[valid])

    control_log = np.log2(np.maximum(control, 0.0) + 1.0)
    degron_log = np.log2(np.maximum(degron, 0.0) + 1.0)
    count_values = np.concatenate([control_log[np.isfinite(control_log)], degron_log[np.isfinite(degron_log)]])
    count_vmax = float(np.quantile(count_values, 0.995)) if count_values.size else 1.0
    fc_vmax = max(0.5, finite_quantile(np.abs(raw_log2fc), 0.995))
    fc_vmax = min(fc_vmax, 3.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("control observed log2(count + 1)", control_log, magmap, 0.0, count_vmax, "log2 count"),
        ("degron observed log2(count + 1)", degron_log, magmap, 0.0, count_vmax, "log2 count"),
        ("observed degron/control log2FC", raw_log2fc, diverging, -fc_vmax, fc_vmax, "log2FC"),
    ]
    for ax, (title, data, cmap, vmin, vmax, label) in zip(axes, panels):
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="upper",
        )
        for boundary in boundaries:
            line_color = "white" if cmap is magmap else "black"
            ax.axhline(boundary - 0.5, color=line_color, lw=0.5, ls="--", alpha=0.35)
            ax.axvline(boundary - 0.5, color=line_color, lw=0.5, ls="--", alpha=0.35)
        draw_masked_diagonal_band(ax, data.shape[0], mask_diagonal_radius)
        ax.set_title(title)
        ax.set_xlabel("monomer index")
        ax.set_ylabel("monomer index")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)

    fig.suptitle(f"Observed maps, masked diagonal band |i-j| <= {mask_diagonal_radius}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    stats = {
        "finite_pixels": int(np.isfinite(raw_log2fc).sum()),
        "mean": float(np.nanmean(raw_log2fc)),
        "median": float(np.nanmedian(raw_log2fc)),
        "p05": finite_quantile(raw_log2fc, 0.05),
        "p95": finite_quantile(raw_log2fc, 0.95),
    }
    return raw_log2fc, stats


def cis_decay(contact_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cm = np.asarray(contact_map, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("contact map must be square")
    distances = []
    means = []
    for offset in range(1, cm.shape[0]):
        diag = np.diagonal(cm, offset=offset)
        finite = diag[np.isfinite(diag)]
        if finite.size == 0:
            continue
        distances.append(offset)
        means.append(float(finite.mean()))
    return np.asarray(distances, dtype=int), np.asarray(means, dtype=float)


def log_smooth(x: np.ndarray, y: np.ndarray, points: int = 250, window: int = 17) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    if x.size == 0:
        return x, y
    logx = np.log10(x)
    grid = np.linspace(logx.min(), logx.max(), min(points, x.size))
    interp = np.interp(grid, logx, y)
    if window > 1 and interp.size >= window:
        kernel = np.hanning(window)
        kernel = kernel / kernel.sum()
        padded = np.pad(interp, (window // 2, window // 2), mode="edge")
        interp = np.convolve(padded, kernel, mode="valid")[: interp.size]
    return 10 ** grid, interp


def plot_cis_decay_fold_change(
    control_raw: np.ndarray,
    degron_raw: np.ndarray,
    output_csv: Path,
    output_png: Path,
    bin_size_bp: int,
) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distances, control_decay = cis_decay(control_raw)
    distances2, degron_decay = cis_decay(degron_raw)
    if not np.array_equal(distances, distances2):
        raise ValueError("cis-decay distances differ")

    valid = (control_decay > 0) & (degron_decay > 0)
    fold_change = np.full(control_decay.shape, np.nan, dtype=float)
    fold_change[valid] = degron_decay[valid] / control_decay[valid]
    bp = distances.astype(float) * int(bin_size_bp)
    smooth_bp, smooth_fc = log_smooth(bp, fold_change)

    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "distance_bins",
                "distance_bp",
                "control_mean_observed",
                "degron_mean_observed",
                "fold_change_degron_over_control",
                "log2_fold_change_degron_over_control",
            ]
        )
        for dist, dist_bp, control_mean, degron_mean, fc in zip(
            distances, bp, control_decay, degron_decay, fold_change
        ):
            writer.writerow(
                [
                    int(dist),
                    int(dist_bp),
                    float(control_mean),
                    float(degron_mean),
                    float(fc) if np.isfinite(fc) else "",
                    float(np.log2(fc)) if np.isfinite(fc) and fc > 0 else "",
                ]
            )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(smooth_bp, smooth_fc, color="#2563eb", lw=1.6, label="Degron vs Control")
    ax.axhline(1.0, color="gray", lw=1.0, ls="--")
    ax.set_title("Fold Change in Cis Decay Curve")
    ax.set_xlabel("Genomic Distance (bp)")
    ax.set_ylabel("Fold Change in Contact Frequency")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    comparable = bp <= 2_500_000
    return {
        "bin_size_bp": int(bin_size_bp),
        "min_bp": int(np.nanmin(bp)),
        "max_bp": int(np.nanmax(bp)),
        "mean_fold_change": float(np.nanmean(fold_change)),
        "median_fold_change": float(np.nanmedian(fold_change)),
        "min_fold_change": float(np.nanmin(fold_change[comparable])),
        "max_fold_change": float(np.nanmax(fold_change[comparable])),
    }


def plot_fold_change(
    log2fc: np.ndarray,
    boundaries: Iterable[int],
    output_path: Path,
    mask_diagonal_radius: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked_cmap = plt.get_cmap("coolwarm").copy()
    masked_cmap.set_bad("#e5e7eb")
    finite = log2fc[np.isfinite(log2fc)]
    vmax = 1.0
    if finite.size:
        vmax = max(0.5, float(np.quantile(np.abs(finite), 0.995)))
        vmax = min(vmax, 3.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        log2fc,
        cmap=masked_cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="black", lw=0.6, ls="--", alpha=0.35)
        ax.axvline(boundary - 0.5, color="black", lw=0.6, ls="--", alpha=0.35)
    draw_masked_diagonal_band(ax, log2fc.shape[0], mask_diagonal_radius)
    ax.set_title(f"Degron / control O/E log2 fold change, masked |i-j| <= {mask_diagonal_radius}")
    ax.set_xlabel("monomer index")
    ax.set_ylabel("monomer index")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log2((degron O/E) / (control O/E))")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def diamond_insulation(contact_map: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if contact_map.ndim != 2 or contact_map.shape[0] != contact_map.shape[1]:
        raise ValueError("contact map must be square")
    if window < 1:
        raise ValueError("window must be >= 1")
    n = contact_map.shape[0]
    if n < 2 * window:
        raise ValueError(f"window {window} too large for map size {n}")

    cm = np.asarray(contact_map, dtype=float)
    valid_pixels = np.isfinite(cm)
    cm = np.where(valid_pixels, cm, 0.0)
    prefix = np.pad(cm.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    count_prefix = np.pad(valid_pixels.astype(float).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    positions = np.arange(window, n - window + 1, dtype=int)
    scores = np.empty(positions.shape[0], dtype=float)

    for idx, boundary in enumerate(positions):
        r1 = boundary - window
        r2 = boundary
        c1 = boundary
        c2 = boundary + window
        total = prefix[r2, c2] - prefix[r1, c2] - prefix[r2, c1] + prefix[r1, c1]
        denom = count_prefix[r2, c2] - count_prefix[r1, c2] - count_prefix[r2, c1] + count_prefix[r1, c1]
        scores[idx] = total / denom if denom > 0 else np.nan

    positive = scores[scores > 0]
    log2_over_mean = np.full(scores.shape, np.nan, dtype=float)
    if positive.size:
        mean_positive = positive.mean()
        valid = scores > 0
        log2_over_mean[valid] = np.log2(scores[valid] / mean_positive)
    return positions, scores, log2_over_mean


def write_insulation_csv(
    path: Path,
    positions: np.ndarray,
    control_scores: np.ndarray,
    degron_scores: np.ndarray,
    control_norm: np.ndarray,
    degron_norm: np.ndarray,
) -> np.ndarray:
    valid = (control_scores > 0) & (degron_scores > 0)
    log2_raw_fc = np.full(control_scores.shape, np.nan, dtype=float)
    log2_raw_fc[valid] = np.log2(degron_scores[valid] / control_scores[valid])
    norm_delta = degron_norm - control_norm

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "position",
                "control_insulation",
                "degron_insulation",
                "control_log2_over_mean",
                "degron_log2_over_mean",
                "degron_minus_control_log2_over_mean",
                "log2_degron_over_control_raw",
            ]
        )
        for row in zip(
            positions,
            control_scores,
            degron_scores,
            control_norm,
            degron_norm,
            norm_delta,
            log2_raw_fc,
        ):
            writer.writerow([int(row[0]), *[float(x) if np.isfinite(x) else "" for x in row[1:]]])
    return log2_raw_fc


def nearest_rows(positions: np.ndarray, targets: Iterable[int]) -> list[tuple[int, int]]:
    rows = []
    for target in targets:
        idx = int(np.argmin(np.abs(positions - int(target))))
        rows.append((int(target), idx))
    return rows


def write_boundary_summary(
    path: Path,
    positions: np.ndarray,
    boundaries: Iterable[int],
    control_scores: np.ndarray,
    degron_scores: np.ndarray,
    control_norm: np.ndarray,
    degron_norm: np.ndarray,
    log2_raw_fc: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "boundary",
                "nearest_position",
                "control_insulation",
                "degron_insulation",
                "control_log2_over_mean",
                "degron_log2_over_mean",
                "degron_minus_control_log2_over_mean",
                "log2_degron_over_control_raw",
            ]
        )
        for boundary, idx in nearest_rows(positions, boundaries):
            writer.writerow(
                [
                    boundary,
                    int(positions[idx]),
                    float(control_scores[idx]),
                    float(degron_scores[idx]),
                    float(control_norm[idx]),
                    float(degron_norm[idx]),
                    float(degron_norm[idx] - control_norm[idx]),
                    float(log2_raw_fc[idx]),
                ]
            )


def plot_insulation(
    positions: np.ndarray,
    control_norm: np.ndarray,
    degron_norm: np.ndarray,
    log2_raw_fc: np.ndarray,
    boundaries: Iterable[int],
    output_path: Path,
    window: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(positions, control_norm, lw=1.4, color="#2563eb", label="control")
    ax.plot(positions, degron_norm, lw=1.4, color="#dc2626", label="degron")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.35)
    ax.set_ylabel("log2(insulation / condition mean)")
    ax.set_title(f"Diamond insulation scores, window {window}")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.plot(positions, log2_raw_fc, lw=1.2, color="#111827")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.35)
    ax.set_ylabel("raw log2FC")
    ax.set_xlabel("monomer index")

    for boundary in boundaries:
        for axis in axes:
            axis.axvline(boundary, color="black", lw=0.7, ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_observed_insulation(
    positions: np.ndarray,
    control_scores: np.ndarray,
    degron_scores: np.ndarray,
    log2_raw_fc: np.ndarray,
    boundaries: Iterable[int],
    output_path: Path,
    window: int,
    mask_diagonal_radius: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(positions, control_scores, lw=1.3, color="#2563eb", label="control observed")
    ax.plot(positions, degron_scores, lw=1.3, color="#dc2626", label="degron observed")
    ax.set_ylabel("observed diamond mean")
    ax.set_title(f"Observed-contact diamond insulation, window {window}, masked |i-j| <= {mask_diagonal_radius}")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.plot(positions, log2_raw_fc, lw=1.2, color="#111827")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.35)
    ax.set_ylabel("log2(degron/control)")
    ax.set_xlabel("monomer index")

    for boundary in boundaries:
        for axis in axes:
            axis.axvline(boundary, color="black", lw=0.7, ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def region_mean(matrix: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    if r0 >= r1 or c0 >= c1:
        return float("nan")
    region = np.asarray(matrix[r0:r1, c0:c1], dtype=float)
    finite = region[np.isfinite(region)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def region_pair_mean(matrix: np.ndarray, regions: list[tuple[int, int, int, int]]) -> float:
    chunks = []
    for r0, r1, c0, c1 in regions:
        if r0 >= r1 or c0 >= c1:
            continue
        region = np.asarray(matrix[r0:r1, c0:c1], dtype=float)
        finite = region[np.isfinite(region)]
        if finite.size:
            chunks.append(finite)
    if not chunks:
        return float("nan")
    return float(np.concatenate(chunks).mean())


def safe_log2_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return float("nan")
    if numerator <= 0 or denominator <= 0:
        return float("nan")
    return float(np.log2(numerator / denominator))


def tad_strength_rows(
    matrix_name: str,
    control_matrix: np.ndarray,
    degron_matrix: np.ndarray,
    tads: list[dict],
) -> list[dict[str, object]]:
    n = control_matrix.shape[0]
    rows: list[dict[str, object]] = []
    for tad in tads:
        start = int(tad["start"])
        end = int(tad["end"])
        size = int(tad.get("size", end - start))
        left_start = max(0, start - size)
        right_end = min(n, end + size)
        inter_regions = [
            (start, end, left_start, start),
            (start, end, end, right_end),
        ]

        control_intra = region_mean(control_matrix, start, end, start, end)
        control_inter = region_pair_mean(control_matrix, inter_regions)
        degron_intra = region_mean(degron_matrix, start, end, start, end)
        degron_inter = region_pair_mean(degron_matrix, inter_regions)

        control_strength = safe_log2_ratio(control_intra, control_inter)
        degron_strength = safe_log2_ratio(degron_intra, degron_inter)
        rows.append(
            {
                "matrix": matrix_name,
                "tad_label": str(tad.get("label", f"{start}-{end}")),
                "start": start,
                "end": end,
                "size": end - start,
                "flank_size": size,
                "control_intra_mean": control_intra,
                "control_flank_mean": control_inter,
                "control_strength_log2": control_strength,
                "degron_intra_mean": degron_intra,
                "degron_flank_mean": degron_inter,
                "degron_strength_log2": degron_strength,
                "strength_delta_degron_minus_control": (
                    degron_strength - control_strength
                    if np.isfinite(degron_strength) and np.isfinite(control_strength)
                    else float("nan")
                ),
                "intra_log2fc_degron_over_control": safe_log2_ratio(degron_intra, control_intra),
                "flank_log2fc_degron_over_control": safe_log2_ratio(degron_inter, control_inter),
            }
        )
    return rows


def write_rows_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        "" if isinstance(row.get(field), float) and not np.isfinite(row[field])
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )


def plot_tad_strengths(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix_names = list(dict.fromkeys(str(row["matrix"]) for row in rows))
    labels = list(dict.fromkeys(str(row["tad_label"]) for row in rows))
    fig, axes = plt.subplots(len(matrix_names), 2, figsize=(14, 3.4 * len(matrix_names)), squeeze=False)
    x = np.arange(len(labels))
    width = 0.38

    for row_idx, matrix_name in enumerate(matrix_names):
        subset = [row for row in rows if row["matrix"] == matrix_name]
        by_label = {str(row["tad_label"]): row for row in subset}
        control = np.array([by_label[label]["control_strength_log2"] for label in labels], dtype=float)
        degron = np.array([by_label[label]["degron_strength_log2"] for label in labels], dtype=float)
        delta = np.array([by_label[label]["strength_delta_degron_minus_control"] for label in labels], dtype=float)

        ax = axes[row_idx, 0]
        ax.bar(x - width / 2, control, width, color="#2563eb", label="control")
        ax.bar(x + width / 2, degron, width, color="#dc2626", label="degron")
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.35)
        ax.set_title(f"{matrix_name}: TAD strength")
        ax.set_ylabel("log2(intra / flank)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend(frameon=False)

        ax = axes[row_idx, 1]
        colors = np.where(delta >= 0, "#dc2626", "#2563eb")
        ax.bar(x, delta, color=colors)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.35)
        ax.set_title(f"{matrix_name}: degron - control")
        ax.set_ylabel("delta log2 strength")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_tad_strengths(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for matrix_name in dict.fromkeys(str(row["matrix"]) for row in rows):
        subset = [row for row in rows if row["matrix"] == matrix_name]
        deltas = np.array([row["strength_delta_degron_minus_control"] for row in subset], dtype=float)
        summary[matrix_name] = {
            "mean_delta_degron_minus_control": float(np.nanmean(deltas)),
            "median_delta_degron_minus_control": float(np.nanmedian(deltas)),
            "min_delta_degron_minus_control": float(np.nanmin(deltas)),
            "max_delta_degron_minus_control": float(np.nanmax(deltas)),
        }
    return summary


def boundary_strength_rows(
    positions: np.ndarray,
    boundaries: Iterable[int],
    control_norm: np.ndarray,
    degron_norm: np.ndarray,
    log2_raw_fc: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for boundary, idx in nearest_rows(positions, boundaries):
        control_strength = -float(control_norm[idx])
        degron_strength = -float(degron_norm[idx])
        rows.append(
            {
                "boundary": boundary,
                "nearest_position": int(positions[idx]),
                "control_boundary_strength": control_strength,
                "degron_boundary_strength": degron_strength,
                "strength_delta_degron_minus_control": degron_strength - control_strength,
                "raw_insulation_log2fc_degron_over_control": float(log2_raw_fc[idx]),
            }
        )
    return rows


def tad_boundary_strength_rows(
    tads: list[dict],
    boundary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_boundary = {int(row["boundary"]): row for row in boundary_rows}
    rows: list[dict[str, object]] = []
    for tad in tads:
        start = int(tad["start"])
        end = int(tad["end"])
        sides = [b for b in (start, end) if b in by_boundary]
        if not sides:
            continue
        control = np.array([by_boundary[b]["control_boundary_strength"] for b in sides], dtype=float)
        degron = np.array([by_boundary[b]["degron_boundary_strength"] for b in sides], dtype=float)
        rows.append(
            {
                "tad_label": str(tad.get("label", f"{start}-{end}")),
                "start": start,
                "end": end,
                "boundaries_used": ";".join(str(b) for b in sides),
                "control_boundary_strength_mean": float(np.nanmean(control)),
                "degron_boundary_strength_mean": float(np.nanmean(degron)),
                "strength_delta_degron_minus_control": float(np.nanmean(degron - control)),
            }
        )
    return rows


def plot_tad_boundary_strengths(rows: list[dict[str, object]], output_path: Path, window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["tad_label"]) for row in rows]
    x = np.arange(len(labels))
    control = np.array([row["control_boundary_strength_mean"] for row in rows], dtype=float)
    degron = np.array([row["degron_boundary_strength_mean"] for row in rows], dtype=float)
    delta = np.array([row["strength_delta_degron_minus_control"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.38
    axes[0].bar(x - width / 2, control, width, color="#2563eb", label="control")
    axes[0].bar(x + width / 2, degron, width, color="#dc2626", label="degron")
    axes[0].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    axes[0].set_title(f"TAD boundary strength, window {window}")
    axes[0].set_ylabel("-log2(insulation / condition mean)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].legend(frameon=False)

    colors = np.where(delta >= 0, "#dc2626", "#2563eb")
    axes[1].bar(x, delta, color=colors)
    axes[1].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    axes[1].set_title("degron - control")
    axes[1].set_ylabel("delta boundary strength")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_tads(runs_dir: Path, condition: str, n: int) -> list[dict]:
    meta_path = runs_dir / condition / "bridging_viewer_elements.json"
    if not meta_path.exists():
        return []
    data = json.loads(meta_path.read_text())
    tads = []
    for tad in data.get("tads", []):
        start = int(tad["start"])
        end = int(tad["end"])
        if 0 <= start < end <= n:
            item = dict(tad)
            item["start"] = start
            item["end"] = end
            item["size"] = int(item.get("size", end - start))
            item["label"] = str(item.get("label", f"{start}-{end}"))
            tads.append(item)
    return tads


def load_boundaries(runs_dir: Path, condition: str, n: int) -> list[int]:
    meta_path = runs_dir / condition / "bridging_viewer_elements.json"
    if not meta_path.exists():
        return []
    data = json.loads(meta_path.read_text())
    boundaries = set()
    for tad in data.get("tads", []):
        start = int(tad["start"])
        end = int(tad["end"])
        if 0 < start < n:
            boundaries.add(start)
        if 0 < end < n:
            boundaries.add(end)
    return sorted(boundaries)


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    output_dir = (args.output_dir or runs_dir / "degron_control_compare").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    control_dir = runs_dir / args.condition_a
    degron_dir = runs_dir / args.condition_b
    control_oe = load_array(control_dir / "contact_map_oe.npy")
    degron_oe = load_array(degron_dir / "contact_map_oe.npy")
    control_raw = load_array(control_dir / "contact_map.npy")
    degron_raw = load_array(degron_dir / "contact_map.npy")

    if control_raw.shape != degron_raw.shape:
        raise ValueError(f"shape mismatch: control {control_raw.shape}, degron {degron_raw.shape}")
    if control_raw.shape != control_oe.shape:
        raise ValueError(f"raw/OE shape mismatch: raw {control_raw.shape}, OE {control_oe.shape}")

    boundaries = load_boundaries(runs_dir, args.condition_a, control_oe.shape[0])
    tads = load_tads(runs_dir, args.condition_a, control_oe.shape[0])
    mask_radius = int(args.mask_diagonals)

    control_raw_masked = mask_diagonals(control_raw, mask_radius)
    degron_raw_masked = mask_diagonals(degron_raw, mask_radius)
    control_oe_masked = mask_diagonals(control_oe, mask_radius)
    degron_oe_masked = mask_diagonals(degron_oe, mask_radius)

    control_noice_oe = mask_diagonals(observed_over_expected(control_raw), mask_radius)
    degron_noice_oe = mask_diagonals(observed_over_expected(degron_raw), mask_radius)
    np.save(output_dir / f"{args.condition_a}_noice_oe.npy", control_noice_oe)
    np.save(output_dir / f"{args.condition_b}_noice_oe.npy", degron_noice_oe)
    noice_vs_ice_stats = plot_noice_vs_ice(
        [
            (args.condition_a, control_noice_oe, control_oe_masked),
            (args.condition_b, degron_noice_oe, degron_oe_masked),
        ],
        boundaries,
        output_dir / "noice_vs_ice_oe.png",
        mask_radius,
    )
    observed_log2fc, observed_stats = plot_observed_only(
        control_raw_masked,
        degron_raw_masked,
        boundaries,
        output_dir / "observed_only.png",
        mask_radius,
    )
    np.save(output_dir / "observed_degron_over_control_log2fc.npy", observed_log2fc)
    cis_decay_stats = plot_cis_decay_fold_change(
        control_raw_masked,
        degron_raw_masked,
        output_dir / "observed_cis_decay_fold_change_bp.csv",
        output_dir / "observed_cis_decay_fold_change_bp.png",
        args.bin_size_bp,
    )

    fc, log2fc = fold_change_oe(control_oe_masked, degron_oe_masked)
    np.save(output_dir / "degron_over_control_oe_fold_change.npy", fc)
    np.save(output_dir / "degron_over_control_oe_log2fc.npy", log2fc)
    plot_fold_change(log2fc, boundaries, output_dir / "degron_over_control_oe_log2fc.png", mask_radius)

    tad_rows: list[dict[str, object]] = []
    if tads:
        tad_rows.extend(tad_strength_rows("observed", control_raw_masked, degron_raw_masked, tads))
        tad_rows.extend(tad_strength_rows("noice_oe", control_noice_oe, degron_noice_oe, tads))
        tad_rows.extend(tad_strength_rows("ice_oe", control_oe_masked, degron_oe_masked, tads))
        tad_fields = [
            "matrix",
            "tad_label",
            "start",
            "end",
            "size",
            "flank_size",
            "control_intra_mean",
            "control_flank_mean",
            "control_strength_log2",
            "degron_intra_mean",
            "degron_flank_mean",
            "degron_strength_log2",
            "strength_delta_degron_minus_control",
            "intra_log2fc_degron_over_control",
            "flank_log2fc_degron_over_control",
        ]
        write_rows_csv(output_dir / "tad_strengths.csv", tad_rows, tad_fields)
        write_rows_csv(
            output_dir / "tad_strength_differences.csv",
            tad_rows,
            [
                "matrix",
                "tad_label",
                "start",
                "end",
                "strength_delta_degron_minus_control",
                "intra_log2fc_degron_over_control",
                "flank_log2fc_degron_over_control",
            ],
        )
        plot_tad_strengths(tad_rows, output_dir / "tad_strengths.png")

    positions, control_ins, control_norm = diamond_insulation(control_raw_masked, args.window)
    positions2, degron_ins, degron_norm = diamond_insulation(degron_raw_masked, args.window)
    if not np.array_equal(positions, positions2):
        raise ValueError("insulation positions differ")

    scores_csv = output_dir / f"insulation_scores_window{args.window}.csv"
    log2_raw_fc = write_insulation_csv(
        scores_csv,
        positions,
        control_ins,
        degron_ins,
        control_norm,
        degron_norm,
    )
    observed_scores_csv = output_dir / f"observed_insulation_scores_window{args.window}.csv"
    write_insulation_csv(
        observed_scores_csv,
        positions,
        control_ins,
        degron_ins,
        control_norm,
        degron_norm,
    )
    write_boundary_summary(
        output_dir / f"insulation_boundary_summary_window{args.window}.csv",
        positions,
        boundaries,
        control_ins,
        degron_ins,
        control_norm,
        degron_norm,
        log2_raw_fc,
    )
    write_boundary_summary(
        output_dir / f"observed_insulation_boundary_summary_window{args.window}.csv",
        positions,
        boundaries,
        control_ins,
        degron_ins,
        control_norm,
        degron_norm,
        log2_raw_fc,
    )
    boundary_rows = boundary_strength_rows(
        positions,
        boundaries,
        control_norm,
        degron_norm,
        log2_raw_fc,
    )
    boundary_fields = [
        "boundary",
        "nearest_position",
        "control_boundary_strength",
        "degron_boundary_strength",
        "strength_delta_degron_minus_control",
        "raw_insulation_log2fc_degron_over_control",
    ]
    write_rows_csv(
        output_dir / f"boundary_strengths_window{args.window}.csv",
        boundary_rows,
        boundary_fields,
    )
    tad_boundary_rows = tad_boundary_strength_rows(tads, boundary_rows) if tads else []
    if tad_boundary_rows:
        write_rows_csv(
            output_dir / f"tad_boundary_strengths_window{args.window}.csv",
            tad_boundary_rows,
            [
                "tad_label",
                "start",
                "end",
                "boundaries_used",
                "control_boundary_strength_mean",
                "degron_boundary_strength_mean",
                "strength_delta_degron_minus_control",
            ],
        )
        plot_tad_boundary_strengths(
            tad_boundary_rows,
            output_dir / f"tad_boundary_strengths_window{args.window}.png",
            args.window,
        )
    plot_insulation(
        positions,
        control_norm,
        degron_norm,
        log2_raw_fc,
        boundaries,
        output_dir / f"insulation_compare_window{args.window}.png",
        args.window,
    )
    plot_observed_insulation(
        positions,
        control_ins,
        degron_ins,
        log2_raw_fc,
        boundaries,
        output_dir / f"observed_insulation_compare_window{args.window}.png",
        args.window,
        mask_radius,
    )

    summary = {
        "runs_dir": str(runs_dir),
        "output_dir": str(output_dir),
        "map_shape": list(control_oe.shape),
        "window": int(args.window),
        "masked_diagonal_radius": mask_radius,
        "bin_size_bp": int(args.bin_size_bp),
        "boundaries": boundaries,
        "tads": [
            {
                "label": str(tad.get("label", "")),
                "start": int(tad["start"]),
                "end": int(tad["end"]),
            }
            for tad in tads
        ],
        "oe_log2fc": {
            "finite_pixels": int(np.isfinite(log2fc).sum()),
            "mean": float(np.nanmean(log2fc)),
            "median": float(np.nanmedian(log2fc)),
            "p05": finite_quantile(log2fc, 0.05),
            "p95": finite_quantile(log2fc, 0.95),
            "p995_abs": finite_quantile(np.abs(log2fc), 0.995),
        },
        "observed_cis_decay_fold_change": cis_decay_stats,
        "tad_strengths": summarize_tad_strengths(tad_rows) if tad_rows else {},
        "tad_boundary_strengths": {
            "mean_delta_degron_minus_control": float(
                np.nanmean([row["strength_delta_degron_minus_control"] for row in tad_boundary_rows])
            ),
            "median_delta_degron_minus_control": float(
                np.nanmedian([row["strength_delta_degron_minus_control"] for row in tad_boundary_rows])
            ),
        }
        if tad_boundary_rows
        else {},
        "noice_vs_ice": noice_vs_ice_stats,
        "observed_only_log2fc": observed_stats,
        "insulation_raw_log2fc": {
            "mean": float(np.nanmean(log2_raw_fc)),
            "median": float(np.nanmedian(log2_raw_fc)),
            "p05": finite_quantile(log2_raw_fc, 0.05),
            "p95": finite_quantile(log2_raw_fc, 0.95),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
