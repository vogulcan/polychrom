#!/usr/bin/env python3
"""Build observed, ICE O/E, and fold-change maps for several contact cutoffs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from polychrom import contactmaps
from polychrom.hdf5_format import list_URIs
from polychrom.pipelines.loop_extrusion.plugins.sampling import (
    balanced_observed_over_expected,
    iterative_correction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", nargs="?", default="runs", type=Path)
    parser.add_argument("--condition-a", default="control", help="Denominator condition.")
    parser.add_argument("--condition-b", default="degron", help="Numerator condition.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[2, 3, 4, 5, 6])
    parser.add_argument("--nproc", type=int, default=6)
    parser.add_argument("--ice-max-iter", type=int, default=2000)
    parser.add_argument("--ice-tol", type=float, default=1.0e-6)
    parser.add_argument("--ice-ignore-diagonals", type=int, default=2)
    parser.add_argument(
        "--mask-diagonals",
        type=int,
        default=1,
        help="Mask |i-j| <= N in fold-change arrays and plots.",
    )
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--map-starts", nargs="*", type=int, default=None)
    parser.add_argument("--insulation-window", type=int, default=50)
    parser.add_argument("--boundaries", nargs="*", type=int, default=[500, 1000, 1500, 2000])
    parser.add_argument(
        "--heatmap-cmap",
        default="Reds",
        help="Sequential colormap used for observed count heatmaps.",
    )
    parser.add_argument(
        "--delta-cmap",
        default="coolwarm",
        help="Diverging colormap used for O/E log2 and fold-change/delta heatmaps.",
    )
    parser.add_argument("--reuse-existing", action="store_true", help="Load saved observed maps if present.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_chain_layout(run_dir: Path) -> tuple[int, int]:
    with h5py.File(run_dir / "LEFPositions.h5", "r") as fh:
        n_sites = int(fh.attrs["N"])
        chain_length = int(fh.attrs.get("chain_length", n_sites))
        num_chains = int(fh.attrs.get("num_chains", max(1, n_sites // chain_length)))
    return chain_length, num_chains


def effective_map_starts(
    run_dir: Path,
    *,
    map_size: int | None,
    map_starts: list[int] | None,
) -> tuple[int, list[int]]:
    chain_length, num_chains = read_chain_layout(run_dir)
    size = chain_length if map_size is None else int(map_size)
    starts = [0] if map_starts is None else [int(start) for start in map_starts]
    expanded: list[int] = []
    for chain_idx in range(num_chains):
        offset = chain_idx * chain_length
        for start in starts:
            if not (0 <= start and start + size <= chain_length):
                raise ValueError(
                    f"map_start {start} with map_size {size} does not fit inside "
                    f"chain length {chain_length}"
                )
            expanded.append(offset + start)
    return size, expanded


def mask_diagonals(matrix: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(matrix, dtype=float).copy()
    if radius < 0:
        return out
    n = out.shape[0]
    idx = np.arange(n)
    for offset in range(int(radius) + 1):
        rows = idx[: n - offset]
        cols = rows + offset
        out[rows, cols] = np.nan
        if offset:
            out[cols, rows] = np.nan
    return out


def ratio_and_log2(numerator: np.ndarray, denominator: np.ndarray, mask_radius: int) -> tuple[np.ndarray, np.ndarray]:
    if numerator.shape != denominator.shape:
        raise ValueError(f"shape mismatch: {numerator.shape} vs {denominator.shape}")
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > 0)
        & (denominator > 0)
    )
    fc = np.full(numerator.shape, np.nan, dtype=float)
    fc[valid] = numerator[valid] / denominator[valid]
    log2fc = np.full(numerator.shape, np.nan, dtype=float)
    log2fc[valid] = np.log2(fc[valid])
    return mask_diagonals(fc, mask_radius), mask_diagonals(log2fc, mask_radius)


def finite_stats(matrix: np.ndarray) -> dict[str, float]:
    finite = np.asarray(matrix)[np.isfinite(matrix)]
    if finite.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def sample_observed(
    run_dir: Path,
    *,
    cutoff: float,
    map_size: int,
    map_starts: list[int],
    nproc: int,
    verbose: bool,
) -> np.ndarray:
    uris = list_URIs(str(run_dir))
    if not uris:
        raise FileNotFoundError(f"No trajectory blocks found under {run_dir}")
    return contactmaps.monomerResolutionContactMapSubchains(
        filenames=uris,
        mapStarts=map_starts,
        mapN=map_size,
        cutoff=float(cutoff),
        n=int(nproc),
        verbose=verbose,
    )


def save_heatmap(
    panels: list[tuple[str, np.ndarray, str, float | None, float | None, float | None]],
    output_path: Path,
    *,
    title: str,
    mask_radius: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.8), squeeze=False)
    for ax, (label, matrix, cmap_name, vmin, vmax, center) in zip(axes[0], panels):
        data = mask_diagonals(matrix, mask_radius)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e5e7eb")
        imshow_kwargs = {
            "cmap": cmap,
            "interpolation": "nearest",
            "origin": "upper",
        }
        if center is None:
            imshow_kwargs.update({"vmin": vmin, "vmax": vmax})
        else:
            from matplotlib.colors import TwoSlopeNorm

            imshow_kwargs["norm"] = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(data, **imshow_kwargs)
        ax.set_title(label)
        ax.set_xlabel("monomer")
        ax.set_ylabel("monomer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
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


def nearest_boundary_rows(positions: np.ndarray, boundaries: list[int]) -> list[tuple[int, int]]:
    rows = []
    for boundary in boundaries:
        idx = int(np.argmin(np.abs(positions - int(boundary))))
        rows.append((int(boundary), idx))
    return rows


def write_insulation_outputs(
    cutoff_dir: Path,
    label: str,
    condition_a: str,
    condition_b: str,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    window: int,
    boundaries: list[int],
    mask_radius: int,
) -> dict[str, float]:
    positions, scores_a, norm_a = diamond_insulation(mask_diagonals(matrix_a, mask_radius), window)
    positions_b, scores_b, norm_b = diamond_insulation(mask_diagonals(matrix_b, mask_radius), window)
    if not np.array_equal(positions, positions_b):
        raise ValueError("insulation positions differ")

    valid = (scores_a > 0) & (scores_b > 0)
    log2_raw_fc = np.full(scores_a.shape, np.nan, dtype=float)
    log2_raw_fc[valid] = np.log2(scores_b[valid] / scores_a[valid])
    delta_norm = norm_b - norm_a

    with (cutoff_dir / f"{label}_insulation_scores_window{window}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "position",
            f"{condition_a}_insulation",
            f"{condition_b}_insulation",
            f"{condition_a}_log2_over_mean",
            f"{condition_b}_log2_over_mean",
            f"{condition_b}_minus_{condition_a}_log2_over_mean",
            f"log2_{condition_b}_over_{condition_a}_raw",
        ])
        for row in zip(positions, scores_a, scores_b, norm_a, norm_b, delta_norm, log2_raw_fc):
            writer.writerow([int(row[0]), *[float(x) if np.isfinite(x) else "" for x in row[1:]]])

    boundary_delta = []
    boundary_raw_fc = []
    with (cutoff_dir / f"{label}_insulation_boundary_summary_window{window}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "boundary",
            "nearest_position",
            f"{condition_a}_insulation",
            f"{condition_b}_insulation",
            f"{condition_a}_log2_over_mean",
            f"{condition_b}_log2_over_mean",
            f"{condition_b}_minus_{condition_a}_log2_over_mean",
            f"log2_{condition_b}_over_{condition_a}_raw",
        ])
        for boundary, idx in nearest_boundary_rows(positions, boundaries):
            boundary_delta.append(delta_norm[idx])
            boundary_raw_fc.append(log2_raw_fc[idx])
            writer.writerow([
                boundary,
                int(positions[idx]),
                float(scores_a[idx]),
                float(scores_b[idx]),
                float(norm_a[idx]),
                float(norm_b[idx]),
                float(delta_norm[idx]),
                float(log2_raw_fc[idx]),
            ])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(positions, norm_a, lw=1.2, color="#2563eb", label=condition_a)
    axes[0].plot(positions, norm_b, lw=1.2, color="#dc2626", label=condition_b)
    axes[0].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    axes[0].set_ylabel("log2(insulation / condition mean)")
    axes[0].set_title(f"{label} diamond insulation, window {window}")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(positions, log2_raw_fc, lw=1.1, color="#111827")
    axes[1].axhline(0.0, color="black", lw=0.8, alpha=0.35)
    axes[1].set_ylabel(f"log2({condition_b}/{condition_a})")
    axes[1].set_xlabel("monomer")
    for boundary in boundaries:
        for ax in axes:
            ax.axvline(boundary, color="black", lw=0.7, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(cutoff_dir / f"{label}_insulation_compare_window{window}.png", dpi=170)
    plt.close(fig)

    boundary_delta = np.asarray(boundary_delta, dtype=float)
    boundary_raw_fc = np.asarray(boundary_raw_fc, dtype=float)
    return {
        f"{label}_insulation_delta_mean": float(np.nanmean(delta_norm)),
        f"{label}_insulation_delta_median": float(np.nanmedian(delta_norm)),
        f"{label}_boundary_delta_mean": float(np.nanmean(boundary_delta)),
        f"{label}_boundary_delta_median": float(np.nanmedian(boundary_delta)),
        f"{label}_boundary_raw_log2fc_mean": float(np.nanmean(boundary_raw_fc)),
        f"{label}_boundary_raw_log2fc_median": float(np.nanmedian(boundary_raw_fc)),
    }


def build_tads(map_size: int, boundaries: list[int]) -> list[dict[str, object]]:
    edges = [0, *sorted(int(b) for b in boundaries if 0 < int(b) < map_size), map_size]
    return [
        {"label": f"T{i}", "start": start, "end": end, "size": end - start}
        for i, (start, end) in enumerate(zip(edges[:-1], edges[1:]))
    ]


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
    matrix_label: str,
    condition_a: str,
    condition_b: str,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    tads: list[dict[str, object]],
) -> list[dict[str, object]]:
    n = matrix_a.shape[0]
    rows = []
    for tad in tads:
        start = int(tad["start"])
        end = int(tad["end"])
        size = int(tad.get("size", end - start))
        left_start = max(0, start - size)
        right_end = min(n, end + size)
        flank_regions = [
            (start, end, left_start, start),
            (start, end, end, right_end),
        ]

        a_intra = region_mean(matrix_a, start, end, start, end)
        a_flank = region_pair_mean(matrix_a, flank_regions)
        b_intra = region_mean(matrix_b, start, end, start, end)
        b_flank = region_pair_mean(matrix_b, flank_regions)
        a_strength = safe_log2_ratio(a_intra, a_flank)
        b_strength = safe_log2_ratio(b_intra, b_flank)
        rows.append({
            "matrix": matrix_label,
            "tad_label": str(tad["label"]),
            "start": start,
            "end": end,
            "flank_size": size,
            f"{condition_a}_intra_mean": a_intra,
            f"{condition_a}_flank_mean": a_flank,
            f"{condition_a}_strength_log2": a_strength,
            f"{condition_b}_intra_mean": b_intra,
            f"{condition_b}_flank_mean": b_flank,
            f"{condition_b}_strength_log2": b_strength,
            f"strength_delta_{condition_b}_minus_{condition_a}": (
                b_strength - a_strength if np.isfinite(a_strength) and np.isfinite(b_strength) else float("nan")
            ),
            f"intra_log2fc_{condition_b}_over_{condition_a}": safe_log2_ratio(b_intra, a_intra),
            f"flank_log2fc_{condition_b}_over_{condition_a}": safe_log2_ratio(b_flank, a_flank),
        })
    return rows


def write_tad_strength_outputs(
    cutoff_dir: Path,
    condition_a: str,
    condition_b: str,
    matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    boundaries: list[int],
    mask_radius: int,
) -> dict[str, float]:
    map_size = next(iter(matrices.values()))[0].shape[0]
    tads = build_tads(map_size, boundaries)
    rows: list[dict[str, object]] = []
    for label, (matrix_a, matrix_b) in matrices.items():
        rows.extend(tad_strength_rows(
            label,
            condition_a,
            condition_b,
            mask_diagonals(matrix_a, mask_radius),
            mask_diagonals(matrix_b, mask_radius),
            tads,
        ))

    delta_key = f"strength_delta_{condition_b}_minus_{condition_a}"
    fields = [
        "matrix",
        "tad_label",
        "start",
        "end",
        "flank_size",
        f"{condition_a}_intra_mean",
        f"{condition_a}_flank_mean",
        f"{condition_a}_strength_log2",
        f"{condition_b}_intra_mean",
        f"{condition_b}_flank_mean",
        f"{condition_b}_strength_log2",
        delta_key,
        f"intra_log2fc_{condition_b}_over_{condition_a}",
        f"flank_log2fc_{condition_b}_over_{condition_a}",
    ]
    with (cutoff_dir / "tad_strengths.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: "" if isinstance(row.get(field), float) and not np.isfinite(row[field]) else row.get(field, "")
                for field in fields
            })

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix_labels = list(matrices)
    tad_labels = [str(tad["label"]) for tad in tads]
    fig, axes = plt.subplots(len(matrix_labels), 2, figsize=(13, 3.4 * len(matrix_labels)), squeeze=False)
    x = np.arange(len(tad_labels))
    width = 0.38
    summary: dict[str, float] = {}
    for row_idx, label in enumerate(matrix_labels):
        subset = [row for row in rows if row["matrix"] == label]
        by_tad = {str(row["tad_label"]): row for row in subset}
        a_strength = np.array([by_tad[t][f"{condition_a}_strength_log2"] for t in tad_labels], dtype=float)
        b_strength = np.array([by_tad[t][f"{condition_b}_strength_log2"] for t in tad_labels], dtype=float)
        delta = np.array([by_tad[t][delta_key] for t in tad_labels], dtype=float)
        summary[f"{label}_tad_strength_delta_mean"] = float(np.nanmean(delta))
        summary[f"{label}_tad_strength_delta_median"] = float(np.nanmedian(delta))

        ax = axes[row_idx, 0]
        ax.bar(x - width / 2, a_strength, width, color="#2563eb", label=condition_a)
        ax.bar(x + width / 2, b_strength, width, color="#dc2626", label=condition_b)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.35)
        ax.set_title(f"{label}: TAD strength")
        ax.set_ylabel("log2(intra / flank)")
        ax.set_xticks(x)
        ax.set_xticklabels(tad_labels)
        ax.legend(frameon=False)

        ax = axes[row_idx, 1]
        colors = np.where(delta >= 0, "#dc2626", "#2563eb")
        ax.bar(x, delta, color=colors)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.35)
        ax.set_title(f"{label}: {condition_b} - {condition_a}")
        ax.set_ylabel("delta log2 strength")
        ax.set_xticks(x)
        ax.set_xticklabels(tad_labels)

    fig.tight_layout()
    fig.savefig(cutoff_dir / "tad_strengths.png", dpi=170)
    plt.close(fig)
    return summary


def save_cutoff_outputs(
    cutoff_dir: Path,
    cutoff: float,
    condition_a: str,
    condition_b: str,
    observed_a: np.ndarray,
    observed_b: np.ndarray,
    *,
    ice_max_iter: int,
    ice_tol: float,
    ice_ignore_diagonals: int,
    mask_radius: int,
    insulation_window: int,
    boundaries: list[int],
    heatmap_cmap: str,
    delta_cmap: str,
) -> dict[str, object]:
    cutoff_dir.mkdir(parents=True, exist_ok=True)

    np.save(cutoff_dir / f"{condition_a}_observed.npy", observed_a)
    np.save(cutoff_dir / f"{condition_b}_observed.npy", observed_b)

    ice_observed_a = iterative_correction(
        observed_a,
        max_iter=ice_max_iter,
        tol=ice_tol,
        ignore_diagonals=ice_ignore_diagonals,
    )
    ice_observed_b = iterative_correction(
        observed_b,
        max_iter=ice_max_iter,
        tol=ice_tol,
        ignore_diagonals=ice_ignore_diagonals,
    )
    ice_oe_a = balanced_observed_over_expected(
        observed_a,
        max_iter=ice_max_iter,
        tol=ice_tol,
        ignore_diagonals=ice_ignore_diagonals,
    )
    ice_oe_b = balanced_observed_over_expected(
        observed_b,
        max_iter=ice_max_iter,
        tol=ice_tol,
        ignore_diagonals=ice_ignore_diagonals,
    )

    np.save(cutoff_dir / f"{condition_a}_ice_observed.npy", ice_observed_a)
    np.save(cutoff_dir / f"{condition_b}_ice_observed.npy", ice_observed_b)
    np.save(cutoff_dir / f"{condition_a}_ice_oe.npy", ice_oe_a)
    np.save(cutoff_dir / f"{condition_b}_ice_oe.npy", ice_oe_b)

    obs_fc, obs_log2fc = ratio_and_log2(observed_b, observed_a, mask_radius)
    ice_obs_fc, ice_obs_log2fc = ratio_and_log2(ice_observed_b, ice_observed_a, mask_radius)
    ice_oe_fc, ice_oe_log2fc = ratio_and_log2(ice_oe_b, ice_oe_a, mask_radius)

    np.save(cutoff_dir / f"{condition_b}_over_{condition_a}_observed_fc.npy", obs_fc)
    np.save(cutoff_dir / f"{condition_b}_over_{condition_a}_observed_log2fc.npy", obs_log2fc)
    np.save(cutoff_dir / f"{condition_b}_over_{condition_a}_ice_observed_fc.npy", ice_obs_fc)
    np.save(cutoff_dir / f"{condition_b}_over_{condition_a}_ice_observed_log2fc.npy", ice_obs_log2fc)
    np.save(cutoff_dir / f"{condition_b}_over_{condition_a}_ice_oe_fc.npy", ice_oe_fc)
    np.save(cutoff_dir / f"{condition_b}_over_{condition_a}_ice_oe_log2fc.npy", ice_oe_log2fc)

    with np.errstate(divide="ignore", invalid="ignore"):
        obs_log_a = np.log10(observed_a + 1)
        obs_log_b = np.log10(observed_b + 1)
        oe_log_a = np.log2(ice_oe_a)
        oe_log_b = np.log2(ice_oe_b)

    save_heatmap(
        [
            (f"{condition_a} obs log10(count+1)", obs_log_a, heatmap_cmap, None, None, None),
            (f"{condition_b} obs log10(count+1)", obs_log_b, heatmap_cmap, None, None, None),
            (f"{condition_b}/{condition_a} obs log2FC", obs_log2fc, delta_cmap, -1.5, 1.5, 0.0),
        ],
        cutoff_dir / "observed_maps_and_fc.png",
        title=f"Observed maps, cutoff {cutoff:g}",
        mask_radius=mask_radius,
    )
    save_heatmap(
        [
            (f"{condition_a} ICE O/E log2", oe_log_a, delta_cmap, -1.5, 1.5, 0.0),
            (f"{condition_b} ICE O/E log2", oe_log_b, delta_cmap, -1.5, 1.5, 0.0),
            (f"{condition_b}/{condition_a} ICE O/E log2FC", ice_oe_log2fc, delta_cmap, -1.5, 1.5, 0.0),
        ],
        cutoff_dir / "ice_oe_maps_and_fc.png",
        title=f"ICE observed/expected maps, cutoff {cutoff:g}",
        mask_radius=mask_radius,
    )
    save_heatmap(
        [
            (f"{condition_b}/{condition_a} observed log2FC", obs_log2fc, delta_cmap, -1.5, 1.5, 0.0),
            (f"{condition_b}/{condition_a} ICE observed log2FC", ice_obs_log2fc, delta_cmap, -1.5, 1.5, 0.0),
            (f"{condition_b}/{condition_a} ICE O/E log2FC", ice_oe_log2fc, delta_cmap, -1.5, 1.5, 0.0),
        ],
        cutoff_dir / "fold_changes.png",
        title=f"Fold changes, cutoff {cutoff:g}",
        mask_radius=mask_radius,
    )

    observed_insulation = write_insulation_outputs(
        cutoff_dir,
        "observed",
        condition_a,
        condition_b,
        observed_a,
        observed_b,
        window=insulation_window,
        boundaries=boundaries,
        mask_radius=mask_radius,
    )
    ice_oe_insulation = write_insulation_outputs(
        cutoff_dir,
        "ice_oe",
        condition_a,
        condition_b,
        ice_oe_a,
        ice_oe_b,
        window=insulation_window,
        boundaries=boundaries,
        mask_radius=mask_radius,
    )
    tad_strength_summary = write_tad_strength_outputs(
        cutoff_dir,
        condition_a,
        condition_b,
        {
            "observed": (observed_a, observed_b),
            "ice_observed": (ice_observed_a, ice_observed_b),
            "ice_oe": (ice_oe_a, ice_oe_b),
        },
        boundaries=boundaries,
        mask_radius=mask_radius,
    )

    return {
        "cutoff": float(cutoff),
        "observed_total_a": float(np.nansum(observed_a)),
        "observed_total_b": float(np.nansum(observed_b)),
        "observed_log2fc": finite_stats(obs_log2fc),
        "ice_observed_log2fc": finite_stats(ice_obs_log2fc),
        "ice_oe_log2fc": finite_stats(ice_oe_log2fc),
        **observed_insulation,
        **ice_oe_insulation,
        **tad_strength_summary,
    }


def write_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    with (output_dir / "summary.json").open("w") as fh:
        json.dump(rows, fh, indent=2)

    fields = [
        "cutoff",
        "observed_total_a",
        "observed_total_b",
        "observed_log2fc_mean",
        "observed_log2fc_median",
        "ice_observed_log2fc_mean",
        "ice_observed_log2fc_median",
        "ice_oe_log2fc_mean",
        "ice_oe_log2fc_median",
        "observed_boundary_delta_mean",
        "observed_boundary_delta_median",
        "observed_boundary_raw_log2fc_mean",
        "ice_oe_boundary_delta_mean",
        "ice_oe_boundary_delta_median",
        "ice_oe_boundary_raw_log2fc_mean",
        "observed_tad_strength_delta_mean",
        "observed_tad_strength_delta_median",
        "ice_observed_tad_strength_delta_mean",
        "ice_observed_tad_strength_delta_median",
        "ice_oe_tad_strength_delta_mean",
        "ice_oe_tad_strength_delta_median",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "cutoff": row["cutoff"],
                "observed_total_a": row["observed_total_a"],
                "observed_total_b": row["observed_total_b"],
                "observed_log2fc_mean": row["observed_log2fc"]["mean"],
                "observed_log2fc_median": row["observed_log2fc"]["median"],
                "ice_observed_log2fc_mean": row["ice_observed_log2fc"]["mean"],
                "ice_observed_log2fc_median": row["ice_observed_log2fc"]["median"],
                "ice_oe_log2fc_mean": row["ice_oe_log2fc"]["mean"],
                "ice_oe_log2fc_median": row["ice_oe_log2fc"]["median"],
                "observed_boundary_delta_mean": row["observed_boundary_delta_mean"],
                "observed_boundary_delta_median": row["observed_boundary_delta_median"],
                "observed_boundary_raw_log2fc_mean": row["observed_boundary_raw_log2fc_mean"],
                "ice_oe_boundary_delta_mean": row["ice_oe_boundary_delta_mean"],
                "ice_oe_boundary_delta_median": row["ice_oe_boundary_delta_median"],
                "ice_oe_boundary_raw_log2fc_mean": row["ice_oe_boundary_raw_log2fc_mean"],
                "observed_tad_strength_delta_mean": row["observed_tad_strength_delta_mean"],
                "observed_tad_strength_delta_median": row["observed_tad_strength_delta_median"],
                "ice_observed_tad_strength_delta_mean": row["ice_observed_tad_strength_delta_mean"],
                "ice_observed_tad_strength_delta_median": row["ice_observed_tad_strength_delta_median"],
                "ice_oe_tad_strength_delta_mean": row["ice_oe_tad_strength_delta_mean"],
                "ice_oe_tad_strength_delta_median": row["ice_oe_tad_strength_delta_median"],
            })


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.runs_dir / "degron_control_contact_cutoffs")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_a = args.runs_dir / args.condition_a
    run_b = args.runs_dir / args.condition_b
    map_size_a, starts_a = effective_map_starts(
        run_a,
        map_size=args.map_size,
        map_starts=args.map_starts,
    )
    map_size_b, starts_b = effective_map_starts(
        run_b,
        map_size=args.map_size,
        map_starts=args.map_starts,
    )
    if map_size_a != map_size_b or starts_a != starts_b:
        raise ValueError("conditions must use same map size and starts")

    metadata = {
        "runs_dir": str(args.runs_dir),
        "condition_a": args.condition_a,
        "condition_b": args.condition_b,
        "cutoffs": [float(c) for c in args.cutoffs],
        "nproc": int(args.nproc),
        "map_size": int(map_size_a),
        "map_starts": starts_a,
        "ice_max_iter": int(args.ice_max_iter),
        "ice_tol": float(args.ice_tol),
        "ice_ignore_diagonals": int(args.ice_ignore_diagonals),
        "mask_diagonals": int(args.mask_diagonals),
        "insulation_window": int(args.insulation_window),
        "boundaries": [int(b) for b in args.boundaries],
        "heatmap_cmap": args.heatmap_cmap,
        "delta_cmap": args.delta_cmap,
    }
    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)

    rows = []
    for cutoff in args.cutoffs:
        cutoff_label = str(float(cutoff)).replace(".", "p")
        cutoff_dir = output_dir / f"cutoff_{cutoff_label}"
        observed_a_path = cutoff_dir / f"{args.condition_a}_observed.npy"
        observed_b_path = cutoff_dir / f"{args.condition_b}_observed.npy"
        if args.reuse_existing and observed_a_path.exists() and observed_b_path.exists():
            print(f"[cutoff {cutoff:g}] loading existing observed maps", flush=True)
            observed_a = np.load(observed_a_path)
            observed_b = np.load(observed_b_path)
        else:
            print(f"[cutoff {cutoff:g}] sampling {args.condition_a}", flush=True)
            observed_a = sample_observed(
                run_a,
                cutoff=cutoff,
                map_size=map_size_a,
                map_starts=starts_a,
                nproc=args.nproc,
                verbose=args.verbose,
            )
            print(f"[cutoff {cutoff:g}] sampling {args.condition_b}", flush=True)
            observed_b = sample_observed(
                run_b,
                cutoff=cutoff,
                map_size=map_size_a,
                map_starts=starts_a,
                nproc=args.nproc,
                verbose=args.verbose,
            )
        print(f"[cutoff {cutoff:g}] ICE + fold changes", flush=True)
        row = save_cutoff_outputs(
            cutoff_dir,
            cutoff,
            args.condition_a,
            args.condition_b,
            observed_a,
            observed_b,
            ice_max_iter=args.ice_max_iter,
            ice_tol=args.ice_tol,
            ice_ignore_diagonals=args.ice_ignore_diagonals,
            mask_radius=args.mask_diagonals,
            insulation_window=args.insulation_window,
            boundaries=[int(b) for b in args.boundaries],
            heatmap_cmap=args.heatmap_cmap,
            delta_cmap=args.delta_cmap,
        )
        rows.append(row)
        write_summary(output_dir, rows)

    print(f"[done] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
