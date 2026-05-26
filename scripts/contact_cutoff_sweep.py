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
    panels: list[tuple[str, np.ndarray, str, float | None, float | None]],
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
    for ax, (label, matrix, cmap_name, vmin, vmax) in zip(axes[0], panels):
        data = mask_diagonals(matrix, mask_radius)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e5e7eb")
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
        ax.set_title(label)
        ax.set_xlabel("monomer")
        ax.set_ylabel("monomer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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
            (f"{condition_a} obs log10(count+1)", obs_log_a, "viridis", None, None),
            (f"{condition_b} obs log10(count+1)", obs_log_b, "viridis", None, None),
            (f"{condition_b}/{condition_a} obs log2FC", obs_log2fc, "coolwarm", -1.5, 1.5),
        ],
        cutoff_dir / "observed_maps_and_fc.png",
        title=f"Observed maps, cutoff {cutoff:g}",
        mask_radius=mask_radius,
    )
    save_heatmap(
        [
            (f"{condition_a} ICE O/E log2", oe_log_a, "coolwarm", -1.5, 1.5),
            (f"{condition_b} ICE O/E log2", oe_log_b, "coolwarm", -1.5, 1.5),
            (f"{condition_b}/{condition_a} ICE O/E log2FC", ice_oe_log2fc, "coolwarm", -1.5, 1.5),
        ],
        cutoff_dir / "ice_oe_maps_and_fc.png",
        title=f"ICE observed/expected maps, cutoff {cutoff:g}",
        mask_radius=mask_radius,
    )
    save_heatmap(
        [
            (f"{condition_b}/{condition_a} observed log2FC", obs_log2fc, "coolwarm", -1.5, 1.5),
            (f"{condition_b}/{condition_a} ICE observed log2FC", ice_obs_log2fc, "coolwarm", -1.5, 1.5),
            (f"{condition_b}/{condition_a} ICE O/E log2FC", ice_oe_log2fc, "coolwarm", -1.5, 1.5),
        ],
        cutoff_dir / "fold_changes.png",
        title=f"Fold changes, cutoff {cutoff:g}",
        mask_radius=mask_radius,
    )

    return {
        "cutoff": float(cutoff),
        "observed_total_a": float(np.nansum(observed_a)),
        "observed_total_b": float(np.nansum(observed_b)),
        "observed_log2fc": finite_stats(obs_log2fc),
        "ice_observed_log2fc": finite_stats(ice_obs_log2fc),
        "ice_oe_log2fc": finite_stats(ice_oe_log2fc),
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
    }
    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)

    rows = []
    for cutoff in args.cutoffs:
        cutoff_label = str(float(cutoff)).replace(".", "p")
        cutoff_dir = output_dir / f"cutoff_{cutoff_label}"
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
        )
        rows.append(row)
        write_summary(output_dir, rows)

    print(f"[done] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
