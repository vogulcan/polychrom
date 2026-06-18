#!/usr/bin/env python
"""Grid search over Type A fraction x lesion density at a FIXED cohesin block prob.

Sister script to ``gen_lesion_grid_and_heatmaps.py``. There the swept knob is
``lesion_block_prob`` (cohesin stall strength); here that is held STATIC
(``--block-prob``) and the two grid axes are instead:

  Y -- ``lesion_type_a_prob``: P(a gene-body lesion is Type A / TC-NER). Type A
       lesions block cohesin already during pre-recognition (a long ~1200 s
       window, with RNAPII off the lesion does the stalling intrinsically); Type
       B only blocks once in the REPAIR state. Sweeping this dials the lesion
       population from all-GG-NER (B) to all-TC-NER (A). Off-gene lesions are
       always Type B regardless.
  X -- lesion DENSITY in lesions/Mbp, set via ``lesion_spacing`` (1 monomer =
       1 kb so density = 1000/spacing), exactly as in the sister script.

Every config otherwise matches the sister script's transform: RNAPII fully off
(``max_rnapii: 0``, ``rnapii_load``/``rnapii_translocate`` nulled), boundary
strengths x ``--bstr-mult`` capped at 1.0, and the lesion plugin enabled. For
each grid point the 1D LEF stage is run (or a matching H5 reused), and the same
outputs are produced with the y-axis relabelled to Type A probability:

  * ``ta_density_heatmaps``               -- per metric, grid mean / BASELINE mean.
  * ``ta_density_measured_stall``         -- per-lesion stall occupancy (measured).
  * ``ta_density_measured_stall_seconds`` -- overall stall duration mean+median (s).
  * ``ta_density_measured_stall_by_type`` -- A-PRE / A-repair / B-repair mean (s),
    the regime that this sweep is built to move.

All measurement / plotting / config-building is imported from the sister module;
this file only swaps which variable is swept.
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported (see sister script).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gen_lesion_grid_and_heatmaps as gg
from gen_lesion_grid_and_heatmaps import (
    DEFAULT_TAD_REPAIR_EXPONENT,
    DEFAULT_TAD_SIZE_EXPONENT,
    LESION_PRERECOGNITION_SECONDS,
    LESION_REPAIR_SECONDS,
    MEASURED_BYTYPE_LABELS,
    MEASURED_LABELS,
    MEASURED_SECONDS_LABELS,
    MEASURED_STAGEFRAC_LABELS,
    METRIC_LABELS,
    SEQ_CMAP,
    _CATEGORIES,
    _LES_TYPE_A,
    _p_values,
    _run_grid_point,
    _safe_label,
    _topology,
    analyze_h5,
    build_grid_config,
    density_per_mbp,
    effective_stall_seconds,
    gene_body_mass,
    lesion_density_axis,
    measure_lesion_stall,
    plot_heatmaps,
)


def ta_label(type_a: float, spacing: int) -> str:
    return _safe_label(f"grid_ta{type_a:.3f}_s{spacing}")


# Measured (from the H5 trajectory) Type-A lesion burden inside gene bodies.
TYPEA_PERKB_KEY = "typea_per_kb_genebody"
TYPEA_PERKB_LABELS = {
    TYPEA_PERKB_KEY: "Type-A lesions per kb of gene body\n(measured from trajectory)",
}
# Same readout inverted to a spacing (kb of gene body per Type-A lesion).
TYPEA_KB_PER_LESION_LABELS = {
    TYPEA_PERKB_KEY: "kb of gene body per Type-A lesion\n(measured from trajectory)",
}


def gene_body_mask(base: dict) -> np.ndarray:
    """Boolean lattice mask (length N) marking gene-body sites for the baseline
    genes + TAD layout (replicated across chains when configured). Mirrors the mask
    inside ``gene_body_mass``. 1 monomer = 1 kb, so ``mask.sum()`` is the total
    gene-body length in kb."""
    lef = base["lef"]
    tk = lef.get("topology_kwargs", {})
    chain_length = int(lef["chain_length"])
    num_chains = int(lef["num_chains"])
    N = chain_length * num_chains
    mask = np.zeros(N, dtype=bool)
    replicate = bool(tk.get("replicate_genes_across_chains", False))
    offsets = [c * chain_length for c in range(num_chains)] if replicate else [0]
    for g in tk.get("genes") or []:
        tss, tes = int(g["tss"]), int(g["tes"])
        lo, hi = (tss, tes) if tss <= tes else (tes, tss)
        for off in offsets:
            a, b = off + lo, off + hi
            if 0 <= a < N:
                mask[a: min(b + 1, N)] = True
    return mask


def measure_typea_per_kb_genebody(
    h5_path: Path, gb_mask: np.ndarray, *, chunk: int = 2000
) -> float:
    """Mean number of Type-A lesions present inside gene bodies, per kb of gene
    body, measured DIRECTLY from the 1D trajectory (time-averaged over frames).

    This is the *realised* Type-A burden a traversing cohesin sees -- it reflects
    the lesion dynamics (placement, recognition, repair/removal) rather than the
    analytic placement density ``P(A) x density``. Type-A lesions live only in gene
    bodies, but we restrict to the gene-body mask explicitly and divide by the total
    gene-body length (1 monomer = 1 kb), so the result is normalised over gene
    bodies. Returns Type-A lesions / kb of gene body (NaN if no gene bodies or the
    H5 has no lesion datasets)."""
    gb_kb = int(gb_mask.sum())
    if gb_kb <= 0:
        return float("nan")
    with h5py.File(h5_path, "r") as h:
        if "lesions" not in h:
            return float("nan")
        N = int(h.attrs["N"])
        ds_sites, ds_types = h["lesions"], h["lesion_types"]
        frames = int(ds_sites.shape[0])
        if frames <= 0:
            return float("nan")
        total = 0
        for start in range(0, frames, chunk):
            end = min(start + chunk, frames)
            sites = ds_sites[start:end]
            types = ds_types[start:end]
            for k in range(end - start):
                s, ty = sites[k], types[k]
                valid = (s >= 0) & (s < N) & (ty == _LES_TYPE_A)
                ss = s[valid]
                if ss.size:
                    total += int(gb_mask[ss].sum())
        return total / frames / gb_kb


def _measured_keys() -> list[str]:
    keys = [
        "stalled_per_lesion", "stalled_per_blocking_lesion",
        "stalled_legs_per_frame", "lesions_per_frame", "blocking_lesions_per_frame",
        "dwell_mean_s", "dwell_median_s", "dwell_p90_s", "dwell_max_s",
        "n_stall_events", "stalled_legtick_fraction",
    ]
    for cat in _CATEGORIES:
        keys += [f"{cat}_dwell_mean_s", f"{cat}_dwell_median_s",
                 f"{cat}_dwell_p90_s", f"{cat}_dwell_max_s",
                 f"{cat}_n_events", f"{cat}_legtick_fraction"]
    keys += list(MEASURED_STAGEFRAC_LABELS)
    return keys


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=Path, required=True, help="baseline config YAML")
    ap.add_argument("--h5", type=Path, required=True, help="baseline LEFPositions.h5 for --config")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory (configs, h5s, heatmaps)")
    ap.add_argument("--block-prob", type=float, default=0.97,
                    help="STATIC lesion_block_prob applied to every grid config (cohesin stall strength)")
    ap.add_argument("--bstr-mult", type=float, default=3.0, help="boundary-strength multiplier (capped at 1.0)")
    ap.add_argument("--lesion-type-b-enabled", action=argparse.BooleanOptionalAction, default=True,
                    help="include Type-B (GG-NER) lesions (default). --no-lesion-type-b-enabled keeps only "
                         "Type-A lesions; the x-axis then uses the gene-body (Type-A) density, scaled per row "
                         "by P(A).")
    ap.add_argument(
        "--prerecognition-seconds", type=float, default=None,
        help="lesion pre-recognition window in SECONDS (PRE->REPAIR). Converted to "
             "lesion_prerecognition_ticks via round(seconds / tick_seconds). Default: "
             "inherit from --config, else ~1200s.",
    )
    ap.add_argument(
        "--repair-seconds", type=float, default=None,
        help="lesion repair window in SECONDS (REPAIR->removed; the window in which a "
             "lesion blocks cohesin). Converted to lesion_repair_ticks via "
             "round(seconds / tick_seconds). Default: inherit from --config, else ~300s.",
    )
    ap.add_argument("--ta-hi", type=float, default=1.0, help="highest lesion_type_a_prob (top row)")
    ap.add_argument("--ta-lo", type=float, default=0.0, help="lowest lesion_type_a_prob (bottom row)")
    ap.add_argument("--ta-step", type=float, default=0.2, help="lesion_type_a_prob increment")
    ap.add_argument(
        "--spacings", type=int, nargs="+",
        default=[10, 25, 50, 100, 150, 200],
        help="lesion_spacing values (columns); density = 1000/spacing lesions/Mbp",
    )
    ap.add_argument("--skip-run", action="store_true",
                    help="only generate the grid configs; do not run 1D or plot")
    ap.add_argument("--force-1d", action="store_true",
                    help="rerun the 1D LEF stage for every grid point even if a matching H5 exists")
    ap.add_argument("--jobs", type=int, default=1,
                    help="number of grid-point 1D sims to run concurrently (process pool, fork)")
    ap.add_argument("--no-measure", action="store_true",
                    help="skip the measured lesion-stall heatmaps (only emit the fold heatmaps)")
    ap.add_argument("--measure-chunk", type=int, default=2000,
                    help="frames per streamed block when measuring lesion stalls from the H5")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    cfg_dir = out_dir / "configs"
    h5_dir = out_dir / "h5"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(args.config.read_text())
    lef = base["lef"]
    tick_seconds = float(lef.get("tick_seconds") or 0.0)
    if tick_seconds <= 0:
        raise SystemExit("config lef.tick_seconds must be a positive number")

    tk = lef["topology_kwargs"]
    lesion_defaults = {
        "prerecognition_ticks": tk.get(
            "lesion_prerecognition_ticks", max(1, int(round(LESION_PRERECOGNITION_SECONDS / tick_seconds)))
        ),
        "repair_ticks": tk.get(
            "lesion_repair_ticks", max(1, int(round(LESION_REPAIR_SECONDS / tick_seconds)))
        ),
        "tad_size_exponent": tk.get("lesion_tad_size_exponent", DEFAULT_TAD_SIZE_EXPONENT),
        "tad_repair_exponent": tk.get("lesion_tad_repair_exponent", DEFAULT_TAD_REPAIR_EXPONENT),
    }
    # Recognition/repair may be given in SECONDS on the CLI; convert to ticks here
    # (tick number = round(seconds / tick_seconds), floored at 1 tick).
    if args.prerecognition_seconds is not None:
        lesion_defaults["prerecognition_ticks"] = max(1, int(round(args.prerecognition_seconds / tick_seconds)))
    if args.repair_seconds is not None:
        lesion_defaults["repair_ticks"] = max(1, int(round(args.repair_seconds / tick_seconds)))

    block_prob = float(args.block_prob)
    # Axis orderings: ascending Type A (rows) and ascending density (cols).
    ta_sorted = sorted(_p_values(args.ta_hi, args.ta_lo, args.ta_step))
    sp_sorted = sorted(set(args.spacings), reverse=True)
    # X-axis density: total (B on). With B off, P(A) is the y-axis here, so the
    # per-column x-axis is the gene-body (Type-A-eligible) density (= Type-A density
    # at P(A)=1); a cell's actual Type-A density is x_density x its row's P(A).
    type_b_enabled = bool(args.lesion_type_b_enabled)
    gbm = gene_body_mass(base, float(lesion_defaults["tad_size_exponent"])) if not type_b_enabled else 1.0
    # Gene-body mask + length (kb) for the measured Type-A-per-kb-of-gene-body readout.
    gb_mask = gene_body_mask(base)
    gb_kb = int(gb_mask.sum())
    density_axis = lesion_density_axis(
        sp_sorted, type_b_enabled=type_b_enabled, gbm=gbm, type_a_prob=1.0)
    x_label = ("Lesion density (lesions/Mbp)" if type_b_enabled
               else "Gene-body (Type-A) lesion density (lesions/Mbp)")
    # Analytic effective stall is fixed (block_prob is static) -- a single number.
    eff_stall = effective_stall_seconds(
        block_prob, tick_seconds, float(lesion_defaults["repair_ticks"]),
        float(lef.get("lifetime_stalled") or lef.get("lifetime") or 0) or None,
    )

    print(f"baseline config : {args.config}")
    print(f"baseline H5     : {args.h5}")
    print(f"tick_seconds    : {tick_seconds:g}")
    print(f"STATIC block_prob : {block_prob:g}  (effective stall ~ {eff_stall:.0f}s, fixed across grid)")
    print(f"grid            : {len(ta_sorted)} type_a x {len(sp_sorted)} density = "
          f"{len(ta_sorted) * len(sp_sorted)} configs")
    print(f"  lesion_type_a_prob : {ta_sorted}")
    print(f"  spacing            : {sp_sorted}")
    if type_b_enabled:
        print(f"  density /Mbp       : {[round(d, 2) for d in density_axis]}  (total; Type-B on)")
    else:
        print(f"  Type-B DISABLED -> x-axis = gene-body (Type-A) density "
              f"(gene_body_mass={gbm:.3f}); per-cell Type-A density = x * row P(A)")
        print(f"  gene-body density /Mbp: {[round(d, 2) for d in density_axis]}")
    prerec_ticks = int(lesion_defaults["prerecognition_ticks"])
    repair_t = int(lesion_defaults["repair_ticks"])
    psrc = "CLI(s)" if args.prerecognition_seconds is not None else "config"
    rsrc = "CLI(s)" if args.repair_seconds is not None else "config"
    print(f"  lesion_prerecognition_ticks: {prerec_ticks} (~{prerec_ticks * tick_seconds:.0f}s) [{psrc}]")
    print(f"  lesion_repair_ticks       : {repair_t} (~{repair_t * tick_seconds:.0f}s) [{rsrc}]")
    print(f"  boundary x{args.bstr_mult:g} (cap 1.0), max_rnapii=0, rnapii_load/translocate=null")

    # 1) generate every grid config ---------------------------------------
    grid_cfg_paths: dict[tuple[float, int], Path] = {}
    for ta in ta_sorted:
        for sp in sp_sorted:
            cfg = build_grid_config(
                base, p=block_prob, spacing=sp, bstr_mult=args.bstr_mult,
                type_a_prob=ta, tick_seconds=tick_seconds, lesion_defaults=lesion_defaults,
                type_b_enabled=type_b_enabled,
            )
            path = cfg_dir / f"{ta_label(ta, sp)}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
            grid_cfg_paths[(ta, sp)] = path
    print(f"wrote {len(grid_cfg_paths)} grid configs to {cfg_dir}")

    if args.skip_run:
        print("--skip-run set: configs generated, skipping 1D + heatmaps")
        return

    # Topology is identical across baseline + grid (type/strength don't move masks).
    topo = _topology(base)
    baseline = analyze_h5("baseline", args.config, args.h5.resolve(), topo)

    # 2) run + analyze + measure each grid point --------------------------
    metrics_seen = list(METRIC_LABELS)
    shape = (len(ta_sorted), len(sp_sorted))
    raws = {m: np.full(shape, np.nan) for m in metrics_seen}
    folds = {m: np.full(shape, np.nan) for m in metrics_seen}
    measured = {k: np.full(shape, np.nan) for k in _measured_keys()}
    measured[TYPEA_PERKB_KEY] = np.full(shape, np.nan)

    tasks = [
        {
            "ri": ri, "ci": ci, "ta": ta, "sp": sp,
            "label": ta_label(ta, sp),
            "cfg_path": str(grid_cfg_paths[(ta, sp)]),
            "h5_dir": str(h5_dir),
            "force_1d": args.force_1d,
        }
        for ri, ta in enumerate(ta_sorted)
        for ci, sp in enumerate(sp_sorted)
    ]
    total = len(tasks)
    jobs = max(1, int(args.jobs))

    def _record(task: dict, h5_path: str) -> None:
        ri, ci, label = task["ri"], task["ci"], task["label"]
        means = analyze_h5(label, Path(task["cfg_path"]), Path(h5_path), topo)
        for m in metrics_seen:
            gm = means.get(m, float("nan"))
            bm = baseline.get(m, float("nan"))
            raws[m][ri, ci] = gm
            folds[m][ri, ci] = gm / bm if np.isfinite(bm) and bm != 0 else float("nan")
        if not args.no_measure:
            ms = measure_lesion_stall(
                Path(h5_path), tick_seconds, chunk=args.measure_chunk,
                prerecognition_ticks=prerec_ticks, repair_ticks=repair_t)
            for k in _measured_keys():
                measured[k][ri, ci] = ms[k]
            measured[TYPEA_PERKB_KEY][ri, ci] = measure_typea_per_kb_genebody(
                Path(h5_path), gb_mask, chunk=args.measure_chunk)

    if jobs == 1:
        for n, task in enumerate(tasks, 1):
            print(f"[{n}/{total}] {task['label']}  "
                  f"(type_a={task['ta']:.3f}, density={density_axis[task['ci']]:.1f}/Mbp)")
            res = _run_grid_point(task)
            _record(task, res["h5"])
    else:
        print(f"running {total} grid points with --jobs {jobs} (process pool, fork context)")
        ctx = mp.get_context("fork")
        by_label = {t["label"]: t for t in tasks}
        n = 0
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for res in ex.map(_run_grid_point, tasks):
                n += 1
                task = by_label[res["label"]]
                print(f"[{n}/{total}] done {res['label']}  "
                      f"(type_a={task['ta']:.3f}, density={density_axis[task['ci']]:.1f}/Mbp)")
                _record(task, res["h5"])

    # 3) outputs ----------------------------------------------------------
    # Y axis labels the row's Type A probability only; the measured per-row
    # Type-A/kb burden has its own dedicated panel below.
    y_label = "Lesion Type A probability"
    y_fmt = "{:.2f}"
    y_ticklabels = None
    sub = (f"STATIC block_prob={block_prob:g}, RNAPII off, "
           f"boundaries x{args.bstr_mult:g} (cap 1.0), tick={tick_seconds:g}s")

    fold_svg = out_dir / "ta_density_heatmaps.svg"
    plot_heatmaps(
        folds, METRIC_LABELS,
        y_values=ta_sorted, y_label=y_label, y_fmt=y_fmt, density_axis=density_axis, x_label=x_label,
        y_ticklabels=y_ticklabels,
        out_path=fold_svg,
        title=f"Type A x density grid vs baseline 1D sim (fold = grid mean / baseline mean)\n{sub}",
        value_label="fold vs baseline mean",
        center=1.0,
    )
    fold_tsv = out_dir / "ta_density_folds.tsv"
    frows = []
    for ri, ta in enumerate(ta_sorted):
        for ci, sp in enumerate(sp_sorted):
            for m in metrics_seen:
                frows.append({
                    "metric": m,
                    "metric_label": METRIC_LABELS[m].replace("\n", " "),
                    "lesion_type_a_prob": ta,
                    "lesion_block_prob": block_prob,
                    "lesion_spacing": sp,
                    "lesion_density_per_mbp": density_axis[ci],
                    "baseline_mean": baseline.get(m, float("nan")),
                    "grid_mean": raws[m][ri, ci],
                    "fold_vs_baseline": folds[m][ri, ci],
                })
    pd.DataFrame(frows).to_csv(fold_tsv, sep="\t", index=False)
    print(f"wrote {fold_svg}")
    print(f"wrote {fold_tsv}")

    if not args.no_measure:
        ta_perkb_svg = out_dir / "ta_density_typea_per_kb_genebody.svg"
        plot_heatmaps(
            measured, TYPEA_PERKB_LABELS,
            y_values=ta_sorted, y_label=y_label, y_fmt=y_fmt, density_axis=density_axis, x_label=x_label,
            y_ticklabels=y_ticklabels,
            out_path=ta_perkb_svg,
            title=(f"Measured Type-A lesion burden inside gene bodies (from trajectory)\n"
                   f"gene-body length={gb_kb} kb; {sub}"),
            value_label="Type-A lesions / kb gene body", cell_fmt="{:.4f}", cell_fontsize=4.5,
            center=None, cmap=SEQ_CMAP,
        )
        # Same readout expressed as a SPACING: kb of gene body per Type-A lesion =
        # 1 / (Type-A per kb). Plotting the reciprocal grid makes the colour scale,
        # colorbar ticks and integer cell annotations all consistent (kb, not /kb).
        # Zero/non-finite density -> no finite spacing -> NaN cell ("n/a").
        ta_perkb_inv_svg = out_dir / "ta_density_typea_one_per_kb_genebody.svg"
        dens = measured[TYPEA_PERKB_KEY]
        with np.errstate(divide="ignore", invalid="ignore"):
            spacing = np.where(dens > 0, 1.0 / dens, np.nan)
        spacing_grid = {TYPEA_PERKB_KEY: spacing}

        def _kb_per_lesion(v: float) -> str:
            return f"{round(v):d}"

        plot_heatmaps(
            spacing_grid, TYPEA_KB_PER_LESION_LABELS,
            y_values=ta_sorted, y_label=y_label, y_fmt=y_fmt, density_axis=density_axis, x_label=x_label,
            y_ticklabels=y_ticklabels,
            out_path=ta_perkb_inv_svg,
            title=(f"Measured Type-A lesion spacing inside gene bodies (from trajectory)\n"
                   f"gene-body length={gb_kb} kb; {sub}"),
            value_label="kb of gene body per Type-A lesion", cell_text=_kb_per_lesion, cell_fontsize=4.0,
            center=None, log_norm=True, cmap=SEQ_CMAP,
        )
        occ_svg = out_dir / "ta_density_measured_stall.svg"
        plot_heatmaps(
            measured, MEASURED_LABELS,
            y_values=ta_sorted, y_label=y_label, y_fmt=y_fmt, density_axis=density_axis, x_label=x_label,
            y_ticklabels=y_ticklabels,
            out_path=occ_svg,
            title=f"Measured per-lesion cohesin stall occupancy (from trajectory)\n{sub}",
            value_label="stalled cohesin legs per lesion", cell_fmt="{:.3f}", cell_fontsize=4.5,
            center=None, cmap=SEQ_CMAP,
        )
        sec_svg = out_dir / "ta_density_measured_stall_seconds.svg"
        plot_heatmaps(
            measured, MEASURED_SECONDS_LABELS,
            y_values=ta_sorted, y_label=y_label, y_fmt=y_fmt, density_axis=density_axis, x_label=x_label,
            y_ticklabels=y_ticklabels,
            out_path=sec_svg,
            title=f"Measured cohesin stall DURATION (s, per event, from trajectory)\n{sub}",
            value_label="stall duration (s)", cell_fmt="{:.0f}", center=None, cmap=SEQ_CMAP,
        )
        bytype_svg = out_dir / "ta_density_measured_stall_by_type.svg"
        plot_heatmaps(
            measured, MEASURED_BYTYPE_LABELS,
            y_values=ta_sorted, y_label=y_label, y_fmt=y_fmt, density_axis=density_axis, x_label=x_label,
            y_ticklabels=y_ticklabels,
            out_path=bytype_svg,
            title=f"Measured cohesin stall by lesion type/state (mean s, from trajectory)\n{sub}",
            value_label="stall duration (s)", cell_fmt="{:.0f}", center=None, cmap=SEQ_CMAP,
        )
        frac_svg = out_dir / "ta_density_measured_stall_fraction.svg"
        plot_heatmaps(
            measured, MEASURED_STAGEFRAC_LABELS,
            y_values=ta_sorted, y_label=y_label, y_fmt=y_fmt, density_axis=density_axis, x_label=x_label,
            y_ticklabels=y_ticklabels,
            out_path=frac_svg,
            title=f"Measured cohesin stall / lesion stage lifetime (per-encounter mean, from trajectory)\n{sub}",
            value_label="stall / stage lifetime", cell_fmt="{:.2f}", center=None, cmap=SEQ_CMAP,
        )
        m_tsv = out_dir / "ta_density_measured_stall.tsv"
        mrows = []
        for ri, ta in enumerate(ta_sorted):
            for ci, sp in enumerate(sp_sorted):
                mrows.append({
                    "lesion_type_a_prob": ta,
                    "lesion_block_prob": block_prob,
                    "lesion_spacing": sp,
                    "lesion_density_per_mbp": density_axis[ci],
                    **{k: measured[k][ri, ci] for k in measured},
                    "effective_stall_seconds_analytic": eff_stall,
                })
        pd.DataFrame(mrows).to_csv(m_tsv, sep="\t", index=False)
        print(f"wrote {ta_perkb_svg}")
        print(f"wrote {ta_perkb_inv_svg}")
        print(f"wrote {occ_svg}")
        print(f"wrote {sec_svg}")
        print(f"wrote {bytype_svg}")
        print(f"wrote {frac_svg}")
        print(f"wrote {m_tsv}")


if __name__ == "__main__":
    main()
