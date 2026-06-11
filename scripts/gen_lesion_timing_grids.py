#!/usr/bin/env python
"""Grid search lesion TIMING (seconds) vs lesion density, at a fixed block prob.

Runs two independent sweeps, both against lesion density on the x-axis:

  1. ``prerecog/`` -- pre-recognition time (y) x lesion density (x), with the
     repair time held fixed. Pre-recognition is how long a lesion sits in the PRE
     state before the repair machinery assembles. For Type A this is the window
     during which the lesion intrinsically stalls cohesin (TC-NER / stalled-Pol-II
     mimic, with RNAPII off); for Type B it sets when the lesion *starts* blocking
     (B blocks only in REPAIR) and so the steady-state REPAIR fraction.
  2. ``repair/`` -- repair time (y) x lesion density (x), with pre-recognition
     held fixed. Repair time is how long a lesion stays in the (blocking) REPAIR
     state before removal -- the dwell cap for both A and B repair stalls.

Times are given in SECONDS and converted to ticks with the config's
``tick_seconds`` (ticks = round(seconds / tick_seconds), >= 1). The y-axis shows
the REALIZED seconds (ticks x tick_seconds) actually used by the simulation.

Every config otherwise matches the sister scripts: RNAPII off, boundary strengths
x ``--bstr-mult`` capped at 1.0, a STATIC ``--block-prob`` and ``--type-a-prob``.
Each sweep writes the same family of outputs as the sister scripts (fold heatmaps
+ measured occupancy / duration / by-type stall), with the y-axis relabelled to
the swept time. The by-type figure is the headline: the pre-recognition sweep
moves the Type A PRE stall; the repair sweep moves the A/B repair stalls.

Building, measurement and plotting are imported from
``gen_lesion_grid_and_heatmaps.py``; this file only chooses what is swept.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

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


def seconds_to_ticks(seconds: float, tick_seconds: float) -> int:
    return max(1, int(round(float(seconds) / float(tick_seconds))))


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


def _seconds_axis(lo: float, hi: float, step: float, tick_seconds: float) -> tuple[list[int], list[float]]:
    """Requested seconds -> (unique tick counts ascending, realized seconds)."""
    realized: dict[int, float] = {}
    for s in sorted(set(_p_values(hi, lo, step))):
        t = seconds_to_ticks(s, tick_seconds)
        realized[t] = t * float(tick_seconds)
    ticks = sorted(realized)
    return ticks, [realized[t] for t in ticks]


def run_sweep(
    *,
    base: dict,
    baseline: dict,
    topo: dict,
    lesion_defaults: dict,
    out_dir: Path,
    prefix: str,
    swept: str,                 # "prerecog" or "repair"
    y_ticks: list[int],
    y_seconds: list[float],
    y_label: str,
    fixed_ticks: int,
    fixed_label: str,
    block_prob: float,
    type_a_prob: float,
    bstr_mult: float,
    tick_seconds: float,
    sp_sorted: list[int],
    density_axis: list[float],
    type_b_enabled: bool,
    x_label: str,
    jobs: int,
    skip_run: bool,
    force_1d: bool,
    no_measure: bool,
    measure_chunk: int,
) -> None:
    sweep_dir = out_dir / prefix
    cfg_dir = sweep_dir / "configs"
    h5_dir = sweep_dir / "h5"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== sweep '{prefix}': {y_label} x density "
          f"({len(y_ticks)} x {len(sp_sorted)} = {len(y_ticks) * len(sp_sorted)} configs) ===")
    print(f"  {y_label:24s}: {[round(s) for s in y_seconds]}  (ticks {y_ticks})")
    print(f"  {fixed_label}")

    def label(t: int, sp: int) -> str:
        return _safe_label(f"{prefix}_t{t}_s{sp}")

    grid_cfg_paths: dict[tuple[int, int], Path] = {}
    for t in y_ticks:
        for sp in sp_sorted:
            cfg = build_grid_config(
                base, p=block_prob, spacing=sp, bstr_mult=bstr_mult,
                type_a_prob=type_a_prob, tick_seconds=tick_seconds, lesion_defaults=lesion_defaults,
                type_b_enabled=type_b_enabled,
            )
            tk = cfg["lef"]["topology_kwargs"]
            # build_grid_config keeps the base config's timing (setdefault); override
            # both explicitly so the swept axis and the fixed companion are exact.
            if swept == "prerecog":
                tk["lesion_prerecognition_ticks"] = int(t)
                tk["lesion_repair_ticks"] = int(fixed_ticks)
            else:
                tk["lesion_repair_ticks"] = int(t)
                tk["lesion_prerecognition_ticks"] = int(fixed_ticks)
            path = cfg_dir / f"{label(t, sp)}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
            grid_cfg_paths[(t, sp)] = path
    print(f"  wrote {len(grid_cfg_paths)} configs to {cfg_dir}")
    if skip_run:
        return

    metrics_seen = list(METRIC_LABELS)
    shape = (len(y_ticks), len(sp_sorted))
    raws = {m: np.full(shape, np.nan) for m in metrics_seen}
    folds = {m: np.full(shape, np.nan) for m in metrics_seen}
    measured = {k: np.full(shape, np.nan) for k in _measured_keys()}

    tasks = [
        {
            "ri": ri, "ci": ci, "t": t, "sp": sp,
            "label": label(t, sp),
            "cfg_path": str(grid_cfg_paths[(t, sp)]),
            "h5_dir": str(h5_dir),
            "force_1d": force_1d,
        }
        for ri, t in enumerate(y_ticks)
        for ci, sp in enumerate(sp_sorted)
    ]
    total = len(tasks)

    def record(task: dict, h5_path: str) -> None:
        ri, ci = task["ri"], task["ci"]
        means = analyze_h5(task["label"], Path(task["cfg_path"]), Path(h5_path), topo)
        for m in metrics_seen:
            gm = means.get(m, float("nan"))
            bm = baseline.get(m, float("nan"))
            raws[m][ri, ci] = gm
            folds[m][ri, ci] = gm / bm if np.isfinite(bm) and bm != 0 else float("nan")
        if not no_measure:
            # Stage windows for the fraction metric: the swept axis is this row's
            # tick value, its fixed companion is the constant.
            if swept == "prerecog":
                prt, rpt = task["t"], fixed_ticks
            else:
                prt, rpt = fixed_ticks, task["t"]
            ms = measure_lesion_stall(
                Path(h5_path), tick_seconds, chunk=measure_chunk,
                prerecognition_ticks=prt, repair_ticks=rpt)
            for k in measured:
                measured[k][ri, ci] = ms[k]

    if jobs <= 1:
        for n, task in enumerate(tasks, 1):
            print(f"  [{n}/{total}] {task['label']}  "
                  f"({y_label}={task['t'] * tick_seconds:.0f}s, density={density_axis[task['ci']]:.1f}/Mbp)")
            record(task, _run_grid_point(task)["h5"])
    else:
        print(f"  running {total} grid points with jobs={jobs} (process pool, fork)")
        by_label = {t["label"]: t for t in tasks}
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as ex:
            for n, res in enumerate(ex.map(_run_grid_point, tasks), 1):
                task = by_label[res["label"]]
                print(f"  [{n}/{total}] done {res['label']}  "
                      f"({y_label}={task['t'] * tick_seconds:.0f}s, density={density_axis[task['ci']]:.1f}/Mbp)")
                record(task, res["h5"])

    # outputs (y-axis = realized seconds) ---------------------------------
    y_fmt = "{:.0f}"
    sub = (f"{fixed_label}; STATIC block_prob={block_prob:g}, type_a={type_a_prob:g}, "
           f"RNAPII off, boundaries x{bstr_mult:g}, tick={tick_seconds:g}s")

    fold_svg = sweep_dir / f"{prefix}_heatmaps.svg"
    plot_heatmaps(
        folds, METRIC_LABELS,
        y_values=y_seconds, y_label=y_label, y_fmt=y_fmt, density_axis=density_axis, x_label=x_label,
        out_path=fold_svg,
        title=f"{y_label} x density vs baseline 1D sim (fold = grid mean / baseline mean)\n{sub}",
        value_label="fold vs baseline mean", center=1.0,
    )
    frows = []
    for ri, (t, ys) in enumerate(zip(y_ticks, y_seconds)):
        for ci, sp in enumerate(sp_sorted):
            for m in metrics_seen:
                frows.append({
                    "metric": m, "metric_label": METRIC_LABELS[m].replace("\n", " "),
                    "swept": prefix, f"{prefix}_seconds": ys, f"{prefix}_ticks": t,
                    "lesion_spacing": sp, "lesion_density_per_mbp": density_axis[ci],
                    "baseline_mean": baseline.get(m, float("nan")),
                    "grid_mean": raws[m][ri, ci], "fold_vs_baseline": folds[m][ri, ci],
                })
    pd.DataFrame(frows).to_csv(sweep_dir / f"{prefix}_folds.tsv", sep="\t", index=False)
    print(f"  wrote {fold_svg}")

    if not no_measure:
        specs = [
            (MEASURED_LABELS, f"{prefix}_measured_stall.svg",
             "Measured per-lesion cohesin stall occupancy", "stalled cohesin legs per lesion", "{:.3f}", SEQ_CMAP),
            (MEASURED_SECONDS_LABELS, f"{prefix}_measured_stall_seconds.svg",
             "Measured cohesin stall DURATION (s, per event)", "stall duration (s)", "{:.0f}", SEQ_CMAP),
            (MEASURED_BYTYPE_LABELS, f"{prefix}_measured_stall_by_type.svg",
             "Measured cohesin stall by lesion type/state (mean s)", "stall duration (s)", "{:.0f}", SEQ_CMAP),
            (MEASURED_STAGEFRAC_LABELS, f"{prefix}_measured_stall_fraction.svg",
             "Measured cohesin stall / lesion stage lifetime (per-encounter mean)",
             "stall / stage lifetime", "{:.2f}", SEQ_CMAP),
        ]
        for labels, fname, ttl, vlab, cfmt, cmap in specs:
            plot_heatmaps(
                measured, labels,
                y_values=y_seconds, y_label=y_label, y_fmt=y_fmt, density_axis=density_axis, x_label=x_label,
                out_path=sweep_dir / fname,
                title=f"{ttl} (from trajectory)\n{sub}",
                value_label=vlab, cell_fmt=cfmt, center=None, cmap=cmap,
            )
        mrows = []
        for ri, (t, ys) in enumerate(zip(y_ticks, y_seconds)):
            for ci, sp in enumerate(sp_sorted):
                mrows.append({
                    "swept": prefix, f"{prefix}_seconds": ys, f"{prefix}_ticks": t,
                    "lesion_spacing": sp, "lesion_density_per_mbp": density_axis[ci],
                    **{k: measured[k][ri, ci] for k in measured},
                })
        pd.DataFrame(mrows).to_csv(sweep_dir / f"{prefix}_measured_stall.tsv", sep="\t", index=False)
        print(f"  wrote measured heatmaps + {prefix}_measured_stall.tsv")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="baseline config YAML")
    ap.add_argument("--h5", type=Path, required=True, help="baseline LEFPositions.h5 for --config")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory (one subdir per sweep)")
    ap.add_argument("--block-prob", type=float, default=0.97, help="STATIC lesion_block_prob")
    ap.add_argument("--type-a-prob", type=float, default=0.5, help="STATIC lesion_type_a_prob")
    ap.add_argument("--lesion-type-b-enabled", action=argparse.BooleanOptionalAction, default=True,
                    help="include Type-B (GG-NER) lesions (default). --no-lesion-type-b-enabled keeps "
                         "only Type-A lesions and switches the x-axis to Type-A density.")
    ap.add_argument("--bstr-mult", type=float, default=3.0, help="boundary-strength multiplier (capped at 1.0)")
    # pre-recognition sweep (seconds), with repair held fixed
    ap.add_argument("--prerecog-lo", type=float, default=300.0, help="lowest pre-recognition time (s)")
    ap.add_argument("--prerecog-hi", type=float, default=1800.0, help="highest pre-recognition time (s)")
    ap.add_argument("--prerecog-step", type=float, default=300.0, help="pre-recognition time step (s)")
    ap.add_argument("--repair-fixed", type=float, default=300.0,
                    help="repair time (s) held fixed during the pre-recognition sweep")
    # repair sweep (seconds), with pre-recognition held fixed
    ap.add_argument("--repair-lo", type=float, default=100.0, help="lowest repair time (s)")
    ap.add_argument("--repair-hi", type=float, default=600.0, help="highest repair time (s)")
    ap.add_argument("--repair-step", type=float, default=100.0, help="repair time step (s)")
    ap.add_argument("--prerecog-fixed", type=float, default=1200.0,
                    help="pre-recognition time (s) held fixed during the repair sweep")
    ap.add_argument("--spacings", type=int, nargs="+", default=[10, 25, 50, 100, 150, 200],
                    help="lesion_spacing values (columns); density = 1000/spacing lesions/Mbp")
    ap.add_argument("--only", choices=["prerecog", "repair"], default=None,
                    help="run only one of the two sweeps (default: both)")
    ap.add_argument("--skip-run", action="store_true", help="only generate configs")
    ap.add_argument("--force-1d", action="store_true", help="rerun 1D even if a matching H5 exists")
    ap.add_argument("--jobs", type=int, default=1, help="concurrent 1D sims (process pool, fork)")
    ap.add_argument("--no-measure", action="store_true", help="skip measured heatmaps")
    ap.add_argument("--measure-chunk", type=int, default=2000, help="frames per streamed measurement block")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(args.config.read_text())
    lef = base["lef"]
    tick_seconds = float(lef.get("tick_seconds") or 0.0)
    if tick_seconds <= 0:
        raise SystemExit("config lef.tick_seconds must be a positive number")

    tk = lef["topology_kwargs"]
    lesion_defaults = {
        "prerecognition_ticks": tk.get(
            "lesion_prerecognition_ticks", max(1, int(round(LESION_PRERECOGNITION_SECONDS / tick_seconds)))),
        "repair_ticks": tk.get(
            "lesion_repair_ticks", max(1, int(round(LESION_REPAIR_SECONDS / tick_seconds)))),
        "tad_size_exponent": tk.get("lesion_tad_size_exponent", DEFAULT_TAD_SIZE_EXPONENT),
        "tad_repair_exponent": tk.get("lesion_tad_repair_exponent", DEFAULT_TAD_REPAIR_EXPONENT),
    }

    sp_sorted = sorted(set(args.spacings), reverse=True)   # 200..10 -> ascending density
    # X-axis density: total (B on) or Type-A density (B off).
    type_b_enabled = bool(args.lesion_type_b_enabled)
    gbm = gene_body_mass(base, float(lesion_defaults["tad_size_exponent"])) if not type_b_enabled else 1.0
    density_axis = lesion_density_axis(
        sp_sorted, type_b_enabled=type_b_enabled, gbm=gbm, type_a_prob=args.type_a_prob)
    x_label = ("Lesion density (lesions/Mbp)" if type_b_enabled
               else "Type-A lesion density (lesions/Mbp)")

    print(f"baseline config : {args.config}")
    print(f"baseline H5     : {args.h5}")
    print(f"tick_seconds    : {tick_seconds:g}")
    print(f"STATIC block_prob={args.block_prob:g}, type_a_prob={args.type_a_prob:g}, "
          f"boundary x{args.bstr_mult:g}, Type-B {'on' if type_b_enabled else 'OFF'}")
    if type_b_enabled:
        print(f"density /Mbp    : {[round(d, 2) for d in density_axis]}  (total)")
    else:
        print(f"Type-A density /Mbp: {[round(d, 2) for d in density_axis]}  "
              f"(gene_body_mass={gbm:.3f} x type_a={args.type_a_prob:g})")

    topo = _topology(base)
    baseline = None
    if not args.skip_run:
        baseline = analyze_h5("baseline", args.config, args.h5.resolve(), topo)

    common = dict(
        base=base, baseline=baseline, topo=topo, lesion_defaults=lesion_defaults,
        out_dir=args.out_dir, block_prob=args.block_prob, type_a_prob=args.type_a_prob,
        bstr_mult=args.bstr_mult, tick_seconds=tick_seconds, sp_sorted=sp_sorted,
        density_axis=density_axis, type_b_enabled=type_b_enabled, x_label=x_label,
        jobs=max(1, int(args.jobs)), skip_run=args.skip_run,
        force_1d=args.force_1d, no_measure=args.no_measure, measure_chunk=args.measure_chunk,
    )

    if args.only in (None, "prerecog"):
        pt, ps = _seconds_axis(args.prerecog_lo, args.prerecog_hi, args.prerecog_step, tick_seconds)
        rfix = seconds_to_ticks(args.repair_fixed, tick_seconds)
        run_sweep(
            prefix="prerecog", swept="prerecog", y_ticks=pt, y_seconds=ps,
            y_label="Pre-recognition time (s)",
            fixed_ticks=rfix, fixed_label=f"repair fixed at {rfix * tick_seconds:.0f}s ({rfix} ticks)",
            **common,
        )

    if args.only in (None, "repair"):
        rt, rs = _seconds_axis(args.repair_lo, args.repair_hi, args.repair_step, tick_seconds)
        pfix = seconds_to_ticks(args.prerecog_fixed, tick_seconds)
        run_sweep(
            prefix="repair", swept="repair", y_ticks=rt, y_seconds=rs,
            y_label="Repair time (s)",
            fixed_ticks=pfix, fixed_label=f"pre-recognition fixed at {pfix * tick_seconds:.0f}s ({pfix} ticks)",
            **common,
        )


if __name__ == "__main__":
    main()
