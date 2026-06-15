#!/usr/bin/env python
"""Lesion grid-search generator + per-metric heatmaps vs a baseline 1D sim.

Given ONE config + its ``LEFPositions.h5`` (the baseline 1D LEF run), this
script derives a 2-D grid of NEW configs that share the baseline's TAD skeleton
and genes but differ in two ways:

  * RNAPII is fully ablated: ``max_rnapii: 0`` and the ``rnapii_load`` /
    ``rnapii_translocate`` plugin slots are nulled.
  * Every boundary strength (per-TAD ``left_strength`` / ``right_strength`` plus
    ``default_boundary_strength``) is multiplied by ``--bstr-mult`` (default 3)
    and capped at 1.0.

The grid axes (independent variables) are:

  Y -- ``lesion_block_prob`` (``p``), the swept config knob itself: a lesion holds
       a cohesin leg with per-tick probability ``p`` (bypass probability
       ``1 - p``). ``p`` is swept 1.0 -> 0.8 in 0.025 steps. Earlier versions put
       a derived "stall time in seconds" on this axis, but no closed form is
       trustworthy here: a Type-B lesion only blocks while in its REPAIR state and
       is then REMOVED, and a held leg can also unload, so the realized stall
       saturates rather than tracking ``tick/(1-p)``. Instead of trusting a
       formula we MEASURE the stall directly (see below). Reference seconds
       columns (analytic effective stall and naive bypass) are still written to
       the fold TSV.

Two figures are produced:

  * ``lesion_grid_heatmaps`` -- per metric, grid mean / BASELINE mean (fold,
    colour centred at 1).
  * ``lesion_grid_measured_stall`` -- lesion-induced cohesin stall MEASURED from
    each run's trajectory (a leg is stalled when its next extrusion site holds a
    blocking lesion), as per-lesion OCCUPANCY. NOT vs baseline; no stall formula.
  * ``lesion_grid_measured_stall_seconds`` -- the realized stall DURATION in
    SECONDS (mean + median per stall event), measured as runs of consecutive
    blocked frames x ``tick_seconds``. This is the concrete "how long is a cohesin
    stalled at a lesion" answer. See :func:`measure_lesion_stall`. The measured
    TSV also carries p90/max/n_events and the analytic effective-stall reference.
    Disable both with ``--no-measure``.

  X -- lesion DENSITY in lesions/Mbp, set via ``lesion_spacing``. The steady
       state holds ``num_sites // spacing`` lesions; with 1 monomer = 1 kb the
       density is ``1000 / spacing`` lesions/Mbp (independent of chain length).
       ``spacing`` is swept over {10, 25, 50, 100, 150, 200} -> {100, 40, 20, 10,
       6.67, 5} lesions/Mbp (6 columns).

For every grid config the 1D LEF stage is run (or an existing matching H5 is
reused), the same per-replicate-chain metrics used by
``compare_config_chain_metrics.py`` are computed, and each metric's chain-mean is
divided by the BASELINE config's chain-mean (the input config + its H5). One
heatmap per metric is drawn with the colour bar centred at 1.0 (fold = 1 means
"same as the baseline 1D sim").
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
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

# Diverging fold-change colormap shared with sweep_rnapoff_boundary_strength_1d.py:
# slate-blue (below baseline) -> white (fold = 1.0, the TwoSlopeNorm centre) ->
# brick-red (above baseline). Used for every "fold vs baseline" heatmap.
FOLD_CMAP = LinearSegmentedColormap.from_list(
    "fold_slate_red", ["#465775", "#ffffff", "#A63446"]
)
# Single-hue sequential map for the non-centered (no natural midpoint) heatmaps:
# white at the data minimum -> brick-red at the maximum (same red as FOLD_CMAP).
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_white_red", ["#ffffff", "#A63446"]
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the exact topology + metric machinery from the comparison script so the
# grid folds are computed identically to the pairwise comparison.
from compare_config_chain_metrics import (  # noqa: E402
    BOUNDARY_LABELS,
    CORE_LABELS,
    _resolve_h5,
    _safe_label,
    _topology,
    analyze_run,
)

LESION_PLUGIN = "polychrom.pipelines.loop_extrusion.plugins.lesions:update_lesions"

# Biological stage durations (seconds) used only when the baseline config has no
# lesion kwargs to inherit; mirrors gen_realistic_configs_variable_tick.py.
LESION_PRERECOGNITION_SECONDS = 1200.0
LESION_REPAIR_SECONDS = 300.0
DEFAULT_TYPE_A_PROB = 0.25
DEFAULT_TAD_SIZE_EXPONENT = 1.0
DEFAULT_TAD_REPAIR_EXPONENT = 1.0

# Metrics to draw, in display order. The union of the core + boundary sets used
# by compare_config_chain_metrics.py, de-duplicated (some names live in both).
METRIC_LABELS: dict[str, str] = {}
for _src in (CORE_LABELS, BOUNDARY_LABELS):
    for _k, _v in _src.items():
        METRIC_LABELS.setdefault(_k, _v.replace("\n", " "))


def raw_bypass_seconds(p: float, tick_seconds: float) -> float:
    """Naive mean time to *stochastically* bypass a persistent lesion (s):
    tick / (1 - p). Diverges as p -> 1. This is the OLD y-axis and it is wrong
    as a stall time because it assumes the lesion never disappears."""
    if p >= 1.0:
        return float("inf")
    return float(tick_seconds) / (1.0 - p)


def effective_stall_seconds(
    p: float,
    tick_seconds: float,
    repair_ticks: float,
    lifetime_stalled_ticks: float | None = None,
) -> float:
    """Mean time a cohesin leg is ACTUALLY held at a lesion (s).

    A Type-B lesion only blocks a cohesin leg while it is in the REPAIR state;
    the block is leaky (per-tick bypass probability ``1 - p``) and the lesion is
    *removed* when repair completes (mean ``repair_ticks``). A held leg can also
    unload via the cohesin stalled lifetime (mean ``lifetime_stalled_ticks``).

    These are independent memory-less escapes, so their per-tick hazards add and
    the realized stall is their reciprocal sum::

        rate = (1 - p)/tick + 1/repair_seconds [+ 1/lifetime_stalled_seconds]
        stall = 1 / rate

    Unlike :func:`raw_bypass_seconds` this SATURATES: as ``p -> 1`` the bypass
    term vanishes and the stall is capped by lesion repair (and, if included, the
    cohesin lifetime) instead of diverging."""
    rate = (1.0 - p) / float(tick_seconds) + 1.0 / (float(repair_ticks) * float(tick_seconds))
    if lifetime_stalled_ticks:
        rate += 1.0 / (float(lifetime_stalled_ticks) * float(tick_seconds))
    return 1.0 / rate if rate > 0 else float("inf")


def density_per_mbp(spacing: int) -> float:
    """Lesions/Mbp for a given spacing (1 monomer = 1 kb -> 1000/spacing)."""
    return 1000.0 / float(spacing)


def gene_body_mass(base: dict, tad_size_exponent: float) -> float:
    """Fraction of the TAD-weighted placement mass that lands in gene bodies for
    the baseline genes + TAD layout.

    Mirrors ``lesions.precompute_lesion_fields`` (gene_body_mass = sum of
    ``lesion_site_p`` over gene-body sites). With Type-B disabled only gene-body
    lesions can be Type A, so the realised Type-A density is
    ``total_density * gene_body_mass * type_a_prob``. Constant across a grid (it
    depends only on the genes, TAD layout and ``tad_size_exponent``)."""
    from polychrom.pipelines.loop_extrusion.plugins.lesions import _tad_length_per_site

    lef = base["lef"]
    tk = lef.get("topology_kwargs", {})
    chain_length = int(lef["chain_length"])
    num_chains = int(lef["num_chains"])
    N = chain_length * num_chains
    tad_len, _ = _tad_length_per_site(num_chains, chain_length, tk.get("tad_positions") or [])
    weights = tad_len ** (-float(tad_size_exponent))
    total = float(weights.sum())
    p = (weights / total) if total > 0 else np.full(N, 1.0 / N)
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
    return float(p[mask].sum())


def lesion_density_axis(
    sp_sorted: list[int],
    *,
    type_b_enabled: bool,
    gbm: float = 1.0,
    type_a_prob: float = 1.0,
) -> list[float]:
    """Per-column x-axis lesion density for the heatmaps.

    Type-B enabled -> TOTAL lesion density (``1000/spacing``), as usual. Type-B
    disabled -> Type-A lesion density ``total * gbm * type_a_prob`` (only gene-body
    lesions can be Type A). Pass ``type_a_prob=1.0`` when P(A) is itself a grid
    axis to get the per-column gene-body (Type-A-eligible) density, scaled per row
    by the y-axis."""
    factor = 1.0 if type_b_enabled else float(gbm) * float(type_a_prob)
    return [density_per_mbp(sp) * factor for sp in sp_sorted]


def _p_values(p_hi: float, p_lo: float, step: float) -> list[float]:
    """p from p_hi down to p_lo (inclusive) in `step` decrements."""
    n = int(round((p_hi - p_lo) / step))
    return [round(p_hi - i * step, 6) for i in range(n + 1)]


def build_grid_config(
    base: dict,
    *,
    p: float,
    spacing: int,
    bstr_mult: float,
    type_a_prob: float,
    tick_seconds: float,
    lesion_defaults: dict,
    type_b_enabled: bool = True,
) -> dict:
    """Return a deep-copied config with RNAPII off, boosted boundaries, and the
    lesion block-prob / spacing set for one grid point. ``type_a_prob`` is a
    strict, fixed override applied to every config (like ``bstr_mult``).
    ``type_b_enabled`` toggles Type-B (GG-NER) lesions: when False only Type-A
    lesions are kept (the Type-A count is unchanged, no backfill)."""
    cfg = copy.deepcopy(base)
    lef = cfg["lef"]

    # --- RNAPII fully off -------------------------------------------------
    lef["max_rnapii"] = 0
    plugins = lef.setdefault("plugins", {})
    plugins["rnapii_load"] = None
    plugins["rnapii_translocate"] = None
    # Enable the lesion plugin (config may not have had it).
    plugins["lesion"] = LESION_PLUGIN

    tk = lef["topology_kwargs"]

    # --- boundary strengths x bstr_mult, capped at 1.0 -------------------
    cap = lambda v: round(min(float(v) * bstr_mult, 1.0), 4)
    for tad in tk.get("tads", []):
        if "left_strength" in tad:
            tad["left_strength"] = cap(tad["left_strength"])
        if "right_strength" in tad:
            tad["right_strength"] = cap(tad["right_strength"])
    if "default_boundary_strength" in tk:
        tk["default_boundary_strength"] = cap(tk["default_boundary_strength"])

    # --- lesion grid point ------------------------------------------------
    tk["lesion_spacing"] = int(spacing)
    tk["lesion_block_prob"] = round(float(p), 4)
    # Strict overrides (same kind of fixed change as bstr_mult): always set, never
    # inherited from the baseline. lesion_defaults already resolves each to the
    # CLI override, else the baseline config value, else the seconds-based default.
    tk["lesion_type_a_prob"] = round(float(type_a_prob), 4)
    tk["lesion_type_b_enabled"] = bool(type_b_enabled)
    tk["lesion_prerecognition_ticks"] = int(lesion_defaults["prerecognition_ticks"])
    tk["lesion_repair_ticks"] = int(lesion_defaults["repair_ticks"])
    # TAD-size exponents are not (yet) CLI-controlled; carry from the baseline.
    tk.setdefault("lesion_tad_size_exponent", lesion_defaults["tad_size_exponent"])
    tk.setdefault("lesion_tad_repair_exponent", lesion_defaults["tad_repair_exponent"])
    return cfg


def metric_means(rows: list[dict]) -> dict[str, float]:
    """Chain-mean raw value per metric from analyze_run rows."""
    df = pd.DataFrame(rows)
    return df.groupby("metric")["raw_value"].mean().to_dict()


def analyze_h5(label: str, cfg_path: Path, h5_path: Path, topo: dict) -> dict[str, float]:
    core, boundary, _aux, _lengths = analyze_run(label, cfg_path, h5_path, topo)
    means = metric_means(core)
    means.update(metric_means(boundary))  # identical for shared names
    return means


# Lesion codes (must match plugins/lesions.py).
_LES_TYPE_A, _LES_TYPE_B, _LES_STATE_PRE, _LES_STATE_REPAIR = 0, 1, 0, 1

# Blocking regimes, as category codes painted onto the lattice each frame. These
# are exactly the (type, state) rows of lesions.lesion_stalls_cohesin that stall
# cohesin (B-PRE never blocks; A-PRE only blocks when RNAPII is absent).
_CAT_A_PRE, _CAT_A_REPAIR, _CAT_B_REPAIR = 1, 2, 3
_CATEGORIES = {"a_pre": _CAT_A_PRE, "a_repair": _CAT_A_REPAIR, "b_repair": _CAT_B_REPAIR}

MEASURED_LABELS = {
    "stalled_per_lesion": "Cohesin legs stalled at a lesion\n(per lesion)",
    "stalled_per_blocking_lesion": "Cohesin legs stalled at a lesion\n(per REPAIR-state lesion)",
}

MEASURED_SECONDS_LABELS = {
    "dwell_mean_s": "Mean cohesin stall at a lesion\n(measured, s)",
    "dwell_median_s": "Median cohesin stall at a lesion\n(measured, s)",
}

# Per-(type, state) mean stall durations -- the detailed breakdown.
MEASURED_BYTYPE_LABELS = {
    "a_pre_dwell_mean_s": "Type A pre-recognition stall\n(mean, s)",
    "a_repair_dwell_mean_s": "Type A repair stall\n(mean, s)",
    "b_repair_dwell_mean_s": "Type B repair stall\n(mean, s)",
}

# Per-(type, state) stall as a FRACTION of the lesion's lifetime in that stage:
# per-encounter mean stall / stage-window seconds (PRE -> pre-recognition window,
# REPAIR -> repair window). 1.0 = a held leg occupies the whole stage on average.
MEASURED_STAGEFRAC_LABELS = {
    "a_pre_stall_frac_of_stage": "Type A pre-recognition\nstall / stage lifetime",
    "a_repair_stall_frac_of_stage": "Type A repair\nstall / stage lifetime",
    "b_repair_stall_frac_of_stage": "Type B repair\nstall / stage lifetime",
}


def _pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def _dwell_summary(run_ticks: list[int], tick_seconds: float, prefix: str,
                   legframes: int, frames: int, n_lefs: int) -> dict[str, float]:
    """Mean/median/p90/max/n + leg-time fraction for one set of stall segments."""
    d = np.asarray(run_ticks, dtype=float) * float(tick_seconds)
    return {
        f"{prefix}_dwell_mean_s": float(d.mean()) if d.size else float("nan"),
        f"{prefix}_dwell_median_s": _pct(d, 50),
        f"{prefix}_dwell_p90_s": _pct(d, 90),
        f"{prefix}_dwell_max_s": float(d.max()) if d.size else float("nan"),
        f"{prefix}_n_events": int(d.size),
        f"{prefix}_legtick_fraction": legframes / max(frames * n_lefs * 2, 1),
    }


def _stage_frac(out: dict[str, float], tick_seconds: float,
                prerecognition_ticks: float | None, repair_ticks: float | None) -> None:
    """Add ``<cat>_stall_frac_of_stage`` to ``out``: the per-encounter mean stall
    divided by the lesion's mean lifetime in that stage (PRE -> pre-recognition
    window, REPAIR -> repair window). NaN when the window is unknown (the caller
    did not pass it) or the category had no measured stall."""
    stage_ticks = {"a_pre": prerecognition_ticks,
                   "a_repair": repair_ticks, "b_repair": repair_ticks}
    for cat in _CATEGORIES:
        win_t = stage_ticks.get(cat)
        win_s = float(win_t) * float(tick_seconds) if win_t else 0.0
        dwell = out.get(f"{cat}_dwell_mean_s", float("nan"))
        out[f"{cat}_stall_frac_of_stage"] = (
            dwell / win_s if win_s > 0 and np.isfinite(dwell) else float("nan"))


def measure_lesion_stall(
    h5_path: Path, tick_seconds: float, chunk: int = 2000,
    *, prerecognition_ticks: float | None = None, repair_ticks: float | None = None,
) -> dict[str, float]:
    """Measure lesion-induced cohesin stalling DIRECTLY from a 1D trajectory.

    A cohesin leg is counted as lesion-stalled in a frame when the site it would
    extrude into holds a *blocking* lesion: the left leg (``positions[:,j,0]``)
    steps to ``pos-1``, the right leg (``positions[:,j,1]``) to ``pos+1`` (see
    lef_dynamics; left.pos <= right.pos always, and slot ``j`` is a stable column
    so a reload jump ends a run). A lesion blocks cohesin when it is in REPAIR
    (either type) or PRE as Type A with RNAPII absent -- exactly
    ``lesions.lesion_stalls_cohesin``.

    Two complementary readouts, neither using an analytic stall formula:

    * Occupancy (time-averaged): ``stalled_per_lesion`` etc. -- how busy lesions
      are (folds in dwell + arrival rate).
    * Dwell DURATION in seconds: a stall event is a maximal run of consecutive
      blocked frames for one ``(slot, leg)``; its length x ``tick_seconds`` is the
      realized stall. Runs touching frame 0 (left-censored) or still open at the
      final frame (right-censored) are dropped. Each recorded frame is one tick,
      so this requires FULL frames (no striding); frames are streamed in blocks of
      ``chunk`` to bound memory.

    Returns occupancy keys plus the OVERALL stall duration (``dwell_mean_s`` /
    ``dwell_median_s`` / ``dwell_p90_s`` / ``dwell_max_s`` / ``n_stall_events`` /
    ``stalled_legtick_fraction``) AND a per-regime breakdown for each blocking
    category -- ``a_pre`` (Type A pre-recognition), ``a_repair`` (Type A repair),
    ``b_repair`` (Type B repair) -- with the same ``<cat>_dwell_*`` / ``_n_events``
    / ``_legtick_fraction`` keys. The overall run is the full continuous stall;
    the per-category segments split it where the blocking lesion changes
    (type, state), e.g. a Type A lesion's PRE phase vs its REPAIR phase.
    """
    keys = ("stalled_legs_per_frame", "lesions_per_frame", "blocking_lesions_per_frame",
            "stalled_per_lesion", "stalled_per_blocking_lesion",
            "dwell_mean_s", "dwell_median_s", "dwell_p90_s", "dwell_max_s",
            "n_stall_events", "stalled_legtick_fraction")
    with h5py.File(h5_path, "r") as h:
        N = int(h.attrs["N"])
        chain = int(h.attrs["chain_length"])
        rnapii = bool(h.attrs.get("rnapii_enabled", False))
        if "lesions" not in h:
            base = {k: float("nan") for k in keys}
            for cat in _CATEGORIES:
                base.update(_dwell_summary([], tick_seconds, cat, 0, 1, 1))
            _stage_frac(base, tick_seconds, prerecognition_ticks, repair_ticks)
            return base
        frames = int(h["positions"].shape[0])
        n_lefs = int(h["positions"].shape[1])
        ds_sites, ds_states, ds_types = h["lesions"], h["lesion_states"], h["lesion_types"]
        ds_pos = h["positions"]

        stalled = lesions_tot = blocking_tot = 0
        # Overall continuous stall (any blocking lesion).
        runs: list[int] = []
        cur = np.zeros((n_lefs, 2), dtype=np.int64)
        from_zero = np.zeros((n_lefs, 2), dtype=bool)
        # Per-category segments (split when the blocking category changes).
        cat_runs: dict[int, list[int]] = {c: [] for c in _CATEGORIES.values()}
        cat_legframes: dict[int, int] = {c: 0 for c in _CATEGORIES.values()}
        cur_cat = np.zeros((n_lefs, 2), dtype=np.int8)
        cur_len = np.zeros((n_lefs, 2), dtype=np.int64)
        from_zero_cat = np.zeros((n_lefs, 2), dtype=bool)
        gf = 0
        for start in range(0, frames, chunk):
            end = min(start + chunk, frames)
            sites = ds_sites[start:end]
            states = ds_states[start:end]
            types = ds_types[start:end]
            pos = ds_pos[start:end]
            for k in range(end - start):
                s, st, ty = sites[k], states[k], types[k]
                valid = (s >= 0) & (s < N)
                s, st, ty = s[valid], st[valid], ty[valid]
                lesions_tot += s.size

                # Paint each site with its blocking-regime category code.
                catmap = np.zeros(N, dtype=np.int8)
                if not rnapii:
                    catmap[s[(ty == _LES_TYPE_A) & (st == _LES_STATE_PRE)]] = _CAT_A_PRE
                catmap[s[(ty == _LES_TYPE_A) & (st == _LES_STATE_REPAIR)]] = _CAT_A_REPAIR
                catmap[s[(ty == _LES_TYPE_B) & (st == _LES_STATE_REPAIR)]] = _CAT_B_REPAIR
                blocking_tot += int((catmap > 0).sum())

                left = pos[k, :, 0]
                right = pos[k, :, 1]
                legcat = np.zeros((n_lefs, 2), dtype=np.int8)  # blocking category at each leg's next site
                l_ok = (left % chain) != 0           # left leg's -1 step stays in-chain
                r_ok = ((right + 1) % chain) != 0    # right leg's +1 step stays in-chain
                legcat[l_ok, 0] = catmap[left[l_ok] - 1]
                legcat[r_ok, 1] = catmap[right[r_ok] + 1]
                blocked = legcat > 0
                stalled += int(blocked.sum())
                for c in _CATEGORIES.values():
                    cat_legframes[c] += int((legcat == c).sum())

                # --- overall continuous-stall runs (any category) ---
                ended = (~blocked) & (cur > 0)
                keep = ended & (~from_zero)
                if keep.any():
                    runs.extend(int(x) for x in cur[keep])
                cur[ended] = 0
                from_zero[ended] = False
                cur[blocked] += 1
                if gf == 0:
                    from_zero[blocked] = True

                # --- per-category segments (split on category change) ---
                cont = (legcat == cur_cat) & (legcat > 0)
                seg_ended = (cur_len > 0) & (~cont)
                for c in _CATEGORIES.values():
                    sel = seg_ended & (cur_cat == c) & (~from_zero_cat)
                    if sel.any():
                        cat_runs[c].extend(int(x) for x in cur_len[sel])
                cur_len[cont] += 1
                cur_len[seg_ended] = 0
                cur_cat[seg_ended] = 0
                from_zero_cat[seg_ended] = False
                new_seg = (legcat > 0) & (~cont)
                cur_len[new_seg] = 1
                cur_cat[new_seg] = legcat[new_seg]
                if gf == 0:
                    from_zero_cat[new_seg] = True
                gf += 1

        # open runs at trajectory end are right-censored -> dropped
        dwell = np.asarray(runs, dtype=float) * float(tick_seconds)
        out = {
            "stalled_legs_per_frame": stalled / frames,
            "lesions_per_frame": lesions_tot / frames,
            "blocking_lesions_per_frame": blocking_tot / frames,
            "stalled_per_lesion": stalled / max(lesions_tot, 1),
            "stalled_per_blocking_lesion": stalled / max(blocking_tot, 1),
            "dwell_mean_s": float(dwell.mean()) if dwell.size else float("nan"),
            "dwell_median_s": _pct(dwell, 50),
            "dwell_p90_s": _pct(dwell, 90),
            "dwell_max_s": float(dwell.max()) if dwell.size else float("nan"),
            "n_stall_events": int(dwell.size),
            "stalled_legtick_fraction": stalled / max(frames * n_lefs * 2, 1),
        }
        for cat, code in _CATEGORIES.items():
            out.update(_dwell_summary(cat_runs[code], tick_seconds, cat,
                                      cat_legframes[code], frames, n_lefs))
        _stage_frac(out, tick_seconds, prerecognition_ticks, repair_ticks)
        return out


def grid_label(p: float, spacing: int) -> str:
    return _safe_label(f"grid_p{p:.3f}_s{spacing}")


def _run_grid_point(task: dict) -> dict:
    """Worker: run (or reuse) the 1D LEF stage for one grid point and return its
    H5 path. Top-level + picklable so it can be dispatched to a process pool.
    Each process re-seeds numpy from the config seed inside ``lef.run``, so
    results are deterministic and isolated from sibling workers."""
    h5_path = _resolve_h5(
        cfg_path=Path(task["cfg_path"]),
        requested_h5=None,
        out_dir=Path(task["h5_dir"]),
        label=task["label"],
        force_1d=task["force_1d"],
    )
    return {"ri": task["ri"], "ci": task["ci"], "label": task["label"], "h5": str(h5_path)}


def plot_heatmaps(
    grids: dict[str, np.ndarray],
    labels: dict[str, str],
    *,
    y_values: list[float],
    y_label: str,
    density_axis: list[float],
    out_path: Path,
    title: str,
    value_label: str,
    y_fmt: str = "{:.3f}",
    cell_fmt: str = "{:.2f}",
    center: float | None = 1.0,
    cmap=FOLD_CMAP,
    x_label: str = "Lesion density (lesions/Mbp)",
    y_ticklabels: list[str] | None = None,
) -> None:
    """Grid of heatmaps, one per metric in ``grids``. Rows = ``y_values``
    (ascending), cols = density (lesions/Mbp, ascending).

    ``center`` selects the colour scale: a float centres a diverging map there
    (e.g. fold = 1.0); ``None`` uses a plain sequential scale spanning the data
    (for standalone measured quantities that have no natural midpoint)."""
    metrics = [m for m in labels if m in grids]
    ncols = min(3, max(1, len(metrics)))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)

    if y_ticklabels is None:
        y_ticklabels = [("∞" if not np.isfinite(v) else y_fmt.format(v)) for v in y_values]
    x_ticklabels = [f"{d:.1f}" for d in density_axis]

    for idx, metric in enumerate(metrics):
        ax = axes.flat[idx]
        grid = grids[metric]
        finite = grid[np.isfinite(grid)]
        dmin, dmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if center is not None:
            vmin, vmax = min(dmin, center - 1e-6), max(dmax, center + 1e-6)
            norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=dmin, vmax=dmax if dmax > dmin else dmin + 1e-6)
        cmap_obj = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
        cmap_obj.set_bad("#f1f3f5")
        im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap_obj, norm=norm)

        ax.set_xticks(range(len(density_axis)))
        ax.set_xticklabels(x_ticklabels, fontsize=8)
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels(y_ticklabels, fontsize=8)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(labels[metric], fontsize=10, pad=6)

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid[r, c]
                txt = "n/a" if not np.isfinite(v) else cell_fmt.format(v)
                ax.text(c, r, txt, ha="center", va="center", fontsize=6.0, color="#111111")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(value_label, fontsize=8)

    for ax in axes.flat[len(metrics):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)  # format inferred from the file suffix
    plt.close(fig)


def write_grid_tsv(
    folds: dict[str, np.ndarray],
    raws: dict[str, np.ndarray],
    *,
    p_values: list[float],
    spacings: list[int],
    stall_axis: list[float],
    raw_bypass_axis: list[float],
    density_axis: list[float],
    tick_seconds: float,
    baseline: dict[str, float],
    out_path: Path,
) -> None:
    rows = []
    for ri, p in enumerate(p_values):
        for ci, sp in enumerate(spacings):
            for metric in folds:
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "lesion_block_prob": p,
                        "lesion_spacing": sp,
                        # Realized stall used for the y-axis (repair-/lifetime-capped)
                        # plus the naive bypass time for reference/comparison.
                        "effective_stall_seconds": stall_axis[ri],
                        "raw_bypass_seconds": raw_bypass_axis[ri],
                        "lesion_density_per_mbp": density_axis[ci],
                        "tick_seconds": tick_seconds,
                        "baseline_mean": baseline.get(metric, float("nan")),
                        "grid_mean": raws[metric][ri, ci],
                        "fold_vs_baseline": folds[metric][ri, ci],
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=Path, required=True, help="baseline config YAML")
    ap.add_argument("--h5", type=Path, required=True, help="baseline LEFPositions.h5 for --config")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory (configs, h5s, heatmaps)")
    ap.add_argument("--bstr-mult", type=float, default=3.0, help="boundary-strength multiplier (capped at 1.0)")
    ap.add_argument(
        "--type-a-prob",
        type=float,
        default=DEFAULT_TYPE_A_PROB,
        help="strict lesion_type_a_prob override applied to every grid config "
             "(P(Type A | in a gene body); off-gene lesions are always Type B)",
    )
    ap.add_argument(
        "--lesion-type-b-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include Type-B (GG-NER) lesions (default). --no-lesion-type-b-enabled "
             "keeps only Type-A lesions and switches the heatmap x-axis to Type-A density.",
    )
    ap.add_argument(
        "--repair-seconds",
        type=float,
        default=None,
        help="lesion repair window in SECONDS (REPAIR->removed; the window in which a "
             "Type-B lesion blocks cohesin). Converted to lesion_repair_ticks via "
             "round(seconds / tick_seconds). Default: inherit from --config, else ~300s.",
    )
    ap.add_argument(
        "--prerecognition-seconds",
        type=float,
        default=None,
        help="lesion pre-recognition window in SECONDS (PRE->REPAIR). Converted to "
             "lesion_prerecognition_ticks via round(seconds / tick_seconds). Longer PRE "
             "means a smaller steady-state REPAIR (blocking) fraction. Default: inherit "
             "from --config, else ~1200s.",
    )
    ap.add_argument(
        "--block-probs", type=float, nargs="+", default=None,
        help="explicit lesion_block_prob values (rows). When given, these are used "
             "directly and --p-hi/--p-lo/--p-step are ignored.",
    )
    ap.add_argument("--p-hi", type=float, default=1.0, help="highest lesion_block_prob (top row)")
    ap.add_argument("--p-lo", type=float, default=0.8, help="lowest lesion_block_prob (bottom row)")
    ap.add_argument("--p-step", type=float, default=0.025, help="lesion_block_prob decrement")
    ap.add_argument(
        "--spacings",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 150, 200],
        help="lesion_spacing values (columns); density = 1000/spacing lesions/Mbp",
    )
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="only generate the grid configs; do not run 1D or plot",
    )
    ap.add_argument(
        "--force-1d",
        action="store_true",
        help="rerun the 1D LEF stage for every grid point even if a matching H5 exists",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of grid-point 1D sims to run concurrently (process pool); "
             "1 = sequential. Each sim is an independent process writing its own "
             "H5, so results are identical to sequential.",
    )
    ap.add_argument(
        "--no-lifetime-cap",
        action="store_true",
        help="effective-stall reference column: cap only by lesion repair (bypass + "
             "repair), excluding the cohesin stalled-lifetime route. Default includes it.",
    )
    ap.add_argument(
        "--no-measure",
        action="store_true",
        help="skip the measured lesion-stall heatmap (only emit the fold heatmaps)",
    )
    ap.add_argument(
        "--measure-chunk",
        type=int,
        default=2000,
        help="frames per streamed block when measuring lesion stalls from the H5 "
             "(memory knob; dwell durations need full frames so striding is not used)",
    )
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
        raise SystemExit("config lef.tick_seconds must be a positive number to convert stall times")

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
    # Strict CLI overrides (control these like --type-a-prob): given in SECONDS,
    # converted to ticks (round(seconds / tick_seconds), floored at 1), they replace
    # the inherited value uniformly across every grid config.
    if args.repair_seconds is not None:
        lesion_defaults["repair_ticks"] = max(1, int(round(args.repair_seconds / tick_seconds)))
    if args.prerecognition_seconds is not None:
        lesion_defaults["prerecognition_ticks"] = max(1, int(round(args.prerecognition_seconds / tick_seconds)))

    if args.block_probs is not None:
        p_values = sorted(set(float(p) for p in args.block_probs))
    else:
        p_values = _p_values(args.p_hi, args.p_lo, args.p_step)
    spacings = list(args.spacings)
    # Axis orderings: ascending stall (rows) and ascending density (cols).
    p_sorted = sorted(p_values)  # ascending p -> ascending stall (0.8..1.0)
    sp_sorted = sorted(spacings, reverse=True)  # 200..10 -> ascending density

    # Effective cohesin stall (s): the realized hold time at a lesion, bounded by
    # lesion repair (the lesion is removed) and -- unless --no-lifetime-cap -- the
    # cohesin stalled lifetime. repair_ticks is the same across the grid (it is
    # not a swept variable); lifetime_stalled is read from the config.
    repair_ticks = float(lesion_defaults["repair_ticks"])
    lifetime_stalled = float(lef.get("lifetime_stalled") or lef.get("lifetime") or 0) or None
    life_term = None if args.no_lifetime_cap else lifetime_stalled
    stall_axis = [effective_stall_seconds(p, tick_seconds, repair_ticks, life_term) for p in p_sorted]
    raw_bypass_axis = [raw_bypass_seconds(p, tick_seconds) for p in p_sorted]
    # X-axis density: total (B on) or Type-A density (B off, scaled by the
    # gene-body mass and the fixed type_a_prob).
    type_b_enabled = bool(args.lesion_type_b_enabled)
    gbm = gene_body_mass(base, float(lesion_defaults["tad_size_exponent"])) if not type_b_enabled else 1.0
    density_axis = lesion_density_axis(
        sp_sorted, type_b_enabled=type_b_enabled, gbm=gbm, type_a_prob=args.type_a_prob)
    x_label = ("Lesion density (lesions/Mbp)" if type_b_enabled
               else "Type-A lesion density (lesions/Mbp)")

    print(f"baseline config : {args.config}")
    print(f"baseline H5     : {args.h5}")
    print(f"tick_seconds    : {tick_seconds:g}")
    print(f"grid            : {len(p_sorted)} stall x {len(sp_sorted)} density = "
          f"{len(p_sorted) * len(sp_sorted)} configs")
    repair_seconds = repair_ticks * tick_seconds
    life_note = (
        f"+ cohesin stalled lifetime {lifetime_stalled * tick_seconds:.0f}s"
        if life_term else "(repair only; cohesin lifetime excluded)"
    )
    print(f"  lesion_block_prob : {p_sorted}")
    print(f"  eff stall (s)     : {[round(s, 1) for s in stall_axis]}")
    print(f"    capped by lesion repair {repair_seconds:.0f}s {life_note}")
    print(f"  raw bypass (old)  : {['inf' if not np.isfinite(s) else round(s, 1) for s in raw_bypass_axis]}  <- diverges; not used")
    print(f"  spacing           : {sp_sorted}")
    if type_b_enabled:
        print(f"  density /Mbp      : {[round(d,2) for d in density_axis]}  (total; Type-B enabled)")
    else:
        print(f"  Type-B DISABLED -> x-axis = Type-A density (gene_body_mass={gbm:.3f} x type_a={args.type_a_prob:g})")
        print(f"  Type-A density /Mbp: {[round(d,2) for d in density_axis]}")
    prerec_ticks = int(lesion_defaults["prerecognition_ticks"])
    repair_t = int(lesion_defaults["repair_ticks"])
    # Type-B lesions only block while in REPAIR; the steady-state REPAIR fraction
    # is repair/(prerec+repair) (uniform-TAD approximation).
    repair_frac = repair_t / (prerec_ticks + repair_t)
    rsrc = "CLI(s)" if args.repair_seconds is not None else "config"
    psrc = "CLI(s)" if args.prerecognition_seconds is not None else "config"
    print(f"  lesion_repair_ticks       : {repair_t} (~{repair_t * tick_seconds:.0f}s) [{rsrc}]")
    print(f"  lesion_prerecognition_ticks: {prerec_ticks} (~{prerec_ticks * tick_seconds:.0f}s) [{psrc}]")
    print(f"    -> steady-state REPAIR (blocking) fraction ~ {repair_frac:.2f}")
    print(f"  boundary x{args.bstr_mult:g} (cap 1.0), lesion_type_a_prob={args.type_a_prob:g}, "
          f"max_rnapii=0, rnapii_load/translocate=null")

    # 1) generate every grid config (always) ------------------------------
    grid_cfg_paths: dict[tuple[float, int], Path] = {}
    for p in p_sorted:
        for sp in sp_sorted:
            cfg = build_grid_config(
                base, p=p, spacing=sp, bstr_mult=args.bstr_mult,
                type_a_prob=args.type_a_prob,
                tick_seconds=tick_seconds, lesion_defaults=lesion_defaults,
                type_b_enabled=type_b_enabled,
            )
            path = cfg_dir / f"{grid_label(p, sp)}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
            grid_cfg_paths[(p, sp)] = path
    print(f"wrote {len(grid_cfg_paths)} grid configs to {cfg_dir}")

    if args.skip_run:
        print("--skip-run set: configs generated, skipping 1D + heatmaps")
        return

    # Topology is identical across baseline + grid (strengths don't move masks),
    # so compute it once and reuse for every analyze_run -> topology-matched folds.
    topo = _topology(base)

    # 2) baseline metric means -------------------------------------------
    baseline = analyze_h5("baseline", args.config, args.h5.resolve(), topo)

    # 3) run + analyze every grid point ----------------------------------
    metrics_seen = list(METRIC_LABELS)
    shape = (len(p_sorted), len(sp_sorted))
    raws = {m: np.full(shape, np.nan) for m in metrics_seen}
    folds = {m: np.full(shape, np.nan) for m in metrics_seen}
    # Measured (not vs baseline): lesion-induced cohesin stall, read from the H5 --
    # both time-averaged occupancy and realized dwell DURATION in seconds.
    measured_keys = [
        "stalled_per_lesion", "stalled_per_blocking_lesion",
        "stalled_legs_per_frame", "lesions_per_frame", "blocking_lesions_per_frame",
        "dwell_mean_s", "dwell_median_s", "dwell_p90_s", "dwell_max_s",
        "n_stall_events", "stalled_legtick_fraction",
    ]
    # Per-(type, state) breakdown: a_pre / a_repair / b_repair x stat.
    for cat in _CATEGORIES:
        measured_keys += [f"{cat}_dwell_mean_s", f"{cat}_dwell_median_s",
                          f"{cat}_dwell_p90_s", f"{cat}_dwell_max_s",
                          f"{cat}_n_events", f"{cat}_legtick_fraction"]
    # Per-(type, state) stall fraction of the lesion's stage lifetime.
    measured_keys += list(MEASURED_STAGEFRAC_LABELS)
    measured = {k: np.full(shape, np.nan) for k in measured_keys}

    tasks = [
        {
            "ri": ri, "ci": ci, "p": p, "sp": sp,
            "label": grid_label(p, sp),
            "cfg_path": str(grid_cfg_paths[(p, sp)]),
            "h5_dir": str(h5_dir),
            "force_1d": args.force_1d,
        }
        for ri, p in enumerate(p_sorted)
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
            # Fold is undefined when the baseline mean is 0 or non-finite (e.g. a
            # metric that never fires in the baseline run); leave NaN so the
            # heatmap masks that cell instead of drawing inf.
            folds[m][ri, ci] = gm / bm if np.isfinite(bm) and bm != 0 else float("nan")
        if not args.no_measure:
            ms = measure_lesion_stall(
                Path(h5_path), tick_seconds, chunk=args.measure_chunk,
                prerecognition_ticks=prerec_ticks, repair_ticks=repair_t)
            for k in measured:
                measured[k][ri, ci] = ms[k]

    if jobs == 1:
        for n, task in enumerate(tasks, 1):
            print(f"[{n}/{total}] {task['label']}  "
                  f"(p={task['p']:.3f}, density={density_axis[task['ci']]:.1f}/Mbp)")
            res = _run_grid_point(task)
            _record(task, res["h5"])
    else:
        # Process pool: each grid point is an independent process that re-seeds
        # numpy from the config and writes its own H5 -> bitwise-identical to
        # sequential, just concurrent. A fork context avoids re-importing this
        # module (and matplotlib) in every worker.
        print(f"running {total} grid points with --jobs {jobs} (process pool, fork context)")
        ctx = mp.get_context("fork")
        by_label = {t["label"]: t for t in tasks}
        n = 0
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for res in ex.map(_run_grid_point, tasks):
                n += 1
                task = by_label[res["label"]]
                print(f"[{n}/{total}] done {res['label']}  "
                      f"(p={task['p']:.3f}, density={density_axis[task['ci']]:.1f}/Mbp)")
                _record(task, res["h5"])

    # 4) outputs ----------------------------------------------------------
    plots_dir = out_dir
    svg_path = plots_dir / "lesion_grid_heatmaps.svg"
    tsv_path = plots_dir / "lesion_grid_folds.tsv"
    y_label = "Lesion block probability p (per tick)"
    # Fold heatmaps: y-axis is now the raw swept knob (lesion_block_prob), not a
    # derived seconds value; colour bar still centred at fold = 1.0.
    plot_heatmaps(
        folds, METRIC_LABELS,
        y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
        out_path=svg_path,
        title=(
            f"Lesion grid vs baseline 1D sim (fold = grid mean / baseline mean)\n"
            f"RNAPII off, boundaries x{args.bstr_mult:g} (cap 1.0), tick={tick_seconds:g}s"
        ),
        value_label="fold vs baseline mean",
        center=1.0,
    )
    write_grid_tsv(
        folds, raws,
        p_values=p_sorted, spacings=sp_sorted,
        stall_axis=stall_axis, raw_bypass_axis=raw_bypass_axis, density_axis=density_axis,
        tick_seconds=tick_seconds, baseline=baseline, out_path=tsv_path,
    )
    print(f"wrote {svg_path}")
    print(f"wrote {tsv_path}")

    # Measured lesion-stall heatmaps (standalone; NOT vs baseline) ---------
    if not args.no_measure:
        # (a) per-lesion occupancy: how busy each lesion is.
        occ_svg = plots_dir / "lesion_grid_measured_stall.svg"
        plot_heatmaps(
            measured, MEASURED_LABELS,
            y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
            out_path=occ_svg,
            title=(
                "Measured lesion-induced cohesin stall (from 1D trajectory, not vs baseline)\n"
                f"per-lesion normalised; RNAPII off, tick={tick_seconds:g}s"
            ),
            value_label="stalled cohesin legs per lesion",
            cell_fmt="{:.3f}",
            center=None,
            cmap=SEQ_CMAP,
        )
        # (b) realized dwell DURATION in seconds: how long a stall lasts.
        sec_svg = plots_dir / "lesion_grid_measured_stall_seconds.svg"
        plot_heatmaps(
            measured, MEASURED_SECONDS_LABELS,
            y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
            out_path=sec_svg,
            title=(
                "Measured cohesin stall DURATION at a lesion (from 1D trajectory)\n"
                f"per stall event, in seconds; RNAPII off, tick={tick_seconds:g}s"
            ),
            value_label="stall duration (s)",
            cell_fmt="{:.0f}",
            center=None,
            cmap=SEQ_CMAP,
        )
        # (c) per-(type, state) breakdown: A-PRE / A-REPAIR / B-REPAIR mean stall.
        bytype_svg = plots_dir / "lesion_grid_measured_stall_by_type.svg"
        plot_heatmaps(
            measured, MEASURED_BYTYPE_LABELS,
            y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
            out_path=bytype_svg,
            title=(
                "Measured cohesin stall by lesion type/state (from 1D trajectory)\n"
                f"mean per-regime stall, in seconds; RNAPII off, tick={tick_seconds:g}s"
            ),
            value_label="stall duration (s)",
            cell_fmt="{:.0f}",
            center=None,
            cmap=SEQ_CMAP,
        )
        # (d) per-(type, state) stall as a FRACTION of the lesion's stage lifetime.
        frac_svg = plots_dir / "lesion_grid_measured_stall_fraction.svg"
        plot_heatmaps(
            measured, MEASURED_STAGEFRAC_LABELS,
            y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
            out_path=frac_svg,
            title=(
                "Measured cohesin stall as a fraction of the lesion's stage lifetime\n"
                f"per-encounter mean stall / stage window; RNAPII off, tick={tick_seconds:g}s"
            ),
            value_label="stall / stage lifetime",
            cell_fmt="{:.2f}",
            center=None,
            cmap=SEQ_CMAP,
        )
        # measured TSV: occupancy + dwell distribution + per-type + analytic ref.
        m_tsv = plots_dir / "lesion_grid_measured_stall.tsv"
        m_rows = []
        for ri, p in enumerate(p_sorted):
            for ci, sp in enumerate(sp_sorted):
                m_rows.append({
                    "lesion_block_prob": p,
                    "lesion_spacing": sp,
                    "lesion_density_per_mbp": density_axis[ci],
                    **{k: measured[k][ri, ci] for k in measured},
                    "effective_stall_seconds_analytic": stall_axis[ri],
                })
        pd.DataFrame(m_rows).to_csv(m_tsv, sep="\t", index=False)
        print(f"wrote {occ_svg}")
        print(f"wrote {sec_svg}")
        print(f"wrote {bytype_svg}")
        print(f"wrote {frac_svg}")
        print(f"wrote {m_tsv}")


if __name__ == "__main__":
    main()
