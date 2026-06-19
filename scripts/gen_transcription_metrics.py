#!/usr/bin/env python
"""Transcription-kinetics metrics for a transcription-ON (RNAPII-enabled) config.

Given ONE config (and optionally its ``LEFPositions.h5``; the 1D stage is run or
reused if absent), this measures per-gene + locus-aggregate transcription rates
DIRECTLY from the RNAPII trajectory -- no analytic formula. It is the
transcription counterpart to ``gen_lesion_grid_and_heatmaps.py``'s measured
cohesin-stall readout, and reuses the same ``_resolve_h5`` 1D runner.

The trajectory exposes (see loop_extrusion/lef.py):

  * ``rnapii_positions`` (T, R, 2) = [global_site, gene_id], -1 = absent.
  * ``rnapii_states``    (T, R)    = PRE-INITIATION/PAUSED/ELONGATING/TERMINATING/STALLED.
  * ``rnapii_ids``       (T, R)    = stable per-Pol uid (survives column reshuffle
                                     on unload) -> reconstructs each polymerase's
                                     load -> unload life, the basis for completion
                                     counts, per-molecule velocity and dwell times.
  * ``genes``            struct    = (gene_id, tss, tes, direction, load_prob).
  * ``lesions`` / ``positions``    (optional) -> co-transcriptional stall attribution.

Each recorded frame is one tick; ``tick_seconds`` (from the config) converts ticks
to seconds. Four metric families are computed per gene and as a locus aggregate:

  CORE RATES & DENSITY
    initiation_rate_per_min    new Pol II loaded at the TSS / time (bursty input)
    completion_rate_per_min    Pol II reaching the TES (= mRNA production rate)
    synthesis_rate_nt_per_s    sites transcribed / time (nascent RNA throughput)
    pol2_occupancy             mean Pol II present per tick (ChIP analog, per allele; incl. the 3' window)
    pol2_density_per_kb        occupancy / gene length (kb)
    nascent_signal             mean ELONGATING|STALLED Pol II per tick (engaged)
    term_zone_occupancy        mean Pol II in the 3' termination window past the TES (per tick)
    elongation_velocity_bp_s   sites advanced / elongating-tick (bp/s; excludes the 3' termination crawl)

  Position classes are window-aware: rel = (pos-TSS)*direction; rel<=pause_offset is the
  promoter/pause, pause_offset<rel<=gene_length is the gene body (sets the pausing-index
  denominator), and rel>gene_length is the 3' termination window (Schwalb 2016) -- the last
  is excluded from gene-body density so the pausing index is not diluted by terminating Pol.

  PAUSING & STATE KINETICS
    pausing_index              (TSS..pause-site) / gene-body density (travelling ratio)
    pausing_index_pausesite    pause-site-only / gene-body density (undiluted pause peak)
    frac_{pre_initiation,paused,elongating,stalled,terminating}   state-time fractions
    mean_{pre_initiation,pause,terminating}_dwell_s               per-state residence (s)
    mean_residence_s           full load->unload transcription time (s)
    pause_release_efficiency   completed / initiated

  CO-TRANSCRIPTIONAL INTERFERENCE
    stall_frac                 STALLED / (ELONGATING+STALLED) engaged-time
    lesion_stall_frac          of STALLED ticks, fraction with a blocking lesion
                               immediately ahead (NaN if no ``lesions`` dataset)
    cohesin_stall_frac         of STALLED ticks, fraction with a cohesin leg
                               immediately ahead (NaN if no ``positions`` dataset)

  TRANSCRIPTIONAL BURSTING
    active_fraction            fraction of ticks the gene holds >=1 Pol II
    burst_rate_per_min         onsets of activity / time
    mean_burst_size            new Pol II initiated per active burst
    mean_burst_duration_s      mean active-interval length (s)
    mean_interburst_s          mean silent-interval length (s)

Outputs (to ``--out-dir``, default alongside the H5):

  * ``transcription_metrics.tsv``        one row per gene + an ``ALL`` aggregate row.
  * ``transcription_metrics.svg``        per-gene panels (rates, density, pausing,
                                         state fractions, velocity, bursting).
  * ``transcription_tracks.svg``         Pol II ChIP + nascent per-site tracks,
                                         folded across alleles onto one locus.
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported (matches the sister
# scripts; the 1D stage is numpy-light and we never want threaded BLAS here).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_config_chain_metrics import _resolve_h5, _safe_label  # noqa: E402
from polychrom.pipelines.loop_extrusion.plugins.lesions import (  # noqa: E402
    LESION_REPAIR,
    LESION_TYPE_A,
)
from polychrom.pipelines.loop_extrusion.plugins.rnapii import (  # noqa: E402
    STATE_ELONGATING,
    STATE_PAUSED,
    STATE_PRE_INITIATION,
    STATE_STALLED,
    STATE_TERMINATING,
)

# Engaged Pol II carries a nascent transcript (productive or transiently
# obstacle-stalled in the gene body); same convention as polii_chip_nascent_track.py.
_ENGAGED = (STATE_ELONGATING, STATE_STALLED)
_STATE_ORDER = [STATE_PRE_INITIATION, STATE_PAUSED, STATE_ELONGATING, STATE_STALLED, STATE_TERMINATING]
_STATE_NAMES = {
    STATE_PRE_INITIATION: "pre_initiation", STATE_PAUSED: "paused", STATE_ELONGATING: "elongating",
    STATE_STALLED: "stalled", STATE_TERMINATING: "terminating",
}

# Per-gene metric columns, in TSV/display order (the aggregate row fills the same).
_RATE_KEYS = [
    "initiation_rate_per_min", "completion_rate_per_min", "synthesis_rate_nt_per_s",
    "pol2_occupancy", "pol2_density_per_kb", "nascent_signal", "term_zone_occupancy",
    "elongation_velocity_bp_s",
]
_PAUSE_KEYS = [
    "pausing_index", "pausing_index_pausesite", "frac_pre_initiation", "frac_paused", "frac_elongating", "frac_stalled",
    "frac_terminating", "mean_pre_initiation_dwell_s", "mean_pause_dwell_s",
    "apparent_pause_duration_s", "mean_terminating_dwell_s", "mean_residence_s",
    "pause_release_efficiency", "productive_fraction", "premature_termination_frac",
]
_INTERFERENCE_KEYS = ["stall_frac", "lesion_stall_frac", "cohesin_stall_frac"]
_BURST_KEYS = [
    "active_fraction", "burst_rate_per_min", "mean_burst_size",
    "mean_burst_duration_s", "mean_interburst_s",
]

# --- human-biology reference ranges -----------------------------------------
# Accepted in-vivo mammalian ranges for the trackable observables, in the SAME
# native units the metric is measured in (so the per-gene median compares
# directly). `scale`/`unit` are display-only. `soft=True` marks activity-
# dependent quantities with a deliberately wide band (plausibility, not a tight
# law).
HUMAN_RANGES: dict[str, dict] = {
    "elongation_velocity_bp_s": dict(
        lo=8.3, hi=100.0, scale=0.06, unit="kb/min", disp="0.5-6 kb/min", soft=False,
        src="genome-wide in-vivo: 0.5-4 kb/min mESC mean ~2; median ~1.5 human; "
            "2-6 kb/min HeLa mode 3.5; cf. typical-gene 1.3-4.3"),
    "mean_pause_dwell_s": dict(
        lo=5.0, hi=60.0, scale=1 / 60, unit="min", disp="~0.1-1 min (single-Pol)", soft=True,
        src="single-Pol kinetic dwell 1/(k_release+k_term): ~42 s human live-cell residence "
            "(Steurer 2018, high conf); mammalian PRO-seq models imply faster, sub-minute (~6-20 s, "
            "less certain). NOT the minutes-scale apparent pause."),
    "apparent_pause_duration_s": dict(
        lo=60.0, hi=900.0, scale=1 / 60, unit="min", disp="~1-15 min (occ/output)", soft=True,
        src="apparent = pause occupancy / PRODUCTIVE initiation (NOT single-Pol dwell): human mRNA "
            "median ~1 min (Gressel 2019); MOUSE 6.9 min (Jonkers 2014); >15 min for many genes "
            "(Shao & Zeitlinger; Krebs). = single-Pol dwell / productive_fraction"),
    "mean_terminating_dwell_s": dict(
        lo=60.0, hi=480.0, scale=1 / 60, unit="min", disp="1-8 min", soft=True,
        src="DERIVED, not directly measured in seconds (medium confidence): HEK293 3' decel to "
            "0.7-0.9 kb/min over ~1-4 kb past the PAS => ~1-6 min (Cortazar 2019; Fong 2015)"),
    "mean_residence_s": dict(
        lo=180.0, hi=2400.0, scale=1 / 60, unit="min", disp="~3-40 min", soft=True,
        src="avg human gene ~24kb at ~1.7-3.8 kb/min -> ~7-15min "
            "elongation + pause + 3' dwell; direct total dwell 179-357s for short genes"),
    "pol2_occupancy": dict(
        lo=0.3, hi=10.0, scale=1.0, unit="Pol/gene", disp="~0.5-1.5 typ, <=10 active", soft=True,
        src="~1.1 elongating Pol/gene body; promoter occupancy <10%; "
            "~2/gene derived from model rates (~0.7 elongating + 3'-stalled barrier)"),
    "pol2_density_per_kb": dict(
        lo=0.005, hi=0.25, scale=1.0, unit="Pol/kb", disp="~0.005-0.25 Pol/kb", soft=True,
        src="derived (occupancy / gene length)"),
    "initiation_rate_per_min": dict(
        lo=0.005, hi=3.0, scale=60.0, unit="/h", disp="~0.3-180 /h (loading; bursty)", soft=True,
        src="LOADING rate incl. ~75-80% abortive Pol -- NOT productive output (compare completion_rate). "
            "Human PRODUCTIVE initiation: median ~1/min mRNA, 0.3 lincRNA, 0.15 eRNA, up to ~87/min (HSPA1A) "
            "(Gressel/Cramer 2019 K562); loading runs ~1/productive_fraction higher. lo=0.005/min floor for least-active expressed genes"),
    "completion_rate_per_min": dict(
        lo=0.005, hi=1.5, scale=60.0, unit="/h", disp="~0.3-90 /h (productive)", soft=True,
        src="PRODUCTIVE initiation = mRNA output (Pol reaching the 3' end): human median ~1/min mRNA, "
            "0.3 lincRNA, 0.15 eRNA, up to ~87/min (Gressel/Cramer 2019 K562). The literature-anchored "
            "band (vs loading above); lo=0.005/min floor for least-active expressed genes"),
    "pausing_index": dict(
        lo=1.5, hi=20.0, scale=1.0, unit="", disp="~2-20 (TSS->pause)", soft=True,
        src="travelling ratio = promoter-proximal (TSS..pause site) / gene-body Pol density; paused"),
    "pausing_index_pausesite": dict(
        lo=2.0, hi=20.0, scale=1.0, unit="", disp="~2-20 (pause site)", soft=True,
        src="pause-site-only travelling ratio (pause peak / gene-body density), undiluted by the TSS"),
    "productive_fraction": dict(
        lo=0.05, hi=0.30, scale=1.0, unit="", disp="~0.15-0.25 (premat. term.)", soft=False,
        src="~75-80% of paused Pol II prematurely terminate -> ~0.20-0.25 productive "
            "(STL-seq Drosophila ~20%; Mukherjee & Guertin mammalian ~25%; cross-species, not direct human)"),
}


def realism_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gene median of each trackable metric vs its human-biology range.

    Flags each as OK / LOW / HIGH (median inside / below / above the band) and
    reports the fraction of genes inside the band."""
    rows = []
    for key, r in HUMAN_RANGES.items():
        if key in df.columns:
            s = pd.to_numeric(df[key], errors="coerce").dropna()
        else:
            continue
        if s.empty:
            continue
        med = float(s.median())
        frac_in = float(((s >= r["lo"]) & (s <= r["hi"])).mean())
        flag = "OK" if r["lo"] <= med <= r["hi"] else ("LOW" if med < r["lo"] else "HIGH")
        rows.append({
            "metric": key, "median_native": med, "median_display": med * r["scale"],
            "unit": r["unit"], "human_range": r["disp"], "flag": flag,
            "frac_genes_in_range": frac_in, "activity_dependent": r["soft"], "source": r["src"],
        })
    return pd.DataFrame(rows)


def print_realism_report(rr: pd.DataFrame) -> None:
    """Pretty-print the realism table (a '*' marks activity-dependent soft bands)."""
    glyph = {"OK": "OK  ✓", "LOW": "LOW ↓", "HIGH": "HIGH↑"}
    print("\n=== human-biology range check (per-gene median; * = activity-dependent, wide band) ===")
    print(f"  {'metric':26s} {'median':>16s}   {'human range':24s} {'flag':6s} {'in-rng':>6s}  source")
    for _, r in rr.iterrows():
        name = r["metric"] + ("*" if r["activity_dependent"] else "")
        med = f"{r['median_display']:.3g} {r['unit']}".strip()
        print(f"  {name:26s} {med:>16s}   {r['human_range']:24s} "
              f"{glyph[r['flag']]:6s} {r['frac_genes_in_range']:>5.0%}  {r['source']}")
    hard = rr[~rr["activity_dependent"]]
    nok = int((hard["flag"] == "OK").sum())
    print(f"  -> hard-range metrics in human range: {nok}/{len(hard)} "
          f"({int((hard['flag']=='HIGH').sum())} high, {int((hard['flag']=='LOW').sum())} low); "
          f"soft (activity-dependent) shown with *")


def activity_report(df: pd.DataFrame) -> dict:
    """Measure the per-gene transcriptional ACTIVITY distribution (the expression
    heavy-tail), which the per-metric range check does NOT capture. A typical
    locus is mostly QUIET with a minority of active genes, spanning orders of
    magnitude (log-normal / Zipf). Expressed-gene initiation should fall in the
    human PRO-seq band 0.2-1.5/min."""
    occ = pd.to_numeric(df["pol2_occupancy"], errors="coerce")
    load = pd.to_numeric(df["initiation_rate_per_min"], errors="coerce")   # LOADING (incl. abortive)
    prod = pd.to_numeric(df["completion_rate_per_min"], errors="coerce")   # PRODUCTIVE (reach 3' = mRNA)
    m = occ.notna()
    occ, load, prod = occ[m], load[m], prod[m]
    f = lambda mask: 100.0 * float(mask.mean())
    silent = occ < 0.3; quiet = (occ >= 0.3) & (occ < 1.0)
    mod = (occ >= 1.0) & (occ < 3.0); act = occ >= 3.0
    nz = occ[occ > 1e-3]
    spread = (np.percentile(nz, 95) / max(np.percentile(nz, 5), 1e-9)) if len(nz) > 5 else float("nan")
    expr = occ >= 1.0
    # The 0.2-1.5/min band is the PRODUCTIVE/effective rate (omega), so compare it to the
    # completion (3'-arrival = mRNA) rate, NOT loading (which counts the ~75% that prematurely
    # terminate at the pause).
    in_prod_band = 100.0 * float(((prod >= 0.2) & (prod <= 1.5) & expr).sum()) / max(int(expr.sum()), 1)
    med_expr_prod = float(prod[expr].median()) if expr.any() else float("nan")
    med_expr_load = float(load[expr].median()) if expr.any() else float("nan")
    print("\n=== transcriptional ACTIVITY distribution (expression heavy-tail) ===")
    print(f"  Pol-occupancy classes: silent(<0.3)={f(silent):.0f}%  quiet(0.3-1)={f(quiet):.0f}%  "
          f"moderate(1-3)={f(mod):.0f}%  active(>3)={f(act):.0f}%")
    print(f"    (a QUIET majority + a small active tail -- log-normal / Zipf)")
    print(f"  activity spread (occupancy p95/p5) = {spread:.0f}x   "
          f"(~10-1000x; expression spans 3-4 orders genome-wide)")
    print(f"  expressed genes: loading median {med_expr_load:.2f}/min; PRODUCTIVE median "
          f"{med_expr_prod:.2f}/min; {in_prod_band:.0f}% PRODUCTIVE in 0.2-1.5/min band")
    quiet_maj = (f(silent) + f(quiet)) >= 30.0
    heavy_tail = (spread >= 10.0) if spread == spread else False
    verdict = ("OK (quiet majority + heavy tail)" if (quiet_maj and heavy_tail)
               else "CHECK: distribution too narrow/uniform" if not heavy_tail
               else "CHECK: too few quiet genes (locus uniformly active)")
    print(f"  -> activity distribution: {verdict}")
    return {"silent_pct": f(silent), "quiet_pct": f(quiet), "moderate_pct": f(mod),
            "active_pct": f(act), "spread_p95_p5": spread,
            "expr_load_median_per_min": med_expr_load,
            "expr_prod_median_per_min": med_expr_prod, "expr_in_prod_band_pct": in_prod_band}


# Per-class productive-output references (Gressel/Cramer 2019 K562): housekeeping/active genes
# approach the mRNA median ~1/min; cell-type-specific genes sit at lincRNA level ~0.3/min; poised
# developmental genes are mostly off, ~0.15/min. Printed as guidance, not a pass/fail flag -- the
# point is to judge output PER CLASS rather than against a single band that conflates the
# correctly-quiet cell-type majority with genuinely under-active housekeeping genes.
_CLASS_OUTPUT_TARGET = {
    "hk_const": "~1/min (mRNA median)", "hk_high": "~1/min+ (active mRNA)",
    "celltype": "~0.3/min (lincRNA / cell-type)", "dev": "~0.15/min (poised / developmental)",
}


def class_breakdown(df: pd.DataFrame, gene_table: np.ndarray) -> "pd.DataFrame | None":
    """Per regulatory-class transcription summary, keyed by the ``gene_class`` tag carried on the
    H5 ``genes`` dataset (emitted by the config generator). Returns None for trajectories that
    predate the tag. Reports, per class: gene count, % silent / % expressed (by occupancy),
    median output (all + expressed), and median Pol II occupancy, alongside the per-class
    literature target."""
    names = getattr(gene_table.dtype, "names", None) or ()
    if "gene_class" not in names:
        return None

    def _dec(v):
        return v.decode("ascii", "ignore") if isinstance(v, bytes) else str(v)

    cls_by_id = {int(r["gene_id"]): _dec(r["gene_class"]) for r in gene_table}
    g = df.copy()
    g["gene_class"] = g["gene_id"].astype(int).map(cls_by_id).fillna("")
    occ = pd.to_numeric(g["pol2_occupancy"], errors="coerce")
    comp = pd.to_numeric(g["completion_rate_per_min"], errors="coerce")
    known = ["hk_const", "hk_high", "celltype", "dev"]
    order = [c for c in known if (g["gene_class"] == c).any()]
    order += sorted(c for c in g["gene_class"].unique() if c and c not in order)
    rows = []
    for c in order:
        m = g["gene_class"] == c
        o, p = occ[m], comp[m]
        expr = o >= 1
        rows.append({
            "gene_class": c, "n": int(m.sum()),
            "pct_silent": round(100.0 * float((o < 0.3).mean()), 1),
            "pct_expressed": round(100.0 * float(expr.mean()), 1),
            "median_completion_per_min": round(float(p.median()), 4),
            "median_completion_expressed": (round(float(p[expr].median()), 4) if expr.any() else float("nan")),
            "median_pol2_occupancy": round(float(o.median()), 3),
            "literature_target": _CLASS_OUTPUT_TARGET.get(c, ""),
        })
    return pd.DataFrame(rows)


def print_class_breakdown(cb: pd.DataFrame) -> None:
    """Pretty-print the per-class table; output is completion_rate_per_min (productive, /min)."""
    print("\n=== per regulatory-class transcription (productive output vs per-class literature) ===")
    print(f"  {'class':10s} {'n':>4s} {'%sil':>5s} {'%expr':>6s} {'med_out':>8s} {'out_expr':>9s}  literature target")
    for _, r in cb.iterrows():
        oe = r["median_completion_expressed"]
        print(f"  {r['gene_class']:10s} {int(r['n']):>4d} {r['pct_silent']:>4.0f}% {r['pct_expressed']:>5.0f}% "
              f"{r['median_completion_per_min']:>8.3f} {oe:>9.3f}  {r['literature_target']}")
    print("  -> housekeeping should approach ~1/min; cell-type ~0.3; developmental ~0.15 "
          "(per-class medians, not a single band)")


class _UidTrack:
    """Accumulated life of one polymerase (keyed by its stable uid)."""

    __slots__ = ("gene", "first_frame", "last_frame", "prev_pos", "adv_sites",
                 "n_pre_initiation", "n_paused", "n_pause_episodes", "n_elong", "n_term",
                 "in_pause", "reached_terminating")

    def __init__(self, gene: int, frame: int, pos: int):
        self.gene = gene
        self.first_frame = frame
        self.last_frame = frame
        self.prev_pos = pos
        self.adv_sites = 0
        self.n_pre_initiation = 0
        self.n_paused = 0
        self.n_pause_episodes = 0
        self.n_elong = 0
        self.n_term = 0
        self.in_pause = False
        self.reached_terminating = False


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) runs where the boolean timeseries is True."""
    if not active.any():
        return []
    d = np.diff(active.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if active[0]:
        starts = [0] + starts
    if active[-1]:
        ends = ends + [active.size]
    return list(zip(starts, ends))


def measure_transcription(
    h5_path: Path, tick_seconds: float, *, pause_offset: int = 1, chunk: int = 2000
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Per-gene transcription metrics measured from a 1D RNAPII trajectory.

    Returns ``(per_gene_df, aggregate_row, chip_track, nascent_track, gene_table)``.
    The two tracks are per-allele mean occupancy folded onto one locus (length
    ``chain_length``); ``gene_table`` is the structured ``genes`` dataset.
    """
    with h5py.File(h5_path, "r") as h:
        if not bool(h.attrs.get("rnapii_enabled", False)):
            raise SystemExit(f"{h5_path}: RNAPII not enabled in this run (tx-OFF config).")
        if "genes" not in h:
            raise SystemExit(f"{h5_path}: no 'genes' dataset; nothing to transcribe.")
        N = int(h.attrs["N"])
        chain = int(h.attrs["chain_length"])
        num_chains = int(h.attrs.get("num_chains", 1))
        genes = h["genes"][:]
        n_genes = genes.shape[0]
        gids = genes["gene_id"].astype(int)
        tss = genes["tss"].astype(int)
        tes = genes["tes"].astype(int)
        direction = genes["direction"].astype(int)
        glen = np.abs(tes - tss)                       # sites == kb (1 monomer = 1 kb)
        gidx = np.full(int(gids.max()) + 1, -1, dtype=np.int64)
        gidx[gids] = np.arange(n_genes)

        T = int(h["rnapii_positions"].shape[0])
        has_lesions = "lesions" in h
        has_cohesin = "positions" in h

        # Per-gene per-frame occupancy (any present Pol / engaged Pol).
        present_ct = np.zeros((n_genes, T), dtype=np.int32)
        engaged_ct = np.zeros((n_genes, T), dtype=np.int32)
        state_ticks = np.zeros((n_genes, 5), dtype=np.int64)   # cols indexed by state code
        tss_pause_ct = np.zeros(n_genes, dtype=np.int64)   # Pol in [TSS, pause site]: rel in [0, pause_offset]
        pausesite_ct = np.zeros(n_genes, dtype=np.int64)   # Pol exactly at the pause site: rel == pause_offset
        body_ct = np.zeros(n_genes, dtype=np.int64)        # Pol in the gene body: pause_offset < rel <= glen
        term_zone_ct = np.zeros(n_genes, dtype=np.int64)   # Pol in the 3' termination window: rel > glen (past the TES)
        stall_total = np.zeros(n_genes, dtype=np.int64)
        stall_lesion = np.zeros(n_genes, dtype=np.int64)
        stall_cohesin = np.zeros(n_genes, dtype=np.int64)
        chip_fold = np.zeros(chain, dtype=np.float64)
        nascent_fold = np.zeros(chain, dtype=np.float64)
        tracks: dict[int, _UidTrack] = {}

        ds_pos, ds_st, ds_id = h["rnapii_positions"], h["rnapii_states"], h["rnapii_ids"]
        ds_les = h["lesions"] if has_lesions else None
        ds_lty = h["lesion_types"] if has_lesions else None
        ds_lst = h["lesion_states"] if has_lesions else None
        ds_lef = h["positions"] if has_cohesin else None

        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            pos = ds_pos[start:end]
            st = ds_st[start:end]
            ids = ds_id[start:end]
            les = ds_les[start:end] if has_lesions else None
            lty = ds_lty[start:end] if has_lesions else None
            lst = ds_lst[start:end] if has_lesions else None
            lef = ds_lef[start:end] if has_cohesin else None

            for k in range(end - start):
                gf = start + k
                site = pos[k, :, 0]
                present = site >= 0
                if not present.any():
                    continue
                sp = site[present]
                gp = gidx[pos[k, present, 1]]
                stp = st[k][present].astype(int)
                idp = ids[k][present]

                np.add.at(present_ct[:, gf], gp, 1)
                eng = np.isin(stp, _ENGAGED)
                np.add.at(engaged_ct[:, gf], gp[eng], 1)
                np.add.at(state_ticks, (gp, stp), 1)
                # pausing index from 1D position: rel = distance past TSS toward TES.
                # The pause sits at rel == pause_offset; the body is pause_offset < rel <= glen.
                # Pol with rel > glen are PAST the TES in the 3' termination window (they crawl
                # there before release) -- they must NOT count toward gene-body density, else the
                # pausing index is diluted by the terminating Pol the window now lets exist.
                rel = (sp - tss[gp]) * direction[gp]
                glen_p = glen[gp]
                np.add.at(tss_pause_ct, gp[rel <= pause_offset], 1)
                np.add.at(pausesite_ct, gp[rel == pause_offset], 1)
                np.add.at(body_ct, gp[(rel > pause_offset) & (rel <= glen_p)], 1)
                np.add.at(term_zone_ct, gp[rel > glen_p], 1)
                # per-site ChIP / nascent tracks, folded onto one locus
                np.add.at(chip_fold, sp % chain, 1.0)
                np.add.at(nascent_fold, sp[eng] % chain, 1.0)

                # Co-transcriptional stall attribution: paint blocking lesions and
                # cohesin legs onto the lattice, then look one site ahead of each
                # STALLED Pol (it stalls just upstream of the obstacle).
                les_block = None
                if has_lesions:
                    s = les[k]
                    v = (s >= 0) & (s < N)
                    sv, tv, stv = s[v], lty[k][v], lst[k][v]
                    block = (stv == LESION_REPAIR) | (tv == LESION_TYPE_A)
                    les_block = np.zeros(N, dtype=bool)
                    les_block[sv[block]] = True
                coh = None
                if has_cohesin:
                    legs = lef[k].reshape(-1)
                    legs = legs[(legs >= 0) & (legs < N)]
                    coh = np.zeros(N, dtype=bool)
                    coh[legs] = True

                # Per-uid bookkeeping (stateful in prev_pos / pause episodes).
                for j in range(sp.size):
                    g = int(gp[j])
                    s_code = int(stp[j])
                    p = int(sp[j])
                    uid = int(idp[j])
                    tr = tracks.get(uid)
                    if tr is None:
                        tr = tracks[uid] = _UidTrack(g, gf, p)
                    tr.last_frame = gf
                    if s_code == STATE_PRE_INITIATION:
                        tr.n_pre_initiation += 1
                    elif s_code == STATE_PAUSED:
                        tr.n_paused += 1
                        if not tr.in_pause:
                            tr.n_pause_episodes += 1
                            tr.in_pause = True
                    elif s_code == STATE_TERMINATING:
                        tr.n_term += 1
                        tr.reached_terminating = True
                    if s_code != STATE_PAUSED:
                        tr.in_pause = False
                    if s_code == STATE_ELONGATING:
                        tr.n_elong += 1
                    # forward progress (sites) for velocity / synthesis throughput. EXCLUDE the
                    # 3' termination-window crawl (STATE_TERMINATING): those sites advance at the
                    # decelerated termination rate, not productive elongation, and counting them
                    # against elongating-only ticks would inflate elongation_velocity.
                    adv = (p - tr.prev_pos) * direction[g]
                    if adv > 0 and s_code == STATE_ELONGATING:
                        tr.adv_sites += int(adv)
                    tr.prev_pos = p
                    # stall attribution
                    if s_code == STATE_STALLED:
                        stall_total[g] += 1
                        ahead = p + int(direction[g])
                        if 0 <= ahead < N:
                            if les_block is not None and les_block[ahead]:
                                stall_lesion[g] += 1
                            elif coh is not None and coh[ahead]:
                                stall_cohesin[g] += 1

    # ---- reduce per-uid tracks to per-gene aggregates --------------------
    init_ct = np.zeros(n_genes, dtype=np.int64)        # loaded in-window (uncensored start)
    completed_ct = np.zeros(n_genes, dtype=np.int64)   # reached TES (terminating)
    aborted_ct = np.zeros(n_genes, dtype=np.int64)     # paused then premature-terminated (never elongated)
    adv_sites = np.zeros(n_genes, dtype=np.int64)
    pre_initiation_dwell: list[list[float]] = [[] for _ in range(n_genes)]
    term_dwell: list[list[float]] = [[] for _ in range(n_genes)]
    residence: list[list[float]] = [[] for _ in range(n_genes)]
    pause_ticks = np.zeros(n_genes, dtype=np.int64)
    pause_episodes = np.zeros(n_genes, dtype=np.int64)
    for tr in tracks.values():
        g = tr.gene
        adv_sites[g] += tr.adv_sites
        pause_ticks[g] += tr.n_paused
        pause_episodes[g] += tr.n_pause_episodes
        new = tr.first_frame > 0                       # not present at frame 0
        if new:
            init_ct[g] += 1
            pre_initiation_dwell[g].append(tr.n_pre_initiation * tick_seconds)
        if tr.reached_terminating:
            completed_ct[g] += 1
            term_dwell[g].append(tr.n_term * tick_seconds)
            if new:
                residence[g].append((tr.last_frame - tr.first_frame) * tick_seconds)
        elif new and tr.n_elong == 0 and tr.n_paused > 0:
            # loaded, paused, never elongated, gone before completing -> promoter-
            # proximal premature termination (the abortive fate).
            aborted_ct[g] += 1

    t_obs = T * tick_seconds
    elong_ticks = state_ticks[:, STATE_ELONGATING].astype(float)
    total_ticks = state_ticks.sum(axis=1).astype(float)

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    rows = []
    for g in range(n_genes):
        occ = present_ct[g].sum() / T
        nasc = engaged_ct[g].sum() / T
        prox_window = pause_offset + 1                      # TSS through the pause site (sites)
        body_len = max(glen[g] - prox_window, 1)
        body_d = body_ct[g] / body_len
        st_tot = total_ticks[g] if total_ticks[g] > 0 else float("nan")
        engaged_t = elong_ticks[g] + state_ticks[g, STATE_STALLED]
        row = {
            "gene_id": int(gids[g]),
            "tss": int(tss[g]), "tes": int(tes[g]),
            "chain_relative_tss": int(tss[g] % chain),
            "direction": int(direction[g]),
            "gene_length_kb": float(glen[g]),
            # core rates & density
            "initiation_rate_per_min": init_ct[g] / t_obs * 60.0,
            "completion_rate_per_min": completed_ct[g] / t_obs * 60.0,
            "synthesis_rate_nt_per_s": adv_sites[g] * 1000.0 / t_obs,
            "pol2_occupancy": occ,
            "pol2_density_per_kb": occ / max(glen[g], 1),
            "nascent_signal": nasc,
            # mean Pol II in the 3' termination window (past the TES). The window lets a terminating
            # Pol vacate the single TES slot, so the 3' end no longer jams; this is the 3' Pol
            # accumulation that also reinforces the cohesin barrier downstream of the gene.
            "term_zone_occupancy": term_zone_ct[g] / T,
            "elongation_velocity_bp_s": (adv_sites[g] / elong_ticks[g] * 1000.0 / tick_seconds
                                         if elong_ticks[g] > 0 else float("nan")),
            # pausing & state kinetics
            # two windows: TSS->pause-site (standard travelling-ratio window) and the
            # pause site alone (the pause peak, undiluted by the ~empty TSS site).
            "pausing_index": ((tss_pause_ct[g] / prox_window) / body_d if body_d > 0 else float("nan")),
            "pausing_index_pausesite": (pausesite_ct[g] / body_d if body_d > 0 else float("nan")),
            "mean_pre_initiation_dwell_s": _mean(pre_initiation_dwell[g]),
            # single-Pol pause dwell: real time one Pol II stays paused before it
            # releases OR terminates (1/(k_release+k_termination); ~0.4 min, short).
            "mean_pause_dwell_s": (pause_ticks[g] / pause_episodes[g] * tick_seconds
                                   if pause_episodes[g] > 0 else float("nan")),
            # apparent pause duration (occupancy/output): pause occupancy
            # divided by PRODUCTIVE initiation -- inflated ~1/productive_fraction-fold
            # vs the single-Pol dwell because terminated Pol II are dropped from output.
            "apparent_pause_duration_s": (state_ticks[g, STATE_PAUSED] * tick_seconds / completed_ct[g]
                                          if completed_ct[g] > 0 else float("nan")),
            "mean_terminating_dwell_s": _mean(term_dwell[g]),
            "mean_residence_s": _mean(residence[g]),
            "pause_release_efficiency": (completed_ct[g] / init_ct[g]
                                         if init_ct[g] > 0 else float("nan")),
            "productive_fraction": (completed_ct[g] / init_ct[g]
                                    if init_ct[g] > 0 else float("nan")),
            "premature_termination_frac": (aborted_ct[g] / init_ct[g]
                                           if init_ct[g] > 0 else float("nan")),
            # co-transcriptional interference
            "stall_frac": (state_ticks[g, STATE_STALLED] / engaged_t
                           if engaged_t > 0 else float("nan")),
            "lesion_stall_frac": (stall_lesion[g] / stall_total[g]
                                  if has_lesions and stall_total[g] > 0 else float("nan")),
            "cohesin_stall_frac": (stall_cohesin[g] / stall_total[g]
                                   if has_cohesin and stall_total[g] > 0 else float("nan")),
        }
        for code in _STATE_ORDER:
            row[f"frac_{_STATE_NAMES[code]}"] = (state_ticks[g, code] / st_tot
                                                 if st_tot == st_tot else float("nan"))
        # bursting (from the per-gene presence timeseries)
        active = present_ct[g] > 0
        bursts = _runs(active)
        gaps = _runs(~active)
        row["active_fraction"] = float(active.mean())
        row["burst_rate_per_min"] = len(bursts) / t_obs * 60.0
        row["mean_burst_size"] = (init_ct[g] / len(bursts) if bursts else float("nan"))
        row["mean_burst_duration_s"] = (_mean([(e - s) * tick_seconds for s, e in bursts])
                                        if bursts else float("nan"))
        row["mean_interburst_s"] = (_mean([(e - s) * tick_seconds for s, e in gaps])
                                    if gaps else float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Locus aggregate: rates/counts sum, intensive quantities are occupancy-weighted.
    agg = _aggregate(df, n_alleles=n_genes)
    chip_track = chip_fold / (T * max(num_chains, 1))
    nascent_track = nascent_fold / (T * max(num_chains, 1))
    return df, agg, chip_track, nascent_track, genes


def _aggregate(df: pd.DataFrame, *, n_alleles: int) -> dict:
    """Locus-level summary row. Extensive rates are summed across genes; intensive
    quantities (densities, fractions, dwell times, velocity) are averaged so the
    row reads as 'the typical gene' rather than a sum of incommensurate ratios."""
    agg: dict = {"gene_id": "ALL", "tss": -1, "tes": -1, "chain_relative_tss": -1,
                 "direction": 0, "gene_length_kb": float(df["gene_length_kb"].mean())}
    summed = {"initiation_rate_per_min", "completion_rate_per_min",
              "synthesis_rate_nt_per_s", "burst_rate_per_min"}
    for col in df.columns:
        if col in agg:
            continue
        if col in summed:
            agg[col] = float(df[col].sum())
        else:
            agg[col] = float(np.nanmean(df[col].to_numpy(dtype=float)))
    agg["n_genes"] = n_alleles
    return agg


# Each panel: (metric column, descriptive title, x-axis label) — the band/scale come
# from HUMAN_RANGES so every panel shades the human range and is self-explaining.
# (column, title, x-label, log-x, display-scale [None -> HUMAN_RANGES scale])
_DIST_PANELS = [
    ("elongation_velocity_bp_s", "Elongation velocity\n(Pol II speed along the gene)", "kb / min", False, None),
    ("mean_pause_dwell_s", "Promoter-proximal pause\n(time Pol II pauses near the TSS)", "minutes", False, None),
    ("mean_terminating_dwell_s", "3′ termination dwell\n(time Pol II spends at the gene 3′ end)", "minutes", False, None),
    ("initiation_rate_per_min", "Initiation rate\n(new Pol II loaded)", "events / h", False, None),
    ("pol2_occupancy", "Pol II per gene\n(occupancy ≈ Pol II ChIP)", "polymerases / gene", True, None),
    ("pausing_index", "Pausing index\n(promoter ÷ gene-body density)", "ratio", False, None),
    ("completion_rate_per_min", "mRNA production\n(productive Pol II reaching the 3′ end)", "mRNA / h", False, 60.0),
    ("synthesis_rate_nt_per_s", "Nascent RNA synthesis\n(engaged Pol II throughput ≈ GRO/PRO-seq)", "nt / s", True, 1.0),
]


def _dist_panel(ax, vals, key, title, xlabel, logx, scale=None):
    """One distribution panel: histogram + median line. ``scale`` overrides the
    HUMAN_RANGES display scale for metrics not in that table (e.g. mRNA output)."""
    sc = scale if scale is not None else HUMAN_RANGES.get(key, {}).get("scale", 1.0)
    v = np.asarray(pd.to_numeric(vals, errors="coerce"), float) * sc
    v = v[np.isfinite(v)]
    if logx:
        v = v[v > 0]
        ax.set_xscale("log")
        bins = np.logspace(np.log10(max(v.min(), 1e-3)), np.log10(v.max()), 28) if v.size else 12
        vplot = v
    else:
        # clip a heavy tail to the 99th pct so a few outliers don't stretch the axis
        hi_clip = np.percentile(v, 99) if v.size else 1.0
        vplot = np.clip(v, None, hi_clip)
        bins = np.linspace(min(v.min(), 0) if v.size else 0, hi_clip, 26)
    ax.hist(vplot, bins=bins, color="#6f8fbf", edgecolor="white", linewidth=0.3, zorder=2)
    med = float(np.median(v)) if v.size else np.nan
    ax.axvline(med, color="#A63446", lw=2.0, zorder=3)
    left = med < (np.nanmean(ax.get_xlim()) if not logx else 10 ** np.mean(np.log10(ax.get_xlim())))
    ax.annotate(f"median\n{med:.2g}", xy=(med, 0.97), xycoords=("data", "axes fraction"),
                ha="left" if left else "right", va="top", fontsize=9, color="#A63446", fontweight="bold")
    ax.set_title(title, fontsize=11.5, pad=5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("number of genes", fontsize=9.5)
    ax.tick_params(labelsize=9)


def plot_distributions(df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    """Publication figure: per-gene distributions of each transcription metric
    (median marked in red), including mRNA production and nascent-RNA synthesis,
    an occupancy->output scatter (Pol II density vs mRNA), where Pol II spends its
    time, and the activity (expression) distribution. Each panel is self-contained:
    the title says what it is and the x-axis gives the unit."""
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 17))
    for ax, (key, ttl, xl, logx, sc) in zip(axes.flat[:8], _DIST_PANELS):
        _dist_panel(ax, df.get(key), key, ttl, xl, logx, sc)

    # Panel 9: occupancy -> output (does Pol II density translate into mRNA, or saturate?).
    ax = axes.flat[8]
    occ_s = pd.to_numeric(df.get("pol2_occupancy"), errors="coerce")
    out_s = pd.to_numeric(df.get("completion_rate_per_min"), errors="coerce") * 60.0
    m = np.isfinite(occ_s) & np.isfinite(out_s) & (occ_s > 0) & (out_s > 0)
    ax.scatter(occ_s[m], out_s[m], s=6, alpha=0.25, color="#3b6b8a", edgecolor="none", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Occupancy → output\n(does Pol II density yield mRNA?)", fontsize=11.5, pad=5)
    ax.set_xlabel("Pol II per gene (occupancy)", fontsize=10)
    ax.set_ylabel("mRNA / h", fontsize=9.5); ax.tick_params(labelsize=9)

    # Panel 10: where Pol II spends its time (mean state-time fractions, one stacked bar).
    ax = axes.flat[9]
    palette = [("pre_initiation", "#c9d6df"), ("paused", "#f2c14e"), ("elongating", "#3b8a5a"),
               ("stalled", "#A63446"), ("terminating", "#6b4e9e")]
    left = 0.0
    for name, color in palette:
        frac = float(np.nanmean(pd.to_numeric(df.get(f"frac_{name}"), errors="coerce")))
        ax.barh(0, frac, left=left, color=color, edgecolor="white")
        if frac > 0.04:
            ax.text(left + frac / 2, 0, f"{name}\n{frac*100:.0f}%", ha="center", va="center",
                    fontsize=8.5, color="white" if name != "pre_initiation" else "#333")
        left += frac
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.set_title("Where Pol II spends its time\n(mean fraction per state)", fontsize=11.5, pad=5)
    ax.set_xlabel("fraction of polymerase-time", fontsize=10); ax.tick_params(labelsize=9)

    # Panel 11: activity (expression) distribution across genes — quiet majority + active tail.
    ax = axes.flat[10]
    occ = pd.to_numeric(df.get("pol2_occupancy"), errors="coerce").dropna()
    cats = [("silent\n<0.3", (occ < 0.3).mean(), "#c9d6df"),
            ("quiet\n0.3–1", ((occ >= 0.3) & (occ < 1)).mean(), "#9ab0cc"),
            ("moderate\n1–3", ((occ >= 1) & (occ < 3)).mean(), "#6f8fbf"),
            ("active\n>3", (occ >= 3).mean(), "#A63446")]
    ax.bar(range(4), [c[1] * 100 for c in cats], color=[c[2] for c in cats], edgecolor="white")
    for i, c in enumerate(cats):
        ax.text(i, c[1] * 100 + 1, f"{c[1]*100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cats], fontsize=9)
    ax.set_ylabel("% of genes", fontsize=9.5)
    ax.set_title("Activity (expression) across genes\n(quiet majority + active tail)",
                 fontsize=11.5, pad=5); ax.tick_params(labelsize=9)

    # Panel 12: plain-language summary.
    ax = axes.flat[11]; ax.axis("off")
    spread = (occ.quantile(0.95) / max(occ.quantile(0.05), 1e-6)) if len(occ) else float("nan")
    lines = [
        "How to read this figure",
        "",
        "• red line = median across genes",
        "• distributions, not per-gene bars,",
        "   because most genes are quiet and a",
        "   few are highly active (log-normal).",
        "",
        f"genes measured: {len(df)}",
        f"median Pol II / gene: {occ.median():.2f}",
        f"activity spread (p95/p5): {spread:.0f}×",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, family="monospace")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Genome-browser metric tracks: (column, label, unit, display scale, colour, log-y).
# Heavy-tailed metrics (occupancy, pause, pausing index) use a LOG y-axis so a few
# outliers don't squash the typical genes.
_GENOME_TRACKS = [
    ("pol2_occupancy", "Pol II / gene", "Pol", 1.0, "#465775", True),
    ("initiation_rate_per_min", "initiation", "/min", 1.0, "#1b6ca8", False),
    ("elongation_velocity_bp_s", "elongation", "kb/min", 0.06, "#3b8a5a", False),
    ("mean_pause_dwell_s", "pause", "min", 1 / 60, "#c77d18", True),
    ("pausing_index", "pausing index", "ratio", 1.0, "#6b4e9e", True),
]


def _draw_boundaries(ax, boundaries, *, label=False, top=False):
    """Mark TAD boundaries as vertical dashed lines; optionally label them (kb)."""
    for b in boundaries or []:
        ax.axvline(b, color="#c0392b", lw=0.9, ls="--", alpha=0.55, zorder=1)
    if label and top and boundaries and len(boundaries) <= 16:
        for b in boundaries:
            ax.annotate(f"{b}", xy=(b, 1.02), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7.5, color="#c0392b")


def plot_genome_tracks(df: pd.DataFrame, out_path: Path, *, title: str,
                       boundaries=None, chain: int | None = None) -> None:
    """Publication figure (genome-browser view): each transcription metric plotted
    ALONG the genome coordinate -- one marker per gene at its TSS (alleles averaged) --
    with TAD boundaries marked (red dashed). Shows WHERE active genes and metric
    outliers sit relative to TADs."""
    g = df.copy()
    g["pos"] = pd.to_numeric(g["chain_relative_tss"], errors="coerce")
    agg = g.dropna(subset=["pos"]).groupby("pos").mean(numeric_only=True).reset_index().sort_values("pos")
    pos = agg["pos"].to_numpy()
    n = len(_GENOME_TRACKS)
    fig, axes = plt.subplots(n, 1, figsize=(15, 1.9 * n + 0.8), sharex=True)
    for ax, (key, lab, unit, sc, col, logy) in zip(axes, _GENOME_TRACKS):
        if key not in agg.columns:
            continue
        y = pd.to_numeric(agg[key], errors="coerce").to_numpy() * sc
        if logy:
            yy = np.where(y > 0, y, np.nan)
            ax.set_yscale("log")
            base = np.nanmin(yy) if np.isfinite(yy).any() else 1.0
            ax.vlines(pos, base, yy, color=col, lw=0.8, alpha=0.35, zorder=2)
            ax.scatter(pos, yy, s=18, color=col, zorder=3, edgecolor="white", linewidth=0.3)
            ax.set_ylabel(f"{lab}\n({unit}, log)", fontsize=10)
        else:
            ax.vlines(pos, 0, y, color=col, lw=1.0, alpha=0.5, zorder=2)
            ax.scatter(pos, y, s=18, color=col, zorder=3, edgecolor="white", linewidth=0.3)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(f"{lab}\n({unit})", fontsize=10)
        _draw_boundaries(ax, boundaries)
        ax.tick_params(labelsize=8.5); ax.margins(x=0.01)
    _draw_boundaries(axes[0], boundaries, label=True, top=True)
    axes[0].annotate("red dashed = TAD boundary",
                     xy=(0.99, 1.16), xycoords="axes fraction", ha="right", fontsize=9,
                     style="italic", color="#555")
    axes[-1].set_xlabel("genome coordinate (kb; all alleles folded onto one chain)", fontsize=11.5)
    if chain:
        axes[-1].set_xlim(0, chain)
    fig.suptitle(title, fontsize=14.5, y=0.995)
    fig.text(0.5, 0.004, "one point per gene at its TSS (alleles averaged); height = metric value",
             ha="center", fontsize=9, style="italic", color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def plot_tracks(chip: np.ndarray, nascent: np.ndarray, genes: np.ndarray,
                chain: int, out_path: Path, *, title: str, boundaries=None) -> None:
    """Pol II ChIP (all present) and nascent (engaged) per-site tracks, folded
    across alleles onto one locus; gene bodies shaded; TAD boundaries marked."""
    xs = np.arange(chain)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    a1.fill_between(xs, chip, color="#465775", lw=0)
    a1.set_ylabel("Pol II per site\n(per allele)", fontsize=10.5)
    a1.set_title("Pol II occupancy  —  all polymerases (ChIP-seq analog)", fontsize=12, loc="left")
    a2.fill_between(xs, nascent, color="#3b8a5a", lw=0)
    a2.set_ylabel("nascent RNA per site\n(per allele)", fontsize=10.5)
    a2.set_title("Nascent RNA  —  engaged (elongating + stalled) Pol II (GRO/PRO-seq analog)",
                 fontsize=12, loc="left")
    for g in genes:
        lo, hi = sorted((int(g["tss"]) % chain, int(g["tes"]) % chain))
        for ax in (a1, a2):
            ax.axvspan(lo, hi, color="#000000", alpha=0.06, lw=0)
    for ax in (a1, a2):
        _draw_boundaries(ax, boundaries)
    _draw_boundaries(a1, boundaries, label=True, top=True)
    a2.set_xlabel("genome coordinate (kb; all alleles folded onto one chain)", fontsize=11)
    for ax in (a1, a2):
        ax.tick_params(labelsize=9)
        ax.margins(x=0)
    a1.annotate("grey = gene bodies · red dashed = TAD boundary · sharp 5′ peaks = promoter Pol II",
                xy=(0.5, 1.20), xycoords="axes fraction", ha="center", fontsize=9,
                style="italic", color="#555")
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="transcription-ON config YAML")
    ap.add_argument("--h5", type=Path, default=None,
                    help="LEFPositions.h5 for --config (default: run/reuse next to --out-dir)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory for TSV + figures (default: the H5's parent)")
    ap.add_argument("--force-1d", action="store_true",
                    help="rerun the 1D stage even if a matching H5 exists")
    ap.add_argument("--pause-offset", type=int, default=1,
                    help="pause site is this many sites (kb) downstream of the TSS "
                         "(matches gene.pause_offset); the pausing index uses TSS->pause "
                         "and pause-site-only windows")
    ap.add_argument("--measure-chunk", type=int, default=2000,
                    help="frames per streamed block when reading the trajectory")
    ap.add_argument("--no-plot", action="store_true", help="write only the TSV")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(args.config.read_text())
    lef = base["lef"]
    tick_seconds = float(lef.get("tick_seconds") or 0.0)
    if tick_seconds <= 0:
        raise SystemExit("config lef.tick_seconds must be a positive number")

    label = _safe_label(args.config.stem)
    out_dir = args.out_dir or (args.h5.parent if args.h5 else args.config.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = _resolve_h5(
        cfg_path=args.config, requested_h5=args.h5.resolve() if args.h5 else None,
        out_dir=out_dir, label=label, force_1d=args.force_1d,
    )

    print(f"config       : {args.config}")
    print(f"trajectory   : {h5_path}")
    print(f"tick_seconds : {tick_seconds:g}")
    df, agg, chip, nascent, genes = measure_transcription(
        h5_path, tick_seconds, pause_offset=args.pause_offset, chunk=args.measure_chunk)

    tsv = out_dir / "transcription_metrics.tsv"
    out_df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    out_df.to_csv(tsv, sep="\t", index=False)
    print(f"\nmeasured {len(df)} gene(s); locus aggregate:")
    for k in ("initiation_rate_per_min", "completion_rate_per_min", "synthesis_rate_nt_per_s",
              "pol2_occupancy", "elongation_velocity_bp_s", "pausing_index",
              "stall_frac", "active_fraction"):
        print(f"  {k:28s}: {agg[k]:.4g}")
    print(f"wrote {tsv}")

    # Human-biology range check (the headline "are my transcription numbers in range?").
    rr = realism_report(df)
    print_realism_report(rr)
    act = activity_report(df)
    realism_tsv = out_dir / "transcription_realism.tsv"
    rr.to_csv(realism_tsv, sep="\t", index=False)
    pd.DataFrame([act]).to_csv(out_dir / "transcription_activity.tsv", sep="\t", index=False)
    print(f"wrote {realism_tsv} + transcription_activity.tsv")

    # Per regulatory-class breakdown (when the trajectory carries the gene_class tag).
    cb = class_breakdown(df, genes)
    if cb is not None:
        print_class_breakdown(cb)
        class_tsv = out_dir / "transcription_class_breakdown.tsv"
        cb.to_csv(class_tsv, sep="\t", index=False)
        print(f"wrote {class_tsv}")

    if not args.no_plot:
        with h5py.File(h5_path, "r") as h:
            chain = int(h.attrs["chain_length"])
        # TAD boundaries (chain-relative) for the genome-coordinate plots. Internal
        # boundaries appear as adjacent edge pairs (~1 kb apart); merge them and drop
        # the chromosome ends so each boundary is one labelled line.
        tads = lef.get("topology_kwargs", {}).get("tads", [])
        raw_b = sorted({int(t["left"]) for t in tads} | {int(t["right"]) for t in tads})
        boundaries: list[int] = []
        for b in raw_b:
            if 3 < b < chain - 3 and (not boundaries or b - boundaries[-1] > 5):
                boundaries.append(b)
        sub = f"{args.config.name}, tick={tick_seconds:g}s"
        gene_svg = out_dir / "transcription_metrics.svg"
        plot_genome_tracks(df, gene_svg, title=f"Transcription metrics along the genome\n{sub}",
                           boundaries=boundaries, chain=chain)
        dist_svg = out_dir / "transcription_distributions.svg"
        plot_distributions(df, dist_svg,
                           title=f"Transcription metric distributions\n{sub}")
        track_svg = out_dir / "transcription_tracks.svg"
        plot_tracks(chip, nascent, genes, chain, track_svg,
                    title=f"Pol II ChIP & nascent RNA tracks\n{sub}", boundaries=boundaries)
        print(f"wrote {gene_svg}, {dist_svg}, {track_svg}")


if __name__ == "__main__":
    main()
