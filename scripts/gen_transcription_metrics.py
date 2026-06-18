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
    pol2_occupancy             mean Pol II present per tick (ChIP analog, per allele)
    pol2_density_per_kb        occupancy / gene length (kb)
    nascent_signal             mean ELONGATING|STALLED Pol II per tick (engaged)
    elongation_velocity_bp_s   sites advanced / elongating-tick (bp/s)

  PAUSING & STATE KINETICS
    pausing_index              promoter-proximal density / gene-body density
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
    "pol2_occupancy", "pol2_density_per_kb", "nascent_signal", "elongation_velocity_bp_s",
]
_PAUSE_KEYS = [
    "pausing_index", "frac_pre_initiation", "frac_paused", "frac_elongating", "frac_stalled",
    "frac_terminating", "mean_pre_initiation_dwell_s", "mean_pause_dwell_s",
    "mean_terminating_dwell_s", "mean_residence_s", "pause_release_efficiency",
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
# law). Sources include Banigan et al. 2023 PNAS's own simulation values (the
# model these configs are calibrated to) for direct cross-reference.
HUMAN_RANGES: dict[str, dict] = {
    "elongation_velocity_bp_s": dict(
        lo=21.7, hi=71.7, scale=0.06, unit="kb/min", disp="1.3-4.3 kb/min", soft=False,
        src="Jonkers2014/Fuchs2014/Veloso2014; Banigan vp=6.0"),
    "mean_pause_dwell_s": dict(
        lo=300.0, hi=720.0, scale=1 / 60, unit="min", disp="5-12 min", soft=False,
        src="Jonkers2014 ~7min; Banigan 8.3min (kunpause 0.002/s)"),
    "mean_terminating_dwell_s": dict(
        lo=60.0, hi=480.0, scale=1 / 60, unit="min", disp="1-8 min", soft=False,
        src="Cortazar2019/Fong2015; Banigan 8.3min (kunbind 0.002/s)"),
    "mean_residence_s": dict(
        lo=300.0, hi=2400.0, scale=1 / 60, unit="min", disp="5-40 min", soft=True,
        src=">=30min avg gene (Shao&Zeitlinger; Maiuri2011)"),
    "pol2_occupancy": dict(
        lo=0.3, hi=10.0, scale=1.0, unit="Pol/gene", disp="~0.5-2 typ, <=10 active", soft=True,
        src="bimodal 5' peak; Banigan sim ~2/gene"),
    "pol2_density_per_kb": dict(
        lo=0.005, hi=0.25, scale=1.0, unit="Pol/kb", disp="~0.005-0.25 Pol/kb", soft=True,
        src="derived (occupancy / gene length)"),
    "initiation_rate_per_min": dict(
        lo=0.005, hi=1.0, scale=60.0, unit="/h", disp="~0.3-60 /h (bursty)", soft=True,
        src="pause-init limit Gressel2017; Banigan kload 0.001/s=0.06/min"),
    "pausing_index": dict(
        lo=1.5, hi=50.0, scale=1.0, unit="", disp="~2-50 (paused genes)", soft=True,
        src="Jonkers/Lis2014 [NOTE: 5kb window here, not the ~50bp pause]"),
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
    heavy-tail), which the per-metric range check does NOT capture. A realistic
    locus is mostly QUIET with a minority of active genes, spanning orders of
    magnitude (log-normal / Zipf). Expressed-gene initiation should fall in the
    human PRO-seq band 0.2-1.5/min (Zhao, Liu & Siepel 2023, NAR e106)."""
    occ = pd.to_numeric(df["pol2_occupancy"], errors="coerce")
    init = pd.to_numeric(df["initiation_rate_per_min"], errors="coerce")   # events/min
    m = occ.notna()
    occ, init = occ[m], init[m]
    f = lambda mask: 100.0 * float(mask.mean())
    silent = occ < 0.3; quiet = (occ >= 0.3) & (occ < 1.0)
    mod = (occ >= 1.0) & (occ < 3.0); act = occ >= 3.0
    nz = occ[occ > 1e-3]
    spread = (np.percentile(nz, 95) / max(np.percentile(nz, 5), 1e-9)) if len(nz) > 5 else float("nan")
    expr = occ >= 1.0
    in_zhao = 100.0 * float(((init >= 0.2) & (init <= 1.5) & expr).sum()) / max(int(expr.sum()), 1)
    med_expr_init = float(init[expr].median()) if expr.any() else float("nan")
    print("\n=== transcriptional ACTIVITY distribution (expression heavy-tail) ===")
    print(f"  Pol-occupancy classes: silent(<0.3)={f(silent):.0f}%  quiet(0.3-1)={f(quiet):.0f}%  "
          f"moderate(1-3)={f(mod):.0f}%  active(>3)={f(act):.0f}%")
    print(f"    (realistic: a QUIET majority + a small active tail -- log-normal / Zipf)")
    print(f"  activity spread (occupancy p95/p5) = {spread:.0f}x   "
          f"(realistic ~10-1000x; expression spans 3-4 orders genome-wide)")
    print(f"  expressed-gene initiation: median {med_expr_init:.2f}/min; "
          f"{in_zhao:.0f}% in Zhao 0.2-1.5/min band")
    quiet_maj = (f(silent) + f(quiet)) >= 30.0
    heavy_tail = (spread >= 10.0) if spread == spread else False
    verdict = ("REALISTIC (quiet majority + heavy tail)" if (quiet_maj and heavy_tail)
               else "CHECK: distribution too narrow/uniform" if not heavy_tail
               else "CHECK: too few quiet genes (locus uniformly active)")
    print(f"  -> activity distribution: {verdict}")
    return {"silent_pct": f(silent), "quiet_pct": f(quiet), "moderate_pct": f(mod),
            "active_pct": f(act), "spread_p95_p5": spread,
            "expr_init_median_per_min": med_expr_init, "expr_in_zhao_band_pct": in_zhao}


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
    h5_path: Path, tick_seconds: float, *, promoter_window: int, chunk: int = 2000
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
        promoter_ct = np.zeros(n_genes, dtype=np.int64)
        body_ct = np.zeros(n_genes, dtype=np.int64)
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
                # promoter-proximal vs gene-body (rel = distance past TSS toward TES)
                rel = (sp - tss[gp]) * direction[gp]
                prox = rel < promoter_window
                np.add.at(promoter_ct, gp[prox], 1)
                np.add.at(body_ct, gp[~prox], 1)
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
                    # forward progress (sites) for velocity / synthesis throughput
                    adv = (p - tr.prev_pos) * direction[g]
                    if adv > 0:
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

    t_obs = T * tick_seconds
    elong_ticks = state_ticks[:, STATE_ELONGATING].astype(float)
    total_ticks = state_ticks.sum(axis=1).astype(float)

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    rows = []
    for g in range(n_genes):
        occ = present_ct[g].sum() / T
        nasc = engaged_ct[g].sum() / T
        prox_d = promoter_ct[g] / max(promoter_window, 1)
        body_len = max(glen[g] - promoter_window, 1)
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
            "elongation_velocity_bp_s": (adv_sites[g] / elong_ticks[g] * 1000.0 / tick_seconds
                                         if elong_ticks[g] > 0 else float("nan")),
            # pausing & state kinetics
            "pausing_index": (prox_d / body_d if body_d > 0 else float("nan")),
            "mean_pre_initiation_dwell_s": _mean(pre_initiation_dwell[g]),
            "mean_pause_dwell_s": (pause_ticks[g] / pause_episodes[g] * tick_seconds
                                   if pause_episodes[g] > 0 else float("nan")),
            "mean_terminating_dwell_s": _mean(term_dwell[g]),
            "mean_residence_s": _mean(residence[g]),
            "pause_release_efficiency": (completed_ct[g] / init_ct[g]
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
# from HUMAN_RANGES so every panel shades the realistic human range and is self-explaining.
_DIST_PANELS = [
    ("elongation_velocity_bp_s", "Elongation velocity\n(Pol II speed along the gene)", "kb / min", False),
    ("mean_pause_dwell_s", "Promoter-proximal pause\n(time Pol II pauses near the TSS)", "minutes", False),
    ("mean_terminating_dwell_s", "3′ termination dwell\n(time Pol II spends at the gene 3′ end)", "minutes", False),
    ("initiation_rate_per_min", "Initiation rate\n(new Pol II loaded)", "events / h", False),
    ("pol2_occupancy", "Pol II per gene\n(occupancy ≈ Pol II ChIP)", "polymerases / gene", True),
    ("pausing_index", "Pausing index\n(promoter ÷ gene-body density)", "ratio", False),
]


def _dist_panel(ax, vals, key, title, xlabel, logx):
    """One distribution panel: histogram + shaded human range + median line."""
    r = HUMAN_RANGES.get(key, {})
    sc = r.get("scale", 1.0)
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
        if r:
            hi_clip = max(hi_clip, r["hi"] * sc * 1.15)        # always show the full human range
        vplot = np.clip(v, None, hi_clip)
        bins = np.linspace(min(v.min(), 0) if v.size else 0, hi_clip, 26)
    ax.hist(vplot, bins=bins, color="#6f8fbf", edgecolor="white", linewidth=0.3, zorder=2)
    if r:
        lo, hi = r["lo"] * sc, r["hi"] * sc
        ax.axvspan(lo, hi, color="#3b8a5a", alpha=0.15, lw=0, zorder=0, label="human range")
    med = float(np.median(v)) if v.size else np.nan
    ax.axvline(med, color="#A63446", lw=2.0, zorder=3)
    left = med < (np.nanmean(ax.get_xlim()) if not logx else 10 ** np.mean(np.log10(ax.get_xlim())))
    ax.annotate(f"median\n{med:.2g}", xy=(med, 0.97), xycoords=("data", "axes fraction"),
                ha="left" if left else "right", va="top", fontsize=9, color="#A63446", fontweight="bold")
    ax.set_title(title, fontsize=11.5, pad=5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("number of genes", fontsize=9.5)
    ax.tick_params(labelsize=9)
    if r:
        ax.text(0.5, -0.30, f"realistic: {lo:.2g}–{hi:.2g} {xlabel}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color="#3b6b46", style="italic")


def plot_distributions(df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    """Publication figure: per-gene distributions of each transcription metric with
    the realistic human-biology range shaded (green) and the median marked (red),
    plus where Pol II spends its time and the activity (expression) distribution.
    Each panel is self-contained: title says what it is, x-axis gives the unit, and
    the green band shows the realistic range."""
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 13))
    for ax, (key, ttl, xl, logx) in zip(axes.flat[:6], _DIST_PANELS):
        _dist_panel(ax, df.get(key), key, ttl, xl, logx)

    # Panel 7: where Pol II spends its time (mean state-time fractions, one stacked bar).
    ax = axes[2, 0]
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

    # Panel 8: activity (expression) distribution across genes — quiet majority + active tail.
    ax = axes[2, 1]
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
    ax.set_title("Activity (expression) across genes\n(realistic: quiet majority + active tail)",
                 fontsize=11.5, pad=5); ax.tick_params(labelsize=9)

    # Panel 9: plain-language summary.
    ax = axes[2, 2]; ax.axis("off")
    spread = (occ.quantile(0.95) / max(occ.quantile(0.05), 1e-6)) if len(occ) else float("nan")
    lines = [
        "How to read this figure",
        "",
        "• green band = realistic human range",
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
    with the realistic human-range band shaded (green) and TAD boundaries marked
    (red dashed). Shows WHERE active genes and metric outliers sit relative to TADs."""
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
        r = HUMAN_RANGES.get(key)
        if r:
            ax.axhspan(r["lo"] * sc, r["hi"] * sc, color="#3b8a5a", alpha=0.12, lw=0, zorder=0)
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
    axes[0].annotate("green = realistic human range   ·   red dashed = TAD boundary",
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
    ap.add_argument("--promoter-window", type=int, default=5,
                    help="sites downstream of the TSS counted as promoter-proximal "
                         "for the pausing index")
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
        h5_path, tick_seconds, promoter_window=args.promoter_window, chunk=args.measure_chunk)

    tsv = out_dir / "transcription_metrics.tsv"
    out_df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    out_df.to_csv(tsv, sep="\t", index=False)
    print(f"\nmeasured {len(df)} gene(s); locus aggregate:")
    for k in ("initiation_rate_per_min", "completion_rate_per_min", "synthesis_rate_nt_per_s",
              "pol2_occupancy", "elongation_velocity_bp_s", "pausing_index",
              "stall_frac", "active_fraction"):
        print(f"  {k:28s}: {agg[k]:.4g}")
    print(f"wrote {tsv}")

    # Human-biology range check (the headline "are my transcription numbers realistic?").
    rr = realism_report(df)
    print_realism_report(rr)
    act = activity_report(df)
    realism_tsv = out_dir / "transcription_realism.tsv"
    rr.to_csv(realism_tsv, sep="\t", index=False)
    pd.DataFrame([act]).to_csv(out_dir / "transcription_activity.tsv", sep="\t", index=False)
    print(f"wrote {realism_tsv} + transcription_activity.tsv")

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
                           title=f"Transcription metric distributions vs human ranges\n{sub}")
        track_svg = out_dir / "transcription_tracks.svg"
        plot_tracks(chip, nascent, genes, chain, track_svg,
                    title=f"Pol II ChIP & nascent RNA tracks\n{sub}", boundaries=boundaries)
        print(f"wrote {gene_svg}, {dist_svg}, {track_svg}")


if __name__ == "__main__":
    main()
