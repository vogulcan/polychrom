#!/usr/bin/env python
"""Transcription-kinetics metrics for a transcription-ON (RNAPII-enabled) config.

Given ONE config (and optionally its ``LEFPositions.h5``; the 1D stage is run or
reused if absent), this measures per-gene + locus-aggregate transcription rates
DIRECTLY from the RNAPII trajectory -- no analytic formula. It is the
transcription counterpart to ``gen_lesion_grid_and_heatmaps.py``'s measured
cohesin-stall readout, and reuses the same ``_resolve_h5`` 1D runner.

The trajectory exposes (see loop_extrusion/lef.py):

  * ``rnapii_positions`` (T, R, 2) = [global_site, gene_id], -1 = absent.
  * ``rnapii_states``    (T, R)    = POISED/PAUSED/ELONGATING/TERMINATING/STALLED.
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
    frac_{poised,paused,elongating,stalled,terminating}   state-time fractions
    mean_{poised,pause,terminating}_dwell_s               per-state residence (s)
    mean_residence_s           full load->unload transcription time (s)
    pause_release_efficiency   completed / initiated (productive fraction)

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
    STATE_POISED,
    STATE_STALLED,
    STATE_TERMINATING,
)

# Engaged Pol II carries a nascent transcript (productive or transiently
# obstacle-stalled in the gene body); same convention as polii_chip_nascent_track.py.
_ENGAGED = (STATE_ELONGATING, STATE_STALLED)
_STATE_ORDER = [STATE_POISED, STATE_PAUSED, STATE_ELONGATING, STATE_STALLED, STATE_TERMINATING]
_STATE_NAMES = {
    STATE_POISED: "poised", STATE_PAUSED: "paused", STATE_ELONGATING: "elongating",
    STATE_STALLED: "stalled", STATE_TERMINATING: "terminating",
}

# Per-gene metric columns, in TSV/display order (the aggregate row fills the same).
_RATE_KEYS = [
    "initiation_rate_per_min", "completion_rate_per_min", "synthesis_rate_nt_per_s",
    "pol2_occupancy", "pol2_density_per_kb", "nascent_signal", "elongation_velocity_bp_s",
]
_PAUSE_KEYS = [
    "pausing_index", "frac_poised", "frac_paused", "frac_elongating", "frac_stalled",
    "frac_terminating", "mean_poised_dwell_s", "mean_pause_dwell_s",
    "mean_terminating_dwell_s", "mean_residence_s", "pause_release_efficiency",
]
_INTERFERENCE_KEYS = ["stall_frac", "lesion_stall_frac", "cohesin_stall_frac"]
_BURST_KEYS = [
    "active_fraction", "burst_rate_per_min", "mean_burst_size",
    "mean_burst_duration_s", "mean_interburst_s",
]


class _UidTrack:
    """Accumulated life of one polymerase (keyed by its stable uid)."""

    __slots__ = ("gene", "first_frame", "last_frame", "prev_pos", "adv_sites",
                 "n_poised", "n_paused", "n_pause_episodes", "n_elong", "n_term",
                 "in_pause", "reached_terminating")

    def __init__(self, gene: int, frame: int, pos: int):
        self.gene = gene
        self.first_frame = frame
        self.last_frame = frame
        self.prev_pos = pos
        self.adv_sites = 0
        self.n_poised = 0
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
                    if s_code == STATE_POISED:
                        tr.n_poised += 1
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
    poised_dwell: list[list[float]] = [[] for _ in range(n_genes)]
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
            poised_dwell[g].append(tr.n_poised * tick_seconds)
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
            "mean_poised_dwell_s": _mean(poised_dwell[g]),
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


def plot_per_gene(df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    """Six per-gene panels covering each metric family."""
    labels = [str(g) for g in df["gene_id"]]
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0, 0]
    w = 0.4
    ax.bar(x - w / 2, df["initiation_rate_per_min"], w, label="initiation", color="#465775")
    ax.bar(x + w / 2, df["completion_rate_per_min"], w, label="completion (mRNA)", color="#A63446")
    ax.set_title("Initiation vs completion rate"); ax.set_ylabel("events / min"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(x, df["pol2_density_per_kb"], color="#1b6ca8")
    ax.set_title("Pol II density"); ax.set_ylabel("Pol II / kb (per allele)")

    ax = axes[0, 2]
    ax.bar(x, df["elongation_velocity_bp_s"], color="#3b8a5a")
    ax.set_title("Elongation velocity"); ax.set_ylabel("bp / s")

    ax = axes[1, 0]
    ax.bar(x, df["pausing_index"], color="#c77d18")
    ax.axhline(1.0, color="#888", lw=0.8, ls="--")
    ax.set_title("Pausing index (promoter / body)"); ax.set_ylabel("ratio")

    ax = axes[1, 1]
    bottom = np.zeros(len(df))
    palette = {"poised": "#c9d6df", "paused": "#f2c14e", "elongating": "#3b8a5a",
               "stalled": "#A63446", "terminating": "#6b4e9e"}
    for name, color in palette.items():
        vals = df[f"frac_{name}"].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, label=name)
        bottom += np.nan_to_num(vals)
    ax.set_title("State-time fractions"); ax.set_ylabel("fraction of Pol-ticks")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 2]
    ax.bar(x, df["active_fraction"], color="#1b6ca8")
    ax.set_title("Active fraction (burst occupancy)"); ax.set_ylabel("fraction of ticks")

    for ax in axes.flat:
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=90)
        ax.set_xlabel("gene_id", fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def plot_tracks(chip: np.ndarray, nascent: np.ndarray, genes: np.ndarray,
                chain: int, out_path: Path, *, title: str) -> None:
    """Pol II ChIP (all present) and nascent (engaged) per-site tracks, folded
    across alleles onto one locus; gene bodies shaded."""
    xs = np.arange(chain)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    a1.fill_between(xs, chip, color="#465775", lw=0); a1.set_ylabel("Pol II ChIP\n(per allele)")
    a1.set_title("Pol II occupancy (all states)")
    a2.fill_between(xs, nascent, color="#3b8a5a", lw=0); a2.set_ylabel("nascent RNA\n(engaged)")
    a2.set_title("Nascent RNA signal (ELONGATING|STALLED)")
    for g in genes:
        lo, hi = sorted((int(g["tss"]) % chain, int(g["tes"]) % chain))
        for ax in (a1, a2):
            ax.axvspan(lo, hi, color="#000000", alpha=0.05, lw=0)
    a2.set_xlabel("locus position (sites, folded onto one chain)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
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

    if not args.no_plot:
        with h5py.File(h5_path, "r") as h:
            chain = int(h.attrs["chain_length"])
        sub = f"{args.config.name}, tick={tick_seconds:g}s"
        gene_svg = out_dir / "transcription_metrics.svg"
        plot_per_gene(df, gene_svg, title=f"Per-gene transcription metrics\n{sub}")
        track_svg = out_dir / "transcription_tracks.svg"
        plot_tracks(chip, nascent, genes, chain, track_svg,
                    title=f"Pol II ChIP & nascent RNA tracks\n{sub}")
        print(f"wrote {gene_svg}")
        print(f"wrote {track_svg}")


if __name__ == "__main__":
    main()
