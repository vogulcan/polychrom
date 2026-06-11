#!/usr/bin/env python
"""Banigan-style evaluation of RNAPII as a moving barrier to cohesin.

Companion to ``gen_transcription_metrics.py`` (which measures the Pol II side).
This measures the COHESIN side of the RNAPII<->cohesin interference -- the actual
result of Banigan et al. 2023 (PNAS, "Transcription shapes 3D chromatin
organization by interacting with loop extrusion"): translocating RNAP is a moving
barrier that obstructs, slows and pushes cohesin, so cohesin ACCUMULATES at active
promoters and is RELOCALIZED toward gene 3' ends.

The signal is the DIFFERENCE between transcription ON and a matched RNAPII-OFF
control (same cohesin parameters, ``max_rnapii: 0``, rnapii plugins nulled): any
cohesin structure that is present ON but absent OFF is caused by transcription,
not by CTCF boundaries that happen to sit near genes. The control is auto-derived
and run unless ``--no-control`` (or an explicit ``--control-h5``) is given.

Measured from the 1D trajectory (``positions`` = cohesin legs, ``genes`` table,
and ``rnapii_positions`` for per-gene activity):

  * Cohesin occupancy meta-profiles, ORIENTED 5'->3' (TSS on the left, TES on the
    right), anchored three ways:
      - TSS-anchored   (promoter accumulation; the Banigan "barrier at the 5' end")
      - TES-anchored   (3' accumulation from terminating Pol II; relocalization)
      - body-scaled    (TSS..TES rescaled to [0,1] with flanks; the whole gene)
    Each is reported as ENRICHMENT = occupancy / genome-mean occupancy, for ON,
    OFF, and ON-OFF.
  * Activity dependence: meta-profiles split into the most- vs least-transcribed
    gene terciles (by measured Pol II occupancy), and a per-gene scatter of
    cohesin enrichment vs Pol II occupancy with Pearson/Spearman correlation --
    Banigan predicts accumulation that SCALES with transcription.

Outputs (to ``--out-dir``):

  * ``cohesin_barrier_metagene.svg``   TSS/TES/body-scaled enrichment, ON vs OFF.
  * ``cohesin_barrier_activity.svg``   active vs silent metagene + enrichment-vs-
                                        activity scatter.
  * ``cohesin_barrier_profiles.tsv``   the metagene curves (ON/OFF/diff).
  * ``cohesin_barrier_per_gene.tsv``   per-gene cohesin enrichment + Pol occupancy.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
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

# RNAPII state codes that carry a transcript / are present at all -- only used to
# weight per-gene activity; any present Pol II counts for occupancy.


def read_cohesin_occupancy(h5_path: Path, chunk: int = 2000):
    """Time-averaged cohesin-leg occupancy per lattice site.

    Returns ``(occ, genome_mean, genes, N, chain, num_chains, T, rnapii_enabled)``
    where ``occ[s]`` is the mean number of cohesin legs (either leg of any LEF)
    sitting at site ``s`` per tick."""
    with h5py.File(h5_path, "r") as h:
        N = int(h.attrs["N"])
        chain = int(h.attrs["chain_length"])
        num_chains = int(h.attrs.get("num_chains", 1))
        rnapii = bool(h.attrs.get("rnapii_enabled", False))
        # The `genes` dataset is only written for RNAPII-enabled runs; an OFF
        # control has none. Callers that need the gene table read the ON run; here
        # we just return None so the control can reuse the ON topology.
        genes = h["genes"][:] if "genes" in h else None
        ds = h["positions"]                       # (T, n_lefs, 2) = [left, right] leg sites
        T = int(ds.shape[0])
        occ = np.zeros(N, dtype=np.float64)
        for start in range(0, T, chunk):
            legs = ds[start:min(start + chunk, T)].reshape(-1)
            legs = legs[(legs >= 0) & (legs < N)]
            np.add.at(occ, legs, 1.0)
        occ /= T
    genome_mean = float(occ.mean()) or 1.0
    return occ, genome_mean, genes, N, chain, num_chains, T, rnapii


def read_pol_occupancy_per_gene(h5_path: Path, genes, chunk: int = 2000):
    """Mean Pol II present per tick at each gene (its trajectory activity).
    Returns an array aligned to ``genes`` rows, or None if RNAPII is disabled."""
    gids = genes["gene_id"].astype(int)
    gidx = np.full(int(gids.max()) + 1, -1, dtype=np.int64)
    gidx[gids] = np.arange(len(gids))
    with h5py.File(h5_path, "r") as h:
        if not bool(h.attrs.get("rnapii_enabled", False)):
            return None
        ds = h["rnapii_positions"]                # (T, R, 2) = [site, gene_id]
        T = int(ds.shape[0])
        occ = np.zeros(len(gids), dtype=np.float64)
        for start in range(0, T, chunk):
            block = ds[start:min(start + chunk, T)]
            site = block[:, :, 0]
            present = site >= 0
            g = gidx[block[:, :, 1][present]]
            np.add.at(occ, g, 1.0)
        occ /= T
    return occ


def build_control_config(base: dict) -> dict:
    """RNAPII-OFF twin of ``base``: identical cohesin/topology, transcription off."""
    cfg = copy.deepcopy(base)
    lef = cfg["lef"]
    lef["max_rnapii"] = 0
    plugins = lef.setdefault("plugins", {})
    plugins["rnapii_load"] = None
    plugins["rnapii_translocate"] = None
    return cfg


def _oriented_profile(occ, genes, N, chain, *, flank, gene_mask=None):
    """TSS- and TES-anchored mean cohesin occupancy, oriented 5'->3'.

    Offset ``d`` runs -flank..+flank in the transcription direction (so +d is
    downstream / toward 3'). Sites crossing a chain boundary are dropped per
    offset. Returns ``(offsets, tss_mean, tes_mean)`` (occupancy, not enrichment)."""
    offsets = np.arange(-flank, flank + 1)
    tss_sum = np.zeros(offsets.size); tss_cnt = np.zeros(offsets.size)
    tes_sum = np.zeros(offsets.size); tes_cnt = np.zeros(offsets.size)
    for gi in range(len(genes)):
        if gene_mask is not None and not gene_mask[gi]:
            continue
        tss = int(genes["tss"][gi]); tes = int(genes["tes"][gi]); d = int(genes["direction"][gi])
        c = tss // chain
        for anchor, ssum, scnt in ((tss, tss_sum, tss_cnt), (tes, tes_sum, tes_cnt)):
            sites = anchor + d * offsets
            ok = (sites >= 0) & (sites < N) & (sites // chain == c)
            ssum[ok] += occ[sites[ok]]
            scnt[ok] += 1.0
    tss_mean = np.divide(tss_sum, tss_cnt, out=np.full(offsets.size, np.nan), where=tss_cnt > 0)
    tes_mean = np.divide(tes_sum, tes_cnt, out=np.full(offsets.size, np.nan), where=tes_cnt > 0)
    return offsets, tss_mean, tes_mean


def _scaled_metagene(occ, genes, N, chain, *, flank, nbody, gene_mask=None):
    """Body-scaled metagene: upstream flank (sites) | TSS..TES rescaled to nbody
    bins | downstream flank. Returns the mean occupancy vector and the x-axis."""
    up = np.zeros(flank); up_c = np.zeros(flank)
    body = np.zeros(nbody); body_c = np.zeros(nbody)
    dn = np.zeros(flank); dn_c = np.zeros(flank)
    body_grid = np.linspace(0.0, 1.0, nbody)
    for gi in range(len(genes)):
        if gene_mask is not None and not gene_mask[gi]:
            continue
        tss = int(genes["tss"][gi]); tes = int(genes["tes"][gi]); d = int(genes["direction"][gi])
        c = tss // chain
        L = abs(tes - tss) + 1
        body_sites = tss + d * np.arange(L)
        ok = (body_sites >= 0) & (body_sites < N) & (body_sites // chain == c)
        if ok.sum() >= 2:
            vals = occ[body_sites[ok]]
            xs = np.linspace(0.0, 1.0, ok.sum())
            body += np.interp(body_grid, xs, vals); body_c += 1.0
        up_sites = tss + d * np.arange(-flank, 0)
        uok = (up_sites >= 0) & (up_sites < N) & (up_sites // chain == c)
        up[uok] += occ[up_sites[uok]]; up_c[uok] += 1.0
        dn_sites = tes + d * np.arange(1, flank + 1)
        dok = (dn_sites >= 0) & (dn_sites < N) & (dn_sites // chain == c)
        dn[dok] += occ[dn_sites[dok]]; dn_c[dok] += 1.0
    prof = np.concatenate([
        np.divide(up, up_c, out=np.full(flank, np.nan), where=up_c > 0),
        np.divide(body, body_c, out=np.full(nbody, np.nan), where=body_c > 0),
        np.divide(dn, dn_c, out=np.full(flank, np.nan), where=dn_c > 0),
    ])
    x = np.concatenate([
        np.linspace(-1.0, 0.0, flank, endpoint=False),   # upstream flank
        body_grid,                                        # TSS(0)..TES(1)
        np.linspace(1.0, 2.0, flank + 1)[1:],             # downstream flank
    ])
    return x, prof


def per_gene_enrichment(occ, genes, genome_mean, N, chain, *, promoter=5):
    """Per-gene cohesin enrichment over genome mean, for the gene body and the
    promoter-proximal window (``promoter`` sites downstream of the TSS)."""
    body_e = np.full(len(genes), np.nan)
    prom_e = np.full(len(genes), np.nan)
    tes_e = np.full(len(genes), np.nan)
    for gi in range(len(genes)):
        tss = int(genes["tss"][gi]); tes = int(genes["tes"][gi]); d = int(genes["direction"][gi])
        c = tss // chain
        lo, hi = (tss, tes) if tss <= tes else (tes, tss)
        body_e[gi] = occ[lo:hi + 1].mean() / genome_mean
        psites = tss + d * np.arange(0, promoter + 1)
        psites = psites[(psites >= 0) & (psites < N) & (psites // chain == c)]
        if psites.size:
            prom_e[gi] = occ[psites].mean() / genome_mean
        tsites = tes + d * np.arange(-promoter, promoter + 1)
        tsites = tsites[(tsites >= 0) & (tsites < N) & (tsites // chain == c)]
        if tsites.size:
            tes_e[gi] = occ[tsites].mean() / genome_mean
    return body_e, prom_e, tes_e


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ar = np.argsort(np.argsort(a[m])); br = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ar, br)[0, 1])


def plot_metagene(x_tss, x_tes, xs_scaled, on, off, flank, out_path, *, title):
    """TSS-anchored, TES-anchored, and body-scaled cohesin enrichment, ON vs OFF,
    plus the ON-OFF difference -- the isolated transcription effect (Banigan)."""
    ncol = 4 if off is not None else 3
    fig, axes = plt.subplots(1, ncol, figsize=(4.0 * ncol, 4.6))
    d = np.arange(-flank, flank + 1)

    ax = axes[0]
    ax.plot(d, on["tss"], color="#A63446", lw=1.6, label="tx ON")
    if off is not None:
        ax.plot(d, off["tss"], color="#465775", lw=1.4, label="RNAPII OFF")
    ax.axvline(0, color="#888", lw=0.8, ls="--"); ax.axhline(1, color="#bbb", lw=0.7)
    ax.set_title("TSS-anchored (5')"); ax.set_xlabel("sites from TSS (oriented 5'->3')")
    ax.set_ylabel("cohesin enrichment (/genome mean)"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(d, on["tes"], color="#A63446", lw=1.6, label="tx ON")
    if off is not None:
        ax.plot(d, off["tes"], color="#465775", lw=1.4, label="RNAPII OFF")
    ax.axvline(0, color="#888", lw=0.8, ls="--"); ax.axhline(1, color="#bbb", lw=0.7)
    ax.set_title("TES-anchored (3')"); ax.set_xlabel("sites from TES (oriented 5'->3')")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(xs_scaled, on["scaled"], color="#A63446", lw=1.6, label="tx ON")
    if off is not None:
        ax.plot(xs_scaled, off["scaled"], color="#465775", lw=1.4, label="RNAPII OFF")
    for xv in (0.0, 1.0):
        ax.axvline(xv, color="#888", lw=0.8, ls="--")
    ax.axhline(1, color="#bbb", lw=0.7)
    ax.set_title("body-scaled metagene"); ax.set_xlabel("TSS(0) -> gene body -> TES(1)")
    ax.legend(fontsize=8)
    ax.text(0.0, ax.get_ylim()[1], " TSS", fontsize=7, va="top")
    ax.text(1.0, ax.get_ylim()[1], "TES ", fontsize=7, va="top", ha="right")

    if off is not None:
        # ON - OFF = the cohesin redistribution CAUSED by transcription. A positive
        # bump at/just inside the TSS is the moving-barrier accumulation (Banigan);
        # a dip is RNAP-driven eviction of cohesin.
        ax = axes[3]
        diff = on["scaled"] - off["scaled"]
        ax.plot(xs_scaled, diff, color="#3b8a5a", lw=1.7)
        ax.fill_between(xs_scaled, 0, diff, where=diff > 0, color="#3b8a5a", alpha=0.25)
        ax.fill_between(xs_scaled, 0, diff, where=diff < 0, color="#A63446", alpha=0.20)
        for xv in (0.0, 1.0):
            ax.axvline(xv, color="#888", lw=0.8, ls="--")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title("ON - OFF (transcription effect)")
        ax.set_xlabel("TSS(0) -> gene body -> TES(1)")
        ax.set_ylabel("Δ cohesin enrichment")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path); plt.close(fig)


def plot_activity(xs_scaled, active_prof, silent_prof, pol_occ, body_e, out_path, *,
                  title, pearson, spearman):
    """Active vs silent metagene + per-gene cohesin enrichment vs Pol II activity."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.plot(xs_scaled, active_prof, color="#A63446", lw=1.8, label="most transcribed (top 1/3)")
    a1.plot(xs_scaled, silent_prof, color="#1b6ca8", lw=1.6, label="least transcribed (bottom 1/3)")
    for xv in (0.0, 1.0):
        a1.axvline(xv, color="#888", lw=0.8, ls="--")
    a1.axhline(1, color="#bbb", lw=0.7)
    a1.set_title("cohesin metagene by activity (tx ON)")
    a1.set_xlabel("TSS(0) -> gene body -> TES(1)")
    a1.set_ylabel("cohesin enrichment (/genome mean)"); a1.legend(fontsize=8)

    m = np.isfinite(pol_occ) & np.isfinite(body_e)
    a2.scatter(pol_occ[m], body_e[m], s=10, alpha=0.5, color="#3b3b6b")
    a2.axhline(1, color="#bbb", lw=0.7)
    a2.set_xscale("log")
    a2.set_title(f"cohesin vs activity  (Pearson r={pearson:.2f}, Spearman ρ={spearman:.2f})")
    a2.set_xlabel("Pol II occupancy at gene (per tick)")
    a2.set_ylabel("gene-body cohesin enrichment")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path); plt.close(fig)


def _profiles(occ, genes, genome_mean, N, chain, *, flank, nbody, gene_mask=None):
    _, tss, tes = _oriented_profile(occ, genes, N, chain, flank=flank, gene_mask=gene_mask)
    xs, scaled = _scaled_metagene(occ, genes, N, chain, flank=flank, nbody=nbody, gene_mask=gene_mask)
    return xs, {"tss": tss / genome_mean, "tes": tes / genome_mean, "scaled": scaled / genome_mean}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="transcription-ON config YAML")
    ap.add_argument("--h5", type=Path, default=None, help="LEFPositions.h5 for --config (else run/reuse)")
    ap.add_argument("--out-dir", type=Path, default=None, help="output dir (default: the H5's parent)")
    ap.add_argument("--control-h5", type=Path, default=None,
                    help="precomputed RNAPII-OFF control H5 (else auto-derive + run, unless --no-control)")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the RNAPII-OFF control (report ON only; no causal ON-OFF difference)")
    ap.add_argument("--force-1d", action="store_true", help="rerun the 1D stage(s) even if an H5 exists")
    ap.add_argument("--flank", type=int, default=50, help="flank window (sites) around TSS/TES")
    ap.add_argument("--nbody", type=int, default=50, help="bins the gene body is rescaled into")
    ap.add_argument("--promoter", type=int, default=5, help="promoter-proximal window (sites) for per-gene enrichment")
    ap.add_argument("--measure-chunk", type=int, default=2000, help="frames per streamed block")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(args.config.read_text())
    label = _safe_label(args.config.stem)
    out_dir = args.out_dir or (args.h5.parent if args.h5 else args.config.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    on_h5 = _resolve_h5(cfg_path=args.config, requested_h5=args.h5.resolve() if args.h5 else None,
                        out_dir=out_dir, label=label, force_1d=args.force_1d)
    occ_on, gm_on, genes, N, chain, num_chains, T, rnapii = read_cohesin_occupancy(
        on_h5, chunk=args.measure_chunk)
    if not rnapii or genes is None:
        raise SystemExit(f"{on_h5}: RNAPII not enabled / no gene table -- this is the tx-ON evaluator.")
    pol_occ = read_pol_occupancy_per_gene(on_h5, genes, chunk=args.measure_chunk)

    print(f"config        : {args.config}")
    print(f"tx-ON traj    : {on_h5}  ({T} frames, {len(genes)} genes)")
    print(f"cohesin genome-mean occupancy : {gm_on:.5f} legs/site/tick")

    # --- matched RNAPII-OFF control -------------------------------------
    occ_off = gm_off = None
    if not args.no_control:
        if args.control_h5 is not None:
            ctrl_h5 = args.control_h5.resolve()
            occ_off, gm_off, *_ = read_cohesin_occupancy(ctrl_h5, chunk=args.measure_chunk)
        else:
            ctrl_dir = out_dir / "control_rnapoff"
            ctrl_dir.mkdir(parents=True, exist_ok=True)
            ctrl_cfg = ctrl_dir / "control_rnapoff.yaml"
            ctrl_cfg.write_text(yaml.safe_dump(build_control_config(base), sort_keys=False))
            ctrl_h5 = _resolve_h5(cfg_path=ctrl_cfg, requested_h5=None, out_dir=ctrl_dir,
                                  label=f"{label}_rnapoff", force_1d=args.force_1d)
            occ_off, gm_off, *_ = read_cohesin_occupancy(ctrl_h5, chunk=args.measure_chunk)
        print(f"control traj  : {ctrl_h5}  (RNAPII OFF; cohesin genome-mean {gm_off:.5f})")

    # --- profiles -------------------------------------------------------
    xs, on_prof = _profiles(occ_on, genes, gm_on, N, chain, flank=args.flank, nbody=args.nbody)
    off_prof = None
    if occ_off is not None:
        _, off_prof = _profiles(occ_off, genes, gm_off, N, chain, flank=args.flank, nbody=args.nbody)

    body_e, prom_e, tes_e = per_gene_enrichment(
        occ_on, genes, gm_on, N, chain, promoter=args.promoter)

    # activity terciles (by measured Pol II occupancy)
    finite = np.isfinite(pol_occ)
    q1, q2 = np.nanquantile(pol_occ[finite], [1 / 3, 2 / 3])
    active_mask = pol_occ >= q2
    silent_mask = pol_occ <= q1
    _, act_prof = _profiles(occ_on, genes, gm_on, N, chain, flank=args.flank,
                            nbody=args.nbody, gene_mask=active_mask)
    _, sil_prof = _profiles(occ_on, genes, gm_on, N, chain, flank=args.flank,
                            nbody=args.nbody, gene_mask=silent_mask)
    pear = float(np.corrcoef(np.log(pol_occ[finite & (pol_occ > 0)]),
                             body_e[finite & (pol_occ > 0)])[0, 1])
    spear = _spearman(pol_occ, body_e)

    # --- summary --------------------------------------------------------
    def _peak(p, lo, hi):
        seg = p[lo:hi]; seg = seg[np.isfinite(seg)]
        return float(seg.max()) if seg.size else float("nan")
    fl = args.flank
    print("\ncohesin enrichment (/genome mean):")
    print(f"  TSS peak   ON={_peak(on_prof['tss'], fl-3, fl+8):.2f}"
          + (f"  OFF={_peak(off_prof['tss'], fl-3, fl+8):.2f}" if off_prof else ""))
    print(f"  TES peak   ON={_peak(on_prof['tes'], fl-8, fl+3):.2f}"
          + (f"  OFF={_peak(off_prof['tes'], fl-8, fl+3):.2f}" if off_prof else ""))
    print(f"  body mean  active={np.nanmean(body_e[active_mask]):.2f}  silent={np.nanmean(body_e[silent_mask]):.2f}")
    print(f"  cohesin-vs-activity  Pearson r={pear:.2f}  Spearman ρ={spear:.2f}")
    if off_prof is not None:
        # Isolated transcription effect: mean ON-OFF cohesin enrichment over the
        # gene body (TSS..TES). >0 = moving-barrier accumulation; <0 = net eviction.
        body = slice(args.flank, args.flank + args.nbody)
        accum = float(np.nanmean((on_prof["scaled"] - off_prof["scaled"])[body]))
        print(f"  ON-OFF gene-body accumulation index = {accum:+.3f}  "
              f"({'barrier-dominated (Banigan)' if accum > 0 else 'eviction-dominated'})")

    # --- plots ----------------------------------------------------------
    sub = f"{args.config.name}  (cohesin legs; oriented 5'->3')"
    plot_metagene(None, None, xs, on_prof, off_prof, args.flank,
                  out_dir / "cohesin_barrier_metagene.svg",
                  title=f"Cohesin accumulation around genes (Banigan moving-barrier)\n{sub}")
    plot_activity(xs, act_prof["scaled"], sil_prof["scaled"], pol_occ, body_e,
                  out_dir / "cohesin_barrier_activity.svg",
                  title=f"Cohesin accumulation scales with transcription\n{sub}",
                  pearson=pear, spearman=spear)

    # --- tsv ------------------------------------------------------------
    prof_df = {"metagene_x": xs, "on_scaled": on_prof["scaled"]}
    if off_prof is not None:
        prof_df["off_scaled"] = off_prof["scaled"]
        prof_df["on_minus_off_scaled"] = on_prof["scaled"] - off_prof["scaled"]
    d = np.arange(-args.flank, args.flank + 1)
    pd.DataFrame(prof_df).to_csv(out_dir / "cohesin_barrier_profiles.tsv", sep="\t", index=False)
    anch = {"offset": d, "on_tss": on_prof["tss"], "on_tes": on_prof["tes"]}
    if off_prof is not None:
        anch["off_tss"] = off_prof["tss"]; anch["off_tes"] = off_prof["tes"]
    pd.DataFrame(anch).to_csv(out_dir / "cohesin_barrier_tss_tes.tsv", sep="\t", index=False)
    pd.DataFrame({
        "gene_id": genes["gene_id"], "tss": genes["tss"], "tes": genes["tes"],
        "direction": genes["direction"], "pol2_occupancy": pol_occ,
        "cohesin_body_enrichment": body_e, "cohesin_promoter_enrichment": prom_e,
        "cohesin_tes_enrichment": tes_e,
    }).to_csv(out_dir / "cohesin_barrier_per_gene.tsv", sep="\t", index=False)

    print(f"\nwrote {out_dir/'cohesin_barrier_metagene.svg'}")
    print(f"wrote {out_dir/'cohesin_barrier_activity.svg'}")
    print(f"wrote {out_dir/'cohesin_barrier_profiles.tsv'}, cohesin_barrier_tss_tes.tsv, cohesin_barrier_per_gene.tsv")


if __name__ == "__main__":
    main()
