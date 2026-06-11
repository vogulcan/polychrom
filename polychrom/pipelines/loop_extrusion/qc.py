"""Stage QC: 1D + 3D simulation quality control / analysis.

Reads:
* ``cfg.lef.output_path`` (LEFPositions.h5)        -- 1D cohesin/RNAPII/lesions
* ``cfg.contacts.raw_output_path`` (contact_map.npy) -- 3D contact map (optional)

Writes (to ``<contacts.trajectory_folder>/qc/``):
* ``metrics.json``  -- every numeric metric, structured
* ``report.md``     -- human-readable summary
* ``plots/*.png``   -- loop-length, P(s), insulation panels, contact-map heatmap

Each metric function is pure (numpy in, numpy/dict out) so it is reusable from
notebooks. ``run(cfg)`` only orchestrates: load -> compute -> dump.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from . import annotate
from .plugins.sampling import iterative_correction


# ---------------------------------------------------------------------------
# 1D metrics
# ---------------------------------------------------------------------------

def loop_length_stats(positions: np.ndarray, edges: List[int]) -> Dict[str, Any]:
    """positions: (T, n_lefs, 2). Returns histogram + percentiles of |R-L|."""
    L = np.minimum(positions[:, :, 0], positions[:, :, 1])
    R = np.maximum(positions[:, :, 0], positions[:, :, 1])
    sizes = (R - L).ravel().astype(float)
    hist, _ = np.histogram(sizes, bins=edges)
    return {
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "p10": float(np.percentile(sizes, 10)),
        "p90": float(np.percentile(sizes, 90)),
        "histogram_edges_kb": [int(e) for e in edges],
        "histogram_fraction": [float(x) for x in hist / max(hist.sum(), 1)],
        "histogram_counts": [int(x) for x in hist],
    }


def cohesin_occupancy(positions: np.ndarray, n_sites: int) -> np.ndarray:
    """Per-site mean cohesin-leg occupancy (over frames)."""
    occ = np.zeros(n_sites, dtype=float)
    for fr in positions:
        for l, r in fr:
            if 0 <= int(l) < n_sites:
                occ[int(l)] += 1
            if 0 <= int(r) < n_sites:
                occ[int(r)] += 1
    return occ / max(len(positions), 1)


def anchor_set(boundary_positions: List[int], chain_length: int) -> set:
    """CTCF anchor sites (interval left edge + interval right edge - 1) with +-1 slack."""
    edges = [0, *boundary_positions, chain_length]
    out: set = set()
    for start, end in zip(edges[:-1], edges[1:]):
        for s in (start, end - 1):
            out.update({s - 1, s, s + 1})
    return out


def cohesin_classification(positions: np.ndarray, anch: set) -> Dict[str, float]:
    """Per (cohesin, frame): both/one/zero legs anchored at CTCF = corner / stripe / free."""
    anchl = np.array(sorted(anch))
    L = np.minimum(positions[:, :, 0], positions[:, :, 1])
    R = np.maximum(positions[:, :, 0], positions[:, :, 1])
    la = np.isin(L, anchl)
    ra = np.isin(R, anchl)
    n = la.astype(int) + ra.astype(int)
    return {
        "corner_pct": float(100 * (n == 2).mean()),
        "stripe_pct": float(100 * (n == 1).mean()),
        "free_pct": float(100 * (n == 0).mean()),
    }


def boundary_crossing(positions: np.ndarray, boundaries: List[int]) -> Dict[str, float]:
    L = np.minimum(positions[:, :, 0], positions[:, :, 1])
    R = np.maximum(positions[:, :, 0], positions[:, :, 1])
    out = {}
    for b in boundaries:
        out[str(b)] = float((((L < b) & (R > b)).any(1)).mean())
    out["mean"] = float(np.mean(list(out.values())))
    return out


def boundary_crossing_stripes(positions: np.ndarray, boundaries: List[int],
                              slack: int = 1) -> Dict[str, Any]:
    """Per-boundary crossings split by whether a crossing LEF is stripe-like.

    A stripe-like crossing spans boundary ``b`` and has exactly one leg at the
    boundary anchor window around ``b - 1``/``b``. This is conditioned on the
    same crossing event as ``boundary_crossing`` rather than all LEFs globally.
    """
    L = np.minimum(positions[:, :, 0], positions[:, :, 1])
    R = np.maximum(positions[:, :, 0], positions[:, :, 1])
    out: Dict[str, Any] = {}
    event_totals = {"cross": 0, "stripe": 0, "corner": 0, "free": 0}

    for b in boundaries:
        cross = (L < b) & (R > b)
        anchor_sites = np.arange(b - 1 - slack, b + slack + 1)
        n_anchor_legs = np.isin(L, anchor_sites).astype(int) + np.isin(R, anchor_sites).astype(int)
        stripe = cross & (n_anchor_legs == 1)
        corner = cross & (n_anchor_legs == 2)
        free = cross & (n_anchor_legs == 0)

        cross_frames = cross.any(1)
        stripe_frames = stripe.any(1)
        corner_frames = corner.any(1)
        free_frames = free.any(1)
        n_cross_frames = int(cross_frames.sum())
        n_cross_events = int(cross.sum())
        n_stripe_events = int(stripe.sum())
        n_corner_events = int(corner.sum())
        n_free_events = int(free.sum())

        event_totals["cross"] += n_cross_events
        event_totals["stripe"] += n_stripe_events
        event_totals["corner"] += n_corner_events
        event_totals["free"] += n_free_events

        out[str(b)] = {
            "crossing_frame_fraction": float(cross_frames.mean()),
            "stripe_crossing_frame_fraction": float(stripe_frames.mean()),
            "corner_crossing_frame_fraction": float(corner_frames.mean()),
            "free_crossing_frame_fraction": float(free_frames.mean()),
            "stripe_share_of_crossing_frames": (
                float(stripe_frames.sum() / n_cross_frames) if n_cross_frames else 0.0
            ),
            "crossing_event_count": n_cross_events,
            "stripe_crossing_event_pct": (
                float(100 * n_stripe_events / n_cross_events) if n_cross_events else 0.0
            ),
            "corner_crossing_event_pct": (
                float(100 * n_corner_events / n_cross_events) if n_cross_events else 0.0
            ),
            "free_crossing_event_pct": (
                float(100 * n_free_events / n_cross_events) if n_cross_events else 0.0
            ),
        }

    keys = [str(b) for b in boundaries]
    total_cross_events = event_totals["cross"]
    out["mean"] = {
        "crossing_frame_fraction": float(np.mean([out[k]["crossing_frame_fraction"] for k in keys])),
        "stripe_crossing_frame_fraction": float(np.mean([out[k]["stripe_crossing_frame_fraction"] for k in keys])),
        "corner_crossing_frame_fraction": float(np.mean([out[k]["corner_crossing_frame_fraction"] for k in keys])),
        "free_crossing_frame_fraction": float(np.mean([out[k]["free_crossing_frame_fraction"] for k in keys])),
        "stripe_share_of_crossing_frames": float(np.mean([out[k]["stripe_share_of_crossing_frames"] for k in keys])),
        "crossing_event_count": total_cross_events,
        "stripe_crossing_event_pct": (
            float(100 * event_totals["stripe"] / total_cross_events) if total_cross_events else 0.0
        ),
        "corner_crossing_event_pct": (
            float(100 * event_totals["corner"] / total_cross_events) if total_cross_events else 0.0
        ),
        "free_crossing_event_pct": (
            float(100 * event_totals["free"] / total_cross_events) if total_cross_events else 0.0
        ),
    }
    return out


def asymmetry_index(positions: np.ndarray) -> float:
    """Mean | dleft - dright | / (|dleft| + |dright|) per cohesin between frames.

    1 = fully one-sided (stripe-like); 0 = symmetric extrusion.
    """
    diffs = np.diff(positions, axis=0)
    dl = np.abs(diffs[:, :, 0]).astype(float)
    dr = np.abs(diffs[:, :, 1]).astype(float)
    denom = dl + dr
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.abs(dl - dr)[mask].sum() / denom[mask].sum())


def rnapii_metrics(rnapii_positions: np.ndarray, rnapii_states: np.ndarray,
                   tick_seconds: float, rnapii_ids: np.ndarray) -> Dict[str, Any]:
    """State mix + convoy + realized elongation speed.

    Recording columns are the live, compacted RNAPII list, so a column is NOT a
    stable Pol II across ticks. Elongation speed is measured per ``rnapii_ids``
    (uid) track: consecutive ELONGATING ticks of the same uid are true steps.
    """
    present = rnapii_positions[:, :, 0] >= 0
    states = rnapii_states.astype(int)
    tot = present.sum()
    out: Dict[str, Any] = {
        "frames_with_any_polii_pct": float(100 * present.any(1).mean()),
        "mean_count_when_present": float(present.sum(1)[present.any(1)].mean()) if present.any() else 0.0,
        "max_simultaneous": int(present.sum(1).max()),
    }
    if tot > 0:
        out["state_mix_pct"] = {
            "POISED": float(100 * ((states == 0) & present).sum() / tot),
            "PAUSED": float(100 * ((states == 1) & present).sum() / tot),
            "ELONGATING": float(100 * ((states == 2) & present).sum() / tot),
            "TERMINATING": float(100 * ((states == 3) & present).sum() / tot),
            "STALLED": float(100 * ((states == 4) & present).sum() / tot),
        }
        sites = rnapii_positions[:, :, 0]
        adv = 0
        eticks = 0
        for k in range(sites.shape[1]):
            for t in range(1, sites.shape[0]):
                if states[t, k] != 2 or states[t - 1, k] != 2:
                    continue
                if rnapii_ids[t, k] < 0 or rnapii_ids[t, k] != rnapii_ids[t - 1, k]:
                    continue
                d = int(sites[t, k] - sites[t - 1, k])
                adv += abs(d)
                eticks += 1
        sites_per_tick = adv / max(eticks, 1)
        out["elongation_speed_sites_per_tick"] = float(sites_per_tick)
        out["elongation_speed_kb_per_min"] = float(sites_per_tick / tick_seconds * 60.0)
    return out


def rnapii_per_gene_throughput(rnapii_positions: np.ndarray,
                               genes: List[Tuple[int, int]],
                               rnapii_ids: np.ndarray) -> Dict[str, int]:
    """Completed transcripts per gene = # distinct Pol II (uid) that reach TES.

    Counts distinct completing Pol II, not TES-frames: a terminating Pol II
    dwells at the TES for several ticks across shuffled columns, so summing
    ``pos==tes`` frames (the old method) massively overcounts.
    """
    sites = rnapii_positions[:, :, 0]
    out: Dict[str, int] = {}
    for i, (tss, tes) in enumerate(genes):
        uids = rnapii_ids[sites == tes]
        out[f"gene{i}_completions"] = int(len(set(uids[uids >= 0].tolist())))
    return out


def lesion_metrics(lesions: np.ndarray, gene_bodies: List[Tuple[int, int]]) -> Dict[str, Any]:
    n_per = (lesions >= 0).sum(1)
    out: Dict[str, Any] = {
        "mean_simultaneous": float(n_per.mean()),
        "max_simultaneous": int(n_per.max()),
        "frames_with_any_pct": float(100 * (n_per > 0).mean()),
    }
    # observed repair half-life: track distinct lesion-site lifetimes (when seen first->last absent)
    # simplified: total lesion-site-frames / distinct site appearances
    sites_per_frame = [set(int(s) for s in fr if s >= 0) for fr in lesions]
    site_runs: Dict[int, int] = {}
    prev: set = set()
    for cur in sites_per_frame:
        for s in cur:
            site_runs[s] = site_runs.get(s, 0) + 1
        prev = cur
    if site_runs:
        out["mean_observed_lifetime_ticks"] = float(np.mean(list(site_runs.values())))
    # all lesions in gene bodies?
    in_bodies = 0
    total = 0
    for fr in lesions:
        for s in fr:
            if s < 0:
                continue
            total += 1
            for a, b in gene_bodies:
                if a <= int(s) <= b:
                    in_bodies += 1
                    break
    out["lesions_in_gene_body_pct"] = float(100 * in_bodies / max(total, 1))
    return out


def cohesin_at_lesion_flanks(positions: np.ndarray, lesions: np.ndarray, n_sites: int) -> Dict[str, float]:
    occ = cohesin_occupancy(positions, n_sites)
    sites = np.unique(lesions[lesions >= 0])
    if len(sites) == 0:
        return {"flank_mean_occ": 0.0, "global_mean_occ": float(occ.mean()), "enrichment_x": 0.0}
    flanks = [occ[max(0, s - 1)] + occ[min(n_sites - 1, s + 1)] for s in sites]
    fm = float(np.mean(flanks))
    gm = float(occ.mean())
    return {"flank_mean_occ": fm, "global_mean_occ": gm, "enrichment_x": fm / max(gm, 1e-12)}


# ---------------------------------------------------------------------------
# 3D metrics
# ---------------------------------------------------------------------------

def ps_curve(m: np.ndarray) -> np.ndarray:
    """Contact probability vs genomic separation, from a 3D contact map."""
    N = m.shape[0]
    out = np.full(N, np.nan)
    for s in range(1, N):
        out[s] = np.diagonal(m, s).mean()
    return out


def ps_curve_1d(positions: np.ndarray, chain_length: int, num_chains: int) -> np.ndarray:
    """Bridge-contact probability vs genomic separation, from 1D LEF positions.

    Each cohesin bridges its two legs ``(l, r)``; a captured loop of span
    ``s = |r - l|`` is one contact at separation ``s``. Normalised by the number
    of frames and same-chain site pairs at separation ``s`` so the curve is
    shape-comparable to the 3D contact-map P(s).
    """
    counts = np.zeros(chain_length, dtype=float)
    n_frames = max(positions.shape[0], 1)
    for fr in positions:
        for l, r in fr:
            li, ri = int(l), int(r)
            if li < 0 or ri < 0:
                continue
            s = abs(ri - li)
            if 0 < s < chain_length:
                counts[s] += 1
    out = np.full(chain_length, np.nan)
    for s in range(1, chain_length):
        pairs = num_chains * (chain_length - s)
        if pairs > 0:
            out[s] = counts[s] / (n_frames * pairs)
    return out


#: Insulation-score window sizes in **monomers** (~kb). These are physical
#: genomic scales, independent of contact-map resolution; see
#: :func:`insulation_windows_for_resolution`.
INSULATION_WINDOWS = [5, 10, 20, 40, 80, 120]


def insulation_windows_for_resolution(factor: int) -> List[Tuple[int, int]]:
    """Insulation windows (in bins) usable at a ``factor``-monomer resolution.

    Each physical window in :data:`INSULATION_WINDOWS` (monomers) is converted to
    ``window // factor`` bins. Windows finer than 2 bins are dropped -- you cannot
    meaningfully measure insulation below the contact-map resolution -- and
    duplicates are removed. Returns ``(window_bins, window_monomers)`` pairs where
    ``window_monomers = window_bins * factor`` is the honest physical label.
    At ``factor == 1`` this reproduces the historical bin windows exactly.
    """
    factor = max(1, int(factor))
    seen: set = set()
    out: List[Tuple[int, int]] = []
    for w in INSULATION_WINDOWS:
        wb = w // factor
        if wb < 2 or wb in seen:
            continue
        seen.add(wb)
        out.append((wb, wb * factor))
    if not out:  # extreme coarsening: keep one 2-bin window so the metric exists
        out.append((2, 2 * factor))
    return out


def insulation_profile(m: np.ndarray, window: int) -> np.ndarray:
    """Raw windowed-mean insulation profile (legacy helper).

    The qc/compare pipeline no longer uses this -- all insulation analysis is
    cooltools-style via :func:`insulation_score_cooltools` (log2(diamond/median),
    masked-bin aware). Kept only as a low-level primitive for notebooks.
    """
    N = m.shape[0]
    out = np.full(N, np.nan)
    for i in range(window, N - window):
        out[i] = m[i - window:i, i:i + window].mean()
    return out


def insulation_boundary_strength(profile: np.ndarray, boundaries: List[int], window: int) -> Dict[str, float]:
    """Local min around each boundary / global mean. Lower = stronger insulation.

    Legacy ratio metric, unused by the pipeline -- boundary strength is now the
    cooltools log2 prominence (:func:`insulation_boundary_prominence`).
    """
    valid = profile[~np.isnan(profile)]
    gm = float(valid.mean()) if valid.size else float("nan")
    out: Dict[str, float] = {"global_mean": gm}
    for b in boundaries:
        lo = max(b - window // 2, 0)
        hi = min(b + window // 2, len(profile))
        out[str(b)] = float(np.nanmin(profile[lo:hi]) / gm) if gm > 0 else float("nan")
    return out


def aggregate_insulation(profile: np.ndarray, boundaries: List[int], half: int = 40) -> Dict[str, Any]:
    """Aggregate (mean over boundaries) insulation profile centered at boundaries.

    Legacy ratio aggregate, unused by the pipeline -- the cooltools log2 variant
    :func:`aggregate_insulation_log2` (dip_depth = flank - center) is used instead.

    Stacks the +/-``half`` window of the full-chain insulation ``profile`` around
    each boundary (offset 0 = boundary) and averages over boundaries. The central
    dip relative to the flanks is the aggregate boundary-insulation score
    (``dip_ratio`` = center / flank-mean; lower = stronger insulation). Boundaries
    whose window falls outside the valid (non-NaN) profile range are skipped.
    """
    snips = []
    for b in boundaries:
        lo, hi = b - half, b + half + 1
        if lo < 0 or hi > len(profile):
            continue
        seg = profile[lo:hi]
        if np.isnan(seg).any():
            continue
        snips.append(seg)
    if not snips:
        return {"n": 0, "half": half, "offsets": [], "mean_profile": []}
    mean_prof = np.vstack(snips).mean(axis=0)
    center = float(mean_prof[half])
    edge = max(half // 2, 1)
    flank = float(np.concatenate([mean_prof[:edge], mean_prof[-edge:]]).mean())
    return {
        "n": len(snips),
        "half": half,
        "offsets": list(range(-half, half + 1)),
        "mean_profile": [float(x) for x in mean_prof],
        "center": center,
        "flank_mean": flank,
        "dip_ratio": float(center / flank) if flank > 0 else float("nan"),
    }


def insulation_score_cooltools(m: np.ndarray, window: int,
                               min_valid_frac: float = 0.66) -> np.ndarray:
    """cooltools-style log2 insulation score for a square contact map.

    Mirrors ``cooltools.insulation``: for each bin ``i`` the diamond window
    ``m[i-w:i, i:i+w]`` is **nansum**-reduced with a valid-pixel count; bins whose
    diamond has fewer than ``min_valid_frac`` finite pixels are set NaN (masked /
    low-coverage). The per-bin diamond sums are then divided by their genome-wide
    **median** and ``log2``-transformed, so 0 = chromosome-average insulation,
    negative = insulated (boundary-like), positive = enriched.

    Unlike :func:`insulation_profile` (raw windowed mean), this normalizes and
    log-transforms like cooltools, and excludes masked bins instead of counting
    them as zero. Masked bins are reconstructed as fully-zero / all-NaN rows of
    the ICE-balanced map. Uses a naive sliding window (like ``insulation_profile``)
    so it stays memory-light on large maps.
    """
    m = np.asarray(m, dtype=float)
    N = m.shape[0]
    if window < 1 or N < 2 * window + 1:
        return np.full(N, np.nan)
    # Reconstruct masked (dead) bins without copying the whole map: ICE sets dead
    # rows/cols to NaN, which the qc/compare loaders then turn into 0. Treat all-
    # NaN or all-zero rows as bad. Contact maps are non-negative, so a plain row
    # sum of 0 identifies a zeroed (masked) bin.
    finite_any = np.isfinite(m).any(axis=1)
    with np.errstate(invalid="ignore"):
        row_sum = m.sum(axis=1)
    bad = (~finite_any) | (row_sum == 0)
    if bad.any():
        mm = m.copy()
        mm[bad, :] = np.nan
        mm[:, bad] = np.nan
    else:
        mm = m

    raw = np.full(N, np.nan)
    thresh = min_valid_frac * float(window * window)
    for i in range(window, N - window):
        block = mm[i - window:i, i:i + window]
        finite = np.isfinite(block)
        nvalid = int(finite.sum())
        if nvalid >= thresh:
            raw[i] = float(np.nansum(block))
    med = np.nanmedian(raw)
    if not np.isfinite(med) or med <= 0:
        return np.full(N, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(raw / med)


def insulation_boundary_prominence(score: np.ndarray, boundaries: List[int]) -> Dict[str, float]:
    """cooltools-style boundary strength = prominence of the insulation minimum.

    For each (known) boundary ``b`` in the log2 insulation ``score``, the strength
    is ``min(left_peak, right_peak) - score_min`` where ``score_min`` is the local
    minimum near ``b`` and the flanking peaks are the maxima of the profile on each
    side, bounded by the neighbouring boundaries. Larger = stronger insulation
    (deeper, more prominent dip), in log2 units. Boundaries whose flanks have no
    finite score are skipped. Returns per-boundary values plus their ``mean``.
    """
    n = len(score)
    bs = sorted(int(b) for b in boundaries if 0 <= int(b) < n)
    out: Dict[str, float] = {}
    proms: List[float] = []
    for idx, b in enumerate(bs):
        left_lim = bs[idx - 1] if idx > 0 else 0
        right_lim = bs[idx + 1] if idx + 1 < len(bs) else n - 1
        half = max(1, (right_lim - left_lim) // 8)
        lo = max(left_lim, b - half)
        hi = min(right_lim, b + half) + 1
        seg = score[lo:hi]
        left_seg = score[left_lim:b + 1]
        right_seg = score[b:right_lim + 1]
        if not (np.isfinite(seg).any() and np.isfinite(left_seg).any()
                and np.isfinite(right_seg).any()):
            continue
        smin = float(np.nanmin(seg))
        lp = float(np.nanmax(left_seg))
        rp = float(np.nanmax(right_seg))
        prom = float(min(lp, rp) - smin)
        out[str(b)] = prom
        proms.append(prom)
    out["mean"] = float(np.mean(proms)) if proms else float("nan")
    return out


#: Half-width (in **monomers**) of the boundary-centered insulation aggregate
#: window. Converted to bins per resolution by :func:`insulation_aggregate_half`.
INSULATION_AGGREGATE_HALF = 40


def insulation_aggregate_half(factor: int) -> int:
    """Aggregate half-window in bins for a ``factor``-monomer resolution.

    Keeps the physical +/-``INSULATION_AGGREGATE_HALF`` monomer span constant
    across resolutions (so a coarsened map doesn't use a 10x larger window), with
    a floor of 2 bins so the dip is still resolvable. ``factor == 1`` reproduces
    the historical 40-bin half.
    """
    return max(2, INSULATION_AGGREGATE_HALF // max(1, int(factor)))


def aggregate_insulation_log2(score: np.ndarray, boundaries: List[int],
                              half: int = 40) -> Dict[str, Any]:
    """Boundary-centered average of a log2 insulation ``score``.

    Like :func:`aggregate_insulation` but for the cooltools log2 score: the dip is
    a **difference** (``dip_depth = flank_mean - center``, positive = insulated),
    not a ratio, because the score is already in log2 space.
    """
    snips = []
    for b in boundaries:
        lo, hi = b - half, b + half + 1
        if lo < 0 or hi > len(score):
            continue
        seg = score[lo:hi]
        if np.isnan(seg).any():
            continue
        snips.append(seg)
    if not snips:
        return {"n": 0, "half": half, "offsets": [], "mean_profile": []}
    mean_prof = np.vstack(snips).mean(axis=0)
    center = float(mean_prof[half])
    edge = max(half // 2, 1)
    flank = float(np.concatenate([mean_prof[:edge], mean_prof[-edge:]]).mean())
    return {
        "n": len(snips),
        "half": half,
        "offsets": list(range(-half, half + 1)),
        "mean_profile": [float(x) for x in mean_prof],
        "center": center,
        "flank_mean": flank,
        "dip_depth": float(flank - center),
    }


def tad_strength(m: np.ndarray, tads: List[Tuple[int, int]]) -> Dict[str, Any]:
    intra = [float(m[a:b, a:b].mean()) for a, b in tads]
    inter = []
    for i in range(len(tads) - 1):
        a, b = tads[i]
        c, d = tads[i + 1]
        inter.append(float(m[a:b, c:d].mean()))
    return {
        "intra_means": intra,
        "inter_means": inter,
        "intra_over_inter": float(np.mean(intra) / max(np.mean(inter), 1e-12)),
    }


def corner_dot_intensities(m: np.ndarray, tads: List[Tuple[int, int]]) -> List[float]:
    """Mean contact in a 3x3 window at the (left-anchor, right-anchor) corner of each TAD."""
    out = []
    for a, b in tads:
        out.append(float(m[max(a - 1, 0):a + 2, b - 2:b + 1].mean()))
    return out


def rescale_square(snip: np.ndarray, target: int) -> np.ndarray:
    """Bilinear-rescale a square matrix to (target, target)."""
    from scipy.ndimage import zoom
    factor = target / snip.shape[0]
    out = zoom(snip, (factor, factor), order=1)
    # ndimage.zoom can be off-by-one; trim or pad to exact target.
    if out.shape[0] >= target:
        out = out[:target, :target]
    else:
        pad = ((0, target - out.shape[0]), (0, target - out.shape[1]))
        out = np.pad(out, pad, mode="edge")
    return out


#: Side length (pixels) of the rescaled-TAD pile-up. With median TADs ~700-800
#: monomers, the native 3xTAD window is ~2100-2400 bins, so 300 still downsamples
#: real data (no interpolation) while matching the rendered panel size at dpi 120.
#: Must stay divisible by 3 so the central-third TAD square stays exact.
TAD_PILEUP_RESOLUTION = 300


def rescaled_tad_pileup(m: np.ndarray, tads: List[Tuple[int, int]],
                        target: int = TAD_PILEUP_RESOLUTION) -> Tuple[Optional[np.ndarray], List[np.ndarray], List[int]]:
    """Flyamer-2017-style rescaled TAD pile-up.

    For each TAD ``(a, b)`` of length L, take the 3L x 3L window
    ``m[a-L : b+L, a-L : b+L]`` (one TAD on each side), rescale to
    ``target x target`` (bilinear). Returns ``(average, per_tad_snippets,
    valid_tad_indices)``. The central ``[target/3 : 2*target/3]`` square is the
    TAD itself; the flanks are the neighbours.

    TADs whose window runs past the map edge are skipped.
    """
    N = m.shape[0]
    snippets: List[np.ndarray] = []
    kept: List[int] = []
    for idx, (a, b) in enumerate(tads):
        L = b - a
        if a - L < 0 or b + L > N or L <= 0:
            continue
        sub = m[a - L:b + L, a - L:b + L]
        snippets.append(rescale_square(sub, target))
        kept.append(idx)
    if not snippets:
        return None, [], []
    return np.mean(np.stack(snippets, axis=0), axis=0), snippets, kept


def tad_strength_from_pileup(pile: np.ndarray) -> Dict[str, float]:
    """Within-TAD / between-TAD intensity ratio on a rescaled pile-up.

    Per Flyamer 2017: pile-up is 3 TADs wide, so the central third is the TAD
    itself and the side thirds are the neighbours. The thirds are derived from
    the pile-up size, so any ``target`` divisible by 3 works (e.g. 300 gives the
    [100:200, 100:200] central square).
    """
    n = pile.shape[0]
    third = n // 3
    within = float(pile[third:2 * third, third:2 * third].sum())
    upper = float(pile[:third, third:2 * third].sum())
    lower = float(pile[third:2 * third, 2 * third:].sum())
    between = 0.5 * (upper + lower)
    return {
        "within": within,
        "between": between,
        "strength": float(within / between) if between > 0 else float("nan"),
    }


def coarsen_contact_map(m: np.ndarray, factor: int) -> np.ndarray:
    """Bin a square contact map into ``factor x factor`` blocks (summed).

    Coarsens a per-monomer contact map to ``factor``-monomer resolution by
    summing each non-overlapping ``factor x factor`` block (the standard
    contact-map binning: a contact between monomers ``i, j`` becomes a contact
    between bins ``i // factor, j // factor``). An incomplete trailing block
    (when the size is not divisible by ``factor``) is dropped. ``factor <= 1``
    returns the map unchanged. Intended to run on the *raw* map, before ICE.
    """
    m = np.asarray(m, dtype=float)
    if factor <= 1:
        return m
    n = m.shape[0]
    nb = n // factor
    if nb == 0:
        raise ValueError(f"coarsening factor {factor} exceeds map size {n}")
    trimmed = m[: nb * factor, : nb * factor]
    return trimmed.reshape(nb, factor, nb, factor).sum(axis=(1, 3))


def scale_resolution_coords(
    boundaries: List[int],
    gene_tss: List[int],
    gene_tes: List[int],
    lesion_sites: Optional[List[int]],
    factor: int,
    n_bins: int,
) -> Dict[str, Any]:
    """Map monomer-unit annotation coordinates onto a coarsened map's bins.

    Every coordinate ``p`` lands in bin ``p // factor`` (clipped to the last
    valid bin). Boundaries are de-duplicated and sorted (two boundaries closer
    than ``factor`` monomers collapse to one bin); TADs are rebuilt from the
    scaled boundaries spanning ``[0, n_bins]`` with any zero-width interval
    dropped. ``factor == 1`` is the identity (no clipping/dedup) so the native
    analysis is byte-for-byte unchanged.
    """
    if factor <= 1:
        tads = [(s, e) for s, e in zip([0, *boundaries], [*boundaries, n_bins])]
        return {
            "boundaries": list(boundaries),
            "tads": tads,
            "gene_tss": list(gene_tss),
            "gene_tes": list(gene_tes),
            "lesion_sites": list(lesion_sites) if lesion_sites is not None else None,
        }

    def clip(p: int) -> int:
        return min(max(int(p) // factor, 0), n_bins - 1)

    b = sorted({clip(x) for x in boundaries})
    tads = [(s, e) for s, e in zip([0, *b], [*b, n_bins]) if e > s]
    return {
        "boundaries": b,
        "tads": tads,
        "gene_tss": [clip(x) for x in gene_tss],
        "gene_tes": [clip(x) for x in gene_tes],
        "lesion_sites": ([clip(x) for x in lesion_sites]
                         if lesion_sites is not None else None),
    }


def observed_over_expected(m: np.ndarray) -> np.ndarray:
    """Divide each contact by the mean at its genomic separation (P(s) baseline).

    Returns ratio map (1 = expected, >1 = enriched, <1 = depleted). NaN-safe.
    """
    N = m.shape[0]
    exp = np.zeros_like(m)
    for s in range(N):
        d = np.diagonal(m, s).mean()
        if d <= 0:
            continue
        idx = np.arange(N - s)
        exp[idx, idx + s] = d
        exp[idx + s, idx] = d
    with np.errstate(divide="ignore", invalid="ignore"):
        oe = np.where(exp > 0, m / exp, np.nan)
    return oe


def pileup(m: np.ndarray, centers: List[int], half: int = 40) -> Optional[np.ndarray]:
    """Average (2*half+1)^2 contact-map snippet centered on each ``centers[i]``.

    Centers within ``half`` of the map edge are dropped. Returns ``None`` if no
    valid centers.
    """
    N = m.shape[0]
    side = 2 * half + 1
    snippets: List[np.ndarray] = []
    for c in centers:
        if c - half < 0 or c + half + 1 > N:
            continue
        snippets.append(m[c - half:c + half + 1, c - half:c + half + 1])
    if not snippets:
        return None
    return np.mean(np.stack(snippets, axis=0), axis=0)


def stripe_enrichment(m: np.ndarray, anchor: int, length: int = 80) -> Dict[str, float]:
    """Mean contact along horizontal extension from anchor vs background."""
    N = m.shape[0]
    right = [float(m[anchor, j]) for j in range(anchor + 5, min(anchor + length, N))]
    left = [float(m[anchor, j]) for j in range(max(anchor - length, 0), max(anchor - 5, 0))]
    bg = float(m.mean())
    return {
        "right_mean": float(np.mean(right)) if right else 0.0,
        "left_mean": float(np.mean(left)) if left else 0.0,
        "right_enrichment_x": (float(np.mean(right)) / bg) if (right and bg > 0) else 0.0,
        "left_enrichment_x": (float(np.mean(left)) / bg) if (left and bg > 0) else 0.0,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_1d(positions: np.ndarray, chain_length: int, num_chains: int) -> Dict[str, Any]:
    return {
        "n_frames": int(positions.shape[0]),
        "n_lefs": int(positions.shape[1]),
        "any_nan": bool(np.isnan(positions).any()),
        "any_out_of_range": bool((positions < 0).any() or (positions >= chain_length * num_chains).any()),
        "any_cross_chain_leg": bool(((positions[:, :, 0] // chain_length) != (positions[:, :, 1] // chain_length)).any()),
    }


# ---------------------------------------------------------------------------
# Orchestrator + plotting
# ---------------------------------------------------------------------------

def _plot_loop_hist(stats: Dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = stats["histogram_edges_kb"]
    frac = stats["histogram_fraction"]
    widths = [edges[i + 1] - edges[i] for i in range(len(frac))]
    # Plot probability DENSITY (fraction per kb), not raw per-bin fraction. The
    # bins are unequal width, so raw fraction makes the wide bins look tall and
    # breaks the monotonic decay (e.g. the wide 200-300 / 500+ bins appear as
    # bumps). Dividing by bin width gives a bin-width-independent distribution
    # that decreases continuously, and renders the wide tail bin honestly small.
    density = [f / w if w > 0 else 0.0 for f, w in zip(frac, widths)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(density)), density,
           tick_label=[f"{edges[i]}-{edges[i + 1]}" for i in range(len(density))])
    ax.set_xlabel("loop length bin (kb)")
    ax.set_ylabel("fraction per kb (density)")
    ax.set_title("Cohesin loop length distribution")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_ps(ps: np.ndarray, out: Path,
             title: str = "Contact probability vs genomic separation") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    s = np.arange(1, len(ps))
    ax.loglog(s, ps[1:])
    ax.set_xlabel("s (kb)")
    ax.set_ylabel("P(s)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_insulation(scores: List[Tuple[int, int, np.ndarray]],
                     boundaries: List[int], out: Path, factor: int = 1) -> None:
    """cooltools-style log2 insulation score per window (one panel each).

    ``scores`` is a list of ``(window_bins, window_monomers, score_array)``; the
    x-axis is drawn in monomers (~kb) and boundaries are marked. 0 = chromosome
    average, negative = insulated (boundary-like)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(scores)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    for k, (wb, wphys, score) in enumerate(scores):
        axes[k].plot(np.arange(len(score)) * factor, score)
        axes[k].axhline(0, color="grey", ls="-", alpha=0.3, lw=0.8)
        for b in boundaries:
            axes[k].axvline(b * factor, color="k", ls=":", alpha=0.4)
        axes[k].set_title(f"window={wphys} kb")
        axes[k].set_ylabel("log2 insulation")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    axes[-1].set_xlabel("position (kb)")
    fig.suptitle("cooltools-style insulation score (log2; lower = insulated)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_insulation_aggregate(scores: List[Tuple[int, int, np.ndarray]],
                               boundaries: List[int], out: Path, factor: int = 1) -> None:
    """Boundary-centered average of the cooltools log2 insulation score per window.

    ``scores`` is a list of ``(window_bins, window_monomers, score_array)``; the
    offset axis is drawn in monomers (~kb). The central dip (``dip_depth`` =
    flank - center, in log2) is the aggregate boundary-insulation strength."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(scores)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    half = insulation_aggregate_half(factor)
    for k, (wb, wphys, score) in enumerate(scores):
        agg = aggregate_insulation_log2(score, boundaries, half=half)
        if agg["mean_profile"]:
            axes[k].plot(np.asarray(agg["offsets"]) * factor, agg["mean_profile"])
        axes[k].axvline(0, color="k", ls=":", alpha=0.4)
        axes[k].axhline(0, color="grey", ls="-", alpha=0.3, lw=0.8)
        axes[k].set_title(f"window={wphys} kb (n={agg['n']}, "
                          f"dip={agg.get('dip_depth', float('nan')):.2f})")
        axes[k].set_ylabel("mean log2 insulation")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    axes[-1].set_xlabel("offset from boundary (kb)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_pileups(m: np.ndarray, anchor_groups: Dict[str, List[int]],
                  out: Path, half: int = 40) -> Dict[str, Any]:
    """One pile-up heatmap per anchor group; return per-group mean center value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = [(k, v) for k, v in anchor_groups.items() if v]
    if not items:
        return {}
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    summary: Dict[str, Any] = {}
    for k, (name, centers) in enumerate(items):
        ax = axes[k // cols][k % cols]
        snip = pileup(m, centers, half=half)
        if snip is None:
            ax.set_title(f"{name}: no valid centers")
            ax.axis("off")
            summary[name] = {"n_centers": 0}
            continue
        ax.imshow(np.log1p(snip), cmap="inferno", origin="lower",
                  extent=[-half, half, -half, half])
        ax.axhline(0, color="cyan", ls=":", alpha=0.5, lw=0.5)
        ax.axvline(0, color="cyan", ls=":", alpha=0.5, lw=0.5)
        ax.set_title(f"{name} pile-up (n={len(centers)})")
        ax.set_xlabel("Δ kb")
        ax.set_ylabel("Δ kb")
        summary[name] = {
            "n_centers": int(len(centers)),
            "center_value": float(snip[half, half]),
            "mean_value": float(snip.mean()),
            "corner_value": float(snip[0, -1]),  # off-diagonal corner of the snippet
        }
    for k in range(len(items), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return summary


def _plot_rescaled_tad_pileup(avg_obs: Optional[np.ndarray], avg_oe: Optional[np.ndarray],
                              strengths: Dict[str, Any], out: Path) -> None:
    """Side-by-side avg rescaled-TAD pile-ups (obs + O/E) with within/between boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, label, key in (
        (axes[0], avg_obs, "obs (ICE)", "obs"),
        (axes[1], avg_oe, "O/E", "oe"),
    ):
        if mat is None:
            ax.set_title(f"{label}: no valid TADs"); ax.axis("off"); continue
        n = mat.shape[0]; t = n // 3
        if label.startswith("O"):
            v = np.log2(np.clip(mat, 1e-3, None))
            vlim = float(np.percentile(np.abs(v[np.isfinite(v)]), 99))
            im = ax.imshow(v, cmap="bwr", vmin=-vlim, vmax=vlim, origin="lower")
        else:
            v = np.log1p(mat)
            lo, hi = np.percentile(v[np.isfinite(v)], (1, 99))
            im = ax.imshow(v, cmap="inferno", vmin=lo, vmax=hi, origin="lower")
        # boxes: central within = green, two flanks = cyan
        ax.add_patch(Rectangle((t, t), t, t, fill=False, ec="lime", lw=1.5))
        ax.add_patch(Rectangle((t, 0), t, t, fill=False, ec="cyan", lw=1.0, ls="--"))
        ax.add_patch(Rectangle((2 * t, t), t, t, fill=False, ec="cyan", lw=1.0, ls="--"))
        s = strengths.get(key, {})
        ax.set_title("%s rescaled-TAD pile-up\nstrength=%.2f (within/between)" %
                     (label, s.get("strength", float("nan"))))
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_contact_map(m: np.ndarray, ann: Optional[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.log1p(m), cmap="inferno", origin="lower")
    annotate.draw(ax, ann, legend=True)
    ax.set_title("Contact map (log1p)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


@dataclass
class QCPaths:
    folder: Path
    metrics_json: Path
    report_md: Path
    plots_dir: Path


def _paths(cfg) -> QCPaths:
    base = Path(cfg.contacts.trajectory_folder) / "qc"
    plots = base / "plots"
    base.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return QCPaths(base, base / "metrics.json", base / "report.md", plots)


def _suffix(p: Path, suffix: str) -> Path:
    """Insert ``suffix`` before the file extension (``a.png`` -> ``a_res10.png``)."""
    return p.with_name(f"{p.stem}{suffix}{p.suffix}") if suffix else p


def _analyze_3d(
    m: np.ndarray,
    oe: np.ndarray,
    *,
    boundaries: List[int],
    tads: List[Tuple[int, int]],
    gene_tss: List[int],
    gene_tes: List[int],
    lesion_sites: Optional[List[int]],
    plots_dir: Path,
    annotations: Optional[dict],
    resolution: int = 1,
    suffix: str = "",
) -> Dict[str, Any]:
    """Compute every 3D metric on a (already ICE-balanced) contact map + O/E.

    ``m``/``oe`` are the balanced obs and observed/expected maps at the working
    resolution; ``boundaries``/``tads``/``gene_*``/``lesion_sites`` are in that
    map's bin units (use :func:`scale_resolution_coords` to project monomer
    coordinates onto a coarsened map). Plots are written into ``plots_dir`` with
    ``suffix`` appended before the extension so multiple resolutions coexist.
    Returns the ``m3d`` metrics dict.
    """
    ps = ps_curve(m)
    # Insulation windows are physical (monomers); convert to bins for this
    # resolution and key/label the metrics by the honest monomer scale, so a
    # coarsened map never reports a window finer than its bin size.
    windows = insulation_windows_for_resolution(resolution)
    m3d: Dict[str, Any] = {
        "shape": list(m.shape),
        "resolution": int(resolution),
        "ice_balanced": True,
        "tad_strength": tad_strength(m, tads),
        "corner_dot_intensities": corner_dot_intensities(m, tads),
        # O/E corner dot: distance-normalized CTCF-CTCF loop enrichment, the
        # proper corner-score analog of the 1D corner_both metric. The raw-obs
        # corner dot is distance-confounded (far-apart anchors -> low count).
        "corner_dot_intensities_oe": corner_dot_intensities(oe, tads),
        "ps_at": {str(s): float(ps[s]) for s in (5, 10, 20, 50, 100, 150, 200, 300, 500) if s < len(ps)},
        "stripe_enrichment_per_boundary": {
            str(b): stripe_enrichment(m, b) for b in boundaries
        },
    }
    # Insulation: cooltools-style log2(diamond / median) score (masked-bin aware).
    # Computed once per window and reused for boundary strength, aggregate, and
    # the plots. Boundary strength = log2 prominence of the minimum (higher =
    # stronger); aggregate dip_depth = flank - center in log2 (higher = stronger).
    insul_scores = [(wb, wphys, insulation_score_cooltools(m, wb)) for wb, wphys in windows]
    agg_half = insulation_aggregate_half(resolution)  # physical 40 monomers -> bins
    m3d["insulation_boundary_strength"] = {
        str(wphys): insulation_boundary_prominence(score, boundaries)
        for wb, wphys, score in insul_scores
    }
    m3d["insulation_aggregate"] = {
        str(wphys): aggregate_insulation_log2(score, boundaries, half=agg_half)
        for wb, wphys, score in insul_scores
    }
    _plot_ps(ps, _suffix(plots_dir / "Ps_3d.png", suffix), title="P(s) — 3D contact map")
    _plot_insulation(insul_scores, boundaries,
                     _suffix(plots_dir / "insulation_windows.png", suffix), factor=resolution)
    _plot_insulation_aggregate(insul_scores, boundaries,
                               _suffix(plots_dir / "insulation_aggregate.png", suffix),
                               factor=resolution)
    _plot_contact_map(m, annotations, _suffix(plots_dir / "contact_map.png", suffix))
    # Pile-ups (centered on each anchor type) -- obs and O/E, both ICE-balanced.
    anchor_groups: Dict[str, List[int]] = {"CTCF boundary": list(boundaries)}
    if gene_tss or gene_tes:
        anchor_groups["gene TSS"] = list(gene_tss)
        anchor_groups["gene TES"] = list(gene_tes)
    if lesion_sites:
        anchor_groups["lesions"] = list(lesion_sites)
    m3d["pileups_obs"] = _plot_pileups(m, anchor_groups, _suffix(plots_dir / "pileups_obs.png", suffix), half=40)
    m3d["pileups_oe"] = _plot_pileups(oe, anchor_groups, _suffix(plots_dir / "pileups_oe.png", suffix), half=40)

    # Flyamer 2017 rescaled-TAD pile-ups (obs + O/E). 3xTAD window scaled
    # to TAD_PILEUP_RESOLUTION**2; within = central third, between = mean of
    # the two flanking blocks; strength = within/between (see
    # tad_strength_from_pileup, which derives the thirds from the size).
    avg_obs, snips_obs, kept = rescaled_tad_pileup(m, tads)
    avg_oe, snips_oe, _ = rescaled_tad_pileup(oe, tads)
    tad_pile: Dict[str, Any] = {"n_valid_tads": len(kept), "valid_tad_indices": kept}
    if avg_obs is not None:
        tad_pile["obs"] = tad_strength_from_pileup(avg_obs)
        tad_pile["per_tad"] = {}
        for i, idx in enumerate(kept):
            obs_s = tad_strength_from_pileup(snips_obs[i])
            oe_s = tad_strength_from_pileup(snips_oe[i]) if snips_oe else {}
            tad_pile["per_tad"][str(idx)] = {"obs": obs_s, "oe": oe_s}
    if avg_oe is not None:
        tad_pile["oe"] = tad_strength_from_pileup(avg_oe)
    m3d["tad_pileup"] = tad_pile
    _plot_rescaled_tad_pileup(avg_obs, avg_oe, tad_pile,
                              _suffix(plots_dir / "pileups_tad_rescaled.png", suffix))
    return m3d


def run(cfg) -> Path:
    """Run QC on the lef stage's h5 + (optionally) the contacts contact map."""
    paths = _paths(cfg)

    lef_h5 = Path(cfg.lef.output_path)
    if not lef_h5.exists():
        raise FileNotFoundError(f"LEFPositions h5 not found: {lef_h5}")
    with h5py.File(lef_h5, "r") as fh:
        positions = fh["positions"][:]
        n_sites = int(fh.attrs["N"])
        chain_length = int(fh.attrs.get("chain_length", n_sites))
        num_chains = int(fh.attrs.get("num_chains", 1))
        rnapii_enabled = bool(fh.attrs.get("rnapii_enabled", False))
        rnapii_positions = fh["rnapii_positions"][:] if "rnapii_positions" in fh else None
        rnapii_states = fh["rnapii_states"][:] if "rnapii_states" in fh else None
        rnapii_ids = fh["rnapii_ids"][:] if "rnapii_ids" in fh else None
        lesion_enabled = bool(fh.attrs.get("lesion_enabled", False))
        lesions = fh["lesions"][:] if "lesions" in fh else None
        genes_ds = fh["genes"][:] if "genes" in fh else None

    boundaries = annotate.boundaries_from_topology_kwargs(cfg.lef.topology_kwargs, chain_length)
    # TADs for the 3D analysis are derived per-resolution inside the 3D block
    # (see scale_resolution_coords); the 1D metrics below only need boundaries.
    anch = anchor_set(boundaries, chain_length)
    gene_bodies: List[Tuple[int, int]] = []
    if genes_ds is not None:
        for g in genes_ds:
            lo, hi = sorted((int(g["tss"]), int(g["tes"])))
            gene_bodies.append((lo, hi))
    # Directional (tss, tes) per gene -- kept separate from the sorted gene_bodies
    # so per-gene completions count Pol reaching the real TES even on reverse-
    # strand genes (tes < tss), where sorting would mistake the TSS for the TES.
    gene_dirs: List[Tuple[int, int]] = (
        [(int(g["tss"]), int(g["tes"])) for g in genes_ds] if genes_ds is not None else []
    )

    # tad_positions / anchors are chain-relative, but recorded positions are
    # absolute across all chains. Fold to chain-relative so every chain's legs
    # are matched against the (replicated) boundaries -- otherwise only chain 0
    # contributes while the denominator spans all chains, diluting the metrics
    # by ~num_chains.
    positions_rel = positions % chain_length

    metrics: Dict[str, Any] = {"chain_length": chain_length, "num_chains": num_chains}
    metrics["sanity_1d"] = sanity_1d(positions, chain_length, num_chains)
    # Finer high-end bins so the tail is resolved instead of dumped into one
    # giant 500..chain_length catch-all (was 500-15000 for a 15 kb chain). The
    # final overflow bin still spans up to chain_length but holds few loops; the
    # density-based plot renders it honestly small.
    loop_edges = [
        e for e in [0, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
        if e < chain_length
    ] + [chain_length]
    metrics["loop_length"] = loop_length_stats(positions, edges=loop_edges)
    metrics["classification"] = cohesin_classification(positions_rel, anch)
    metrics["boundary_crossing"] = boundary_crossing(positions_rel, boundaries)
    metrics["asymmetry_index"] = asymmetry_index(positions)

    ps1d = ps_curve_1d(positions, chain_length, num_chains)
    metrics["ps_1d"] = {
        "ps_at": {
            str(s): float(ps1d[s])
            for s in (5, 10, 20, 50, 100, 150, 200, 300, 500)
            if s < len(ps1d) and np.isfinite(ps1d[s])
        }
    }

    occ = cohesin_occupancy(positions, chain_length * num_chains)
    # Replicate the chain-relative anchors onto every chain before summing the
    # absolute per-site occupancy.
    anch_abs = {a + c * chain_length for c in range(num_chains) for a in anch}
    metrics["cohesin_at_ctcf_anchor_sum"] = float(
        sum(occ[s] for s in anch_abs if 0 <= s < len(occ)))
    if gene_bodies:
        metrics["cohesin_at_gene_bodies_sum"] = float(sum(occ[a:b + 1].sum() for a, b in gene_bodies))

    if (rnapii_enabled and rnapii_positions is not None
            and rnapii_states is not None and rnapii_ids is not None):
        tick_s = getattr(cfg.lef, "tick_seconds", None)
        if tick_s is None:
            tick_s = float(cfg.polymer.md_steps_per_block) * 0.0063
        metrics["rnapii"] = rnapii_metrics(
            rnapii_positions, rnapii_states, tick_seconds=float(tick_s), rnapii_ids=rnapii_ids)
        if gene_dirs:
            metrics["rnapii"]["per_gene_completions"] = rnapii_per_gene_throughput(
                rnapii_positions, gene_dirs, rnapii_ids=rnapii_ids)

    if lesion_enabled and lesions is not None:
        metrics["lesions"] = lesion_metrics(lesions, gene_bodies)
        metrics["lesions"]["cohesin_flank_enrichment"] = cohesin_at_lesion_flanks(
            positions, lesions, chain_length * num_chains)

    # plots: 1D loop length + 1D bridge-contact P(s)
    _plot_loop_hist(metrics["loop_length"], paths.plots_dir / "loop_length.png")
    _plot_ps(ps1d, paths.plots_dir / "Ps_1d.png", title="P(s) — 1D bridge contacts")

    # 3D (if available). Repeated at each configured analysis resolution: the
    # native per-monomer ICE'd map (resolution 1, written to the historical
    # keys/paths) plus any coarsened resolution N (raw map binned N x N, then
    # ICE'd), whose metrics land under ``3d_resN`` and ``*_resN.png`` plots.
    cmap_path = Path(getattr(cfg.contacts, "raw_output_path", ""))
    has_3d = cmap_path.exists() and cmap_path.suffix == ".npy"
    if has_3d:
        raw = np.load(cmap_path).astype(float)
        les_sites = (
            [int(s) for s in np.unique(lesions[lesions >= 0])]
            if (lesion_enabled and lesions is not None) else None
        )
        tss = [int(g["tss"]) for g in genes_ds] if (gene_bodies and genes_ds is not None) else []
        tes = [int(g["tes"]) for g in genes_ds] if (gene_bodies and genes_ds is not None) else []
        # Per-gene enhancer lists for the coarsened contact-map annotation.
        gene_enh: List[List[int]] = []
        if genes_ds is not None:
            for g in genes_ds:
                try:
                    raw_enh = g["enhancers"]
                except (KeyError, ValueError, TypeError):
                    raw_enh = None
                gene_enh.append([int(e) for e in raw_enh] if raw_enh is not None else [])

        resolutions = cfg.contacts.resolution_list
        oe_path = Path(getattr(cfg.contacts, "oe_output_path", ""))
        for factor in resolutions:
            # Skip resolutions too coarse for this map (would leave <3 bins, or
            # exceed the map entirely) instead of crashing the whole qc run.
            if factor > 1 and raw.shape[0] // factor < 3:
                print(f"[qc] skipping analysis_resolution {factor}: contact map of "
                      f"size {raw.shape[0]} coarsens to <3 bins")
                continue
            suffix = "" if factor == 1 else f"_res{factor}"
            # ICE balance the (optionally coarsened) map -- all downstream QC
            # uses the balanced (bias-corrected) map, per Imakaev 2012. Diagonals
            # ignored for the balancing fit so near-diagonal bias doesn't drive
            # normalisation. Coarsening happens on the *raw* map, before ICE.
            base_raw = coarsen_contact_map(raw, factor)
            m = iterative_correction(base_raw, ignore_diagonals=2, max_iter=200, tol=1e-5)
            m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
            # The cached on-disk O/E only matches the native (un-coarsened) map.
            if factor == 1 and oe_path.exists() and oe_path.suffix == ".npy":
                oe = np.load(oe_path).astype(float)
                oe = np.nan_to_num(oe, nan=1.0, posinf=1.0, neginf=1.0)
            else:
                oe = observed_over_expected(m)
                oe = np.nan_to_num(oe, nan=1.0, posinf=1.0, neginf=1.0)

            n_bins = m.shape[0]
            sc = scale_resolution_coords(boundaries, tss, tes, les_sites, factor, n_bins)
            if factor == 1:
                ann = annotate.from_lef_cfg(cfg.lef)
            else:
                ann = annotate.from_lists(sc["boundaries"], sc["gene_tss"], sc["gene_tes"],
                                          [[e // factor for e in lst] for lst in gene_enh],
                                          origin=0, span=n_bins)
            m3d = _analyze_3d(
                m, oe,
                boundaries=sc["boundaries"], tads=sc["tads"],
                gene_tss=sc["gene_tss"], gene_tes=sc["gene_tes"],
                lesion_sites=sc["lesion_sites"],
                plots_dir=paths.plots_dir, annotations=ann,
                resolution=factor, suffix=suffix,
            )
            metrics[f"3d{suffix}"] = m3d

    paths.metrics_json.write_text(json.dumps(metrics, indent=2))
    _write_report(metrics, paths)
    return paths.folder


def _write_report(metrics: Dict[str, Any], paths: QCPaths) -> None:
    lines: List[str] = ["# Simulation QC report\n"]
    lines.append(f"- chain_length: {metrics['chain_length']}")
    lines.append(f"- num_chains: {metrics['num_chains']}")
    s1 = metrics["sanity_1d"]
    lines.append("\n## Sanity (1D)")
    for k, v in s1.items():
        lines.append(f"- {k}: {v}")
    ll = metrics["loop_length"]
    lines.append("\n## Loop length")
    lines.append(f"- mean: {ll['mean']:.1f}  median: {ll['median']:.1f}  p10: {ll['p10']:.1f}  p90: {ll['p90']:.1f}")
    cls = metrics["classification"]
    lines.append("\n## Cohesin classification")
    lines.append(f"- corner: {cls['corner_pct']:.1f}%   stripe: {cls['stripe_pct']:.1f}%   free: {cls['free_pct']:.1f}%")
    lines.append(f"- asymmetry index: {metrics['asymmetry_index']:.3f}")
    bc = metrics["boundary_crossing"]
    lines.append("\n## Boundary crossing")
    lines.append(f"- mean: {bc['mean']:.3f}")
    if "ps_1d" in metrics:
        lines.append("\n## P(s) — 1D bridge contacts")
        lines.append("- P(s) at separations: " + "  ".join(
            f"s={k}:{v:.2e}" for k, v in metrics["ps_1d"]["ps_at"].items()))
    if "rnapii" in metrics:
        r = metrics["rnapii"]
        lines.append("\n## RNAPII")
        lines.append(f"- present: {r['frames_with_any_polii_pct']:.1f}%   mean#: {r['mean_count_when_present']:.2f}   max#: {r['max_simultaneous']}")
        if "elongation_speed_kb_per_min" in r:
            lines.append(f"- realized elongation: {r['elongation_speed_kb_per_min']:.2f} kb/min ({r['elongation_speed_sites_per_tick']:.3f} sites/tick)")
        if "state_mix_pct" in r:
            lines.append(f"- state mix %: {r['state_mix_pct']}")
    if "lesions" in metrics:
        l = metrics["lesions"]
        lines.append("\n## Lesions")
        lines.append(f"- mean: {l['mean_simultaneous']:.2f}   max: {l['max_simultaneous']}   in-body: {l['lesions_in_gene_body_pct']:.1f}%")
        lines.append(f"- cohesin flank enrichment: {l['cohesin_flank_enrichment']['enrichment_x']:.2f}x")
    if "3d" in metrics:
        m3 = metrics["3d"]
        lines.append("\n## 3D")
        lines.append(f"- contact-map shape: {m3['shape']}")
        lines.append(f"- TAD strength (intra/inter): {m3['tad_strength']['intra_over_inter']:.2f}")
        lines.append(f"- corner-dot intensities per TAD: {[round(x, 1) for x in m3['corner_dot_intensities']]}")
        lines.append("- P(s) (3D contact map) at separations: " + "  ".join(f"s={k}:{v:.2e}" for k, v in m3["ps_at"].items()))
        lines.append("- insulation boundary strength (cooltools log2 prominence; higher = stronger; per window):")
        for w, d in m3["insulation_boundary_strength"].items():
            lines.append(f"  - w={w}: mean={d.get('mean', float('nan')):.3f}")
        if "insulation_aggregate" in m3:
            lines.append("- aggregate insulation centered at boundaries (cooltools log2 dip_depth = flank - center; higher = stronger):")
            for w, d in m3["insulation_aggregate"].items():
                dip = d.get("dip_depth", float("nan"))
                lines.append(f"  - w={w}: dip_depth={dip:.3f} (n={d.get('n', 0)})")
    lines.append("\n## Plots")
    lines.append("- [loop_length.png](plots/loop_length.png)")
    lines.append("- [Ps_1d.png](plots/Ps_1d.png) -- 1D bridge-contact P(s)")
    if "3d" in metrics:
        lines.append("- All 3D metrics use the **ICE-balanced** contact map.")
        lines.append("- [Ps_3d.png](plots/Ps_3d.png) -- 3D contact-map P(s)")
        lines.append("- [insulation_windows.png](plots/insulation_windows.png) -- cooltools-style log2 insulation score")
        lines.append("- [insulation_aggregate.png](plots/insulation_aggregate.png) -- aggregate log2 insulation centered at boundaries")
        lines.append("- [contact_map.png](plots/contact_map.png)")
        lines.append("- [pileups_obs.png](plots/pileups_obs.png) -- observed pile-ups around CTCF / TSS / TES / lesions")
        lines.append("- [pileups_oe.png](plots/pileups_oe.png) -- observed/expected pile-ups (same anchors)")
        lines.append("- [pileups_tad_rescaled.png](plots/pileups_tad_rescaled.png) -- Flyamer 2017 rescaled-TAD pile-up (obs + O/E)")
        if "tad_pileup" in metrics["3d"]:
            tp = metrics["3d"]["tad_pileup"]
            lines.append(f"\n## Rescaled-TAD pile-up (Flyamer 2017, n_valid_tads={tp.get('n_valid_tads')})")
            lines.append(
                "For cross-config comparisons, treat O/E strength as primary; "
                "raw obs strength is sensitive to P(s) and global contact redistribution."
            )
            if "obs" in tp:
                o = tp["obs"]
                lines.append(f"- raw obs:  within={o['within']:.2f}  between={o['between']:.2f}  **strength={o['strength']:.2f}**")
            if "oe" in tp:
                o = tp["oe"]
                lines.append(f"- O/E primary:  within={o['within']:.2f}  between={o['between']:.2f}  **strength={o['strength']:.2f}**")
            if "per_tad" in tp:
                lines.append("- per-TAD strengths (obs / O/E):")
                for k, v in tp["per_tad"].items():
                    lines.append(f"  - TAD {k}: obs={v['obs']['strength']:.2f}   O/E={v.get('oe', {}).get('strength', float('nan')):.2f}")

    # Coarsened-resolution 3D analyses (raw map binned N x N, then ICE'd).
    for key in sorted(k for k in metrics if k.startswith("3d_res")):
        factor = key[len("3d_res"):]
        suffix = f"_res{factor}"
        m3 = metrics[key]
        lines.append(f"\n## 3D @ {factor}-monomer resolution (coarsened then ICE-balanced)")
        lines.append(f"- contact-map shape: {m3['shape']}")
        lines.append(f"- TAD strength (intra/inter): {m3['tad_strength']['intra_over_inter']:.2f}")
        lines.append(f"- corner-dot intensities per TAD: {[round(x, 1) for x in m3['corner_dot_intensities']]}")
        lines.append("- P(s) (3D contact map) at separations: " + "  ".join(
            f"s={k}:{v:.2e}" for k, v in m3["ps_at"].items()))
        if "insulation_boundary_strength" in m3:
            lines.append("- insulation boundary strength (cooltools log2 prominence; higher = stronger; per window):")
            for w, d in m3["insulation_boundary_strength"].items():
                lines.append(f"  - w={w}: mean={d.get('mean', float('nan')):.3f}")
        if "insulation_aggregate" in m3:
            lines.append("- aggregate insulation dip_depth (cooltools log2 flank - center; higher = stronger):")
            for w, d in m3["insulation_aggregate"].items():
                dip = d.get("dip_depth", float("nan"))
                lines.append(f"  - w={w}: dip_depth={dip:.3f} (n={d.get('n', 0)})")
        tp = m3.get("tad_pileup", {})
        if "obs" in tp:
            o = tp["obs"]
            lines.append(f"- rescaled-TAD raw obs:  within={o['within']:.2f}  between={o['between']:.2f}  **strength={o['strength']:.2f}**")
        if "oe" in tp:
            o = tp["oe"]
            lines.append(f"- rescaled-TAD O/E primary:  within={o['within']:.2f}  between={o['between']:.2f}  **strength={o['strength']:.2f}**")
        for fn in ("Ps_3d", "insulation_windows", "insulation_aggregate",
                   "contact_map", "pileups_obs", "pileups_oe", "pileups_tad_rescaled"):
            lines.append(f"- [{fn}{suffix}.png](plots/{fn}{suffix}.png)")

    paths.report_md.write_text("\n".join(lines))
