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
                   tick_seconds: float = 8.0) -> Dict[str, Any]:
    """State mix + convoy + realized elongation speed."""
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
        }
        # realized elongation speed: per slot, sum |step| while ELONG. Filter
        # to |delta| <= 1 to drop slot-reuse jumps (a slot is recycled when a
        # Pol II unloads and another loads -> apparent huge delta in 1 tick).
        adv = 0
        eticks = 0
        gids = rnapii_positions[:, :, 1]
        for k in range(rnapii_positions.shape[1]):
            pos = rnapii_positions[:, k, 0]
            stt = states[:, k]
            gid = gids[:, k]
            for t in range(1, rnapii_positions.shape[0]):
                if stt[t] != 2 or stt[t - 1] != 2:
                    continue
                if pos[t] < 0 or pos[t - 1] < 0:
                    continue
                if gid[t] != gid[t - 1]:        # slot recycled
                    continue
                d = int(pos[t] - pos[t - 1])
                if abs(d) > 1:                  # >1 site/tick = not a true step
                    continue
                adv += abs(d)
                eticks += 1
        sites_per_tick = adv / max(eticks, 1)
        out["elongation_speed_sites_per_tick"] = float(sites_per_tick)
        out["elongation_speed_kb_per_min"] = float(sites_per_tick / tick_seconds * 60.0)
    return out


def rnapii_per_gene_throughput(rnapii_positions: np.ndarray, genes: List[Tuple[int, int]]) -> Dict[str, int]:
    """Approx throughput = # distinct slot-tracks whose final pos crosses TES."""
    # crude: count slot-frames where pos==tes
    out: Dict[str, int] = {}
    for i, (tss, tes) in enumerate(genes):
        hits = int((rnapii_positions[:, :, 0] == tes).sum())
        out[f"gene{i}_tes_visits"] = hits
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
    """Contact probability vs genomic separation."""
    N = m.shape[0]
    out = np.full(N, np.nan)
    for s in range(1, N):
        out[s] = np.diagonal(m, s).mean()
    return out


def insulation_profile(m: np.ndarray, window: int) -> np.ndarray:
    N = m.shape[0]
    out = np.full(N, np.nan)
    for i in range(window, N - window):
        out[i] = m[i - window:i, i:i + window].mean()
    return out


def insulation_boundary_strength(profile: np.ndarray, boundaries: List[int], window: int) -> Dict[str, float]:
    """Local min around each boundary / global mean. Lower = stronger insulation."""
    valid = profile[~np.isnan(profile)]
    gm = float(valid.mean()) if valid.size else float("nan")
    out: Dict[str, float] = {"global_mean": gm}
    for b in boundaries:
        lo = max(b - window // 2, 0)
        hi = min(b + window // 2, len(profile))
        out[str(b)] = float(np.nanmin(profile[lo:hi]) / gm) if gm > 0 else float("nan")
    return out


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


def rescaled_tad_pileup(m: np.ndarray, tads: List[Tuple[int, int]],
                        target: int = 90) -> Tuple[Optional[np.ndarray], List[np.ndarray], List[int]]:
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
    itself and the side thirds are the neighbours. ``target=90`` gives the
    classic [30:60, 30:60] / [0:30, 30:60] / [30:60, 60:90] slicing.
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
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(frac))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(frac)), frac, tick_label=[f"{edges[i]}-{edges[i + 1]}" for i in range(len(frac))])
    ax.set_xlabel("loop length bin (kb)")
    ax.set_ylabel("fraction")
    ax.set_title("Cohesin loop length distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_ps(ps: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    s = np.arange(1, len(ps))
    ax.loglog(s, ps[1:])
    ax.set_xlabel("s (kb)")
    ax.set_ylabel("P(s)")
    ax.set_title("Contact probability vs genomic separation")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_insulation(m: np.ndarray, windows: List[int], boundaries: List[int], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(windows)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    for k, w in enumerate(windows):
        prof = insulation_profile(m, w)
        axes[k].plot(prof)
        for b in boundaries:
            axes[k].axvline(b, color="k", ls=":", alpha=0.4)
        axes[k].set_title(f"window={w} kb")
        axes[k].set_ylabel("insulation (raw)")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    axes[-1].set_xlabel("position (kb)")
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


def _plot_contact_map(m: np.ndarray, boundaries: List[int], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.log1p(m), cmap="inferno", origin="lower")
    for b in boundaries:
        ax.axhline(b, color="cyan", ls="--", alpha=0.5, lw=0.7)
        ax.axvline(b, color="cyan", ls="--", alpha=0.5, lw=0.7)
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
        lesion_enabled = bool(fh.attrs.get("lesion_enabled", False))
        lesions = fh["lesions"][:] if "lesions" in fh else None
        genes_ds = fh["genes"][:] if "genes" in fh else None

    boundaries = list(cfg.lef.topology_kwargs.get("tad_positions", [])) if isinstance(cfg.lef.topology_kwargs, dict) else []
    tads = [(s, e) for s, e in zip([0, *boundaries], [*boundaries, chain_length])]
    anch = anchor_set(boundaries, chain_length)
    gene_bodies: List[Tuple[int, int]] = []
    if genes_ds is not None:
        for g in genes_ds:
            lo, hi = sorted((int(g["tss"]), int(g["tes"])))
            gene_bodies.append((lo, hi))

    metrics: Dict[str, Any] = {"chain_length": chain_length, "num_chains": num_chains}
    metrics["sanity_1d"] = sanity_1d(positions, chain_length, num_chains)
    metrics["loop_length"] = loop_length_stats(positions, edges=[0, 50, 100, 150, 200, 300, 500, chain_length])
    metrics["classification"] = cohesin_classification(positions, anch)
    metrics["boundary_crossing"] = boundary_crossing(positions, boundaries)
    metrics["asymmetry_index"] = asymmetry_index(positions)

    occ = cohesin_occupancy(positions, chain_length * num_chains)
    metrics["cohesin_at_ctcf_anchor_sum"] = float(sum(occ[s] for s in anch if s < len(occ)))
    if gene_bodies:
        metrics["cohesin_at_gene_bodies_sum"] = float(sum(occ[a:b + 1].sum() for a, b in gene_bodies))

    if rnapii_enabled and rnapii_positions is not None and rnapii_states is not None:
        metrics["rnapii"] = rnapii_metrics(rnapii_positions, rnapii_states)
        if gene_bodies:
            metrics["rnapii"]["per_gene_tes_visits"] = rnapii_per_gene_throughput(
                rnapii_positions, gene_bodies)

    if lesion_enabled and lesions is not None:
        metrics["lesions"] = lesion_metrics(lesions, gene_bodies)
        metrics["lesions"]["cohesin_flank_enrichment"] = cohesin_at_lesion_flanks(
            positions, lesions, chain_length * num_chains)

    # plots: 1D loop length
    _plot_loop_hist(metrics["loop_length"], paths.plots_dir / "loop_length.png")

    # 3D (if available)
    cmap_path = Path(getattr(cfg.contacts, "raw_output_path", ""))
    has_3d = cmap_path.exists() and cmap_path.suffix == ".npy"
    if has_3d:
        raw = np.load(cmap_path).astype(float)
        # ICE balance the contact map -- all downstream QC uses the balanced
        # (bias-corrected) map, per Imakaev 2012. Diagonals ignored for the
        # balancing fit so the near-diagonal bias doesn't drive normalisation.
        m = iterative_correction(raw, ignore_diagonals=2, max_iter=200, tol=1e-5)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        # Load (or compute) ICE-balanced O/E for the O/E pile-ups.
        oe_path = Path(getattr(cfg.contacts, "oe_output_path", ""))
        if oe_path.exists() and oe_path.suffix == ".npy":
            oe = np.load(oe_path).astype(float)
            oe = np.nan_to_num(oe, nan=1.0, posinf=1.0, neginf=1.0)
        else:
            oe = observed_over_expected(m)
            oe = np.nan_to_num(oe, nan=1.0, posinf=1.0, neginf=1.0)

        ps = ps_curve(m)
        windows = [5, 10, 20, 40, 80, 120]
        m3d: Dict[str, Any] = {
            "shape": list(m.shape),
            "ice_balanced": True,
            "tad_strength": tad_strength(m, tads),
            "corner_dot_intensities": corner_dot_intensities(m, tads),
            "ps_at": {str(s): float(ps[s]) for s in (5, 10, 20, 50, 100, 150, 200, 300, 500) if s < len(ps)},
            "insulation_boundary_strength": {
                str(w): insulation_boundary_strength(insulation_profile(m, w), boundaries, w)
                for w in windows
            },
            "stripe_enrichment_per_boundary": {
                str(b): stripe_enrichment(m, b) for b in boundaries
            },
        }
        metrics["3d"] = m3d
        _plot_ps(ps, paths.plots_dir / "Ps.png")
        _plot_insulation(m, windows, boundaries, paths.plots_dir / "insulation_windows.png")
        _plot_contact_map(m, boundaries, paths.plots_dir / "contact_map.png")
        # Pile-ups (centered on each anchor type) -- obs and O/E, both ICE-balanced.
        anchor_groups: Dict[str, List[int]] = {"CTCF boundary": list(boundaries)}
        if gene_bodies:
            tss = [int(g["tss"]) for g in genes_ds] if genes_ds is not None else []
            tes = [int(g["tes"]) for g in genes_ds] if genes_ds is not None else []
            anchor_groups["gene TSS"] = tss
            anchor_groups["gene TES"] = tes
        if lesion_enabled and lesions is not None:
            les_sites = [int(s) for s in np.unique(lesions[lesions >= 0])]
            if les_sites:
                anchor_groups["lesions"] = les_sites
        m3d["pileups_obs"] = _plot_pileups(m, anchor_groups, paths.plots_dir / "pileups_obs.png", half=40)
        m3d["pileups_oe"] = _plot_pileups(oe, anchor_groups, paths.plots_dir / "pileups_oe.png", half=40)

        # Flyamer 2017 rescaled-TAD pile-ups (obs + O/E). 3xTAD window scaled
        # to 90x90; within = central [30:60,30:60], between = mean of the two
        # flanking [0:30,30:60] and [30:60,60:90] blocks; strength = within/between.
        avg_obs, snips_obs, kept = rescaled_tad_pileup(m, tads, target=90)
        avg_oe, snips_oe, _ = rescaled_tad_pileup(oe, tads, target=90)
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
                                  paths.plots_dir / "pileups_tad_rescaled.png")

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
        lines.append("- P(s) at separations: " + "  ".join(f"s={k}:{v:.2e}" for k, v in m3["ps_at"].items()))
        lines.append("- insulation_boundary_strength (lower = stronger; per window):")
        for w, d in m3["insulation_boundary_strength"].items():
            lines.append(f"  - w={w}: {d}")
    lines.append("\n## Plots")
    lines.append("- [loop_length.png](plots/loop_length.png)")
    if "3d" in metrics:
        lines.append("- All 3D metrics use the **ICE-balanced** contact map.")
        lines.append("- [Ps.png](plots/Ps.png)")
        lines.append("- [insulation_windows.png](plots/insulation_windows.png)")
        lines.append("- [contact_map.png](plots/contact_map.png)")
        lines.append("- [pileups_obs.png](plots/pileups_obs.png) -- observed pile-ups around CTCF / TSS / TES / lesions")
        lines.append("- [pileups_oe.png](plots/pileups_oe.png) -- observed/expected pile-ups (same anchors)")
        lines.append("- [pileups_tad_rescaled.png](plots/pileups_tad_rescaled.png) -- Flyamer 2017 rescaled-TAD pile-up (obs + O/E)")
        if "tad_pileup" in metrics["3d"]:
            tp = metrics["3d"]["tad_pileup"]
            lines.append(f"\n## Rescaled-TAD pile-up (Flyamer 2017, n_valid_tads={tp.get('n_valid_tads')})")
            if "obs" in tp:
                o = tp["obs"]
                lines.append(f"- obs:  within={o['within']:.2f}  between={o['between']:.2f}  **strength={o['strength']:.2f}**")
            if "oe" in tp:
                o = tp["oe"]
                lines.append(f"- O/E:  within={o['within']:.2f}  between={o['between']:.2f}  **strength={o['strength']:.2f}**")
            if "per_tad" in tp:
                lines.append("- per-TAD strengths (obs / O/E):")
                for k, v in tp["per_tad"].items():
                    lines.append(f"  - TAD {k}: obs={v['obs']['strength']:.2f}   O/E={v.get('oe', {}).get('strength', float('nan')):.2f}")
    paths.report_md.write_text("\n".join(lines))
