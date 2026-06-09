"""Pairwise and baseline-vs-many comparison of pipeline outputs (1D + 3D).

Takes two configs (``cfg_a``, ``cfg_b``) pointing to two run folders (each with
``LEFPositions.h5`` + ``contact_map.npy`` + ``contact_map_oe.npy``) and produces
side-by-side metrics + comparison plots into ``out_dir``:

* ``compare.json``  -- A, B and their folds/diffs, every metric
* ``compare.md``    -- short readable summary
* ``plots/``        -- loop-length, P(s), insulation, contact-map, anchor
                       pile-ups (obs + O/E), Flyamer rescaled-TAD pile-up.

All 3D metrics use the **ICE-balanced** map; all plot color scales use
**1st/99th percentile** clipping so outliers don't dominate.

CLI: ``polychrom-loopext compare cfgA.yaml cfgB.yaml [--folder-a PATH]
[--folder-b PATH] [--out DIR] [--label-a NAME] [--label-b NAME]``.
Multiple comparisons are accepted as ``polychrom-loopext compare baseline_run
run2 run3 ...`` and write pairwise subdirectories plus ``compare_many.*``.

With ``--cutoffs 2 3 4 5 6`` each cutoff is run into its own ``cutoff_<c>/``
subfolder and a consolidated ``tad_strength_vs_cutoff.{md,json}`` is written at
the top level: Flyamer rescaled-TAD strength (obs + O/E, plus B/A fold) per
contact cutoff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from ...hdf5_format import list_URIs
from . import annotate
from .config import resolve_plugin
from .contacts import _effective_map_starts
from .qc import (
    aggregate_insulation_log2, anchor_set, asymmetry_index,
    boundary_crossing, boundary_crossing_stripes, coarsen_contact_map,
    cohesin_at_lesion_flanks, cohesin_classification, cohesin_occupancy,
    corner_dot_intensities, insulation_aggregate_half,
    insulation_boundary_prominence,
    insulation_score_cooltools, insulation_windows_for_resolution,
    lesion_metrics, loop_length_stats, observed_over_expected, pileup,
    ps_curve, ps_curve_1d, rescaled_tad_pileup, rnapii_metrics, sanity_1d,
    scale_resolution_coords, stripe_enrichment, tad_strength,
    tad_strength_from_pileup, TAD_PILEUP_RESOLUTION,
)
from .plugins.sampling import iterative_correction


# Upper bound on the genomic separation (in bins/kb) used for the 3D P(s)
# comparison, as a fraction of the contact-map size. The large-s tail is
# dominated by a handful of long-range contacts and is too noisy to compare;
# curves are truncated here before AUC-normalization. Tune to taste.
PS_3D_S_MAX_FRAC = 0.8


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    label: str
    positions: np.ndarray
    rnapii_positions: Optional[np.ndarray]
    rnapii_states: Optional[np.ndarray]
    rnapii_ids: Optional[np.ndarray]
    rnapii_enabled: bool
    tick_seconds: float
    lesions: Optional[np.ndarray]
    lesion_enabled: bool
    genes_ds: Optional[np.ndarray]
    gene_tss: List[int]
    gene_tes: List[int]
    gene_enhancers: List[List[int]]
    chain_length: int
    num_chains: int
    boundaries: List[int]
    tads: List[Tuple[int, int]]
    gene_bodies: List[Tuple[int, int]]
    cmap_obs: Optional[np.ndarray]
    cmap_oe: Optional[np.ndarray]
    # Raw (un-ICE'd, un-coarsened) contact map, kept so the coarsened-resolution
    # analyses can bin it before ICE. None when no 3D map is available.
    cmap_raw: Optional[np.ndarray] = None


def _gene_coordinates(genes: Any) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    if genes is None:
        return [], [], []
    tss: List[int] = []
    tes: List[int] = []
    bodies: List[Tuple[int, int]] = []
    for g in genes:
        start = int(g["tss"])
        end = int(g["tes"])
        lo, hi = sorted((start, end))
        tss.append(start)
        tes.append(end)
        bodies.append((lo, hi))
    return tss, tes, bodies


def _gene_enhancers(genes: Any) -> List[List[int]]:
    """Per-gene list of enhancer positions (a gene may have several)."""
    if genes is None:
        return []
    out: List[List[int]] = []
    for g in genes:
        try:
            raw = g["enhancers"]
        except (KeyError, ValueError, TypeError):
            raw = None
        if raw is None:
            try:
                ep = g["enhancer_pos"]
            except (KeyError, ValueError, TypeError):
                ep = None
            raw = [] if ep is None else [ep]
        out.append([int(e) for e in raw])
    return out


def _sample_raw(cfg, cutoff: float) -> np.ndarray:
    """Re-sample a raw contact map from the run's trajectory at ``cutoff``.

    Mirrors the ``contacts`` stage sampler so per-cutoff comparisons use the
    same map_starts / sampler plugin the run was configured with.
    """
    contacts_cfg = replace(cfg.contacts, cutoff=float(cutoff))
    contacts_cfg = replace(
        contacts_cfg, map_starts=_effective_map_starts(contacts_cfg, cfg.lef))
    uris = list_URIs(str(contacts_cfg.trajectory_folder))
    if not uris:
        raise FileNotFoundError(
            f"No trajectory blocks under {contacts_cfg.trajectory_folder}; "
            "cutoff resampling needs the polymer trajectory"
        )
    sampler = resolve_plugin(contacts_cfg.plugins.sampler)
    raw = sampler(uris, cfg=contacts_cfg, **contacts_cfg.plugins.sampler.kwargs)
    return np.asarray(raw, dtype=float)


def _load(cfg, label: str, cutoff: Optional[float] = None) -> RunData:
    lef_h5 = Path(cfg.lef.output_path)
    if not lef_h5.exists():
        raise FileNotFoundError(f"[{label}] LEFPositions.h5 not found: {lef_h5}")
    with h5py.File(lef_h5, "r") as fh:
        pos = fh["positions"][:]
        n_sites = int(fh.attrs["N"])
        chain_length = int(fh.attrs.get("chain_length", n_sites))
        num_chains = int(fh.attrs.get("num_chains", 1))
        rnapii_enabled = bool(fh.attrs.get("rnapii_enabled", False))
        rp = fh["rnapii_positions"][:] if "rnapii_positions" in fh else None
        rs = fh["rnapii_states"][:] if "rnapii_states" in fh else None
        ri = fh["rnapii_ids"][:] if "rnapii_ids" in fh else None
        lesion_enabled = bool(fh.attrs.get("lesion_enabled", False))
        les = fh["lesions"][:] if "lesions" in fh else None
        genes_ds = fh["genes"][:] if "genes" in fh else None

    boundaries = list(cfg.lef.topology_kwargs.get("tad_positions", []))
    tads = [(s, e) for s, e in zip([0, *boundaries], [*boundaries, chain_length])]
    config_genes = cfg.lef.topology_kwargs.get("genes", [])
    gene_source = genes_ds if genes_ds is not None else config_genes
    gene_tss, gene_tes, gene_bodies = _gene_coordinates(gene_source)
    gene_enhancers = _gene_enhancers(config_genes)

    cmap_obs = None
    cmap_oe = None
    cmap_raw = None
    if cutoff is not None:
        raw = _sample_raw(cfg, cutoff)
        cmap_raw = raw
        cmap_obs = np.nan_to_num(
            iterative_correction(raw, ignore_diagonals=2, max_iter=200, tol=1e-5)
        )
        cmap_oe = np.nan_to_num(observed_over_expected(cmap_obs), nan=1.0)
    else:
        raw_p = Path(getattr(cfg.contacts, "raw_output_path", ""))
        if raw_p.exists():
            raw = np.load(raw_p).astype(float)
            cmap_raw = raw
            cmap_obs = np.nan_to_num(
                iterative_correction(raw, ignore_diagonals=2, max_iter=200, tol=1e-5)
            )
            oe_p = Path(getattr(cfg.contacts, "oe_output_path", ""))
            if oe_p.exists():
                cmap_oe = np.nan_to_num(np.load(oe_p).astype(float), nan=1.0)
            else:
                cmap_oe = np.nan_to_num(observed_over_expected(cmap_obs), nan=1.0)

    polymer_cfg = getattr(cfg, "polymer", None)
    tick_seconds = getattr(cfg.lef, "tick_seconds", None)
    if tick_seconds is None:
        md_steps_per_block = getattr(polymer_cfg, "md_steps_per_block", 0.0)
        tick_seconds = float(md_steps_per_block) * 0.0063
    return RunData(
        label=label, positions=pos, rnapii_positions=rp, rnapii_states=rs,
        rnapii_ids=ri, rnapii_enabled=rnapii_enabled,
        tick_seconds=float(tick_seconds),
        lesions=les, lesion_enabled=lesion_enabled,
        genes_ds=genes_ds, gene_tss=gene_tss, gene_tes=gene_tes,
        gene_enhancers=gene_enhancers,
        chain_length=chain_length, num_chains=num_chains,
        boundaries=boundaries, tads=tads, gene_bodies=gene_bodies,
        cmap_obs=cmap_obs, cmap_oe=cmap_oe, cmap_raw=cmap_raw,
    )


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def _collect_1d(r: RunData) -> Dict[str, Any]:
    anch = anchor_set(r.boundaries, r.chain_length)
    # Boundaries/anchors are chain-relative, while LEFPositions stores absolute
    # coordinates across replicated chains. Fold to chain-relative for boundary
    # and anchor-classification metrics, matching qc.py; otherwise only chain 0
    # matches the anchors while all chains remain in the denominator.
    positions_rel = r.positions % r.chain_length
    occ = cohesin_occupancy(r.positions, r.chain_length * r.num_chains)
    anch_abs = {
        a + c * r.chain_length
        for c in range(r.num_chains)
        for a in anch
    }
    out: Dict[str, Any] = {
        "sanity": sanity_1d(r.positions, r.chain_length, r.num_chains),
        "loop_length": loop_length_stats(
            r.positions,
            edges=[e for e in [0, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
                   if e < r.chain_length] + [r.chain_length]),
        "classification": cohesin_classification(positions_rel, anch),
        "boundary_crossing": boundary_crossing(positions_rel, r.boundaries),
        "boundary_crossing_stripes": boundary_crossing_stripes(positions_rel, r.boundaries),
        "asymmetry_index": asymmetry_index(r.positions),
        "cohesin_at_ctcf_anchor_sum": float(
            sum(occ[s] for s in anch_abs if 0 <= s < len(occ))
        ),
    }
    ps1d = ps_curve_1d(r.positions, r.chain_length, r.num_chains)
    out["ps_1d"] = {
        "ps_at": {
            str(s): float(ps1d[s])
            for s in (5, 10, 20, 50, 100, 150, 200, 300, 500)
            if s < len(ps1d) and np.isfinite(ps1d[s])
        }
    }
    if r.gene_bodies:
        out["cohesin_at_gene_bodies_sum"] = float(sum(occ[a:b + 1].sum() for a, b in r.gene_bodies))
    if (r.rnapii_enabled and r.rnapii_positions is not None
            and r.rnapii_states is not None and r.rnapii_ids is not None):
        out["rnapii"] = rnapii_metrics(
            r.rnapii_positions, r.rnapii_states,
            tick_seconds=r.tick_seconds, rnapii_ids=r.rnapii_ids)
    if r.lesion_enabled and r.lesions is not None:
        out["lesions"] = lesion_metrics(r.lesions, r.gene_bodies)
        out["lesions"]["cohesin_flank_enrichment"] = cohesin_at_lesion_flanks(
            r.positions, r.lesions, r.chain_length * r.num_chains)
    return out


def _rescaled_crossing_stripes(pile: np.ndarray, kind: str, band: Optional[int] = None) -> Dict[str, float]:
    n = pile.shape[0]
    third = n // 3
    two_thirds = 2 * third
    # Stripe half-width: a fixed fraction of the pile (3 px at the legacy 90-px
    # resolution). Scale with size so the *physical* band stays constant when
    # TAD_PILEUP_RESOLUTION changes, keeping the metric comparable across runs.
    if band is None:
        band = max(1, round(3 * n / 90))
    if kind == "oe":
        mat = np.log2(np.clip(pile, 1e-3, None))
    elif kind == "obs":
        mat = np.log1p(pile)
    else:
        raise ValueError(f"Unknown pileup kind: {kind}")

    def mean(mask: np.ndarray) -> float:
        vals = mat[mask]
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else 0.0

    yy, xx = np.indices(mat.shape)
    left_cross = (
        ((yy >= third) & (yy < two_thirds) & (xx < third)) |
        ((xx >= third) & (xx < two_thirds) & (yy < third))
    )
    right_cross = (
        ((yy >= third) & (yy < two_thirds) & (xx >= two_thirds)) |
        ((xx >= third) & (xx < two_thirds) & (yy >= two_thirds))
    )
    left_line = left_cross & ((np.abs(xx - third) < band) | (np.abs(yy - third) < band))
    right_line = right_cross & ((np.abs(xx - two_thirds) < band) | (np.abs(yy - two_thirds) < band))
    left_bg = left_cross & ~left_line
    right_bg = right_cross & ~right_line

    left_line_mean = mean(left_line)
    left_bg_mean = mean(left_bg)
    right_line_mean = mean(right_line)
    right_bg_mean = mean(right_bg)
    left_delta = left_line_mean - left_bg_mean
    right_delta = right_line_mean - right_bg_mean
    return {
        "left_line_mean": left_line_mean,
        "left_background_mean": left_bg_mean,
        "left_contrast": left_delta,
        "right_line_mean": right_line_mean,
        "right_background_mean": right_bg_mean,
        "right_contrast": right_delta,
        "mean_contrast": float(np.mean([left_delta, right_delta])),
    }


def _mean_values(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else 0.0


def _fountain_stats(m: np.ndarray, gene_tss: List[int], gene_tes: List[int],
                    gene_enhancers: List[List[int]], *,
                    is_oe: bool, window: int = 80, inner: int = 5) -> Dict[str, Any]:
    """Gene-centered two-sided row/column arm enrichment.

    Fountain score summarizes contacts extending from each TSS to both genomic
    sides. For OBS, score is arm mean / local square background. For O/E, score
    is arm mean because 1 is expected.
    """
    n = m.shape[0]
    left_vals: List[float] = []
    right_vals: List[float] = []
    bg_vals: List[float] = []
    per_gene: Dict[str, Any] = {}

    for idx, tss in enumerate(gene_tss):
        center = int(tss)
        if center < 0 or center >= n:
            continue
        left = np.arange(max(0, center - window), max(0, center - inner))
        right = np.arange(min(n, center + inner + 1), min(n, center + window + 1))
        if not len(left) or not len(right):
            continue

        left_arm = _mean_values([
            *m[center, left].astype(float).tolist(),
            *m[left, center].astype(float).tolist(),
        ])
        right_arm = _mean_values([
            *m[center, right].astype(float).tolist(),
            *m[right, center].astype(float).tolist(),
        ])
        lo = max(0, center - window)
        hi = min(n, center + window + 1)
        local = m[lo:hi, lo:hi].astype(float).copy()
        local_center = center - lo
        local[max(0, local_center - inner):local_center + inner + 1, :] = np.nan
        local[:, max(0, local_center - inner):local_center + inner + 1] = np.nan
        bg = _mean_values(local.ravel().tolist())

        left_vals.append(left_arm)
        right_vals.append(right_arm)
        bg_vals.append(bg)
        mean_arm = 0.5 * (left_arm + right_arm)
        per_gene[str(idx)] = {
            "tss": center,
            "tes": int(gene_tes[idx]) if idx < len(gene_tes) else None,
            "left_arm_mean": left_arm,
            "right_arm_mean": right_arm,
            "background_mean": bg,
            "fountain_score": mean_arm if is_oe else (mean_arm / bg if bg else 0.0),
            "symmetry": min(left_arm, right_arm) / max(left_arm, right_arm) if max(left_arm, right_arm) else 0.0,
        }

    mean_left = _mean_values(left_vals)
    mean_right = _mean_values(right_vals)
    mean_bg = _mean_values(bg_vals)
    mean_arm = 0.5 * (mean_left + mean_right)
    enh_vals: List[float] = []
    for tss, enhancers in zip(gene_tss, gene_enhancers):
        for enhancer in enhancers:
            if 0 <= int(tss) < n and 0 <= int(enhancer) < n:
                enh_vals.append(float(0.5 * (m[int(tss), int(enhancer)] + m[int(enhancer), int(tss)])))

    return {
        "n_genes": len(per_gene),
        "left_arm_mean": mean_left,
        "right_arm_mean": mean_right,
        "background_mean": mean_bg,
        "fountain_score": mean_arm if is_oe else (mean_arm / mean_bg if mean_bg else 0.0),
        "symmetry": min(mean_left, mean_right) / max(mean_left, mean_right) if max(mean_left, mean_right) else 0.0,
        "enhancer_tss_mean": _mean_values(enh_vals),
        "n_enhancer_pairs": len(enh_vals),
        "per_gene": per_gene,
    }


def _collect_3d_maps(
    cmap_obs: Optional[np.ndarray], cmap_oe: Optional[np.ndarray],
    tads: List[Tuple[int, int]], boundaries: List[int],
    gene_tss: List[int], gene_tes: List[int],
    gene_enhancers: List[List[int]], *, resolution: int = 1,
) -> Optional[Dict[str, Any]]:
    """Every 3D metric on explicit obs/expected maps + bin-unit coordinates."""
    if cmap_obs is None:
        return None
    m = cmap_obs
    ps = ps_curve(m)
    # Physical (monomer) insulation windows -> bins for this resolution; key by
    # the honest monomer scale so a coarsened map never claims a sub-bin window.
    windows = insulation_windows_for_resolution(resolution)
    # Insulation: cooltools-style log2(diamond / median) score (masked-bin aware),
    # computed once per window. Boundary strength = log2 prominence (higher =
    # stronger); aggregate dip_depth = flank - center in log2 (higher = stronger).
    insul_scores = [(wphys, insulation_score_cooltools(m, wb)) for wb, wphys in windows]
    agg_half = insulation_aggregate_half(resolution)
    out: Dict[str, Any] = {
        "shape": list(m.shape),
        "resolution": int(resolution),
        "tad_strength": tad_strength(m, tads),
        "corner_dot_intensities": corner_dot_intensities(m, tads),
        # O/E corner dot: distance-normalized CTCF-CTCF loop enrichment (proper
        # corner-score; the raw-obs corner dot is distance-confounded).
        "corner_dot_intensities_oe": (
            corner_dot_intensities(cmap_oe, tads) if cmap_oe is not None else None),
        "ps_at": {str(s): float(ps[s]) for s in (5, 10, 20, 50, 100, 150, 200, 300, 500) if s < len(ps)},
        "insulation_boundary_strength": {
            str(wphys): insulation_boundary_prominence(score, boundaries)
            for wphys, score in insul_scores
        },
        "insulation_aggregate": {
            str(wphys): aggregate_insulation_log2(score, boundaries, half=agg_half)
            for wphys, score in insul_scores
        },
        "stripe_enrichment_per_boundary": {
            str(b): stripe_enrichment(m, b) for b in boundaries
        },
    }
    if gene_tss:
        out["fountain_obs"] = _fountain_stats(
            m, gene_tss, gene_tes, gene_enhancers, is_oe=False)
    avg, snips, kept = rescaled_tad_pileup(m, tads)
    if avg is not None:
        out["tad_pileup_obs"] = tad_strength_from_pileup(avg)
        out["tad_pileup_obs_crossing_stripes"] = _rescaled_crossing_stripes(avg, "obs")
        out["tad_pileup_per_tad"] = {
            str(idx): tad_strength_from_pileup(snips[i]) for i, idx in enumerate(kept)
        }
    if cmap_oe is not None:
        avg_oe, _, _ = rescaled_tad_pileup(cmap_oe, tads)
        if avg_oe is not None:
            out["tad_pileup_oe"] = tad_strength_from_pileup(avg_oe)
            out["tad_pileup_oe_crossing_stripes"] = _rescaled_crossing_stripes(avg_oe, "oe")
        if gene_tss:
            out["fountain_oe"] = _fountain_stats(
                cmap_oe, gene_tss, gene_tes, gene_enhancers, is_oe=True)
    return out


def _collect_3d(r: RunData) -> Optional[Dict[str, Any]]:
    return _collect_3d_maps(
        r.cmap_obs, r.cmap_oe, r.tads, r.boundaries,
        r.gene_tss, r.gene_tes, r.gene_enhancers)


# ---------------------------------------------------------------------------
# Plots (all percentile-clipped to suppress outliers)
# ---------------------------------------------------------------------------

def _pclip(arr: np.ndarray, p: Tuple[float, float] = (1.0, 99.0)) -> Tuple[float, float]:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0, 1.0
    return float(np.percentile(a, p[0])), float(np.percentile(a, p[1]))


def _pabs(arr: np.ndarray, p: float = 99.0) -> float:
    a = np.abs(arr[np.isfinite(arr)])
    return float(np.percentile(a, p)) if a.size else 1.0


def _mask_central_diags(mat: np.ndarray, k: int = 1) -> np.ndarray:
    """Return a float copy of ``mat`` with the main and ``±k`` diagonals set to NaN."""
    out = np.array(mat, dtype=float)
    r, c = np.indices(out.shape)
    out[np.abs(r - c) <= k] = np.nan
    return out


def _plot_loop_length(a: Dict, b: Dict, la: str, lb: str, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ea = a["loop_length"]["histogram_edges_kb"]
    fa = a["loop_length"]["histogram_fraction"]
    fb = b["loop_length"]["histogram_fraction"]
    ca = a["loop_length"].get("histogram_counts")
    cb = b["loop_length"].get("histogram_counts")
    labels = [f"{ea[i]}-{ea[i + 1]}" for i in range(len(fa))]
    x = np.arange(len(fa)); w = 0.4
    # Plot probability DENSITY (fraction per kb): bins are unequal width, so raw
    # per-bin fraction makes wide bins look tall and breaks the monotonic decay.
    # Dividing by bin width gives a bin-width-independent comparison.
    widths = [ea[i + 1] - ea[i] for i in range(len(fa))]
    da = [f / wd if wd > 0 else 0.0 for f, wd in zip(fa, widths)]
    db = [f / wd if wd > 0 else 0.0 for f, wd in zip(fb, widths)]
    panels = [("fraction per kb (density)", da, db)]
    if ca is not None and cb is not None:
        panels.append(("counts", ca, cb))
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 4), squeeze=False)
    for ax, (ylabel, va, vb) in zip(axes[0], panels):
        ax.bar(x - w / 2, va, w, label=la); ax.bar(x + w / 2, vb, w, label=lb)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30)
        ax.set_ylabel(ylabel); ax.set_xlabel("loop bin (kb)")
        ax.legend()
    fig.suptitle(f"Loop length distribution: {la} vs {lb}")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_ps(a: np.ndarray, b: np.ndarray, la: str, lb: str, out: Path,
             title: str = "P(s)", auc_normalize: bool = False,
             s_max: Optional[int] = None) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(len(a), len(b))
    if s_max is not None:
        # Drop the noisy large-s tail before doing anything else, so it neither
        # shows on the plot nor skews the AUC used for normalization.
        n = min(n, int(s_max) + 1)
    a, b = a[:n].astype(float), b[:n].astype(float)
    s = np.arange(1, n)
    ylabel = "P(s)"
    if auc_normalize:
        # Normalize each curve to unit area under the curve (over the retained
        # s-range) so the comparison reflects shape rather than overall scale.
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        def _auc_norm(y: np.ndarray) -> np.ndarray:
            area = _trapz(y[1:], s)
            return y / area if area > 0 else y
        a, b = _auc_norm(a), _auc_norm(b)
        ylabel = "P(s) (AUC-normalized)"
        title = f"{title} (AUC-normalized)"
    if s_max is not None:
        title = f"{title}, s≤{int(s_max)}"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].loglog(s, a[1:], label=la); ax[0].loglog(s, b[1:], label=lb)
    ax[0].set_xlabel("s (kb)"); ax[0].set_ylabel(ylabel); ax[0].set_title(title); ax[0].legend()
    fold = b[1:] / np.where(a[1:] > 0, a[1:], np.nan)
    ax[1].semilogx(s, fold); ax[1].axhline(1, color="k", ls="--")
    ax[1].set_xlabel("s (kb)"); ax[1].set_ylabel(f"{lb}/{la}")
    ax[1].set_title("P(s) fold")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_insulation(ma: np.ndarray, mb: np.ndarray, boundaries: List[int],
                     la: str, lb: str, out: Path, factor: int = 1) -> None:
    """A-vs-B cooltools-style log2 insulation score per window (lower = insulated)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    windows = insulation_windows_for_resolution(factor)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axflat = axes.reshape(-1)
    for k, (wb, wphys) in enumerate(windows):
        ax = axflat[k]
        sa = insulation_score_cooltools(ma, wb)
        sb = insulation_score_cooltools(mb, wb)
        ax.plot(np.arange(len(sa)) * factor, sa, label=la)
        ax.plot(np.arange(len(sb)) * factor, sb, label=lb)
        ax.axhline(0, color="grey", ls="-", alpha=0.3, lw=0.8)
        for b in boundaries: ax.axvline(b * factor, color="k", ls=":", alpha=0.4)
        ax.set_title(f"window={wphys} kb"); ax.set_ylabel("log2 insulation")
        if k == 0: ax.legend(fontsize=8)
    for k in range(len(windows), len(axflat)):
        axflat[k].axis("off")
    axes[-1, -1].set_xlabel("position (kb)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_insulation_aggregate(ma: np.ndarray, mb: np.ndarray, boundaries: List[int],
                               la: str, lb: str, out: Path, factor: int = 1) -> None:
    """A-vs-B boundary-centered average of the cooltools log2 insulation score."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    windows = insulation_windows_for_resolution(factor)
    half = insulation_aggregate_half(factor)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axflat = axes.reshape(-1)
    for k, (wb, wphys) in enumerate(windows):
        ax = axflat[k]
        aa = aggregate_insulation_log2(insulation_score_cooltools(ma, wb), boundaries, half=half)
        ab = aggregate_insulation_log2(insulation_score_cooltools(mb, wb), boundaries, half=half)
        if aa["mean_profile"]:
            ax.plot(np.asarray(aa["offsets"]) * factor, aa["mean_profile"], label=la)
        if ab["mean_profile"]:
            ax.plot(np.asarray(ab["offsets"]) * factor, ab["mean_profile"], label=lb)
        ax.axvline(0, color="k", ls=":", alpha=0.4)
        ax.axhline(0, color="grey", ls="-", alpha=0.3, lw=0.8)
        ax.set_title(f"window={wphys} kb (dip {la}={aa.get('dip_depth', float('nan')):.2f}, "
                     f"{lb}={ab.get('dip_depth', float('nan')):.2f})")
        ax.set_ylabel("mean log2 insulation")
        if k == 0: ax.legend(fontsize=8)
    for k in range(len(windows), len(axflat)):
        axflat[k].axis("off")
    axes[-1, -1].set_xlabel("offset from boundary (kb)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _log2_fold(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    vals = np.concatenate([
        np.asarray(num)[np.isfinite(num) & (num > 0)],
        np.asarray(den)[np.isfinite(den) & (den > 0)],
    ])
    eps = max(float(np.percentile(vals, 1)) * 1e-3, 1e-12) if vals.size else 1e-12
    return np.log2(np.clip(num, eps, None) / np.clip(den, eps, None))


def _plot_contact_maps(ma: np.ndarray, mb: np.ndarray, ea: np.ndarray, eb: np.ndarray,
                       boundaries: List[int], la: str, lb: str, out: Path,
                       ann: Optional[dict] = None) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    if ann is None:
        ann = {"boundaries": list(boundaries), "gene_bodies": [],
               "promoters": [], "enhancers": [], "ep_pairs": []}
    _marked = {"n": 0}

    def mark_boundaries(ax):
        annotate.draw(ax, ann, legend=(_marked["n"] == 0))
        _marked["n"] += 1

    obs_a = np.log1p(ma); obs_b = np.log1p(mb)
    lo, hi = _pclip(np.concatenate([obs_a.ravel(), obs_b.ravel()]))
    for ax, mat, title in (
        (axes[0, 0], obs_a, f"OBS {la} (log1p)"),
        (axes[0, 1], obs_b, f"OBS {lb} (log1p)"),
    ):
        im = ax.imshow(mat, cmap="inferno", vmin=lo, vmax=hi, origin="lower")
        mark_boundaries(ax); ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.045)
    obs_fold = _log2_fold(mb, ma)
    vm = _pabs(obs_fold)
    im = axes[0, 2].imshow(obs_fold, cmap="bwr", vmin=-vm, vmax=vm, origin="lower")
    mark_boundaries(axes[0, 2]); axes[0, 2].set_title(f"OBS log2({lb}/{la})")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.045)

    oe_a = np.log2(np.clip(ea, 1e-3, None))
    oe_b = np.log2(np.clip(eb, 1e-3, None))
    vm = _pabs(np.concatenate([oe_a.ravel(), oe_b.ravel()]))
    for ax, mat, title in (
        (axes[1, 0], oe_a, f"O/E {la} (log2)"),
        (axes[1, 1], oe_b, f"O/E {lb} (log2)"),
    ):
        im = ax.imshow(mat, cmap="bwr", vmin=-vm, vmax=vm, origin="lower")
        mark_boundaries(ax); ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.045)
    oe_fold = _log2_fold(eb, ea)
    vm = _pabs(oe_fold)
    im = axes[1, 2].imshow(oe_fold, cmap="bwr", vmin=-vm, vmax=vm, origin="lower")
    mark_boundaries(axes[1, 2]); axes[1, 2].set_title(f"O/E log2({lb}/{la})")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.045)

    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_tad_pileup_compare(ma: np.ndarray, mb: np.ndarray, ea: np.ndarray, eb: np.ndarray,
                             tads: List[Tuple[int, int]], la: str, lb: str, out: Path) -> Dict[str, Any]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ap_o, _, _ = rescaled_tad_pileup(ma, tads)
    aq_o, _, _ = rescaled_tad_pileup(mb, tads)
    ap_e, _, _ = rescaled_tad_pileup(ea, tads)
    aq_e, _, _ = rescaled_tad_pileup(eb, tads)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    t = TAD_PILEUP_RESOLUTION // 3  # central-third side; thirds at [t, 2t]
    def boxes(ax):
        ax.add_patch(Rectangle((t, t), t, t, fill=False, ec="lime", lw=1.5))
        ax.add_patch(Rectangle((t, 0), t, t, fill=False, ec="cyan", lw=1.0, ls="--"))
        ax.add_patch(Rectangle((2 * t, t), t, t, fill=False, ec="cyan", lw=1.0, ls="--"))
    # OBS row — mask the main and ±1 diagonals (self-contacts dominate the scale)
    obs_a_disp = _mask_central_diags(np.log1p(ap_o))
    obs_b_disp = _mask_central_diags(np.log1p(aq_o))
    obs_all = np.concatenate([obs_a_disp.ravel(), obs_b_disp.ravel()])
    lo, hi = _pclip(obs_all)
    for col, (mat, label) in enumerate([(obs_a_disp, la), (obs_b_disp, lb)]):
        ax = axes[0, col]
        im = ax.imshow(mat, cmap="inferno", vmin=lo, vmax=hi, origin="lower")
        plt.colorbar(im, ax=ax, fraction=.045); boxes(ax)
        ax.set_title(f"OBS log1p rescaled-TAD: {label}")
    do = _mask_central_diags(aq_o - ap_o); vm = _pabs(do)
    ax = axes[0, 2]; im = ax.imshow(do, cmap="bwr", vmin=-vm, vmax=vm, origin="lower")
    plt.colorbar(im, ax=ax, fraction=.045); boxes(ax)
    ax.set_title(f"OBS  {lb} − {la}")
    # O/E row — mask the main and ±1 diagonals
    oe_a_disp = _mask_central_diags(np.log2(np.clip(ap_e, 1e-3, None)))
    oe_b_disp = _mask_central_diags(np.log2(np.clip(aq_e, 1e-3, None)))
    oe_log = np.concatenate([oe_a_disp.ravel(), oe_b_disp.ravel()])
    ve = _pabs(oe_log)
    for col, (mat, label) in enumerate([(oe_a_disp, la), (oe_b_disp, lb)]):
        ax = axes[1, col]
        im = ax.imshow(mat, cmap="bwr", vmin=-ve, vmax=ve, origin="lower")
        plt.colorbar(im, ax=ax, fraction=.045); boxes(ax)
        ax.set_title(f"O/E log2 rescaled-TAD: {label}")
    de = _mask_central_diags(aq_e - ap_e); vm = _pabs(de)
    ax = axes[1, 2]; im = ax.imshow(de, cmap="bwr", vmin=-vm, vmax=vm, origin="lower")
    plt.colorbar(im, ax=ax, fraction=.045); boxes(ax)
    ax.set_title(f"O/E  {lb} − {la}")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    # quantify
    out_q: Dict[str, Any] = {}
    if ap_o is not None: out_q[f"{la}_obs"] = tad_strength_from_pileup(ap_o)
    if aq_o is not None: out_q[f"{lb}_obs"] = tad_strength_from_pileup(aq_o)
    if ap_e is not None: out_q[f"{la}_oe"] = tad_strength_from_pileup(ap_e)
    if aq_e is not None: out_q[f"{lb}_oe"] = tad_strength_from_pileup(aq_e)
    return out_q


def _plot_anchor_pileups(ma: np.ndarray, mb: np.ndarray, anchor_groups: Dict[str, List[int]],
                         la: str, lb: str, out: Path, half: int = 40,
                         use_log2: bool = False) -> Dict[str, Any]:
    """One row per anchor group, cols = A | B | B-A. Color scales percentile-clipped."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [(k, v) for k, v in anchor_groups.items() if v]
    if not groups:
        return {}
    fig, axes = plt.subplots(len(groups), 3, figsize=(12, 4 * len(groups)), squeeze=False)
    summary: Dict[str, Any] = {}
    for r, (name, centers) in enumerate(groups):
        snip_a = pileup(ma, centers, half=half)
        snip_b = pileup(mb, centers, half=half)
        if snip_a is None or snip_b is None:
            for c in range(3):
                axes[r, c].set_title(f"{name}: no valid centers"); axes[r, c].axis("off")
            continue
        if use_log2:
            va = np.log2(np.clip(snip_a, 1e-3, None)); vb = np.log2(np.clip(snip_b, 1e-3, None))
            both = np.concatenate([va.ravel(), vb.ravel()]); vmax = _pabs(both)
            cmap = "bwr"; vmin = -vmax
        else:
            va = np.log1p(snip_a); vb = np.log1p(snip_b)
            both = np.concatenate([va.ravel(), vb.ravel()]); vmin, vmax = _pclip(both)
            cmap = "inferno"
        ext = [-half, half, -half, half]
        for c, (mat, label) in enumerate([(va, la), (vb, lb)]):
            ax = axes[r, c]
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", extent=ext)
            ax.axhline(0, color="cyan", ls=":", alpha=0.5, lw=0.5)
            ax.axvline(0, color="cyan", ls=":", alpha=0.5, lw=0.5)
            ax.set_title(f"{name}: {label} (n={len(centers)})")
            plt.colorbar(im, ax=ax, fraction=.045)
        d = snip_b - snip_a; vm = _pabs(d)
        ax = axes[r, 2]
        im = ax.imshow(d, cmap="bwr", vmin=-vm, vmax=vm, origin="lower", extent=ext)
        ax.axhline(0, color="k", ls=":", alpha=0.4, lw=0.5)
        ax.axvline(0, color="k", ls=":", alpha=0.4, lw=0.5)
        ax.set_title(f"{name}: {lb} − {la}")
        plt.colorbar(im, ax=ax, fraction=.045)
        summary[name] = {
            f"{la}_center": float(snip_a[half, half]),
            f"{lb}_center": float(snip_b[half, half]),
            f"{la}_mean": float(snip_a.mean()),
            f"{lb}_mean": float(snip_b.mean()),
            "diff_mean": float((snip_b - snip_a).mean()),
        }
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _anchor_groups_from(boundaries: List[int], gene_tss: List[int],
                        gene_tes: List[int],
                        lesion_sites: Optional[List[int]]) -> Dict[str, List[int]]:
    g: Dict[str, List[int]] = {"CTCF boundary": list(boundaries)}
    if gene_tss or gene_tes:
        g["gene TSS"] = list(gene_tss)
        g["gene TES"] = list(gene_tes)
    if lesion_sites:
        g["lesions"] = list(lesion_sites)
    return g


def _anchor_groups(r: RunData) -> Dict[str, List[int]]:
    sites = None
    if r.lesion_enabled and r.lesions is not None:
        ls = [int(s) for s in np.unique(r.lesions[r.lesions >= 0])]
        sites = ls or None
    return _anchor_groups_from(r.boundaries, r.gene_tss, r.gene_tes, sites)


def _coarsen_run(r: RunData, factor: int):
    """Coarsen a run's raw map to ``factor``-monomer bins, ICE + O/E, scale coords.

    Returns ``(cmap_obs, cmap_oe, scaled_coords)`` where ``scaled_coords`` is the
    :func:`scale_resolution_coords` dict plus a ``gene_enhancers`` entry, all in
    the coarsened map's bin units.
    """
    rc = coarsen_contact_map(r.cmap_raw, factor)
    obs = np.nan_to_num(
        iterative_correction(rc, ignore_diagonals=2, max_iter=200, tol=1e-5))
    oe = np.nan_to_num(observed_over_expected(obs), nan=1.0)
    n_bins = obs.shape[0]
    les_sites = None
    if r.lesion_enabled and r.lesions is not None:
        ls = [int(s) for s in np.unique(r.lesions[r.lesions >= 0])]
        les_sites = ls or None
    sc = scale_resolution_coords(r.boundaries, r.gene_tss, r.gene_tes,
                                 les_sites, factor, n_bins)
    sc["gene_enhancers"] = [[e // factor for e in lst] for lst in r.gene_enhancers]
    return obs, oe, sc


def _plot_3d_pair(a_obs, b_obs, a_oe, b_oe, *, boundaries, tads,
                  gene_tss, gene_tes, gene_enhancers, anchor_groups,
                  label_a, label_b, plots: Path, prefix: str,
                  factor: int = 1) -> Dict[str, Any]:
    """Render the full 3D comparison plot set for one resolution; return the
    quantified pile-up summaries keyed for the metrics dict."""
    _plot_contact_maps(a_obs, b_obs, a_oe, b_oe, boundaries, label_a, label_b,
                       plots / f"{prefix}contact_map_compare.png",
                       ann=annotate.from_lists(boundaries, gene_tss, gene_tes,
                                               gene_enhancers, origin=0,
                                               span=a_obs.shape[0]))
    _plot_ps(ps_curve(a_obs), ps_curve(b_obs), label_a, label_b,
             plots / f"{prefix}Ps_3d_compare.png", title="P(s) — 3D contact map",
             auc_normalize=True,
             s_max=int(min(a_obs.shape[0], b_obs.shape[0]) * PS_3D_S_MAX_FRAC))
    _plot_insulation(a_obs, b_obs, boundaries, label_a, label_b,
                     plots / f"{prefix}insulation_compare.png", factor=factor)
    _plot_insulation_aggregate(a_obs, b_obs, boundaries, label_a, label_b,
                               plots / f"{prefix}insulation_aggregate_compare.png", factor=factor)
    out: Dict[str, Any] = {}
    out["tad_pileup_compare"] = _plot_tad_pileup_compare(
        a_obs, b_obs, a_oe, b_oe, tads, label_a, label_b,
        plots / f"{prefix}tad_pileup_compare.png")
    out["anchor_pileups_obs"] = _plot_anchor_pileups(
        a_obs, b_obs, anchor_groups, label_a, label_b,
        plots / f"{prefix}anchor_pileups_compare_obs.png", use_log2=False)
    out["anchor_pileups_oe"] = _plot_anchor_pileups(
        a_oe, b_oe, anchor_groups, label_a, label_b,
        plots / f"{prefix}anchor_pileups_compare_oe.png", use_log2=True)
    return out


def _compute_3d_folds(a3: Dict[str, Any], b3: Dict[str, Any]) -> Dict[str, Any]:
    """B/A folds (and deltas) for the 3D metrics of one resolution."""
    folds: Dict[str, Any] = {}
    folds["corner_dot_intensities_per_tad"] = [
        _fold(p, q) for p, q in zip(
            a3["corner_dot_intensities"], b3["corner_dot_intensities"])
    ]
    if a3.get("corner_dot_intensities_oe") and b3.get("corner_dot_intensities_oe"):
        folds["corner_dot_intensities_oe_per_tad"] = [
            _fold(p, q) for p, q in zip(
                a3["corner_dot_intensities_oe"], b3["corner_dot_intensities_oe"])
        ]
    if "stripe_enrichment_per_boundary" in a3 and "stripe_enrichment_per_boundary" in b3:
        folds["stripe_enrichment_per_boundary"] = {}
        for boundary in sorted(set(a3["stripe_enrichment_per_boundary"]) &
                               set(b3["stripe_enrichment_per_boundary"]), key=int):
            a_stripe = a3["stripe_enrichment_per_boundary"][boundary]
            b_stripe = b3["stripe_enrichment_per_boundary"][boundary]
            folds["stripe_enrichment_per_boundary"][boundary] = {
                "left_enrichment_x": _fold(a_stripe.get("left_enrichment_x"),
                                           b_stripe.get("left_enrichment_x")),
                "right_enrichment_x": _fold(a_stripe.get("right_enrichment_x"),
                                            b_stripe.get("right_enrichment_x")),
            }
    # Insulation is cooltools-style log2: compare boundary strength (prominence)
    # and aggregate dip_depth with a delta (B - A; positive = stronger in B), not
    # a ratio, since both are already in log2 units.
    if "insulation_boundary_strength" in a3 and "insulation_boundary_strength" in b3:
        folds["insulation_boundary_strength_delta"] = {
            w: (b3["insulation_boundary_strength"][w].get("mean", float("nan"))
                - a3["insulation_boundary_strength"][w].get("mean", float("nan")))
            for w in a3["insulation_boundary_strength"]
            if w in b3["insulation_boundary_strength"]
        }
    if "insulation_aggregate" in a3 and "insulation_aggregate" in b3:
        folds["insulation_aggregate_dip_delta"] = {
            w: (b3["insulation_aggregate"][w].get("dip_depth", float("nan"))
                - a3["insulation_aggregate"][w].get("dip_depth", float("nan")))
            for w in a3["insulation_aggregate"] if w in b3["insulation_aggregate"]
        }
    if "tad_pileup_obs" in a3 and "tad_pileup_obs" in b3:
        folds["tad_strength_obs"] = _fold(a3["tad_pileup_obs"]["strength"],
                                          b3["tad_pileup_obs"]["strength"])
    if "tad_pileup_obs_crossing_stripes" in a3 and "tad_pileup_obs_crossing_stripes" in b3:
        folds["tad_pileup_obs_crossing_stripe_contrast_delta"] = (
            b3["tad_pileup_obs_crossing_stripes"]["mean_contrast"] -
            a3["tad_pileup_obs_crossing_stripes"]["mean_contrast"]
        )
    if "tad_pileup_oe" in a3 and "tad_pileup_oe" in b3:
        folds["tad_strength_oe"] = _fold(a3["tad_pileup_oe"]["strength"],
                                         b3["tad_pileup_oe"]["strength"])
    if "tad_pileup_oe_crossing_stripes" in a3 and "tad_pileup_oe_crossing_stripes" in b3:
        folds["tad_pileup_oe_crossing_stripe_contrast_delta"] = (
            b3["tad_pileup_oe_crossing_stripes"]["mean_contrast"] -
            a3["tad_pileup_oe_crossing_stripes"]["mean_contrast"]
        )
    if "fountain_obs" in a3 and "fountain_obs" in b3:
        folds["fountain_obs_score"] = _fold(
            a3["fountain_obs"]["fountain_score"], b3["fountain_obs"]["fountain_score"])
        folds["fountain_obs_symmetry"] = _fold(
            a3["fountain_obs"]["symmetry"], b3["fountain_obs"]["symmetry"])
        folds["fountain_obs_enhancer_tss_mean"] = _fold(
            a3["fountain_obs"].get("enhancer_tss_mean"),
            b3["fountain_obs"].get("enhancer_tss_mean"))
    if "fountain_oe" in a3 and "fountain_oe" in b3:
        folds["fountain_oe_score"] = _fold(
            a3["fountain_oe"]["fountain_score"], b3["fountain_oe"]["fountain_score"])
        folds["fountain_oe_symmetry"] = _fold(
            a3["fountain_oe"]["symmetry"], b3["fountain_oe"]["symmetry"])
        folds["fountain_oe_enhancer_tss_mean"] = _fold(
            a3["fountain_oe"].get("enhancer_tss_mean"),
            b3["fountain_oe"].get("enhancer_tss_mean"))
    return folds


def _fold(a: float, b: float) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        a, b = float(a), float(b)
    except Exception:
        return None
    return b / a if a not in (0.0, 0) else None


def _run_pair(cfg_a, cfg_b, out_dir: Path, label_a: str, label_b: str,
              report_name: str = "compare.md", json_name: str = "compare.json",
              plot_prefix: str = "", cutoff: Optional[float] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    a = _load(cfg_a, label_a, cutoff=cutoff)
    b = _load(cfg_b, label_b, cutoff=cutoff)
    if a.boundaries != b.boundaries:
        print(f"WARNING: boundaries differ between {label_a} and {label_b}; comparison uses {label_a}'s.")

    metrics: Dict[str, Any] = {
        "labels": {"A": label_a, "B": label_b},
        "cutoff": cutoff,
        label_a: {"1d": _collect_1d(a)},
        label_b: {"1d": _collect_1d(b)},
    }
    md3a = _collect_3d(a); md3b = _collect_3d(b)
    if md3a is not None: metrics[label_a]["3d"] = md3a
    if md3b is not None: metrics[label_b]["3d"] = md3b

    # ---- 1D plots
    _plot_loop_length(metrics[label_a]["1d"], metrics[label_b]["1d"],
                      label_a, label_b, plots / f"{plot_prefix}loop_length_compare.png")
    _plot_ps(ps_curve_1d(a.positions, a.chain_length, a.num_chains),
             ps_curve_1d(b.positions, b.chain_length, b.num_chains),
             label_a, label_b, plots / f"{plot_prefix}Ps_1d_compare.png",
             title="P(s) — 1D bridge contacts")

    # ---- 3D plots (need both) -- native per-monomer ICE'd map (resolution 1)
    if a.cmap_obs is not None and b.cmap_obs is not None:
        pl = _plot_3d_pair(
            a.cmap_obs, b.cmap_obs, a.cmap_oe, b.cmap_oe,
            boundaries=a.boundaries, tads=a.tads,
            gene_tss=a.gene_tss, gene_tes=a.gene_tes, gene_enhancers=a.gene_enhancers,
            anchor_groups=_anchor_groups(a),  # use A's anchor set
            label_a=label_a, label_b=label_b, plots=plots, prefix=plot_prefix,
            factor=1)
        metrics.update(pl)

    # ---- folds for top-line numbers
    folds: Dict[str, Any] = {}
    a1 = metrics[label_a]["1d"]; b1 = metrics[label_b]["1d"]
    folds["loop_mean"] = _fold(a1["loop_length"]["mean"], b1["loop_length"]["mean"])
    folds["cohesin_at_ctcf_anchor_sum"] = _fold(a1.get("cohesin_at_ctcf_anchor_sum"),
                                                  b1.get("cohesin_at_ctcf_anchor_sum"))
    folds["cohesin_at_gene_bodies_sum"] = _fold(a1.get("cohesin_at_gene_bodies_sum"),
                                                  b1.get("cohesin_at_gene_bodies_sum"))
    folds["corner_pct"] = _fold(a1["classification"]["corner_pct"], b1["classification"]["corner_pct"])
    folds["stripe_pct"] = _fold(a1["classification"]["stripe_pct"], b1["classification"]["stripe_pct"])
    folds["boundary_crossing_mean"] = _fold(a1["boundary_crossing"]["mean"],
                                              b1["boundary_crossing"]["mean"])
    folds["boundary_crossing_stripe_share"] = _fold(
        a1["boundary_crossing_stripes"]["mean"]["stripe_share_of_crossing_frames"],
        b1["boundary_crossing_stripes"]["mean"]["stripe_share_of_crossing_frames"],
    )
    if "3d" in metrics[label_a] and "3d" in metrics[label_b]:
        folds.update(_compute_3d_folds(metrics[label_a]["3d"], metrics[label_b]["3d"]))
    metrics["folds"] = folds

    # ---- coarsened-resolution analyses: bin the raw map N x N, ICE, then run the
    # full 3D comparison again. Independent of ``cutoff``; native (res 1) keeps
    # the historical keys/paths, each extra resolution is suffixed ``_resN``.
    resolutions = [f for f in getattr(cfg_a.contacts, "resolution_list", [1]) if f != 1]
    if resolutions and a.cmap_raw is not None and b.cmap_raw is not None:
        for factor in resolutions:
            # Skip resolutions too coarse for either map (<3 bins) rather than
            # crashing the comparison.
            if min(a.cmap_raw.shape[0], b.cmap_raw.shape[0]) // factor < 3:
                print(f"[compare] skipping analysis_resolution {factor}: contact map "
                      f"coarsens to <3 bins")
                continue
            suffix = f"_res{factor}"
            rprefix = f"{plot_prefix}res{factor}_"
            a_obs, a_oe, a_sc = _coarsen_run(a, factor)
            b_obs, b_oe, b_sc = _coarsen_run(b, factor)
            md3a = _collect_3d_maps(a_obs, a_oe, a_sc["tads"], a_sc["boundaries"],
                                    a_sc["gene_tss"], a_sc["gene_tes"],
                                    a_sc["gene_enhancers"], resolution=factor)
            md3b = _collect_3d_maps(b_obs, b_oe, b_sc["tads"], b_sc["boundaries"],
                                    b_sc["gene_tss"], b_sc["gene_tes"],
                                    b_sc["gene_enhancers"], resolution=factor)
            if md3a is not None:
                metrics[label_a][f"3d{suffix}"] = md3a
            if md3b is not None:
                metrics[label_b][f"3d{suffix}"] = md3b
            pl = _plot_3d_pair(
                a_obs, b_obs, a_oe, b_oe,
                boundaries=a_sc["boundaries"], tads=a_sc["tads"],
                gene_tss=a_sc["gene_tss"], gene_tes=a_sc["gene_tes"],
                gene_enhancers=a_sc["gene_enhancers"],
                anchor_groups=_anchor_groups_from(
                    a_sc["boundaries"], a_sc["gene_tss"], a_sc["gene_tes"],
                    a_sc["lesion_sites"]),
                label_a=label_a, label_b=label_b, plots=plots, prefix=rprefix,
                factor=factor)
            for k, v in pl.items():
                metrics[f"{k}{suffix}"] = v
            if md3a is not None and md3b is not None:
                metrics[f"folds{suffix}"] = _compute_3d_folds(md3a, md3b)

    (out_dir / json_name).write_text(json.dumps(metrics, indent=2, default=str))
    _write_report(metrics, out_dir / report_name, label_a, label_b, plot_prefix=plot_prefix)
    return metrics


def _cutoff_dirname(cutoff: float) -> str:
    c = float(cutoff)
    return f"cutoff_{int(c)}" if c.is_integer() else f"cutoff_{c}"


def _tad_strength_row(cutoff: float, metrics: Dict[str, Any],
                      la: str, lb: str) -> Dict[str, Any]:
    """Pull Flyamer obs + O/E TAD strength (A, B, fold) out of a pair's metrics."""
    a3 = (metrics.get(la, {}) or {}).get("3d", {}) or {}
    b3 = (metrics.get(lb, {}) or {}).get("3d", {}) or {}
    folds = metrics.get("folds", {})
    return {
        "cutoff": float(cutoff),
        "obs_a": a3.get("tad_pileup_obs", {}).get("strength"),
        "obs_b": b3.get("tad_pileup_obs", {}).get("strength"),
        "obs_fold": folds.get("tad_strength_obs"),
        "oe_a": a3.get("tad_pileup_oe", {}).get("strength"),
        "oe_b": b3.get("tad_pileup_oe", {}).get("strength"),
        "oe_fold": folds.get("tad_strength_oe"),
    }


def _write_tad_strength_vs_cutoff(rows: List[Dict[str, Any]], path: Path,
                                  la: str, lb: str) -> None:
    """Consolidated Flyamer TAD-strength (obs + O/E) across contact cutoffs."""
    lines = [f"# Flyamer TAD strength vs contact cutoff: {la} vs {lb}\n"]
    lines.append(
        "Use the O/E columns for cross-config TAD-strength interpretation; "
        "OBS is the raw within/between ratio and can move with P(s) or map-scale "
        "contact redistribution.\n"
    )
    lines.append(
        f"| cutoff | OBS {la} | OBS {lb} | OBS {lb}/{la} | "
        f"O/E {la} | O/E {lb} | O/E {lb}/{la} |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['cutoff']:g} | {_fmt_summary(r['obs_a'])} | {_fmt_summary(r['obs_b'])} | "
            f"{_fmt_summary(r['obs_fold'])} | {_fmt_summary(r['oe_a'])} | "
            f"{_fmt_summary(r['oe_b'])} | {_fmt_summary(r['oe_fold'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run(cfg_a, cfg_b, out_dir: Path,
        label_a: str = "A", label_b: str = "B",
        cutoffs: Optional[List[float]] = None) -> Path:
    out_dir = Path(out_dir)
    if cutoffs:
        rows: List[Dict[str, Any]] = []
        for c in cutoffs:
            metrics = _run_pair(cfg_a, cfg_b, out_dir / _cutoff_dirname(c),
                                label_a, label_b, cutoff=c)
            rows.append(_tad_strength_row(c, metrics, label_a, label_b))
        (out_dir / "tad_strength_vs_cutoff.json").write_text(
            json.dumps(rows, indent=2, default=str))
        _write_tad_strength_vs_cutoff(rows, out_dir / "tad_strength_vs_cutoff.md",
                                      label_a, label_b)
        return out_dir
    _run_pair(cfg_a, cfg_b, out_dir, label_a, label_b)
    return out_dir


def _slug(label: str) -> str:
    out = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(label))
    return out.strip("._") or "run"


def _fmt_summary(x: Any, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "n/a"


def run_many(cfg_baseline, cfg_others: List[Any], out_dir: Path,
             baseline_label: str = "baseline",
             comparison_labels: Optional[List[str]] = None,
             cutoffs: Optional[List[float]] = None,
             cutoff: Optional[float] = None) -> Path:
    """Compare many runs against one baseline.

    Writes flat files in ``out_dir``:
    * ``compare_<label>.json/md`` -- pairwise details per comparison
    * ``compare_many.json`` -- paths + top-line fold summary
    * ``compare_many.md``   -- compact table linking pairwise reports

    With ``cutoffs`` set, the whole comparison is repeated per contact-distance
    cutoff (maps resampled from each run's trajectory) into ``cutoff_<c>/``
    subfolders.
    """
    out_dir = Path(out_dir)
    if cutoffs:
        per_cutoff: List[Tuple[float, Dict[str, Any]]] = []
        for c in cutoffs:
            summ = _run_many_once(cfg_baseline, cfg_others, out_dir / _cutoff_dirname(c),
                                  baseline_label, comparison_labels, cutoff=c)
            per_cutoff.append((float(c), summ))
        agg = {
            "baseline": baseline_label,
            "cutoffs": [
                {"cutoff": c,
                 "comparisons": {lbl: info["top_line"]
                                 for lbl, info in s["comparisons"].items()}}
                for c, s in per_cutoff
            ],
        }
        (out_dir / "tad_strength_vs_cutoff.json").write_text(
            json.dumps(agg, indent=2, default=str))
        _write_many_tad_strength_vs_cutoff(
            per_cutoff, baseline_label, out_dir / "tad_strength_vs_cutoff.md")
        return out_dir
    _run_many_once(cfg_baseline, cfg_others, out_dir, baseline_label,
                   comparison_labels)
    return out_dir


def _run_many_once(cfg_baseline, cfg_others: List[Any], out_dir: Path,
                   baseline_label: str, comparison_labels: Optional[List[str]],
                   cutoff: Optional[float] = None) -> Dict[str, Any]:
    """One baseline-vs-many pass at a single cutoff. Writes compare_many.* and
    returns the summary dict."""
    if not cfg_others:
        raise ValueError("run_many requires at least one comparison config")
    if comparison_labels is None:
        comparison_labels = [f"run{i + 1}" for i in range(len(cfg_others))]
    if len(comparison_labels) != len(cfg_others):
        raise ValueError("comparison_labels length must match cfg_others")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "baseline": baseline_label,
        "comparisons": {},
    }
    used_slugs: set = set()
    for cfg, label in zip(cfg_others, comparison_labels):
        comparison_slug = _slug(label)
        suffix = 2
        while comparison_slug in used_slugs:
            comparison_slug = f"{_slug(label)}_{suffix}"
            suffix += 1
        used_slugs.add(comparison_slug)

        json_name = f"compare_{comparison_slug}.json"
        report_name = f"compare_{comparison_slug}.md"
        plot_prefix = f"{comparison_slug}_"
        metrics = _run_pair(
            cfg_baseline, cfg, out_dir, baseline_label, label,
            report_name=report_name, json_name=json_name, plot_prefix=plot_prefix,
            cutoff=cutoff,
        )
        folds = metrics.get("folds", {})
        baseline_3d = metrics.get(baseline_label, {}).get("3d", {})
        comparison_3d = metrics.get(label, {}).get("3d", {})
        summary["comparisons"][label] = {
            "json": str(out_dir / json_name),
            "report": str(out_dir / report_name),
            "plot_prefix": plot_prefix,
            "folds": folds,
            "top_line": {
                "loop_mean": folds.get("loop_mean"),
                "cohesin_at_ctcf_anchor_sum": folds.get("cohesin_at_ctcf_anchor_sum"),
                "cohesin_at_gene_bodies_sum": folds.get("cohesin_at_gene_bodies_sum"),
                "corner_pct": folds.get("corner_pct"),
                "global_anchored_stripe_pct": folds.get("stripe_pct"),
                "boundary_crossing_mean": folds.get("boundary_crossing_mean"),
                "boundary_crossing_stripe_share": folds.get("boundary_crossing_stripe_share"),
                "tad_strength_obs_baseline": (
                    baseline_3d.get("tad_pileup_obs", {}).get("strength")
                ),
                "tad_strength_obs_comparison": (
                    comparison_3d.get("tad_pileup_obs", {}).get("strength")
                ),
                "tad_strength_obs": folds.get("tad_strength_obs"),
                "tad_strength_oe_baseline": (
                    baseline_3d.get("tad_pileup_oe", {}).get("strength")
                ),
                "tad_strength_oe_comparison": (
                    comparison_3d.get("tad_pileup_oe", {}).get("strength")
                ),
                "tad_strength_oe": folds.get("tad_strength_oe"),
                "tad_pileup_oe_crossing_stripe_contrast_delta": (
                    folds.get("tad_pileup_oe_crossing_stripe_contrast_delta")
                ),
                "fountain_oe_score": folds.get("fountain_oe_score"),
                "fountain_oe_symmetry": folds.get("fountain_oe_symmetry"),
            },
        }

    (out_dir / "compare_many.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_many_report(summary, out_dir / "compare_many.md")
    return summary


def _write_many_tad_strength_vs_cutoff(
    per_cutoff: List[Tuple[float, Dict[str, Any]]], baseline_label: str,
    path: Path,
) -> None:
    """Flyamer obs + O/E TAD strength vs contact cutoff, one section per comparison."""
    lines = [f"# Flyamer TAD strength vs contact cutoff (baseline {baseline_label})\n"]
    lines.append(
        "Use O/E as the primary cross-config TAD-strength metric. OBS is retained "
        "as the raw within/between ratio and is sensitive to distance-dependent "
        "contact decay and global map redistribution.\n"
    )
    labels: List[str] = []
    for _c, summ in per_cutoff:
        for lbl in summ["comparisons"]:
            if lbl not in labels:
                labels.append(lbl)
    for lbl in labels:
        lines.append(f"\n## {baseline_label} vs {lbl}")
        lines.append(
            f"| cutoff | OBS {baseline_label} | OBS {lbl} | OBS {lbl}/{baseline_label} | "
            f"O/E {baseline_label} | O/E {lbl} | O/E {lbl}/{baseline_label} |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for c, summ in per_cutoff:
            top = summ["comparisons"].get(lbl, {}).get("top_line", {})
            lines.append(
                f"| {c:g} | "
                f"{_fmt_summary(top.get('tad_strength_obs_baseline'))} | "
                f"{_fmt_summary(top.get('tad_strength_obs_comparison'))} | "
                f"{_fmt_summary(top.get('tad_strength_obs'))} | "
                f"{_fmt_summary(top.get('tad_strength_oe_baseline'))} | "
                f"{_fmt_summary(top.get('tad_strength_oe_comparison'))} | "
                f"{_fmt_summary(top.get('tad_strength_oe'))} |"
            )
    path.write_text("\n".join(lines) + "\n")


def _write_many_report(summary: Dict[str, Any], path: Path) -> None:
    baseline = summary["baseline"]
    lines = [f"# Multi Comparison: baseline {baseline}\n"]
    lines.append("| comparison | report | loop | CTCF | genes | corner | global stripe | boundary cross | crossing stripe | TAD O/E fold | O/E crossing stripe delta | fountain O/E | fountain symmetry |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, info in summary["comparisons"].items():
        top = info["top_line"]
        report = Path(info["report"])
        rel = report.relative_to(path.parent) if report.is_relative_to(path.parent) else report
        lines.append(
            f"| {label} | [{label}]({rel.as_posix()}) | "
            f"{_fmt_summary(top.get('loop_mean'))} | "
            f"{_fmt_summary(top.get('cohesin_at_ctcf_anchor_sum'))} | "
            f"{_fmt_summary(top.get('cohesin_at_gene_bodies_sum'))} | "
            f"{_fmt_summary(top.get('corner_pct'))} | "
            f"{_fmt_summary(top.get('global_anchored_stripe_pct'))} | "
            f"{_fmt_summary(top.get('boundary_crossing_mean'))} | "
            f"{_fmt_summary(top.get('boundary_crossing_stripe_share'))} | "
            f"{_fmt_summary(top.get('tad_strength_oe'))} | "
            f"{_fmt_summary(top.get('tad_pileup_oe_crossing_stripe_contrast_delta'), 3)} | "
            f"{_fmt_summary(top.get('fountain_oe_score'))} | "
            f"{_fmt_summary(top.get('fountain_oe_symmetry'))} |"
        )
    lines.append("\n## TAD Strength")
    lines.append("| comparison | OBS baseline | OBS comparison | OBS fold | O/E baseline | O/E comparison | O/E fold |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, info in summary["comparisons"].items():
        top = info["top_line"]
        lines.append(
            f"| {label} | "
            f"{_fmt_summary(top.get('tad_strength_obs_baseline'))} | "
            f"{_fmt_summary(top.get('tad_strength_obs_comparison'))} | "
            f"{_fmt_summary(top.get('tad_strength_obs'))} | "
            f"{_fmt_summary(top.get('tad_strength_oe_baseline'))} | "
            f"{_fmt_summary(top.get('tad_strength_oe_comparison'))} | "
            f"{_fmt_summary(top.get('tad_strength_oe'))} |"
        )
    path.write_text("\n".join(lines))


def _write_report(m: Dict[str, Any], path: Path, la: str, lb: str,
                  plot_prefix: str = "") -> None:
    lines: List[str] = [f"# Comparison: {la} vs {lb}\n"]
    a1, b1 = m[la]["1d"], m[lb]["1d"]
    lines.append("## 1D top-line")
    lines.append("| metric | {la} | {lb} | {lb}/{la} |".format(la=la, lb=lb))
    lines.append("|---|---|---|---|")
    f = m.get("folds", {})
    def row(k, va, vb, fold):
        return f"| {k} | {va} | {vb} | {fold} |"
    lines.append(row("mean loop", f"{a1['loop_length']['mean']:.1f}",
                     f"{b1['loop_length']['mean']:.1f}", f"{f.get('loop_mean'):.2f}"))
    lines.append(row("cohesin@CTCF", f"{a1.get('cohesin_at_ctcf_anchor_sum', 0):.2f}",
                     f"{b1.get('cohesin_at_ctcf_anchor_sum', 0):.2f}",
                     f"{f.get('cohesin_at_ctcf_anchor_sum') or float('nan'):.2f}"))
    if 'cohesin_at_gene_bodies_sum' in a1 or 'cohesin_at_gene_bodies_sum' in b1:
        va = a1.get('cohesin_at_gene_bodies_sum'); vb = b1.get('cohesin_at_gene_bodies_sum')
        fmt = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"
        fold = f.get('cohesin_at_gene_bodies_sum')
        lines.append(row("cohesin@genes", fmt(va), fmt(vb), fmt(fold)))
    lines.append(row("corner%", f"{a1['classification']['corner_pct']:.1f}",
                     f"{b1['classification']['corner_pct']:.1f}",
                     f"{f.get('corner_pct') or float('nan'):.2f}"))
    lines.append(row("global anchored-stripe%", f"{a1['classification']['stripe_pct']:.1f}",
                     f"{b1['classification']['stripe_pct']:.1f}",
                     f"{f.get('stripe_pct') or float('nan'):.2f}"))
    lines.append(row("boundary cross", f"{a1['boundary_crossing']['mean']:.3f}",
                     f"{b1['boundary_crossing']['mean']:.3f}",
                     f"{f.get('boundary_crossing_mean') or float('nan'):.2f}"))
    a_cross_stripe = 100 * a1["boundary_crossing_stripes"]["mean"]["stripe_share_of_crossing_frames"]
    b_cross_stripe = 100 * b1["boundary_crossing_stripes"]["mean"]["stripe_share_of_crossing_frames"]
    lines.append(row("boundary-crossing stripe share", f"{a_cross_stripe:.1f}%",
                     f"{b_cross_stripe:.1f}%",
                     f"{f.get('boundary_crossing_stripe_share') or float('nan'):.2f}"))
    if "ps_1d" in a1 and "ps_1d" in b1:
        lines.append("\n## P(s) — 1D bridge contacts")
        lines.append(f"| s (kb) | {la} | {lb} | {lb}/{la} |")
        lines.append("|---|---|---|---|")
        pa, pb = a1["ps_1d"]["ps_at"], b1["ps_1d"]["ps_at"]
        for s in sorted(set(pa) & set(pb), key=int):
            va, vb = pa[s], pb[s]
            fold = vb / va if va else float("nan")
            lines.append(f"| {s} | {va:.2e} | {vb:.2e} | {fold:.2f} |")
    if "3d" in m[la] and "3d" in m[lb]:
        a3 = m[la]["3d"]; b3 = m[lb]["3d"]
        lines.append("\n## 3D")
        if "ps_at" in a3 and "ps_at" in b3:
            lines.append("\n### P(s) — 3D contact map")
            lines.append(f"| s (kb) | {la} | {lb} | {lb}/{la} |")
            lines.append("|---|---|---|---|")
            pa, pb = a3["ps_at"], b3["ps_at"]
            for s in sorted(set(pa) & set(pb), key=int):
                va, vb = pa[s], pb[s]
                fold = vb / va if va else float("nan")
                lines.append(f"| {s} | {va:.2e} | {vb:.2e} | {fold:.2f} |")
        if "tad_pileup_obs" in a3:
            lines.append(row("TAD strength, Flyamer (raw OBS)",
                             f"{a3['tad_pileup_obs']['strength']:.2f}",
                             f"{b3['tad_pileup_obs']['strength']:.2f}",
                             f"{f.get('tad_strength_obs') or float('nan'):.2f}"))
        if "tad_pileup_oe" in a3:
            lines.append(row("TAD strength, Flyamer (O/E, primary)",
                             f"{a3['tad_pileup_oe']['strength']:.2f}",
                             f"{b3['tad_pileup_oe']['strength']:.2f}",
                             f"{f.get('tad_strength_oe') or float('nan'):.2f}"))
            lines.append(
                "\nFor cross-config TAD strength, interpret the O/E row first; "
                "the raw OBS row is useful for diagnosing global contact shifts."
            )
        if "tad_pileup_obs_crossing_stripes" in a3 or "tad_pileup_oe_crossing_stripes" in a3:
            lines.append("\n### 3D Rescaled-TAD Crossing Stripes")
            lines.append(f"| map | {la} contrast | {lb} contrast | {lb}-{la} |")
            lines.append("|---|---|---|---|")
            stripe_rows = [
                (
                    "OBS log1p",
                    "tad_pileup_obs_crossing_stripes",
                    "tad_pileup_obs_crossing_stripe_contrast_delta",
                ),
                (
                    "O/E log2",
                    "tad_pileup_oe_crossing_stripes",
                    "tad_pileup_oe_crossing_stripe_contrast_delta",
                ),
            ]
            for label, key, delta_key in stripe_rows:
                if key in a3 and key in b3:
                    lines.append(row(
                        label,
                        f"{a3[key]['mean_contrast']:.3f}",
                        f"{b3[key]['mean_contrast']:.3f}",
                        f"{f.get(delta_key, float('nan')):.3f}",
                    ))
        if "fountain_obs" in a3 or "fountain_oe" in a3:
            lines.append("\n### Gene Fountain Stats")
            lines.append(f"| map | metric | {la} | {lb} | {lb}/{la} |")
            lines.append("|---|---|---|---|---|")
            fountain_rows = [
                ("OBS", "fountain_obs", "fountain_score", "score", "fountain_obs_score"),
                ("OBS", "fountain_obs", "symmetry", "symmetry", "fountain_obs_symmetry"),
                ("OBS", "fountain_obs", "enhancer_tss_mean", "enhancer-TSS", "fountain_obs_enhancer_tss_mean"),
                ("O/E", "fountain_oe", "fountain_score", "score", "fountain_oe_score"),
                ("O/E", "fountain_oe", "symmetry", "symmetry", "fountain_oe_symmetry"),
                ("O/E", "fountain_oe", "enhancer_tss_mean", "enhancer-TSS", "fountain_oe_enhancer_tss_mean"),
            ]
            for map_label, key, metric_key, metric_label, fold_key in fountain_rows:
                if key in a3 and key in b3:
                    lines.append(
                        f"| {map_label} | {metric_label} | "
                        f"{a3[key].get(metric_key, 0.0):.3f} | "
                        f"{b3[key].get(metric_key, 0.0):.3f} | "
                        f"{_fmt_summary(f.get(fold_key), 2)} |"
                    )
        lines.append("\n### Corner-dot intensities per TAD (fold)")
        for i, fold in enumerate(f.get("corner_dot_intensities_per_tad", [])):
            lines.append(f"- TAD{i}: {fold:.2f}" if fold is not None else f"- TAD{i}: n/a")
        if "stripe_enrichment_per_boundary" in a3 and "stripe_enrichment_per_boundary" in b3:
            lines.append("\n### 3D Contact-Map Stripe Enrichment Per Boundary")
            lines.append(
                f"| boundary | {la} left | {lb} left | left fold | "
                f"{la} right | {lb} right | right fold |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            stripe_folds = f.get("stripe_enrichment_per_boundary", {})
            for boundary in sorted(set(a3["stripe_enrichment_per_boundary"]) &
                                   set(b3["stripe_enrichment_per_boundary"]), key=int):
                a_stripe = a3["stripe_enrichment_per_boundary"][boundary]
                b_stripe = b3["stripe_enrichment_per_boundary"][boundary]
                folds_for_boundary = stripe_folds.get(boundary, {})
                left_fold = folds_for_boundary.get("left_enrichment_x")
                right_fold = folds_for_boundary.get("right_enrichment_x")
                fmt_fold = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"
                lines.append(
                    f"| {boundary} | "
                    f"{a_stripe['left_enrichment_x']:.2f} | "
                    f"{b_stripe['left_enrichment_x']:.2f} | "
                    f"{fmt_fold(left_fold)} | "
                    f"{a_stripe['right_enrichment_x']:.2f} | "
                    f"{b_stripe['right_enrichment_x']:.2f} | "
                    f"{fmt_fold(right_fold)} |"
                )
    lines.append("\n## Plots")
    for fn in ("loop_length_compare.png", "Ps_1d_compare.png", "Ps_3d_compare.png",
               "insulation_compare.png", "insulation_aggregate_compare.png",
               "contact_map_compare.png", "tad_pileup_compare.png",
               "anchor_pileups_compare_obs.png", "anchor_pileups_compare_oe.png"):
        prefixed = f"{plot_prefix}{fn}"
        if (path.parent / "plots" / prefixed).exists():
            lines.append(f"- [{fn}](plots/{prefixed})")

    # Coarsened-resolution 3D comparisons (raw map binned N x N, then ICE'd).
    res_keys = sorted(k for k in m.get(la, {}) if k.startswith("3d_res"))
    for key in res_keys:
        factor = key[len("3d_res"):]
        suffix = f"_res{factor}"
        a3 = m[la].get(key); b3 = m[lb].get(key)
        fr = m.get(f"folds{suffix}", {})
        lines.append(f"\n## 3D @ {factor}-monomer resolution (coarsened then ICE-balanced)")
        if a3 and b3:
            lines.append(f"| metric | {la} | {lb} | {lb}/{la} |")
            lines.append("|---|---|---|---|")
            if "tad_pileup_obs" in a3 and "tad_pileup_obs" in b3:
                lines.append(
                    f"| TAD strength, Flyamer (raw OBS) | {a3['tad_pileup_obs']['strength']:.2f} | "
                    f"{b3['tad_pileup_obs']['strength']:.2f} | "
                    f"{_fmt_summary(fr.get('tad_strength_obs'))} |")
            if "tad_pileup_oe" in a3 and "tad_pileup_oe" in b3:
                lines.append(
                    f"| TAD strength, Flyamer (O/E, primary) | {a3['tad_pileup_oe']['strength']:.2f} | "
                    f"{b3['tad_pileup_oe']['strength']:.2f} | "
                    f"{_fmt_summary(fr.get('tad_strength_oe'))} |")
            if "fountain_oe" in a3 and "fountain_oe" in b3:
                lines.append(
                    f"| fountain O/E score | {a3['fountain_oe']['fountain_score']:.3f} | "
                    f"{b3['fountain_oe']['fountain_score']:.3f} | "
                    f"{_fmt_summary(fr.get('fountain_oe_score'))} |")
            ca = a3.get("insulation_boundary_strength", {})
            cb = b3.get("insulation_boundary_strength", {})
            shared = [w for w in ca if w in cb]
            if shared:
                w = max(shared, key=int)  # largest (most robust) window
                cdelta = fr.get("insulation_boundary_strength_delta", {}).get(w)
                lines.append(
                    f"| insulation strength, cooltools (w={w}, B−A) | "
                    f"{ca[w].get('mean', float('nan')):.3f} | "
                    f"{cb[w].get('mean', float('nan')):.3f} | "
                    f"{_fmt_summary(cdelta)} |")
        for fn in ("Ps_3d_compare", "insulation_compare", "insulation_aggregate_compare",
                   "contact_map_compare", "tad_pileup_compare",
                   "anchor_pileups_compare_obs", "anchor_pileups_compare_oe"):
            prefixed = f"{plot_prefix}res{factor}_{fn}.png"
            if (path.parent / "plots" / prefixed).exists():
                lines.append(f"- [{fn}{suffix}.png](plots/{prefixed})")

    path.write_text("\n".join(lines))
