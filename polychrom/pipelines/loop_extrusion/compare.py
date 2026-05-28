"""Pairwise comparison of two pipeline outputs (1D + 3D).

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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from .qc import (
    anchor_set, asymmetry_index, boundary_crossing, cohesin_at_lesion_flanks,
    cohesin_classification, cohesin_occupancy, corner_dot_intensities,
    insulation_boundary_strength, insulation_profile, lesion_metrics,
    loop_length_stats, observed_over_expected, pileup, ps_curve,
    rescaled_tad_pileup, rnapii_metrics, sanity_1d,
    stripe_enrichment, tad_strength, tad_strength_from_pileup,
)
from .plugins.sampling import iterative_correction


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    label: str
    positions: np.ndarray
    rnapii_positions: Optional[np.ndarray]
    rnapii_states: Optional[np.ndarray]
    rnapii_enabled: bool
    lesions: Optional[np.ndarray]
    lesion_enabled: bool
    genes_ds: Optional[np.ndarray]
    chain_length: int
    num_chains: int
    boundaries: List[int]
    tads: List[Tuple[int, int]]
    gene_bodies: List[Tuple[int, int]]
    cmap_obs: Optional[np.ndarray]
    cmap_oe: Optional[np.ndarray]


def _load(cfg, label: str) -> RunData:
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
        lesion_enabled = bool(fh.attrs.get("lesion_enabled", False))
        les = fh["lesions"][:] if "lesions" in fh else None
        genes_ds = fh["genes"][:] if "genes" in fh else None

    boundaries = list(cfg.lef.topology_kwargs.get("tad_positions", []))
    tads = [(s, e) for s, e in zip([0, *boundaries], [*boundaries, chain_length])]
    gene_bodies: List[Tuple[int, int]] = []
    if genes_ds is not None:
        for g in genes_ds:
            lo, hi = sorted((int(g["tss"]), int(g["tes"])))
            gene_bodies.append((lo, hi))

    cmap_obs = None
    cmap_oe = None
    raw_p = Path(getattr(cfg.contacts, "raw_output_path", ""))
    if raw_p.exists():
        raw = np.load(raw_p).astype(float)
        cmap_obs = np.nan_to_num(
            iterative_correction(raw, ignore_diagonals=2, max_iter=200, tol=1e-5)
        )
        oe_p = Path(getattr(cfg.contacts, "oe_output_path", ""))
        if oe_p.exists():
            cmap_oe = np.nan_to_num(np.load(oe_p).astype(float), nan=1.0)
        else:
            cmap_oe = np.nan_to_num(observed_over_expected(cmap_obs), nan=1.0)

    return RunData(
        label=label, positions=pos, rnapii_positions=rp, rnapii_states=rs,
        rnapii_enabled=rnapii_enabled, lesions=les, lesion_enabled=lesion_enabled,
        genes_ds=genes_ds, chain_length=chain_length, num_chains=num_chains,
        boundaries=boundaries, tads=tads, gene_bodies=gene_bodies,
        cmap_obs=cmap_obs, cmap_oe=cmap_oe,
    )


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def _collect_1d(r: RunData) -> Dict[str, Any]:
    anch = anchor_set(r.boundaries, r.chain_length)
    occ = cohesin_occupancy(r.positions, r.chain_length * r.num_chains)
    out: Dict[str, Any] = {
        "sanity": sanity_1d(r.positions, r.chain_length, r.num_chains),
        "loop_length": loop_length_stats(
            r.positions, edges=[0, 50, 100, 150, 200, 300, 500, r.chain_length]),
        "classification": cohesin_classification(r.positions, anch),
        "boundary_crossing": boundary_crossing(r.positions, r.boundaries),
        "asymmetry_index": asymmetry_index(r.positions),
        "cohesin_at_ctcf_anchor_sum": float(sum(occ[s] for s in anch if s < len(occ))),
    }
    if r.gene_bodies:
        out["cohesin_at_gene_bodies_sum"] = float(sum(occ[a:b + 1].sum() for a, b in r.gene_bodies))
    if r.rnapii_enabled and r.rnapii_positions is not None and r.rnapii_states is not None:
        out["rnapii"] = rnapii_metrics(r.rnapii_positions, r.rnapii_states)
    if r.lesion_enabled and r.lesions is not None:
        out["lesions"] = lesion_metrics(r.lesions, r.gene_bodies)
        out["lesions"]["cohesin_flank_enrichment"] = cohesin_at_lesion_flanks(
            r.positions, r.lesions, r.chain_length * r.num_chains)
    return out


def _collect_3d(r: RunData) -> Optional[Dict[str, Any]]:
    if r.cmap_obs is None:
        return None
    m = r.cmap_obs
    ps = ps_curve(m)
    windows = [5, 10, 20, 40, 80, 120]
    out: Dict[str, Any] = {
        "shape": list(m.shape),
        "tad_strength": tad_strength(m, r.tads),
        "corner_dot_intensities": corner_dot_intensities(m, r.tads),
        "ps_at": {str(s): float(ps[s]) for s in (5, 10, 20, 50, 100, 150, 200, 300, 500) if s < len(ps)},
        "insulation_boundary_strength": {
            str(w): insulation_boundary_strength(insulation_profile(m, w), r.boundaries, w)
            for w in windows
        },
        "stripe_enrichment_per_boundary": {
            str(b): stripe_enrichment(m, b) for b in r.boundaries
        },
    }
    avg, snips, kept = rescaled_tad_pileup(m, r.tads, target=90)
    if avg is not None:
        out["tad_pileup_obs"] = tad_strength_from_pileup(avg)
        out["tad_pileup_per_tad"] = {
            str(idx): tad_strength_from_pileup(snips[i]) for i, idx in enumerate(kept)
        }
    if r.cmap_oe is not None:
        avg_oe, _, _ = rescaled_tad_pileup(r.cmap_oe, r.tads, target=90)
        if avg_oe is not None:
            out["tad_pileup_oe"] = tad_strength_from_pileup(avg_oe)
    return out


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


def _plot_loop_length(a: Dict, b: Dict, la: str, lb: str, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ea = a["loop_length"]["histogram_edges_kb"]
    fa = a["loop_length"]["histogram_fraction"]
    fb = b["loop_length"]["histogram_fraction"]
    labels = [f"{ea[i]}-{ea[i + 1]}" for i in range(len(fa))]
    x = np.arange(len(fa)); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, fa, w, label=la); ax.bar(x + w / 2, fb, w, label=lb)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("fraction"); ax.set_xlabel("loop bin (kb)")
    ax.set_title(f"Loop length distribution: {la} vs {lb}")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_ps(a: np.ndarray, b: np.ndarray, la: str, lb: str, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = np.arange(1, len(a))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].loglog(s, a[1:], label=la); ax[0].loglog(s, b[1:], label=lb)
    ax[0].set_xlabel("s (kb)"); ax[0].set_ylabel("P(s)"); ax[0].set_title("P(s)"); ax[0].legend()
    fold = b[1:] / np.where(a[1:] > 0, a[1:], np.nan)
    ax[1].semilogx(s, fold); ax[1].axhline(1, color="k", ls="--")
    ax[1].set_xlabel("s (kb)"); ax[1].set_ylabel(f"{lb}/{la}")
    ax[1].set_title("P(s) fold")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_insulation(ma: np.ndarray, mb: np.ndarray, boundaries: List[int],
                     la: str, lb: str, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    windows = [5, 10, 20, 40, 80, 120]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for k, w in enumerate(windows):
        ax = axes[k // 3][k % 3]
        ax.plot(insulation_profile(ma, w), label=la)
        ax.plot(insulation_profile(mb, w), label=lb)
        for b in boundaries: ax.axvline(b, color="k", ls=":", alpha=0.4)
        ax.set_title(f"window={w} kb"); ax.set_ylabel("insulation (raw)")
        if k == 0: ax.legend(fontsize=8)
    axes[-1, -1].set_xlabel("position (kb)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_contact_maps(ma: np.ndarray, mb: np.ndarray, boundaries: List[int],
                       la: str, lb: str, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, m, label in ((axes[0], ma, la), (axes[1], mb, lb)):
        v = np.log1p(m); lo, hi = _pclip(v)
        im = ax.imshow(v, cmap="inferno", vmin=lo, vmax=hi, origin="lower")
        for b in boundaries:
            ax.axhline(b, color="cyan", ls="--", alpha=0.5, lw=0.7)
            ax.axvline(b, color="cyan", ls="--", alpha=0.5, lw=0.7)
        ax.set_title(f"{label}  contact map (log1p, p1-p99)")
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_tad_pileup_compare(ma: np.ndarray, mb: np.ndarray, ea: np.ndarray, eb: np.ndarray,
                             tads: List[Tuple[int, int]], la: str, lb: str, out: Path) -> Dict[str, Any]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ap_o, _, _ = rescaled_tad_pileup(ma, tads, 90)
    aq_o, _, _ = rescaled_tad_pileup(mb, tads, 90)
    ap_e, _, _ = rescaled_tad_pileup(ea, tads, 90)
    aq_e, _, _ = rescaled_tad_pileup(eb, tads, 90)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    def boxes(ax):
        ax.add_patch(Rectangle((30, 30), 30, 30, fill=False, ec="lime", lw=1.5))
        ax.add_patch(Rectangle((30, 0), 30, 30, fill=False, ec="cyan", lw=1.0, ls="--"))
        ax.add_patch(Rectangle((60, 30), 30, 30, fill=False, ec="cyan", lw=1.0, ls="--"))
    # OBS row
    obs_all = np.concatenate([np.log1p(ap_o).ravel(), np.log1p(aq_o).ravel()])
    lo, hi = _pclip(obs_all)
    for col, (mat, label) in enumerate([(np.log1p(ap_o), la), (np.log1p(aq_o), lb)]):
        ax = axes[0, col]
        im = ax.imshow(mat, cmap="inferno", vmin=lo, vmax=hi, origin="lower")
        plt.colorbar(im, ax=ax, fraction=.045); boxes(ax)
        ax.set_title(f"OBS log1p rescaled-TAD: {label}")
    do = aq_o - ap_o; vm = _pabs(do)
    ax = axes[0, 2]; im = ax.imshow(do, cmap="bwr", vmin=-vm, vmax=vm, origin="lower")
    plt.colorbar(im, ax=ax, fraction=.045); boxes(ax)
    ax.set_title(f"OBS  {lb} − {la}")
    # O/E row
    oe_log = np.concatenate([
        np.log2(np.clip(ap_e, 1e-3, None)).ravel(),
        np.log2(np.clip(aq_e, 1e-3, None)).ravel(),
    ])
    ve = _pabs(oe_log)
    for col, (mat, label) in enumerate([(np.log2(np.clip(ap_e, 1e-3, None)), la),
                                         (np.log2(np.clip(aq_e, 1e-3, None)), lb)]):
        ax = axes[1, col]
        im = ax.imshow(mat, cmap="bwr", vmin=-ve, vmax=ve, origin="lower")
        plt.colorbar(im, ax=ax, fraction=.045); boxes(ax)
        ax.set_title(f"O/E log2 rescaled-TAD: {label}")
    de = aq_e - ap_e; vm = _pabs(de)
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

def _anchor_groups(r: RunData) -> Dict[str, List[int]]:
    g: Dict[str, List[int]] = {"CTCF boundary": list(r.boundaries)}
    if r.genes_ds is not None:
        g["gene TSS"] = [int(x["tss"]) for x in r.genes_ds]
        g["gene TES"] = [int(x["tes"]) for x in r.genes_ds]
    if r.lesion_enabled and r.lesions is not None:
        sites = [int(s) for s in np.unique(r.lesions[r.lesions >= 0])]
        if sites:
            g["lesions"] = sites
    return g


def _fold(a: float, b: float) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        a, b = float(a), float(b)
    except Exception:
        return None
    return b / a if a not in (0.0, 0) else None


def run(cfg_a, cfg_b, out_dir: Path,
        label_a: str = "A", label_b: str = "B") -> Path:
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    a = _load(cfg_a, label_a)
    b = _load(cfg_b, label_b)
    if a.boundaries != b.boundaries:
        print(f"WARNING: boundaries differ between {label_a} and {label_b}; comparison uses {label_a}'s.")

    metrics: Dict[str, Any] = {
        "labels": {"A": label_a, "B": label_b},
        label_a: {"1d": _collect_1d(a)},
        label_b: {"1d": _collect_1d(b)},
    }
    md3a = _collect_3d(a); md3b = _collect_3d(b)
    if md3a is not None: metrics[label_a]["3d"] = md3a
    if md3b is not None: metrics[label_b]["3d"] = md3b

    # ---- 1D plots
    _plot_loop_length(metrics[label_a]["1d"], metrics[label_b]["1d"],
                      label_a, label_b, plots / "loop_length_compare.png")

    # ---- 3D plots (need both)
    if a.cmap_obs is not None and b.cmap_obs is not None:
        _plot_contact_maps(a.cmap_obs, b.cmap_obs, a.boundaries, label_a, label_b,
                           plots / "contact_map_compare.png")
        _plot_ps(ps_curve(a.cmap_obs), ps_curve(b.cmap_obs),
                 label_a, label_b, plots / "Ps_compare.png")
        _plot_insulation(a.cmap_obs, b.cmap_obs, a.boundaries, label_a, label_b,
                         plots / "insulation_compare.png")
        # Flyamer rescaled-TAD pile-up
        tp = _plot_tad_pileup_compare(
            a.cmap_obs, b.cmap_obs, a.cmap_oe, b.cmap_oe, a.tads,
            label_a, label_b, plots / "tad_pileup_compare.png")
        metrics["tad_pileup_compare"] = tp
        # anchor pile-ups (obs + O/E)
        groups = _anchor_groups(a)  # use A's anchor set
        metrics["anchor_pileups_obs"] = _plot_anchor_pileups(
            a.cmap_obs, b.cmap_obs, groups, label_a, label_b,
            plots / "anchor_pileups_compare_obs.png", use_log2=False)
        metrics["anchor_pileups_oe"] = _plot_anchor_pileups(
            a.cmap_oe, b.cmap_oe, groups, label_a, label_b,
            plots / "anchor_pileups_compare_oe.png", use_log2=True)

    # ---- folds for top-line numbers
    folds: Dict[str, Any] = {}
    a1 = metrics[label_a]["1d"]; b1 = metrics[label_b]["1d"]
    folds["loop_mean"] = _fold(a1["loop_length"]["mean"], b1["loop_length"]["mean"])
    folds["cohesin_at_ctcf_anchor_sum"] = _fold(a1.get("cohesin_at_ctcf_anchor_sum"),
                                                  b1.get("cohesin_at_ctcf_anchor_sum"))
    folds["cohesin_at_gene_bodies_sum"] = _fold(a1.get("cohesin_at_gene_bodies_sum"),
                                                  b1.get("cohesin_at_gene_bodies_sum"))
    folds["corner_pct"] = _fold(a1["classification"]["corner_pct"], b1["classification"]["corner_pct"])
    folds["boundary_crossing_mean"] = _fold(a1["boundary_crossing"]["mean"],
                                              b1["boundary_crossing"]["mean"])
    if "3d" in metrics[label_a] and "3d" in metrics[label_b]:
        folds["corner_dot_intensities_per_tad"] = [
            _fold(p, q) for p, q in zip(
                metrics[label_a]["3d"]["corner_dot_intensities"],
                metrics[label_b]["3d"]["corner_dot_intensities"],
            )
        ]
        if "tad_pileup_obs" in metrics[label_a]["3d"]:
            folds["tad_strength_obs"] = _fold(metrics[label_a]["3d"]["tad_pileup_obs"]["strength"],
                                                metrics[label_b]["3d"]["tad_pileup_obs"]["strength"])
        if "tad_pileup_oe" in metrics[label_a]["3d"]:
            folds["tad_strength_oe"] = _fold(metrics[label_a]["3d"]["tad_pileup_oe"]["strength"],
                                               metrics[label_b]["3d"]["tad_pileup_oe"]["strength"])
    metrics["folds"] = folds

    (out_dir / "compare.json").write_text(json.dumps(metrics, indent=2, default=str))
    _write_report(metrics, out_dir / "compare.md", label_a, label_b)
    return out_dir


def _write_report(m: Dict[str, Any], path: Path, la: str, lb: str) -> None:
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
    lines.append(row("boundary cross", f"{a1['boundary_crossing']['mean']:.3f}",
                     f"{b1['boundary_crossing']['mean']:.3f}",
                     f"{f.get('boundary_crossing_mean') or float('nan'):.2f}"))
    if "3d" in m[la] and "3d" in m[lb]:
        a3 = m[la]["3d"]; b3 = m[lb]["3d"]
        lines.append("\n## 3D")
        if "tad_pileup_obs" in a3:
            lines.append(row("Flyamer strength (OBS)",
                             f"{a3['tad_pileup_obs']['strength']:.2f}",
                             f"{b3['tad_pileup_obs']['strength']:.2f}",
                             f"{f.get('tad_strength_obs') or float('nan'):.2f}"))
        if "tad_pileup_oe" in a3:
            lines.append(row("Flyamer strength (O/E)",
                             f"{a3['tad_pileup_oe']['strength']:.2f}",
                             f"{b3['tad_pileup_oe']['strength']:.2f}",
                             f"{f.get('tad_strength_oe') or float('nan'):.2f}"))
        lines.append("\n### Corner-dot intensities per TAD (fold)")
        for i, fold in enumerate(f.get("corner_dot_intensities_per_tad", [])):
            lines.append(f"- TAD{i}: {fold:.2f}" if fold is not None else f"- TAD{i}: n/a")
    lines.append("\n## Plots")
    for fn in ("loop_length_compare.png", "Ps_compare.png", "insulation_compare.png",
               "contact_map_compare.png", "tad_pileup_compare.png",
               "anchor_pileups_compare_obs.png", "anchor_pileups_compare_oe.png"):
        if (path.parent / "plots" / fn).exists():
            lines.append(f"- [{fn}](plots/{fn})")
    path.write_text("\n".join(lines))
