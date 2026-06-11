"""Shared genome-feature overlays for contact-map heatmaps.

Single source of truth for drawing CTCF boundaries, gene bodies, promoters
(TSS), enhancers and enhancer-promoter links on a contact-map ``Axes``. Used by
the ``contacts`` viz plugin (``default_oe_heatmap``), ``qc`` and ``compare`` so
every heatmap carries the same annotation legend.

Coordinates are chain-relative monomer indices (1 site = 1 kb), matching the
single-chain window the maps are sampled over. Drawing uses blended
data/axes-fraction transforms so it works for both ``origin="upper"`` and
``origin="lower"`` imshow and needs no extra axes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# marker / colour conventions, shared across every heatmap
_C_BOUNDARY = "cyan"
_C_GENE = "orange"
_C_TSS = "lime"
_C_ENH = "magenta"
_C_EP = "yellow"


def _window(ann: Dict[str, Any], origin: int, span: int) -> Dict[str, Any]:
    """Shift to window-relative coords and drop features outside [0, span)."""
    lo, hi = int(origin), int(origin) + int(span)

    def _ok(p: int) -> bool:
        return lo <= int(p) < hi

    return {
        "boundaries": [int(b) - lo for b in ann["boundaries"] if _ok(b)],
        "gene_bodies": [(int(s) - lo, int(e) - lo) for s, e in ann["gene_bodies"]
                        if _ok(s) or _ok(e)],
        "promoters": [int(p) - lo for p in ann["promoters"] if _ok(p)],
        "enhancers": [int(e) - lo for e in ann["enhancers"] if _ok(e)],
        "ep_pairs": [(int(e) - lo, int(p) - lo) for e, p in ann["ep_pairs"]
                     if _ok(e) and _ok(p)],
    }


def from_lists(
    boundaries: Sequence[int],
    gene_tss: Sequence[int],
    gene_tes: Sequence[int],
    gene_enhancers: Sequence[Sequence[int]],
    *,
    origin: int = 0,
    span: int | None = None,
) -> Dict[str, Any]:
    """Build an annotation dict from already-extracted per-gene lists."""
    bodies = [(int(s), int(e)) for s, e in zip(gene_tss, gene_tes)]
    promoters = [int(s) for s in gene_tss]
    enhancers: List[int] = []
    ep_pairs: List[Tuple[int, int]] = []
    for i, enh in enumerate(gene_enhancers):
        tss = int(gene_tss[i])
        for e in enh:
            enhancers.append(int(e))
            ep_pairs.append((int(e), tss))
    ann = {"boundaries": [int(b) for b in boundaries], "gene_bodies": bodies,
           "promoters": promoters, "enhancers": enhancers, "ep_pairs": ep_pairs}
    if span is None:
        return ann
    return _window(ann, origin, span)


def boundaries_from_topology_kwargs(tk: Any, chain_length: int) -> List[int]:
    """Chain-relative interior TAD boundaries from either topology schema.

    Supports the legacy ``tad_positions`` list and the per-TAD ``tads`` schema
    (each record's ``left`` and ``right + 1``, reconstructing both edges of any
    inter-TAD gap so the chain stays fully tiled). Returns sorted, unique
    positions strictly inside ``(0, chain_length)`` -- the single source of truth
    for every package consumer that draws TAD boundaries / intervals.
    """
    if not isinstance(tk, dict):
        return []
    tads = tk.get("tads")
    if tads:
        inner = set()
        for rec in tads:
            left = int(rec["left"])
            right = int(rec["right"])
            if left > 0:
                inner.add(left)
            if right + 1 < chain_length:
                inner.add(right + 1)
    else:
        inner = {int(p) for p in (tk.get("tad_positions") or [])}
    return sorted(p for p in inner if 0 < p < chain_length)


def from_lef_cfg(lef_cfg, *, origin: int = 0, span: int | None = None) -> Dict[str, Any]:
    """Build an annotation dict from a ``LEFConfig`` topology.

    Reads boundaries (legacy ``tad_positions`` or the per-TAD ``tads`` schema via
    :func:`boundaries_from_topology_kwargs`) and ``genes`` (tss/tes + ``enhancers``
    or ``enhancer_pos``) from ``topology_kwargs``. ``span`` defaults to the chain
    length so a single-chain window is annotated.
    """
    tk = lef_cfg.topology_kwargs if isinstance(lef_cfg.topology_kwargs, dict) else {}
    tss: List[int] = []
    tes: List[int] = []
    enh: List[List[int]] = []
    for g in (tk.get("genes") or []):
        tss.append(int(g["tss"]))
        tes.append(int(g["tes"]))
        e = g.get("enhancers")
        if e is None:
            ep = g.get("enhancer_pos")
            e = [] if ep is None else [ep]
        enh.append([int(x) for x in e])
    chain_length = int(getattr(lef_cfg, "chain_length", 0) or 0)
    span = chain_length if span is None else span
    boundaries = boundaries_from_topology_kwargs(tk, chain_length)
    return from_lists(boundaries, tss, tes, enh,
                      origin=origin, span=span or None)


def is_empty(ann: Dict[str, Any] | None) -> bool:
    return not ann or not any(ann.get(k) for k in
                              ("boundaries", "gene_bodies", "promoters", "enhancers"))


def draw(ax, ann: Dict[str, Any] | None, *, legend: bool = False) -> None:
    """Overlay boundaries + gene/promoter/enhancer/E-P features on ``ax``.

    Boundaries are dashed grid lines; gene bodies, TSS and enhancers sit in thin
    margin tracks just inside the bottom and left edges; E-P pairs are off-diagonal
    square markers. Safe to call with ``None`` / empty annotations (no-op).
    """
    if is_empty(ann):
        return
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D

    # margin-track transforms: x in data / y in axes-fraction (bottom) and mirror.
    bottom = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    left = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    for b in ann["boundaries"]:
        ax.axhline(b, color=_C_BOUNDARY, ls="--", alpha=0.5, lw=0.7)
        ax.axvline(b, color=_C_BOUNDARY, ls="--", alpha=0.5, lw=0.7)

    for s, e in ann["gene_bodies"]:
        ax.plot([s, e], [0.018, 0.018], transform=bottom, color=_C_GENE,
                lw=3, solid_capstyle="butt", alpha=0.9, clip_on=False)
        ax.plot([0.018, 0.018], [s, e], transform=left, color=_C_GENE,
                lw=3, solid_capstyle="butt", alpha=0.9, clip_on=False)

    if ann["promoters"]:
        ax.scatter(ann["promoters"], [0.05] * len(ann["promoters"]), transform=bottom,
                   marker="^", s=20, color=_C_TSS, edgecolors="black", linewidths=0.3,
                   clip_on=False, zorder=5)
        ax.scatter([0.05] * len(ann["promoters"]), ann["promoters"], transform=left,
                   marker="^", s=20, color=_C_TSS, edgecolors="black", linewidths=0.3,
                   clip_on=False, zorder=5)
    if ann["enhancers"]:
        ax.scatter(ann["enhancers"], [0.05] * len(ann["enhancers"]), transform=bottom,
                   marker="o", s=14, color=_C_ENH, edgecolors="black", linewidths=0.3,
                   clip_on=False, zorder=5)
        ax.scatter([0.05] * len(ann["enhancers"]), ann["enhancers"], transform=left,
                   marker="o", s=14, color=_C_ENH, edgecolors="black", linewidths=0.3,
                   clip_on=False, zorder=5)

    for e, p in ann["ep_pairs"]:
        ax.scatter([e, p], [p, e], marker="s", s=22, facecolors="none",
                   edgecolors=_C_EP, linewidths=0.8, zorder=6)

    if legend:
        handles = [
            Line2D([0], [0], color=_C_BOUNDARY, ls="--", lw=1, label="CTCF boundary"),
            Line2D([0], [0], color=_C_GENE, lw=3, label="gene body"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=_C_TSS,
                   markeredgecolor="black", markersize=7, ls="", label="TSS"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_C_ENH,
                   markeredgecolor="black", markersize=6, ls="", label="enhancer"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                   markeredgecolor=_C_EP, markersize=7, ls="", label="E-P pair"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.6)
