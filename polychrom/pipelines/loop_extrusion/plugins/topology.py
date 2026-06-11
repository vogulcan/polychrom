"""Linear-setup plugins: chain layout and CTCF site placement.

A topology plugin returns the ``args`` dict consumed by the LEF dynamics,
already populated with ``N``, ``LIFETIME``, ``LIFETIME_STALLED``,
``ctcfCapture`` and ``ctcfRelease``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from ..config import LEFConfig
from .rnapii import build_genes
from .lesions import precompute_lesion_fields

# Boundary strength is either a single value applied to every anchor, or a
# per-anchor mapping {position: strength} keyed by the chain-relative site.
BoundaryStrength = Union[float, Mapping[Any, float]]


def _empty_ctcf() -> Dict[int, Dict[int, float]]:
    return {-1: {}, 1: {}}


def _strength_at(
    boundary_strength: BoundaryStrength,
    position: int,
    default: float,
) -> float:
    """Resolve the capture strength for one anchor.

    ``boundary_strength`` may be a scalar (same value for every anchor) or a
    mapping keyed by chain-relative position. Positions absent from the mapping
    fall back to ``default``. YAML may parse int keys as strings, so both forms
    are accepted.
    """
    if isinstance(boundary_strength, Mapping):
        if position in boundary_strength:
            return float(boundary_strength[position])
        if str(position) in boundary_strength:
            return float(boundary_strength[str(position)])
        return float(default)
    return float(boundary_strength)


@dataclass
class TadRecord:
    """One TAD's two oriented anchors (chain-relative).

    ``left``/``right`` are the chain-relative anchor sites (``right`` is the
    inclusive right-anchor site). ``left_side``/``right_side`` are the cohesin
    leg directions each anchor captures: convergent TADs use ``-1`` (left-facing,
    captures the leftward leg) and ``+1`` (right-facing, captures the rightward
    leg). Strengths are the resolved capture probabilities BEFORE chromosome-end
    forcing, which :func:`_apply_convergent_tads` applies.
    """

    left: int
    right: int
    left_strength: float
    right_strength: float
    left_side: int = -1
    right_side: int = 1


def _opt_strength(spec: Mapping[str, Any], key: str, default: float) -> float:
    """Per-side strength from a ``tads`` record, falling back to ``default``."""
    val = spec.get(key)
    return float(default) if val is None else float(val)


def _records_from_tads(
    cfg: LEFConfig,
    tads: Iterable[Mapping[str, Any]],
    default_boundary_strength: float,
) -> List[TadRecord]:
    """Build canonical records from the explicit per-TAD ``tads`` schema.

    Each entry is a mapping with required ``left``/``right`` (chain-relative,
    ``right`` inclusive) and optional ``left_strength``/``right_strength``
    (default ``default_boundary_strength``) and ``left_side``/``right_side``
    (default convergent ``-1``/``+1``). Records are returned sorted by ``left``;
    overlapping TADs are rejected. Gaps (``next.left > rec.right + 1``) are
    allowed and simply leave anchor-free sites between TADs.
    """
    records: List[TadRecord] = []
    for i, spec in enumerate(tads):
        if not isinstance(spec, Mapping):
            raise TypeError(f"tads[{i}] must be a mapping, got {type(spec).__name__}")
        if "left" not in spec or "right" not in spec:
            raise ValueError(f"tads[{i}] must contain 'left' and 'right'")
        left = int(spec["left"])
        right = int(spec["right"])
        if not (0 <= left <= right < cfg.chain_length):
            raise ValueError(
                f"tads[{i}]: require 0 <= left <= right < chain_length "
                f"({cfg.chain_length}); got left={left}, right={right}"
            )
        records.append(
            TadRecord(
                left=left,
                right=right,
                left_strength=_opt_strength(spec, "left_strength", default_boundary_strength),
                right_strength=_opt_strength(spec, "right_strength", default_boundary_strength),
                left_side=int(spec.get("left_side", -1)),
                right_side=int(spec.get("right_side", 1)),
            )
        )
    records.sort(key=lambda r: r.left)
    for prev, nxt in zip(records[:-1], records[1:]):
        if nxt.left <= prev.right:
            raise ValueError(
                f"Overlapping TADs: [{prev.left},{prev.right}] and "
                f"[{nxt.left},{nxt.right}] share sites"
            )
    return records


def _canonical_tads(
    cfg: LEFConfig,
    *,
    tads: Optional[Iterable[Mapping[str, Any]]] = None,
    tad_positions: Optional[Iterable[int]] = None,
    boundary_strength: BoundaryStrength = 0.5,
    default_boundary_strength: float = 0.5,
) -> List[TadRecord]:
    """Canonicalise a TAD layout into a list of :class:`TadRecord`.

    Two mutually exclusive inputs are accepted:

    - **New** ``tads``: an explicit list of per-TAD records (see
      :func:`_records_from_tads`). Each TAD owns independent left/right anchors,
      and inter-TAD gaps are allowed.
    - **Legacy** ``tad_positions``: the chain is tiled by intervals
      ``[0, *tad_positions, chain_length]`` and each interval ``[start, end)``
      becomes a record with ``left=start``, ``right=end-1``. The left anchor's
      strength is keyed by ``start`` and the right by ``end`` (historical
      convention), resolved via :func:`_strength_at`.
    """
    if tads is not None and tad_positions:
        raise ValueError("Provide either 'tads' or 'tad_positions', not both")
    if tads is not None:
        return _records_from_tads(cfg, tads, default_boundary_strength)

    inner = [int(pos) for pos in (tad_positions or [])]
    boundaries = [0, *inner, cfg.chain_length]
    records = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        records.append(
            TadRecord(
                left=start,
                right=end - 1,
                left_strength=_strength_at(boundary_strength, start, default_boundary_strength),
                right_strength=_strength_at(boundary_strength, end, default_boundary_strength),
            )
        )
    return records


def _reconstruct_boundaries(cfg: LEFConfig, records: List[TadRecord]) -> List[int]:
    """Chain-relative interval edges for downstream consumers (lesions, drawing).

    Returns every internal segment edge: each TAD's ``left`` (when > 0) and the
    site just past its ``right`` (when < chain_length). For an abutting (gap-free)
    layout this reproduces the legacy ``tad_positions`` exactly; with gaps it
    includes BOTH edges of every gap so the chain stays fully tiled by
    alternating TAD and gap segments (keeping ``_tad_length_per_site`` correct).
    """
    edges = set()
    for rec in records:
        if rec.left > 0:
            edges.add(rec.left)
        if rec.right + 1 < cfg.chain_length:
            edges.add(rec.right + 1)
    return sorted(edges)


def _base_args(cfg: LEFConfig) -> Dict[str, Any]:
    return {
        "N": cfg.num_sites,
        "chain_length": cfg.chain_length,
        "num_chains": cfg.num_chains,
        "LIFETIME": cfg.lifetime,
        "LIFETIME_STALLED": cfg.lifetime_stalled,
        "LIFETIME_CTCF": int(cfg.lifetime_ctcf) if cfg.lifetime_ctcf is not None else cfg.lifetime,
        "ctcfCapture": _empty_ctcf(),
        "ctcfRelease": _empty_ctcf(),
    }


def uniform_tad_topology(
    cfg: LEFConfig,
    *,
    tad_positions: Iterable[int] = (300, 800, 1500, 2300, 2900, 3400),
    capture_prob: float = 0.9,
    release_prob: float = 0.003,
    symmetric: bool = True,
) -> Dict[str, Any]:
    """Repeat the same TAD CTCF layout across every chain.

    Mirrors the layout used in ``extrusion_1D_newCode.ipynb``: every chain
    of length ``cfg.chain_length`` gets the same set of CTCF positions, each
    acting on both sides (``symmetric=True``).
    """
    args = _base_args(cfg)
    sides = (-1, 1) if symmetric else (1,)
    for chain_idx in range(cfg.num_chains):
        chain_offset = chain_idx * cfg.chain_length
        for pos in tad_positions:
            site = chain_offset + pos
            for side in sides:
                args["ctcfCapture"][side][site] = capture_prob
                args["ctcfRelease"][side][site] = release_prob
    return args


def _apply_convergent_tads(
    args: Dict[str, Any],
    cfg: LEFConfig,
    *,
    tad_records: List[TadRecord],
    release_prob: float,
    include_chromosome_ends: bool,
) -> List[tuple]:
    """Place inward-facing CTCF barriers from canonical per-TAD records.

    Each record contributes a left anchor (captures the ``left_side`` leg) and a
    right anchor (captures the ``right_side`` leg), replicated across every chain.
    The chromosome-start anchor (``left == 0``) and chromosome-end anchor
    (``right == chain_length - 1``) are forced to hard ``1.0`` walls; when
    ``include_chromosome_ends`` is False those two anchors are omitted entirely.
    Sites belonging to no record (inter-TAD gaps) receive no anchor.

    Returns the ground-truth anchor table as a list of tuples
    ``(abs_position, chain_position, chain_index, side, strength, tad_index, edge)``
    -- one row per anchor actually placed, per chain -- for H5 persistence. The
    recorded strength is the value actually written (``1.0`` at forced ends).
    """
    chain_end = cfg.chain_length - 1
    anchors: List[tuple] = []
    for chain_idx in range(cfg.num_chains):
        chain_offset = chain_idx * cfg.chain_length
        for tad_idx, rec in enumerate(tad_records):
            left_site = chain_offset + rec.left
            right_site = chain_offset + rec.right
            if include_chromosome_ends or rec.left != 0:
                strength = 1.0 if rec.left == 0 else rec.left_strength
                args["ctcfCapture"][rec.left_side][left_site] = strength
                args["ctcfRelease"][rec.left_side][left_site] = float(release_prob)
                anchors.append(
                    (left_site, rec.left, chain_idx, rec.left_side, strength, tad_idx, 0)
                )
            if include_chromosome_ends or rec.right != chain_end:
                strength = 1.0 if rec.right == chain_end else rec.right_strength
                args["ctcfCapture"][rec.right_side][right_site] = strength
                args["ctcfRelease"][rec.right_side][right_site] = float(release_prob)
                anchors.append(
                    (right_site, rec.right, chain_idx, rec.right_side, strength, tad_idx, 1)
                )
    return anchors


def _expand_genes_across_chains(
    cfg: LEFConfig,
    genes: Optional[List[dict]],
    *,
    replicate_genes_across_chains: bool,
) -> List[dict]:
    """Expand chain-relative gene specs to absolute lattice coordinates."""
    base = [dict(g) for g in (genes or [])]
    if not replicate_genes_across_chains:
        return base

    expanded: List[dict] = []
    for chain_idx in range(cfg.num_chains):
        offset = chain_idx * cfg.chain_length
        for spec in base:
            out = dict(spec)

            def _shift(site: int, label: str) -> int:
                site = int(site)
                if not (0 <= site < cfg.chain_length):
                    raise ValueError(
                        f"Gene {label}={site} is outside one chain of length "
                        f"{cfg.chain_length}; disable replicate_genes_across_chains "
                        "for absolute coordinates"
                    )
                return site + offset

            for key in ("tss", "tes", "enhancer_pos"):
                if key not in out or out[key] is None:
                    continue
                out[key] = _shift(out[key], key)
            if out.get("enhancers"):
                out["enhancers"] = [_shift(e, "enhancers") for e in out["enhancers"]]
            expanded.append(out)
    return expanded


def _set_loading_sites(
    args: Dict[str, Any],
    gene_objs,
    *,
    targeted_load_prob: float,
    loading_window: int,
    target_enhancers: bool,
    target_tss: bool,
    weight_by_activity: bool = True,
) -> None:
    """Populate ``args`` with targeted cohesin-loading sites (enhancers/TSS).

    Each site carries a loading WEIGHT, exposed as a normalized probability in
    ``args["loading_probs"]``. With ``weight_by_activity`` (default) the weight
    is the owning gene's *productive*-transcription rate
    (``initiation_prob * pause_release_prob``), so cohesin loads preferentially
    at ACTIVE enhancers/promoters -- matching NIPBL/MAU2 recruitment to active
    elements -- rather than uniformly across every enhancer. Poised/silent loci
    (e.g. developmental enhancers) keep their *regulatory* enhancers but
    contribute little *loading*. Without it every site weighs 1.0 (uniform),
    reproducing the legacy behaviour; uniform-kinetics genes (all probs 1.0)
    are unaffected either way.

    Consumed by ``lef_dynamics.load_targeted``. ``targeted_load_prob == 0``
    (default) leaves loading uniform even if this is called.
    """
    weights: Dict[int, float] = {}
    for g in gene_objs:
        w = float(g.initiation_prob * g.pause_release_prob) if weight_by_activity else 1.0
        if target_enhancers:
            for e in g.enhancers:
                weights[int(e)] = weights.get(int(e), 0.0) + w
        if target_tss:
            weights[int(g.tss)] = weights.get(int(g.tss), 0.0) + w
    sites = sorted(weights)
    total = sum(weights[s] for s in sites)
    args["loading_sites"] = sites
    # Normalized sampling distribution (same order as ``loading_sites``); ``None``
    # -> ``load_targeted`` falls back to a uniform pick over the sites.
    args["loading_probs"] = (
        [weights[s] / total for s in sites] if sites and total > 0 else None
    )
    args["targeted_load_prob"] = float(targeted_load_prob)
    args["loading_window"] = int(loading_window)


def convergent_tad_topology(
    cfg: LEFConfig,
    *,
    tad_positions: Iterable[int] = (),
    tads: Optional[Iterable[Mapping[str, Any]]] = None,
    boundary_strength: BoundaryStrength = 0.5,
    release_prob: float = 0.0,
    include_chromosome_ends: bool = True,
    default_boundary_strength: float = 0.5,
) -> Dict[str, Any]:
    """TAD layout with directional, inward-facing CTCF barriers.

    For each TAD interval, the left edge captures the left-moving cohesin leg
    and the right edge captures the right-moving leg. With ``release_prob=0``,
    captured legs remain at CTCF until cohesin unloads, matching the NRMCB
    supplementary-box assumption.

    Two mutually exclusive layout inputs are accepted (see :func:`_canonical_tads`):

    - ``tads``: explicit per-TAD records, each owning independent left/right
      anchors (strength + orientation), with inter-TAD gaps allowed.
    - ``tad_positions`` + ``boundary_strength``: the legacy shared-boundary form;
      ``boundary_strength`` is a scalar or ``{position: strength}`` mapping with
      ``default_boundary_strength`` as fallback.

    ``args["tad_positions"]`` is set to the reconstructed chain-relative interval
    edges (legacy positions for an abutting layout; both gap edges when gaps
    exist) so downstream consumers -- lesion fields, drawing -- stay correct.
    """
    args = _base_args(cfg)
    records = _canonical_tads(
        cfg,
        tads=tads,
        tad_positions=tad_positions,
        boundary_strength=boundary_strength,
        default_boundary_strength=default_boundary_strength,
    )
    args["ctcf_anchors"] = _apply_convergent_tads(
        args,
        cfg,
        tad_records=records,
        release_prob=release_prob,
        include_chromosome_ends=include_chromosome_ends,
    )
    args["tad_positions"] = _reconstruct_boundaries(cfg, records)
    return args


def gene_aware_topology(
    cfg: LEFConfig,
    *,
    tad_positions: Iterable[int] = (),
    capture_prob: float = 0.9,
    release_prob: float = 0.003,
    symmetric: bool = True,
    genes: Optional[List[dict]] = None,
    rnapii_stride: int = 1,
    rnapii_stall_prob: float = 0.4,
    rnapii_push_prob: float = 0.3,
    rnapii_headon_push_prob: float = 0.0,
    rnapii_pause_cohesin_restraint: float = 1.0,
    rnapii_pause_restraint_window: int = 1,
    rnapii_poised_block_prob: float = 1.0,
    rnapii_paused_block_prob: Optional[float] = None,
    rnapii_elongating_block_prob: Optional[float] = None,
    rnapii_terminating_block_prob: Optional[float] = None,
    rnapii_block_prob: float = 1.0,
    lifetime_rnapii_stalled: Optional[int] = None,
    rnapii_default_load_prob: float = 0.02,
    ep_contact_tolerance: int = 2,
    replicate_genes_across_chains: bool = False,
    targeted_load_prob: float = 0.0,
    loading_window: int = 2,
    target_enhancers: bool = True,
    target_tss: bool = True,
    weight_loading_by_activity: bool = True,
    lesion_spacing: int = 10,
    lesion_block_prob: float = 0.95,
    lesion_type_a_prob: float = 0.5,
    lesion_prerecognition_ticks: int = 100,
    lesion_repair_ticks: int = 100,
    lesion_tad_size_exponent: float = 1.0,
    lesion_tad_repair_exponent: float = 1.0,
    lesion_type_b_enabled: bool = True,
) -> Dict[str, Any]:
    """CTCF TAD layout + per-gene transcription units.

    Mirrors :func:`uniform_tad_topology` for CTCFs (TAD pattern repeated
    across every chain) and additionally populates the gene / RNAPII
    bookkeeping fields consumed by ``rnapii.translocate_rnapii`` and
    ``lef_dynamics.translocate_with_rnapii``.
    """
    args = _base_args(cfg)
    sides = (-1, 1) if symmetric else (1,)

    for chain_idx in range(cfg.num_chains):
        chain_offset = chain_idx * cfg.chain_length
        for pos in tad_positions:
            site = chain_offset + pos
            for side in sides:
                args["ctcfCapture"][side][site] = capture_prob
                args["ctcfRelease"][side][site] = release_prob

    gene_specs = _expand_genes_across_chains(
        cfg,
        genes,
        replicate_genes_across_chains=replicate_genes_across_chains,
    )
    gene_objs = build_genes(gene_specs, default_load_prob=rnapii_default_load_prob)
    args["genes"] = gene_objs
    args["tss_by_pos"] = {g.tss: g.gene_id for g in gene_objs}
    args["tes_by_pos"] = {g.tes: g.gene_id for g in gene_objs}
    args["rnapii_by_pos"] = {}
    args["cohesin_leg_by_pos"] = {}
    args["rnapii_stride"] = int(rnapii_stride)
    args["rnapii_stall_prob"] = float(rnapii_stall_prob)
    args["rnapii_push_prob"] = float(rnapii_push_prob)
    args["rnapii_headon_push_prob"] = float(rnapii_headon_push_prob)
    args["rnapii_pause_cohesin_restraint"] = float(rnapii_pause_cohesin_restraint)
    args["rnapii_pause_restraint_window"] = int(rnapii_pause_restraint_window)
    fallback_block_prob = float(rnapii_block_prob)
    args["rnapii_poised_block_prob"] = float(rnapii_poised_block_prob)
    args["rnapii_paused_block_prob"] = (
        fallback_block_prob
        if rnapii_paused_block_prob is None
        else float(rnapii_paused_block_prob)
    )
    args["rnapii_elongating_block_prob"] = (
        fallback_block_prob
        if rnapii_elongating_block_prob is None
        else float(rnapii_elongating_block_prob)
    )
    # Terminating Pol II defaults to the paused block prob (both stationary).
    args["rnapii_terminating_block_prob"] = (
        args["rnapii_paused_block_prob"]
        if rnapii_terminating_block_prob is None
        else float(rnapii_terminating_block_prob)
    )
    args["rnapii_block_prob"] = fallback_block_prob
    args["LIFETIME_RNAPII_STALLED"] = (
        int(args["LIFETIME_STALLED"])
        if lifetime_rnapii_stalled is None
        else int(lifetime_rnapii_stalled)
    )
    args["ep_contact_tolerance"] = int(ep_contact_tolerance)
    _set_loading_sites(
        args, gene_objs,
        targeted_load_prob=targeted_load_prob,
        loading_window=loading_window,
        target_enhancers=target_enhancers,
        target_tss=target_tss,
        weight_by_activity=weight_loading_by_activity,
    )
    # Lesion (UV-damage) two-state machine -- see plugins.lesions. Population is
    # homeostatic (lesion_spacing sets the steady-state count N // spacing) and
    # placement is TAD-size-weighted; type & state govern cohesin / RNAPII
    # blocking. Lesions actually appear only when the lesion plugin is enabled
    # (it runs update_lesions, which fills + advances them); the fields below
    # just arm the spawner. lesion_spacing == 0 disables lesions entirely.
    args["lesions"] = {}
    args["lesion_block_prob"] = float(lesion_block_prob)
    args["lesion_type_a_prob"] = float(lesion_type_a_prob)
    args["lesion_prerecognition_ticks"] = int(lesion_prerecognition_ticks)
    args["lesion_repair_ticks"] = int(lesion_repair_ticks)
    args["lesion_spacing"] = int(lesion_spacing)
    args["lesion_tad_size_exponent"] = float(lesion_tad_size_exponent)
    args["lesion_tad_repair_exponent"] = float(lesion_tad_repair_exponent)
    args["lesion_type_b_enabled"] = bool(lesion_type_b_enabled)
    if lesion_spacing > 0:
        precompute_lesion_fields(
            args,
            tad_positions=args.get("tad_positions", tad_positions),
            gene_objs=gene_objs,
            tad_size_exponent=lesion_tad_size_exponent,
            tad_repair_exponent=lesion_tad_repair_exponent,
            spacing=lesion_spacing,
        )
    return args


def gene_aware_convergent_tad_topology(
    cfg: LEFConfig,
    *,
    tad_positions: Iterable[int] = (),
    tads: Optional[Iterable[Mapping[str, Any]]] = None,
    boundary_strength: BoundaryStrength = 0.5,
    release_prob: float = 0.0,
    include_chromosome_ends: bool = True,
    default_boundary_strength: float = 0.5,
    genes: Optional[List[dict]] = None,
    rnapii_stride: int = 1,
    rnapii_stall_prob: float = 0.4,
    rnapii_push_prob: float = 0.3,
    rnapii_headon_push_prob: float = 0.0,
    rnapii_pause_cohesin_restraint: float = 1.0,
    rnapii_pause_restraint_window: int = 1,
    rnapii_poised_block_prob: float = 1.0,
    rnapii_paused_block_prob: Optional[float] = None,
    rnapii_elongating_block_prob: Optional[float] = None,
    rnapii_terminating_block_prob: Optional[float] = None,
    rnapii_block_prob: float = 1.0,
    lifetime_rnapii_stalled: Optional[int] = None,
    rnapii_default_load_prob: float = 0.02,
    ep_contact_tolerance: int = 2,
    replicate_genes_across_chains: bool = False,
    targeted_load_prob: float = 0.0,
    loading_window: int = 2,
    target_enhancers: bool = True,
    target_tss: bool = True,
    weight_loading_by_activity: bool = True,
    lesion_spacing: int = 10,
    lesion_block_prob: float = 0.95,
    lesion_type_a_prob: float = 0.5,
    lesion_prerecognition_ticks: int = 100,
    lesion_repair_ticks: int = 100,
    lesion_tad_size_exponent: float = 1.0,
    lesion_tad_repair_exponent: float = 1.0,
    lesion_type_b_enabled: bool = True,
) -> Dict[str, Any]:
    """Directional TAD CTCFs plus per-gene RNAPII bookkeeping."""
    args = convergent_tad_topology(
        cfg,
        tad_positions=tad_positions,
        tads=tads,
        boundary_strength=boundary_strength,
        release_prob=release_prob,
        include_chromosome_ends=include_chromosome_ends,
        default_boundary_strength=default_boundary_strength,
    )

    gene_specs = _expand_genes_across_chains(
        cfg,
        genes,
        replicate_genes_across_chains=replicate_genes_across_chains,
    )
    gene_objs = build_genes(gene_specs, default_load_prob=rnapii_default_load_prob)
    args["genes"] = gene_objs
    args["tss_by_pos"] = {g.tss: g.gene_id for g in gene_objs}
    args["tes_by_pos"] = {g.tes: g.gene_id for g in gene_objs}
    args["rnapii_by_pos"] = {}
    args["cohesin_leg_by_pos"] = {}
    args["rnapii_stride"] = int(rnapii_stride)
    args["rnapii_stall_prob"] = float(rnapii_stall_prob)
    args["rnapii_push_prob"] = float(rnapii_push_prob)
    args["rnapii_headon_push_prob"] = float(rnapii_headon_push_prob)
    args["rnapii_pause_cohesin_restraint"] = float(rnapii_pause_cohesin_restraint)
    args["rnapii_pause_restraint_window"] = int(rnapii_pause_restraint_window)
    fallback_block_prob = float(rnapii_block_prob)
    args["rnapii_poised_block_prob"] = float(rnapii_poised_block_prob)
    args["rnapii_paused_block_prob"] = (
        fallback_block_prob
        if rnapii_paused_block_prob is None
        else float(rnapii_paused_block_prob)
    )
    args["rnapii_elongating_block_prob"] = (
        fallback_block_prob
        if rnapii_elongating_block_prob is None
        else float(rnapii_elongating_block_prob)
    )
    # Terminating Pol II defaults to the paused block prob (both stationary).
    args["rnapii_terminating_block_prob"] = (
        args["rnapii_paused_block_prob"]
        if rnapii_terminating_block_prob is None
        else float(rnapii_terminating_block_prob)
    )
    args["rnapii_block_prob"] = fallback_block_prob
    args["LIFETIME_RNAPII_STALLED"] = (
        int(args["LIFETIME_STALLED"])
        if lifetime_rnapii_stalled is None
        else int(lifetime_rnapii_stalled)
    )
    args["ep_contact_tolerance"] = int(ep_contact_tolerance)
    _set_loading_sites(
        args, gene_objs,
        targeted_load_prob=targeted_load_prob,
        loading_window=loading_window,
        target_enhancers=target_enhancers,
        target_tss=target_tss,
        weight_by_activity=weight_loading_by_activity,
    )
    # Lesion (UV-damage) two-state machine -- see plugins.lesions. Population is
    # homeostatic (lesion_spacing sets the steady-state count N // spacing) and
    # placement is TAD-size-weighted; type & state govern cohesin / RNAPII
    # blocking. Lesions actually appear only when the lesion plugin is enabled
    # (it runs update_lesions, which fills + advances them); the fields below
    # just arm the spawner. lesion_spacing == 0 disables lesions entirely.
    args["lesions"] = {}
    args["lesion_block_prob"] = float(lesion_block_prob)
    args["lesion_type_a_prob"] = float(lesion_type_a_prob)
    args["lesion_prerecognition_ticks"] = int(lesion_prerecognition_ticks)
    args["lesion_repair_ticks"] = int(lesion_repair_ticks)
    args["lesion_spacing"] = int(lesion_spacing)
    args["lesion_tad_size_exponent"] = float(lesion_tad_size_exponent)
    args["lesion_tad_repair_exponent"] = float(lesion_tad_repair_exponent)
    args["lesion_type_b_enabled"] = bool(lesion_type_b_enabled)
    if lesion_spacing > 0:
        precompute_lesion_fields(
            args,
            tad_positions=args.get("tad_positions", tad_positions),
            gene_objs=gene_objs,
            tad_size_exponent=lesion_tad_size_exponent,
            tad_repair_exponent=lesion_tad_repair_exponent,
            spacing=lesion_spacing,
        )
    return args


def ep_pair_topology(
    cfg: LEFConfig,
    *,
    n_pairs: int = 7,
    ep_distance: int = 400,
    pair_spacing: int = 10_000,
    first_pair_offset: Optional[int] = None,
    boundary_strength: BoundaryStrength = 0.5,
    default_boundary_strength: float = 0.5,
    convergent_orientation: bool = True,
) -> Dict[str, Any]:
    """E-P pair layout from the NRMCB supplementary box 1.

    Places ``n_pairs`` cognate enhancer/promoter pairs along the lattice:

        E_i at offset + i * pair_spacing
        P_i at offset + i * pair_spacing + ep_distance

    Each pair is flanked by one CBS at ``E_i - 1`` and one at ``P_i + 1``.
    Convergent orientation (standard convergent-CTCF loop anchor): the left
    CBS stalls the left-moving (-1) leg and the right CBS stalls the
    right-moving (+1) leg, so a cohesin extruding between the two CBSs is
    bracketed and the resulting loop encloses E and P, bringing them into
    proximity. Release probability is zero -- stalled cohesins only leave
    when the LEF unloads (per paper section 2).

    The list of E and P monomer indices is written to
    ``args["sticky_particles"]`` so the polymer-stage force builder can
    consume the same layout via YAML mirroring (or programmatic glue).
    """
    args = _base_args(cfg)
    cfg_N = cfg.num_sites

    if first_pair_offset is None:
        used = (n_pairs - 1) * pair_spacing + ep_distance
        first_pair_offset = max(0, (cfg_N - used) // 2)

    ep_pairs: list = []
    sticky: list = []
    for i in range(n_pairs):
        e = first_pair_offset + i * pair_spacing
        p = e + ep_distance
        if not (0 <= e and p < cfg_N):
            raise ValueError(
                f"E-P pair {i} positions ({e}, {p}) fall outside lattice of size {cfg_N}"
            )
        ep_pairs.append((e, p))
        sticky.extend([e, p])

        left_ctcf = e - 1
        right_ctcf = p + 1
        if 0 <= left_ctcf:
            s = _strength_at(boundary_strength, left_ctcf, default_boundary_strength)
            if convergent_orientation:
                args["ctcfCapture"][-1][left_ctcf] = s   # stalls left-moving (-1) leg
            else:
                args["ctcfCapture"][1][left_ctcf] = s
        if right_ctcf < cfg_N:
            s = _strength_at(boundary_strength, right_ctcf, default_boundary_strength)
            if convergent_orientation:
                args["ctcfCapture"][1][right_ctcf] = s   # stalls right-moving (+1) leg
            else:
                args["ctcfCapture"][-1][right_ctcf] = s

    args["ep_pairs"] = ep_pairs
    args["sticky_particles"] = sticky
    args["boundary_strength"] = boundary_strength
    return args


def explicit_ctcf_topology(
    cfg: LEFConfig,
    *,
    left_capture: Mapping[int, float] | None = None,
    right_capture: Mapping[int, float] | None = None,
    left_release: Mapping[int, float] | None = None,
    right_release: Mapping[int, float] | None = None,
) -> Dict[str, Any]:
    """User supplies CTCF site dictionaries directly via YAML kwargs."""
    args = _base_args(cfg)
    if left_capture:
        args["ctcfCapture"][-1].update({int(k): float(v) for k, v in left_capture.items()})
    if right_capture:
        args["ctcfCapture"][1].update({int(k): float(v) for k, v in right_capture.items()})
    if left_release:
        args["ctcfRelease"][-1].update({int(k): float(v) for k, v in left_release.items()})
    if right_release:
        args["ctcfRelease"][1].update({int(k): float(v) for k, v in right_release.items()})
    return args
