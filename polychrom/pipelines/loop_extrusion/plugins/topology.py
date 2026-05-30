"""Linear-setup plugins: chain layout and CTCF site placement.

A topology plugin returns the ``args`` dict consumed by the LEF dynamics,
already populated with ``N``, ``LIFETIME``, ``LIFETIME_STALLED``,
``ctcfCapture`` and ``ctcfRelease``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from ..config import LEFConfig
from .rnapii import build_genes
from .lesions import seed_periodic_lesions

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


def _base_args(cfg: LEFConfig) -> Dict[str, Any]:
    return {
        "N": cfg.num_sites,
        "chain_length": cfg.chain_length,
        "num_chains": cfg.num_chains,
        "LIFETIME": cfg.lifetime,
        "LIFETIME_STALLED": cfg.lifetime_stalled,
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
    tad_positions: Iterable[int],
    boundary_strength: BoundaryStrength,
    release_prob: float,
    include_chromosome_ends: bool,
    default_boundary_strength: float = 0.5,
) -> None:
    """Place inward-facing CTCF barriers at each TAD interval edge.

    ``boundary_strength`` is resolved per anchor: pass a scalar for a uniform
    barrier, or a ``{position: strength}`` mapping (keyed by the chain-relative
    TAD boundary position) to tune each anchor individually. The left-facing
    anchor of an interval is keyed by its ``start`` boundary, the right-facing
    anchor (sitting at ``end - 1``) by its ``end`` boundary.
    """
    inner = [int(pos) for pos in tad_positions]
    boundaries = [0, *inner, cfg.chain_length]
    for chain_idx in range(cfg.num_chains):
        chain_offset = chain_idx * cfg.chain_length
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            left_site = chain_offset + start
            right_site = chain_offset + end - 1
            if include_chromosome_ends or start != 0:
                args["ctcfCapture"][-1][left_site] = _strength_at(
                    boundary_strength, start, default_boundary_strength)
                args["ctcfRelease"][-1][left_site] = float(release_prob)
            if include_chromosome_ends or end != cfg.chain_length:
                args["ctcfCapture"][1][right_site] = _strength_at(
                    boundary_strength, end, default_boundary_strength)
                args["ctcfRelease"][1][right_site] = float(release_prob)


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
) -> None:
    """Populate ``args`` with targeted cohesin-loading sites (enhancers/TSS).

    Consumed by ``lef_dynamics.load_targeted``. ``targeted_load_prob == 0``
    (default) leaves loading uniform even if this is called.
    """
    sites: set = set()
    if target_enhancers:
        sites |= {e for g in gene_objs for e in g.enhancers}
    if target_tss:
        sites |= {g.tss for g in gene_objs}
    args["loading_sites"] = sorted(int(s) for s in sites)
    args["targeted_load_prob"] = float(targeted_load_prob)
    args["loading_window"] = int(loading_window)


def convergent_tad_topology(
    cfg: LEFConfig,
    *,
    tad_positions: Iterable[int] = (),
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

    ``boundary_strength`` accepts a scalar (uniform) or a ``{position: strength}``
    mapping to set each TAD boundary individually; positions missing from the
    mapping use ``default_boundary_strength``.
    """
    args = _base_args(cfg)
    _apply_convergent_tads(
        args,
        cfg,
        tad_positions=tad_positions,
        boundary_strength=boundary_strength,
        release_prob=release_prob,
        include_chromosome_ends=include_chromosome_ends,
        default_boundary_strength=default_boundary_strength,
    )
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
    lesion_prob: float = 0.0,
    lesion_lifetime: int = 100,
    lesion_block_prob: float = 0.95,
    lesion_max: int = 64,
    lesion_spacing: int = 0,
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
    )
    # Lesion (UV-damage) state + parameters, consumed by lesions.update_lesions.
    args["lesions"] = {}
    args["lesion_prob"] = float(lesion_prob)
    args["lesion_lifetime"] = int(lesion_lifetime)
    args["lesion_block_prob"] = float(lesion_block_prob)
    args["lesion_max"] = int(lesion_max)
    # Optional: deterministic CPD seeding -- a UV pulse depositing lesions at
    # every ``lesion_spacing`` monomer in each gene body, persisting for
    # lesion_lifetime ticks (set very large for permanent damage).
    if lesion_spacing > 0:
        seed_periodic_lesions(args, int(lesion_spacing), int(lesion_lifetime))
    return args


def gene_aware_convergent_tad_topology(
    cfg: LEFConfig,
    *,
    tad_positions: Iterable[int] = (),
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
    lesion_prob: float = 0.0,
    lesion_lifetime: int = 100,
    lesion_block_prob: float = 0.95,
    lesion_max: int = 64,
    lesion_spacing: int = 0,
) -> Dict[str, Any]:
    """Directional TAD CTCFs plus per-gene RNAPII bookkeeping."""
    args = convergent_tad_topology(
        cfg,
        tad_positions=tad_positions,
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
    )
    # Lesion (UV-damage) state + parameters, consumed by lesions.update_lesions.
    args["lesions"] = {}
    args["lesion_prob"] = float(lesion_prob)
    args["lesion_lifetime"] = int(lesion_lifetime)
    args["lesion_block_prob"] = float(lesion_block_prob)
    args["lesion_max"] = int(lesion_max)
    # Optional: deterministic CPD seeding -- a UV pulse depositing lesions at
    # every ``lesion_spacing`` monomer in each gene body, persisting for
    # lesion_lifetime ticks (set very large for permanent damage).
    if lesion_spacing > 0:
        seed_periodic_lesions(args, int(lesion_spacing), int(lesion_lifetime))
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
