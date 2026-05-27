"""RNA polymerase II agent on the 1D loop-extrusion lattice.

Two translocate plugins live here:

* :func:`translocate_rnapii` — single-state RNAPII that walks at a fixed
  integer ``rnapii_stride`` per tick (v1, kept for backwards compatibility).
* :func:`stateful_translocate_rnapii` — three-state biological model
  (POISED -> PAUSED -> ELONGATING) with per-state stochastic step
  probability and optional enhancer-dependent pause release. The 1D
  enhancer-promoter contact is detected by :func:`compute_ep_contacts`
  using a cohesin-loop-containment proxy.

Lattice encoding (``occupied``)::

    0 = free
    1 = cohesin leg
    2 = RNAPII body

RNAPII state codes (stored in ``RNAPII.attrs["state"]`` and the optional
``rnapii_states`` HDF5 dataset)::

    0 = POISED      (bound at TSS, not yet initiated)
    1 = PAUSED      (initiated, promoter-proximal pause)
    2 = ELONGATING  (productive elongation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

FREE = 0
COHESIN = 1
RNAPII_CELL = 2

STATE_POISED = 0
STATE_PAUSED = 1
STATE_ELONGATING = 2


def _chain_length(args: Dict) -> int:
    return int(args.get("chain_length", args["N"]))


def _same_chain(a: int, b: int, args: Dict) -> bool:
    chain_length = _chain_length(args)
    return int(a) // chain_length == int(b) // chain_length


def _valid_step(pos: int, target: int, args: Dict) -> bool:
    return 0 <= target < args["N"] and _same_chain(pos, target, args)


@dataclass
class Gene:
    """A single transcription unit on the lattice."""

    gene_id: int
    tss: int
    tes: int
    direction: int                          # +1 (TSS<TES) or -1 (TSS>TES)
    load_prob: float                        # per-tick recruitment to TSS
    enhancer_pos: Optional[int] = None      # required iff requires_enhancer
    requires_enhancer: bool = False
    load_requires_enhancer: bool = False    # recruitment also requires E-P contact
    initiation_prob: float = 1.0            # POISED -> PAUSED per tick
    pause_release_prob: float = 1.0         # PAUSED  -> ELONGATING per tick
    elongation_step_prob: float = 1.0       # per-tick step prob during ELONGATING
    pause_offset: int = 0                   # PAUSED site = TSS + direction*pause_offset


class RNAPII:
    """One translocating polymerase."""

    __slots__ = ("pos", "gene_id", "direction", "attrs")

    def __init__(self, pos: int, gene_id: int, direction: int):
        self.pos = pos
        self.gene_id = gene_id
        self.direction = direction
        self.attrs: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Topology bookkeeping helpers
# ---------------------------------------------------------------------------

def build_genes(
    gene_specs: List[dict],
    *,
    default_load_prob: float = 0.02,
) -> List[Gene]:
    """Materialise YAML gene specs into :class:`Gene` instances."""
    genes: List[Gene] = []
    for gid, spec in enumerate(gene_specs):
        tss = int(spec["tss"])
        tes = int(spec["tes"])
        if tss == tes:
            raise ValueError(f"Gene {gid}: TSS and TES must differ")
        direction = 1 if tes > tss else -1
        load_prob = float(spec.get("load_prob", default_load_prob))
        enhancer_pos = spec.get("enhancer_pos")
        if enhancer_pos is not None:
            enhancer_pos = int(enhancer_pos)
        requires_enhancer = bool(spec.get("requires_enhancer", False))
        if requires_enhancer and enhancer_pos is None:
            raise ValueError(
                f"Gene {gid} requires_enhancer=True but no enhancer_pos given"
            )
        genes.append(Gene(
            gene_id=gid,
            tss=tss,
            tes=tes,
            direction=direction,
            load_prob=load_prob,
            enhancer_pos=enhancer_pos,
            requires_enhancer=requires_enhancer,
            load_requires_enhancer=bool(spec.get("load_requires_enhancer", False)),
            initiation_prob=float(spec.get("initiation_prob", 1.0)),
            pause_release_prob=float(spec.get("pause_release_prob", 1.0)),
            elongation_step_prob=float(spec.get("elongation_step_prob", 1.0)),
            pause_offset=int(spec.get("pause_offset", 0)),
        ))
    return genes


# ---------------------------------------------------------------------------
# E-P contact (1D loop-containment proxy)
# ---------------------------------------------------------------------------

def compute_ep_contacts(
    cohesins,
    genes: Iterable[Gene],
    tolerance: int = 2,
) -> Set[int]:
    """Return the set of gene_ids whose (TSS, enhancer_pos) pair is bracketed
    by at least one cohesin's loop interval, within ``tolerance`` sites.

    The tolerance compensates for cohesin legs that cannot sit exactly at
    the TSS when a POISED/PAUSED RNAPII occupies that site -- the loop
    still topologically encloses the EP pair when its endpoints are one
    or two sites off the EP coordinates.

    Genes with ``enhancer_pos is None`` are skipped (never reported).
    """
    intervals = [
        (min(c.left.pos, c.right.pos), max(c.left.pos, c.right.pos))
        for c in cohesins
    ]
    out: Set[int] = set()
    for g in genes:
        if g.enhancer_pos is None:
            continue
        lo = min(g.tss, g.enhancer_pos) + tolerance
        hi = max(g.tss, g.enhancer_pos) - tolerance
        for L, R in intervals:
            if L <= lo and R >= hi:
                out.add(g.gene_id)
                break
    return out


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def load_rnapii(rnapiis: List[RNAPII], occupied: np.ndarray, args: Dict) -> None:
    """For each gene, stochastically spawn a new RNAPII at its TSS.

    New RNAPIIs start in the POISED state.
    """
    genes: List[Gene] = args["genes"]
    rnapii_by_pos: Dict[int, RNAPII] = args["rnapii_by_pos"]
    ep_contacts = args.get("current_ep_contacts", set())
    for gene in genes:
        if gene.load_requires_enhancer and gene.gene_id not in ep_contacts:
            continue
        if np.random.random() >= gene.load_prob:
            continue
        if occupied[gene.tss] != FREE:
            continue
        r = RNAPII(pos=gene.tss, gene_id=gene.gene_id, direction=gene.direction)
        r.attrs["state"] = STATE_POISED
        occupied[gene.tss] = RNAPII_CELL
        rnapii_by_pos[gene.tss] = r
        rnapiis.append(r)


def _resolve_head_on(r: "RNAPII", leg, args: Dict) -> str:
    """Resolve an RNAPII stepping into a cohesin leg.

    Returns ``"stall"`` (RNAPII blocked, stays put) or ``"push"`` (RNAPII
    advances, displacing the cohesin leg).

    Biology (Fursova & Larson, *Curr. Opin. Struct. Biol.* 2024, Fig. 3a):

    * Only ELONGATING RNAPII translocates productively, so only it can
      push a cohesin. POISED / PAUSED RNAPII is a stationary block and
      always stalls on contact -- it never displaces cohesin.
    * An elongating RNAPII pushes a *co-directional* cohesin leg (rear
      encounter) far more readily than a *head-on* (converging) leg, which
      tends to stall or slow the polymerase.

    Backwards-compatible defaults: a v1 single-state RNAPII carries no
    ``state`` attr and is treated as always-elongating; a leg with no
    recorded ``dir`` is treated as co-directional.
    """
    state = r.attrs.get("state")
    if state is not None and state != STATE_ELONGATING:
        return "stall"

    # Intrinsic stall floor (Pol II pausing/slowing on any obstacle),
    # applied regardless of orientation.
    if np.random.random() < float(args.get("rnapii_stall_prob", 0.0)):
        return "stall"

    leg_dir = leg.attrs.get("dir") if leg is not None else None
    head_on = leg_dir is not None and leg_dir == -r.direction
    push_p = float(args.get(
        "rnapii_headon_push_prob" if head_on else "rnapii_push_prob", 0.0
    ))
    return "push" if np.random.random() < push_p else "stall"


def _try_single_step(r: RNAPII, gene: Gene, occupied: np.ndarray, args: Dict) -> bool:
    """Attempt a single +direction step. Returns True if the RNAPII advanced.

    Handles all four target-cell cases (FREE / RNAPII / COHESIN / TES). On
    a cohesin head-on encounter resolves stall / push / pass per
    :func:`_resolve_head_on`. On TES arrival the RNAPII steps into the
    TES slot even if a cohesin sits there (bypass slot).
    """
    rnapii_by_pos = args["rnapii_by_pos"]
    target = r.pos + r.direction
    if not _valid_step(r.pos, target, args):
        return False

    if target == gene.tes:
        rnapii_by_pos.pop(r.pos, None)
        if occupied[r.pos] == RNAPII_CELL:
            occupied[r.pos] = FREE
        r.pos = target
        rnapii_by_pos[target] = r
        return True

    cell = occupied[target]

    if cell == RNAPII_CELL:
        return False

    if cell == COHESIN:
        leg = args["cohesin_leg_by_pos"].get(target)
        if _resolve_head_on(r, leg, args) == "stall":
            return False
        # push: displace the co-directional leg backward (in RNAPII's direction).
        behind = target + r.direction
        if not (_valid_step(target, behind, args) and occupied[behind] == FREE):
            return False
        occupied[behind] = COHESIN
        occupied[target] = RNAPII_CELL
        rnapii_by_pos.pop(r.pos, None)
        if occupied[r.pos] == RNAPII_CELL:
            occupied[r.pos] = FREE
        args["cohesin_leg_by_pos"].pop(target, None)
        if leg is not None:
            leg.pos = behind
            leg.attrs["pushed"] = True
            args["cohesin_leg_by_pos"][behind] = leg
        r.pos = target
        rnapii_by_pos[target] = r
        return True

    # cell == FREE
    if occupied[r.pos] == RNAPII_CELL:
        occupied[r.pos] = FREE
    rnapii_by_pos.pop(r.pos, None)
    occupied[target] = RNAPII_CELL
    rnapii_by_pos[target] = r
    r.pos = target
    return True


def _unload_at_tes(r: RNAPII, gene: Gene, occupied: np.ndarray, args: Dict) -> bool:
    """If the RNAPII sits at its TES, clear it and return True."""
    if r.pos != gene.tes:
        return False
    if occupied[r.pos] == RNAPII_CELL:
        occupied[r.pos] = FREE
    args["rnapii_by_pos"].pop(r.pos, None)
    return True


# ---------------------------------------------------------------------------
# v1: single-state translocate (kept for back-compat)
# ---------------------------------------------------------------------------

def translocate_rnapii(
    rnapiis: List[RNAPII],
    cohesins,                # unused but kept for symmetry with v2
    occupied: np.ndarray,
    args: Dict,
) -> None:
    """v1 RNAPII dynamics: single state, integer stride per tick."""
    genes: List[Gene] = args["genes"]
    stride: int = int(args.get("rnapii_stride", 1))

    for idx in range(len(rnapiis) - 1, -1, -1):
        r = rnapiis[idx]
        gene = genes[r.gene_id]
        if _unload_at_tes(r, gene, occupied, args):
            del rnapiis[idx]
            continue
        for _ in range(stride):
            if not _try_single_step(r, gene, occupied, args):
                break


# ---------------------------------------------------------------------------
# v2: biological state machine (POISED / PAUSED / ELONGATING)
# ---------------------------------------------------------------------------

def stateful_translocate_rnapii(
    rnapiis: List[RNAPII],
    cohesins,
    occupied: np.ndarray,
    args: Dict,
) -> None:
    """v2 RNAPII dynamics with POISED/PAUSED/ELONGATING states.

    Per tick (per RNAPII):

    * Unload at TES (existing one-tick dwell + cohesin bypass behaviour).
    * **POISED**: with probability ``gene.initiation_prob`` transition to
      PAUSED; if ``gene.pause_offset > 0`` and the pause site is free,
      hop there. Otherwise PAUSE in place.
    * **PAUSED**: if the gene requires enhancer contact and the gene is
      not currently in E-P contact (loop-containment proxy), stay PAUSED.
      Otherwise transition to ELONGATING with probability
      ``gene.pause_release_prob``. On transition, fall through to one
      elongation step in the same tick.
    * **ELONGATING**: attempt one ``+direction`` step with probability
      ``gene.elongation_step_prob`` (sub-site biological speed).
    """
    genes: List[Gene] = args["genes"]
    tol = int(args.get("ep_contact_tolerance", 2))
    ep_contacts = compute_ep_contacts(cohesins, genes, tolerance=tol)

    for idx in range(len(rnapiis) - 1, -1, -1):
        r = rnapiis[idx]
        gene = genes[r.gene_id]

        if _unload_at_tes(r, gene, occupied, args):
            del rnapiis[idx]
            continue

        state = r.attrs.get("state", STATE_POISED)

        if state == STATE_POISED:
            if np.random.random() < gene.initiation_prob:
                if gene.pause_offset > 0:
                    # Try to hop to pause site; if blocked, pause in place.
                    if _try_single_step(r, gene, occupied, args):
                        pass  # advanced one step toward pause
                r.attrs["state"] = STATE_PAUSED
            continue

        if state == STATE_PAUSED:
            if gene.requires_enhancer and gene.gene_id not in ep_contacts:
                continue
            if np.random.random() >= gene.pause_release_prob:
                continue
            r.attrs["state"] = STATE_ELONGATING
            # fall through to elongation step this tick

        # state == STATE_ELONGATING (either entering this tick or carried over)
        if np.random.random() < gene.elongation_step_prob:
            _try_single_step(r, gene, occupied, args)
