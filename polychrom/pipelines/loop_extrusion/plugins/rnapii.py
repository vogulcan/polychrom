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

    0 = POISED       (bound at TSS, not yet initiated)
    1 = PAUSED       (initiated, promoter-proximal pause)
    2 = ELONGATING   (productive elongation)
    3 = TERMINATING  (reached TES, dwelling there before unloading)
    4 = STALLED      (in the gene body, attempted a step but physically blocked
                      this tick -- cohesin / traffic / lesion; NOT promoter-
                      proximal pause and NOT productive elongation)

POISED, PAUSED, STALLED and TERMINATING Pol II are stationary blocks to
cohesin; only ELONGATING Pol II can push it (Fursova & Larson 2024, Fig 3a).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

FREE = 0
COHESIN = 1
RNAPII_CELL = 2

STATE_POISED = 0
STATE_PAUSED = 1
STATE_ELONGATING = 2
STATE_TERMINATING = 3
STATE_STALLED = 4


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
    enhancer_pos: Optional[int] = None      # back-compat: primary (first) enhancer
    # All enhancers regulating this gene (shadow / super-enhancer). One gene can
    # integrate several, per ``enhancer_logic``. ``enhancer_pos`` mirrors the
    # first entry for back-compat.
    enhancers: Tuple[int, ...] = ()
    # How simultaneous E-P contacts combine into transcriptional output:
    #   "any"         -> redundancy: any one contact suffices (shadow enhancers)
    #   "all"         -> obligate cooperativity: every enhancer must contact
    #   "additive"    -> output scales with #contacts (DEFAULT; empirical baseline,
    #                    Bothma/Levo eLife 2015; housekeeping enhancers eLife 2024)
    #   "synergistic" -> super-additive output (#contacts ** enhancer_synergy;
    #                    developmental enhancers)
    enhancer_logic: str = "additive"
    enhancer_synergy: float = 1.5           # exponent used iff logic == "synergistic"
    requires_enhancer: bool = False
    load_requires_enhancer: bool = False    # recruitment also requires E-P contact
    initiation_prob: float = 1.0            # POISED -> PAUSED per tick
    pause_release_prob: float = 1.0         # PAUSED  -> ELONGATING per tick
    elongation_step_prob: float = 1.0       # per-tick step prob during ELONGATING
    pause_offset: int = 0                   # PAUSED site = TSS + direction*pause_offset
    termination_prob: float = 1.0           # TERMINATING -> unload per tick (1.0 = no dwell)

    def __post_init__(self) -> None:
        # Keep ``enhancers`` (canonical) and ``enhancer_pos`` (legacy single) in
        # sync however the Gene was constructed.
        if not self.enhancers and self.enhancer_pos is not None:
            self.enhancers = (int(self.enhancer_pos),)
        elif self.enhancers:
            self.enhancers = tuple(int(e) for e in self.enhancers)
            if self.enhancer_pos is None:
                self.enhancer_pos = self.enhancers[0]


class RNAPII:
    """One translocating polymerase."""

    __slots__ = ("pos", "gene_id", "direction", "uid", "attrs")

    def __init__(self, pos: int, gene_id: int, direction: int, uid: int = -1):
        self.pos = pos
        self.gene_id = gene_id
        self.direction = direction
        self.uid = uid              # stable per-Pol identity for trajectory tracks
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
        # Accept a list under ``enhancers`` (multi-enhancer genes) or a single
        # ``enhancer_pos`` (legacy). The list wins when both are present.
        raw_enhancers = spec.get("enhancers")
        if raw_enhancers is None:
            ep = spec.get("enhancer_pos")
            enhancers: Tuple[int, ...] = () if ep is None else (int(ep),)
        else:
            enhancers = tuple(int(e) for e in raw_enhancers)
        requires_enhancer = bool(spec.get("requires_enhancer", False))
        if requires_enhancer and not enhancers:
            raise ValueError(
                f"Gene {gid} requires_enhancer=True but no enhancers given"
            )
        logic = str(spec.get("enhancer_logic", "additive"))
        if logic not in ("any", "all", "additive", "synergistic"):
            raise ValueError(
                f"Gene {gid} enhancer_logic={logic!r} not in "
                "{'any','all','additive','synergistic'}"
            )
        genes.append(Gene(
            gene_id=gid,
            tss=tss,
            tes=tes,
            direction=direction,
            load_prob=load_prob,
            enhancer_pos=enhancers[0] if enhancers else None,
            enhancers=enhancers,
            enhancer_logic=logic,
            enhancer_synergy=float(spec.get("enhancer_synergy", 1.5)),
            requires_enhancer=requires_enhancer,
            load_requires_enhancer=bool(spec.get("load_requires_enhancer", False)),
            initiation_prob=float(spec.get("initiation_prob", 1.0)),
            pause_release_prob=float(spec.get("pause_release_prob", 1.0)),
            elongation_step_prob=float(spec.get("elongation_step_prob", 1.0)),
            pause_offset=int(spec.get("pause_offset", 0)),
            termination_prob=float(spec.get("termination_prob", 1.0)),
        ))
    return genes


# ---------------------------------------------------------------------------
# E-P contact (1D loop-containment proxy)
# ---------------------------------------------------------------------------

def compute_ep_contacts(
    cohesins,
    genes: Iterable[Gene],
    tolerance: int = 2,
) -> Dict[int, int]:
    """Map each gene_id to HOW MANY of its enhancers are currently in E-P contact.

    An enhancer is "in contact" with its promoter (TSS) when at least one
    cohesin loop interval brackets the (TSS, enhancer) pair within ``tolerance``
    sites. The tolerance compensates for cohesin legs that cannot sit exactly at
    the TSS when a POISED/PAUSED RNAPII occupies that site -- the loop still
    topologically encloses the EP pair when its endpoints are one or two sites
    off the EP coordinates.

    Only genes with >=1 enhancer in contact appear in the returned mapping, so
    ``gene_id in result`` still tests "this gene has E-P contact" (a multi-
    enhancer generalisation of the old set return). The count drives dosage in
    :func:`enhancer_factor`. Genes with no enhancers are skipped.
    """
    intervals = [
        (min(c.left.pos, c.right.pos), max(c.left.pos, c.right.pos))
        for c in cohesins
    ]
    out: Dict[int, int] = {}
    for g in genes:
        if not g.enhancers:
            continue
        count = 0
        for enhancer in g.enhancers:
            lo = min(g.tss, enhancer) + tolerance
            hi = max(g.tss, enhancer) - tolerance
            if any(L <= lo and R >= hi for L, R in intervals):
                count += 1
        if count:
            out[g.gene_id] = count
    return out


def _contact_count(ep_contacts, gene_id: int) -> int:
    """#enhancers in contact for ``gene_id``, tolerating a dict or a plain set.

    ``compute_ep_contacts`` returns a count dict, but callers (and tests) may
    still pass a bare set of gene_ids; for a set, membership counts as one.
    """
    if isinstance(ep_contacts, dict):
        return int(ep_contacts.get(gene_id, 0))
    return 1 if gene_id in ep_contacts else 0


def enhancer_factor(gene: Gene, n_contacts: int) -> float:
    """Transcriptional-output multiplier from #enhancers in contact.

    Returns 0.0 when the enhancer requirement is unmet (gate closed). For a
    single-enhancer gene this is always 0.0 / 1.0 -- identical to the old
    boolean behaviour -- regardless of ``enhancer_logic``. Multi-enhancer genes:

      * ``any``         -> 1.0 if >=1 contact (redundant shadow enhancers)
      * ``all``         -> 1.0 only if every enhancer is in contact
      * ``additive``    -> min(n_contacts, k): linear dosage (DEFAULT)
      * ``synergistic`` -> n_contacts ** enhancer_synergy: super-additive
    """
    k = len(gene.enhancers)
    if k == 0:
        return 1.0
    logic = gene.enhancer_logic
    if logic == "all":
        return 1.0 if n_contacts >= k else 0.0
    if n_contacts <= 0:
        return 0.0
    if logic == "any":
        return 1.0
    if logic == "synergistic":
        return float(n_contacts) ** float(gene.enhancer_synergy)
    return float(min(n_contacts, k))   # additive


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
        load_prob = gene.load_prob
        if gene.load_requires_enhancer:
            factor = enhancer_factor(gene, _contact_count(ep_contacts, gene.gene_id))
            if factor <= 0.0:
                continue
            load_prob = min(1.0, gene.load_prob * factor)
        if np.random.random() >= load_prob:
            continue
        if occupied[gene.tss] != FREE:
            continue
        uid = int(args.get("_rnapii_uid_next", 0))
        args["_rnapii_uid_next"] = uid + 1
        r = RNAPII(pos=gene.tss, gene_id=gene.gene_id, direction=gene.direction, uid=uid)
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

    # A lesion blocks RNAPII -- it cannot transcribe through the damage and
    # stalls just upstream (lesion-stalled Pol II).
    lesions = args.get("lesions")
    if lesions and target in lesions:
        r.attrs["lesion_stalled"] = True
        return False

    if target == gene.tes:
        # TES is a normal occupiable slot: a terminating Pol II dwelling here must
        # block cohesin (rnapii_terminating_block_prob) and exclude other Pol II.
        # Stall if the slot is taken rather than stacking / erasing a marker.
        if occupied[target] != FREE:
            return False
        rnapii_by_pos.pop(r.pos, None)
        if occupied[r.pos] == RNAPII_CELL:
            occupied[r.pos] = FREE
        occupied[target] = RNAPII_CELL
        r.pos = target
        rnapii_by_pos[target] = r
        return True

    cell = occupied[target]

    if cell == RNAPII_CELL:
        return False

    if cell == COHESIN:
        # The grid may show COHESIN while an RNAPII is masked under a tunnelling
        # cohesin (bypass). Never step onto a cell that still holds a Pol II.
        if target in rnapii_by_pos:
            return False
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
            # Pushed by elongating Pol II == transcription collision -> mark for
            # Pol II-specific fast eviction (Busslinger 2017; Jeppsson 2022).
            leg.attrs["rnapii_stalled"] = True
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
    """v2 RNAPII dynamics with POISED/PAUSED/ELONGATING/TERMINATING states.

    Per tick (per RNAPII):

    * **At TES**: enter TERMINATING and dwell there as a stationary block
      (Fursova & Larson 2024, Fig 3a: terminating Pol II blocks cohesin,
      giving TES accumulation). Unload with probability
      ``gene.termination_prob`` (``1.0`` = old one-tick behaviour).
    * **POISED**: with probability ``gene.initiation_prob`` transition to
      PAUSED; if ``gene.pause_offset > 0`` and the pause site is free,
      hop there. Otherwise PAUSE in place.
    * **PAUSED**: if the gene requires enhancer contact and the gene is
      not currently in E-P contact (loop-containment proxy), stay PAUSED.
      Otherwise transition to ELONGATING with probability
      ``gene.pause_release_prob``. On transition, fall through to one
      elongation step in the same tick.

      Cohesin gatekeeper (Tei et al. 2026): a cohesin leg physically
      associated within ``rnapii_pause_restraint_window`` sites of the
      paused Pol II RESTRAINS (delays) pause release, multiplying the
      release probability by ``rnapii_pause_cohesin_restraint`` (<1). This
      is the second, opposing cohesin arm to E-P-mediated activation: on
      cohesin loss the restraint is relieved (faster release), which
      compensates for reduced recruitment and keeps steady-state output
      roughly constant (the cohesin-loss paradox). Default ``1.0`` = off.
    * **ELONGATING**: attempt one ``+direction`` step with probability
      ``gene.elongation_step_prob`` (sub-site biological speed).
    """
    genes: List[Gene] = args["genes"]
    tol = int(args.get("ep_contact_tolerance", 2))
    ep_contacts = compute_ep_contacts(cohesins, genes, tolerance=tol)

    # Tei 2026 gatekeeper: precompute cohesin leg positions once per tick so the
    # PAUSED branch can detect a cohesin physically associated with the paused
    # Pol II and restrain its release. Skipped entirely when restraint is off.
    restraint = float(args.get("rnapii_pause_cohesin_restraint", 1.0))
    restraint_window = int(args.get("rnapii_pause_restraint_window", 1))
    leg_positions: Optional[np.ndarray] = None
    if restraint < 1.0:
        legs = [c.left.pos for c in cohesins] + [c.right.pos for c in cohesins]
        leg_positions = np.array(legs, dtype=np.int64) if legs else None

    for idx in range(len(rnapiis) - 1, -1, -1):
        r = rnapiis[idx]
        gene = genes[r.gene_id]

        # At TES: terminating Pol II dwells as a stationary block, then unloads.
        if r.pos == gene.tes:
            r.attrs["state"] = STATE_TERMINATING
            if np.random.random() < gene.termination_prob:
                _unload_at_tes(r, gene, occupied, args)
                del rnapiis[idx]
            continue

        state = r.attrs.get("state", STATE_POISED)

        if state == STATE_POISED:
            if np.random.random() < gene.initiation_prob:
                # Hop up to pause_offset sites toward the pause site; stop early
                # if blocked. pause_offset is a distance, not a flag.
                for _ in range(gene.pause_offset):
                    if not _try_single_step(r, gene, occupied, args):
                        break
                r.attrs["state"] = STATE_PAUSED
            continue

        if state == STATE_PAUSED:
            release_prob = gene.pause_release_prob
            if gene.requires_enhancer:
                factor = enhancer_factor(
                    gene, _contact_count(ep_contacts, gene.gene_id)
                )
                if factor <= 0.0:
                    continue                       # gate closed: stay PAUSED
                release_prob = min(1.0, gene.pause_release_prob * factor)
            if leg_positions is not None and np.any(
                np.abs(leg_positions - r.pos) <= restraint_window
            ):
                release_prob *= restraint           # cohesin restrains release
            if np.random.random() >= release_prob:
                continue
            r.attrs["state"] = STATE_ELONGATING
            # fall through to elongation step this tick

        # state == STATE_ELONGATING / STATE_STALLED (entering or carried over).
        # Rolling the speed dice and failing = slow but productive elongation
        # (stays ELONGATING). Rolling it and being physically blocked = STALLED
        # (cohesin / traffic / lesion); this keeps obstacle stalls out of the
        # elongation metric and out of the %paused denominator downstream.
        if np.random.random() < gene.elongation_step_prob:
            advanced = _try_single_step(r, gene, occupied, args)
            r.attrs["state"] = STATE_ELONGATING if advanced else STATE_STALLED
        else:
            r.attrs["state"] = STATE_ELONGATING
