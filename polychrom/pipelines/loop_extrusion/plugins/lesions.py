"""DNA-lesion (UV-damage) dynamics on the 1D loop-extrusion lattice.

A lesion is a damaged lattice site -- a standalone DNA feature (independent of
who occupies the site) -- carrying a **type** and progressing through a
**two-state machine**.

Type (assigned at spawn):

* **Type A** (transcription-coupled, TC-NER): only in a gene body. Mimics a
  Pol II stalled at the damage. In pre-recognition it blocks RNAPII (creating a
  lesion-stalled Pol II, which in turn blocks cohesin); where RNAPII is *not*
  part of the model it stalls cohesin itself, marked for Pol II-style fast
  eviction (``LIFETIME_RNAPII_STALLED``).
* **Type B** (global-genome, GG-NER): the only type off gene bodies, and the
  alternative type inside them. It does NOT block RNAPII or cohesin during
  pre-recognition (the polymerase reads through); only once the repair machinery
  has assembled (repair state) does it become a roadblock.

State (both durations are stochastic, parameterised in ticks like the cohesin
lifetime -- a memory-less per-tick transition probability of ``1/ticks``)::

    PRE    (pre-recognition) --1/lesion_prerecognition_ticks--> REPAIR
    REPAIR (machinery bound)  --1/lesion_repair_ticks---------> repaired (removed)

Interaction matrix consumed by ``lef_dynamics`` (cohesin) and ``rnapii``::

    (type, state)   blocks RNAPII?   stalls cohesin?
    A, PRE          yes              via stalled Pol II if RNAPII in model, else
                                     intrinsic (rnapii_stalled fast eviction)
    B, PRE          no               no
    A/B, REPAIR     yes              yes (generic stall, LIFETIME_STALLED)

Population is **homeostatic**: ``lesion_spacing`` sets the steady-state count
``target = N // lesion_spacing`` (e.g. N=10000, spacing=10 -> 1000 lesions). Each
tick the state machine removes repaired lesions and the spawner refills back to
``target``. New lesions are placed genome-wide with a per-site probability
proportional to ``L_TAD ** (-lesion_tad_size_exponent)`` (``L_TAD`` = length of
the TAD containing the site), so shorter TADs accumulate relatively more damage.
A lesion in a gene body is Type A with probability ``lesion_type_a_prob`` (else
B); off gene bodies a lesion is always Type B.

Recognition and repair are **TAD-size-dependent**: both transition rates are
scaled per lesion by ``(L_ref / L_TAD) ** lesion_tad_repair_exponent`` (``L_ref``
= mean TAD length), so lesions in shorter TADs are recognised and repaired faster.
The exponent ``0`` recovers uniform rates.

State lives in ``args["lesions"]``: ``Dict[int, Lesion]`` keyed by site (so the
``site in lesions`` / ``lesions.get(site)`` checks elsewhere stay valid).
Topology calls :func:`precompute_lesion_fields` once to populate
``args["gene_body_mask"]`` (bool[N]), ``args["lesion_site_p"]`` (normalised
placement weights, float[N]), ``args["lesion_rate_mult"]`` (rate multipliers,
float[N] or None) and ``args["lesion_target"]`` (int).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Lesion type codes (stored in Lesion.ltype and the lesion_types HDF5 dataset).
LESION_TYPE_A = 0   # transcription-coupled (TC-NER): mimics Pol II stalled at damage
LESION_TYPE_B = 1   # global-genome (GG-NER)

# Lesion state codes (stored in Lesion.state and the lesion_states HDF5 dataset).
LESION_PRE = 0      # pre-recognition
LESION_REPAIR = 1   # repair (machinery assembled)


class Lesion:
    """One damaged site: a type label and a repair-state.

    No per-lesion countdown: the PRE->REPAIR->repaired transitions fire
    stochastically each tick (mirroring cohesin's ``1/LIFETIME`` unloading).
    """

    __slots__ = ("site", "ltype", "state")

    def __init__(self, site: int, ltype: int, state: int = LESION_PRE):
        self.site = site
        self.ltype = ltype
        self.state = state

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        t = "A" if self.ltype == LESION_TYPE_A else "B"
        s = "pre" if self.state == LESION_PRE else "repair"
        return f"Lesion(site={self.site}, type={t}, state={s})"


def is_lesion(site: int, args: Dict) -> bool:
    """True if ``site`` currently carries a lesion."""
    return int(site) in args.get("lesions", {})


# ---------------------------------------------------------------------------
# Interaction predicates (consumed by rnapii.py and lef_dynamics.py)
# ---------------------------------------------------------------------------

def lesion_blocks_rnapii(les: Lesion) -> bool:
    """Whether a lesion blocks an RNAPII trying to step onto its site.

    Type A in pre-recognition is a transcription-blocking lesion (-> stalled
    Pol II). Type B in pre-recognition is read through. Once in repair, the
    assembled machinery blocks transcription regardless of type.
    """
    if les.state == LESION_REPAIR:
        return True
    return les.ltype == LESION_TYPE_A


def lesion_stalls_cohesin(les: Lesion, rnapii_enabled: bool) -> Tuple[bool, bool]:
    """Whether a lesion intrinsically stalls an incoming cohesin leg.

    Returns ``(stalls, rnapii_evict)``; ``rnapii_evict`` requests the
    transcription-specific fast-eviction regime (``LIFETIME_RNAPII_STALLED``).

    * REPAIR (either type): stalls, generic eviction (the repair complex is a
      physical roadblock, not transcription).
    * PRE + Type A: stalls **intrinsically only when RNAPII is absent** from the
      model (otherwise the real stalled Pol II blocks cohesin via the RNAPII
      path); the intrinsic stall mimics a stalled Pol II -> fast eviction.
    * PRE + Type B: never stalls cohesin.
    """
    if les.state == LESION_REPAIR:
        return True, False
    if les.ltype == LESION_TYPE_A:
        return (not rnapii_enabled), True
    return False, False


# ---------------------------------------------------------------------------
# Static setup (called once by the topology plugin)
# ---------------------------------------------------------------------------

def _tad_length_per_site(num_chains: int, chain_length: int, tad_positions):
    """Per-site TAD length and the per-chain TAD interval lengths.

    Boundaries are the chain-relative ``tad_positions`` plus the chain ends; a
    chain with no interior boundary is one TAD of length ``chain_length``. The
    pattern repeats on every chain. Returns ``(tad_len, intervals)`` where
    ``tad_len`` is float[N] (length of the TAD containing each site) and
    ``intervals`` is the per-chain list of distinct TAD lengths.
    """
    N = num_chains * chain_length
    tad_len = np.empty(N, dtype=np.float64)
    bounds = sorted({int(p) for p in (tad_positions or []) if 0 < int(p) < chain_length})
    edges = [0, *bounds, chain_length]
    for ci in range(num_chains):
        off = ci * chain_length
        for lo, hi in zip(edges[:-1], edges[1:]):
            tad_len[off + lo: off + hi] = float(hi - lo)
    intervals = np.diff(edges).astype(np.float64)
    return tad_len, intervals


def precompute_lesion_fields(
    args: Dict,
    *,
    tad_positions,
    gene_objs,
    tad_size_exponent: float,
    spacing: int,
    tad_repair_exponent: float = 0.0,
) -> None:
    """Precompute the static fields the spawner / state machine need (once at setup).

    Populates ``args`` with:

    * ``gene_body_mask``  -- bool[N], True at sites inside any gene body.
    * ``lesion_site_p``   -- float[N] (or ``None``), normalised placement weights;
      per-site spawn probability proportional to ``L_TAD ** (-tad_size_exponent)``
      (the ``L_ref`` constant cancels under normalisation, so it is omitted).
    * ``lesion_rate_mult`` -- float[N] (or ``None`` when ``tad_repair_exponent`` is 0),
      the per-site transition-rate multiplier ``(L_ref / L_TAD) ** tad_repair_exponent``
      with ``L_ref`` the mean TAD length. Both recognition (PRE->REPAIR) and repair
      (REPAIR->removed) rates are scaled by this, so lesions in shorter TADs are
      recognised and repaired faster.
    * ``lesion_target``   -- int, steady-state lesion count ``N // spacing``.
    """
    N = int(args["N"])
    num_chains = int(args.get("num_chains", 1))
    chain_length = int(args.get("chain_length", N))

    mask = np.zeros(N, dtype=bool)
    for g in gene_objs or []:
        lo, hi = (g.tss, g.tes) if g.tss <= g.tes else (g.tes, g.tss)
        mask[int(lo): int(hi) + 1] = True
    args["gene_body_mask"] = mask

    tad_len, intervals = _tad_length_per_site(num_chains, chain_length, tad_positions)
    weights = tad_len ** (-float(tad_size_exponent))
    total = float(weights.sum())
    args["lesion_site_p"] = (weights / total) if total > 0 else None

    beta = float(tad_repair_exponent)
    if beta != 0.0:
        l_ref = float(intervals.mean()) if intervals.size else float(chain_length)
        args["lesion_rate_mult"] = (l_ref / tad_len) ** beta
    else:
        args["lesion_rate_mult"] = None

    args["lesion_target"] = int(N // spacing) if spacing > 0 else 0


# ---------------------------------------------------------------------------
# Per-tick dynamics
# ---------------------------------------------------------------------------

def _assign_type(site: int, args: Dict) -> int:
    """Type for a new lesion at ``site``: in a gene body -> A with probability
    ``lesion_type_a_prob`` else B; off gene bodies -> always B."""
    mask = args.get("gene_body_mask")
    in_gene = bool(mask[site]) if mask is not None else False
    if in_gene and np.random.random() < float(args.get("lesion_type_a_prob", 0.5)):
        return LESION_TYPE_A
    return LESION_TYPE_B


def refill_lesions(args: Dict) -> None:
    """Spawn new lesions until the population reaches ``lesion_target``.

    Sites are sampled with probability ``lesion_site_p`` (TAD-size weighted) and
    placed in the PRE state with a type from :func:`_assign_type`. Already-lesioned
    draws are rejected (occupancy ~ 1/spacing, so rejection is cheap). The whole
    deficit is drawn per batch in one weighted ``np.random.choice`` call; the loop
    only repeats if collisions left a residual deficit.
    """
    target = int(args.get("lesion_target", 0))
    if target <= 0:
        return
    lesions: Dict[int, Lesion] = args.setdefault("lesions", {})
    if len(lesions) >= target:
        return
    N = int(args["N"])
    p = args.get("lesion_site_p")
    for _ in range(8):
        need = target - len(lesions)
        if need <= 0:
            break
        k = need + max(4, need // 2)  # headroom for collisions
        if p is not None:
            cand = np.random.choice(N, size=k, p=p)
        else:
            cand = np.random.randint(0, N, size=k)
        for site in cand:
            site = int(site)
            if site not in lesions:
                lesions[site] = Lesion(site, _assign_type(site, args), LESION_PRE)
                if len(lesions) >= target:
                    break


def update_lesions(args: Dict) -> None:
    """Advance lesion dynamics one tick: stochastic state transitions, remove
    repaired lesions, then refill the population to ``lesion_target``.

    Transitions are memory-less (geometric), parameterised in ticks like cohesin
    lifetime: PRE->REPAIR at base rate ``1/lesion_prerecognition_ticks`` and
    REPAIR->repaired at base rate ``1/lesion_repair_ticks``. Both rates are scaled
    per lesion by ``lesion_rate_mult[site]`` (clamped to <= 1), so lesions in
    shorter TADs are recognised and repaired faster; absent that array (no TAD
    speed modulation) the multiplier is 1. A repaired lesion is deleted here; an
    RNAPII stalled against it is evicted in the RNAPII phase
    (see :func:`rnapii.stateful_translocate_rnapii`), which runs after this.

    The initial population is seeded implicitly: on the first tick the dict is
    empty, transitions are a no-op, and the refill fills it to ``target``.
    """
    lesions: Dict[int, Lesion] = args.setdefault("lesions", {})
    if lesions:
        pre_rate = 1.0 / max(1, int(args.get("lesion_prerecognition_ticks", 1)))
        rep_rate = 1.0 / max(1, int(args.get("lesion_repair_ticks", 1)))
        rate_mult = args.get("lesion_rate_mult")  # float[N] or None (no modulation)
        u = np.random.random(len(lesions))
        to_remove: List[int] = []
        for i, les in enumerate(lesions.values()):
            m = rate_mult[les.site] if rate_mult is not None else 1.0
            if les.state == LESION_PRE:
                if u[i] < min(1.0, pre_rate * m):
                    les.state = LESION_REPAIR
            elif u[i] < min(1.0, rep_rate * m):
                to_remove.append(les.site)
        for site in to_remove:
            del lesions[site]
    refill_lesions(args)
