"""Default LEF mechanics: load/unload, CTCF capture/release, translocation.

Ported from ``examples/loopExtrusion/extrusion_1D_newCode.ipynb``.

The :class:`Leg` and :class:`Cohesin` classes are the same data structures
used in the notebook; the four core functions accept an ``args`` dict so a
user can supply alternative implementations with the same signature.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .lesions import lesion_stalls_cohesin


FREE = 0
COHESIN = 1
RNAPII_CELL = 2
STATE_POISED = 0
STATE_PAUSED = 1
STATE_ELONGATING = 2
STATE_TERMINATING = 3
STATE_STALLED = 4


class Leg:
    """One end of a cohesin."""

    __slots__ = ("pos", "attrs")

    def __init__(self, pos: int, attrs: Dict[str, bool] | None = None):
        self.pos = pos
        self.attrs = dict(attrs) if attrs else {"stalled": False, "CTCF": False}


class Cohesin:
    """Two legs that share fate and unloading probability."""

    __slots__ = ("left", "right")

    def __init__(self, left: Leg, right: Leg):
        self.left = left
        self.right = right
        # Per-leg extrusion direction, consumed by the RNAPII head-on
        # resolver: the left leg extrudes toward decreasing coordinates
        # (-1), the right toward increasing (+1). Stamped on every cohesin
        # (initial load and reloads) so orientation is always available.
        left.attrs["dir"] = -1
        right.attrs["dir"] = 1

    def any(self, attr: str) -> bool:
        return self.left.attrs.get(attr, False) or self.right.attrs.get(attr, False)

    def all(self, attr: str) -> bool:
        return self.left.attrs.get(attr, False) and self.right.attrs.get(attr, False)

    def __getitem__(self, item: int) -> Leg:
        if item == -1:
            return self.left
        if item == 1:
            return self.right
        raise ValueError("Cohesin side must be -1 (left) or 1 (right)")


# ---------------------------------------------------------------------------
# Default mechanic plugins
# ---------------------------------------------------------------------------

def _chain_length(args: Dict) -> int:
    return int(args.get("chain_length", args["N"]))


def _same_chain(a: int, b: int, args: Dict) -> bool:
    chain_length = _chain_length(args)
    return int(a) // chain_length == int(b) // chain_length


def _valid_step(pos: int, target: int, args: Dict) -> bool:
    return 0 <= target < args["N"] and _same_chain(pos, target, args)


def unload_prob(cohesin: Cohesin, args: Dict) -> float:
    """Per-step unload probability. Stalled (not at CTCF) cohesins die faster.

    A cohesin stalled or pushed *by RNAPII* (``rnapii_stalled`` flag) unloads at
    the transcription-specific rate ``LIFETIME_RNAPII_STALLED`` -- modelling
    active Pol II eviction of cohesin (Busslinger 2017; Jeppsson 2022). This is
    kept separate from generic obstacle stalling (cohesin-cohesin traffic, lesion)
    so depleting RNAPII restores full residence ONLY where transcription was the
    cause, instead of globally shrinking loops.
    """
    if cohesin.any("CTCF"):
        return 1.0 / args.get("LIFETIME_CTCF", args["LIFETIME"])
    if cohesin.any("rnapii_stalled"):
        return 1.0 / args.get("LIFETIME_RNAPII_STALLED", args["LIFETIME_STALLED"])
    if cohesin.any("stalled"):
        return 1.0 / args["LIFETIME_STALLED"]
    return 1.0 / args["LIFETIME"]


def load_one(cohesins: List[Cohesin], occupied: np.ndarray, args: Dict) -> None:
    """Randomly place one cohesin onto two adjacent same-chain empty sites."""
    n = args["N"]
    for _ in range(max(100, 2 * n)):
        a = np.random.randint(n - 1)
        if not _same_chain(a, a + 1, args):
            continue
        if occupied[a] == FREE and occupied[a + 1] == FREE:
            occupied[a] = COHESIN
            occupied[a + 1] = COHESIN
            cohesins.append(Cohesin(Leg(a), Leg(a + 1)))
            return
    for a in range(n - 1):
        if not _same_chain(a, a + 1, args):
            continue
        if occupied[a] == FREE and occupied[a + 1] == FREE:
            occupied[a] = COHESIN
            occupied[a + 1] = COHESIN
            cohesins.append(Cohesin(Leg(a), Leg(a + 1)))
            return
    raise RuntimeError("No same-chain adjacent empty sites available for cohesin loading")


def _try_place(cohesins: List[Cohesin], occupied: np.ndarray, args: Dict, a: int) -> bool:
    """Place a cohesin on the adjacent pair ``(a, a+1)`` if both free + same chain."""
    n = args["N"]
    if 0 <= a and a + 1 < n and _same_chain(a, a + 1, args) \
            and occupied[a] == FREE and occupied[a + 1] == FREE:
        occupied[a] = COHESIN
        occupied[a + 1] = COHESIN
        cohesins.append(Cohesin(Leg(a), Leg(a + 1)))
        return True
    return False


def load_targeted(cohesins: List[Cohesin], occupied: np.ndarray, args: Dict) -> None:
    """Bias cohesin loading toward active enhancers / TSS (targeted loading).

    Models NIPBL/MAU2-mediated targeted cohesin loading at active enhancers
    (Fursova & Larson 2024, Fig 4b): with probability ``targeted_load_prob`` the
    cohesin is placed at a free adjacent pair within ``loading_window`` sites of
    a random loading site (``args["loading_sites"]`` -- enhancers/TSS populated
    by the topology). Otherwise, or if every nearby slot is occupied, it falls
    back to uniform :func:`load_one`. With no loading sites or
    ``targeted_load_prob == 0`` this is identical to :func:`load_one`.
    """
    sites = args.get("loading_sites") or []
    p = float(args.get("targeted_load_prob", 0.0))
    if sites and np.random.random() < p:
        window = int(args.get("loading_window", 2))
        # Activity-weighted pick (NIPBL prefers active enhancers/promoters); falls
        # back to uniform when no weights were supplied.
        probs = args.get("loading_probs")
        idx = (
            int(np.random.choice(len(sites), p=probs))
            if probs is not None
            else np.random.randint(len(sites))
        )
        site = int(sites[idx])
        # Search outward from the target: (site, site+1), then +/- offsets.
        # `_same_chain(a, site)` keeps the placement inside the target site's own
        # chain -- a site near a chain boundary must never spill the cohesin into
        # a neighbouring replicate chain.
        for off in range(window + 1):
            for a in ((site + off, site - off) if off else (site + off,)):
                if _same_chain(a, site, args) and _try_place(cohesins, occupied, args, a):
                    return
        # Target neighbourhood fully occupied -> fall back to uniform.
    load_one(cohesins, occupied, args)


def capture(cohesin: Cohesin, occupied: np.ndarray, args: Dict) -> Cohesin:
    """Stochastically capture each leg at its side's CTCF site."""
    for side in (-1, 1):
        prob = args["ctcfCapture"][side].get(cohesin[side].pos, 0.0)
        if np.random.random() < prob:
            cohesin[side].attrs["CTCF"] = True
    return cohesin


def release(cohesin: Cohesin, occupied: np.ndarray, args: Dict) -> Cohesin:
    """Stochastically release each captured leg."""
    if not cohesin.any("CTCF"):
        return cohesin
    for side in (-1, 1):
        prob = args["ctcfRelease"][side].get(cohesin[side].pos, 0.0)
        if cohesin[side].attrs["CTCF"] and np.random.random() < prob:
            cohesin[side].attrs["CTCF"] = False
    return cohesin


def translocate(
    cohesins: List[Cohesin],
    occupied: np.ndarray,
    args: Dict,
    *,
    unload_prob_fn=unload_prob,
    load_fn=load_one,
    capture_fn=capture,
    release_fn=release,
) -> None:
    """Drive one step of LEF dynamics. Accepts injectable per-mechanic hooks."""
    # 1. Unload + reload to maintain constant LEF count.
    for i in range(len(cohesins) - 1, -1, -1):
        prob = unload_prob_fn(cohesins[i], args)
        if np.random.random() < prob:
            occupied[cohesins[i].left.pos] = 0
            occupied[cohesins[i].right.pos] = 0
            del cohesins[i]
            load_fn(cohesins, occupied, args)
            # Reload into slot i so each list slot tracks one cohesin over time
            # (stable trajectory identity for visualisation); reordering would
            # otherwise scramble per-slot tracks. Bond set per frame unchanged.
            cohesins.insert(i, cohesins.pop())

    # 2. CTCF capture / release.
    for i in range(len(cohesins)):
        cohesins[i] = capture_fn(cohesins[i], occupied, args)
        cohesins[i] = release_fn(cohesins[i], occupied, args)

    # 3. Translocation step.
    for i in range(len(cohesins)):
        coh = cohesins[i]
        for side in (-1, 1):
            leg = coh[side]
            if leg.attrs["CTCF"]:
                continue
            target = leg.pos + side
            if not _valid_step(leg.pos, target, args) or occupied[target] != FREE:
                leg.attrs["stalled"] = True
                continue
            leg.attrs["stalled"] = False
            occupied[leg.pos] = FREE
            occupied[target] = COHESIN
            leg.pos = target


def _rnapii_blocks_cohesin(rnapii, args: Dict) -> bool:
    """Return whether an RNAPII body blocks an incoming cohesin leg."""
    state = rnapii.attrs.get("state", STATE_POISED)
    if state == STATE_POISED:
        prob = float(args.get("rnapii_poised_block_prob", 1.0))
    elif state == STATE_PAUSED:
        prob = float(
            args.get("rnapii_paused_block_prob", args.get("rnapii_block_prob", 1.0))
        )
    elif state == STATE_ELONGATING:
        prob = float(
            args.get("rnapii_elongating_block_prob", args.get("rnapii_block_prob", 1.0))
        )
    elif state == STATE_STALLED:
        # Stalled Pol II is physically the same body as elongating, just not
        # advancing this tick; block cohesin at the elongating rate.
        prob = float(
            args.get("rnapii_elongating_block_prob", args.get("rnapii_block_prob", 1.0))
        )
    elif state == STATE_TERMINATING:
        # Terminating Pol II is a stationary block (like paused); default to the
        # paused block prob if a terminating-specific one isn't given.
        prob = float(
            args.get(
                "rnapii_terminating_block_prob",
                args.get("rnapii_paused_block_prob", args.get("rnapii_block_prob", 1.0)),
            )
        )
    else:
        prob = float(args.get("rnapii_block_prob", 1.0))
    return np.random.random() < prob


def translocate_with_rnapii(
    cohesins: List[Cohesin],
    occupied: np.ndarray,
    args: Dict,
    *,
    unload_prob_fn=unload_prob,
    load_fn=load_one,
    capture_fn=capture,
    release_fn=release,
) -> None:
    """Cohesin dynamics aware of RNAPII bodies on the lattice.

    Identical to :func:`translocate` except the per-leg step also handles
    ``occupied[target] == 2`` (RNAPII): cohesin bypasses RNAPII when the
    RNAPII sits on its gene's TES, otherwise it stalls with state-dependent
    RNAPII blocking probabilities. Maintains
    ``args["cohesin_leg_by_pos"]`` so the RNAPII translocate phase can
    push specific legs when needed.
    """
    leg_by_pos: Dict[int, "Leg"] = args.setdefault("cohesin_leg_by_pos", {})
    rnapii_by_pos = args.get("rnapii_by_pos", {})

    def _vacate(p: int) -> None:
        # A cohesin leg leaving site ``p``: if an RNAPII still occupies that site
        # (the leg had bypassed/tunnelled through it), restore the RNAPII marker
        # instead of clearing to FREE -- otherwise a stationary Pol II (e.g. one
        # terminating at its TES) loses its occupancy and another Pol stacks on it.
        occupied[p] = RNAPII_CELL if p in rnapii_by_pos else FREE
    lesions = args.get("lesions")
    lesion_block_prob = float(args.get("lesion_block_prob", 1.0))
    rnapii_enabled = bool(args.get("rnapii_enabled", False))

    # 1. Unload + reload.
    for i in range(len(cohesins) - 1, -1, -1):
        prob = unload_prob_fn(cohesins[i], args)
        if np.random.random() < prob:
            for side in (-1, 1):
                leg_by_pos.pop(cohesins[i][side].pos, None)
            _vacate(cohesins[i].left.pos)
            _vacate(cohesins[i].right.pos)
            del cohesins[i]
            load_fn(cohesins, occupied, args)
            # Newly loaded cohesin: register its leg positions.
            new = cohesins[-1]
            leg_by_pos[new.left.pos] = new.left
            leg_by_pos[new.right.pos] = new.right
            # Reload into slot i to keep per-slot trajectory identity stable.
            cohesins.insert(i, cohesins.pop())

    # 2. CTCF capture / release.
    for i in range(len(cohesins)):
        cohesins[i] = capture_fn(cohesins[i], occupied, args)
        cohesins[i] = release_fn(cohesins[i], occupied, args)

    # 3. Per-leg translocation step.
    for coh in cohesins:
        for side in (-1, 1):
            leg = coh[side]
            if leg.attrs.get("CTCF"):
                continue
            target = leg.pos + side
            if not _valid_step(leg.pos, target, args):
                leg.attrs["stalled"] = True
                leg.attrs["rnapii_stalled"] = False
                continue

            # A lesion may stall an incoming cohesin leg, depending on its type
            # and state (see lesions.lesion_stalls_cohesin). Type-A pre-recognition
            # lesions defer to a real stalled Pol II when RNAPII is in the model
            # (rnapii_evict -> Pol II-style fast eviction); repair-state lesions
            # stall generically. The other leg is unaffected, so the cohesin
            # extrudes asymmetrically.
            if lesions:
                les = lesions.get(target)
                if les is not None:
                    stalls, rnapii_evict = lesion_stalls_cohesin(les, rnapii_enabled)
                    if stalls and np.random.random() < lesion_block_prob:
                        leg.attrs["stalled"] = True
                        leg.attrs["rnapii_stalled"] = rnapii_evict
                        continue

            cell = occupied[target]

            if cell == FREE:
                leg_by_pos.pop(leg.pos, None)
                leg.attrs["stalled"] = False
                leg.attrs["pushed"] = False
                leg.attrs["rnapii_stalled"] = False
                _vacate(leg.pos)
                occupied[target] = COHESIN
                leg.pos = target
                leg_by_pos[target] = leg
                continue

            if cell == COHESIN:
                leg.attrs["stalled"] = True
                leg.attrs["rnapii_stalled"] = False
                continue

            # cell == RNAPII_CELL
            r = rnapii_by_pos.get(target)
            if r is not None and not _rnapii_blocks_cohesin(r, args):
                # Probabilistic bypass: keep the RNAPII bookkeeping entry,
                # but the occupancy grid marks the cohesin leg at this site.
                # When RNAPII advances later, it will leave the cohesin marker
                # in place because occupied[r.pos] is no longer RNAPII_CELL.
                leg_by_pos.pop(leg.pos, None)
                leg.attrs["stalled"] = False
                leg.attrs["rnapii_stalled"] = False
                _vacate(leg.pos)
                occupied[target] = COHESIN
                leg.pos = target
                leg_by_pos[target] = leg
            else:
                # Blocked by RNAPII: stalled by transcription -> eligible for
                # Pol II-specific fast eviction (LIFETIME_RNAPII_STALLED).
                leg.attrs["stalled"] = True
                leg.attrs["rnapii_stalled"] = True


def color(cohesins: List[Cohesin], args: Dict) -> np.ndarray:
    """Helper for visualisation: 1 free, 2 stalled, 3 at CTCF."""
    def state(attrs: Dict[str, bool]) -> int:
        if attrs.get("stalled"):
            return 2
        if attrs.get("CTCF"):
            return 3
        return 1

    ar = np.zeros(args["N"])
    for c in cohesins:
        ar[c.left.pos] = state(c.left.attrs)
        ar[c.right.pos] = state(c.right.attrs)
    return ar
