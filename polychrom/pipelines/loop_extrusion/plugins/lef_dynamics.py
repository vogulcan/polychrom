"""Default LEF mechanics: load/unload, CTCF capture/release, translocation.

Ported from ``examples/loopExtrusion/extrusion_1D_newCode.ipynb``.

The :class:`Leg` and :class:`Cohesin` classes are the same data structures
used in the notebook; the four core functions accept an ``args`` dict so a
user can supply alternative implementations with the same signature.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


FREE = 0
COHESIN = 1
RNAPII_CELL = 2
STATE_POISED = 0
STATE_PAUSED = 1
STATE_ELONGATING = 2


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
    """Per-step unload probability. Stalled (not at CTCF) cohesins die faster."""
    if cohesin.any("stalled") and not cohesin.any("CTCF"):
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
    tes_by_pos: Dict[int, int] = args.get("tes_by_pos", {})
    rnapii_by_pos = args.get("rnapii_by_pos", {})

    # 1. Unload + reload.
    for i in range(len(cohesins) - 1, -1, -1):
        prob = unload_prob_fn(cohesins[i], args)
        if np.random.random() < prob:
            for side in (-1, 1):
                leg_by_pos.pop(cohesins[i][side].pos, None)
            occupied[cohesins[i].left.pos] = FREE
            occupied[cohesins[i].right.pos] = FREE
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
                continue

            cell = occupied[target]

            if cell == FREE:
                leg_by_pos.pop(leg.pos, None)
                leg.attrs["stalled"] = False
                leg.attrs["pushed"] = False
                occupied[leg.pos] = FREE
                occupied[target] = COHESIN
                leg.pos = target
                leg_by_pos[target] = leg
                continue

            if cell == COHESIN:
                leg.attrs["stalled"] = True
                continue

            # cell == RNAPII_CELL
            r = rnapii_by_pos.get(target)
            if r is not None and target in tes_by_pos and tes_by_pos[target] == r.gene_id:
                # RNAPII parked at its own TES -> bypass.
                # Step onto the TES site; both markers coexist until the
                # RNAPII unloads next tick (RNAPII translocate phase
                # handles the cleanup and won't overwrite COHESIN).
                leg_by_pos.pop(leg.pos, None)
                leg.attrs["stalled"] = False
                occupied[leg.pos] = FREE
                occupied[target] = COHESIN  # bypass: cohesin takes the slot
                leg.pos = target
                leg_by_pos[target] = leg
            elif r is not None and not _rnapii_blocks_cohesin(r, args):
                # Probabilistic bypass: keep the RNAPII bookkeeping entry,
                # but the occupancy grid marks the cohesin leg at this site.
                # When RNAPII advances later, it will leave the cohesin marker
                # in place because occupied[r.pos] is no longer RNAPII_CELL.
                leg_by_pos.pop(leg.pos, None)
                leg.attrs["stalled"] = False
                occupied[leg.pos] = FREE
                occupied[target] = COHESIN
                leg.pos = target
                leg_by_pos[target] = leg
            else:
                leg.attrs["stalled"] = True


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
