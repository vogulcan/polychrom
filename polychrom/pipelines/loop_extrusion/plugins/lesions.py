"""DNA-lesion (UV-damage) dynamics on the 1D loop-extrusion lattice.

A lesion is a damaged lattice site inside a gene body. It is a standalone DNA
feature (independent of who occupies the site) that:

* **stalls an elongating RNAPII** trying to step onto it -- the polymerase
  cannot transcribe through the damage and piles up just upstream
  (lesion-stalled Pol II, the transcription-coupled-repair trigger);
* **blocks a cohesin leg** trying to step onto it, with probability
  ``lesion_block_prob`` (strong / near-stable). Because each leg steps
  independently, the *other* leg keeps extruding -- so the cohesin extrudes
  **asymmetrically** while one side is pinned at the lesion.

Lesions arise stochastically (``lesion_prob`` per gene per tick), live for
``lesion_lifetime`` ticks (countdown to repair), and are capped at
``lesion_max`` simultaneous lesions PER CHAIN (total cap is
``lesion_max * num_chains``).

State lives in ``args["lesions"]``: ``Dict[int, int]`` mapping lesion site to
its remaining lifetime in ticks.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def is_lesion(site: int, args: Dict) -> bool:
    """True if ``site`` currently carries a lesion."""
    return int(site) in args.get("lesions", {})


def seed_periodic_lesions(args: Dict, spacing: int, lifetime: int) -> int:
    """Pre-place lesions at every ``spacing``-th monomer in each gene body.

    Models a UV pulse that deposits CPDs at regular intervals along transcribed
    DNA. Lesions get ``lifetime`` ticks until repair (set very high for
    persistent damage over the trajectory). Returns the number of lesions seeded.
    """
    lesions: Dict[int, int] = args.setdefault("lesions", {})
    if spacing <= 0:
        return 0
    seeded = 0
    for gene in args.get("genes", []):
        lo, hi = (gene.tss, gene.tes) if gene.tss <= gene.tes else (gene.tes, gene.tss)
        for site in range(lo, hi + 1, spacing):
            if site not in lesions:
                lesions[site] = int(lifetime)
                seeded += 1
    return seeded


def update_lesions(args: Dict) -> None:
    """Advance lesion state one tick: repair existing, then spawn new ones.

    Repair runs first (countdown each lesion's lifetime, drop expired), then
    occurrence places at most one new lesion per gene with probability
    ``lesion_prob`` at a uniform-random site in that gene's body.
    """
    lesions: Dict[int, int] = args.setdefault("lesions", {})

    # 1. Repair: countdown, remove fully-repaired lesions.
    for site in list(lesions):
        lesions[site] -= 1
        if lesions[site] <= 0:
            del lesions[site]

    # 2. Occurrence in gene bodies.
    prob = float(args.get("lesion_prob", 0.0))
    if prob <= 0.0:
        return
    lifetime = max(1, int(args.get("lesion_lifetime", 1)))
    # lesion_max is per-chain; total simultaneous cap scales with num_chains.
    lmax = int(args.get("lesion_max", 64)) * int(args.get("num_chains", 1))
    for gene in args.get("genes", []):
        if len(lesions) >= lmax:
            break
        if np.random.random() >= prob:
            continue
        lo, hi = (gene.tss, gene.tes) if gene.tss <= gene.tes else (gene.tes, gene.tss)
        site = int(np.random.randint(lo, hi + 1))
        if site not in lesions:
            lesions[site] = lifetime
