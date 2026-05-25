"""Default polymer force builder + initial conformation generator.

Replace these with custom callables to swap in different physics (e.g.,
different chain layouts, alternative repulsive potentials, multi-block
copolymers, etc.). Plugin signatures:

    force_builder(sim, *, num_chains, chain_length, **kwargs) -> None
    initial_conformation(*, num_sites, box, **kwargs) -> np.ndarray
"""

from __future__ import annotations

import numpy as np

from .... import forcekits, forces
from ....starting_conformations import grow_cubic


def default_force_builder(
    sim,
    *,
    num_chains: int,
    chain_length: int,
    bond_length: float = 1.0,
    bond_wiggle: float = 0.1,
    angle_k: float = 1.5,
    repulsive_trunc: float = 1.5,
    repulsive_radius_mult: float = 1.05,
) -> None:
    """Build a polymer with harmonic bonds, angles, polynomial repulsion.

    Each chain is added as a separate, non-ring segment. Mirrors the force
    setup in ``extrusion_3D.ipynb`` but generalised to N chains.
    """
    chains = [
        (chain_idx * chain_length, (chain_idx + 1) * chain_length, False)
        for chain_idx in range(num_chains)
    ]
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=chains,
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": bond_length,
                "bondWiggleDistance": bond_wiggle,
            },
            angle_force_func=forces.angle_force,
            angle_force_kwargs={"k": angle_k},
            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                "trunc": repulsive_trunc,
                "radiusMult": repulsive_radius_mult,
            },
            except_bonds=True,
        )
    )


def grow_cubic_conformation(*, num_sites: int, box: float, **_: object) -> np.ndarray:
    """Default starting conformation: one compact polymer in a cubic box."""
    return grow_cubic(num_sites, int(box) - 2)


def paper_force_builder(
    sim,
    *,
    num_chains: int,
    chain_length: int,
    sticky_particles: list = (),
    ep_pairs: list = (),
    extra_hard_particles: list = (),
    bond_length: float = 1.0,
    bond_wiggle: float = 0.1,
    angle_k=None,
    repulsion_energy: float = 50.0,
    repulsion_radius: float = 1.05,
    attraction_energy: float = 0.0,
    attraction_radius: float = 2.0,
    selective_attraction_energy: float = 1.0,
    selective_repulsion_energy: float = 0.0,
    confinement_density: float = 0.2,
    confinement_k: float = 5.0,
) -> None:
    """Force kit from Nat. Rev. Mol. Cell Biol. supplementary box 1.

    Harmonic bonds (k from ``bond_wiggle``), angle force, selective_SSW
    with E/P-pair sticky particles, and spherical confinement at the
    requested DNA volume fraction.

    ``sticky_particles`` is the flat list of E and P monomer indices
    (14 ints for 7 cognate E-P pairs).
    """
    chains = [
        (chain_idx * chain_length, (chain_idx + 1) * chain_length, False)
        for chain_idx in range(num_chains)
    ]

    if ep_pairs:
        monomer_types = np.zeros(sim.N, dtype=int)
        interaction_matrix = np.zeros((len(ep_pairs) + 1, len(ep_pairs) + 1), dtype=float)
        for pair_idx, pair in enumerate(ep_pairs, start=1):
            if len(pair) != 2:
                raise ValueError(f"ep_pairs entries must have two monomers, got {pair!r}")
            enhancer, promoter = (int(pair[0]), int(pair[1]))
            monomer_types[enhancer] = pair_idx
            monomer_types[promoter] = pair_idx
            interaction_matrix[pair_idx, pair_idx] = 1.0
        nonbonded_force_func = forces.heteropolymer_SSW
        nonbonded_force_kwargs = {
            "interactionMatrix": interaction_matrix,
            "monomerTypes": monomer_types,
            "extraHardParticlesIdxs": list(extra_hard_particles),
            "repulsionEnergy": repulsion_energy,
            "repulsionRadius": repulsion_radius,
            "attractionEnergy": attraction_energy,
            "attractionRadius": attraction_radius,
            "selectiveAttractionEnergy": selective_attraction_energy,
            "selectiveRepulsionEnergy": selective_repulsion_energy,
        }
    else:
        nonbonded_force_func = forces.selective_SSW
        nonbonded_force_kwargs = {
            "stickyParticlesIdxs": list(sticky_particles),
            "extraHardParticlesIdxs": list(extra_hard_particles),
            "repulsionEnergy": repulsion_energy,
            "repulsionRadius": repulsion_radius,
            "attractionEnergy": attraction_energy,
            "attractionRadius": attraction_radius,
            "selectiveAttractionEnergy": selective_attraction_energy,
            "selectiveRepulsionEnergy": selective_repulsion_energy,
        }

    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=chains,
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": bond_length,
                "bondWiggleDistance": bond_wiggle,
            },
            angle_force_func=None if angle_k is None else forces.angle_force,
            angle_force_kwargs={} if angle_k is None else {"k": angle_k},
            nonbonded_force_func=nonbonded_force_func,
            nonbonded_force_kwargs=nonbonded_force_kwargs,
            except_bonds=True,
        )
    )

    sim.add_force(
        forces.spherical_confinement(
            sim,
            density=confinement_density,
            k=confinement_k,
        )
    )
