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
    restrict_nonbonded_to_chains: bool = False,
) -> None:
    """Build a polymer with harmonic bonds, angles, polynomial repulsion.

    Each chain is added as a separate, non-ring segment. Mirrors the force
    setup in ``extrusion_3D.ipynb`` but generalised to N chains.
    """
    chains = [
        (chain_idx * chain_length, (chain_idx + 1) * chain_length, False)
        for chain_idx in range(num_chains)
    ]
    nonbonded_force_func = forces.polynomial_repulsive
    if restrict_nonbonded_to_chains:
        nonbonded_force_func = _chain_restricted_nonbonded(
            nonbonded_force_func,
            num_chains=num_chains,
            chain_length=chain_length,
        )

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
            nonbonded_force_func=nonbonded_force_func,
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


def _expand_ep_pairs_across_chains(
    ep_pairs: list,
    *,
    num_chains: int,
    chain_length: int,
    replicate_ep_pairs_across_chains: bool,
) -> list:
    """Expand chain-relative E-P pairs to absolute monomer indices."""
    base = []
    for pair in ep_pairs:
        if len(pair) != 2:
            raise ValueError(f"ep_pairs entries must have two monomers, got {pair!r}")
        base.append((int(pair[0]), int(pair[1])))

    if not replicate_ep_pairs_across_chains:
        return base

    expanded = []
    for chain_idx in range(num_chains):
        offset = chain_idx * chain_length
        for enhancer, promoter in base:
            for site in (enhancer, promoter):
                if not (0 <= site < chain_length):
                    raise ValueError(
                        f"ep_pairs monomer index {site} is outside one chain of "
                        f"length {chain_length}; disable replicate_ep_pairs_across_chains "
                        "for absolute coordinates"
                    )
            expanded.append((enhancer + offset, promoter + offset))
    return expanded


def _add_chain_interaction_groups(
    nonbonded_force,
    *,
    num_chains: int,
    chain_length: int,
):
    """Restrict a nonbonded force to intra-chain pairs only."""
    if not hasattr(nonbonded_force, "addInteractionGroup"):
        raise TypeError(
            f"{getattr(nonbonded_force, 'name', type(nonbonded_force).__name__)} "
            "does not support interaction groups"
        )
    for chain_idx in range(num_chains):
        start = chain_idx * chain_length
        group = set(range(start, start + chain_length))
        nonbonded_force.addInteractionGroup(group, group)
    return nonbonded_force


def _chain_restricted_nonbonded(force_func, *, num_chains: int, chain_length: int):
    def build(sim, **kwargs):
        force = force_func(sim, **kwargs)
        return _add_chain_interaction_groups(
            force,
            num_chains=num_chains,
            chain_length=chain_length,
        )

    return build


def _spherical_confinement_for_replicates(
    sim,
    *,
    num_chains: int,
    chain_length: int,
    density: float,
    k: float,
):
    """Add one same-density spherical confinement per replicate chain."""
    radius = (3 * chain_length / (4 * np.pi * density)) ** (1.0 / 3.0)
    return [
        forces.spherical_confinement(
            sim,
            r=radius,
            k=k,
            particles=range(chain_idx * chain_length, (chain_idx + 1) * chain_length),
            name=f"spherical_confinement_chain_{chain_idx}",
        )
        for chain_idx in range(num_chains)
    ]


def paper_force_builder(
    sim,
    *,
    num_chains: int,
    chain_length: int,
    sticky_particles: list = (),
    ep_pairs: list = (),
    replicate_ep_pairs_across_chains: bool = False,
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
    restrict_nonbonded_to_chains: bool = False,
    confinement_per_chain: bool = False,
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

    expanded_ep_pairs = _expand_ep_pairs_across_chains(
        ep_pairs,
        num_chains=num_chains,
        chain_length=chain_length,
        replicate_ep_pairs_across_chains=replicate_ep_pairs_across_chains,
    )

    if expanded_ep_pairs:
        sticky_sites = sorted({site for pair in expanded_ep_pairs for site in pair})
        site_to_type = {site: idx for idx, site in enumerate(sticky_sites, start=1)}
        monomer_types = np.zeros(sim.N, dtype=int)
        for site, site_type in site_to_type.items():
            if not (0 <= site < sim.N):
                raise ValueError(f"ep_pairs monomer index {site} is outside polymer length {sim.N}")
            monomer_types[site] = site_type

        interaction_matrix = np.zeros((len(sticky_sites) + 1, len(sticky_sites) + 1), dtype=float)
        for enhancer, promoter in expanded_ep_pairs:
            enhancer_type = site_to_type[enhancer]
            promoter_type = site_to_type[promoter]
            interaction_matrix[enhancer_type, promoter_type] = 1.0
            interaction_matrix[promoter_type, enhancer_type] = 1.0
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

    if restrict_nonbonded_to_chains:
        nonbonded_force_func = _chain_restricted_nonbonded(
            nonbonded_force_func,
            num_chains=num_chains,
            chain_length=chain_length,
        )

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

    if confinement_per_chain:
        sim.add_force(
            _spherical_confinement_for_replicates(
                sim,
                num_chains=num_chains,
                chain_length=chain_length,
                density=confinement_density,
                k=confinement_k,
            )
        )
    else:
        sim.add_force(
            forces.spherical_confinement(
                sim,
                density=confinement_density,
                k=confinement_k,
            )
        )
