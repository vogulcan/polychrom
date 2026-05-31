"""Stage 2 driver: 3D molecular-dynamics simulation with dynamic SMC bonds.

Ported from ``examples/loopExtrusion/extrusion_3D.ipynb``.

The :class:`BondUpdater` is the same incremental bond-swapping helper used
in the notebook; the new :func:`run` wraps the original loop, makes the
force kit pluggable, and parameterises every magic number through
:class:`PolymerConfig`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

from ...hdf5_format import HDF5Reporter
from ...simulation import Simulation
from .config import PolymerConfig, resolve_plugin
from .plugins.rnapii import STATE_ELONGATING, STATE_STALLED
from .progress import ProgressMeter, log, stepped_run


def _gene_bodies(genes_dataset) -> dict:
    """Map gene_id -> set of (absolute) monomer indices spanning [TSS, TES].

    ``genes_dataset`` is the structured ``genes`` HDF5 dataset written by the
    1D stage (fields ``gene_id, tss, tes, ...``); coordinates are absolute,
    already replicated across chains by the 1D topology.
    """
    bodies: dict = {}
    for row in genes_dataset:
        gid = int(row["gene_id"])
        lo, hi = sorted((int(row["tss"]), int(row["tes"])))
        bodies.setdefault(gid, set()).update(range(lo, hi + 1))
    return bodies


class BondUpdater:
    """Incrementally activate/deactivate SMC bonds in an OpenMM context."""

    def __init__(self, lef_positions: np.ndarray):
        self.lef_positions = lef_positions
        self.cur_time = 0
        self.all_bonds: List[List[Tuple[int, int]]] = []
        self.cur_bonds: List[Tuple[int, int]] = []
        self.unique_bonds: List[Tuple[int, int]] = []
        self.bond_inds: List[int] = []
        self.bond_to_ind: dict = {}
        self.bond_force = None
        self.active_params: dict = {}
        self.inactive_params: dict = {}

    def set_params(self, active: dict, inactive: dict) -> None:
        self.active_params = active
        self.inactive_params = inactive

    def setup(self, bond_force, blocks: int = 100) -> Tuple[List[Tuple[int, int]], list]:
        if self.all_bonds:
            raise ValueError(f"Not all bonds were used; {len(self.all_bonds)} sets left")

        self.bond_force = bond_force
        loaded = self.lef_positions[self.cur_time : self.cur_time + blocks]
        self.all_bonds = [
            [(int(loaded[i, j, 0]), int(loaded[i, j, 1])) for j in range(loaded.shape[1])]
            for i in range(loaded.shape[0])
        ]
        self.unique_bonds = list({b for frame in self.all_bonds for b in frame})

        self.bond_inds = []
        self.cur_bonds = self.all_bonds.pop(0)
        for bond in self.unique_bonds:
            params = self.active_params if bond in self.cur_bonds else self.inactive_params
            ind = bond_force.addBond(bond[0], bond[1], **params)
            self.bond_inds.append(ind)
        self.bond_to_ind = dict(zip(self.unique_bonds, self.bond_inds))

        self.cur_time += blocks
        return self.cur_bonds, []

    def step(self, context, verbose: bool = False) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        if not self.all_bonds:
            raise ValueError("No bonds left; restart simulation and call setup() again")
        past = self.cur_bonds
        self.cur_bonds = self.all_bonds.pop(0)
        past_set = set(past)
        cur_set = set(self.cur_bonds)
        to_add = list(cur_set - past_set)
        to_remove = list(past_set - cur_set)
        if verbose:
            print(f"{len(past_set & cur_set)} stay, {len(to_add)} new, {len(to_remove)} removed")
        for bond in to_add:
            ind = self.bond_to_ind[bond]
            self.bond_force.setBondParameters(ind, bond[0], bond[1], **self.active_params)
        for bond in to_remove:
            ind = self.bond_to_ind[bond]
            self.bond_force.setBondParameters(ind, bond[0], bond[1], **self.inactive_params)
        self.bond_force.updateParametersInContext(context)
        return self.cur_bonds, past


class StickyUpdater:
    """Toggle Pol II self-affinity per frame to model transcription-driven
    compaction (Fursova & Larson 2024, Fig 2a).

    Mirrors :class:`BondUpdater`'s frame cursor. Each frame, the gene bodies of
    genes that currently carry at least one ELONGATING RNAPII are switched to
    the Pol II monomer type (self-attracting); all other transcribable monomers
    are switched back to the normal type. Type changes are applied via
    ``setParticleParameters`` on the heteropolymer nonbonded force, exactly the
    pattern :class:`BondUpdater` uses for SMC bonds.
    """

    def __init__(
        self,
        rnapii_positions: np.ndarray,
        rnapii_states: np.ndarray,
        gene_bodies: dict,
        *,
        polii_type: int,
        candidates: list,
        extra_hard: set,
    ):
        self.rnapii_positions = rnapii_positions
        self.rnapii_states = rnapii_states
        self.polii_type = float(polii_type)
        self.candidate_set = {int(m) for m in candidates}
        self.extra_hard = {int(m) for m in extra_hard}
        # Per-gene transcribable monomers (gene body restricted to candidates,
        # i.e. excluding any E/P sticky sites dropped by the force builder).
        self.gene_candidates = {
            gid: (set(body) & self.candidate_set) for gid, body in gene_bodies.items()
        }
        self.cur_time = 0
        self.force = None
        self.cur_active: set = set()
        self.pending: List[set] = []

    def _params(self, on: bool, monomer: int):
        type_id = self.polii_type if on else 0.0
        return (type_id, float(monomer in self.extra_hard))

    def _active_at(self, frame: int) -> set:
        pos = self.rnapii_positions[frame]
        states = self.rnapii_states[frame]
        active_genes = set()
        for k in range(pos.shape[0]):
            gid = int(pos[k, 1])
            if gid < 0:
                continue
            if int(states[k]) in (STATE_ELONGATING, STATE_STALLED):
                active_genes.add(gid)
        out: set = set()
        for gid in active_genes:
            out |= self.gene_candidates.get(gid, set())
        return out

    def setup(self, force, blocks: int, context=None) -> set:
        if self.pending:
            raise ValueError(f"Not all frames were used; {len(self.pending)} left")
        self.force = force
        # The force is freshly rebuilt each chunk with all candidates OFF, so
        # the live state resets to empty.
        self.cur_active = set()
        frames = [self._active_at(self.cur_time + i) for i in range(blocks)]
        first = frames[0]
        self.pending = frames[1:]
        for monomer in first:
            force.setParticleParameters(monomer, self._params(True, monomer))
        self.cur_active = set(first)
        if context is not None:
            force.updateParametersInContext(context)
        self.cur_time += blocks
        return self.cur_active

    def step(self, context) -> set:
        if not self.pending:
            raise ValueError("No frames left; restart simulation and call setup() again")
        nxt = self.pending.pop(0)
        to_add = nxt - self.cur_active
        to_remove = self.cur_active - nxt
        for monomer in to_add:
            self.force.setParticleParameters(monomer, self._params(True, monomer))
        for monomer in to_remove:
            self.force.setParticleParameters(monomer, self._params(False, monomer))
        if to_add or to_remove:
            self.force.updateParametersInContext(context)
        self.cur_active = nxt
        return self.cur_active


def run(cfg: PolymerConfig) -> Path:
    """Run the 3D MD simulation. Returns the output trajectory folder."""

    if cfg.seed is not None:
        np.random.seed(int(cfg.seed))

    plugins = cfg.plugins
    force_builder = resolve_plugin(plugins.force_builder)
    initial_conformation = resolve_plugin(plugins.initial_conformation)

    lef_file = h5py.File(cfg.lef_positions_path, "r")
    try:
        n_sites = int(lef_file.attrs["N"])
        n_frames = lef_file["positions"].shape[0]
        chain_length = int(lef_file.attrs.get("chain_length", n_sites))
        num_chains = int(lef_file.attrs.get("num_chains", 1))

        # Optional Pol II transcription-driven compaction (Fursova & Larson
        # 2024, Fig 2a). Enabled when the force builder is given a positive
        # polii_self_affinity AND the 1D stage recorded RNAPII + genes.
        fb_kwargs = dict(plugins.force_builder.kwargs)
        polii_feature = (
            float(fb_kwargs.get("polii_self_affinity", 0.0)) > 0.0
            and bool(lef_file.attrs.get("rnapii_enabled", False))
            and "genes" in lef_file
            and "rnapii_positions" in lef_file
            and "rnapii_states" in lef_file
        )
        gene_bodies: dict = {}
        if polii_feature:
            gene_bodies = _gene_bodies(lef_file["genes"])
            fb_kwargs["transcribed_particles"] = sorted(
                {m for body in gene_bodies.values() for m in body}
            )

        if n_frames % cfg.restart_every_blocks != 0:
            raise ValueError(
                "trajectory frames must be a multiple of restart_every_blocks "
                f"({n_frames} % {cfg.restart_every_blocks} != 0)"
            )







        if cfg.restart_every_blocks % cfg.save_every_blocks != 0:
            raise ValueError("restart_every_blocks must be a multiple of save_every_blocks")

        sim_inits_total = n_frames // cfg.restart_every_blocks

        box = (n_sites / cfg.density) ** (1.0 / 3.0)
        data = initial_conformation(num_sites=n_sites, box=box, **plugins.initial_conformation.kwargs)

        out_folder = Path(cfg.output_folder)
        out_folder.mkdir(parents=True, exist_ok=True)
        reporter = HDF5Reporter(
            folder=str(out_folder),
            max_data_length=cfg.max_data_length,
            overwrite=cfg.overwrite,
            blocks_only=False,
        )

        milker = BondUpdater(lef_file["positions"])
        sticky: "StickyUpdater | None" = None

        log.info(
            "[polymer] 3D MD: N=%d monomers, %d chains | %d frames = %d inits x %d "
            "blocks, %d MD steps/block | relax=%d, burn-in=%d steps | %s/%s",
            n_sites, num_chains, n_frames, sim_inits_total,
            cfg.restart_every_blocks, cfg.md_steps_per_block,
            cfg.initial_relaxation_steps, cfg.pre_recording_steps,
            cfg.platform, cfg.precision,
        )
        block_meter = ProgressMeter(n_frames, "polymer:record")

        for iteration in range(sim_inits_total):
            sim_kwargs = dict(
                platform=cfg.platform,
                integrator=cfg.integrator,
                error_tol=cfg.error_tol,
                GPU=cfg.gpu,
                collision_rate=cfg.collision_rate,
                N=len(data),
                reporters=[reporter],
                precision=cfg.precision,
            )
            if cfg.pbc:
                sim_kwargs["PBCbox"] = [box, box, box]
            sim = Simulation(**sim_kwargs)
            if cfg.seed is not None and hasattr(sim.integrator, "setRandomNumberSeed"):
                sim.integrator.setRandomNumberSeed(int(cfg.seed) + iteration)

            sim.set_data(data)
            force_builder(
                sim,
                num_chains=num_chains,
                chain_length=chain_length,
                **fb_kwargs,
            )

            if polii_feature and sticky is None:
                meta = getattr(sim, "polii_meta", None)
                if meta is not None and meta["candidates"]:
                    sticky = StickyUpdater(
                        lef_file["rnapii_positions"],
                        lef_file["rnapii_states"],
                        gene_bodies,
                        polii_type=meta["polii_type"],
                        candidates=meta["candidates"],
                        extra_hard=meta["extra_hard"],
                    )

            if iteration == 0 and cfg.initial_relaxation_steps > 0:
                # Paper-style: relax bare polymer (no cohesin bonds) before
                # inserting the first set of SMC bonds.
                log.info("[polymer] minimizing bare polymer energy...")
                sim.local_energy_minimization()
                log.info("[polymer] relaxing bare polymer: %d steps",
                         cfg.initial_relaxation_steps)
                stepped_run(sim.integrator.step, cfg.initial_relaxation_steps,
                            "polymer:relax")

            k_bond = sim.kbondScalingFactor / (cfg.smc_bond_wiggle ** 2)
            bond_dist = cfg.smc_bond_dist * sim.length_scale
            active = {"length": bond_dist, "k": k_bond}
            inactive = {"length": bond_dist, "k": 0}
            milker.set_params(active, inactive)
            milker.setup(
                bond_force=sim.force_dict["harmonic_bonds"],
                blocks=cfg.restart_every_blocks,
            )
            if sticky is not None:
                meta = sim.polii_meta
                sticky.setup(
                    sim.force_dict[meta["force_name"]],
                    blocks=cfg.restart_every_blocks,
                    context=getattr(sim, "context", None),
                )
            if sim.forces_applied:
                # Initial bare-polymer relaxation creates an OpenMM context
                # before dynamic SMC bonds exist. Reinitialize so the context
                # sees the expanded HarmonicBondForce bond list.
                sim.context.reinitialize(preserveState=True)

            if iteration == 0:
                if cfg.initial_relaxation_steps > 0:
                    sim._apply_forces()
                    sim.local_energy_minimization()
                else:
                    sim.local_energy_minimization()
            else:
                sim._apply_forces()

            if iteration == 0 and cfg.pre_recording_steps > 0:
                # Run with cohesin bonds but skip block-level recording.
                log.info("[polymer] burn-in with SMC bonds: %d steps",
                         cfg.pre_recording_steps)
                stepped_run(sim.integrator.step, cfg.pre_recording_steps,
                            "polymer:burn-in")

            for i in range(cfg.restart_every_blocks):
                if i % cfg.save_every_blocks == (cfg.save_every_blocks - 1):
                    sim.do_block(steps=cfg.md_steps_per_block)
                else:
                    sim.integrator.step(cfg.md_steps_per_block)
                block_meter.update(iteration * cfg.restart_every_blocks + i + 1)
                if i < cfg.restart_every_blocks - 1:
                    milker.step(sim.context)
                    if sticky is not None:
                        sticky.step(sim.context)

            data = sim.get_data()
            del sim
            reporter.blocks_only = True
            time.sleep(0.2)

        block_meter.done()
        reporter.dump_data()
        return out_folder
    finally:
        lef_file.close()
