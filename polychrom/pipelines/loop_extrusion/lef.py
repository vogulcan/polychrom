"""Stage 1 driver: 1D LEF (+ optional RNAPII) dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import List

import h5py
import numpy as np

from .config import LEFConfig, resolve_plugin
from .plugins.lef_dynamics import Cohesin
from .plugins.rnapii import compute_ep_contacts
from .progress import ProgressMeter, log


def initial_state(cfg: LEFConfig, args: dict, load_fn) -> tuple[np.ndarray, List[Cohesin]]:
    """Build occupancy + cohesin list."""
    occupied = np.zeros(cfg.num_sites, dtype=np.int8)

    cohesins: List[Cohesin] = []
    for _ in range(cfg.num_lefs):
        load_fn(cohesins, occupied, args)

    # Initialise the cohesin-leg lookup for the RNAPII-aware translocate.
    leg_by_pos: dict = args.setdefault("cohesin_leg_by_pos", {})
    for coh in cohesins:
        leg_by_pos[coh.left.pos] = coh.left
        leg_by_pos[coh.right.pos] = coh.right

    return occupied, cohesins


def _advance_one_step(
    *,
    rnapii_enabled: bool,
    rnapiis: list,
    cohesins: List[Cohesin],
    occupied: np.ndarray,
    args: dict,
    rnapii_load_fn,
    rnapii_translocate_fn,
    translocate_fn,
    unload_fn,
    load_fn,
    capture_fn,
    release_fn,
    lesion_update_fn=None,
) -> None:
    """Advance one 1D dynamics tick for warmup or recording."""
    # Lesions update first: occurrence + repair before cohesin / RNAPII move,
    # so the new lesion landscape is what the polymers see this tick.
    if lesion_update_fn is not None:
        lesion_update_fn(args)

    if rnapii_enabled:
        if args.get("genes"):
            tol = int(args.get("ep_contact_tolerance", 2))
            args["current_ep_contacts"] = compute_ep_contacts(
                cohesins,
                args["genes"],
                tolerance=tol,
            )
        rnapii_load_fn(rnapiis, occupied, args)
        rnapii_translocate_fn(rnapiis, cohesins, occupied, args)

    translocate_fn(
        cohesins,
        occupied,
        args,
        unload_prob_fn=unload_fn,
        load_fn=load_fn,
        capture_fn=capture_fn,
        release_fn=release_fn,
    )


def run(cfg: LEFConfig) -> Path:
    """Simulate LEF (+ optional RNAPII) dynamics and write ``LEFPositions.h5``."""

    if cfg.seed is not None:
        np.random.seed(int(cfg.seed))

    plugins = cfg.plugins

    topology_fn = resolve_plugin(plugins.topology)
    load_fn = resolve_plugin(plugins.load)
    unload_fn = resolve_plugin(plugins.unload_prob)
    capture_fn = resolve_plugin(plugins.capture)
    release_fn = resolve_plugin(plugins.release)
    translocate_fn = resolve_plugin(plugins.translocate)

    rnapii_enabled = (
        plugins.rnapii_load is not None and plugins.rnapii_translocate is not None
    )
    rnapii_load_fn = resolve_plugin(plugins.rnapii_load) if rnapii_enabled else None
    rnapii_translocate_fn = (
        resolve_plugin(plugins.rnapii_translocate) if rnapii_enabled else None
    )

    lesion_enabled = plugins.lesion is not None
    lesion_update_fn = resolve_plugin(plugins.lesion) if lesion_enabled else None

    args = topology_fn(cfg, **cfg.topology_kwargs)

    occupied, cohesins = initial_state(cfg, args, load_fn)
    rnapiis: list = []

    log.info(
        "[lef] 1D dynamics: N=%d sites, %d LEFs, rnapii=%s, lesions=%s | "
        "warmup=%d steps, trajectory=%d steps",
        cfg.num_sites, cfg.num_lefs, rnapii_enabled, lesion_enabled,
        max(0, int(cfg.warmup_steps)), cfg.trajectory_length,
    )

    warmup_steps = max(0, int(cfg.warmup_steps))
    warmup_meter = ProgressMeter(warmup_steps, "lef:warmup")
    for _ in range(warmup_steps):
        _advance_one_step(
            rnapii_enabled=rnapii_enabled,
            rnapiis=rnapiis,
            cohesins=cohesins,
            occupied=occupied,
            args=args,
            rnapii_load_fn=rnapii_load_fn,
            rnapii_translocate_fn=rnapii_translocate_fn,
            translocate_fn=translocate_fn,
            unload_fn=unload_fn,
            load_fn=load_fn,
            capture_fn=capture_fn,
            release_fn=release_fn,
            lesion_update_fn=lesion_update_fn,
        )
        warmup_meter.update()
    if warmup_steps:
        warmup_meter.done()

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_lefs = cfg.num_lefs
    traj_len = cfg.trajectory_length
    chunk = max(1, cfg.chunk_size)
    boundaries = np.linspace(0, traj_len, chunk + 1, dtype=int)

    with h5py.File(out_path, "w") as fh:
        dset = fh.create_dataset(
            "positions",
            shape=(traj_len, n_lefs, 2),
            dtype=np.int32,
            compression="gzip",
        )
        dset_rnapii = None
        dset_states = None
        if rnapii_enabled:
            dset_rnapii = fh.create_dataset(
                "rnapii_positions",
                shape=(traj_len, cfg.max_rnapii, 2),
                dtype=np.int32,
                compression="gzip",
                fillvalue=-1,
            )
            dset_states = fh.create_dataset(
                "rnapii_states",
                shape=(traj_len, cfg.max_rnapii),
                dtype=np.int8,
                compression="gzip",
                fillvalue=-1,
            )

        lesion_max = int(args.get("lesion_max", 64))
        dset_lesions = None
        if lesion_enabled:
            dset_lesions = fh.create_dataset(
                "lesions",
                shape=(traj_len, lesion_max),
                dtype=np.int32,
                compression="gzip",
                fillvalue=-1,
            )

        rec_meter = ProgressMeter(traj_len, "lef:record")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end == start:
                continue
            n_step = end - start
            buffer = np.empty((n_step, n_lefs, 2), dtype=np.int32)
            rbuf = (
                np.full((n_step, cfg.max_rnapii, 2), -1, dtype=np.int32)
                if rnapii_enabled
                else None
            )
            sbuf = (
                np.full((n_step, cfg.max_rnapii), -1, dtype=np.int8)
                if rnapii_enabled
                else None
            )
            lbuf = (
                np.full((n_step, lesion_max), -1, dtype=np.int32)
                if lesion_enabled
                else None
            )

            for i in range(n_step):
                _advance_one_step(
                    rnapii_enabled=rnapii_enabled,
                    rnapiis=rnapiis,
                    cohesins=cohesins,
                    occupied=occupied,
                    args=args,
                    rnapii_load_fn=rnapii_load_fn,
                    rnapii_translocate_fn=rnapii_translocate_fn,
                    translocate_fn=translocate_fn,
                    unload_fn=unload_fn,
                    load_fn=load_fn,
                    capture_fn=capture_fn,
                    release_fn=release_fn,
                    lesion_update_fn=lesion_update_fn,
                )

                for j, coh in enumerate(cohesins):
                    buffer[i, j, 0] = coh.left.pos
                    buffer[i, j, 1] = coh.right.pos

                if rbuf is not None:
                    for j, r in enumerate(rnapiis[: cfg.max_rnapii]):
                        rbuf[i, j, 0] = r.pos
                        rbuf[i, j, 1] = r.gene_id
                        sbuf[i, j] = r.attrs.get("state", -1)

                if lbuf is not None:
                    for j, site in enumerate(sorted(args["lesions"])[:lesion_max]):
                        lbuf[i, j] = site

                rec_meter.update(start + i + 1)

            dset[start:end] = buffer
            if dset_rnapii is not None:
                dset_rnapii[start:end] = rbuf
                dset_states[start:end] = sbuf
            if dset_lesions is not None:
                dset_lesions[start:end] = lbuf

        rec_meter.done()

        fh.attrs["N"] = cfg.num_sites
        fh.attrs["LEFNum"] = n_lefs
        fh.attrs["chain_length"] = cfg.chain_length
        fh.attrs["num_chains"] = cfg.num_chains
        fh.attrs["lifetime"] = cfg.lifetime
        fh.attrs["lifetime_stalled"] = cfg.lifetime_stalled
        fh.attrs["separation"] = cfg.separation
        fh.attrs["warmup_steps"] = cfg.warmup_steps
        if cfg.seed is not None:
            fh.attrs["seed"] = int(cfg.seed)
        fh.attrs["rnapii_enabled"] = bool(rnapii_enabled)
        fh.attrs["lesion_enabled"] = bool(lesion_enabled)
        if lesion_enabled:
            fh.attrs["lesion_max"] = lesion_max
            fh.attrs["lesion_prob"] = float(args.get("lesion_prob", 0.0))
            fh.attrs["lesion_lifetime"] = int(args.get("lesion_lifetime", 0))
            fh.attrs["lesion_block_prob"] = float(args.get("lesion_block_prob", 0.0))
        if rnapii_enabled:
            fh.attrs["max_rnapii"] = cfg.max_rnapii
            genes = args.get("genes", [])
            if genes:
                gene_arr = np.array(
                    [(g.gene_id, g.tss, g.tes, g.direction, g.load_prob)
                     for g in genes],
                    dtype=[("gene_id", "i4"), ("tss", "i4"),
                           ("tes", "i4"), ("direction", "i4"),
                           ("load_prob", "f4")],
                )
                fh.create_dataset("genes", data=gene_arr)

    return out_path
