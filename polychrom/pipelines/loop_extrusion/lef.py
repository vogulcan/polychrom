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
                chain_length=args.get("chain_length"),
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
    args["num_chains"] = cfg.num_chains
    # Lesion logic gates Type-A pre-recognition cohesin stalling on whether real
    # RNAPII is in the model (if so, the stalled Pol II does the blocking).
    args["rnapii_enabled"] = rnapii_enabled

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

    # max_rnapii is per-chain; total recording width scales with num_chains.
    rnapii_cap = cfg.max_rnapii * cfg.num_chains

    # Storage chunks: whole rows (full width, unsplit last axis) with a time depth
    # matching the write block. h5py's auto-chunking otherwise splits the columns
    # AND the size-2 last axis, so a script-side time-slice read (``ds[a:b]``) has
    # to touch many tiny chunks; whole-row chunks let it decompress whole rows
    # (~20% faster reads, e.g. gen_transcription_metrics). Aligning the time depth
    # to the write block also keeps each ``dset[start:end] = buffer`` write within
    # whole chunks. Data is byte-identical; only the on-disk layout changes.
    _blk = int(boundaries[1] - boundaries[0]) if len(boundaries) > 1 else traj_len

    def _tchunk(row_bytes: int, cap_bytes: int = 8 << 20) -> int:
        return max(1, min(traj_len, _blk, max(1, cap_bytes // max(1, row_bytes))))

    with h5py.File(out_path, "w") as fh:
        dset = fh.create_dataset(
            "positions",
            shape=(traj_len, n_lefs, 2),
            dtype=np.int32,
            chunks=(_tchunk(n_lefs * 2 * 4), n_lefs, 2),
            compression="lzf",
        )
        dset_rnapii = None
        dset_states = None
        dset_ids = None
        if rnapii_enabled:
            dset_rnapii = fh.create_dataset(
                "rnapii_positions",
                shape=(traj_len, rnapii_cap, 2),
                dtype=np.int32,
                chunks=(_tchunk(rnapii_cap * 2 * 4), rnapii_cap, 2),
                compression="lzf",
                fillvalue=-1,
            )
            dset_states = fh.create_dataset(
                "rnapii_states",
                shape=(traj_len, rnapii_cap),
                dtype=np.int8,
                chunks=(_tchunk(rnapii_cap), rnapii_cap),
                compression="lzf",
                fillvalue=-1,
            )
            # Stable per-Pol identity: the live rnapii list is compacted on unload,
            # so a column index is NOT a fixed Pol II. Record each Pol's uid so
            # downstream can reconstruct per-Pol tracks across column shuffles.
            dset_ids = fh.create_dataset(
                "rnapii_ids",
                shape=(traj_len, rnapii_cap),
                dtype=np.int32,
                chunks=(_tchunk(rnapii_cap * 4), rnapii_cap),
                compression="lzf",
                fillvalue=-1,
            )

        # Lesion recording capacity = the homeostatic target (N // spacing); the
        # population never exceeds it, so no truncation. Sites, types and states
        # are recorded as parallel -1-padded datasets (cf. RNAPII positions/states).
        lesion_cap = int(args.get("lesion_target", 0))
        lesion_recorded = lesion_enabled and lesion_cap > 0
        dset_lesions = dset_lesion_types = dset_lesion_states = None
        if lesion_recorded:
            dset_lesions = fh.create_dataset(
                "lesions", shape=(traj_len, lesion_cap), dtype=np.int32,
                chunks=(_tchunk(lesion_cap * 4), lesion_cap),
                compression="lzf", fillvalue=-1,
            )
            dset_lesion_types = fh.create_dataset(
                "lesion_types", shape=(traj_len, lesion_cap), dtype=np.int8,
                chunks=(_tchunk(lesion_cap), lesion_cap),
                compression="lzf", fillvalue=-1,
            )
            dset_lesion_states = fh.create_dataset(
                "lesion_states", shape=(traj_len, lesion_cap), dtype=np.int8,
                chunks=(_tchunk(lesion_cap), lesion_cap),
                compression="lzf", fillvalue=-1,
            )

        rec_meter = ProgressMeter(traj_len, "lef:record")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end == start:
                continue
            n_step = end - start
            buffer = np.empty((n_step, n_lefs, 2), dtype=np.int32)
            rbuf = (
                np.full((n_step, rnapii_cap, 2), -1, dtype=np.int32)
                if rnapii_enabled
                else None
            )
            sbuf = (
                np.full((n_step, rnapii_cap), -1, dtype=np.int8)
                if rnapii_enabled
                else None
            )
            ibuf = (
                np.full((n_step, rnapii_cap), -1, dtype=np.int32)
                if rnapii_enabled
                else None
            )
            lbuf = (
                np.full((n_step, lesion_cap), -1, dtype=np.int32)
                if lesion_recorded
                else None
            )
            ltbuf = (
                np.full((n_step, lesion_cap), -1, dtype=np.int8)
                if lesion_recorded
                else None
            )
            lsbuf = (
                np.full((n_step, lesion_cap), -1, dtype=np.int8)
                if lesion_recorded
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

                # Bulk-assign each frame from Python lists rather than per-element
                # numpy __setitem__ (millions of scalar sets per run). Same values,
                # same slots -> byte-identical buffers; the -1 padding tails are
                # left untouched exactly as before.
                buffer[i, : len(cohesins)] = [
                    (coh.left.pos, coh.right.pos) for coh in cohesins
                ]

                if rbuf is not None:
                    live = rnapiis[:rnapii_cap]
                    n = len(live)
                    if n:
                        rbuf[i, :n, 0] = [r.pos for r in live]
                        rbuf[i, :n, 1] = [r.gene_id for r in live]
                        sbuf[i, :n] = [r.attrs.get("state", -1) for r in live]
                        ibuf[i, :n] = [r.uid for r in live]

                if lbuf is not None:
                    cur = args["lesions"]
                    sites = sorted(cur)[:lesion_cap]
                    n = len(sites)
                    if n:
                        lbuf[i, :n] = sites
                        ltbuf[i, :n] = [cur[s].ltype for s in sites]
                        lsbuf[i, :n] = [cur[s].state for s in sites]

                rec_meter.update(start + i + 1)

            dset[start:end] = buffer
            if dset_rnapii is not None:
                dset_rnapii[start:end] = rbuf
                dset_states[start:end] = sbuf
                dset_ids[start:end] = ibuf
            if dset_lesions is not None:
                dset_lesions[start:end] = lbuf
                dset_lesion_types[start:end] = ltbuf
                dset_lesion_states[start:end] = lsbuf

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
        if lesion_recorded:
            fh.attrs["lesion_target"] = lesion_cap
            fh.attrs["lesion_spacing"] = int(args.get("lesion_spacing", 0))
            fh.attrs["lesion_type_a_prob"] = float(args.get("lesion_type_a_prob", 0.5))
            fh.attrs["lesion_prerecognition_ticks"] = int(args.get("lesion_prerecognition_ticks", 0))
            fh.attrs["lesion_repair_ticks"] = int(args.get("lesion_repair_ticks", 0))
            fh.attrs["lesion_tad_size_exponent"] = float(args.get("lesion_tad_size_exponent", 1.0))
            fh.attrs["lesion_tad_repair_exponent"] = float(args.get("lesion_tad_repair_exponent", 0.0))
            fh.attrs["lesion_type_b_enabled"] = bool(args.get("lesion_type_b_enabled", True))
            fh.attrs["lesion_target_a"] = int(args.get("lesion_target_a", 0))
            fh.attrs["lesion_block_prob"] = float(args.get("lesion_block_prob", 0.0))
        if rnapii_enabled:
            fh.attrs["max_rnapii"] = cfg.max_rnapii
            fh.attrs["rnapii_cap"] = rnapii_cap
            fh.attrs["rnapii_pause_term_prob"] = float(args.get("rnapii_pause_term_prob", 0.0))
            genes = args.get("genes", [])
            if genes:
                gene_arr = np.array(
                    [(g.gene_id, g.tss, g.tes, g.direction, g.load_prob,
                      (getattr(g, "gene_class", "") or "").encode("ascii", "ignore")[:16])
                     for g in genes],
                    dtype=[("gene_id", "i4"), ("tss", "i4"),
                           ("tes", "i4"), ("direction", "i4"),
                           ("load_prob", "f4"), ("gene_class", "S16")],
                )
                fh.create_dataset("genes", data=gene_arr)

        # Ground-truth CTCF anchor table (per-TAD oriented boundaries). Written
        # whenever the topology plugin exposes it (convergent topologies); absent
        # for uniform/symmetric topologies, so downstream analysis falls back to
        # config-derived anchors. Tiny -> no compression.
        anchors = args.get("ctcf_anchors")
        if anchors:
            anchor_arr = np.array(
                anchors,
                dtype=[
                    ("abs_position", "i4"),
                    ("chain_position", "i4"),
                    ("chain_index", "i4"),
                    ("side", "i1"),
                    ("strength", "f4"),
                    ("tad_index", "i4"),
                    ("edge", "i1"),
                ],
            )
            fh.create_dataset("ctcf_anchors", data=anchor_arr)
            fh.attrs["boundary_model"] = "per_tad_oriented"
            fh.attrs["ctcf_anchor_schema_version"] = 1
            fh.attrs["ctcf_anchors_per_chain"] = int(
                len(anchor_arr) // max(cfg.num_chains, 1)
            )

    return out_path
