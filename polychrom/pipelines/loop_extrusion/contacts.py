"""Stage 3 driver: contact map sampling + O/E + visualisation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from ...hdf5_format import list_URIs
from .config import ContactsConfig, LEFConfig, resolve_plugin
from .progress import log


def _save(array: np.ndarray, path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, array)
    return out


def _effective_map_starts(cfg: ContactsConfig, lef_cfg: Optional[LEFConfig]) -> list[int]:
    starts = [int(start) for start in cfg.map_starts]
    if not cfg.replicate_map_starts_across_chains:
        return starts

    if lef_cfg is None:
        raise ValueError(
            "replicate_map_starts_across_chains requires the full pipeline "
            "config so LEF chain_length and num_chains are available"
        )

    expanded: list[int] = []
    for chain_idx in range(lef_cfg.num_chains):
        chain_offset = chain_idx * lef_cfg.chain_length
        for start in starts:
            if not (0 <= start and start + cfg.map_size <= lef_cfg.chain_length):
                raise ValueError(
                    f"map_start {start} with map_size {cfg.map_size} does not fit "
                    f"inside one chain of length {lef_cfg.chain_length}"
                )
            expanded.append(chain_offset + start)
    return expanded


def run(cfg: ContactsConfig, lef_cfg: Optional[LEFConfig] = None) -> dict[str, Path]:
    """Sample contact map, compute O/E, render heatmap.

    Returns a dict of stage outputs: ``{"raw": ..., "oe": ..., "viz": ...}``
    (keys present only for stages that ran).
    """
    cfg = replace(cfg, map_starts=_effective_map_starts(cfg, lef_cfg))

    uris = list_URIs(str(cfg.trajectory_folder))
    if not uris:
        raise FileNotFoundError(f"No trajectory blocks found under {cfg.trajectory_folder}")

    log.info(
        "[contacts] sampling %d trajectory blocks, map_size=%d, %d processes "
        "(per-block progress below)",
        len(uris), cfg.map_size, cfg.num_processes,
    )
    sampler = resolve_plugin(cfg.plugins.sampler)
    raw = sampler(uris, cfg=cfg, **cfg.plugins.sampler.kwargs)
    log.info("[contacts] contact map sampled; computing O/E + visualisation")

    outputs: dict[str, Path] = {"raw": _save(raw, cfg.raw_output_path)}

    oe: Optional[np.ndarray] = None
    if cfg.plugins.obs_over_exp is not None:
        oe_fn = resolve_plugin(cfg.plugins.obs_over_exp)
        oe = oe_fn(raw, **cfg.plugins.obs_over_exp.kwargs)
        outputs["oe"] = _save(oe, cfg.oe_output_path)

    if cfg.plugins.post_process is not None:
        post = resolve_plugin(cfg.plugins.post_process)
        target = oe if oe is not None else raw
        outputs["post"] = _save(post(target, **cfg.plugins.post_process.kwargs),
                                cfg.oe_output_path)

    if cfg.plugins.viz is not None:
        viz = resolve_plugin(cfg.plugins.viz)
        target = oe if oe is not None else raw
        viz_path = Path(cfg.viz_output_path)
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        viz(target, output_path=str(viz_path), **cfg.plugins.viz.kwargs)
        outputs["viz"] = viz_path

    return outputs
