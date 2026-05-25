"""Stage 3 driver: contact map sampling + O/E + visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ...hdf5_format import list_URIs
from .config import ContactsConfig, resolve_plugin


def _save(array: np.ndarray, path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, array)
    return out


def run(cfg: ContactsConfig) -> dict[str, Path]:
    """Sample contact map, compute O/E, render heatmap.

    Returns a dict of stage outputs: ``{"raw": ..., "oe": ..., "viz": ...}``
    (keys present only for stages that ran).
    """

    uris = list_URIs(str(cfg.trajectory_folder))
    if not uris:
        raise FileNotFoundError(f"No trajectory blocks found under {cfg.trajectory_folder}")

    sampler = resolve_plugin(cfg.plugins.sampler)
    raw = sampler(uris, cfg=cfg, **cfg.plugins.sampler.kwargs)

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
