"""Stage 3 driver: contact map sampling + O/E + visualisation."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from ...hdf5_format import list_URIs
from . import annotate
from .config import ContactsConfig, LEFConfig, resolve_plugin
from .progress import log


def _save(array: np.ndarray, path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, array)
    return out


def _cutoff_tag(cutoff: float) -> str:
    c = float(cutoff)
    return f"cutoff{int(c)}" if c.is_integer() else f"cutoff{c}"


def _suffix_path(path: str, cutoff: float) -> str:
    """Insert a ``_cutoff<c>`` tag before the suffix of ``path``."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{_cutoff_tag(cutoff)}{p.suffix}"))


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


def _run_one(cfg: ContactsConfig, uris, *, key_suffix: str = "",
             annotations: Optional[dict] = None) -> dict[str, Path]:
    """Sample + O/E + post-process + visualise for a single ``cfg.cutoff``."""
    log.info(
        "[contacts] sampling %d trajectory blocks, map_size=%d, cutoff=%s, "
        "%d processes (per-block progress below)",
        len(uris), cfg.map_size, cfg.cutoff, cfg.num_processes,
    )
    sampler = resolve_plugin(cfg.plugins.sampler)
    raw = sampler(uris, cfg=cfg, **cfg.plugins.sampler.kwargs)
    log.info("[contacts] contact map sampled; computing O/E + visualisation")

    outputs: dict[str, Path] = {f"raw{key_suffix}": _save(raw, cfg.raw_output_path)}

    oe: Optional[np.ndarray] = None
    if cfg.plugins.obs_over_exp is not None:
        oe_fn = resolve_plugin(cfg.plugins.obs_over_exp)
        oe = oe_fn(raw, **cfg.plugins.obs_over_exp.kwargs)
        outputs[f"oe{key_suffix}"] = _save(oe, cfg.oe_output_path)

    if cfg.plugins.post_process is not None:
        post = resolve_plugin(cfg.plugins.post_process)
        target = oe if oe is not None else raw
        outputs[f"post{key_suffix}"] = _save(post(target, **cfg.plugins.post_process.kwargs),
                                             cfg.oe_output_path)

    if cfg.plugins.viz is not None:
        viz = resolve_plugin(cfg.plugins.viz)
        target = oe if oe is not None else raw
        viz_path = Path(cfg.viz_output_path)
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        viz_kwargs = dict(cfg.plugins.viz.kwargs)
        if annotations is not None:
            viz_kwargs.setdefault("annotations", annotations)
        viz(target, output_path=str(viz_path), **viz_kwargs)
        outputs[f"viz{key_suffix}"] = viz_path

    return outputs


def run(cfg: ContactsConfig, lef_cfg: Optional[LEFConfig] = None) -> dict[str, Path]:
    """Sample contact map(s), compute O/E, render heatmap.

    ``cfg.cutoff`` may be a single value or a list. With one cutoff the outputs
    use the configured base paths. With several cutoffs each map is written to a
    ``_cutoff<c>``-tagged path; the first cutoff additionally writes the base
    paths so the ``qc`` stage and existing tooling still find ``contact_map.npy``.

    Returns a dict of stage outputs (keys per stage that ran, tagged per cutoff
    when more than one cutoff is sampled).
    """
    window_origin = int(cfg.map_starts[0]) if cfg.map_starts else 0
    annotations = (annotate.from_lef_cfg(lef_cfg, origin=window_origin, span=cfg.map_size)
                   if lef_cfg is not None else None)
    cfg = replace(cfg, map_starts=_effective_map_starts(cfg, lef_cfg))

    uris = list_URIs(str(cfg.trajectory_folder))
    if not uris:
        raise FileNotFoundError(f"No trajectory blocks found under {cfg.trajectory_folder}")

    cutoffs = cfg.cutoff_list
    if len(cutoffs) == 1:
        return _run_one(replace(cfg, cutoff=cutoffs[0]), uris, annotations=annotations)

    outputs: dict[str, Path] = {}
    for idx, cutoff in enumerate(cutoffs):
        tag = _cutoff_tag(cutoff)
        cut_cfg = replace(
            cfg,
            cutoff=cutoff,
            raw_output_path=_suffix_path(cfg.raw_output_path, cutoff),
            oe_output_path=_suffix_path(cfg.oe_output_path, cutoff),
            viz_output_path=_suffix_path(cfg.viz_output_path, cutoff),
        )
        outputs.update(_run_one(cut_cfg, uris, key_suffix=f"[{tag}]",
                                annotations=annotations))
        # First cutoff also populates the un-tagged base paths so qc + existing
        # tooling that expects contact_map.npy keep working.
        if idx == 0:
            for kind, src, base in (("raw", cut_cfg.raw_output_path, cfg.raw_output_path),
                                    ("oe", cut_cfg.oe_output_path, cfg.oe_output_path),
                                    ("viz", cut_cfg.viz_output_path, cfg.viz_output_path)):
                if Path(src).exists():
                    base_path = Path(base)
                    base_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, base_path)
                    outputs[f"{kind}[base]"] = base_path
    return outputs
