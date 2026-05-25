"""Modular, configurable loop-extrusion pipeline.

Three stages:
    1. ``lef``       -- 1D LEF/cohesin dynamics, writes ``LEFPositions.h5``.
    2. ``polymer``   -- 3D MD simulation with dynamic SMC bonds.
    3. ``contacts``  -- Contact map sampling from the trajectory.

Each stage is a Python entry point plus a thin CLI; mechanics are pluggable
via the ``plugins:`` section of the YAML config.
"""

from .config import (
    PipelineConfig,
    LEFConfig,
    PolymerConfig,
    ContactsConfig,
    ViewerConfig,
    PluginSpec,
    load_config,
    resolve_plugin,
)

__all__ = [
    "PipelineConfig",
    "LEFConfig",
    "PolymerConfig",
    "ContactsConfig",
    "ViewerConfig",
    "PluginSpec",
    "load_config",
    "resolve_plugin",
]
