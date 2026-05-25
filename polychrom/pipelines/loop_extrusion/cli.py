"""Thin CLI wrappers for each pipeline stage.

Usage::

    python -m polychrom.pipelines.loop_extrusion.cli lef     config.yaml
    python -m polychrom.pipelines.loop_extrusion.cli polymer config.yaml
    python -m polychrom.pipelines.loop_extrusion.cli contacts config.yaml
    python -m polychrom.pipelines.loop_extrusion.cli viewer  config.yaml
    python -m polychrom.pipelines.loop_extrusion.cli all     config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import contacts as contacts_stage
from . import lef as lef_stage
from . import polymer as polymer_stage
from . import viewer as viewer_stage
from .config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polychrom-loopext",
        description="Modular loop-extrusion pipeline (LEF + polymer + contacts).",
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("lef", "viewer", "polymer", "contacts", "all"):
        p = sub.add_parser(name, help=f"run the {name} stage")
        p.add_argument("config", type=Path, help="Path to YAML config")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.stage in ("lef", "all"):
        out = lef_stage.run(cfg.lef)
        print(f"[lef]      wrote {out}")
    # Inspect the 1D dynamics before paying for 3D MD.
    if args.stage in ("viewer", "all"):
        out = viewer_stage.run(cfg.viewer, cfg.lef)
        print(f"[viewer]   wrote {out}")
    if args.stage in ("polymer", "all"):
        out = polymer_stage.run(cfg.polymer)
        print(f"[polymer]  wrote trajectory to {out}")
    if args.stage in ("contacts", "all"):
        outs = contacts_stage.run(cfg.contacts)
        for kind, path in outs.items():
            print(f"[contacts] wrote {kind}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
