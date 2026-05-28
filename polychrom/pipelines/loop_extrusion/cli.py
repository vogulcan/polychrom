"""Thin CLI wrappers for each pipeline stage.

Usage::

    python -m polychrom.pipelines.loop_extrusion.cli lef     config.yaml [output_dir]
    python -m polychrom.pipelines.loop_extrusion.cli polymer config.yaml [output_dir]
    python -m polychrom.pipelines.loop_extrusion.cli contacts config.yaml [output_dir]
    python -m polychrom.pipelines.loop_extrusion.cli viewer  config.yaml [output_dir]
    python -m polychrom.pipelines.loop_extrusion.cli all     config.yaml [output_dir]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import compare as compare_stage
from . import contacts as contacts_stage
from . import lef as lef_stage
from . import polymer as polymer_stage
from . import qc as qc_stage
from . import viewer as viewer_stage
from .config import load_config


def _override_run_paths(cfg, folder: Path) -> None:
    """Re-point the lef + contacts paths of ``cfg`` to ``folder/`` (in place)."""
    folder = Path(folder)
    cfg.lef.output_path = str(folder / "LEFPositions.h5")
    cfg.contacts.trajectory_folder = str(folder)
    cfg.contacts.raw_output_path = str(folder / "contact_map.npy")
    cfg.contacts.oe_output_path = str(folder / "contact_map_oe.npy")


def _archive_config(config_path: Path, output_path: Path) -> Path:
    """Copy the input config into the run output directory."""

    output_path.mkdir(parents=True, exist_ok=True)
    archived = output_path / config_path.name
    if config_path.resolve() != archived.resolve():
        shutil.copy2(config_path, archived)
    return archived


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polychrom-loopext",
        description="Modular loop-extrusion pipeline (LEF + polymer + contacts).",
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("lef", "viewer", "polymer", "contacts", "qc", "all"):
        p = sub.add_parser(name, help=f"run the {name} stage")
        p.add_argument("config", type=Path, help="Path to YAML config")
        p.add_argument(
            "output_path",
            type=Path,
            nargs="?",
            help="Run output directory; overrides all derived stage paths",
        )
    cmp_p = sub.add_parser("compare", help="pairwise comparison of two runs")
    cmp_p.add_argument("config_a", type=Path, help="config for run A")
    cmp_p.add_argument("config_b", type=Path, help="config for run B")
    cmp_p.add_argument("--folder-a", type=Path, default=None,
                       help="override A's run folder (LEFPositions.h5 + contact_map*.npy live here)")
    cmp_p.add_argument("--folder-b", type=Path, default=None,
                       help="override B's run folder")
    cmp_p.add_argument("--out", type=Path, default=None,
                       help="output directory for comparison (default: compare_<A>_<B>/)")
    cmp_p.add_argument("--label-a", type=str, default="A")
    cmp_p.add_argument("--label-b", type=str, default="B")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.stage == "compare":
        cfg_a = load_config(args.config_a)
        cfg_b = load_config(args.config_b)
        if args.folder_a is not None:
            _override_run_paths(cfg_a, args.folder_a)
        if args.folder_b is not None:
            _override_run_paths(cfg_b, args.folder_b)
        out = args.out or Path(f"compare_{args.label_a}_{args.label_b}")
        result = compare_stage.run(cfg_a, cfg_b, out, args.label_a, args.label_b)
        print(f"[compare]  wrote {result}")
        return 0
    cfg = load_config(args.config, output_path=args.output_path)
    if args.output_path is not None:
        print(f"[config]   output directory {args.output_path}")
        archived_config = _archive_config(args.config, args.output_path)
        print(f"[config]   saved {archived_config}")

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
        outs = contacts_stage.run(cfg.contacts, cfg.lef)
        for kind, path in outs.items():
            print(f"[contacts] wrote {kind}: {path}")
    if args.stage in ("qc", "all"):
        out = qc_stage.run(cfg)
        print(f"[qc]       wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
