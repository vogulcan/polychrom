"""Thin CLI wrappers for each pipeline stage.

Usage::

    python -m polychrom.pipelines.loop_extrusion.cli lef     config.yaml [output_dir]
    python -m polychrom.pipelines.loop_extrusion.cli polymer config.yaml [output_dir]
    python -m polychrom.pipelines.loop_extrusion.cli contacts config.yaml [output_dir]
    python -m polychrom.pipelines.loop_extrusion.cli viewer  config.yaml [output_dir]
    python -m polychrom.pipelines.loop_extrusion.cli all     config.yaml [output_dir]
    python -m polychrom.pipelines.loop_extrusion.cli compare baseline_run comparison_run [...]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from . import compare as compare_stage
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


def _infer_config_path(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Compare input not found: {path}")

    preferred = [path / f"{path.name}.yaml", path / f"{path.name}.yml"]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    candidates = sorted([*path.glob("*.yaml"), *path.glob("*.yml")])
    if not candidates:
        raise FileNotFoundError(f"No YAML config found in run folder: {path}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"Multiple YAML configs in {path}; pass config path explicitly ({names})")
    return candidates[0]


def _load_compare_input(path: Path, folder: Path | None = None):
    config_path = _infer_config_path(path)
    run_folder = Path(folder) if folder is not None else (Path(path) if Path(path).is_dir() else None)
    cfg = load_config(config_path)
    if run_folder is None and (config_path.parent / "LEFPositions.h5").exists():
        run_folder = config_path.parent
    if run_folder is not None:
        _override_run_paths(cfg, run_folder)
    label = run_folder.name if run_folder is not None else config_path.stem
    return cfg, label


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polychrom-loopext",
        description="Modular loop-extrusion pipeline (LEF + polymer + contacts).",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-q", "--quiet", action="store_true",
        help="only warnings/errors (default is verbose progress + ETA)",
    )
    verbosity.add_argument(
        "--debug", action="store_true", help="extra debug-level logging",
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
    cmp_p = sub.add_parser("compare", help="pairwise or baseline-vs-many comparison")
    cmp_p.add_argument("baseline", type=Path, help="baseline config or run folder")
    cmp_p.add_argument("comparisons", type=Path, nargs="+", help="comparison config(s) or run folder(s)")
    cmp_p.add_argument("--baseline-folder", "--folder-a", dest="baseline_folder", type=Path, default=None,
                       help="override baseline run folder (LEFPositions.h5 + contact_map*.npy live here)")
    cmp_p.add_argument("--folder-b", type=Path, default=None,
                       help="override comparison run folder; only valid with one comparison")
    cmp_p.add_argument("--folders", type=Path, nargs="+", default=None,
                       help="override comparison run folders, one per comparison")
    cmp_p.add_argument("--out", type=Path, default=None,
                       help="output directory (default: compare_<baseline>_<comparison>/ or compare_<baseline>_vs_many/)")
    cmp_p.add_argument("--label-a", "--baseline-label", dest="label_a", type=str, default=None)
    cmp_p.add_argument("--label-b", type=str, default=None,
                       help="comparison label; only valid with one comparison")
    cmp_p.add_argument("--labels", type=str, nargs="+", default=None,
                       help="comparison labels, one per comparison")
    cmp_p.add_argument("--cutoffs", type=float, nargs="+", default=None,
                       help="contact-distance cutoffs (e.g. 2 3 4 5 6); resamples each "
                            "run's contact map from its trajectory and writes one "
                            "cutoff_<c>/ subfolder per value")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # High verbosity by default: surface per-stage progress/ETA and the
    # Simulation.do_block throughput lines (logging.info). --quiet drops to
    # warnings; --debug adds debug detail.
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.debug else logging.INFO)
    logging.basicConfig(level=level, format="%(message)s", force=True)

    if args.stage == "compare":
        if args.folder_b is not None and args.folders is not None:
            parser.error("Use either --folder-b or --folders, not both")
        if args.folder_b is not None and len(args.comparisons) != 1:
            parser.error("--folder-b is only valid with one comparison")
        if args.folders is not None and len(args.folders) != len(args.comparisons):
            parser.error("--folders must contain one folder per comparison")
        if args.label_b is not None and args.labels is not None:
            parser.error("Use either --label-b or --labels, not both")
        if args.label_b is not None and len(args.comparisons) != 1:
            parser.error("--label-b is only valid with one comparison")
        if args.labels is not None and len(args.labels) != len(args.comparisons):
            parser.error("--labels must contain one label per comparison")

        cfg_a, default_label_a = _load_compare_input(args.baseline, args.baseline_folder)
        folders = [args.folder_b] if args.folder_b is not None else args.folders
        cfg_others = []
        default_labels = []
        for idx, comparison in enumerate(args.comparisons):
            folder = folders[idx] if folders is not None else None
            cfg, label = _load_compare_input(comparison, folder)
            cfg_others.append(cfg)
            default_labels.append(label)

        label_a = args.label_a or default_label_a
        labels = args.labels or ([args.label_b] if args.label_b is not None else default_labels)
        if len(cfg_others) == 1:
            out = args.out or Path(f"compare_{label_a}_{labels[0]}")
            result = compare_stage.run(cfg_a, cfg_others[0], out, label_a, labels[0],
                                       cutoffs=args.cutoffs)
        else:
            out = args.out or Path(f"compare_{label_a}_vs_many")
            result = compare_stage.run_many(cfg_a, cfg_others, out, label_a, labels,
                                            cutoffs=args.cutoffs)
        print(f"[compare]  wrote {result}")
        return 0
    cfg = load_config(args.config, output_path=args.output_path)
    if args.output_path is not None:
        print(f"[config]   output directory {args.output_path}")
        archived_config = _archive_config(args.config, args.output_path)
        print(f"[config]   saved {archived_config}")

    if args.stage in ("lef", "all"):
        from . import lef as lef_stage
        out = lef_stage.run(cfg.lef)
        print(f"[lef]      wrote {out}")
    # Inspect the 1D dynamics before paying for 3D MD.
    if args.stage in ("viewer", "all"):
        from . import viewer as viewer_stage
        out = viewer_stage.run(cfg.viewer, cfg.lef)
        print(f"[viewer]   wrote {out}")
    if args.stage in ("polymer", "all"):
        from . import polymer as polymer_stage
        out = polymer_stage.run(cfg.polymer)
        print(f"[polymer]  wrote trajectory to {out}")
    if args.stage in ("contacts", "all"):
        from . import contacts as contacts_stage
        outs = contacts_stage.run(cfg.contacts, cfg.lef)
        for kind, path in outs.items():
            print(f"[contacts] wrote {kind}: {path}")
    if args.stage in ("qc", "all"):
        from . import qc as qc_stage
        out = qc_stage.run(cfg)
        print(f"[qc]       wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
