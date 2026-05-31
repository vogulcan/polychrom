"""Typed configuration for the loop-extrusion pipeline.

Configuration lives in a YAML file and is materialised into nested
dataclasses. Every "mechanic" hook is a :class:`PluginSpec`, which combines
a fully-qualified ``module:attr`` target with an optional ``kwargs`` mapping
that is forwarded to the callable at runtime.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    get_type_hints,
)

try:
    from typing import get_args, get_origin
except ImportError:  # Python 3.7 typing module
    def get_origin(tp):
        return getattr(tp, "__origin__", None)

    def get_args(tp):
        return getattr(tp, "__args__", ())

import yaml

DEFAULT_LEF_DYNAMICS = "polychrom.pipelines.loop_extrusion.plugins.lef_dynamics"
DEFAULT_TOPOLOGY = "polychrom.pipelines.loop_extrusion.plugins.topology"
DEFAULT_FORCES = "polychrom.pipelines.loop_extrusion.plugins.forces"
DEFAULT_SAMPLING = "polychrom.pipelines.loop_extrusion.plugins.sampling"
DEFAULT_RNAPII = "polychrom.pipelines.loop_extrusion.plugins.rnapii"


@dataclass
class PluginSpec:
    """Pointer to a callable plus the kwargs it should be invoked with."""

    target: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any) -> "PluginSpec":
        if obj is None:
            raise ValueError("PluginSpec value is required")
        if isinstance(obj, PluginSpec):
            return obj
        if isinstance(obj, str):
            return cls(target=obj)
        if isinstance(obj, Mapping):
            if "target" not in obj:
                raise ValueError("Plugin mapping must contain a 'target' key")
            return cls(target=str(obj["target"]), kwargs=dict(obj.get("kwargs", {})))
        raise TypeError(f"Cannot build PluginSpec from {type(obj).__name__}")


def resolve_plugin(spec: PluginSpec) -> Callable[..., Any]:
    """Import ``module:attr`` (or ``module.attr``) and return the callable."""

    target = spec.target
    if ":" in target:
        module_name, attr = target.split(":", 1)
    else:
        module_name, _, attr = target.rpartition(".")
        if not module_name:
            raise ValueError(f"Plugin target '{target}' must be 'module:attr'")
    module = importlib.import_module(module_name)
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Plugin '{target}' not found in module '{module_name}'") from exc
    if not callable(obj):
        raise TypeError(f"Plugin '{target}' is not callable")
    return obj


@dataclass
class LEFPlugins:
    """Plugin slots that customise 1D mechanics."""

    topology: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_TOPOLOGY}:uniform_tad_topology")
    )
    load: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:load_one")
    )
    unload_prob: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:unload_prob")
    )
    capture: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:capture")
    )
    release: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:release")
    )
    translocate: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:translocate")
    )
    # Optional RNAPII slots. When both are set, lef.run() will also drive
    # transcription dynamics each step. Default translocate should be swapped
    # to ``translocate_with_rnapii`` so cohesin honours RNAPII presence.
    rnapii_load: Optional[PluginSpec] = None
    rnapii_translocate: Optional[PluginSpec] = None
    # Optional lesion (DNA-damage) slot. When set, lef.run() updates lesions
    # each step (occurrence in gene bodies + stochastic repair); cohesin and
    # RNAPII then honour lesion sites as barriers.
    lesion: Optional[PluginSpec] = None


@dataclass
class LEFConfig:
    """Stage 1: 1D loop-extrusion factor dynamics."""

    # Lattice size: derived from chain_length * num_chains if left blank.
    chain_length: int = 4000
    num_chains: int = 10
    separation: int = 800            # average bp / lattice sites between LEFs
    lifetime: int = 200              # average lifetime (steps)
    lifetime_stalled: int = 200      # lifetime when stalled (not at CTCF)
    warmup_steps: int = 0            # dynamics steps to discard before saving
    trajectory_length: int = 100000  # number of LEF dynamics steps to save
    chunk_size: int = 50             # write chunks to LEFPositions.h5
    output_path: str = "trajectory/LEFPositions.h5"
    seed: Optional[int] = None       # numpy RNG seed for reproducible 1D dynamics

    # Topology kwargs: forwarded to the topology plugin.
    topology_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Maximum concurrent RNAPII count; controls HDF5 padding of the
    # ``rnapii_positions`` dataset when RNAPII dynamics are enabled.
    max_rnapii: int = 64

    plugins: LEFPlugins = field(default_factory=LEFPlugins)

    @property
    def num_sites(self) -> int:
        return self.chain_length * self.num_chains

    @property
    def num_lefs(self) -> int:
        return self.num_sites // self.separation


@dataclass
class PolymerPlugins:
    """Plugin slots that customise 3D mechanics."""

    force_builder: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_FORCES}:default_force_builder")
    )
    initial_conformation: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_FORCES}:grow_cubic_conformation")
    )


@dataclass
class PolymerConfig:
    """Stage 2: 3D molecular-dynamics simulation."""

    lef_positions_path: str = "trajectory/LEFPositions.h5"
    output_folder: str = "trajectory"

    # OpenMM / Simulation parameters.
    platform: str = "cuda"
    gpu: str = "0"
    integrator: str = "variableLangevin"
    error_tol: float = 0.01
    collision_rate: float = 0.03
    precision: str = "mixed"
    seed: Optional[int] = None       # numpy/OpenMM RNG seed for reproducible 3D dynamics

    # Density / box.
    density: float = 0.1
    # If True (default), the simulation runs under periodic boundary
    # conditions with a cubic box sized by ``density``. Set False when
    # the force builder supplies its own confinement (e.g. spherical).
    pbc: bool = True

    # MD timing.
    md_steps_per_block: int = 750
    save_every_blocks: int = 10
    restart_every_blocks: int = 100

    # MD steps to run on iteration 0 BEFORE any cohesin bonds are inserted.
    # Used to relax the initial cubic-lattice conformation to steady state.
    initial_relaxation_steps: int = 0
    # MD steps with cohesin bonds in place but no recording -- gives the
    # system time to equilibrate before the first saved block.
    pre_recording_steps: int = 0

    # SMC bond geometry.
    smc_bond_wiggle: float = 0.2
    smc_bond_dist: float = 0.5

    # Reporter.
    max_data_length: int = 100
    overwrite: bool = True

    plugins: PolymerPlugins = field(default_factory=PolymerPlugins)


@dataclass
class ContactsPlugins:
    sampler: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_SAMPLING}:monomer_resolution_sampler")
    )
    # Observed/expected normaliser. Set to None to skip O/E generation.
    obs_over_exp: Optional[PluginSpec] = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_SAMPLING}:observed_over_expected")
    )
    # Optional second post-process applied after O/E (e.g. log, clip).
    post_process: Optional[PluginSpec] = None
    # Heatmap renderer. Set to None to skip visualisation.
    viz: Optional[PluginSpec] = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_SAMPLING}:default_oe_heatmap")
    )


@dataclass
class ContactsConfig:
    """Stage 3: contact map sampling + O/E generation + visualisation."""

    trajectory_folder: str = "trajectory"
    # Raw contact map output (always written).
    raw_output_path: str = "trajectory/contact_map.npy"
    # Observed/expected output (written when plugins.obs_over_exp is set).
    oe_output_path: str = "trajectory/contact_map_oe.npy"
    # Visualisation PNG (written when plugins.viz is set).
    viz_output_path: str = "trajectory/contact_map_oe.png"
    map_starts: List[int] = field(default_factory=lambda: list(range(0, 39000, 4000)))
    replicate_map_starts_across_chains: bool = False
    map_size: int = 4000
    # One contact-distance cutoff, or a list of cutoffs (e.g. [2, 3, 4, 5, 6]).
    # With a list, the contacts stage samples one map per cutoff.
    cutoff: Union[float, List[float]] = 6.0
    num_processes: int = 6
    verbose: bool = True

    plugins: ContactsPlugins = field(default_factory=ContactsPlugins)

    @property
    def cutoff_list(self) -> List[float]:
        """Normalise ``cutoff`` to a list of floats."""
        c = self.cutoff
        if isinstance(c, (list, tuple)):
            return [float(x) for x in c]
        return [float(c)]


@dataclass
class ViewerConfig:
    """Stage 4: interactive time-driven 1D bridging viewer (standalone HTML).

    Reads the LEF trajectory and renders a self-contained HTML page with a
    playback bar plus four linked panels: E-P proximity, 1D lattice + arcs,
    kymograph (with time cursor), and a bridge map. CTCF / gene / enhancer
    annotations are re-derived from the ``lef`` stage topology, so no extra
    inputs are needed beyond ``LEFPositions.h5``.
    """

    lef_positions_path: str = "trajectory/LEFPositions.h5"
    output_path: str = "trajectory/bridging_viewer.html"

    # Companion exports written next to the HTML. None -> derived from the HTML
    # stem (``<stem>_visited_heatmap.npy`` and ``<stem>_elements.json``).
    heatmap_output_path: Optional[str] = None
    elements_output_path: Optional[str] = None

    # Frame decimation: keep at most ``max_frames`` evenly spaced timepoints
    # (smaller HTML, smoother scrubbing). ``stride`` is applied first.
    stride: int = 1
    max_frames: int = 1000

    # Effective E-P distance: backbone edge = 1 per site, each cohesin loop is
    # a chord edge of cost ``bridge_cost`` (a captured loop ~ one contact).
    bridge_cost: float = 1.0

    # Diamond insulation score window, in lattice sites, for the dynamic
    # per-frame and cumulative bridge-trace line plots.
    insulation_score_window: int = 50

    # Display window of lattice sites [start, end). None = whole lattice.
    site_start: Optional[int] = None
    site_end: Optional[int] = None

    # Static-background render quality.
    kymo_max_rows: int = 1200   # time rows in the kymograph PNG
    dpi: int = 110

    # Optional explicit E-P pairs when the trajectory has no gene annotation:
    # list of {"e": <site>, "p": <site>, "label": <str>}.
    ep_pairs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    lef: LEFConfig = field(default_factory=LEFConfig)
    polymer: PolymerConfig = field(default_factory=PolymerConfig)
    contacts: ContactsConfig = field(default_factory=ContactsConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)


def apply_output_path(
    cfg: PipelineConfig,
    output_path: Union[str, Path],
) -> PipelineConfig:
    """Derive all pipeline input/output paths from one output directory."""

    output_dir = Path(output_path)
    lef_positions = output_dir / "LEFPositions.h5"

    cfg.lef.output_path = str(lef_positions)

    cfg.viewer.lef_positions_path = str(lef_positions)
    cfg.viewer.output_path = str(output_dir / "bridging_viewer.html")
    cfg.viewer.heatmap_output_path = str(
        output_dir / "bridging_viewer_visited_heatmap.npy"
    )
    cfg.viewer.elements_output_path = str(
        output_dir / "bridging_viewer_elements.json"
    )

    cfg.polymer.lef_positions_path = str(lef_positions)
    cfg.polymer.output_folder = str(output_dir)

    cfg.contacts.trajectory_folder = str(output_dir)
    cfg.contacts.raw_output_path = str(output_dir / "contact_map.npy")
    cfg.contacts.oe_output_path = str(output_dir / "contact_map_oe.npy")
    cfg.contacts.viz_output_path = str(output_dir / "contact_map_oe.png")

    return cfg


# ---------------------------------------------------------------------------
# YAML <-> dataclass plumbing
# ---------------------------------------------------------------------------

_PLUGIN_FIELDS = {"plugins"}


def _from_dict(cls, data: Any) -> Any:
    """Recursively build dataclasses from plain mappings."""
    if data is None:
        return cls()
    if cls is PluginSpec:
        return PluginSpec.from_obj(data)
    if not is_dataclass(cls):
        return data
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(data).__name__}")
    kwargs: Dict[str, Any] = {}
    type_hints = {f.name: f.type for f in fields(cls)}
    for key, value in data.items():
        if key not in type_hints:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        field_type = _resolve_type(cls, key)
        kwargs[key] = _coerce(field_type, value)
    return cls(**kwargs)


def _resolve_type(cls, key: str):
    """Return the runtime type of a dataclass field."""

    hints = get_type_hints(cls)
    return hints.get(key)


def _coerce(field_type, value):
    origin = get_origin(field_type)
    args = get_args(field_type)

    if field_type is PluginSpec or (origin is None and isinstance(field_type, type) and issubclass(field_type, PluginSpec)):
        return PluginSpec.from_obj(value)
    if origin is Union:  # Optional[X]
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _coerce(non_none[0], value)
    if is_dataclass(field_type):
        return _from_dict(field_type, value)
    if origin in (list, List):
        inner = args[0] if args else Any
        return [_coerce(inner, v) for v in value]
    if origin in (dict, Dict):
        return dict(value)
    return value


def load_config(
    path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> PipelineConfig:
    """Load a YAML config file into a :class:`PipelineConfig`.

    When ``output_path`` is provided, it is treated as the run output directory
    and all stage path fields are derived from it.
    """
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _from_dict(PipelineConfig, raw)
    if output_path is not None:
        apply_output_path(cfg, output_path)
    return cfg
