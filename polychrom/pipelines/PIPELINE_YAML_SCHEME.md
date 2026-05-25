# Pipeline YAML Scheme

This document describes the **current YAML configuration scheme** for the
loop-extrusion pipeline shipped under `polychrom/pipelines/loop_extrusion/`.

It is a structural reference: every section, key, type, default, and the rules
the loader enforces. The authoritative source is
[`config.py`](loop_extrusion/config.py) — the dataclasses there define the
schema, and this document is generated from them. For the *biophysical
interpretation* of specific parameter values (paper mapping, cohesin density,
processivity), see
[`configs/CONFIG_SCHEME.md`](loop_extrusion/configs/CONFIG_SCHEME.md).

---

## 1. How the config is loaded

`load_config(path)` in [`config.py`](loop_extrusion/config.py):

1. `yaml.safe_load` the file (an empty file is treated as `{}`).
2. Recursively materialise the mapping into nested dataclasses via `_from_dict`.

Key loader rules:

| Rule | Behaviour |
|------|-----------|
| **Unknown key** | Raises `KeyError: Unknown config key '<k>' for <Class>`. Typos are hard errors, not silently ignored. |
| **Missing key** | Falls back to the dataclass default. Any section or key can be omitted. |
| **Empty / missing file** | Produces an all-default `PipelineConfig`. |
| **Wrong type for a dataclass field** | Raises `TypeError` (mapping expected). |
| **`Optional[...]` field set to `null`** | Disables that step (e.g. skip O/E, skip a viz PNG). |

Because unknown keys raise, every key below is the *complete* set the loader
accepts for each section.

---

## 2. Top-level structure

There are **four** top-level sections, each optional:

```yaml
lef:      # Stage 1 — 1D loop-extrusion factor dynamics
viewer:   # Stage 4 (run order: after lef) — standalone HTML 1D viewer
polymer:  # Stage 2 — 3D OpenMM molecular dynamics
contacts: # Stage 3 — contact-map sampling + O/E + heatmap
```

`PipelineConfig` holds one of each: `lef`, `polymer`, `contacts`, `viewer`.

### Stage commands and run order

CLI ([`cli.py`](loop_extrusion/cli.py)):

```bash
python -m polychrom.pipelines.loop_extrusion.cli <stage> config.yaml
```

`<stage>` is one of `lef`, `viewer`, `polymer`, `contacts`, `all`.

The `all` stage runs in this order (note: **viewer runs before polymer** so the
1D dynamics can be inspected before paying for 3D MD):

```text
lef  ->  viewer  ->  polymer  ->  contacts
```

| Stage | Reads | Writes |
|-------|-------|--------|
| `lef` | topology kwargs | `LEFPositions.h5`, kymograph PNG |
| `viewer` | `LEFPositions.h5` + `lef` config | `bridging_viewer.html` |
| `polymer` | `LEFPositions.h5` | `blocks_*.h5` trajectory |
| `contacts` | trajectory folder | raw `.npy`, O/E `.npy`, heatmap PNG |

---

## 3. The plugin mechanism

Every "mechanic" hook is a **`PluginSpec`**: a pointer to a callable plus the
kwargs it is invoked with. In YAML a plugin slot accepts two forms:

**Short form** — a bare string (`module:attr`, no kwargs):

```yaml
load: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:load_one
```

**Long form** — a mapping with `target` (required) and `kwargs` (optional):

```yaml
viz:
  target: polychrom.pipelines.loop_extrusion.plugins.lef_viz:default_kymograph
  kwargs:
    cmap: viridis
    dpi: 150
```

Resolution (`resolve_plugin`):

- `module:attr` — preferred, split on the first `:`.
- `module.attr` — also accepted, split on the last `.`.
- Import error if the attr is missing; `TypeError` if it is not callable.

A plugin slot typed `Optional[PluginSpec]` can be set to `null` to disable that
step entirely.

---

## 4. Stage 1 — `lef`

1D loop-extrusion factor dynamics. Dataclass: `LEFConfig`.

### Scalar keys

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `chain_length` | int | `4000` | Sites per chain. |
| `num_chains` | int | `10` | Number of chains. |
| `separation` | int | `800` | Average sites between LEFs. `num_lefs = num_sites // separation`. |
| `lifetime` | int | `200` | Average LEF lifetime (steps). |
| `lifetime_stalled` | int | `200` | Lifetime when stalled at a non-CTCF obstacle. |
| `warmup_steps` | int | `0` | Dynamics steps discarded before saving (reach steady state). |
| `trajectory_length` | int | `100000` | Number of saved LEF frames (drive the 3D stage). |
| `chunk_size` | int | `50` | LEF frames per HDF5 write chunk (I/O only). |
| `output_path` | str | `trajectory/LEFPositions.h5` | Output LEF trajectory. |
| `seed` | int? | `null` | NumPy RNG seed for reproducible 1D dynamics. |
| `topology_kwargs` | dict | `{}` | Forwarded to the topology plugin (CTCF/TAD layout). |
| `max_rnapii` | int | `64` | Max concurrent RNAPII; sizes HDF5 `rnapii_positions` padding. |

Derived properties (not YAML keys): `num_sites = chain_length * num_chains`,
`num_lefs = num_sites // separation`.

### `lef.plugins` (`LEFPlugins`)

Slots that customise 1D mechanics. Defaults all live in the
`plugins.lef_dynamics` and `plugins.topology` modules.

| Slot | Optional | Default target |
|------|----------|----------------|
| `topology` | no | `topology:uniform_tad_topology` |
| `load` | no | `lef_dynamics:load_one` |
| `unload_prob` | no | `lef_dynamics:unload_prob` |
| `capture` | no | `lef_dynamics:capture` |
| `release` | no | `lef_dynamics:release` |
| `translocate` | no | `lef_dynamics:translocate` |
| `rnapii_load` | yes | `null` |
| `rnapii_translocate` | yes | `null` |

> The `lef` stage writes only `LEFPositions.h5`. To inspect the 1D dynamics
> (kymograph + arcs + E-P proximity), run the `viewer` stage — it renders the
> kymograph as a browser canvas inside the standalone HTML.

> All default targets are under
> `polychrom.pipelines.loop_extrusion.plugins.<module>`.

**RNAPII (transcription) extension.** When `rnapii_load` *and*
`rnapii_translocate` are both set, `lef.run()` also drives transcription each
step. The default `translocate` should then be swapped to
`translocate_with_rnapii` so cohesin honours RNAPII presence:

```yaml
lef:
  plugins:
    topology:
      target: polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_convergent_tad_topology
    translocate:        polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:translocate_with_rnapii
    rnapii_load:        polychrom.pipelines.loop_extrusion.plugins.rnapii:load_rnapii
    rnapii_translocate: polychrom.pipelines.loop_extrusion.plugins.rnapii:stateful_translocate_rnapii
```

RNAPII-aware topologies also accept cohesin-blocking probabilities in
`topology_kwargs`: `rnapii_poised_block_prob` for POISED promoter-bound RNAPII,
`rnapii_paused_block_prob` for PAUSED RNAPII, and
`rnapii_elongating_block_prob` for ELONGATING RNAPII. They default to `1.0`,
matching the previous hard-obstacle behaviour; `rnapii_block_prob` is still
accepted as a shared paused/elongating fallback for older configs.

---

## 5. Stage 2 — `polymer`

3D OpenMM molecular dynamics, driven by the LEF trajectory. Dataclass:
`PolymerConfig`.

### Scalar keys

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `lef_positions_path` | str | `trajectory/LEFPositions.h5` | LEF trajectory input. |
| `output_folder` | str | `trajectory` | Where `blocks_*.h5` are written. |
| `platform` | str | `cuda` | OpenMM platform (`cuda`, `CPU`, `OpenCL`, …). |
| `gpu` | str | `"0"` | GPU device index. |
| `integrator` | str | `variableLangevin` | OpenMM integrator. |
| `error_tol` | float | `0.01` | Variable-timestep error tolerance. |
| `collision_rate` | float | `0.03` | Langevin friction/collision rate. |
| `precision` | str | `mixed` | CUDA precision. |
| `seed` | int? | `null` | NumPy/OpenMM RNG seed for reproducible 3D dynamics. |
| `density` | float | `0.1` | DNA volume fraction; sizes the box (PBC) or confinement. |
| `pbc` | bool | `true` | Periodic boundary conditions. Set `false` when the force builder supplies its own confinement (e.g. spherical). |
| `md_steps_per_block` | int | `750` | OpenMM integrator steps per saved LEF frame. |
| `save_every_blocks` | int | `10` | Save one 3D conformation every N LEF frames. |
| `restart_every_blocks` | int | `100` | Rebuild the OpenMM dynamic bond list every N LEF frames. |
| `initial_relaxation_steps` | int | `0` | MD steps at iteration 0 *before* any cohesin bonds (relax the initial lattice). |
| `pre_recording_steps` | int | `0` | MD steps with bonds in place but unrecorded (equilibrate before first saved block). |
| `smc_bond_wiggle` | float | `0.2` | SMC harmonic bond stiffness (smaller = stiffer). |
| `smc_bond_dist` | float | `0.5` | SMC bond target distance (monomers). |
| `max_data_length` | int | `100` | Saved conformations per HDF5 block file. |
| `overwrite` | bool | `true` | Allow the reporter to overwrite existing output. |

### `polymer.plugins` (`PolymerPlugins`)

| Slot | Optional | Default target |
|------|----------|----------------|
| `force_builder` | no | `forces:default_force_builder` |
| `initial_conformation` | no | `forces:grow_cubic_conformation` |

`force_builder` kwargs configure the OpenMM force field. The
`default_force_builder` accepts e.g. `bond_length`, `bond_wiggle`, `angle_k`,
`repulsive_trunc`, `repulsive_radius_mult`. The biophysical configs use
`paper_force_builder` (spherical confinement, `selective_SSW` excluded volume,
optional cognate E-P attraction via `ep_pairs`).

---

## 6. Stage 3 — `contacts`

Contact-map sampling, O/E normalisation, heatmap. Dataclass: `ContactsConfig`.

### Scalar keys

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `trajectory_folder` | str | `trajectory` | Folder of saved 3D conformations. |
| `raw_output_path` | str | `trajectory/contact_map.npy` | Raw observed map (always written). |
| `oe_output_path` | str | `trajectory/contact_map_oe.npy` | O/E map (written when `plugins.obs_over_exp` is set). |
| `viz_output_path` | str | `trajectory/contact_map_oe.png` | Heatmap PNG (written when `plugins.viz` is set). |
| `map_starts` | list[int] | `range(0, 39000, 4000)` | Window start monomers. Use multiple small windows for large loci. |
| `map_size` | int | `4000` | Window size (monomers). |
| `cutoff` | float | `6.0` | Contact distance threshold (a contact when 3D distance ≤ cutoff). |
| `num_processes` | int | `6` | CPU multiprocessing for counting (does not affect GPU MD). |
| `verbose` | bool | `true` | Verbose logging. |

### `contacts.plugins` (`ContactsPlugins`)

| Slot | Optional | Default target |
|------|----------|----------------|
| `sampler` | no | `sampling:monomer_resolution_sampler` |
| `obs_over_exp` | yes (`null` skips O/E) | `sampling:observed_over_expected` |
| `post_process` | yes | `null` (e.g. log / clip after O/E) |
| `viz` | yes (`null` skips PNG) | `sampling:default_oe_heatmap` |

The biophysical configs swap `obs_over_exp` to
`sampling:balanced_observed_over_expected` (ICE balancing then per-diagonal O/E).

---

## 7. Stage 4 — `viewer`

Interactive, time-driven 1D bridging viewer rendered as a **standalone HTML
page** (no external inputs beyond `LEFPositions.h5`). CTCF / gene / enhancer
annotations are re-derived from the `lef` stage topology. Dataclass:
`ViewerConfig`.

The page has a playback bar plus linked panels: E-P proximity, 1D lattice +
arcs/RNAPII annotations, kymograph (with a time cursor), bridge map, and
dynamic insulation-score line plots.

### Scalar keys

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `lef_positions_path` | str | `trajectory/LEFPositions.h5` | LEF trajectory input. |
| `output_path` | str | `trajectory/bridging_viewer.html` | Output HTML page. |
| `stride` | int | `1` | Frame stride applied first. |
| `max_frames` | int | `1000` | Keep at most this many evenly spaced frames (smaller HTML, smoother scrub). |
| `bridge_cost` | float | `1.0` | E-P distance: backbone edge = 1/site; each cohesin loop is a chord edge of this cost. |
| `insulation_score_window` | int | `50` | Diamond insulation window, in lattice sites, for the dynamic line plots. The per-frame plot uses current bridge contacts; the cumulative plot uses the visited bridge trace summed through the current frame. |
| `site_start` | int? | `null` | Display window start. `null` = whole lattice. |
| `site_end` | int? | `null` | Display window end. |
| `kymo_max_rows` | int | `1200` | Time rows in the static kymograph PNG background. |
| `dpi` | int | `110` | Static-background render DPI. |
| `ep_pairs` | list[dict] | `[]` | Explicit E-P pairs when the trajectory has no gene annotation: `{e: <site>, p: <site>, label: <str>}`. |

The `viewer` stage has **no `plugins` block**.

---

## 8. Minimal example

Smallest useful config (single chain, default plugins via omission). Anything
not listed falls back to the dataclass default:

```yaml
lef:
  chain_length: 800
  num_chains: 1
  separation: 60
  trajectory_length: 3000
  output_path: trajectory_min/LEFPositions.h5
  topology_kwargs:
    tad_positions: [100, 250, 400, 550, 700]

polymer:
  lef_positions_path: trajectory_min/LEFPositions.h5
  output_folder: trajectory_min
  platform: CPU
  density: 0.15
  md_steps_per_block: 200
  save_every_blocks: 3
  restart_every_blocks: 300

contacts:
  trajectory_folder: trajectory_min
  raw_output_path: trajectory_min/contact_map.npy
  map_starts: [0]
  map_size: 800
  cutoff: 5.5

viewer:
  lef_positions_path: trajectory_min/LEFPositions.h5
  output_path: trajectory_min/bridging_viewer.html
  insulation_score_window: 50
```

For a complete annotated example see
[`configs/default.yaml`](loop_extrusion/configs/default.yaml) and
[`configs/minimal.yaml`](loop_extrusion/configs/minimal.yaml).

---

## 9. Shipped config files

Under [`loop_extrusion/configs/`](loop_extrusion/configs/):

| File | Purpose |
|------|---------|
| `default.yaml` | Mirrors the example notebooks; full plugin block spelled out. |
| `minimal.yaml` | CPU smoke test (~5–6 min end-to-end). |
| `experiment1_tads_only_biophysical.yaml` | Directional CTCF TADs + cohesin, no RNAPII. |
| `experiment2_tads_transcription_biophysical.yaml` | Adds RNAPII dynamics + cognate E-P attraction. |
| `experiment1_tads_only.yaml`, `..._hq.yaml` | Toy / high-quality TAD-only variants. |
| `experiment2_tads_transcription.yaml`, `..._hq.yaml` | Toy / high-quality transcription variants. |
| `rnapii_example.yaml`, `rnapii_v2_example.yaml` | RNAPII plugin examples. |
| `paper_NRMCB.yaml` | Paper-scale reference (multi-window contact maps). |

---

## 10. Quick reference — defaults at a glance

```yaml
lef:
  chain_length: 4000
  num_chains: 10
  separation: 800
  lifetime: 200
  lifetime_stalled: 200
  warmup_steps: 0
  trajectory_length: 100000
  chunk_size: 50
  output_path: trajectory/LEFPositions.h5
  seed: null
  topology_kwargs: {}
  max_rnapii: 64
  plugins:
    topology:    polychrom.pipelines.loop_extrusion.plugins.topology:uniform_tad_topology
    load:        polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:load_one
    unload_prob: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:unload_prob
    capture:     polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:capture
    release:     polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:release
    translocate: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:translocate
    rnapii_load: null
    rnapii_translocate: null

polymer:
  lef_positions_path: trajectory/LEFPositions.h5
  output_folder: trajectory
  platform: cuda
  gpu: "0"
  integrator: variableLangevin
  error_tol: 0.01
  collision_rate: 0.03
  precision: mixed
  seed: null
  density: 0.1
  pbc: true
  md_steps_per_block: 750
  save_every_blocks: 10
  restart_every_blocks: 100
  initial_relaxation_steps: 0
  pre_recording_steps: 0
  smc_bond_wiggle: 0.2
  smc_bond_dist: 0.5
  max_data_length: 100
  overwrite: true
  plugins:
    force_builder:        polychrom.pipelines.loop_extrusion.plugins.forces:default_force_builder
    initial_conformation: polychrom.pipelines.loop_extrusion.plugins.forces:grow_cubic_conformation

contacts:
  trajectory_folder: trajectory
  raw_output_path: trajectory/contact_map.npy
  oe_output_path: trajectory/contact_map_oe.npy
  viz_output_path: trajectory/contact_map_oe.png
  map_starts: [0, 4000, 8000, ... , 36000]   # range(0, 39000, 4000)
  map_size: 4000
  cutoff: 6.0
  num_processes: 6
  verbose: true
  plugins:
    sampler:      polychrom.pipelines.loop_extrusion.plugins.sampling:monomer_resolution_sampler
    obs_over_exp: polychrom.pipelines.loop_extrusion.plugins.sampling:observed_over_expected
    post_process: null
    viz:          polychrom.pipelines.loop_extrusion.plugins.sampling:default_oe_heatmap

viewer:
  lef_positions_path: trajectory/LEFPositions.h5
  output_path: trajectory/bridging_viewer.html
  stride: 1
  max_frames: 1000
  bridge_cost: 1.0
  insulation_score_window: 50
  site_start: null
  site_end: null
  kymo_max_rows: 1200
  dpi: 110
  ep_pairs: []
```
