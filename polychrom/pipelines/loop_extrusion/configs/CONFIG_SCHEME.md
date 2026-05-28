# Loop Extrusion Config Scheme

This document describes the current YAML layout for the modular
loop-extrusion pipeline. It is intentionally schema-oriented: it explains the
sections, fields, plugin hooks, runtime path rules, and common mechanics without
binding the scheme to any one config file, biological condition, or experiment
name.

The source of truth is the typed configuration in
[`../config.py`](../config.py). Stage behavior lives in
[`../lef.py`](../lef.py), [`../viewer.py`](../viewer.py),
[`../polymer.py`](../polymer.py), [`../contacts.py`](../contacts.py),
[`../qc.py`](../qc.py), and [`../compare.py`](../compare.py). Built-in plugin
implementations live under [`../plugins/`](../plugins/).

## Top-Level Layout

A pipeline config is a YAML mapping with these top-level sections:

```yaml
lef:      # 1D cohesin, CTCF, optional RNAPII, optional lesions
viewer:   # standalone interactive HTML derived from the 1D trajectory
polymer:  # 3D OpenMM simulation driven by the LEF trajectory
contacts: # contact-map sampling, normalization, and rendering
```

Every section is optional. Missing sections and missing keys fall back to the
dataclass defaults in `config.py`. An empty YAML file therefore means "run the
whole pipeline with defaults".

The generic shape is:

```yaml
lef:
  chain_length: <sites per chain>
  num_chains: <number of independent chains/replicates>
  separation: <average sites per cohesin>
  lifetime: <cohesin lifetime in 1D ticks>
  lifetime_stalled: <lifetime when stalled by non-CTCF obstacles>
  warmup_steps: <discarded 1D ticks before recording>
  trajectory_length: <recorded 1D ticks>
  chunk_size: <number of HDF5 write chunks>
  seed: <integer or null>
  max_rnapii: <recorded RNAPII slots, when RNAPII is enabled>
  topology_kwargs:
    <kwargs for selected topology plugin>
  plugins:
    topology: <PluginSpec>
    load: <PluginSpec>
    unload_prob: <PluginSpec>
    capture: <PluginSpec>
    release: <PluginSpec>
    translocate: <PluginSpec>
    rnapii_load: <PluginSpec or null>
    rnapii_translocate: <PluginSpec or null>
    lesion: <PluginSpec or null>

viewer:
  stride: <frame stride before decimation>
  max_frames: <maximum frames embedded in HTML; 0 means no max>
  bridge_cost: <shortest-path cost of one cohesin chord>
  insulation_score_window: <window in lattice sites>
  site_start: <first displayed site or null>
  site_end: <exclusive last displayed site or null>
  ep_pairs:
    - {e: <enhancer site>, p: <promoter site>, label: <optional label>}

polymer:
  platform: cuda
  gpu: "0"
  integrator: variableLangevin
  density: <density used for box sizing>
  pbc: <true for periodic cube, false for force-builder confinement>
  md_steps_per_block: <OpenMM steps per LEF frame>
  save_every_blocks: <save one conformation every N LEF frames>
  restart_every_blocks: <dynamic-bond rebuild interval in LEF frames>
  plugins:
    force_builder:
      target: <callable>
      kwargs:
        <kwargs for selected force builder>
    initial_conformation: <PluginSpec>

contacts:
  map_starts: [<window starts>]
  replicate_map_starts_across_chains: <true/false>
  map_size: <square map size>
  cutoff: <3D distance cutoff>
  num_processes: <CPU workers>
  plugins:
    sampler: <PluginSpec>
    obs_over_exp: <PluginSpec or null>
    post_process: <PluginSpec or null>
    viz: <PluginSpec or null>
```

Values are in pipeline units. A lattice site in `lef` corresponds to one
polymer monomer in `polymer` and one pixel/bin unit before any contact-map
coarsening. If a config interprets one site as one kilobase, that is a modeling
choice made by that config, not a separate schema field.

## Loading Rules

`load_config(path)` reads YAML into nested dataclasses:

| Situation | Behavior |
| --- | --- |
| Unknown dataclass key | Hard error: `KeyError: Unknown config key '<key>' for <Class>`. |
| Missing key or section | Dataclass default is used. |
| Empty YAML file | Equivalent to all defaults. |
| Required plugin set to `null` | Error, because `PluginSpec` is required. |
| Optional plugin set to `null` | The optional step is skipped. |
| `topology_kwargs` or plugin `kwargs` | Stored as plain dictionaries and passed to the selected callable. |

The loader validates the fixed schema keys, not the contents of arbitrary
`kwargs` dictionaries. If a kwarg does not match the callable selected by a
plugin, the error is raised when that stage calls the plugin.

## Runtime Paths

Path fields exist in the dataclasses, but the CLI is usually run with an output
directory:

```bash
python -m polychrom.pipelines.loop_extrusion.cli <stage> config.yaml output_dir
```

When `output_dir` is supplied, all stage input/output paths are derived from
that one directory:

| Field | Runtime value |
| --- | --- |
| `lef.output_path` | `output_dir/LEFPositions.h5` |
| `viewer.lef_positions_path` | `output_dir/LEFPositions.h5` |
| `viewer.output_path` | `output_dir/bridging_viewer.html` |
| `viewer.heatmap_output_path` | `output_dir/bridging_viewer_visited_heatmap.npy` |
| `viewer.elements_output_path` | `output_dir/bridging_viewer_elements.json` |
| `polymer.lef_positions_path` | `output_dir/LEFPositions.h5` |
| `polymer.output_folder` | `output_dir` |
| `contacts.trajectory_folder` | `output_dir` |
| `contacts.raw_output_path` | `output_dir/contact_map.npy` |
| `contacts.oe_output_path` | `output_dir/contact_map_oe.npy` |
| `contacts.viz_output_path` | `output_dir/contact_map_oe.png` |

The CLI also copies the input config into `output_dir` before running. Prefer
the runtime `output_dir` argument for run location, and use YAML path fields
only when a stage must read or write nonstandard files.

## CLI Stages

```bash
python -m polychrom.pipelines.loop_extrusion.cli lef      config.yaml [output_dir]
python -m polychrom.pipelines.loop_extrusion.cli viewer   config.yaml [output_dir]
python -m polychrom.pipelines.loop_extrusion.cli polymer  config.yaml [output_dir]
python -m polychrom.pipelines.loop_extrusion.cli contacts config.yaml [output_dir]
python -m polychrom.pipelines.loop_extrusion.cli qc       config.yaml [output_dir]
python -m polychrom.pipelines.loop_extrusion.cli all      config.yaml [output_dir]
python -m polychrom.pipelines.loop_extrusion.cli compare  cfgA.yaml cfgB.yaml [options]
```

`all` runs:

```text
lef -> viewer -> polymer -> contacts -> qc
```

The viewer runs before the 3D stage so the 1D dynamics can be inspected before
OpenMM work. The `viewer` stage can generate `LEFPositions.h5` itself if it is
missing. The `polymer`, `contacts`, and `qc` stages require their upstream
outputs to already exist unless they are run through `all`.

The `compare` command is not configured by a top-level YAML section. It loads
two normal pipeline configs, optionally overrides their run folders, and writes
pairwise metrics and plots into a comparison directory.

## PluginSpec

Every pluggable mechanism is a `PluginSpec`: a callable target plus optional
keyword arguments.

Short form:

```yaml
load: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:load_one
```

Long form:

```yaml
force_builder:
  target: polychrom.pipelines.loop_extrusion.plugins.forces:paper_force_builder
  kwargs:
    repulsion_energy: 50.0
```

Targets use `module:attribute` form. `module.attribute` is accepted too, but
`module:attribute` is clearer because module paths and function names are
separated unambiguously.

Optional plugin slots may be `null`:

```yaml
contacts:
  plugins:
    obs_over_exp: null  # write only the raw contact map
    viz: null           # skip PNG rendering
```

Plugin kwargs are always passed at call time. Use kwargs that belong to the
selected plugin implementation.

## Pipeline Data Flow

```text
YAML config
  |
  v
1D LEF stage
  output: LEFPositions.h5
    - positions: recorded cohesin leg pairs
    - optional rnapii_positions / rnapii_states
    - optional lesions
    - attributes describing lattice size and enabled features
  |
  v
viewer stage
  output: bridging_viewer.html
  output: bridging_viewer_visited_heatmap.npy
  output: bridging_viewer_elements.json
  |
  v
3D polymer stage
  input: LEFPositions.h5
  output: HDF5 trajectory block files
  |
  v
contacts stage
  input: 3D trajectory block files
  output: contact_map.npy
  output: optional contact_map_oe.npy
  output: optional contact_map_oe.png
  |
  v
qc stage
  input: LEFPositions.h5 and optional contact_map.npy / contact_map_oe.npy
  output: qc/metrics.json
  output: qc/report.md
  output: qc/plots/*.png
```

## `lef` Section

The `lef` section controls the 1D lattice simulation.

### Fields

| Field | Default | Description |
| --- | ---: | --- |
| `chain_length` | `4000` | Number of lattice sites per chain. |
| `num_chains` | `10` | Number of chains. Many configs use this as independent replicate chains. |
| `separation` | `800` | Average sites per cohesin; `num_lefs = chain_length * num_chains // separation`. |
| `lifetime` | `200` | Average cohesin lifetime in 1D ticks. |
| `lifetime_stalled` | `200` | Lifetime used when a cohesin is stalled by a non-CTCF obstacle. |
| `warmup_steps` | `0` | 1D ticks to run before recording; discarded from output. |
| `trajectory_length` | `100000` | Number of recorded 1D frames. This also drives the number of LEF frames available to OpenMM. |
| `chunk_size` | `50` | Number of HDF5 write chunks used while recording `positions`. |
| `output_path` | `trajectory/LEFPositions.h5` | Written HDF5 path; overridden by CLI `output_dir`. |
| `seed` | `null` | Optional NumPy RNG seed for reproducible 1D dynamics. |
| `topology_kwargs` | `{}` | Keyword arguments passed to `lef.plugins.topology`. |
| `max_rnapii` | `64` | Number of RNAPII slots recorded per frame when RNAPII dynamics are enabled. |
| `plugins` | default `LEFPlugins` | Plugin slots for topology and 1D mechanics. |

`chain_length * num_chains` is the total lattice size. Cohesin count is derived
by integer division; it is not configured directly.

### Default LEF Plugins

| Slot | Default | Purpose |
| --- | --- | --- |
| `topology` | `plugins.topology:uniform_tad_topology` | Builds the CTCF layout and any extra 1D bookkeeping. |
| `load` | `plugins.lef_dynamics:load_one` | Loads one cohesin on adjacent free sites. |
| `unload_prob` | `plugins.lef_dynamics:unload_prob` | Computes per-tick unload probability for one cohesin. |
| `capture` | `plugins.lef_dynamics:capture` | Captures cohesin legs at CTCF sites. |
| `release` | `plugins.lef_dynamics:release` | Releases captured CTCF legs. |
| `translocate` | `plugins.lef_dynamics:translocate` | Advances cohesin dynamics one tick. |
| `rnapii_load` | `null` | Optional RNAPII loading step. |
| `rnapii_translocate` | `null` | Optional RNAPII state/translocation step. |
| `lesion` | `null` | Optional lesion update step. |

RNAPII dynamics are enabled only when both `rnapii_load` and
`rnapii_translocate` are non-null. Lesion dynamics are enabled when `lesion` is
non-null. These feature flags control which extra HDF5 datasets are written.

### Topology Plugins

A topology plugin is called as:

```python
topology_fn(lef_cfg, **lef_cfg.topology_kwargs)
```

It returns an `args` dictionary consumed by load, capture, release,
translocation, RNAPII, and lesion plugins. All built-in topology plugins create
the base fields:

```text
N
chain_length
num_chains
LIFETIME
LIFETIME_STALLED
ctcfCapture
ctcfRelease
```

Built-in topology choices:

| Plugin | Main kwargs | Description |
| --- | --- | --- |
| `uniform_tad_topology` | `tad_positions`, `capture_prob`, `release_prob`, `symmetric` | Repeats the same CTCF positions on each chain. With `symmetric: true`, each CTCF can capture both leg directions. |
| `convergent_tad_topology` | `tad_positions`, `boundary_strength`, `release_prob`, `include_chromosome_ends` | Splits each chain at TAD boundaries and places inward-facing barriers on interval edges. |
| `gene_aware_topology` | uniform TAD kwargs plus `genes`, RNAPII, loading, and lesion kwargs | Uniform/symmetric CTCF layout plus gene/RNAPII/lesion bookkeeping. |
| `gene_aware_convergent_tad_topology` | convergent TAD kwargs plus `genes`, RNAPII, loading, and lesion kwargs | Directional CTCF layout plus gene/RNAPII/lesion bookkeeping. |
| `ep_pair_topology` | `n_pairs`, `ep_distance`, `pair_spacing`, `first_pair_offset`, `boundary_strength`, `convergent_orientation` | Programmatically places enhancer-promoter pairs and flanking CTCF barriers. |
| `explicit_ctcf_topology` | `left_capture`, `right_capture`, `left_release`, `right_release` | Uses user-supplied CTCF dictionaries directly. |

`tad_positions` are interior boundaries in chain-relative coordinates for the
TAD topology plugins. Boundaries are applied to every chain by adding
`chain_idx * chain_length`.

Directional convention:

```text
left-moving cohesin leg  = side -1
right-moving cohesin leg = side +1
```

In a convergent TAD interval, the left interval edge captures the left-moving
leg and the right interval edge captures the right-moving leg, so a cohesin
loaded inside the interval can become bracketed by inward-facing CTCF sites.

### Gene-Aware Topology

The `gene_aware_*` topology plugins add transcription units and optional
targeted loading / lesion state. Gene IDs are assigned from list order after any
replication across chains. A `gene_id` key in YAML is not required by the
builder.

Per-gene fields:

| Field | Required | Default | Description |
| --- | --- | --- | --- |
| `tss` | yes | none | Transcription start site and RNAPII loading site. |
| `tes` | yes | none | Transcription end site. `tes > tss` gives direction `+1`; `tes < tss` gives `-1`. |
| `load_prob` | no | `rnapii_default_load_prob` | Per-tick probability of loading one POISED RNAPII onto a free TSS. |
| `enhancer_pos` | no | `null` | Cognate enhancer site used for E-P contact tests and targeted loading. |
| `requires_enhancer` | no | `false` | If true, PAUSED to ELONGATING transition requires current E-P contact. |
| `load_requires_enhancer` | no | `false` | If true, RNAPII recruitment to the TSS also requires current E-P contact. |
| `initiation_prob` | no | `1.0` | POISED to PAUSED probability per tick. |
| `pause_release_prob` | no | `1.0` | PAUSED to ELONGATING probability per eligible tick. |
| `elongation_step_prob` | no | `1.0` | Probability an ELONGATING RNAPII advances one site per tick. |
| `pause_offset` | no | `0` | Optional promoter-proximal pause offset from the TSS. |
| `termination_prob` | no | `1.0` | TERMINATING unload probability per tick after reaching TES. |

Common `gene_aware_*` kwargs:

| Kwarg | Default | Description |
| --- | ---: | --- |
| `genes` | `null` | List of gene dictionaries. |
| `replicate_genes_across_chains` | `false` | Treat gene coordinates as chain-relative and copy them to each chain. |
| `rnapii_default_load_prob` | `0.02` | Default gene `load_prob`. |
| `rnapii_stride` | `1` | Step count for the legacy single-state RNAPII translocator. |
| `rnapii_stall_prob` | `0.4` | Intrinsic probability that elongating RNAPII stalls on a cohesin obstacle. |
| `rnapii_push_prob` | `0.3` | Probability that an elongating RNAPII pushes a co-directional cohesin leg. |
| `rnapii_headon_push_prob` | `0.0` | Probability that an elongating RNAPII pushes a head-on cohesin leg. |
| `rnapii_poised_block_prob` | `1.0` | Probability that POISED RNAPII blocks an incoming cohesin leg. |
| `rnapii_paused_block_prob` | `rnapii_block_prob` | Probability that PAUSED RNAPII blocks an incoming cohesin leg. |
| `rnapii_elongating_block_prob` | `rnapii_block_prob` | Probability that ELONGATING RNAPII blocks an incoming cohesin leg. |
| `rnapii_terminating_block_prob` | paused block probability | Probability that TERMINATING RNAPII blocks an incoming cohesin leg. |
| `rnapii_block_prob` | `1.0` | Backward-compatible fallback for state-specific block probabilities. |
| `ep_contact_tolerance` | `2` | Slack in the cohesin-loop E-P containment test. |
| `targeted_load_prob` | `0.0` | Probability that `load_targeted` loads at a gene-derived loading site. |
| `loading_window` | `2` | Search window around a targeted loading site. |
| `target_enhancers` | `true` | Include enhancer sites as targeted loading sites. |
| `target_tss` | `true` | Include TSS sites as targeted loading sites. |
| `lesion_prob` | `0.0` | Per-gene, per-tick stochastic lesion occurrence probability. |
| `lesion_lifetime` | `100` | Lesion countdown before repair. |
| `lesion_block_prob` | `0.95` | Probability that a lesion blocks an incoming cohesin leg. |
| `lesion_max` | `64` | Maximum lesion sites recorded per frame. |
| `lesion_spacing` | `0` | If positive, seed periodic lesions in gene bodies at topology creation. |

### RNAPII Dynamics

To drive RNAPII, set both RNAPII plugin slots and use a gene-aware topology:

```yaml
lef:
  plugins:
    topology: polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_convergent_tad_topology
    translocate: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:translocate_with_rnapii
    rnapii_load: polychrom.pipelines.loop_extrusion.plugins.rnapii:load_rnapii
    rnapii_translocate: polychrom.pipelines.loop_extrusion.plugins.rnapii:stateful_translocate_rnapii
```

`stateful_translocate_rnapii` uses four states:

| State | Meaning |
| --- | --- |
| `POISED` | Loaded at TSS, not yet initiated. |
| `PAUSED` | Initiated but promoter-proximal paused. |
| `ELONGATING` | Productive one-site-at-a-time transcription. |
| `TERMINATING` | Reached TES and may dwell before unloading. |

E-P contact during the 1D stage is a loop-containment proxy. A gene with
`enhancer_pos` is considered in contact when at least one cohesin loop brackets
the enhancer and promoter, with `ep_contact_tolerance` sites of slack. This is
used for `requires_enhancer` and `load_requires_enhancer`.

`translocate_with_rnapii` is also the built-in translocator that knows about
lesion barriers. It can be used even when RNAPII plugin slots are `null`, as
long as the selected topology supplies the needed gene/lesion bookkeeping.

### Targeted Cohesin Loading

`load_targeted` biases cohesin loading toward `args["loading_sites"]`, which
the gene-aware topology can populate from enhancer and/or TSS coordinates.

```yaml
lef:
  plugins:
    load: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:load_targeted
  topology_kwargs:
    targeted_load_prob: <0.0 to 1.0>
    loading_window: <sites around target>
    target_enhancers: true
    target_tss: true
```

With probability `targeted_load_prob`, a new cohesin is placed on a free
adjacent pair near a loading site. If no local slot is free, or if the
probability draw fails, loading falls back to uniform `load_one`. Loading is
kept inside the target site's chain.

### Lesion Dynamics

To update and record lesions, set the lesion plugin:

```yaml
lef:
  plugins:
    lesion: polychrom.pipelines.loop_extrusion.plugins.lesions:update_lesions
```

The gene-aware topology supplies lesion parameters and initializes
`args["lesions"]`. Lesions are damaged lattice sites in gene bodies. They can be
stochastic (`lesion_prob`) or seeded at regular gene-body spacing
(`lesion_spacing`). The update step repairs existing lesions by decrementing
their lifetime, then may add new lesions.

Built-in collision behavior:

* RNAPII cannot step onto a lesion and stalls upstream.
* `translocate_with_rnapii` lets a lesion block an incoming cohesin leg with
  probability `lesion_block_prob`.
* The default `translocate` plugin does not check lesion state.

## `viewer` Section

The viewer stage builds a self-contained HTML page for the 1D trajectory. It
also writes a cumulative bridge-contact heatmap and a JSON annotation export.

### Fields

| Field | Default | Description |
| --- | ---: | --- |
| `lef_positions_path` | `trajectory/LEFPositions.h5` | Input 1D trajectory; overridden by CLI `output_dir`. |
| `output_path` | `trajectory/bridging_viewer.html` | HTML output path; overridden by CLI `output_dir`. |
| `heatmap_output_path` | `null` | Optional explicit `.npy` export path. If null, derived from HTML stem. |
| `elements_output_path` | `null` | Optional explicit `.json` annotation export path. If null, derived from HTML stem. |
| `stride` | `1` | Keep every Nth frame before `max_frames` decimation. |
| `max_frames` | `1000` | Maximum frames embedded in the HTML; `0` means no maximum. |
| `bridge_cost` | `1.0` | Graph cost of a cohesin chord for effective E-P distance. |
| `insulation_score_window` | `50` | Window in lattice sites for dynamic insulation traces. |
| `site_start` | `null` | First displayed absolute lattice site. |
| `site_end` | `null` | Exclusive last displayed absolute lattice site. |
| `kymo_max_rows` | `1200` | ViewerConfig render-quality field retained for kymograph/background generation. |
| `dpi` | `110` | ViewerConfig render-quality field retained for static image generation. |
| `ep_pairs` | `[]` | Explicit E-P pairs used only when topology supplies no gene annotation. |

The viewer re-derives CTCF, gene, enhancer, and TAD annotations from the `lef`
topology and `topology_kwargs`. Coordinates in exported JSON are display-window
relative, with `siteOffset` giving the absolute offset.

Effective E-P distance is the shortest path on a graph whose backbone edges
have cost equal to genomic separation and whose cohesin loops add shortcut
chords of cost `bridge_cost`.

The viewer has no plugin section.

## `polymer` Section

The polymer stage reads `LEFPositions.h5`, converts each frame's cohesin pairs
into temporary harmonic bonds, and runs OpenMM molecular dynamics.

### Fields

| Field | Default | Description |
| --- | ---: | --- |
| `lef_positions_path` | `trajectory/LEFPositions.h5` | Input 1D trajectory path; overridden by CLI `output_dir`. |
| `output_folder` | `trajectory` | Output folder for HDF5 trajectory blocks; overridden by CLI `output_dir`. |
| `platform` | `cuda` | OpenMM platform name. |
| `gpu` | `"0"` | CUDA GPU selector passed as `GPU`. |
| `integrator` | `variableLangevin` | OpenMM integrator name used by `Simulation`. |
| `error_tol` | `0.01` | Error tolerance for variable-step integrators. |
| `collision_rate` | `0.03` | Langevin collision/friction rate. |
| `precision` | `mixed` | CUDA precision mode. |
| `seed` | `null` | Optional NumPy/OpenMM seed. Per restart chunk, OpenMM seed is offset by iteration. |
| `density` | `0.1` | Density used to size the initial box and optional periodic box. |
| `pbc` | `true` | If true, run in a periodic cubic box. Set false when the force builder supplies confinement. |
| `md_steps_per_block` | `750` | OpenMM integrator steps per LEF frame. |
| `save_every_blocks` | `10` | Save one 3D conformation every N LEF frames. |
| `restart_every_blocks` | `100` | Number of LEF frames in each dynamic-bond rebuild chunk. |
| `initial_relaxation_steps` | `0` | Bare-polymer relaxation before SMC bonds are inserted. |
| `pre_recording_steps` | `0` | Dynamics with SMC bonds before the first recorded block. |
| `smc_bond_wiggle` | `0.2` | SMC harmonic-bond wiggle distance; smaller means stiffer. |
| `smc_bond_dist` | `0.5` | SMC bond rest length in simulation length units. |
| `max_data_length` | `100` | Maximum saved conformations per HDF5 block file. |
| `overwrite` | `true` | Whether the reporter may overwrite existing trajectory output. |
| `plugins` | default `PolymerPlugins` | Force builder and initial conformation hooks. |

Hard requirements:

* `trajectory_length` in the LEF HDF5 must be a multiple of
  `restart_every_blocks`.
* `restart_every_blocks` must be a multiple of `save_every_blocks`.

The dynamic bond updater looks ahead within each restart chunk, creates every
SMC bond needed in that chunk, and then toggles those bonds active/inactive by
changing force parameters in the OpenMM context.

### Polymer Plugins

| Slot | Default | Purpose |
| --- | --- | --- |
| `force_builder` | `plugins.forces:default_force_builder` | Adds backbone, angle, nonbonded, confinement, and other forces to the simulation. |
| `initial_conformation` | `plugins.forces:grow_cubic_conformation` | Creates initial `(N, 3)` coordinates from `num_sites` and `box`. |

`force_builder` is called as:

```python
force_builder(sim, num_chains=num_chains, chain_length=chain_length, **kwargs)
```

`initial_conformation` is called as:

```python
initial_conformation(num_sites=n_sites, box=box, **kwargs)
```

### Built-In Force Builders

`default_force_builder` builds separate linear chains with harmonic backbone
bonds, angle force, and polynomial repulsion.

Common kwargs:

| Kwarg | Default | Description |
| --- | ---: | --- |
| `bond_length` | `1.0` | Backbone bond rest length. |
| `bond_wiggle` | `0.1` | Backbone bond flexibility. |
| `angle_k` | `1.5` | Angle-force stiffness. |
| `repulsive_trunc` | `1.5` | Polynomial repulsion truncation. |
| `repulsive_radius_mult` | `1.05` | Repulsive radius multiplier. |
| `restrict_nonbonded_to_chains` | `false` | Restrict nonbonded interactions to pairs within the same chain. |

`paper_force_builder` builds linear chains with harmonic backbone bonds, optional
angle force, selective soft-core nonbonded interactions, optional E-P pair
stickiness, optional Pol II self-affinity, and spherical confinement.

Common kwargs:

| Kwarg | Default | Description |
| --- | ---: | --- |
| `sticky_particles` | `[]` | Flat list of sticky monomer indices for the `selective_SSW` path. |
| `ep_pairs` | `[]` | Pair list `[[enhancer, promoter], ...]`; enables pair-specific `heteropolymer_SSW` attraction. |
| `replicate_ep_pairs_across_chains` | `false` | Treat E-P pair coordinates as chain-relative and copy to each chain. |
| `extra_hard_particles` | `[]` | Monomers marked extra hard in the nonbonded force. |
| `transcribed_particles` | `[]` | Usually filled by the polymer stage when Pol II self-affinity is active. |
| `polii_self_affinity` | `0.0` | Adds a transcribed monomer type with self-attraction as a multiple of `selective_attraction_energy`. |
| `bond_length` | `1.0` | Backbone bond rest length. |
| `bond_wiggle` | `0.1` | Backbone bond flexibility. |
| `angle_k` | `null` | Angle stiffness; `null` disables angle force. |
| `repulsion_energy` | `50.0` | Soft excluded-volume barrier height. |
| `repulsion_radius` | `1.05` | Repulsion radius. |
| `attraction_energy` | `0.0` | Generic all-monomer attraction depth. |
| `attraction_radius` | `2.0` | Generic attraction radius. |
| `selective_attraction_energy` | `1.0` | Pair/type-specific attraction strength. |
| `selective_repulsion_energy` | `0.0` | Pair/type-specific repulsion strength. |
| `confinement_density` | `0.2` | Density used for spherical confinement. |
| `confinement_k` | `5.0` | Confinement wall stiffness. |
| `restrict_nonbonded_to_chains` | `false` | Restrict nonbonded interactions to same-chain pairs. |
| `confinement_per_chain` | `false` | Add one spherical confinement per chain instead of one global sphere. |

When `ep_pairs` is non-empty, `paper_force_builder` assigns each listed sticky
site a distinct type and turns on attraction only between listed cognate pairs.
Shared enhancer or promoter sites are allowed by listing multiple pairs.

When `polii_self_affinity > 0`, the polymer stage enables transcription-driven
compaction only if the LEF file contains RNAPII and gene datasets. It marks the
gene bodies of genes that currently carry at least one ELONGATING RNAPII with a
transcribed monomer type for that frame. With `num_chains > 1`, this feature
requires `restrict_nonbonded_to_chains: true` so replicate chains do not
self-attract across chains.

## `contacts` Section

The contact stage reads saved 3D trajectory blocks, builds a raw contact map,
optionally normalizes it, optionally post-processes it, and optionally renders a
PNG.

### Fields

| Field | Default | Description |
| --- | ---: | --- |
| `trajectory_folder` | `trajectory` | Folder containing 3D HDF5 block files; overridden by CLI `output_dir`. |
| `raw_output_path` | `trajectory/contact_map.npy` | Raw contact map output. |
| `oe_output_path` | `trajectory/contact_map_oe.npy` | O/E or post-processed output path. |
| `viz_output_path` | `trajectory/contact_map_oe.png` | Heatmap output path. |
| `map_starts` | `[0, 4000, ..., 36000]` | Subchain starts used by the sampler. |
| `replicate_map_starts_across_chains` | `false` | Expand chain-relative starts across all chains. |
| `map_size` | `4000` | Square contact-map size for each sampled window. |
| `cutoff` | `6.0` | 3D distance cutoff for contact counting. |
| `num_processes` | `6` | CPU workers for contact-map counting. |
| `verbose` | `true` | Verbosity passed to the sampler. |
| `plugins` | default `ContactsPlugins` | Sampler, O/E, post-process, and visualization hooks. |

If `replicate_map_starts_across_chains` is true, every `map_start` must fit
inside one chain with `map_start + map_size <= lef.chain_length`. Effective
starts are then expanded as:

```text
chain_idx * chain_length + map_start
```

### Contact Plugins

| Slot | Default | Purpose |
| --- | --- | --- |
| `sampler` | `plugins.sampling:monomer_resolution_sampler` | Produces the raw contact map. |
| `obs_over_exp` | `plugins.sampling:observed_over_expected` | Optional raw-to-O/E normalizer. |
| `post_process` | `null` | Optional transform after O/E or raw map. |
| `viz` | `plugins.sampling:default_oe_heatmap` | Optional PNG renderer. |

Built-in sampling/processing functions:

| Plugin | Description |
| --- | --- |
| `monomer_resolution_sampler` | Calls `contactmaps.monomerResolutionContactMapSubchains` with `mapStarts`, `mapN`, `cutoff`, workers, and verbosity from `ContactsConfig`. |
| `binned_sampler` | Calls `contactmaps.binnedContactMap` with an optional `bin_size`. |
| `observed_over_expected` | Diagonal-wise O/E normalization. |
| `iterative_correction` | Symmetric ICE-style balancing. Usually used through `balanced_observed_over_expected`. |
| `balanced_observed_over_expected` | Iterative correction followed by diagonal-wise O/E. |
| `default_oe_heatmap` | Matplotlib heatmap renderer, log-transforming by default. |

`post_process` receives the O/E map if `obs_over_exp` ran, otherwise the raw
map, and writes to `oe_output_path`. The visualization step renders the O/E map
when available, otherwise the raw map.

## QC and Compare

The `qc` stage has no separate YAML section. It uses:

* `lef.output_path` for `LEFPositions.h5`;
* `contacts.raw_output_path` and `contacts.oe_output_path` when 3D contact maps
  exist;
* `contacts.trajectory_folder` as the parent for `qc/`.

Outputs:

```text
qc/metrics.json
qc/report.md
qc/plots/loop_length.png
qc/plots/Ps.png
qc/plots/insulation_windows.png
qc/plots/contact_map.png
qc/plots/pileups_*.png
```

When 3D maps are present, QC balances the raw map before computing 3D metrics.
If no contact map exists, QC still reports 1D trajectory metrics.

`compare` loads two pipeline configs and compares their 1D and optional 3D
outputs. It writes:

```text
compare.json
compare.md
plots/*.png
```

Use `--folder-a` and `--folder-b` when the run folders differ from the paths
stored in the YAML configs.

## Coordinate Conventions

Most coordinates are absolute lattice/monomer indices unless a replication flag
says otherwise.

Chain-relative inputs:

* `topology_kwargs.tad_positions` for built-in TAD topologies;
* `topology_kwargs.genes` when `replicate_genes_across_chains: true`;
* `polymer.plugins.force_builder.kwargs.ep_pairs` when
  `replicate_ep_pairs_across_chains: true`;
* `contacts.map_starts` when `replicate_map_starts_across_chains: true`.

Absolute outputs:

* LEF HDF5 `positions`;
* RNAPII HDF5 positions;
* lesion positions;
* viewer annotations before display-window offsetting;
* polymer monomer indices;
* contact-map sampled windows after map-start expansion.

For replicate-chain configs, keep replication flags consistent across 1D
features, 3D E-P pair attraction, and contact-map windows.

## Common Feature Switches

### Cohesin + CTCF Only

Use a topology plugin that supplies CTCF sites, leave RNAPII and lesion plugins
null, and use the default cohesin translocator unless another barrier model is
needed.

```yaml
lef:
  plugins:
    topology: polychrom.pipelines.loop_extrusion.plugins.topology:convergent_tad_topology
    rnapii_load: null
    rnapii_translocate: null
    lesion: null
```

### RNAPII On

Use a gene-aware topology, set both RNAPII plugins, and use
`translocate_with_rnapii` so cohesin responds to RNAPII occupancy.

```yaml
lef:
  plugins:
    topology: polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_convergent_tad_topology
    translocate: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:translocate_with_rnapii
    rnapii_load: polychrom.pipelines.loop_extrusion.plugins.rnapii:load_rnapii
    rnapii_translocate: polychrom.pipelines.loop_extrusion.plugins.rnapii:stateful_translocate_rnapii
```

### RNAPII Off but Genes Kept as Genomic Features

Keep a gene-aware topology if genes are needed for annotations, targeted
loading, E-P pair metadata, or lesions, but set the RNAPII plugin slots to
`null`.

```yaml
lef:
  plugins:
    topology: polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_convergent_tad_topology
    rnapii_load: null
    rnapii_translocate: null
```

### Lesions On

Use a gene-aware topology with lesion kwargs, set the lesion update plugin, and
use `translocate_with_rnapii` if lesions should block cohesin legs.

```yaml
lef:
  plugins:
    topology: polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_convergent_tad_topology
    translocate: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:translocate_with_rnapii
    lesion: polychrom.pipelines.loop_extrusion.plugins.lesions:update_lesions
```

### 3D E-P Attraction

Use `paper_force_builder` with `ep_pairs`. Keep those pairs consistent with the
gene/enhancer coordinates if the same biological pairs should affect both 1D
RNAPII gating and 3D attraction.

```yaml
polymer:
  plugins:
    force_builder:
      target: polychrom.pipelines.loop_extrusion.plugins.forces:paper_force_builder
      kwargs:
        ep_pairs:
          - [<enhancer site>, <promoter site>]
        selective_attraction_energy: <strength>
```

### O/E Contact Maps With Balancing

```yaml
contacts:
  plugins:
    obs_over_exp:
      target: polychrom.pipelines.loop_extrusion.plugins.sampling:balanced_observed_over_expected
      kwargs:
        max_iter: <iterations>
        tol: <tolerance>
        ignore_diagonals: <number of near diagonals>
```

## Practical Checks Before Running

* `separation` should be positive and no larger than the intended scale of
  cohesin density.
* `trajectory_length % polymer.restart_every_blocks == 0`.
* `polymer.restart_every_blocks % polymer.save_every_blocks == 0`.
* If `polymer.pbc: false`, the selected force builder should add confinement or
  the polymer will not have a periodic box or explicit confining force.
* If RNAPII should be enabled, both RNAPII plugin slots must be non-null.
* If RNAPII should be disabled, setting `max_rnapii: 0` is not the switch; set
  `rnapii_load: null` and `rnapii_translocate: null`.
* If lesions should affect cohesin motion, use a translocator that checks
  lesions, such as `translocate_with_rnapii`.
* If using replicate chains, align `replicate_genes_across_chains`,
  `replicate_ep_pairs_across_chains`, and
  `replicate_map_starts_across_chains` with the coordinate system used in the
  YAML.
* If `polii_self_affinity > 0` with multiple chains, set
  `restrict_nonbonded_to_chains: true`.
* Keep runtime output locations outside the YAML when possible by passing the
  CLI `output_dir`.
