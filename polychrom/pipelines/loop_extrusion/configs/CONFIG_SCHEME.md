# Loop Extrusion Config Scheme

This document explains the YAML configuration of the modular loop-extrusion
pipeline: first the **schema** (how a config is structured and loaded), then
the **biophysical interpretation** of the parameter values used by the paper-
scale configs.

> For an exhaustive key/type/default table of every field in every stage, see
> [`../../PIPELINE_YAML_SCHEME.md`](../../PIPELINE_YAML_SCHEME.md). The source of
> truth is the dataclasses in [`../config.py`](../config.py).

Primary current configs:

- `experiment1_tads_only_biophysical.yaml`: directional CTCF TADs plus cohesin, no RNAPII.
- `experiment2_tads_transcription_biophysical.yaml`: same TAD/cohesin/polymer setup plus RNAPII dynamics and cognate E-P attraction.

Reference source:

- `supplementary_material_clean.txt`
- `paper_NRMCB.yaml`

The biophysical configs follow the supplement where it applies directly:
1 kb lattice sites, cohesin separation `d = 240 kb`, processivity
`lambda = 300 kb`, 10,000-step 1D warmup, paper OpenMM polymer physics,
20% spherical confinement, and contact maps generated from 3D distances.

## Pipeline Overview

```text
YAML config
  |
  v
1D loop extrusion stage
  - place cohesins on DNA lattice
  - move cohesin legs
  - stall legs at directional CTCF sites
  - optionally run RNAPII dynamics
  |
  output: LEFPositions.h5
  |
  v
3D polymer stage
  - DNA is a bead chain
  - cohesin positions become temporary harmonic bonds
  - OpenMM runs molecular dynamics
  |
  output: blocks_*.h5 trajectory files
  |
  v
contact stage
  - count bead pairs closer than cutoff
  - write raw contact map
  - normalize to O/E
  - render PNG
  |
  output: contact_map.npy, contact_map_oe.npy, contact_map_oe.png
```

## Schema at a Glance

### Top-level sections

A config has **four** top-level sections, each optional (omit one and its
dataclass defaults apply):

```yaml
lef:      # Stage 1 - 1D cohesin, CTCF, and optional RNAPII dynamics
viewer:   # Stage 4 - standalone interactive HTML of the 1D trajectory
polymer:  # Stage 2 - 3D OpenMM simulation driven by the LEF trajectory
contacts: # Stage 3 - contact-map sampling and normalization
```

### Stage commands and run order

```bash
python -m polychrom.pipelines.loop_extrusion.cli <stage> config.yaml
```

`<stage>` = `lef`, `viewer`, `polymer`, `contacts`, or `all`. In `all` the
order is **`lef -> viewer -> polymer -> contacts`** -- the viewer runs before
the 3D stage so the 1D dynamics can be inspected before paying for MD. Each
stage reads only the file(s) the previous stage produced, so they can also run
independently (the `viewer` and `polymer` stages auto-run `lef` first if
`LEFPositions.h5` is missing).

### The plugin pattern

Every pluggable "mechanic" is a `PluginSpec`: a pointer to a callable plus its
kwargs. Two YAML forms are accepted anywhere a plugin slot appears:

```yaml
# short form - bare string, no kwargs
load: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:load_one

# long form - target (required) + kwargs (optional)
force_builder:
  target: polychrom.pipelines.loop_extrusion.plugins.forces:paper_force_builder
  kwargs:
    repulsion_energy: 50.0
```

The target is `module:attr` (preferred) or `module.attr`; the loader imports it
and forwards `kwargs` at call time. A slot typed *optional* can be set to
`null` to skip that step (e.g. `obs_over_exp: null` skips O/E generation).

### Loader rules

The loader ([`../config.py`](../config.py), `load_config`) materialises the
YAML into nested dataclasses:

| Situation | Behaviour |
|-----------|-----------|
| Unknown key | **Hard error** -- `KeyError: Unknown config key '<k>' for <Class>`. Typos do not pass silently. |
| Missing key / section | Falls back to the dataclass default. |
| Empty file | All-default config. |
| Optional plugin set to `null` | That step is skipped. |

Because unknown keys raise, the kwargs under `topology_kwargs` (and every plugin
`kwargs`) must match the **selected** callable's signature -- e.g. passing
`rnapii_stall_prob` to `convergent_tad_topology` (which lacks it) is an error;
use `gene_aware_convergent_tad_topology` instead.

## Stage 1: `lef`

The `lef` section controls the 1D loop extrusion simulation.

### Lattice Size

```yaml
chain_length: 1600
num_chains: 1
```

Meaning:

```text
num_sites = chain_length * num_chains
          = 1600 * 1
          = 1600 lattice sites
```

In the biophysical interpretation:

```text
1 lattice site = 1 kb DNA
1600 sites = 1.6 Mb locus
```

The supplement uses `70,000` sites, or `70 Mb`. The 1,600-site configs are
a reduced locus for faster iteration.

### Cohesin Density

```yaml
separation: 240
```

Meaning:

```text
num_lefs = num_sites // separation
         = 1600 // 240
         = 6 cohesins
```

Biological meaning:

```text
separation: 240 = one cohesin per about 240 kb
```

This matches the supplement's cohesin separation `d = 240 kb`.

Important caveat:

```text
1600-site locus at paper density has only 6 cohesins.
```

This is biologically scaled but sparse, so TAD contact maps can look weaker
than older toy configs with `separation: 50` and 32 cohesins.

### Cohesin Lifetime and Processivity

```yaml
lifetime: 150
lifetime_stalled: 150
```

Each step moves the two cohesin legs outward by one lattice site if not
blocked. So an unobstructed cohesin extrudes about two sites per step.

```text
processivity ~= 2 * lifetime
             ~= 2 * 150
             ~= 300 kb
```

This matches the supplement's cohesin processivity of `lambda = 300 kb`.

`lifetime_stalled` is the lifetime used when cohesin is stalled by a non-CTCF
obstacle. It is set equal to `lifetime` in current biophysical configs.

### Warmup and Saved LEF Frames

```yaml
warmup_steps: 10000
trajectory_length: 100000
chunk_size: 50
```

`warmup_steps` are 1D loop extrusion steps discarded before saving. This lets
the 1D cohesin pattern reach steady state.

`trajectory_length` is the number of saved 1D LEF frames that will drive 3D
polymer simulation.

`chunk_size` controls how many LEF frames are written per HDF5 write chunk.
It is an I/O detail, not a biological parameter.

Current biophysical math:

```text
discarded LEF warmup frames = 10,000
saved LEF frames = 100,000
```

### CTCF / TAD Topology

Current biophysical configs use directional convergent CTCF barriers:

```yaml
topology_kwargs:
  tad_positions: [200, 500, 800, 1100, 1400]
  boundary_strength: 0.5
  release_prob: 0.0
  include_chromosome_ends: true
```

TAD intervals:

```text
TAD0: 0-199
TAD1: 200-499
TAD2: 500-799
TAD3: 800-1099
TAD4: 1100-1399
TAD5: 1400-1599
```

Directional CTCF layout:

```text
TAD interval:      start                         end
                   |                             |
left boundary:     captures left-moving leg
right boundary:                                  captures right-moving leg

cohesin loop grows outward until each leg reaches an inward-facing CTCF
```

Per-key meaning:

- `tad_positions`: the interior TAD boundaries, in lattice sites. The intervals
  above are formed by splitting `[0, chain_length]` at these positions; the same
  pattern is replicated on every chain when `num_chains > 1`.
- `boundary_strength`: probability that an *eligible* CTCF encounter captures the
  cohesin leg (the supplement's occupancy `b`). "Eligible" = convergent only: the
  left boundary acts on the left-moving (`-1`) leg, the right boundary on the
  right-moving (`+1`) leg, so a cohesin extruding inside the TAD is bracketed.
  `0.5` = supplement `b = 0.5`; `1.0` = the strong-CTCF condition used to show
  stronger bimodality.
- `release_prob`: per-tick probability that a captured leg lets go. `0.0` means a
  captured leg stays put until the whole cohesin unloads -- matching the
  supplement's statement that stalled motor subunits do not move again until
  cohesin dissociates.
- `include_chromosome_ends`: when true, also place capturing barriers at the two
  chain ends (sites `0` and `chain_length - 1`), so the terminal TADs are walled
  in; when false, the chain ends are open.

For a more illustrative but still supplement-supported setting, set:

```yaml
boundary_strength: 1.0
```

This corresponds to the supplement's strong CTCF condition used to illustrate
stronger bimodality.

### RNAPII (transcription) -- used by Exp2 only

Exp1 runs cohesin + CTCF with no transcription. Exp2 adds RNA polymerase II as a
second species of agent on the same 1D lattice. RNAPII is a **model extension**:
the supplement specifies loop extrusion, CTCF, E-P attraction, and polymer
physics, but not transcription-state dynamics, so these rules are our own.

#### Turning it on

RNAPII activates only when **both** `rnapii_load` and `rnapii_translocate`
plugins are set, and `translocate` is swapped to the RNAPII-aware variant so
cohesin reacts to polymerase bodies:

```yaml
plugins:
  topology:
    target: polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_convergent_tad_topology
  translocate:        polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:translocate_with_rnapii
  rnapii_load:        polychrom.pipelines.loop_extrusion.plugins.rnapii:load_rnapii
  rnapii_translocate: polychrom.pipelines.loop_extrusion.plugins.rnapii:stateful_translocate_rnapii
```

The genes live in `topology_kwargs.genes`, so the topology must be a
`gene_aware_*` variant (a plain `convergent_tad_topology` rejects the gene and
`rnapii_*` kwargs -- see [Loader rules](#loader-rules)).

The lattice now carries three occupancy states: `0` = free, `1` = cohesin leg,
`2` = RNAPII body. Only one agent occupies a site at a time.

#### The RNAPII life cycle (per gene)

Each gene is a transcription unit running from its TSS to its TES. The direction
is inferred from their order (`+1` if `tes > tss`, else `-1`). Every 1D tick:

1. **Load.** With probability `load_prob`, a new polymerase is recruited onto a
   free TSS in the **POISED** state. If `load_requires_enhancer` is true, this
   recruitment step is also gated by current E-P contact.
2. **POISED -> PAUSED.** With probability `initiation_prob` the polymerase
   initiates. If `pause_offset > 0` it also takes one step off the TSS into a
   promoter-proximal pause; otherwise it pauses on the TSS.
3. **PAUSED -> ELONGATING.** Gated two ways: if `requires_enhancer` is true the
   gene must currently be in **E-P contact** (see below) or it stays paused;
   when eligible it releases with probability `pause_release_prob` and takes its
   first elongation step the same tick.
4. **ELONGATING.** With probability `elongation_step_prob` per tick it advances
   one site toward the TES (`< 1` models a sub-cohesin elongation speed).
5. **Unload.** On reaching the TES it dwells one tick, then leaves.

#### Per-gene fields

For multi-chain replicate simulations, define each gene once in chain-relative
coordinates and set `replicate_genes_across_chains: true` in `topology_kwargs`.
The topology plugin expands those sites by `chain_idx * chain_length`.

| Field | Meaning |
|-------|---------|
| `tss` | Transcription start site; the load site and POISED position. |
| `tes` | Transcription end site; RNAPII unloads here. Sign of `tes - tss` sets travel direction. |
| `load_prob` | Per-tick probability of recruiting a new POISED RNAPII (only if the TSS is free). |
| `load_requires_enhancer` | If true, RNAPII recruitment is blocked until the gene is in E-P contact. Defaults to false for backward compatibility. |
| `requires_enhancer` | If true, pause release is blocked until the gene is in E-P contact. False = constitutive. |
| `enhancer_pos` | Cognate enhancer site; defines the E-P pair tested for contact. Required when `requires_enhancer`. |
| `initiation_prob` | Per-tick POISED -> PAUSED probability. |
| `pause_release_prob` | Per-tick PAUSED -> ELONGATING probability (after the enhancer gate, if any). |
| `elongation_step_prob` | Per-tick probability of advancing one site while ELONGATING. |
| `pause_offset` | If `> 0`, the polymerase steps one site off the TSS on initiation (promoter-proximal pause) instead of pausing on the TSS. |

Exp2's gene model: 12 genes (2 per TAD) -- one constitutive (`requires_enhancer:
false`) and one enhancer-dependent (`requires_enhancer: true`) per TAD, giving 6
enhancer-promoter links.

#### E-P contact and `ep_contact_tolerance`

The enhancer gate in step 3 needs a definition of "the enhancer is near the
promoter." Rather than a 3D distance (not known during the 1D stage), the model
uses a **cohesin-loop-containment proxy**: a gene is in E-P contact when some
cohesin's loop interval `[L, R]` brackets its `(tss, enhancer_pos)` pair.

`ep_contact_tolerance` is the slack in that bracket test. With `tol =
ep_contact_tolerance`, the pair counts as contained when

```text
L <= min(tss, enhancer_pos) + tol   AND   R >= max(tss, enhancer_pos) - tol
```

i.e. the loop endpoints may fall up to `tol` sites *short* of the exact E-P
coordinates and still count. Why the slack is needed: a cohesin leg cannot sit
exactly on the TSS while a POISED/PAUSED polymerase occupies that site, so
without tolerance an otherwise-enclosing loop would just miss and never release
the pause. Larger `tol` = more permissive contact detection (easier pause
release); `tol = 0` demands the loop reach the exact E-P sites.

#### Cohesin <-> RNAPII collisions

When an RNAPII tries to step into a site held by a cohesin leg, the outcome is
**state- and orientation-aware** (Fursova & Larson 2024, Fig 3a):

1. **State gate.** Only an `ELONGATING` polymerase translocates productively, so
   only it can push a cohesin. A `POISED` / `PAUSED` polymerase is a stationary
   block and always **stalls** on contact. (A v1 single-state RNAPII carries no
   `state` and is treated as always-elongating.)
2. **Intrinsic stall floor.** With probability `rnapii_stall_prob` the elongating
   polymerase stalls regardless of orientation (Pol II pausing/slowing on any
   obstacle).
3. **Orientation.** Otherwise the push probability depends on the cohesin leg's
   extrusion direction relative to the RNAPII:

| Leg orientation | Push parameter | Effect |
|-----------------|----------------|--------|
| co-directional (rear encounter, leg extrudes the same way Pol II travels) | `rnapii_push_prob` | RNAPII shoves the leg back one site (if the site behind is free) and advances. |
| head-on (converging, leg extrudes toward Pol II) | `rnapii_headon_push_prob` (default `0.0`) | Usually stalls; pushing a converging motor is hard. |

If the chosen push roll fails, the polymerase **stalls**. There is no longer a
separate *pass* outcome (it previously duplicated *push*). A leg with no recorded
`dir` (e.g. a bare `Leg` in a unit test) is treated as co-directional.

The reverse interaction (a cohesin leg meeting an RNAPII body) is handled in
`translocate_with_rnapii`: an RNAPII parked on its own TES is always bypassed.
Otherwise the cohesin leg stalls with state-dependent probability:

| RNAPII state | Parameter | Default | Effect |
|--------------|-----------|---------|--------|
| `POISED` | `rnapii_poised_block_prob` | `1.0` | Probability that a promoter-bound poised RNAPII blocks the incoming cohesin leg. |
| `PAUSED` | `rnapii_paused_block_prob` | `rnapii_block_prob` fallback, otherwise `1.0` | Probability that promoter-proximal paused RNAPII blocks the incoming cohesin leg. |
| `ELONGATING` | `rnapii_elongating_block_prob` | `rnapii_block_prob` fallback, otherwise `1.0` | Probability that elongating gene-body RNAPII blocks the incoming cohesin leg. |

If the draw does not block, the cohesin leg bypasses the RNAPII body and steps
onto that lattice site; RNAPII bookkeeping remains attached to the polymerase so
it can advance away on a later RNAPII translocation step. Defaults preserve the
previous hard-obstacle behaviour.

The encounter parameters live in `topology_kwargs` alongside the genes:

```yaml
topology_kwargs:
  rnapii_stall_prob: 0.4
  rnapii_push_prob: 0.25
  rnapii_headon_push_prob: 0.05
  rnapii_poised_block_prob: 1.0
  rnapii_paused_block_prob: 1.0
  rnapii_elongating_block_prob: 1.0
  ep_contact_tolerance: 10
  replicate_genes_across_chains: true
  genes: [...]
```

`rnapii_block_prob` is still accepted as a shared fallback for older configs,
but new configs should prefer the state-specific paused/elongating keys.

## Stage 2: `polymer`

The `polymer` section controls 3D molecular dynamics in OpenMM.

### Inputs and Outputs

```yaml
lef_positions_path: trajectory_exp1_biophysical/LEFPositions.h5
output_folder: trajectory_exp1_biophysical
```

The polymer stage reads the 1D LEF trajectory and writes 3D trajectory block
files into `output_folder`.

### OpenMM Platform

```yaml
platform: cuda
gpu: "0"
integrator: variableLangevin
error_tol: 0.01
collision_rate: 1.0
precision: mixed
```

Meaning:

- `platform: cuda`: run on NVIDIA GPU.
- `gpu: "0"`: use GPU device 0.
- `integrator: variableLangevin`: Langevin dynamics with adaptive timestep.
- `error_tol: 0.01`: variable timestep error tolerance.
- `collision_rate: 1.0`: Langevin collision/friction rate, matching supplement.
- `precision: mixed`: typical CUDA performance/accuracy balance.

### Density and Confinement

```yaml
density: 0.2
pbc: false
```

Current biophysical configs use spherical confinement from the force builder,
not periodic boundary conditions.

```text
density: 0.2 = 20% DNA volume fraction
pbc: false = no periodic box
```

This matches the supplement's spherical confinement at 20% volume fraction.

### MD Timing

```yaml
md_steps_per_block: 1000
save_every_blocks: 10
restart_every_blocks: 200
```

Definitions:

- `md_steps_per_block`: OpenMM integrator steps per saved LEF frame.
- `save_every_blocks`: save one 3D conformation every N LEF frames.
- `restart_every_blocks`: rebuild the OpenMM dynamic bond list every N LEF frames.

Current math:

```text
LEF frames = trajectory_length = 100,000
MD steps per LEF frame = 1,000

recording MD steps = 100,000 * 1,000
                   = 100,000,000 MD steps
```

Saved conformations:

```text
saved conformations = trajectory_length / save_every_blocks
                    = 100,000 / 10
                    = 10,000
```

OpenMM restart chunks:

```text
restart chunks = trajectory_length / restart_every_blocks
               = 100,000 / 200
               = 500
```

`restart_every_blocks` is an implementation batching parameter. Cohesin bonds
change over time. OpenMM can efficiently change parameters of an existing bond,
but changing the number of bonds in a live context is unsafe. The code looks
ahead over `restart_every_blocks` LEF frames, creates every bond that may be
needed in that chunk, then toggles bonds active/inactive by changing spring
strength.

### Relaxation

```yaml
initial_relaxation_steps: 500000
pre_recording_steps: 500000
```

`initial_relaxation_steps`:

- run before cohesin bonds are inserted
- relaxes the compact initial polymer conformation

`pre_recording_steps`:

- run after cohesin bonds are inserted
- not recorded
- lets the polymer equilibrate with cohesin constraints before sampling

Total approximate MD work:

```text
initial relaxation = 500,000
pre-recording = 500,000
recording = 100,000,000

total ~= 101,000,000 MD steps
```

### Cohesin Bond Geometry

```yaml
smc_bond_wiggle: 0.1
smc_bond_dist: 1.0
```

In 3D, a cohesin loop is represented as a harmonic bond between two monomers.

`smc_bond_dist: 1.0`:

- target distance is one monomer size

`smc_bond_wiggle: 0.1`:

- sets spring stiffness
- smaller value = stiffer bond
- matches supplement `delta = 0.1 monomer`

### Reporter

```yaml
max_data_length: 100
overwrite: true
```

`max_data_length` controls how many saved 3D conformations are placed in each
HDF5 block file.

Current file count:

```text
saved conformations = 10,000
max_data_length = 100
HDF5 block files = 10,000 / 100 = 100
```

`overwrite: true` allows the reporter to overwrite existing output folder data.

### Force Builder

Current biophysical configs use:

```yaml
force_builder:
  target: polychrom.pipelines.loop_extrusion.plugins.forces:paper_force_builder
```

`paper_force_builder` assembles the Nat. Rev. Mol. Cell Biol. supplementary
box-1 force field: harmonic backbone bonds, optional angle stiffness, a soft-core
excluded-volume term (`selective_SSW`, or `heteropolymer_SSW` when `ep_pairs` are
given), and a spherical confinement. Each kwarg:

| Key | Meaning |
|-----|---------|
| `bond_length` | Rest length of a backbone bond, in monomer units (`1.0` = one monomer). |
| `bond_wiggle` | Backbone bond flexibility; the spring constant is `~ 2 kT / bond_wiggle^2`, so smaller = stiffer. |
| `angle_k` | Bending stiffness (persistence length, in kT). `null` disables the angle force -- the paper kit uses no angle term. |
| `repulsion_energy` | Height of the soft excluded-volume barrier, in kT (`50` = chains essentially cannot overlap). |
| `repulsion_radius` | Range of that repulsion, in monomers. |
| `attraction_energy` | Depth of a *global* attractive well between all monomers, in kT. `0.0` = no generic stickiness. |
| `attraction_radius` | Range of the global attraction, in monomers. |
| `confinement_density` | Target DNA volume fraction inside the confining sphere (`0.2` = 20%, the supplement value). The sphere radius is sized from this. |
| `confinement_k` | Stiffness of the confining wall, in kT per monomer. |

Exp2 additionally makes the cognate E-P pairs **selectively** sticky:

```yaml
selective_attraction_energy: 1.0   # kT, only between cognate E-P partners
replicate_ep_pairs_across_chains: true
ep_pairs:                          # each [enhancer_site, promoter_site]
  - [30, 170]
  - [230, 470]
  - [530, 770]
  - [830, 1070]
  - [1130, 1370]
  - [1430, 1570]
```

When `ep_pairs` is set the builder switches to `heteropolymer_SSW`, assigns
each sticky site a distinct monomer type, and turns on only the listed pairwise
interactions. So an enhancer is attracted only to its listed promoter(s) -- not
all-to-all among every enhancer and promoter. Shared enhancers are allowed, e.g.
`[[155, 260], [155, 220]]`. `selective_repulsion_energy` (default `0.0`) is the
analogous selective repulsion, left off here.

Set `replicate_ep_pairs_across_chains: true` when `ep_pairs` are written in
chain-relative coordinates and should be copied to every chain. Leave it false
when `ep_pairs` already contain absolute monomer indices.

> The 1D stage and the 3D stage carry the E-P pairs separately: the 1D
> `enhancer_pos`/`tss` gate RNAPII pause release (loop-containment proxy), while
> these `ep_pairs` add the 3D attraction. Keep the two lists consistent if you
> want the same pairs to act in both stages.

#### Pol II transcription-driven compaction (`polii_self_affinity`)

Models Fursova & Larson 2024 (Fig 2a): Pol II-bound chromatin is biochemically
self-affine, so actively transcribed gene bodies fold into a compact globule.

```yaml
force_builder:
  kwargs:
    selective_attraction_energy: 1.0   # E-P stickiness, kT
    polii_self_affinity: 2.0           # Pol II self-attraction = 2 x 1.0 = 2 kT
```

How it works:

* Enabled only when `polii_self_affinity > 0` **and** the 1D stage recorded
  RNAPII (`max_rnapii > 0`) and genes. Otherwise the 3D physics is unchanged.
* A dedicated "transcribed" monomer type is added to the `heteropolymer_SSW`
  type system. Its self-attraction equals `polii_self_affinity *
  selective_attraction_energy` (so it is expressed as a multiple of the E-P
  stickiness; `2.0` here = 2 kT). The 3D stage forces `heteropolymer_SSW` on
  whenever this feature is active, even without `ep_pairs`.
* The transcribable set is every gene-body monomer (`TSS..TES`), minus any that
  coincide with an E/P sticky site. Each frame, `polymer.StickyUpdater` switches
  ON the gene bodies of genes that currently carry at least one **elongating**
  RNAPII (state 2), and OFF the rest -- the same per-block `updateParametersInContext`
  pattern used for SMC bonds. Pol II affinity is OFF during the initial
  bare-polymer relaxation.
* The affinity rides on the same chain-restricted nonbonded force, so replicate
  chains never interact. With `num_chains > 1` the builder **requires**
  `restrict_nonbonded_to_chains: true` and raises otherwise.

Tuning: start low (1-3 kT). Too strong collapses the whole active region into a
frozen globule; the paper notes full phase separation can itself inhibit
transcription.

## Stage 3: `contacts`

The `contacts` section controls contact map generation.

### Inputs and Outputs

```yaml
trajectory_folder: trajectory_exp1_biophysical
raw_output_path: trajectory_exp1_biophysical/contact_map.npy
oe_output_path: trajectory_exp1_biophysical/contact_map_oe.npy
viz_output_path: trajectory_exp1_biophysical/contact_map_oe.png
```

The contact stage reads saved 3D conformations and writes:

- `contact_map.npy`: raw observed contacts
- `contact_map_oe.npy`: observed/expected normalized map
- `contact_map_oe.png`: rendered heatmap

### Contact Window

```yaml
map_starts: [0]
replicate_map_starts_across_chains: true
map_size: 1600
```

This means sample one full-locus contact map per chain. With
`replicate_map_starts_across_chains: true`, `map_starts` are chain-relative and
are expanded by `chain_idx * chain_length`.

```text
start = 0
size = 1600
region = monomers 0-1599
```

For a 70,000-monomer paper-scale config, do not create a full 70,000 x 70,000
matrix. Use multiple smaller windows as in `paper_NRMCB.yaml`.

### Contact Radius

```yaml
cutoff: 5.0
```

Two monomers are counted as a contact when their 3D distance is less than or
equal to this cutoff radius.

### Parallelism

```yaml
num_processes: 6
verbose: true
```

Contact map counting uses CPU multiprocessing. This setting does not affect
OpenMM GPU dynamics.

### Contact Plugins

```yaml
sampler:
  target: polychrom.pipelines.loop_extrusion.plugins.sampling:monomer_resolution_sampler
```

This calls Polychrom's monomer-resolution subchain contact-map sampler.

```yaml
obs_over_exp:
  target: polychrom.pipelines.loop_extrusion.plugins.sampling:balanced_observed_over_expected
```

Current biophysical configs write ICE-balanced O/E maps by default:

1. perform iterative correction / balancing
2. divide each genomic-distance diagonal by its expected value

No-ICE O/E can be prepared from raw maps after the run:

```bash
PYTHONPATH=. micromamba run -n openmm python scripts/prepare_no_ice_contact_maps.py \
  --exp1-config polychrom/pipelines/loop_extrusion/configs/experiment1_tads_only_biophysical.yaml \
  --exp2-config polychrom/pipelines/loop_extrusion/configs/experiment2_tads_transcription_biophysical.yaml
```

## Stage 4: `viewer`

The `viewer` section renders a single self-contained HTML page that animates the
1D LEF trajectory: a playback bar plus linked panels for E-P proximity, the 1D
lattice + cohesin arcs, a kymograph (browser canvas), and a bridge map. CTCF /
gene / enhancer annotations are re-derived from the `lef` stage topology, so the
only input is `LEFPositions.h5`.

```yaml
viewer:
  lef_positions_path: trajectory_exp1_biophysical/LEFPositions.h5
  output_path: trajectory_exp1_biophysical/bridging_viewer.html
  stride: 1            # keep every Nth frame ...
  max_frames: 10000    # ... then decimate to at most this many
  bridge_cost: 1.0     # cost of one cohesin chord edge in the E-P distance graph
  site_start: null     # display window [start, end); null = whole lattice
  site_end: null
```

Notes:

- The 1D kymograph lives **inside this HTML** (drawn client-side). The `lef`
  stage itself writes only `LEFPositions.h5` -- it no longer emits a separate
  kymograph PNG.
- The "effective E-P distance" per frame is the shortest path on a graph: the
  backbone (1 per site) augmented with one chord edge per cohesin loop of weight
  `bridge_cost`. A loop bracketing an E-P pair short-circuits the backbone, so
  the distance drops as loops form.
- If the trajectory has no gene annotation, explicit pairs can be supplied via
  `ep_pairs: [{e: <site>, p: <site>, label: <str>}, ...]`.

## Plugin System

Every plugin field is a callable target:

```yaml
target: module.path:function_name
kwargs:
  parameter: value
```

The config loader imports the callable and passes `kwargs` to it.

Main plugin categories:

```text
lef.plugins.topology
  builds CTCF layout and RNAPII bookkeeping

lef.plugins.load / unload_prob / capture / release / translocate
  define 1D cohesin dynamics

lef.plugins.rnapii_load / rnapii_translocate   (optional; null by default)
  enable transcription dynamics; set both, and swap translocate to
  translocate_with_rnapii, to drive RNAPII each step

polymer.plugins.force_builder
  defines OpenMM force field

polymer.plugins.initial_conformation
  creates initial 3D polymer coordinates

contacts.plugins.sampler
  generates raw contact map

contacts.plugins.obs_over_exp   (optional; null skips O/E)
  normalizes raw map

contacts.plugins.post_process   (optional)
  extra transform after O/E (e.g. log, clip)

contacts.plugins.viz            (optional; null skips PNG)
  renders PNG heatmap

(the viewer stage has no plugins)
```

## Exp1 vs Exp2

Shared:

- 1,600 monomer locus
- 6 TAD intervals
- directional (convergent) CTCF barriers
- `separation: 240`
- `lifetime: 150`
- paper-style polymer physics
- spherical confinement
- 100,000 LEF frames
- ICE-balanced O/E contact maps

Exp1 only:

- `boundary_strength: 1.0` (strong CTCF, to sharpen TAD boxes)
- no RNAPII
- no E-P attraction

Exp2 only:

- `boundary_strength: 0.5` (supplement `b = 0.5`)
- 12 genes
- 6 enhancer-promoter links
- stateful RNAPII dynamics
- cohesin/RNAPII collision logic
- weak cognate E-P attraction using `ep_pairs`

## Current Biophysical Run Size

For each experiment:

```text
num_sites = 1600
num_lefs = 1600 // 240 = 6
saved LEF frames = 100,000
MD steps per LEF frame = 1,000
recording MD steps = 100,000,000
relaxation MD steps = 1,000,000
total MD steps ~= 101,000,000
saved 3D conformations = 10,000
HDF5 block files = 100
```

Expected runtime on the same CUDA setup used in previous tests:

```text
roughly 1.5-2 hours per experiment
```

This is an estimate, not a guarantee.

## Biological vs Visual Tradeoff

The biophysical configs are more biologically grounded than the old toy configs
because they use paper-like cohesin density and processivity.

However:

```text
1.6 Mb at paper density = about 6 cohesins
```

So TAD boxes may look weaker than old configs with:

```yaml
separation: 50
```

Old toy setting:

```text
1600 // 50 = 32 cohesins
```

That produces stronger visual TAD grids but is about 5 times denser than the
paper cohesin density.

Use the biophysical configs for biological claims. Use old/high-density configs
only when explicitly described as visual or sensitivity controls.

## Running

Run Exp1:

```bash
PYTHONPATH=. micromamba run -n openmm python -m polychrom.pipelines.loop_extrusion.cli all \
  polychrom/pipelines/loop_extrusion/configs/experiment1_tads_only_biophysical.yaml
```

Run Exp2:

```bash
PYTHONPATH=. micromamba run -n openmm python -m polychrom.pipelines.loop_extrusion.cli all \
  polychrom/pipelines/loop_extrusion/configs/experiment2_tads_transcription_biophysical.yaml
```

Compare:

```bash
PYTHONPATH=. micromamba run -n openmm python scripts/compare_loop_extrusion_experiments.py \
  --exp1-config polychrom/pipelines/loop_extrusion/configs/experiment1_tads_only_biophysical.yaml \
  --exp2-config polychrom/pipelines/loop_extrusion/configs/experiment2_tads_transcription_biophysical.yaml \
  --outdir loop_extrusion_comparison_biophysical \
  --insulation-window 50 \
  --ep-window-radius 5
```
