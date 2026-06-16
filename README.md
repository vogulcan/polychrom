This fork of **polychrom** extends the original library with a dedicated focus on a comprehensive **1D chromatin simulation pipeline**.

The main additions include:

1. A configurable 1D simulation framework that incorporates transcription, DNA lesions, nucleotide-excision repair, and their interference with loop extrusion dynamics.
2. Automated pipeline management through YAML-based configuration files and a command-line interface.
3. Procedural topology generation with realistic transcription, lesion, and repair dynamics.
4. Downstream analysis tools for interpreting the resulting 1D chromatin structures.
5. Optional integration with polymer simulations, enabling automated workflows from generated YAML configurations, such as:

```text
Config → 1D Simulation → Analysis

or

Config → 1D Simulation → 3D Polymer Simulation → Analysis
```

This fork is primarily intended for **1D simulations and pipeline development**. For 3D polymer simulations, please refer to the original **polychrom** codebase and see the original software release:

> Imakaev, M., Goloborodko, A., & Brandao, H. *mirnylab/polychrom: v0.1.0*. Zenodo, 2019.

## Reproducing the paper results

To reproduce the results presented in our paper, first create the required micromamba environment using the provided `environment-openmm.yml` file:

```bash
micromamba create -f environment-openmm.yml
```

Then activate the environment:

```bash
micromamba activate <environment-name>
```

The environment name is defined in `environment-openmm.yml`.

After activating the environment, run the panel-generation script:

```bash
bash run_panels.sh
```

Before running the full pipeline, please inspect `run_panels.sh` and adjust the parameters according to your system resources. In particular, update settings such as `nproc` to match the number of CPU cores available on your machine.

## Command-line interface

The loop-extrusion pipeline ships a command-line entry point,
`polychrom.pipelines.loop_extrusion`. Installing the package (e.g.
`pip install -e .`) exposes it as the `polychrom-loopext` console script; the
equivalent module form works without installation:

```bash
polychrom-loopext <stage> config.yaml [output_dir]
# equivalently:
python -m polychrom.pipelines.loop_extrusion.cli <stage> config.yaml [output_dir]
```

Every stage is driven by a single YAML config (see `configs/` for worked
examples) that is materialised into typed dataclasses. The config has four
top-level sections — `lef`, `viewer`, `polymer`, `contacts` — and each
"mechanic" is a swappable plugin specified as a `module:attr` target with
optional `kwargs`.

### Pipeline stages

Each of the following subcommands takes a config path and an optional output
directory. When `output_dir` is given it overrides **all** derived stage paths
(so every stage reads/writes inside that one run folder) and a copy of the
config is archived into it.

| Stage | What it does | Primary output |
|-------|--------------|----------------|
| `lef` | Stage 1 — 1D loop-extrusion factor (cohesin) dynamics, optionally with RNAPII transcription and DNA-lesion/repair mechanics. | `LEFPositions.h5` |
| `viewer` | Stage 4 — self-contained interactive HTML viewer of the 1D dynamics (E–P proximity, lattice + arcs, kymograph, bridge map). Inspect 1D behaviour before paying for 3D MD. | `bridging_viewer.html` (+ `*_visited_heatmap.npy`, `*_elements.json`) |
| `polymer` | Stage 2 — 3D molecular-dynamics simulation with dynamic SMC bonds driven by the 1D trajectory (OpenMM). | trajectory in `output_dir/` |
| `contacts` | Stage 3 — contact-map sampling from the 3D trajectory, observed/expected normalisation, and heatmap rendering. Supports one or several contact-distance cutoffs. | `contact_map.npy`, `contact_map_oe.npy`, `contact_map_oe.png` |
| `qc` | 1D + 3D quality-control / analysis: loop-length and P(s) statistics, insulation, contact-map heatmaps, structured metrics. | `qc/metrics.json`, `qc/report.md`, `qc/plots/*.png` |
| `all` | Runs `lef → viewer → polymer → contacts → qc` in sequence. The `viewer` step is skipped unless `viewer.enabled: true`. | all of the above |

Examples:

```bash
# Run the full pipeline into a fresh run folder
polychrom-loopext all configs/config1_10k.yaml runs/config1

# Run just the 1D stage, then inspect it interactively
polychrom-loopext lef    configs/config1_10k.yaml runs/config1
polychrom-loopext viewer configs/config1_10k.yaml runs/config1

# 3D simulation + contact maps + QC on an existing 1D trajectory
polychrom-loopext polymer  configs/config1_10k.yaml runs/config1
polychrom-loopext contacts configs/config1_10k.yaml runs/config1
polychrom-loopext qc       configs/config1_10k.yaml runs/config1
```

### Verbosity

Verbosity is set with global flags placed before the stage. Output is verbose
by default (per-stage progress, ETA, and MD throughput lines):

```bash
polychrom-loopext --quiet all config.yaml runs/x   # warnings/errors only
polychrom-loopext --debug all config.yaml runs/x   # extra debug detail
```

### Comparing runs

The `compare` subcommand produces side-by-side 1D + 3D metrics and comparison
plots (loop-length, P(s), insulation, contact maps, CTCF-anchor pile-ups, and
Flyamer rescaled-TAD pile-ups) into an output directory. It accepts either a
YAML config or a run folder for each side; a run folder is auto-detected for its
`LEFPositions.h5` and `contact_map*.npy`.

```bash
# Pairwise: baseline vs one comparison
polychrom-loopext compare baseline_run comparison_run

# Override run folders / labels / output explicitly
polychrom-loopext compare cfgA.yaml cfgB.yaml \
    --folder-a runs/A --folder-b runs/B \
    --label-a baseline --label-b treated \
    --out compare_A_B

# Baseline vs many: writes pairwise subdirectories + consolidated compare_many.*
polychrom-loopext compare baseline_run run2 run3 run4 \
    --labels treated_1 treated_2 treated_3

# Sweep contact-distance cutoffs: one cutoff_<c>/ subfolder per value plus a
# consolidated tad_strength_vs_cutoff.{md,json} at the top level
polychrom-loopext compare baseline_run comparison_run --cutoffs 2 3 4 5 6
```

Outputs written into the comparison directory: `compare.json` (A, B and their
folds/diffs for every metric), `compare.md` (readable summary), and `plots/`.
All 3D metrics use the ICE-balanced map.
