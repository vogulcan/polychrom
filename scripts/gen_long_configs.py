#!/usr/bin/env python
"""Generate configs/config1_long.yaml (txn ON) and config2_long.yaml (txn OFF).

A 10 Mb (10000-site, 1 site = 1 kb) locus with many variable-size TADs and a
gene set spanning the biological classes the references demand:

  * constitutive / housekeeping  (Kim 2026: short-range, cohesin-INDEPENDENT)
  * long-range enhancer-promoter (Kim 2026: >100 kb, cohesin-DEPENDENT)
  * super-enhancer (multi-enhancer, additive logic; Galitsyna 2026 fountains)
  * promoter-proximal paused      (Tei 2026 gatekeeper / pausing-QC)

config1_long and config2_long are byte-identical except the transcription axis
(max_rnapii + the two rnapii plugin slots) -- exactly the config1-vs-config2
contrast, just on the larger locus.

Deterministic (seeded). Re-run to regenerate.
"""
from __future__ import annotations

import random
from pathlib import Path

CHAIN_LENGTH = 10000          # 10 Mb at 1 site = 1 kb
SEED = 7
random.seed(SEED)

# --- variable, biologically relevant TAD sizes (kb), summing to CHAIN_LENGTH ---
# Mammalian TADs ~0.2-1 Mb (Dixon 2012); draw varied sizes, last TAD = remainder.
sizes: list[int] = []
acc = 0
while acc < CHAIN_LENGTH - 900:
    s = random.randint(250, 900)
    if acc + s > CHAIN_LENGTH - 250:        # leave room for a final TAD >=250
        break
    sizes.append(s)
    acc += s
sizes.append(CHAIN_LENGTH - acc)            # final TAD takes the remainder

# interior boundaries (chain-relative)
bounds = []
c = 0
for s in sizes[:-1]:
    c += s
    bounds.append(c)
tad_intervals = list(zip([0, *bounds], [*bounds, CHAIN_LENGTH]))
assert tad_intervals[-1][1] == CHAIN_LENGTH

# --- gene placement: cycle through the four classes across TADs ----------------
# Body length 40-160 kb; enhancers placed distally (>100 kb) but inside the TAD.
CLASSES = ["constitutive", "longrange_ep", "super_enhancer", "paused"]
genes: list[dict] = []
ci = 0
for (lo, hi) in tad_intervals:
    span = hi - lo
    if span < 120:                          # too small to host a clean gene
        continue
    cls = CLASSES[ci % len(CLASSES)]
    ci += 1
    plus = (ci % 2 == 0)                     # alternate strand
    body = min(max(40, span // 4), 160)

    if cls in ("longrange_ep", "super_enhancer") and span >= 320:
        # put promoter near one TAD edge, enhancer(s) far inside -> E-P >100 kb
        if plus:
            tss = lo + 20; tes = tss + body
            eref = hi - 20
        else:
            tss = hi - 20; tes = tss - body
            eref = lo + 20
        if cls == "super_enhancer":
            enhancers = sorted({int(eref), int((tss + eref) / 2),
                                int(eref - 0.25 * (eref - tss))})
            enhancers = [e for e in enhancers if lo <= e < hi]
            g = {"tss": tss, "tes": tes, "load_prob": 0.06,
                 "requires_enhancer": True, "load_requires_enhancer": True,
                 "enhancers": enhancers, "enhancer_logic": "additive",
                 "initiation_prob": 0.5, "pause_release_prob": 0.05,
                 "elongation_step_prob": 0.6, "pause_offset": 0,
                 "termination_prob": 0.2}
        else:
            g = {"tss": tss, "tes": tes, "load_prob": 0.06,
                 "requires_enhancer": True, "load_requires_enhancer": True,
                 "enhancers": [int(eref)], "enhancer_logic": "additive",
                 "initiation_prob": 0.5, "pause_release_prob": 0.03,
                 "elongation_step_prob": 0.6, "pause_offset": 0,
                 "termination_prob": 0.2}
    elif cls == "paused":
        mid = (lo + hi) // 2
        if plus:
            tss = mid - body // 2; tes = tss + body
        else:
            tss = mid + body // 2; tes = tss - body
        g = {"tss": tss, "tes": tes, "load_prob": 0.06,
             "requires_enhancer": False, "load_requires_enhancer": False,
             "initiation_prob": 0.5, "pause_release_prob": 0.05,   # heavy pausing
             "elongation_step_prob": 0.6, "pause_offset": 0,
             "termination_prob": 0.2}
    else:  # constitutive / housekeeping (also the fallback for small TADs)
        mid = (lo + hi) // 2
        if plus:
            tss = mid - body // 2; tes = tss + body
        else:
            tss = mid + body // 2; tes = tss - body
        g = {"tss": tss, "tes": tes, "load_prob": 0.06,
             "requires_enhancer": False, "load_requires_enhancer": False,
             "initiation_prob": 0.5, "pause_release_prob": 0.15,
             "elongation_step_prob": 0.6, "pause_offset": 0,
             "termination_prob": 0.2}

    # clamp safety
    for k in ("tss", "tes"):
        g[k] = max(lo, min(hi - 1, int(g[k])))
    genes.append(g)

# ------------------------------------------------------------------ emit YAML --
GENE_KEYS = ["tss", "tes", "load_prob", "requires_enhancer",
             "load_requires_enhancer", "enhancers", "enhancer_logic",
             "initiation_prob", "pause_release_prob", "elongation_step_prob",
             "pause_offset", "termination_prob"]


def _fmt_gene(g: dict) -> str:
    parts = []
    for k in GENE_KEYS:
        if k not in g:
            continue
        v = g[k]
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, list):
            v = "[" + ", ".join(str(x) for x in v) + "]"
        elif isinstance(v, str):
            v = v
        parts.append(f"{k}: {v}")
    return "{" + ", ".join(parts) + "}"


def _bstrength_block() -> str:
    lines = []
    for b in bounds:
        lines.append(f"      {b}: 0.60")
    return "\n".join(lines)


def build(txn_on: bool) -> str:
    max_rnapii = 256 if txn_on else 0
    rnapii_load = ("polychrom.pipelines.loop_extrusion.plugins.rnapii:load_rnapii"
                   if txn_on else "null")
    rnapii_tl = ("polychrom.pipelines.loop_extrusion.plugins.rnapii:stateful_translocate_rnapii"
                 if txn_on else "null")
    gene_lines = "\n".join(f"      - {_fmt_gene(g)}" for g in genes)
    tad_pos = "[" + ", ".join(str(b) for b in bounds) + "]"
    return f"""lef:
  chain_length: {CHAIN_LENGTH}
  num_chains: 16
  separation: 200              # ~1 cohesin per 200 kb (Rao 2014 ~1 loop/185 kb)
  lifetime: 150
  lifetime_stalled: 150
  lifetime_ctcf: 200
  warmup_steps: 10000
  trajectory_length: 5000
  chunk_size: 50
  seed: 42
  max_rnapii: {max_rnapii}

  topology_kwargs:
    tad_positions: {tad_pos}
    boundary_strength:
{_bstrength_block()}
    default_boundary_strength: 0.60
    release_prob: 0.0
    include_chromosome_ends: true
    lifetime_rnapii_stalled: 150
    rnapii_stall_prob: 0.35        # Pol intrinsically stalls ~1/3 of cohesin hits (pausing on obstacle)
    rnapii_push_prob: 0.50         # co-directional cohesin pushed (Kim 2026: cohesin follows txn; Banigan-Mirny 2022)
    rnapii_headon_push_prob: 0.02  # head-on collision stalls, not reversed (Brandao 2019; replication-txn conflict)
    rnapii_pause_cohesin_restraint: 0.3   # Tei 2026 pause-release gatekeeper (3.3x slower)
    rnapii_pause_restraint_window: 1
    rnapii_poised_block_prob: 0.5  # leaky at promoter: cohesin loads/passes POISED Pol
    rnapii_paused_block_prob: 0.95
    rnapii_elongating_block_prob: 0.95
    rnapii_terminating_block_prob: 0.95   # strong TES block -> 3' cohesin accumulation (Fursova 2024)
    ep_contact_tolerance: 1
    replicate_genes_across_chains: true
    targeted_load_prob: 0.02
    loading_window: 1
    target_enhancers: true
    target_tss: false
    lesion_prob: 0.0
    genes:
{gene_lines}

  plugins:
    topology:    polychrom.pipelines.loop_extrusion.plugins.topology:gene_aware_convergent_tad_topology
    load:        polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:load_targeted
    unload_prob: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:unload_prob
    capture:     polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:capture
    release:     polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:release
    translocate: polychrom.pipelines.loop_extrusion.plugins.lef_dynamics:translocate_with_rnapii
    rnapii_load:        {rnapii_load}
    rnapii_translocate: {rnapii_tl}

viewer:
  max_frames: 1000
  bridge_cost: 1.0
  insulation_score_window: {min(sizes)}    # ~ smallest TAD scale
  site_start: 0
  site_end: {CHAIN_LENGTH}

polymer:
  platform: cuda
  gpu: "0"
  integrator: variableLangevin
  error_tol: 0.01
  collision_rate: 1.0
  precision: mixed
  seed: 42
  density: 0.2
  pbc: false
  md_steps_per_block: 1270
  save_every_blocks: 1
  restart_every_blocks: 200
  initial_relaxation_steps: 500000   # Hansen Suppl: bare-polymer relax to steady state
  pre_recording_steps: 300000
  smc_bond_wiggle: 0.1
  smc_bond_dist: 1.0
  max_data_length: 100
  overwrite: true
  plugins:
    force_builder:
      target: polychrom.pipelines.loop_extrusion.plugins.forces:paper_force_builder
      kwargs:
        bond_length: 1.0
        bond_wiggle: 0.1
        angle_k: 1.5
        repulsion_energy: 50.0
        repulsion_radius: 1.05
        attraction_energy: 0.0
        attraction_radius: 2.0
        restrict_nonbonded_to_chains: true
        replicate_ep_pairs_across_chains: true
        ep_pairs: []
        selective_attraction_energy: 0.0
        polii_self_affinity: 0.0
        selective_repulsion_energy: 0.0
        confinement_density: 0.2
        confinement_per_chain: true
        confinement_k: 5.0
    initial_conformation:
      target: polychrom.pipelines.loop_extrusion.plugins.forces:grow_cubic_conformation

contacts:
  replicate_map_starts_across_chains: true
  map_starts: [0]
  map_size: {CHAIN_LENGTH}
  cutoff: [2, 3, 4, 5, 6]
  num_processes: 12
  verbose: true
  plugins:
    sampler:
      target: polychrom.pipelines.loop_extrusion.plugins.sampling:monomer_resolution_sampler
    obs_over_exp:
      target: polychrom.pipelines.loop_extrusion.plugins.sampling:balanced_observed_over_expected
      kwargs:
        max_iter: 2000
        tol: 1.0e-6
        ignore_diagonals: 2
    viz:
      target: polychrom.pipelines.loop_extrusion.plugins.sampling:default_oe_heatmap
      kwargs:
        log: true
        cmap: coolwarm
        vmin: -1.0
        vmax: 1.0
        figsize: [12, 12]
        dpi: 150
"""


def main() -> int:
    cfgdir = Path(__file__).resolve().parent.parent / "configs"
    (cfgdir / "config1_long.yaml").write_text(build(txn_on=True))
    (cfgdir / "config2_long.yaml").write_text(build(txn_on=False))
    print(f"# {len(tad_intervals)} TADs (sizes kb): {sizes}")
    print(f"# tad_positions: {bounds}")
    print(f"# {len(genes)} genes:")
    for g in genes:
        d = "+" if g["tes"] > g["tss"] else "-"
        enh = g.get("enhancers", "-")
        print(f"#   tss={g['tss']:>5} tes={g['tes']:>5} {d} "
              f"req_enh={g['requires_enhancer']!s:>5} enh={enh}")
    print("# wrote configs/config1_long.yaml, configs/config2_long.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
