#!/usr/bin/env python
"""Generate biologically-grounded config1/config2 (txn ON/OFF).

Realistic 30 Mb locus (30000 sites, 1 site = 1 kb), constraints sourced from
literature:
  * gene density 12-15 /Mb        (Gene density, Wikipedia)         -> ~13/Mb
  * gene length median ~23 kb, ~15% >100 kb, mean ~67 kb (PMC4053754)
  * TAD size mean ~880 kb, range 100-1500 kb (TAD, Wikipedia/Dixon)
  * E-P distance median ~100 kb, tail >500 kb; ~10% cross a TAD boundary
    (NAR 2024, doi 10.1093/nar/gkad1234)
  * housekeeping/broadly-active ~50-55%; cell-type + developmental the rest
  * cohesin ~1 loop / 185 kb (Rao 2014)

Gene classes (per user's regulatory-architecture table):
  hk_const 45%  hk_high 10%  celltype 30%  developmental 15%

config1 = txn ON, config2 = txn OFF (max_rnapii + 2 rnapii plugin slots);
byte-identical otherwise. Deterministic (seeded). Tuned MD/LEF params are
preserved from the current configs.
"""
from __future__ import annotations
import argparse
import numpy as np

CHAIN = 30000           # 30 Mb @ 1 kb/site
GENE_DENSITY = 16.0     # target /Mb; nets ~12/Mb after no-overlap packing
SEED = 7

# class -> draw ranges
CLASSES = {
    "hk_const": dict(frac=0.45, init=(0.5, 0.7),  prel=(0.10, 0.20), req=False, nenh=(0, 0)),
    "hk_high":  dict(frac=0.10, init=(0.8, 0.95), prel=(0.20, 0.40), req=False, nenh=(0, 1)),
    "celltype": dict(frac=0.30, init=(0.4, 0.6),  prel=(0.05, 0.15), req=True,  nenh=(1, 3)),
    "dev":      dict(frac=0.15, init=(0.1, 0.3),  prel=(0.03, 0.08), req=True,  nenh=(2, 5)),
}
ORDER = list(CLASSES)
FRAC = np.array([CLASSES[c]["frac"] for c in ORDER])

# --- TAD activity coupling ---------------------------------------------------
# Transcriptionally active chromatin forms SMALLER, gene-dense domains; quiet
# chromatin forms LARGER, gene-poor domains. Each TAD is tagged active/poor and
# its size + gene content are drawn accordingly.
ACTIVE_FRAC = 0.5        # fraction of TADs that are transcriptionally active
DENS_ACTIVE = 20.0       # genes/Mb in active TADs (gene-dense)
DENS_POOR   = 7.0        # genes/Mb in poor TADs (gene-sparse)
# per-TAD class mix [hk_const, hk_high, celltype, dev] by activity
FRAC_ACTIVE = np.array([0.35, 0.30, 0.30, 0.05])   # expressed-skewed
FRAC_POOR   = np.array([0.45, 0.03, 0.20, 0.32])   # poised/silent-skewed


def gen_tads(rng):
    """TAD sizes coupled to activity: active TADs SMALLER, poor TADs LARGER.

    Scales with CHAIN (median ~820 at 30 kb-site default). Returns
    ``(bounds, active)`` where ``active[i]`` flags interval i transcriptionally
    active. Mean size is preserved (active small + poor large average out)."""
    base = 820.0 * CHAIN / 30000.0
    med_a = float(np.clip(base * 0.55, 60, 820))      # active -> smaller
    med_p = float(np.clip(base * 1.60, 120, 2200))    # poor   -> larger
    lo_c = float(np.clip(100 * CHAIN / 30000, 30, 100))
    hi_c = float(min(2200, max(med_p * 1.8, CHAIN / 3)))
    sizes, active, acc = [], [], 0
    while acc < CHAIN - lo_c:
        is_a = rng.random() < ACTIVE_FRAC
        med = med_a if is_a else med_p
        s = int(np.clip(rng.lognormal(np.log(med), 0.40), lo_c, hi_c))
        if acc + s > CHAIN - lo_c:
            break
        sizes.append(s); active.append(is_a); acc += s
    sizes.append(CHAIN - acc); active.append(rng.random() < ACTIVE_FRAC)
    bounds = list(np.cumsum(sizes)[:-1])
    return bounds, active


def gene_length(rng):
    """Median ~23 kb, ~11% >100 kb (lognormal mu=ln23, sigma=1.2); tighter than
    the genome-wide dist so non-overlapping packing still reaches ~12 genes/Mb."""
    return int(np.clip(rng.lognormal(mean=np.log(23), sigma=1.2), 3, 800))


def _make_gene(rng, tss, tes, cls):
    """Assemble one gene dict for class ``cls`` (fields match the YAML schema)."""
    C = CLASSES[cls]
    g = dict(tss=int(tss), tes=int(tes), load_prob=0.06,
             requires_enhancer=C["req"], load_requires_enhancer=C["req"],
             initiation_prob=round(rng.uniform(*C["init"]), 2),
             pause_release_prob=round(rng.uniform(*C["prel"]), 2),
             elongation_step_prob=0.6, pause_offset=0, termination_prob=0.2)
    return g, C


def _add_enhancers(rng, g, C, lo, hi, strand):
    """Attach E-P enhancers in the TAD INTERIOR (never pinned to the boundary).

    Enhancers are the cohesin loading sites (``target_enhancers``). The old code
    clamped an overshoot to ``lo+5`` / ``hi-5`` (np.clip), which planted loaders
    5 sites off the boundary and manufactured loading-site fountains in the
    boundary pile-up. Overshoots are now scattered uniformly across the interior,
    keeping loaders >= M sites away from the flank.
    """
    if not C["req"]:
        return
    tss = g["tss"]
    M = 10
    lo_i, hi_i = lo + M, hi - M
    ne = max(1, int(rng.integers(C["nenh"][0], C["nenh"][1] + 1)))
    enh = []
    for _e in range(ne):
        dist = int(rng.integers(50, 300))
        e = (tss - dist) if strand else (tss + dist)
        if rng.random() < 0.10:               # ~10% cross a TAD boundary (NAR 2024)
            e = int(np.clip(e, 5, CHAIN - 5))
        elif hi_i <= lo_i:                     # tiny TAD: centre
            e = (lo + hi) // 2
        elif not (lo_i <= e <= hi_i):          # overshoot -> scatter into interior, not the flank
            e = int(rng.integers(lo_i, hi_i))
        enh.append(int(e))
    g["enhancers"] = sorted(set(enh)); g["enhancer_logic"] = "additive"


def gen_genes(rng, bounds, active):
    intervals = list(zip([0, *bounds], [*bounds, CHAIN]))
    genes = []
    EDGE = 3                                   # minimal end-margin (was 12, which barred promoters from flanks)
    for (lo, hi), is_a in zip(intervals, active):
        span = hi - lo
        if span < 40:
            continue
        # gene count from activity-dependent density (active = dense, poor = sparse)
        dens = DENS_ACTIVE if is_a else DENS_POOR
        n = int(np.clip(round(dens * span / 1000 * rng.lognormal(0, 0.3)), 0, span // 12))
        if n == 0:
            continue
        fr = FRAC_ACTIVE if is_a else FRAC_POOR
        occ = np.zeros(span, bool); occ[:EDGE] = True; occ[-EDGE:] = True
        remaining = n

        # --- boundary-flank promoters: housekeeping / Pol II density is high at
        # TAD boundaries (Dixon 2012). Seed a compact hk promoter just inside each
        # INTERNAL boundary, TSS proximal -> Pol II acts as a barrier reinforcing
        # insulation. hk classes carry no enhancers, so NO cohesin loads here.
        # Capped (<= n//2) so gene-rich TADs keep an interior population.
        nflank = 0 if n <= 1 else (1 if n <= 3 else 2)
        avail = [s for s in ("L", "R") if (lo > 0 if s == "L" else hi < CHAIN)]
        if len(avail) == 2 and nflank == 1 and rng.random() < 0.5:
            avail = avail[::-1]
        for side in avail[:nflank]:
            off = int(rng.integers(3, 9))       # promoter 3-8 sites into the flank
            L = min(gene_length(rng), max(3, span // 4))
            if side == "L":
                s = off
                tss_pos, tes_pos, strand = lo + s, lo + s + L, True
            else:
                s = span - off - L
                tss_pos, tes_pos, strand = hi - off, hi - off - L, False
            if s < EDGE or s + L > span - EDGE or occ[s:s + L].any():
                continue
            occ[max(0, s - 5):s + L + 5] = True
            cls = "hk_const" if rng.random() < 0.7 else "hk_high"
            g, C = _make_gene(rng, tss_pos, tes_pos, cls)
            _add_enhancers(rng, g, C, lo, hi, strand)
            genes.append(g); remaining -= 1

        # --- interior genes (classes drawn by TAD activity), largest first so
        # long genes get space (avoids short-gene packing bias) ---
        draws = []
        for _ in range(remaining):
            cls = ORDER[rng.choice(len(ORDER), p=fr)]
            draws.append((cls, min(gene_length(rng), span - 2 * EDGE - 4)))
        draws.sort(key=lambda d: -d[1])
        for cls, L in draws:
            if L < 3:
                continue
            placed = False
            for _t in range(40):
                hi_s = span - L - EDGE
                if hi_s <= EDGE:
                    break
                s = int(rng.integers(EDGE, hi_s))
                if not occ[s:s + L].any():
                    occ[max(0, s - 5):s + L + 5] = True; placed = True; break
            if not placed:
                continue
            strand = rng.random() < 0.5
            a, b = lo + s, lo + s + L
            tss_pos, tes_pos = (a, b) if strand else (b, a)
            g, C = _make_gene(rng, tss_pos, tes_pos, cls)
            _add_enhancers(rng, g, C, lo, hi, strand)
            genes.append(g)
    return genes, intervals


GENE_KEYS = ["tss", "tes", "load_prob", "requires_enhancer", "load_requires_enhancer",
             "enhancers", "enhancer_logic", "initiation_prob", "pause_release_prob",
             "elongation_step_prob", "pause_offset", "termination_prob"]


def fmt_gene(g):
    parts = []
    for k in GENE_KEYS:
        if k not in g:
            continue
        v = g[k]
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, list):
            v = "[" + ", ".join(str(x) for x in v) + "]"
        parts.append(f"{k}: {v}")
    return "{" + ", ".join(parts) + "}"


def build(txn_on, bounds, bstrength, genes, num_chains=4):
    # Pol II cap and relaxation/recording scale with locus size & chain count
    max_rnapii = (max(64, int(2560 * (CHAIN / 30000) * (num_chains / 4))) if txn_on else 0)
    relax = max(50000, int(1500000 * CHAIN / 30000))
    prerec = max(20000, int(500000 * CHAIN / 30000))
    rl = ("polychrom.pipelines.loop_extrusion.plugins.rnapii:load_rnapii" if txn_on else "null")
    rt = ("polychrom.pipelines.loop_extrusion.plugins.rnapii:stateful_translocate_rnapii" if txn_on else "null")
    bstr_block = "\n".join(f"      {b}: {s:.2f}" for b, s in zip(bounds, bstrength))
    gene_lines = "\n".join(f"      - {fmt_gene(g)}" for g in genes)
    tad_pos = "[" + ", ".join(str(b) for b in bounds) + "]"
    return f"""lef:
  chain_length: {CHAIN}
  num_chains: {num_chains}
  separation: 200              # ~1 cohesin / 185 kb (Rao 2014)
  lifetime: 75
  lifetime_stalled: 75
  lifetime_ctcf: 300
  warmup_steps: 10000
  trajectory_length: 5000
  chunk_size: 50
  seed: 42
  max_rnapii: {max_rnapii}

  topology_kwargs:
    tad_positions: {tad_pos}
    boundary_strength:
{bstr_block}
    default_boundary_strength: 0.60
    release_prob: 0.0
    include_chromosome_ends: true
    lifetime_rnapii_stalled: 75
    rnapii_stall_prob: 0.35
    rnapii_push_prob: 0.50
    rnapii_headon_push_prob: 0.02
    rnapii_pause_cohesin_restraint: 0.3
    rnapii_pause_restraint_window: 1
    rnapii_poised_block_prob: 0.5
    rnapii_paused_block_prob: 0.95
    rnapii_elongating_block_prob: 0.95
    rnapii_terminating_block_prob: 0.95
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
    rnapii_load:        {rl}
    rnapii_translocate: {rt}

viewer:
  max_frames: 1000
  bridge_cost: 1.0
  insulation_score_window: 300
  site_start: 0
  site_end: {CHAIN}

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
  initial_relaxation_steps: {relax}   # scales with locus size
  pre_recording_steps: {prerec}
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
        confinement_density: 0.1
        confinement_per_chain: true
        confinement_k: 5.0
    initial_conformation:
      target: polychrom.pipelines.loop_extrusion.plugins.forces:grow_cubic_conformation

contacts:
  replicate_map_starts_across_chains: true
  map_starts: [0]
  map_size: {CHAIN}
  cutoff: [2]
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


def main():
    global CHAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chain", type=int, default=CHAIN, help="chain length in sites (1 site = 1 kb)")
    ap.add_argument("--num-chains", type=int, default=4)
    ap.add_argument("--suffix", default="real", help="config name suffix: config1_<suffix>.yaml")
    args = ap.parse_args()
    CHAIN = int(args.chain)
    rng = np.random.default_rng(args.seed)
    bounds, active = gen_tads(rng)
    bstrength = np.round(rng.uniform(0.5, 0.9, size=len(bounds)), 2)
    genes, intervals = gen_genes(rng, bounds, active)
    from pathlib import Path
    od = Path(args.out_dir)
    sfx = args.suffix
    (od / f"config1_{sfx}.yaml").write_text(build(True, bounds, bstrength, genes, args.num_chains))
    (od / f"config2_{sfx}.yaml").write_text(build(False, bounds, bstrength, genes, args.num_chains))
    # stats
    lens = [abs(g["tes"] - g["tss"]) for g in genes]
    nenh = [len(g.get("enhancers", [])) for g in genes]
    perTAD = [sum(1 for g in genes if a <= min(g['tss'], g['tes']) < b) for a, b in intervals]
    asz = [hi - lo for (lo, hi), a in zip(intervals, active) if a]
    psz = [hi - lo for (lo, hi), a in zip(intervals, active) if not a]
    print(f"CHAIN={CHAIN} num_chains={args.num_chains}  TADs: {len(intervals)} "
          f"size mean={np.mean([b-a for a,b in intervals]):.0f} "
          f"min={min(b-a for a,b in intervals)} max={max(b-a for a,b in intervals)}")
    print(f"  active TADs: n={len(asz)} median size={int(np.median(asz)) if asz else 0}")
    print(f"  poor   TADs: n={len(psz)} median size={int(np.median(psz)) if psz else 0}")
    print(f"genes: {len(genes)}  = {len(genes)/(CHAIN/1000):.1f}/Mb")
    print(f"gene len kb: median={int(np.median(lens))} mean={int(np.mean(lens))} "
          f">100kb={100*np.mean(np.array(lens)>100):.0f}%")
    print(f"E-P genes: {sum(1 for g in genes if g.get('enhancers'))}  enh/EPgene mean={np.mean([n for n in nenh if n]) if any(nenh) else 0:.1f}")
    print(f"genes/TAD: min={min(perTAD)} median={int(np.median(perTAD))} max={max(perTAD)}")
    print(f"wrote {od}/config1_{sfx}.yaml , config2_{sfx}.yaml")


if __name__ == "__main__":
    main()
