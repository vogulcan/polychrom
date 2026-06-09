#!/usr/bin/env python
"""Generate biologically-grounded config1/config2 (txn ON/OFF).

Realistic 30 Mb locus (30000 sites, 1 site = 1 kb), constraints sourced from
literature:
  * gene density 12-15 /Mb        (Gene density, Wikipedia)         -> ~13/Mb
  * gene length median ~23 kb, ~15% >100 kb, mean ~67 kb (PMC4053754)
  * TAD size mean ~880 kb, range 100-1500 kb (TAD, Wikipedia/Dixon)
  * E-P distance median ~88-125 kb, tail >500 kb (PCHi-C / genomic-proximity
    estimates; eLife 2024 promoter-centered map). ~10% cross a TAD boundary is a
    modeling assumption (boundary insulation reduces but does not abolish crossing).
  * housekeeping/broadly-active ~50-55%; cell-type + developmental the rest
  * cohesin: ~10,000 loops genome-wide, contact-domain median ~185 kb (Rao 2014)

Gene classes (per user's regulatory-architecture table):
  hk_const 45%  hk_high 10%  celltype 30%  developmental 15%

config1 = txn ON, config2 = txn OFF (max_rnapii + 2 rnapii plugin slots);
byte-identical otherwise. Deterministic (seeded). See references/parameter_plan.md
for the full cohesin / RNAPII / interference table.

CALIBRATION: 1 tick = 4 s. A cohesin extrudes 2 monomers/tick (both legs) = 2 kb/tick
=> loop-growth 0.5 kb/s (Davidson 2019 in-vitro; per-leg 0.25 kb/s). Key identity:
bare processivity lambda = 2*lifetime (kb) is calibration-free; the tick only sets
speed (=1 kb/tick per leg) and residence (=lifetime*4 s).

COHESIN:
  * separation 180 -> 5.6 cohesin/Mb (Rao 2014 ~6/Mb). PRIMARY lever on effective loop
    size / P(s) turning point: in the dense regime effective loop ~= separation.
  * lifetime 300 -> free/extruding residence 20 min (Hansen 2017 15-25 min); bare
    lambda = 600 kb > separation, so loops saturate the dense regime. The OBSERVABLE
    is the collision-capped effective loop (~180 kb, matching Rao 185 / Nat Genet
    190-340), NOT bare lambda. lifetime_ctcf = 4*lifetime (anchored pool, kept at 4x).
  * loading ~98% uniform (targeted_load_prob 0.02). target_tss FALSE (Banigan 2023:
    no preferential TSS loading); target_enhancers TRUE (Kagey 2010 / Fursova 2024).
    cohesin@genes is barrier-driven (RNAPII pinning, Busslinger 2017), not loading.

RNAPII (per gene): load_prob 0.015 (~Banigan k_load), pause_release <=0.10 (40-130 s
  promoter pause), elongation_step_prob 0.20 (0.05 kb/s = 3 kb/min; cohesin 5x faster),
  termination_prob 0.03 (~130 s 3' dwell -> 3' barrier that relocalizes cohesin).

INTERFERENCE: rnapii_headon_push_prob 0.85 + rnapii_stall_prob 0.15 (RNAP wins head-on,
  stall force 10 vs 0.1-1 pN; effective head-on push ~0.72). block_prob 0.95 => bypass
  0.05/tick ~ Banigan k_bypass 0.01/s. lifetime_rnapii_stalled = lifetime/6 (50 ticks =
  200 s ~ t_bypass; Pol II evicts cohesin -- was =lifetime, no eviction). MD params preserved.
"""
from __future__ import annotations
import argparse
import numpy as np

CHAIN = 30000           # 30 Mb @ 1 kb/site
GENE_DENSITY = 16.0     # target /Mb; nets ~12/Mb after no-overlap packing
SEED = 7

# --- TAD-size targets (kb) --------------------------------------------------
# Dense ("short") vs sparse ("long") TAD median sizes. These are insulation-
# domain (Dixon) scale, intentionally LARGER than the biological cohesin loop
# reach (~185 kb): loops do NOT span these domains, so the readout is insulation
# / jets, not corner dots. See plan: short-tads-should-have-declarative-pebble.
SHORT_TAD_KB = 750.0    # TARGET median size of dense / short TADs
LONG_TAD_KB  = 1250.0   # TARGET median size of sparse / long TADs
# Per-interval density means cluster around ~0.24..0.83 (not 0..1), so a linear
# size<-density map needs spacing endpoints WIDER than the targets to land the
# realized dense/sparse medians on them. At the half-representative densities
# (d~0.83 dense, d~0.24 sparse) these endpoints interpolate to ~750/~1250.
SHORT_SPACING = 600.0
LONG_SPACING  = 1450.0

# class -> draw ranges. prel (pause_release_prob) capped <=0.10: at 4 s/tick that is a
# 40-130 s promoter-proximal pause (vs the old 10-25 s, too short to insulate the TSS).
CLASSES = {
    "hk_const": dict(frac=0.45, init=(0.5, 0.7),  prel=(0.06, 0.10), req=False, nenh=(0, 0)),
    "hk_high":  dict(frac=0.10, init=(0.8, 0.95), prel=(0.07, 0.10), req=False, nenh=(0, 1)),
    "celltype": dict(frac=0.30, init=(0.4, 0.6),  prel=(0.04, 0.09), req=True,  nenh=(1, 3)),
    "dev":      dict(frac=0.15, init=(0.1, 0.3),  prel=(0.03, 0.07), req=True,  nenh=(2, 5)),
}
ORDER = list(CLASSES)
FRAC = np.array([CLASSES[c]["frac"] for c in ORDER])

# --- density-driven architecture --------------------------------------------
# A smooth gene/promoter-DENSITY field is the causal latent: dense regions pack
# boundaries closer (=> small TADs), hold more genes, and get higher boundary
# strength. TRANSCRIPTION is a SEPARATE per-TAD level, correlated with density
# but set independently, so it can be ablated (txn OFF) without touching the
# domain skeleton. Size is thus a CONSEQUENCE of density, not an input.
DENS_ACTIVE = 20.0       # genes/Mb at max density
DENS_POOR   = 7.0        # genes/Mb at min density
# per-TAD class mix [hk_const, hk_high, celltype, dev] at max / min transcription
FRAC_ACTIVE = np.array([0.35, 0.30, 0.30, 0.05])   # expressed-skewed
FRAC_POOR   = np.array([0.45, 0.03, 0.20, 0.32])   # poised/silent-skewed
TXN_NOISE   = 0.15       # how much transcription decouples from density
BSTR_RANGE  = (0.1, 0.3)   # boundary-strength gradient (low -> high density)
BSTR_TXN_WEIGHT = 0.5        # boundary strength = this*flank_txn + (1-this)*flank_density;
                             # the density term forces corr(flankTADsize, bstrength) negative


def _density_field(rng):
    """Smooth gene/promoter-density profile along the chain, normalized to 0..1.

    Built from a handful of random anchors (interpolated + boxcar-smoothed) so
    the chain has coherent dense and sparse stretches rather than per-site noise.
    """
    n = max(5, CHAIN // 1000)
    anchors = rng.random(n)
    field = np.interp(np.arange(CHAIN), np.linspace(0, CHAIN - 1, n), anchors)
    w = max(1, CHAIN // (n * 4))
    field = np.convolve(field, np.ones(2 * w + 1) / (2 * w + 1), mode="same")
    return (field - field.min()) / (field.max() - field.min() + 1e-9)


def gen_tads(rng, s_small=SHORT_SPACING, s_large=LONG_SPACING):
    """Density-driven TADs: a gene-density field sets local boundary spacing, so
    dense regions EMERGE as small TADs and sparse regions as large TADs (size is
    a consequence of density, not an input). Returns ``(bounds, dens)`` where
    ``dens[i]`` is the mean density of interval i (0..1).

    ``s_small``/``s_large`` are the dense/sparse spacing endpoints (kb). `field`
    is already normalized to [0,1]; per-interval density means cluster toward
    center, so realized dense/sparse medians are compressed vs these endpoints
    -- verify with the printed "dense/sparse median size" and nudge."""
    field = _density_field(rng)
    lo_c = min(300.0, 0.5 * s_small)        # min TAD clip (kb), <= half the dense median
    hi_c = float(min(3000.0, max(s_large * 1.9, CHAIN / 3)))   # max TAD clip (kb)
    bounds, x = [], 0.0
    while x < CHAIN - lo_c:
        d = float(field[int(min(x, CHAIN - 1))])
        med = s_large * (1.0 - d) + s_small * d        # denser -> shorter step
        step = int(np.clip(rng.lognormal(np.log(med), 0.35), lo_c, hi_c))
        x += step
        if x < CHAIN - lo_c:
            bounds.append(int(x))
    intervals = list(zip([0, *bounds], [*bounds, CHAIN]))
    dens = [float(field[lo:hi].mean()) for lo, hi in intervals]
    return bounds, dens


def gene_length(rng):
    """Median ~23 kb, ~11% >100 kb (lognormal mu=ln23, sigma=1.2); tighter than
    the genome-wide dist so non-overlapping packing still reaches ~12 genes/Mb."""
    return int(np.clip(rng.lognormal(mean=np.log(23), sigma=1.2), 3, 800))


def _make_gene(rng, tss, tes, cls):
    """Assemble one gene dict for class ``cls`` (fields match the YAML schema)."""
    C = CLASSES[cls]
    g = dict(tss=int(tss), tes=int(tes), load_prob=0.015,
             requires_enhancer=C["req"], load_requires_enhancer=C["req"],
             initiation_prob=round(rng.uniform(*C["init"]), 2),
             pause_release_prob=round(rng.uniform(*C["prel"]), 2),
             # 0.20 site/tick = 0.05 kb/s = 3 kb/min Pol II; cohesin (0.25 kb/s per leg)
             # is 5x faster -> moving-barrier regime (Banigan 2023, >=3-5x).
             # termination 0.03 -> ~130 s 3' dwell: terminating Pol II is the 3' barrier
             # that relocalizes cohesin (Busslinger 2017); 0.2 (20 s) was far too short.
             elongation_step_prob=0.20, pause_offset=0, termination_prob=0.03)
    return g, C


def _add_enhancers(rng, g, C, lo, hi, strand):
    """Attach E-P enhancers in the TAD INTERIOR (never pinned to the boundary).

    Enhancers are the cohesin loading sites (``target_enhancers``). The old code
    clamped an overshoot to ``lo+5`` / ``hi-5`` (np.clip), which planted loaders
    5 sites off the boundary and manufactured loading-site fountains in the
    boundary pile-up. Overshoots are now scattered uniformly across the interior,
    keeping loaders >= M sites away from the flank.

    Enhancer SIDE is drawn independently of strand (50/50 up/downstream of the
    TSS). Distal enhancers are position- and orientation-INDEPENDENT (unlike
    promoters), so there is no biological up/down preference. The old code placed
    every enhancer on the strand's 5' side (``tss-dist`` fwd / ``tss+dist`` rev),
    which planted ~83% of cohesin loaders upstream of their gene -> a directional
    extrusion bias detectable in transcription-oriented contact analyses.
    ``strand`` is therefore no longer used for enhancer direction.
    """
    if not C["req"]:
        return
    tss = g["tss"]
    M = 10
    lo_i, hi_i = lo + M, hi - M
    ne = max(1, int(rng.integers(C["nenh"][0], C["nenh"][1] + 1)))
    enh = []
    for _e in range(ne):
        dist = int(np.clip(rng.lognormal(np.log(100), 0.9), 20, 800))  # median ~100 kb, >500 kb tail
        sign = 1 if rng.random() < 0.5 else -1     # up/downstream with equal prob
        e = tss + sign * dist
        if rng.random() < 0.10:               # ~10% cross a TAD boundary (modeling assumption)
            e = int(np.clip(e, 5, CHAIN - 5))
        elif hi_i <= lo_i:                     # tiny TAD: centre
            e = (lo + hi) // 2
        elif not (lo_i <= e <= hi_i):          # overshoot -> scatter into interior, not the flank
            e = int(rng.integers(lo_i, hi_i))
        enh.append(int(e))
    g["enhancers"] = sorted(set(enh)); g["enhancer_logic"] = "additive"


def gen_genes(rng, bounds, dens_per, txn_per):
    intervals = list(zip([0, *bounds], [*bounds, CHAIN]))
    genes = []
    EDGE = 3                                   # minimal end-margin (was 12, which barred promoters from flanks)
    for (lo, hi), d, t in zip(intervals, dens_per, txn_per):
        span = hi - lo
        if span < 40:
            continue
        # DENSITY sets gene count; TRANSCRIPTION (t) sets the expressed-class mix
        gdens = DENS_POOR + d * (DENS_ACTIVE - DENS_POOR)
        n = int(np.clip(round(gdens * span / 1000 * rng.lognormal(0, 0.3)), 0, span // 12))
        if n == 0:
            continue
        fr = (1.0 - t) * FRAC_POOR + t * FRAC_ACTIVE
        fr = fr / fr.sum()
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
            cls = "hk_high" if rng.random() < (0.15 + 0.5 * t) else "hk_const"
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


def build(txn_on, bounds, bstrength, genes, num_chains=4, lifetime=300, separation=180,
          trajectory_length=5000):
    # Pol II cap and relaxation/recording scale with locus size & chain count
    max_rnapii = (max(64, int(2560 * (CHAIN / 30000) * (num_chains / 4))) if txn_on else 0)
    life_ctcf = 4 * lifetime
    life_rnapii_stalled = max(1, lifetime // 6)   # RNAPII-evicted cohesin: 1/6 of lifetime
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
  separation: {separation}
  lifetime: {lifetime}
  lifetime_stalled: {lifetime}
  lifetime_ctcf: {life_ctcf}
  warmup_steps: 10000
  trajectory_length: {trajectory_length}
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
    # RNAPII-stalled/pushed cohesin is evicted fast (Busslinger 2017; Jeppsson 2022);
    # lifetime/6 -> at lifetime 300 = 50 ticks = 200 s ~ Banigan t_bypass (100-160 s).
    # NOT = lifetime (that gave no eviction).
    lifetime_rnapii_stalled: {life_rnapii_stalled}
    rnapii_stall_prob: 0.15
    # Head-on (converging) collisions dominate cohesin relocalization: the stronger
    # motor (RNAP, ~10 pN stall force vs cohesin 0.1-1 pN) pushes a converging cohesin
    # leg downstream, piling cohesin at 3' ends / between convergent genes (Busslinger
    # 2017; Banigan 2023 Fig 2C). headon >> codirectional. NB: this intentionally
    # diverges from the co-directional-dominant rationale in the plugin docstring
    # (rnapii.py:301-303), a comment only -- _resolve_head_on applies whatever we emit.
    rnapii_push_prob: 0.20
    rnapii_headon_push_prob: 0.85
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
    target_enhancers: true     # keep: enhancer/NIPBL loading (Kagey 2010; Fursova 2024)
    target_tss: false          # Banigan 2023: no preferential TSS loading (artifact)
    weight_loading_by_activity: true
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
  enabled: false
  max_frames: 1000
  bridge_cost: 1.0
  insulation_score_window: 500
  site_start: 0
  site_end: {CHAIN}

polymer:
  platform: cuda
  gpu: "0"
  integrator: variableLangevin
  error_tol: 0.02
  collision_rate: 0.1
  precision: single
  seed: 2042
  density: 0.2
  pbc: false
  md_steps_per_block: 635
  save_every_blocks: 1
  restart_every_blocks: 5000
  initial_relaxation_steps: {relax}   # scales with locus size
  pre_recording_steps: {prerec}
  smc_bond_wiggle: 0.1
  smc_bond_dist: 1.0
  max_data_length: 1000
  overwrite: true
  plugins:
    force_builder:
      target: polychrom.pipelines.loop_extrusion.plugins.forces:paper_force_builder
      kwargs:
        bond_length: 1.0
        bond_wiggle: 0.1
        angle_k: 1.0
        repulsion_energy: 1.5
        repulsion_radius: 1.05
        attraction_energy: 0.0
        attraction_radius: 1.1
        restrict_nonbonded_to_chains: true
        replicate_ep_pairs_across_chains: true
        ep_pairs: []
        selective_attraction_energy: 0.0
        polii_self_affinity: 0.0
        selective_repulsion_energy: 0.0
        confinement_density: 0.15
        confinement_per_chain: true
        confinement_k: 5.0
    initial_conformation:
      target: polychrom.pipelines.loop_extrusion.plugins.forces:grow_cubic_conformation

contacts:
  replicate_map_starts_across_chains: true
  map_starts: [0]
  map_size: {CHAIN}
  cutoff: [2]
  # Resolutions (in monomers) at which qc + compare run the 3D analysis: the
  # native per-monomer ICE'd map plus a 10-monomer coarsened-then-ICE'd map.
  analysis_resolutions: [1, 10]
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
    ap.add_argument("--short-spacing", type=float, default=SHORT_SPACING,
                    help="dense-region TAD spacing endpoint (kb); realized dense median is a bit higher")
    ap.add_argument("--long-spacing", type=float, default=LONG_SPACING,
                    help="sparse-region TAD spacing endpoint (kb); realized sparse median is a bit lower")
    ap.add_argument("--lifetime", type=int, default=300,
                    help="cohesin free/extruding lifetime (ticks); 4 s/tick -> 20 min residence; "
                         "bare lambda=2*lifetime=600 kb > separation (dense regime). lifetime_ctcf=4x")
    ap.add_argument("--separation", type=int, default=180,
                    help="sites per cohesin; 180 kb -> 5.6 cohesin/Mb (Rao 6/Mb). Primary lever on "
                         "effective loop size / P(s) turning point (effective loop ~= separation)")
    ap.add_argument("--trajectory-length", type=int, default=10000,
                    help="number of recorded LEF trajectory steps")
    ap.add_argument("--bstr-mult", type=float, default=1.5,
                    help="config3 = config2 with all boundary strengths X times larger")

    args = ap.parse_args()
    CHAIN = int(args.chain)
    rng = np.random.default_rng(args.seed)
    bounds, dens = gen_tads(rng, args.short_spacing, args.long_spacing)
    # transcription level per TAD: correlated with density but set independently
    txn = np.clip([d + rng.normal(0, TXN_NOISE) for d in dens], 0.0, 1.0)
    # boundary strength blends two drivers, both mapped into BSTR_RANGE:
    #   * flank TRANSCRIPTION -- active, Pol II-rich domains get stronger boundaries
    #     (transcription reinforces insulation), and
    #   * flank DENSITY -- dense domains pack into small TADs and get stronger
    #     boundaries (density is the causal latent for both size and strength).
    # The density term is what makes corr(flankTADsize, bstrength) reliably NEGATIVE
    # (smaller TAD -> stronger boundary); transcription alone could not, because txn
    # follows density only loosely (TXN_NOISE). Both use the more-extreme flank
    # (max txn / max density) so a dense, active TAD is well-insulated on both edges.
    lo_b, hi_b = BSTR_RANGE
    tb = [max(txn[i], txn[i + 1]) for i in range(len(bounds))]
    db = [max(dens[i], dens[i + 1]) for i in range(len(bounds))]
    drive = [BSTR_TXN_WEIGHT * t + (1.0 - BSTR_TXN_WEIGHT) * d for t, d in zip(tb, db)]
    bstrength = np.round(np.clip([lo_b + (hi_b - lo_b) * x + rng.normal(0, 0.03) for x in drive], lo_b, hi_b), 2)
    genes, intervals = gen_genes(rng, bounds, dens, txn)
    from pathlib import Path
    od = Path(args.out_dir)
    sfx = args.suffix
    (od / f"config1_{sfx}.yaml").write_text(build(True, bounds, bstrength, genes, args.num_chains, args.lifetime, args.separation, args.trajectory_length))
    (od / f"config2_{sfx}.yaml").write_text(build(False, bounds, bstrength, genes, args.num_chains, args.lifetime, args.separation, args.trajectory_length))
    # config3 = byte-identical to config2 (txn OFF) except every boundary strength is
    # X times larger (--bstr-mult). Stronger insulation, same domain skeleton/genes.
    bstrength_x = np.round(np.minimum(bstrength * args.bstr_mult, 1.0), 2)
    (od / f"config3_{sfx}.yaml").write_text(build(False, bounds, bstrength_x, genes, args.num_chains, args.lifetime, args.separation, args.trajectory_length))
    # per-monomer transcription "ratio" (one value per site, 0..1): every monomer in
    # a TAD inherits that TAD's transcription level, so high-tx TADs -> high ratio,
    # low-tx TADs -> low ratio. Authored for one chain (length CHAIN); the simulation
    # replicates it per chain just like genes/tad_positions. Sidecar array for analysis.
    ratio = np.empty(CHAIN, dtype=float)
    for (lo, hi), t in zip(intervals, txn):
        ratio[lo:hi] = t
    np.save(od / f"monomer_ratios_{sfx}.npy", ratio)
    # stats
    lens = [abs(g["tes"] - g["tss"]) for g in genes]
    nenh = [len(g.get("enhancers", [])) for g in genes]
    perTAD = [sum(1 for g in genes if a <= min(g['tss'], g['tes']) < b) for a, b in intervals]
    sizes = np.array([b - a for a, b in intervals], float)
    dA = np.array(dens)
    def corr(x, y): return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    med = np.median(dA)
    flank_size = np.array([min(sizes[i], sizes[i + 1]) for i in range(len(bounds))]) if bounds else np.array([])
    flank_txn = np.array(tb) if bounds else np.array([])
    print(f"CHAIN={CHAIN} num_chains={args.num_chains}  TADs: {len(intervals)} "
          f"size mean={sizes.mean():.0f} min={int(sizes.min())} max={int(sizes.max())}")
    print(f"  dense(top 50%) median size={int(np.median(sizes[dA>=med]))}  "
          f"sparse median size={int(np.median(sizes[dA<med]))}")
    print(f"  corr(density,size)={corr(dA,sizes):+.2f}  corr(txn,density)={corr(np.array(txn),dA):+.2f}")
    if len(bounds) > 2:
        print(f"  corr(flank_txn,bstrength)={corr(flank_txn,bstrength):+.2f} (active->stronger)  "
              f"corr(flankTADsize,bstrength)={corr(flank_size,bstrength):+.2f} (smaller->stronger)")
    print(f"  boundary_strength: min={bstrength.min():.2f} median={np.median(bstrength):.2f} max={bstrength.max():.2f}")
    print(f"genes: {len(genes)}  = {len(genes)/(CHAIN/1000):.1f}/Mb")
    print(f"gene len kb: median={int(np.median(lens))} mean={int(np.mean(lens))} "
          f">100kb={100*np.mean(np.array(lens)>100):.0f}%")
    print(f"E-P genes: {sum(1 for g in genes if g.get('enhancers'))}  enh/EPgene mean={np.mean([n for n in nenh if n]) if any(nenh) else 0:.1f}")
    print(f"genes/TAD: min={min(perTAD)} median={int(np.median(perTAD))} max={max(perTAD)}")
    print(f"  config3 boundary_strength (x{args.bstr_mult:g}): min={bstrength_x.min():.2f} median={np.median(bstrength_x):.2f} max={bstrength_x.max():.2f}")
    print(f"wrote {od}/config1_{sfx}.yaml , config2_{sfx}.yaml , config3_{sfx}.yaml")


if __name__ == "__main__":
    main()
