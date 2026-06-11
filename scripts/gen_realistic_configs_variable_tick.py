#!/usr/bin/env python
"""Generate paper-calibrated config1/config2/config3 for an arbitrary tick.

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

config1 = txn ON, config2 = txn OFF, config3 = txn OFF + stronger CTCF
capture, config4 = config3 + UV lesions (a typed two-state repair machine:
shorter TADs carry MORE lesions but recognise/repair them FASTER; longer TADs
fewer + slower). Deterministic (seeded). This script intentionally does not
replace ``gen_realistic_configs_20s.py``; that remains the fixed 20 s reference.

CALIBRATION: ``--tick-seconds`` sets the biological time represented by one 1D
lattice update. Biological lifetimes/rates are kept fixed and converted to
per-tick YAML values, so changing tick size changes numerical tick counts and
probabilities rather than changing the underlying biology.

COHESIN:
  * separation 240 -> C36 cohesin density estimate from Gabriele et al.
  * free/extruding residence is fixed at 25 min and converted to ticks.
    CTCF-stalled lifetime is 4x the base lifetime.
  * CTCF capture strengths are centered near the Gabriele best fit
    pstall = 1/8 with boost b = 4; config3 strengthens these further.
  * loading ~98% uniform (targeted_load_prob 0.02). target_tss FALSE (Banigan 2023:
    no preferential TSS loading); target_enhancers TRUE (Kagey 2010 / Fursova 2024).
    cohesin@genes is barrier-driven (RNAPII pinning, Busslinger 2017), not loading.

RNAPII (per gene): Banigan et al. PNAS 2023 moving-barrier rates are converted
  to the requested tick using p = 1-exp(-k*t). RNAPII velocity is held at
  0.1 kb/s by using integer ``rnapii_stride`` plus a per-substep probability.

INTERFERENCE: RNAP is modeled as a permeable moving barrier. Banigan's
  tbypass~100 s gives block_prob=exp(-tick_seconds/100) per tick. Head-on RNAP
  pushes cohesin when possible; co-directional encounters block/follow until
  RNAP vacates or cohesin bypasses. No fast RNAP-induced cohesin eviction is
  used; lifetime_rnapii_stalled equals lifetime.
"""
from __future__ import annotations
import argparse
import math
import numpy as np

CHAIN = 30000           # 30 Mb @ 1 kb/site
GENE_DENSITY = 16.0     # target /Mb; nets ~12/Mb after no-overlap packing
SEED = 7
REFERENCE_TICK_SECONDS = 20.0
REFERENCE_TRAJECTORY_LENGTH = 12096
REFERENCE_WARMUP_STEPS = 10000
REFERENCE_MD_STEPS_PER_BLOCK = 5000
# The polymer stage restarts the MD simulation every RESTART_EVERY_BLOCKS frames
# and asserts the recorded trajectory is an exact multiple of it (polymer.py).
# Keep this in sync with the `restart_every_blocks` field emitted in the template.
RESTART_EVERY_BLOCKS = 5000
COHESIN_LIFETIME_SECONDS = 25 * 60
CTCF_LIFETIME_BOOST = 4
RNAP_LOAD_RATE = 0.001
RNAP_UNPAUSE_RATE = 0.002
RNAP_UNBIND_RATE = 0.002
RNAP_SPEED_KB_PER_SECOND = 0.1
RNAP_BYPASS_SECONDS = 100.0

# Lesion (UV-damage) two-state machine (config4 only): biological stage durations
# in seconds, converted to ticks per --tick-seconds. These are the means at an
# AVERAGE-sized TAD; lesion_tad_repair_exponent makes both stages faster in
# shorter TADs and slower in longer ones, and lesion_tad_size_exponent makes
# shorter TADs carry more lesions. See plugins/lesions.py.
LESION_PRERECOGNITION_SECONDS = 1200.0
LESION_REPAIR_SECONDS = 300.0

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

# Class -> reference 20 s draw ranges. The variable-tick generator converts
# these reference per-tick probabilities through implied per-second rates.
CLASSES = {
    "hk_const": dict(frac=0.45, init=(0.5, 0.7),  prel=(0.039, 0.058), req=False, nenh=(0, 0)),
    "hk_high":  dict(frac=0.10, init=(0.8, 0.95), prel=(0.049, 0.077), req=False, nenh=(0, 1)),
    "celltype": dict(frac=0.30, init=(0.4, 0.6),  prel=(0.030, 0.049), req=True,  nenh=(1, 3)),
    "dev":      dict(frac=0.15, init=(0.1, 0.3),  prel=(0.020, 0.039), req=True,  nenh=(2, 5)),
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
BSTR_RANGE  = (0.04, 0.13)  # boundary-strength gradient near pstall=1/8
BSTR_NOISE  = 0.01
BSTR_TXN_WEIGHT = 0.5        # boundary strength = this*flank_txn + (1-this)*flank_density;
                             # the density term forces corr(flankTADsize, bstrength) negative


def _rate_to_prob(rate_per_second: float, tick_seconds: float) -> float:
    """Convert a continuous-time rate to a per-tick event probability."""
    return 1.0 - math.exp(-float(rate_per_second) * float(tick_seconds))


def _ref_prob_to_rate(prob: float, ref_tick_seconds: float = REFERENCE_TICK_SECONDS) -> float:
    """Infer a continuous rate from a reference per-tick probability."""
    p = min(max(float(prob), 0.0), 1.0 - 1e-12)
    return -math.log1p(-p) / float(ref_tick_seconds)


def _scale_ref_prob(prob: float, tick_seconds: float) -> float:
    """Convert a 20 s reference probability to the requested tick size."""
    return _rate_to_prob(_ref_prob_to_rate(prob), tick_seconds)


def _scale_ref_range(bounds, tick_seconds: float):
    return tuple(_scale_ref_prob(x, tick_seconds) for x in bounds)


def _round_prob(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def make_calibration(tick_seconds: float, trajectory_length=None, warmup_steps=None) -> dict:
    """Compute all tick-dependent defaults from biological rates/durations."""
    if tick_seconds <= 0:
        raise ValueError("--tick-seconds must be positive")
    target_sites_per_tick = RNAP_SPEED_KB_PER_SECOND * tick_seconds
    rnapii_stride = max(1, int(math.ceil(target_sites_per_tick - 1e-12)))
    elongation_step_prob = min(1.0, target_sites_per_tick / rnapii_stride)
    lifetime = max(1, int(round(COHESIN_LIFETIME_SECONDS / tick_seconds)))
    traj = (
        max(1, int(round(REFERENCE_TRAJECTORY_LENGTH * REFERENCE_TICK_SECONDS / tick_seconds)))
        if trajectory_length is None else int(trajectory_length)
    )
    # The polymer stage requires the trajectory to be an exact multiple of
    # restart_every_blocks (it errors out otherwise). Snap to the nearest
    # multiple so generated configs are always runnable; never below one block.
    traj = max(RESTART_EVERY_BLOCKS, int(round(traj / RESTART_EVERY_BLOCKS)) * RESTART_EVERY_BLOCKS)
    warm = (
        max(0, int(round(REFERENCE_WARMUP_STEPS * REFERENCE_TICK_SECONDS / tick_seconds)))
        if warmup_steps is None else int(warmup_steps)
    )
    md_steps = max(1, int(round(REFERENCE_MD_STEPS_PER_BLOCK * tick_seconds / REFERENCE_TICK_SECONDS)))
    return {
        "tick_seconds": float(tick_seconds),
        "lifetime": lifetime,
        "lifetime_ctcf": CTCF_LIFETIME_BOOST * lifetime,
        "trajectory_length": traj,
        "restart_every_blocks": RESTART_EVERY_BLOCKS,
        "warmup_steps": warm,
        "md_steps_per_block": md_steps,
        "rnapii_stride": rnapii_stride,
        "elongation_step_prob": elongation_step_prob,
        "rnap_load_prob": _rate_to_prob(RNAP_LOAD_RATE, tick_seconds),
        "rnap_termination_prob": _rate_to_prob(RNAP_UNBIND_RATE, tick_seconds),
        "rnap_block_prob": math.exp(-tick_seconds / RNAP_BYPASS_SECONDS),
        "class_ranges": {
            cls: {
                "init": _scale_ref_range(spec["init"], tick_seconds),
                "prel": _scale_ref_range(spec["prel"], tick_seconds),
            }
            for cls, spec in CLASSES.items()
        },
    }


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


def _make_gene(rng, tss, tes, cls, calibration):
    """Assemble one gene dict for class ``cls`` (fields match the YAML schema)."""
    C = CLASSES[cls]
    ranges = calibration["class_ranges"][cls]
    g = dict(tss=int(tss), tes=int(tes),
             load_prob=_round_prob(calibration["rnap_load_prob"], 4),
             requires_enhancer=C["req"], load_requires_enhancer=C["req"],
             initiation_prob=_round_prob(rng.uniform(*ranges["init"]), 4),
             pause_release_prob=_round_prob(rng.uniform(*ranges["prel"]), 4),
             # stateful_translocate_rnapii uses rnapii_stride below; stride times
             # this probability preserves 0.1 kb/s on average for any tick.
             elongation_step_prob=round(calibration["elongation_step_prob"], 6),
             pause_offset=0,
             termination_prob=_round_prob(calibration["rnap_termination_prob"], 4))
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


def gen_genes(rng, bounds, dens_per, txn_per, calibration):
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
            g, C = _make_gene(rng, tss_pos, tes_pos, cls, calibration)
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
            g, C = _make_gene(rng, tss_pos, tes_pos, cls, calibration)
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


DEFAULT_BSTR = 0.13  # default_boundary_strength emitted below; ends forced to 1.0


def make_tad_records(bounds, bstrength, chain, gap_frac=0.0):
    """Per-TAD oriented-boundary records from shared boundaries + per-boundary
    strengths.

    TAD ``i`` takes its left_strength from the boundary on its left
    (``bstrength[i-1]``) and its right_strength from the boundary on its right
    (``bstrength[i]``); the two chromosome-end outer sides use the default (the
    topology forces them to hard 1.0 walls regardless). This reproduces the
    legacy ``tad_positions`` + ``boundary_strength`` ``ctcfCapture`` exactly.

    With ``gap_frac > 0`` each interior TAD sheds ``gap_frac`` of its span on the
    right, opening an anchor-free spacer before the next TAD. ``gap_frac == 0``
    (default) keeps the anchors byte-identical to the legacy abutting layout.
    """
    edges = [0, *list(bounds), chain]
    n = len(edges) - 1
    records = []
    for i in range(n):
        left = edges[i]
        right = edges[i + 1] - 1
        ls = float(bstrength[i - 1]) if i > 0 else DEFAULT_BSTR
        rs = float(bstrength[i]) if i < len(bounds) else DEFAULT_BSTR
        if gap_frac > 0.0 and i < n - 1:
            gw = int(round(gap_frac * (right - left)))
            if 1 <= gw < right - left:
                right -= gw
        records.append((left, right, ls, rs))
    return records


def build(txn_on, bounds, bstrength, genes, calibration, num_chains=4, separation=240,
          lesions=None, gap_frac=0.0):
    # Pol II cap and relaxation/recording scale with locus size & chain count
    max_rnapii = (max(64, int(2560 * (CHAIN / 30000) * (num_chains / 4))) if txn_on else 0)
    lifetime = calibration["lifetime"]
    life_ctcf = calibration["lifetime_ctcf"]
    life_rnapii_stalled = lifetime
    relax = max(50000, int(1500000 * CHAIN / 30000))
    prerec = max(20000, int(500000 * CHAIN / 30000))
    rl = ("polychrom.pipelines.loop_extrusion.plugins.rnapii:load_rnapii" if txn_on else "null")
    rt = ("polychrom.pipelines.loop_extrusion.plugins.rnapii:stateful_translocate_rnapii" if txn_on else "null")
    records = make_tad_records(bounds, bstrength, CHAIN, gap_frac)
    tads_block = "\n".join(
        f"      - {{left: {l}, right: {r}, left_strength: {ls:.2f}, right_strength: {rs:.2f}}}"
        for l, r, ls, rs in records
    )
    gene_lines = "\n".join(f"      - {fmt_gene(g)}" for g in genes)
    # Lesion block: when ``lesions`` is given (config4) emit the full UV-damage
    # knob set + enable the lesion plugin; otherwise lesions are off. (We always
    # emit ``lesion_spacing`` -- never the retired ``lesion_prob`` -- so the
    # topology plugin accepts the kwargs.)
    if lesions:
        ls = lesions
        lesion_kwargs_block = "\n".join([
            f"    lesion_spacing: {ls['spacing']}              # steady-state lesions = num_sites // spacing",
            f"    lesion_type_a_prob: {ls['type_a_prob']}      # P(Type A | in a gene body); off-gene always Type B",
            f"    lesion_prerecognition_ticks: {ls['prerecognition_ticks']}   # ~{ls['prerecognition_seconds']:g}s mean at an average-sized TAD",
            f"    lesion_repair_ticks: {ls['repair_ticks']}            # ~{ls['repair_seconds']:g}s mean at an average-sized TAD",
            f"    lesion_block_prob: {ls['block_prob']}      # per-tick prob a stalling lesion blocks a cohesin leg",
            f"    lesion_tad_size_exponent: {ls['tad_size_exponent']}   # L_TAD**(-alpha): shorter TADs carry MORE lesions",
            f"    lesion_tad_repair_exponent: {ls['tad_repair_exponent']} # (L_mean/L_TAD)**beta: shorter TADs recognise/repair FASTER",
            f"    lesion_type_b_enabled: {str(ls['type_b_enabled']).lower()}  # false -> only Type-A lesions kept (Type-A count unchanged, no backfill)",
        ])
        lesion_plugin_line = (
            "\n    lesion:      polychrom.pipelines.loop_extrusion.plugins.lesions:update_lesions"
        )
    else:
        lesion_kwargs_block = "    lesion_spacing: 0            # lesions off (config1/2/3)"
        lesion_plugin_line = ""
    return f"""lef:
  chain_length: {CHAIN}
  num_chains: {num_chains}
  separation: {separation}
  lifetime: {lifetime}
  lifetime_stalled: {lifetime}
  lifetime_ctcf: {life_ctcf}
  tick_seconds: {calibration["tick_seconds"]:g}
  warmup_steps: {calibration["warmup_steps"]}
  trajectory_length: {calibration["trajectory_length"]}
  chunk_size: 50
  seed: 42
  max_rnapii: {max_rnapii}

  topology_kwargs:
    tads:
{tads_block}
    default_boundary_strength: 0.13
    release_prob: 0.0
    include_chromosome_ends: true
    # Paper-calibrated: RNAP is a permeable moving barrier (Banigan), not a fast
    # cohesin-eviction state. Keep RNAP-stalled cohesin at the base lifetime.
    lifetime_rnapii_stalled: {life_rnapii_stalled}
    rnapii_stall_prob: 0.0
    rnapii_stride: {calibration["rnapii_stride"]}
    # Banigan moving barrier: head-on RNAP pushes cohesin where geometry allows;
    # co-directional encounters are slowed/follow until RNAP vacates or bypasses.
    rnapii_push_prob: 0.0
    rnapii_headon_push_prob: 1.0
    rnapii_pause_cohesin_restraint: 0.3
    rnapii_pause_restraint_window: 1
    rnapii_poised_block_prob: {_round_prob(calibration["rnap_block_prob"], 3)}
    rnapii_paused_block_prob: {_round_prob(calibration["rnap_block_prob"], 3)}
    rnapii_elongating_block_prob: {_round_prob(calibration["rnap_block_prob"], 3)}
    rnapii_terminating_block_prob: {_round_prob(calibration["rnap_block_prob"], 3)}
    ep_contact_tolerance: 1
    replicate_genes_across_chains: true
    targeted_load_prob: 0.02
    loading_window: 1
    target_enhancers: true     # keep: enhancer/NIPBL loading (Kagey 2010; Fursova 2024)
    target_tss: false          # Banigan 2023: no preferential TSS loading (artifact)
    weight_loading_by_activity: true
{lesion_kwargs_block}
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
    rnapii_translocate: {rt}{lesion_plugin_line}

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
  md_steps_per_block: {calibration["md_steps_per_block"]}
  save_every_blocks: 1
  restart_every_blocks: {calibration["restart_every_blocks"]}
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
    ap.add_argument("--tick-seconds", type=float, required=True,
                    help="biological seconds represented by one 1D lattice tick")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chain", type=int, default=CHAIN, help="chain length in sites (1 site = 1 kb)")
    ap.add_argument("--num-chains", type=int, default=4)
    ap.add_argument("--suffix", default="real", help="config name suffix: config1_<suffix>.yaml")
    ap.add_argument("--short-spacing", type=float, default=SHORT_SPACING,
                    help="dense-region TAD spacing endpoint (kb); realized dense median is a bit higher")
    ap.add_argument("--long-spacing", type=float, default=LONG_SPACING,
                    help="sparse-region TAD spacing endpoint (kb); realized sparse median is a bit lower")
    ap.add_argument("--separation", type=int, default=240,
                    help="sites per cohesin; 240 kb matches the Gabriele C36 density estimate")
    ap.add_argument("--trajectory-length", type=int, default=None,
                    help="number of recorded LEF trajectory steps; default preserves the 20 s generator's real duration")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="discarded 1D warmup ticks; default preserves the 20 s generator's real duration")
    ap.add_argument("--bstr-mult", type=float, default=3.0,
                    help="config3 = config2 with all boundary strengths X times larger")
    ap.add_argument("--gap-frac", type=float, default=0.0,
                    help="fraction of each interior TAD's span left anchor-free as an "
                         "inter-TAD gap (0 = abutting TADs, byte-identical to legacy)")
    # config4 (= config3 + UV lesions) knobs
    ap.add_argument("--lesion-spacing", type=int, default=10,
                    help="config4: steady-state lesion count = num_sites // spacing (0 disables)")
    ap.add_argument("--lesion-type-a-prob", type=float, default=0.25,
                    help="config4: P(Type A | in a gene body); off-gene lesions are always Type B")
    ap.add_argument("--lesion-prerecognition-seconds", type=float, default=LESION_PRERECOGNITION_SECONDS,
                    help="config4: mean pre-recognition duration at an average-sized TAD")
    ap.add_argument("--lesion-repair-seconds", type=float, default=LESION_REPAIR_SECONDS,
                    help="config4: mean repair duration at an average-sized TAD")
    ap.add_argument("--lesion-block-prob", type=float, default=0.97,
                    help="config4: per-tick prob a stalling lesion blocks a cohesin leg")
    ap.add_argument("--lesion-tad-size-exponent", type=float, default=1.0,
                    help="config4: placement weight L_TAD**(-alpha); larger -> shorter TADs carry more lesions")
    ap.add_argument("--lesion-tad-repair-exponent", type=float, default=1.0,
                    help="config4: rate * (L_mean/L_TAD)**beta; larger -> shorter TADs recognise/repair faster")
    ap.add_argument("--lesion-type-b-enabled", action=argparse.BooleanOptionalAction, default=False,
                    help="config4: include Type-B (GG-NER) lesions. DEFAULT OFF -> only Type-A lesions "
                         "(the Type-A count is unchanged, no backfill). Pass --lesion-type-b-enabled to add Type-B.")

    args = ap.parse_args()
    CHAIN = int(args.chain)
    calibration = make_calibration(
        args.tick_seconds,
        trajectory_length=args.trajectory_length,
        warmup_steps=args.warmup_steps,
    )
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
    bstrength = np.round(np.clip([lo_b + (hi_b - lo_b) * x + rng.normal(0, BSTR_NOISE) for x in drive], lo_b, hi_b), 2)
    genes, intervals = gen_genes(rng, bounds, dens, txn, calibration)
    from pathlib import Path
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    sfx = args.suffix
    (od / f"config1_{sfx}.yaml").write_text(build(True, bounds, bstrength, genes, calibration, args.num_chains, args.separation, gap_frac=args.gap_frac))
    (od / f"config2_{sfx}.yaml").write_text(build(False, bounds, bstrength, genes, calibration, args.num_chains, args.separation, gap_frac=args.gap_frac))
    # config3 = byte-identical to config2 (txn OFF) except every boundary strength is
    # X times larger (--bstr-mult). Stronger insulation, same domain skeleton/genes.
    bstrength_x = np.round(np.minimum(bstrength * args.bstr_mult, 1.0), 2)
    (od / f"config3_{sfx}.yaml").write_text(build(False, bounds, bstrength_x, genes, calibration, args.num_chains, args.separation, gap_frac=args.gap_frac))
    # config4 = config3 (txn OFF, stronger CTCF) + UV lesions. Stage durations are
    # biological seconds converted to ticks at this tick size; shorter TADs carry
    # more lesions (size exponent) but recognise/repair them faster (repair exponent).
    ts = calibration["tick_seconds"]
    lesion_params = dict(
        spacing=args.lesion_spacing,
        type_a_prob=args.lesion_type_a_prob,
        prerecognition_seconds=args.lesion_prerecognition_seconds,
        repair_seconds=args.lesion_repair_seconds,
        prerecognition_ticks=max(1, int(round(args.lesion_prerecognition_seconds / ts))),
        repair_ticks=max(1, int(round(args.lesion_repair_seconds / ts))),
        block_prob=args.lesion_block_prob,
        tad_size_exponent=args.lesion_tad_size_exponent,
        tad_repair_exponent=args.lesion_tad_repair_exponent,
        type_b_enabled=args.lesion_type_b_enabled,
    )
    (od / f"config4_{sfx}.yaml").write_text(build(False, bounds, bstrength_x, genes, calibration, args.num_chains, args.separation, lesions=lesion_params, gap_frac=args.gap_frac))
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
    expected_kb_min = calibration["rnapii_stride"] * calibration["elongation_step_prob"] / calibration["tick_seconds"] * 60.0
    pause_ranges = [v["prel"] for v in calibration["class_ranges"].values()]
    pause_min = min(lo for lo, _hi in pause_ranges)
    pause_max = max(hi for _lo, hi in pause_ranges)
    print("calibration:")
    print(f"  tick_seconds={calibration['tick_seconds']:g}")
    print(f"  lifetime={calibration['lifetime']} ticks ({calibration['lifetime'] * calibration['tick_seconds'] / 60:.1f} min)  "
          f"lifetime_ctcf={calibration['lifetime_ctcf']}")
    print(f"  warmup_steps={calibration['warmup_steps']} ({calibration['warmup_steps'] * calibration['tick_seconds'] / 3600:.2f} h)  "
          f"trajectory_length={calibration['trajectory_length']} ({calibration['trajectory_length'] * calibration['tick_seconds'] / 3600:.2f} h)")
    print(f"  RNAP load_prob={calibration['rnap_load_prob']:.6f}  "
          f"pause_release_prob_range={pause_min:.6f}-{pause_max:.6f}  "
          f"termination_prob={calibration['rnap_termination_prob']:.6f}  "
          f"block_prob={calibration['rnap_block_prob']:.6f}")
    print(f"  RNAP stride={calibration['rnapii_stride']}  "
          f"elongation_step_prob={calibration['elongation_step_prob']:.6f}  "
          f"expected_elongation={expected_kb_min:.2f} kb/min")
    print(f"  md_steps_per_block={calibration['md_steps_per_block']}")
    print(f"genes: {len(genes)}  = {len(genes)/(CHAIN/1000):.1f}/Mb")
    print(f"gene len kb: median={int(np.median(lens))} mean={int(np.mean(lens))} "
          f">100kb={100*np.mean(np.array(lens)>100):.0f}%")
    print(f"E-P genes: {sum(1 for g in genes if g.get('enhancers'))}  enh/EPgene mean={np.mean([n for n in nenh if n]) if any(nenh) else 0:.1f}")
    print(f"genes/TAD: min={min(perTAD)} median={int(np.median(perTAD))} max={max(perTAD)}")
    print(f"  config3 boundary_strength (x{args.bstr_mult:g}): min={bstrength_x.min():.2f} median={np.median(bstrength_x):.2f} max={bstrength_x.max():.2f}")
    n_sites = CHAIN * args.num_chains
    target = (n_sites // args.lesion_spacing) if args.lesion_spacing > 0 else 0
    print(f"config4 lesions: spacing={args.lesion_spacing} -> ~{target} lesions held "
          f"({100.0 * target / n_sites:.1f}% of {n_sites} sites); "
          f"pre-recognition={lesion_params['prerecognition_ticks']} ticks (~{args.lesion_prerecognition_seconds:g}s), "
          f"repair={lesion_params['repair_ticks']} ticks (~{args.lesion_repair_seconds:g}s); "
          f"size_exp(alpha)={args.lesion_tad_size_exponent:g}, repair_exp(beta)={args.lesion_tad_repair_exponent:g}")
    print(f"wrote {od}/config1_{sfx}.yaml , config2_{sfx}.yaml , config3_{sfx}.yaml , config4_{sfx}.yaml")


if __name__ == "__main__":
    main()
