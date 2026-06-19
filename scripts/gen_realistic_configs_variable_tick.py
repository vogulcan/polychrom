#!/usr/bin/env python
"""Generate human-calibrated config1-4 for the loop-extrusion + transcription
pipeline at an arbitrary 1D tick size (1 lattice site = 1 kb).

All magnitudes below are TICK-INDEPENDENT biology (rates in /s, durations in s,
sizes in kb, distributions); ``--tick-seconds`` only converts them to per-tick
YAML probabilities via p = 1 - exp(-k*t). Values are sourced from human (or, where
noted, the closest mammalian) in-vivo measurements and were checked against the
human bands in gen_transcription_metrics.py HUMAN_RANGES. Key anchors:

  ELONGATION  ~3.18 kb/min. Genome-wide human in vivo spans ~1-6 kb/min; DRB
              peak methods give a low median (~1.3-2.2 kb/min; biased low on long
              genes), 4sUDRB ~3.5 kb/min, wave-front/live-imaging the high end.
              3.18 sits high-central (Veloso 2014 Genome Res; Fuchs 2014; Jonkers
              & Lis 2015 NRMCB).
  PAUSING     Single-Pol pause dwell ~0.1-1 min; ~75-80% of paused Pol II
              prematurely terminate => productive fraction ~0.15-0.25 (Gressel/
              Cramer 2019 Nat Commun; Zimmer/Adelman STL-seq 2021 Mol Cell;
              Steurer 2018 PNAS live-imaging ~42 s paused residence). The
              minutes-scale "apparent" pause is occupancy/output, not single-Pol.
  INITIATION  Productive initiation median ~1/min for human mRNAs (lincRNA ~0.3,
              eRNA ~0.15, extreme ~80/min); loading anchor kept moderate because
              realized initiation is PAUSE-ESCAPE-limited, not loading-limited
              (Gressel/Cramer 2019; Zhao/Siepel 2023 NAR).
  3' DWELL    ~3 min terminating dwell (0.7-0.9 kb/min over a ~1-4 kb termination
              zone; Cortazar 2019 Mol Cell; Fong/Bentley 2015).
  TADs        Kept at the classical/directionality scale (mean ~833 kb). NB the
              convergent-CTCF + cohesin-extrusion MECHANISM here corresponds to
              Rao 2014 (Cell) human contact/loop domains (median ~185 kb, 92%
              convergent CTCF); a model TAD is therefore a classical domain that
              may contain nested CTCF loops. Dixon 2012's ~880 kb median is MOUSE
              and a different (directionality) definition -- not relabeled human.
  CTCF        Per-encounter cohesin stall at a boundary ~0.06-0.18 (centered on
              Gabriele 2022 Science 12.5%); config3 strengthens it x2.5.
  COHESIN     density 1/240 kb and base residence ~20 min, CTCF-stabilized x4
              (Gabriele 2022 Science; Hansen 2017 eLife ~22 min; Cattoglio 2019).
  CLASSES     ~20% housekeeping (Eisenberg & Levanon 2013 ~3804 HK; HRT Atlas
              ~2176), the rest cell-type / developmental; E-P distance median
              ~30 kb (Gasperini 2019 ~24 kb; Engreitz/ABC 87% <100 kb).

config1 = txn ON; config2 = txn OFF (same skeleton/genes); config3 = config2 +
stronger CTCF (x bstr-mult); config4 = config3 + UV lesions. Deterministic (seeded).
"""

from __future__ import annotations
import argparse
import math
import numpy as np

# Each constant below notes its formula and the effect of a small INCREASE (↑);
# a small decrease (↓) is the opposite unless stated otherwise. "this" = the
# constant's value, "tick" = tick_seconds.

# ============================== Locus & RNG ==============================
# Lattice length in sites (1 site = 1 kb) -> 10 Mb locus; --chain overrides.
# ↑ longer locus: more genes & cohesins, more compute.
CHAIN = 10000
GENE_DENSITY = 16.0            # target genes/Mb before no-overlap packing; ↑ more, denser genes
SEED = 7                       # default RNG seed (--seed overrides); changes the random layout, not a magnitude

# ========================= Timing / tick rescaling =======================
# Per-tick probabilities below are authored at REFERENCE_TICK_SECONDS and
# rescaled to the requested --tick-seconds via continuous-time rates.
REFERENCE_TICK_SECONDS = 20.0        # authoring tick for the prob ranges; change only when re-deriving them
REFERENCE_TRAJECTORY_LENGTH = 12096  # recorded LEF steps (rescaled by tick); ↑ longer recording, better stats, slower
REFERENCE_WARMUP_STEPS = 10000       # discarded warm-up ticks (rescaled); ↑ longer equilibration before recording
REFERENCE_MD_STEPS_PER_BLOCK = 5000  # 3D MD steps per recorded block (rescaled); ↑ more relaxation/block, slower
RESTART_EVERY_BLOCKS = 5000          # polymer restart interval (traj snapped to a multiple); ↑ fewer restarts

# ========================== Cohesin (loop extrusion) =====================
# Mean cohesin residence /s; lifetime_ticks = this/tick. ↑ longer-lived cohesin -> longer loops.
# 20 min = the live-imaging base residence (Gabriele 2022 Science ~20 min; Hansen 2017
# eLife ~22 min mESC FRAP). 25 min was the top of the measured range; 20 min is central.
COHESIN_LIFETIME_SECONDS = 20 * 60
# CTCF-anchored lifetime = this * free-cohesin lifetime. ↑ stronger CTCF retention (sharper
# boundaries / corner dots). x4 is verbatim Gabriele 2022 ("stabilized fourfold"): ~80 min boosted.
CTCF_LIFETIME_BOOST = 4

# ====================== RNAPII transcription kinetics ====================
# Expressed-gene loading rate /s (geo-mean anchor); load_prob = 1-exp(-this*tick).
# With the 3' termination window the single-slot-TES throughput ceiling is gone, so output is now
# loading-limited. Two soft ceilings bound the hottest genes: (a) gene-body DENSITY (steady-state
# density = productive_body_entry / elongation_velocity <= ~0.25 Pol/kb), and (b) the promoter
# pause-site clearance -- the paused Pol occupies tss+pause_offset for ~one pause dwell, so a gene
# re-initiates at most ~1/dwell ~2/min, i.e. completion <= ~0.5/min (productive_fraction ~0.25).
# The geo-mean sets the typical gene; LOAD_RATE_MAX caps the active tail near those ceilings.
RNAP_LOAD_RATE = 0.038
# Hard cap on any single gene's loading rate /s. Raised to 0.07 once the 3' window removed the
# single-slot jam: the active tail now reaches completion ~0.45-0.6/min (more of the human 0.2-1.5
# band) at density ~0.2 Pol/kb, bounded by the promoter pause-site clearance rather than by 3'
# throughput. body_entry = load * init(~0.9) * productive_fraction(~0.25). Paired with a MODEST
# PAUSE_ACTIVE_BOOST (the active tail's density is load_cap x boost; keep their product in check).
LOAD_RATE_MAX = 0.085
# Premature-termination rate /s at the pause; productive_fraction = k_release/(k_release+this).
# ↑ more abortion -> lower productive_fraction & mRNA output. 0.025 centers productive_fraction at
# ~0.20-0.25 (STL-seq / Mukherjee & Guertin ~20-25%); 0.034 was the high end and pinned it at ~0.16,
# which (with pause-escape-limited loading) held expressed-gene output below the human band.
RNAP_PAUSE_TERM_RATE = 0.025
# 3'-termination/unbind rate /s; per-tick release probability for a Pol in the termination window
# (and the legacy single-slot dwell). ↑ faster release -> shorter 3' dwell, weaker 3' barrier.
RNAP_UNBIND_RATE = 0.005677
# 3' TERMINATION WINDOW: a Pol reaching the TES walks DOWNSTREAM through a per-gene window before
# releasing, instead of blocking the single TES slot. Schwalb 2016 (TT-seq, human K562): termination
# sites lie in a window of median width ~3.3 kb past the last poly(A) site, up to >10 kb. Modeled as
# a lognormal per-gene length (sites=kb). The window VACATES the TES on the Pol's first step, so the
# 3' end no longer caps gene output at 1/dwell (which packed long active genes solid). 0 = legacy slot.
TERM_WINDOW_KB_MEDIAN = 3.3        # median 3' termination-window width (Schwalb 2016: median ~3.3 kb)
TERM_WINDOW_SIGMA = 0.8            # lognormal sigma -> ~8% of genes have windows >10 kb (Schwalb tail)
# Decelerated elongation speed of a terminating Pol crawling the window (Cortazar 2019: 3' rate drops
# to ~0.7-0.9 kb/min). Sets rnapii_termination_step_prob; with the window this gives ~1-3 min total
# termination time and a 3' throughput of ~1/min (vs 0.34/min for the old single TES slot).
TERM_SPEED_KB_PER_SECOND = 0.017
# Elongation speed; kb/min = 60*this -> elongation_step_prob & stride. ↑ faster transit -> lower gene-body occupancy.
# 0.053 kb/s = 3.18 kb/min: high-central in the human in-vivo range (DRB-peak median ~1.3-2.2,
# 4sUDRB ~3.5, wave-front higher; Veloso 2014, Fuchs 2014, Jonkers & Lis 2015). ~0.042 (2.5
# kb/min) is the population-method central if a slower elongation is preferred.
RNAP_SPEED_KB_PER_SECOND = 0.053
# Max x-multiplier on pause-release for the most active genes (ramped over the top activity half).
# Models CDK9/P-TEFb-accelerated pause release at active genes (Gressel 2017: shorter pause -> higher
# productive initiation). Faster release also clears the single-occupancy pause site sooner, relieving
# the promoter back-pressure that throttles re-initiation, so the active tail's output rises toward
# the human band. SAFE only because the 3' termination window now absorbs the extra body->3' flux
# (with the single-slot TES this jammed). Kept MODEST (2.5): a larger boost drives the most active
# genes' productive_fraction = k_rel/(k_rel+k_term) well above the realistic ~0.25 and their gene-body
# density past ~0.25 Pol/kb. Median productive_fraction is unchanged (only the top half is boosted).
# 2.0 is the sweet spot: 2.5 lifted output more but pushed ~3% of genes' density >0.5 Pol/kb (mild
# re-crowding) and the active-gene productive_fraction tail too high.
PAUSE_ACTIVE_BOOST = 2.0
# Floor on pause duration (s); release-prob ceiling = 1-exp(-tick/this). ↑ slower max escape ->
# more paused-state time (higher pausing index) and slower body entry (less jamming). 25 s sits at
# the upper end of the single-Pol paused residence (Steurer 2018 ~42 s); a smaller floor let the
# hottest genes dump Pol into already-throughput-capped bodies and flattened the promoter pause peak.
PAUSE_MIN_S = 25

# ============== Activity distribution (loading heterogeneity) ============
# Per-gene loading spread (log-normal). ↑ wider activity range: heavier active tail + more near-silent genes.
LOAD_LOGNORM_SIGMA = 1.3
# Per-class fraction of genes forced near-silent (cell-type/developmental off in this cell). ↑ more silent genes.
SILENT_PROB = {"hk_const": 0.0, "hk_high": 0.0, "celltype": 0.4, "dev": 0.7}
LOAD_SILENT_FRAC = 0.03        # silent genes load at this x the expressed anchor; ↑ leakier (less silent)
# Loading factor = (median_TAD_size / TAD_size)^this. ↑ stronger size->activity coupling (small TADs more active).
LOAD_TADSIZE_EXPONENT = 0.2

# ============== RNAPII <-> cohesin coupling (moving barrier) =============
# Cohesin-bypass time (s) for an elongating/paused Pol; block_prob = exp(-tick/this), k_bypass = 1/this.
# ↑ stronger, less-permeable Pol barrier -> more loop shrinkage / cohesin eviction.
RNAP_BYPASS_SECONDS = 100.0
# Same, for a terminating Pol at the 3' end; block_prob = exp(-tick/this). ↑ stronger 3' barrier.
RNAP_TERM_BYPASS_SECONDS = 100.0
# Pre-initiation Pol bypass time (s); block_prob = exp(-tick/this). ↑ less-leaky pre-init barrier (stronger).
RNAP_PRE_INITIATION_BYPASS_SECONDS = 12.0
# Prob an elongating Pol pushes a converging (head-on) cohesin leg (1 = always; Banigan). ↑ relocates cohesin more.
RNAP_HEADON_PUSH_PROB = 1.0
# Prob it pushes a co-directional (trailing) leg (0 = impede only). ↑ Pol also drags trailing cohesin.
RNAP_PUSH_PROB = 0.0
# Intrinsic prob the Pol stalls on any cohesin contact (0 = off; non-Banigan extension). ↑ Pol loses more encounters.
RNAP_STALL_PROB = 0.0
# Cohesin next to a paused Pol scales its pause-release by this (1 = off). ↓ <1 slows release (stronger restraint).
RNAP_PAUSE_RESTRAINT = 1.0

# ============================= Lesions (config4) =========================
# Mean time (s) a UV lesion sits unrecognized before repair starts. ↑ lesions linger / block longer.
LESION_PRERECOGNITION_SECONDS = 2400.0
# Mean repair duration (s) once recognized. ↑ slower repair (lesions persist longer).
LESION_REPAIR_SECONDS = 360.0

# ======================= TAD architecture & boundaries ===================
# A smooth gene-density field is the causal latent -> TAD size, class mix, and
# boundary strength all follow from it (size is a consequence, not an input).
SHORT_TAD_KB = 750.0           # target median size of dense/short TADs (diagnostic; SHORT_SPACING is the knob)
LONG_TAD_KB  = 1250.0          # target median size of sparse/long TADs (diagnostic; LONG_SPACING is the knob)
SHORT_SPACING = 600.0          # dense-region boundary-spacing endpoint (kb); ↑ larger dense TADs
LONG_SPACING  = 1450.0         # sparse-region boundary-spacing endpoint (kb); ↑ larger sparse TADs
DENS_ACTIVE = 20.0             # genes/Mb at max density; ↑ more genes in dense regions
DENS_POOR   = 7.0              # genes/Mb at min density; ↑ more genes in sparse regions
TXN_NOISE   = 0.12             # transcription<->density/size decoupling; ↑ txn less predictable from architecture
# txn = this*(inverse TAD size) + (1-this)*density + noise. ↑ size dominates txn over density (long TADs less active).
TAD_SIZE_TXN_WEIGHT = 0.78
BSTR_RANGE  = (0.06, 0.18)     # CTCF boundary per-encounter cohesin-stall prob; ↑ stronger insulation.
                               # Centered on Gabriele 2022 Science (12.5% per-encounter stall, Fbn2 mESC);
                               # plausible in-vivo band ~0.125-0.25. Was (0.04,0.13), low vs the measured value.
BSTR_NOISE  = 0.01             # jitter added to boundary strength; ↑ more boundary variability
# boundary strength = this*flank_txn + (1-this)*flank_density. ↑ strength driven more by transcription than density.
BSTR_TXN_WEIGHT = 0.5

# =========================== Gene classes & mix ==========================
# frac = genome fraction; init = PRE-INITIATION->PAUSED prob range (ref 20 s); prel = pause-release
# rate range k_release (ref 20 s); req = enhancer-gated; nenh = (min,max) enhancers. Classes:
#   hk_const=housekeeping, hk_high=housekeeping/high, celltype=cell-type-specific, dev=developmental.
# frac = documentation-only genome-wide target mix (NOT used: gene class is drawn per TAD
# from FRAC_ACTIVE/FRAC_POOR below). Housekeeping is ~20% of protein-coding genes (Eisenberg
# & Levanon 2013 ~3804 strict HK; HRT Atlas 2021 ~2176), the rest cell-type/developmental.
CLASSES = {
    "hk_const": dict(frac=0.15, init=(0.5, 0.7),  prel=(0.098, 0.145), req=False, nenh=(0, 0)),
    "hk_high":  dict(frac=0.05, init=(0.8, 0.95), prel=(0.123, 0.193), req=False, nenh=(0, 1)),
    "celltype": dict(frac=0.55, init=(0.4, 0.6),  prel=(0.075, 0.123), req=True,  nenh=(1, 3)),
    "dev":      dict(frac=0.25, init=(0.1, 0.3),  prel=(0.050, 0.098), req=True,  nenh=(2, 5)),
}
ORDER = list(CLASSES)
FRAC = np.array([CLASSES[c]["frac"] for c in ORDER])   # documentation-only; see FRAC_ACTIVE/FRAC_POOR
# Per-TAD class mix [hk_const, hk_high, celltype, dev], interpolated by the TAD's transcription
# level. Housekeeping genes are enriched in active compartments but are still a minority even
# there (~35% of active-TAD genes); silent/gene-poor TADs are dominated by developmental +
# off cell-type genes. The activity-weighted blend nets ~20-25% housekeeping genome-wide.
FRAC_ACTIVE = np.array([0.22, 0.13, 0.57, 0.08])   # mix in max-txn TADs; ↑ a weight -> more of that class where active
FRAC_POOR   = np.array([0.10, 0.02, 0.45, 0.43])   # mix in min-txn TADs; ↑ a weight -> more of that class where silent

# ===================== Per-class output stratification ===================
# The deficit in mRNA output is housekeeping-specific (the per-class QC shows hk genes far below the
# ~1/min mRNA median while cell-type/developmental genes are correctly at lincRNA/poised level). Two
# class-conditioned levers, both literature-grounded, lift ONLY housekeeping toward ~1/min:
#   (1) LOADING -- housekeeping promoters are strong / heavily loaded. A per-class multiplier on the
#       loading anchor (renormalized, so it is a RELATIVE shift; cell-type ~ geo-mean, dev below).
#   (2) PREMATURE TERMINATION -- highly-expressed housekeeping genes terminate LESS at the pause
#       (higher productive_fraction); regulated/developmental genes default to high Pol II turnover
#       (Mol Cell 2024 / Genes&Dev 2025 promoter-proximal QC). Per-class k_termination /s below; the
#       engine reads a per-gene pause_term_prob (falls back to the global RNAP_PAUSE_TERM_RATE).
#   (3) 3' TERMINATION DWELL -- active genes terminate FASTER (shorter per-Pol 3' residence; residence
#       is inversely related to transcription rate, Erickson 2018 PNAS), which clears the 3' zone fast
#       and relieves the body back-pressure that jams short, heavily-loaded hk genes. Carried by the
#       per-class UNBIND rate (CLASS_UNBIND_RATE); the readthrough-WINDOW length is left UNIFORM at the
#       Schwalb median (CLASS_TERM_WINDOW_MULT all 1.0) so the two 3' effects don't compound.
CLASS_LOAD_MULT = {"hk_const": 1.8, "hk_high": 2.5, "celltype": 1.0, "dev": 0.7}
CLASS_PAUSE_TERM_RATE = {"hk_const": 0.020, "hk_high": 0.016, "celltype": 0.028, "dev": 0.038}
CLASS_TERM_WINDOW_MULT = {"hk_const": 1.0, "hk_high": 1.0, "celltype": 1.0, "dev": 1.0}
# Per-class gene-length multiplier on the lognormal median. Housekeeping genes are COMPACT (shorter
# introns/exons/UTRs; "human housekeeping genes are compact", Eisenberg & Levanon 2003; Drosophila
# hk introns ~4x shorter than developmental), tissue-specific/developmental genes are LONGER (more
# domains, regulatory introns). Kept moderate so short hk genes do not become unrealistically dense.
CLASS_GENE_LEN_MULT = {"hk_const": 0.7, "hk_high": 0.7, "celltype": 1.15, "dev": 1.6}
# Per-class 3' TERMINATION DWELL: the per-Pol termination/unbind rate /s, ANTI-correlated with gene
# activity. At highly expressed genes paused/terminating Pol II is short-lived with rapid turnover
# (inverse of residence time vs transcription rate; Erickson 2018 PNAS), so housekeeping genes
# terminate FASTER (shorter 3' dwell, faster Pol recycling) and developmental genes SLOWER. Per-gene
# termination_prob below is set from this (falls back to the global RNAP_UNBIND_RATE). The implied
# mean dwell 1/rate stays in the 1-8 min band: hk ~1.5-2 min, cell-type ~2.8, dev ~3.7.
CLASS_UNBIND_RATE = {"hk_const": 0.007, "hk_high": 0.0085, "celltype": 0.006, "dev": 0.0045}


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
        # Decelerated 3' termination crawl: TERM_SPEED sites/s -> per-tick step prob (stride 1;
        # term speed is slow enough that term_speed*tick < 1 for any realistic tick).
        "rnapii_termination_step_prob": min(1.0, TERM_SPEED_KB_PER_SECOND * tick_seconds),
        "rnapii_stride": rnapii_stride,
        "elongation_step_prob": elongation_step_prob,
        "rnap_load_prob": _rate_to_prob(RNAP_LOAD_RATE, tick_seconds),
        "load_prob_max": _rate_to_prob(LOAD_RATE_MAX, tick_seconds),
        "rnap_termination_prob": _rate_to_prob(RNAP_UNBIND_RATE, tick_seconds),
        "rnap_pause_term_prob": _rate_to_prob(RNAP_PAUSE_TERM_RATE, tick_seconds),
        "rnap_block_prob": math.exp(-tick_seconds / RNAP_BYPASS_SECONDS),
        "rnap_term_block_prob": math.exp(-tick_seconds / RNAP_TERM_BYPASS_SECONDS),
        "rnap_pre_initiation_block_prob": math.exp(-tick_seconds / RNAP_PRE_INITIATION_BYPASS_SECONDS),
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


def gene_length(rng, cls=""):
    """Genomic span (TSS->TES), lognormal median 24 kb, sigma 1.3 -> mean ~56 kb,
    ~14% >100 kb. Matches human genomic-span stats (Piovesan 2019: median ~26 kb,
    mean ~67 kb, ~15-16% >100 kb); kept slightly tight so non-overlapping packing
    still reaches ~12 genes/Mb in this gene-dense locus. CLASS_GENE_LEN_MULT scales the
    median per class: housekeeping shorter (compact), developmental longer."""
    med = 24.0 * CLASS_GENE_LEN_MULT.get(cls, 1.0)
    return int(np.clip(rng.lognormal(mean=np.log(med), sigma=1.3), 3, 800))


def term_window(rng, cls=""):
    """3' termination-window length in sites (=kb) DOWNSTREAM of the TES. Lognormal
    median ~3.3 kb with a tail to >10 kb (Schwalb 2016 TT-seq: human TTS window median
    ~3.3 kb past the pA site, up to >10 kb). Per-class CLASS_TERM_WINDOW_MULT scales the median:
    housekeeping terminate TIGHTER (shorter window), developmental looser. Clipped to [1, 20] kb."""
    med = TERM_WINDOW_KB_MEDIAN * CLASS_TERM_WINDOW_MULT.get(cls, 1.0)
    return int(np.clip(rng.lognormal(mean=np.log(med), sigma=TERM_WINDOW_SIGMA), 1, 20))


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
             # pause_offset=1: paused Pol sits 1 site (1 kb) downstream of the TSS
             # (realistic promoter-proximal pause site) so the TSS frees for the next
             # loader -> enables the dense Pol "train" of highly-active loci. With
             # offset 0 a paused Pol blocks its own TSS for the whole ~7 min pause,
             # throttling re-loading to ~1 Pol/gene regardless of load_prob.
             pause_offset=1,
             # Class-specific 3' termination DWELL: faster unbind (shorter dwell) at active
             # housekeeping genes, slower at developmental (Erickson 2018: residence anti-
             # correlated with transcription rate). Falls back to the global RNAP_UNBIND_RATE.
             termination_prob=_round_prob(
                 _rate_to_prob(CLASS_UNBIND_RATE.get(cls, RNAP_UNBIND_RATE),
                               calibration["tick_seconds"]), 4),
             # 3' termination window (sites downstream of TES) the terminating Pol walks
             # before release -> vacates the TES quickly, removing the single-slot 3' jam.
             # Class-specific: housekeeping terminate tighter (shorter window).
             termination_window=term_window(rng, cls))
    g["_cls"] = cls   # transient class tag for the activity post-pass (not emitted; see fmt_gene)
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
        dist = int(np.clip(rng.lognormal(np.log(30), 1.0), 5, 800))  # median ~30 kb (Gasperini 2019 ~24 kb;
        #                                                              Engreitz/ABC 87% <100 kb), heavy tail to ~500 kb.
        #                                                              The old ~100 kb median was the mis-cited PCHi-C value.
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

        # --- boundary-flank promoter: TAD boundaries are MODESTLY enriched (~1.5-2.6x)
        # for housekeeping genes / TSSs (Dixon 2012; stable-boundary studies), NOT
        # saturated. So place at most ONE flank promoter per TAD, only probabilistically
        # (FLANK_PROB), spread over a WIDE window (3-25 kb), and only ~70% housekeeping
        # (boundaries are hk-ENRICHED, not hk-exclusive). The old rule pinned 1-2 hk
        # promoters 3-8 kb inside EVERY boundary -> ~10x enrichment, far too strong.
        # A TSS-bound Pol still reinforces insulation, just at realistic density.
        FLANK_PROB = 0.5
        avail = [s for s in ("L", "R") if (lo > 0 if s == "L" else hi < CHAIN)]
        if avail and n > 2 and rng.random() < FLANK_PROB:
            side = avail[int(rng.integers(len(avail)))]
            off = int(rng.integers(3, 26))      # 3-25 kb into the flank (was 3-8: too tight a peak)
            # Flank-promoter class first (boundaries are hk-ENRICHED, not exclusive), so the gene
            # length can be class-specific (housekeeping compact).
            if rng.random() < 0.70:
                cls = "hk_high" if rng.random() < (0.15 + 0.5 * t) else "hk_const"
            else:
                cls = ORDER[rng.choice(len(ORDER), p=fr)]
            L = min(gene_length(rng, cls), max(3, span // 4))
            if side == "L":
                s = off
                tss_pos, tes_pos, strand = lo + s, lo + s + L, True
            else:
                s = span - off - L
                tss_pos, tes_pos, strand = hi - off, hi - off - L, False
            if EDGE <= s and s + L <= span - EDGE and not occ[s:s + L].any():
                occ[max(0, s - 5):s + L + 5] = True
                g, C = _make_gene(rng, tss_pos, tes_pos, cls, calibration)
                _add_enhancers(rng, g, C, lo, hi, strand)
                genes.append(g); remaining -= 1

        # --- interior genes (classes drawn by TAD activity), largest first so
        # long genes get space (avoids short-gene packing bias) ---
        draws = []
        for _ in range(remaining):
            cls = ORDER[rng.choice(len(ORDER), p=fr)]
            draws.append((cls, min(gene_length(rng, cls), span - 2 * EDGE - 4)))
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
             "elongation_step_prob", "pause_offset", "termination_prob", "termination_window",
             "pause_term_prob", "gene_class"]


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


DEFAULT_BSTR = 0.15  # default_boundary_strength emitted below (near the BSTR_RANGE center); ends forced to 1.0


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
    # max_rnapii is PER CHAIN and sets the recording-buffer width; the total width
    # is rnapii_cap = max_rnapii * num_chains (one frame holds the MAX SIMULTANEOUS
    # Pol, columns reused frame-to-frame, identity via uid -- do NOT scale by
    # num_chains here, rnapii_cap multiplies again). lef.py records only
    # rnapiis[:rnapii_cap] and SILENTLY drops any surplus, which biases-to-zero
    # whole genes and erases the promoter pause when the buffer saturates.
    #
    # Size it to the HARD physical ceiling so it can never truncate: RNAPII obey
    # steric exclusion (<=1 Pol per 1 kb site) and exist between a gene's TSS and TES
    # PLUS its 3' termination window (terminating Pol crawl up to termination_window
    # sites past the TES before release), and genes never overlap on a chain, so the
    # maximum simultaneous Pol per chain is the total (gene-body + termination-window)
    # length in sites. Omitting the window would undercount and silently drop
    # terminating Pol. This is the
    # TIGHTEST value that is guaranteed safe (you cannot have more Pol than gene
    # sites) and it auto-adapts to the gene set (count/length) -- unlike a fixed
    # CHAIN-scaled guess, which was actually a few % UNDER this ceiling. The buffer
    # is -1-padded + lzf-compressed, so on-disk cost tracks the real Pol count, not
    # the cap; only transient in-memory buffers scale with it (lower the metrics
    # --measure-chunk if RAM is tight at very high num_chains).
    max_rnapii = (max(64, sum(abs(g["tes"] - g["tss"]) + 1 + int(g.get("termination_window", 0))
                              for g in genes)) if txn_on else 0)
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
  lifetime_ctcf: {life_ctcf} # Cohesin lifetime when its captured by a CTCF site (boosted from the base lifetime)
  tick_seconds: {calibration["tick_seconds"]:g}
  warmup_steps: {calibration["warmup_steps"]}
  trajectory_length: {calibration["trajectory_length"]}
  chunk_size: 50
  seed: 42
  max_rnapii: {max_rnapii}

  topology_kwargs:
    tads:
{tads_block}
    default_boundary_strength: {DEFAULT_BSTR:g}
    release_prob: 0.0
    include_chromosome_ends: true
    # Paper-calibrated: RNAP is a permeable moving barrier (Banigan), NOT a fast
    # cohesin-eviction state. Keep RNAP-stalled cohesin at the base lifetime -- a
    # SHORT eviction lifetime makes transcription DEPLETE (not accumulate) cohesin
    # at genes, the opposite of the Banigan moving-barrier effect (verified in 1D).
    lifetime_rnapii_stalled: {life_rnapii_stalled}
    rnapii_stride: {calibration["rnapii_stride"]}
    # Decelerated step prob for a Pol crawling the per-gene 3' termination window (Schwalb 2016
    # window median ~3.3 kb; Cortazar 2019 3' rate ~0.7-0.9 kb/min). Vacates the TES quickly so
    # the 3' end stops capping output at 1/dwell (which packed long active gene bodies solid).
    rnapii_termination_step_prob: {round(calibration["rnapii_termination_step_prob"], 6)}
    # Banigan moving barrier (per-encounter probs, tick-independent): elongating RNAP
    # wins head-on and pushes the converging cohesin; co-directional contact mostly
    # only slows cohesin. stall_prob is an intrinsic floor (effective head-on push ~0.72).
    rnapii_stall_prob: {RNAP_STALL_PROB}
    rnapii_push_prob: {RNAP_PUSH_PROB}
    rnapii_headon_push_prob: {RNAP_HEADON_PUSH_PROB}
    rnapii_pause_cohesin_restraint: {RNAP_PAUSE_RESTRAINT}
    rnapii_pause_restraint_window: 1
    # Promoter-proximal premature termination as a per-tick channel competing with
    # productive release (pause_release_prob): single-Pol pause dwell ~0.4-0.5 min
    # (Lysakovskaia 2025), productive fraction ~0.1-0.25 (activity-dependent).
    rnapii_pause_term_prob: {_round_prob(calibration["rnap_pause_term_prob"], 4)}
    # Per-tick cohesin bypass of a Pol II barrier (tbypass ~100 s -> k ~0.01/s, Banigan).
    # pre-initiation Pol II is a LEAKY barrier (cohesin bypasses ~8x faster);
    # paused / elongating / terminating Pol II are strong barriers.
    rnapii_pre_initiation_block_prob: {_round_prob(calibration["rnap_pre_initiation_block_prob"], 3)}
    rnapii_paused_block_prob: {_round_prob(calibration["rnap_block_prob"], 3)}
    rnapii_elongating_block_prob: {_round_prob(calibration["rnap_block_prob"], 3)}
    rnapii_terminating_block_prob: {_round_prob(calibration["rnap_term_block_prob"], 3)}
    ep_contact_tolerance: 1
    replicate_genes_across_chains: true
    targeted_load_prob: 0.0
    loading_window: 1
    target_enhancers: false
    target_tss: false
    weight_loading_by_activity: false
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
    ap.add_argument("--bstr-mult", type=float, default=2.5,
                    help="config3 = config2 with all boundary strengths X times larger "
                         "(2.5x BSTR_RANGE -> ~0.15-0.45, a strong but not deterministic CTCF perturbation)")
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
    ap.add_argument("--lesion-tad-repair-exponent", type=float, default=0.25,
                    help="config4: rate * (L_mean/L_TAD)**beta; larger -> shorter TADs recognise/repair faster")
    ap.add_argument("--lesion-type-b-enabled", action=argparse.BooleanOptionalAction, default=True,
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
    # transcription level per TAD: PRIMARILY driven by inverse TAD size (longer TADs
    # = less transcription / fewer active genes), blended with the density latent and
    # noise so the size->txn trend is strong but not deterministic. The rng.normal is
    # drawn once per TAD (same count as before) so downstream draws are unperturbed.
    _iv0 = list(zip([0, *bounds], [*bounds, CHAIN]))
    _sz0 = np.array([hi - lo for lo, hi in _iv0], dtype=float)
    _inv_size = 1.0 - (_sz0 - _sz0.min()) / (_sz0.max() - _sz0.min() + 1e-9)  # 1 = smallest TAD
    txn = np.clip(
        [TAD_SIZE_TXN_WEIGHT * iv + (1.0 - TAD_SIZE_TXN_WEIGHT) * d + rng.normal(0, TXN_NOISE)
         for iv, d in zip(_inv_size, dens)],
        0.0, 1.0,
    )
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
    # Per-gene loading heterogeneity -> a realistic, heavy-tailed (log-normal / Zipf)
    # ACTIVITY distribution: most genes quiet, a few hotspots, with a silent fraction.
    #   (1) pause-initiation limit (Gressel et al. 2019): load ~ 1/pause = pause_release_prob;
    #   (2) TAD compartmentalisation: shorter TADs hold more-active genes, load ~ (1/size)^exp;
    #   (3) wide log-normal scatter (LOAD_LOGNORM_SIGMA ~1.3) -> 2-3 orders of magnitude spread;
    #   (4) SILENT genes: cell-type-specific / developmental genes are largely OFF in a given
    #       cell (SILENT_PROB per class) -> near-zero load.
    # The EXPRESSED genes' GEOMETRIC mean (the typical active gene) is held at RNAP_LOAD_RATE.
    if genes and LOAD_LOGNORM_SIGMA > 0:
        cls = [g.pop("_cls", "") for g in genes]              # consume the transient class tag
        prel = np.array([g["pause_release_prob"] for g in genes], dtype=float)
        gpos = np.array([min(g["tss"], g["tes"]) for g in genes])
        gsize = np.array([next((hi - lo for lo, hi in intervals if lo <= p < hi), np.nan)
                          for p in gpos], dtype=float)
        gsize[~np.isfinite(gsize)] = np.nanmedian(gsize)
        size_factor = (np.median(gsize) / gsize) ** LOAD_TADSIZE_EXPONENT
        class_mult = np.array([CLASS_LOAD_MULT.get(c, 1.0) for c in cls])
        raw = prel * size_factor * rng.lognormal(0.0, LOAD_LOGNORM_SIGMA, len(genes))
        silent = np.array([rng.random() < SILENT_PROB.get(c, 0.0) for c in cls])
        target = calibration["rnap_load_prob"]
        expressed = ~silent
        geo = np.exp(np.log(raw[expressed]).mean()) if expressed.any() else raw.mean()
        # Class-blind renorm to the geo-mean target, THEN an ABSOLUTE per-class multiplier so lifting
        # housekeeping does NOT sink cell-type/dev (cell-type stays ~target, housekeeping above it).
        loads = raw / geo * target * class_mult
        loads[silent] = LOAD_SILENT_FRAC * target             # silent genes near-off (class-independent)
        # Hard-cap the loading tail at the gene-body density ceiling so genes saturate near the
        # density ceiling instead of piling Pol II up in the body.
        loads = np.clip(loads, 1e-4, calibration["load_prob_max"])
        rank = (np.argsort(np.argsort(loads)) + 0.5) / len(loads)       # activity rank in (0,1)
        # CDK9-like pause-release boost is HOUSEKEEPING-targeted: only hk genes shorten their pause
        # with activity (Gressel 2017). Cell-type/developmental genes keep their longer class pauses
        # (developmental genes are the most stably paused -- Zeitlinger), so they are NOT boosted.
        is_hk = np.array([c in ("hk_const", "hk_high") for c in cls])
        ramp = 1.0 + (PAUSE_ACTIVE_BOOST - 1.0) * np.clip((rank - 0.5) / 0.5, 0.0, 1.0)
        boost = np.where(is_hk, ramp, 1.0)
        prel_cap = _rate_to_prob(1.0 / PAUSE_MIN_S, calibration["tick_seconds"])  # shortest pause
        for g, lp, b, c in zip(genes, loads, boost, cls):
            g["load_prob"] = _round_prob(float(lp), 5)
            g["pause_release_prob"] = _round_prob(min(float(g["pause_release_prob"]) * b, prel_cap), 4)
            # Per-class premature-termination rate -> class-specific productive_fraction (housekeeping
            # terminate less -> more productive; developmental default to high turnover).
            g["pause_term_prob"] = _round_prob(
                _rate_to_prob(CLASS_PAUSE_TERM_RATE.get(c, RNAP_PAUSE_TERM_RATE),
                              calibration["tick_seconds"]), 4)
            g["gene_class"] = c            # emit the regulatory class for per-class QC (engine ignores it)
    else:
        for g in genes:
            c = g.pop("_cls", "")
            g["gene_class"] = c
            g["pause_term_prob"] = _round_prob(
                _rate_to_prob(CLASS_PAUSE_TERM_RATE.get(c, RNAP_PAUSE_TERM_RATE),
                              calibration["tick_seconds"]), 4)
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
