"""3' termination-window behaviour for stateful RNAPII (no OpenMM needed).

A terminating Pol II should walk DOWNSTREAM of the TES through its per-gene
termination window (Schwalb 2016) and vacate the TES on its first step, so a
following Pol can complete -- this is the fix for the single-slot 3'-end jam.
``termination_window == 0`` must keep the legacy single-slot TES dwell.
"""
import numpy as np

from polychrom.pipelines.loop_extrusion.plugins.rnapii import (
    FREE,
    RNAPII,
    RNAPII_CELL,
    STATE_ELONGATING,
    STATE_TERMINATING,
    Gene,
    stateful_translocate_rnapii,
)


def _args(n, genes, **overrides):
    args = dict(
        N=n, chain_length=n, num_chains=1, genes=genes,
        rnapii_by_pos={}, cohesin_leg_by_pos={}, current_ep_contacts={},
        rnapii_stride=1, rnapii_termination_step_prob=1.0,
        rnapii_pause_term_prob=0.0, rnapii_pause_cohesin_restraint=1.0,
    )
    args.update(overrides)
    return args


def _place(pos, gene_id, direction, occupied, state=STATE_ELONGATING):
    r = RNAPII(pos=pos, gene_id=gene_id, direction=direction, uid=pos)
    r.attrs["state"] = state
    occupied[pos] = RNAPII_CELL
    return r


def test_window_vacates_tes_then_unloads_at_ultimate_tts():
    # Gene body 0..10 (+ direction), 3' termination window of 5 sites (11..15).
    g = Gene(gene_id=0, tss=0, tes=10, direction=1, load_prob=0.0,
             initiation_prob=1.0, pause_release_prob=1.0, elongation_step_prob=1.0,
             termination_prob=0.0, termination_window=5)
    occupied = np.zeros(20, dtype=np.int8)
    r = _place(10, 0, 1, occupied)              # Pol sitting at the TES
    rnapiis = [r]
    args = _args(20, [g], rnapii_by_pos={10: r})

    # term_step_prob = 1.0 and termination_prob = 0.0 -> deterministic crawl, release only at TTS.
    stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert r.attrs["state"] == STATE_TERMINATING
    assert r.pos == 11                # walked one site into the window
    assert occupied[10] == FREE       # TES VACATED -> next Pol could complete here

    for _ in range(20):               # keep crawling; releases at the ultimate TTS (15)
        if not rnapiis:
            break
        stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert rnapiis == []              # unloaded
    assert occupied[11:16].sum() == 0 # window cleared


def test_zero_window_keeps_single_slot_tes_dwell():
    g = Gene(gene_id=0, tss=0, tes=10, direction=1, load_prob=0.0,
             initiation_prob=1.0, pause_release_prob=1.0, elongation_step_prob=1.0,
             termination_prob=0.0, termination_window=0)
    occupied = np.zeros(20, dtype=np.int8)
    r = _place(10, 0, 1, occupied)
    rnapiis = [r]
    args = _args(20, [g], rnapii_by_pos={10: r})

    for _ in range(10):               # termination_prob 0 -> never releases, dwells AT the TES
        stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert rnapiis == [r]
    assert r.pos == 10                # still parked on the single TES slot
    assert r.attrs["state"] == STATE_TERMINATING
    assert occupied[10] == RNAPII_CELL


def test_reverse_strand_window_walks_downstream():
    # Reverse gene: tss=15, tes=5, direction=-1; window extends to sites 4..0.
    g = Gene(gene_id=0, tss=15, tes=5, direction=-1, load_prob=0.0,
             initiation_prob=1.0, pause_release_prob=1.0, elongation_step_prob=1.0,
             termination_prob=0.0, termination_window=4)
    occupied = np.zeros(20, dtype=np.int8)
    r = _place(5, 0, -1, occupied)
    rnapiis = [r]
    args = _args(20, [g], rnapii_by_pos={5: r})

    stateful_translocate_rnapii(rnapiis, [], occupied, args)
    assert r.pos == 4                 # walked downstream (decreasing coord) into the window
    assert occupied[5] == FREE        # TES vacated
