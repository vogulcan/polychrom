#!/usr/bin/env python
"""1D comparison of two loop-extrusion configs with topology-aware loop classes
plus precise cohesin-at-CTCF and cohesin-at-corner metrics.

CTCF geometry follows plugins.topology._apply_convergent_tads:
  for each TAD interval [start, end] (boundaries = [0, *tad_positions, N]):
    left  leg captured at  left_site  = start        (inward, left-facing anchor)
    right leg captured at  right_site = end - 1       (inward, right-facing anchor)
  => a TAD "corner-dot" cohesin has left leg at `start` and right leg at `end-1`.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from polychrom.pipelines.loop_extrusion.config import load_config


def topology(cfg):
    lc = cfg.lef
    N = int(lc.chain_length)
    inner = [int(p) for p in lc.topology_kwargs["tad_positions"]]
    edges = np.array([0, *inner, N])                 # per-chain TAD edges
    sizes = np.diff(edges)
    # capture sites within a chain (local coords)
    left_sites = set(int(e) for e in edges[:-1])     # start of each interval
    right_sites = set(int(e) - 1 for e in edges[1:]) # end-1 of each interval
    # internal CTCF (exclude chromosome ends 0 and N-1)
    ctcf_left = set(left_sites) - {0}
    ctcf_right = set(right_sites) - {N - 1}
    return dict(N=N, chain=int(lc.chain_length), nchains=int(lc.num_chains),
                edges=edges, sizes=sizes,
                left_sites=left_sites, right_sites=right_sites,
                ctcf_left=ctcf_left, ctcf_right=ctcf_right,
                # corner-dot map: left start -> matching right (end-1) per interval
                corner_pairs={int(edges[i]): int(edges[i + 1]) - 1
                              for i in range(len(sizes))})


def analyze(h5, topo):
    N, chain = topo["N"], topo["chain"]
    edges, sizes = topo["edges"], topo["sizes"]
    median = float(np.median(sizes))
    with h5py.File(h5, "r") as f:
        pos = f["positions"][:]
        les = None
        if "lesions" in f:
            les = dict(lesions=f["lesions"][:], states=f["lesion_states"][:],
                       types=f["lesion_types"][:], attrs=dict(f.attrs))
    F, L, _ = pos.shape
    l = pos[..., 0]; r = pos[..., 1]
    ll = l % chain; lr = r % chain                   # local coords
    span = (r - l).astype(float)

    def tad_of(x):
        return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(sizes) - 1)
    tl, tr = tad_of(ll), tad_of(lr)
    same = tl == tr
    ncross = tr - tl
    own = sizes[tl]

    short_abs = span < 0.25 * median
    local_short = same & (span < 0.25 * own)
    tad_level = same & (span >= 0.5 * own)
    near_full = same & (span >= 0.75 * own)

    # ---- cohesin-at-CTCF (per leg, EXACT capture-site match) ----
    def inset(local, sites):
        s = np.array(sorted(sites))
        return np.isin(local, s) if len(s) else np.zeros_like(local, bool)
    left_at_ctcf = inset(ll, topo["ctcf_left"])      # left leg parked at internal CTCF
    right_at_ctcf = inset(lr, topo["ctcf_right"])     # right leg parked at internal CTCF
    legs_at_ctcf = left_at_ctcf.sum(1) + right_at_ctcf.sum(1)   # per frame
    n_legs = 2 * L
    # chromosome-end anchors (the two strong N-end barriers)
    left_at_end = ll == 0
    right_at_end = lr == (chain - 1)
    legs_at_end = left_at_end.sum(1) + right_at_end.sum(1)

    # ---- cohesin-at-corner ----
    # strict: left leg at a boundary start AND right leg at that interval's end-1
    starts = np.array(sorted(topo["corner_pairs"].keys()))
    corner_strict = np.zeros((F, L), bool)
    for st in starts:
        en1 = topo["corner_pairs"][st]
        corner_strict |= (ll == st) & (lr == en1)
    # any: both legs each at some inward anchor (may span >1 TAD)
    corner_any = (inset(ll, topo["left_sites"]) & inset(lr, topo["right_sites"]))
    # internal-only corner (exclude chromosome-end anchors)
    corner_ctcf = corner_strict.copy()
    # drop corners that use a chromosome end as one anchor
    end_anchor = (ll == 0) | (lr == (chain - 1))
    corner_ctcf &= ~end_anchor

    def pf(m): return float(m.sum(1).mean())
    def pct(m): return float(m.mean() * 100)

    out = dict(
        frames=F, lefs=L, lefs_per_frame=float(L),
        mean_span=float(span.mean()), median_span=float(np.median(span)),
        p90_span=float(np.percentile(span, 90)),
        within_tad_pct=pct(same), between_tad_pct=pct(~same),
        boundary_cross_pct=pct(ncross >= 1), mean_boundaries=float(ncross.mean()),
        short_abs_pct=pct(short_abs), local_short_pct=pct(local_short),
        tad_level_pct=pct(tad_level), near_full_pct=pct(near_full),
        # --- the two headline metrics ---
        ctcf_legs_per_frame=float(legs_at_ctcf.mean()),
        ctcf_leg_pct=float(legs_at_ctcf.mean() / n_legs * 100),
        end_anchor_legs_per_frame=float(legs_at_end.mean()),
        corner_per_frame=pf(corner_ctcf), corner_pct=pct(corner_ctcf),
        corner_any_per_frame=pf(corner_any), corner_any_pct=pct(corner_any),
    )
    if les is not None:
        a = les["attrs"]; lz = les["lesions"]; st = les["states"]; ty = les["types"]
        valid = lz >= 0
        Nl = valid.sum()
        PRE, REP, A, B = 0, 1, 0, 1
        pre = (st == PRE) & valid; rep = (st == REP) & valid
        tA = (ty == A) & valid
        rnapii = bool(a.get("rnapii_enabled", False))
        stall = rep | (pre & tA) if not rnapii else rep  # PRE+A stalls only w/o RNAPII
        out["lesion"] = dict(
            mean_count_pf=float(valid.sum(1).mean()),
            pre_frac=float(pre.sum() / Nl), repair_frac=float(rep.sum() / Nl),
            typeA_frac=float(tA.sum() / Nl),
            stalling_pf=float(stall.sum(1).mean()), stalling_frac=float(stall.sum() / Nl),
        )
    return out


def main():
    base = ROOT / "runs_test/1d_15k_corner_ctcf"
    cfg1 = load_config(ROOT / "configs_test/config1_15k.yaml")
    cfg4 = load_config(ROOT / "configs_test/config4_15k.yaml")
    t1, t4 = topology(cfg1), topology(cfg4)
    r1 = analyze(base / "config1/LEFPositions.h5", t1)
    r4 = analyze(base / "config4/LEFPositions.h5", t4)
    res = dict(config1=r1, config4=r4)
    fc = {}
    for k, v in r1.items():
        if isinstance(v, (int, float)) and isinstance(r4.get(k), (int, float)) and v:
            fc[k] = round(r4[k] / v, 3)
    res["fold_c4_over_c1"] = fc
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
