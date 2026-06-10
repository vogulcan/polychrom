#!/usr/bin/env python
"""Comprehensive 1D loop-length report for two loop-extrusion configs.

Covers, per config and as config4/config1 folds:
  * full loop-length distribution (mean/median/std + percentile ladder)
  * canonical qc histogram (edges 0,50,100,150,200,300,500,N) + a fine 50-kb hist
  * topology-aware loop classes (short-abs, local-short, TAD-level, near-full)
  * loop length CONDITIONED on state: within- vs between-TAD, CTCF-anchored
    (>=1 leg on a CTCF site) vs free, and corner-dot loops
  * per-TAD loop-length breakdown (mean span + fill fraction span/TAD-size)
  * lesion state-space (config4)

Writes JSON + a markdown report under <run_root>/analysis/.
CTCF geometry per plugins.topology._apply_convergent_tads (left leg captured at
TAD start, right leg at end-1).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from polychrom.pipelines.loop_extrusion.config import load_config
from polychrom.pipelines.loop_extrusion.qc import loop_length_stats

PCTS = [5, 10, 25, 50, 75, 90, 95, 99]

# (config_yaml, LEFPositions.h5) per label; overridable via argv:
#   python loop_length_report_1d.py LABEL=cfg.yaml:run_dir ...
DEFAULT = {
    "config1": ("configs_test/config1_15k.yaml",
                "runs_test/1d_15k_corner_ctcf/config1/LEFPositions.h5"),
    "config4": ("configs_test/config4_15k.yaml",
                "runs_test/1d_15k_corner_ctcf/config4/LEFPositions.h5"),
}


def topology(cfg):
    lc = cfg.lef
    N = int(lc.chain_length)
    inner = [int(p) for p in lc.topology_kwargs["tad_positions"]]
    edges = np.array([0, *inner, N])
    sizes = np.diff(edges)
    left_sites = set(int(e) for e in edges[:-1])
    right_sites = set(int(e) - 1 for e in edges[1:])
    return dict(N=N, chain=N, edges=edges, sizes=sizes,
                ctcf_left=set(left_sites) - {0},
                ctcf_right=set(right_sites) - {N - 1},
                corner_pairs={int(edges[i]): int(edges[i + 1]) - 1
                              for i in range(len(sizes))})


def _isin(local, sites):
    s = np.array(sorted(sites))
    return np.isin(local, s) if len(s) else np.zeros_like(local, bool)


def dist(sizes):
    s = sizes.ravel()
    d = {"n": int(s.size), "mean": float(s.mean()), "median": float(np.median(s)),
         "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}
    for p in PCTS:
        d[f"p{p}"] = float(np.percentile(s, p))
    return d


def analyze(h5, topo):
    chain, edges, sizes = topo["chain"], topo["edges"], topo["sizes"]
    median_tad = float(np.median(sizes))
    with h5py.File(h5, "r") as f:
        pos = f["positions"][:]
        les = None
        if "lesions" in f:
            les = dict(lz=f["lesions"][:], st=f["lesion_states"][:],
                       ty=f["lesion_types"][:], attrs=dict(f.attrs))
    F, L, _ = pos.shape
    lo = np.minimum(pos[..., 0], pos[..., 1])
    hi = np.maximum(pos[..., 0], pos[..., 1])
    span = (hi - lo).astype(float)
    ll = lo % chain; lr = hi % chain

    def tad_of(x):
        return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(sizes) - 1)
    tl, tr = tad_of(ll), tad_of(lr)
    same = tl == tr
    own = sizes[tl].astype(float)

    # ---- canonical + fine histograms ----
    qc_edges = [e for e in [0, 50, 100, 150, 200, 300, 500] if e < chain] + [chain]
    qc = loop_length_stats(pos, qc_edges)
    fine_edges = list(range(0, 1001, 50)) + [chain]
    fh, _ = np.histogram(span.ravel(), bins=fine_edges)

    # ---- loop classes ----
    short_abs = span < 0.25 * median_tad
    local_short = same & (span < 0.25 * own)
    tad_level = same & (span >= 0.5 * own)
    near_full = same & (span >= 0.75 * own)

    # ---- CTCF / corner anchoring ----
    left_ctcf = _isin(ll, topo["ctcf_left"])
    right_ctcf = _isin(lr, topo["ctcf_right"])
    any_ctcf = left_ctcf | right_ctcf            # >=1 leg parked at internal CTCF
    starts = sorted(topo["corner_pairs"])
    corner = np.zeros((F, L), bool)
    for st in starts:
        corner |= (ll == st) & (lr == topo["corner_pairs"][st])
    end_anchor = (ll == 0) | (lr == (chain - 1))
    corner &= ~end_anchor

    def cond(mask):
        s = span[mask]
        return dict(frac=float(mask.mean()), n=int(mask.sum()),
                    mean=float(s.mean()) if s.size else 0.0,
                    median=float(np.median(s)) if s.size else 0.0)

    # ---- per-TAD (within-TAD loops only) ----
    per_tad = []
    for i in range(len(sizes)):
        m = same & (tl == i)
        s = span[m]
        per_tad.append(dict(tad=i, start=int(edges[i]), end=int(edges[i + 1]),
                            size=int(sizes[i]), n=int(m.sum()),
                            mean_span=float(s.mean()) if s.size else 0.0,
                            fill_frac=float((s / sizes[i]).mean()) if s.size else 0.0))

    out = dict(
        frames=F, lefs=L,
        distribution=dist(span),
        qc_histogram=dict(edges=qc["histogram_edges_kb"],
                          fraction=qc["histogram_fraction"],
                          counts=qc["histogram_counts"],
                          mean=qc["mean"], median=qc["median"],
                          p10=qc["p10"], p90=qc["p90"]),
        fine_histogram=dict(edges=fine_edges,
                            fraction=[float(x) for x in fh / fh.sum()]),
        classes=dict(
            short_abs=cond(short_abs), local_short=cond(local_short),
            tad_level=cond(tad_level), near_full=cond(near_full),
            within_tad=cond(same), between_tad=cond(~same)),
        anchoring=dict(
            ctcf_anchored=cond(any_ctcf), free=cond(~any_ctcf),
            corner=cond(corner),
            ctcf_leg_pct=float((left_ctcf.sum() + right_ctcf.sum()) / (2 * F * L) * 100)),
        per_tad=per_tad,
        median_tad=median_tad,
    )
    if les is not None:
        a = les["attrs"]; lz = les["lz"]; st = les["st"]; ty = les["ty"]
        valid = lz >= 0; Nl = valid.sum()
        pre = (st == 0) & valid; rep = (st == 1) & valid; tA = (ty == 0) & valid
        rnapii = bool(a.get("rnapii_enabled", False))
        stall = rep | (pre & tA) if not rnapii else rep
        out["lesion"] = dict(mean_count_pf=float(valid.sum(1).mean()),
                             pre_frac=float(pre.sum() / Nl), repair_frac=float(rep.sum() / Nl),
                             typeA_frac=float(tA.sum() / Nl),
                             stalling_pf=float(stall.sum(1).mean()),
                             stalling_frac=float(stall.sum() / Nl))
    return out


def main():
    spec = dict(DEFAULT)
    out_dir = ROOT / "runs_test/1d_15k_corner_ctcf/analysis"
    for arg in sys.argv[1:]:
        if arg.startswith("--out="):
            out_dir = Path(arg.split("=", 1)[1]); continue
        label, rhs = arg.split("=", 1)
        cfgp, h5 = rhs.split(":", 1)
        spec[label] = (cfgp, h5)
    res = {}
    for name, (cfgp, h5) in spec.items():
        cfg = load_config(ROOT / cfgp if not Path(cfgp).is_absolute() else cfgp)
        h5p = ROOT / h5 if not Path(h5).is_absolute() else Path(h5)
        res[name] = analyze(h5p, topology(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loop_length_report.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
