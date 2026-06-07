#!/usr/bin/env python
"""Measure loop-extrusion asymmetry from a 1D LEF sim (LEFPositions.h5).

Builds a per-site loop-contact map from cohesin leg bridges (summed over frames,
cohesins, chains), then runs:
  (1) strand-BLIND meta-TAD reversal asymmetry  (should be ~0 either way),
  (2) transcription-ORIENTED meta-TSS asymmetry (the loader-bias readout).
Pass one or more run dirs; prints a comparison table.
"""
import sys, re
import numpy as np
import h5py


def load_bridges(h5path):
    with h5py.File(h5path, "r") as f:
        pos = f["positions"][:]            # (frames, n_lef, 2)
        N = int(f.attrs["N"]); nchain = int(f.attrs["num_chains"])
        genes = f["genes"][:] if "genes" in f else None
    L = N // nchain                         # per-chain lattice length
    legs = pos.reshape(-1, 2)               # all (a,b) leg pairs over all frames/lefs
    legs = legs[(legs[:, 0] >= 0) & (legs[:, 1] >= 0)]
    a = np.minimum(legs[:, 0], legs[:, 1]); b = np.maximum(legs[:, 0], legs[:, 1])
    a_local, b_local = a % L, b % L         # fold chains onto a common [0,L) lattice
    same = (a // L) == (b // L)             # keep intra-chain loops only
    return a_local[same], b_local[same], L, genes


def contact_map(a, b, L):
    idx = a.astype(np.int64) * L + b.astype(np.int64)
    flat = np.bincount(idx, minlength=L * L).astype(np.float64)
    M = flat.reshape(L, L)
    M = M + M.T
    return M


def observed_over_expected(M):
    N = M.shape[0]; oe = np.full_like(M, np.nan)
    for s in range(N):
        d = np.diagonal(M, s)
        mu = d.mean()
        if mu <= 0:
            continue
        i = np.arange(N - s)
        oe[i, i + s] = M[i, i + s] / mu
        oe[i + s, i] = oe[i, i + s]
    return oe


def parse_genes_yaml(cfgpath):
    g = []
    for line in open(cfgpath):
        s = line.strip()
        if not s.startswith("- {"):
            continue
        tss = int(re.search(r'tss: (\d+)', s).group(1))
        tes = int(re.search(r'tes: (\d+)', s).group(1))
        req = 'requires_enhancer: true' in s
        g.append((tss, tes, req))
    return g


def oriented_asym(oe, genes, W=150, lo=10, hi=150):
    """Per-gene transcription-oriented asym; + = 5'/upstream enriched, - = 3'/down."""
    N = oe.shape[0]
    out = {}
    for label, sel in [("ALL", genes),
                       ("ENH", [x for x in genes if x[2]]),
                       ("nonENH", [x for x in genes if not x[2]])]:
        vals = []
        for tss, tes, req in sel:
            if tss - W < 0 or tss + W >= N:
                continue
            row = np.nan_to_num(oe[tss, tss - W:tss + W + 1].copy())
            c = W
            if tss > tes:
                row = row[::-1]
            up = row[c - hi:c - lo].sum(); dn = row[c + lo:c + hi].sum()
            if up + dn > 0:
                vals.append((up - dn) / (up + dn))
        vals = np.array(vals)
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
        out[label] = (len(vals), float(vals.mean()), float(se), float(vals.mean() / se) if se else float("nan"))
    return out


if __name__ == "__main__":
    runs = sys.argv[1:]
    print(f"{'run':22s} {'group':7s} {'n':>4s} {'mean':>8s} {'SE':>7s} {'z':>7s}")
    for run in runs:
        a, b, L, _ = load_bridges(f"{run}/LEFPositions.h5")
        M = contact_map(a, b, L)
        oe = observed_over_expected(M)
        # find the config to get genes
        import glob
        cfg = sorted(glob.glob(f"{run}/config1*.yaml") + glob.glob(f"{run}/*.yaml"))
        cfg = cfg[0]
        genes = parse_genes_yaml(cfg)
        res = oriented_asym(oe, genes)
        for grp in ("ALL", "ENH", "nonENH"):
            n, m, se, z = res[grp]
            print(f"{run.split('/')[-1]:22s} {grp:7s} {n:4d} {m:+8.4f} {se:7.4f} {z:+7.2f}")
        print(f"  total loop bridges: {len(a):,}   lattice L={L}")
