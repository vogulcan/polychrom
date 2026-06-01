#!/usr/bin/env python
"""cooltools-style cis-decay slope plot for polychrom sim contact maps.

Mirrors ggner-3d/notebooks/cis_decay.ipynb method, applied to dense
``contact_map.npy`` outputs instead of balanced .mcool / cooltools expected:

  1. P(s): mean of each diagonal of the contact map.
  2. log-bin P(s) into geometric distance bins (cooltools logbin_expected
     style), pair-count weighted -> 'balanced.avg.smoothed.agg' analogue.
  3. mask first 2 diagonals (notebook drops dist < 2).
  4. normalize agg by sum (notebook norm_type='sum').
  5. slope = np.gradient(log10(agg), log10(dist_bp))  <- the diagnostic panel.

Output: 2-panel figure (slope top, fold-change-vs-base bottom) + CSV.

Note: uses raw counts, not ICE-balanced. In cis the balancing bias is
near-flat in s and does not move the loop/TAD shoulder, which is what the
slope panel reads off.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

MONOMER_BP = 1000          # 1 monomer = 1 kb (separation 200 = 200 kb)
BINS_PER_DECADE = 10       # cooltools logbin default
IGNORE_DIAGS = 2           # notebook: dist < 2 -> NaN

# loop / TAD / compartment guide lines (bp); >chain are clipped by xlim
GENOME_SCALE_GUIDES = [150_000, 500_000, 1_000_000]


def ps_curve(m):
    """Mean of each diagonal s = per-separation contact frequency."""
    N = m.shape[0]
    s = np.arange(1, N)
    p = np.array([np.diagonal(m, k).mean() for k in s], dtype=float)
    return s, p


def logbin(s, p, N):
    """Pair-count-weighted average of P(s) in geometric distance bins."""
    lo, hi = np.log10(s[0]), np.log10(s[-1])
    nbins = max(int(np.ceil((hi - lo) * BINS_PER_DECADE)), 1)
    edges = np.unique(np.round(np.logspace(lo, hi, nbins + 1)).astype(int))
    weight = (N - s).astype(float)        # # of pairs at separation s
    cs, ca = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (s >= a) & (s < b)
        if not mask.any():
            continue
        w = weight[mask]
        cs.append(np.exp(np.average(np.log(s[mask]), weights=w)))  # geo center
        ca.append(np.average(p[mask], weights=w))
    return np.array(cs), np.array(ca)


def prepare(map_path):
    m = np.load(map_path)
    N = m.shape[0]
    s, p = ps_curve(m)
    keep = s >= IGNORE_DIAGS
    s, p = s[keep], p[keep]
    s_bin, agg = logbin(s, p, N)
    dist_bp = s_bin * MONOMER_BP
    agg_norm = agg / agg.sum()
    der = np.gradient(np.log10(agg), np.log10(dist_bp))
    return {"dist_bp": dist_bp, "agg": agg, "agg_normalized": agg_norm, "der": der}


def setup_logx(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs containing contact_map.npy")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--map-name", default="contact_map.npy")
    ap.add_argument("--base", default=None, help="label used as fold-change baseline")
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs]
    labels = args.labels or [r.name for r in runs]
    assert len(labels) == len(runs)
    base = args.base or labels[0]

    data = {}
    for lab, r in zip(labels, runs):
        mp = r / args.map_name
        print(f"[{lab}] {mp}")
        data[lab] = prepare(mp)

    colors = ['#465775', '#A63446', '#F5B841', '#9DBBAE', '#8A6E59']
    cmap = {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, figsize=(8.5, 8.5), sharex=True,
        constrained_layout=True, gridspec_kw={"hspace": 0.08})

    # --- slope panel
    for lab in labels:
        v = data[lab]
        sns.lineplot(ax=ax1, x=v["dist_bp"], y=v["der"], label=lab,
                     color=cmap[lab], linewidth=1.8, errorbar=None)
    setup_logx(ax1)
    ax1.set_ylabel(r"Slope of $\log_{10}(\mathrm{P}(s))$")
    ax1.legend(frameon=False, handlelength=2.2)
    ax1.tick_params(axis="both", which="both", direction="out", length=3)
    sns.despine(ax=ax1)

    # --- fold-change panel vs base
    bx = data[base]["dist_bp"]
    by = data[base]["agg_normalized"]
    for lab in labels:
        if lab == base:
            continue
        fc = data[lab]["agg_normalized"] / by
        sns.lineplot(ax=ax2, x=bx, y=fc, label=f"{lab} / {base}",
                     color=cmap[lab], linewidth=1.8, errorbar=None)
    ax2.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0, zorder=0)
    setup_logx(ax2)
    ax2.set_xlabel("Genomic distance (bp)")
    ax2.set_ylabel("Fold change\n(normalized P(s))")
    ax2.legend(frameon=False, handlelength=2.2)
    ax2.tick_params(axis="both", which="both", direction="out", length=3)
    sns.despine(ax=ax2)

    for ax in (ax1, ax2):
        for pos in GENOME_SCALE_GUIDES:
            if bx.min() <= pos <= bx.max():
                ax.axvline(pos, color="0.4", linestyle="--", linewidth=1.0, zorder=0)
    ax1.set_xlim(bx.min(), bx.max())

    svg = out / "cis_decay_slope.svg"
    png = out / "cis_decay_slope.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=150)
    print(f"wrote {svg}\nwrote {png}")

    # --- CSV
    csv = out / "cis_decay_slope.csv"
    cols = ["label", "dist_bp", "agg", "agg_normalized", "der"]
    with open(csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for lab in labels:
            v = data[lab]
            for i in range(len(v["dist_bp"])):
                f.write(f"{lab},{v['dist_bp'][i]:.1f},{v['agg'][i]:.6e},"
                        f"{v['agg_normalized'][i]:.6e},{v['der'][i]:.4f}\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
