#!/usr/bin/env python
"""Per-replicate (per-chain) fold change of config2 vs the baseline mean of
config1 replicates, for three cohesin metrics, drawn as a single swarm plot
(one column per metric).

config1 = transcription ON  (baseline; fold denominator = mean over its chains)
config2 = transcription OFF (numerator; each chain is one swarm point)
Same topology in both, so CTCF sites / gene bodies are identical.

Metrics (all normalized, intensive -> safe to compare across chains and fold):
  * Cohesin at gene bodies   : fraction of cohesin legs inside a gene body
  * Cohesin at CTCF boundaries: fraction of legs parked at a CTCF capture site
                                (left leg at a TAD start, right leg at end-1)
  * Boundary crossing events : boundary crossings per cohesin per frame
                               (summed over all internal boundaries -> "all")

Metric definitions follow scripts/compare4_loops_1d.py topology conventions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from polychrom.pipelines.loop_extrusion.config import load_config

C1_H5 = ROOT / "dummy/1d_20k/config1-20k/LEFPositions.h5"
C2_H5 = ROOT / "dummy/1d_20k/config2-20k/LEFPositions.h5"
C1_CFG = ROOT / "configs_test/config1_20k.yaml"
OUT_PNG = ROOT / "dummy/1d_20k/config2_vs_config1_swarm.png"

METRICS = ["Cohesin at\ngene bodies",
           "Cohesin at\nCTCF boundaries",
           "Boundary crossing\nevents (norm.)"]


def topology(cfg_path):
    cfg = load_config(str(cfg_path))
    lc = cfg.lef
    chain = int(lc.chain_length)
    num_chains = int(getattr(lc, "num_chains", 1))
    inner = sorted(int(p) for p in lc.topology_kwargs["tad_positions"])
    edges = np.array([0, *inner, chain])
    ctcf_left = set(int(e) for e in edges[:-1]) - {0}          # TAD starts
    ctcf_right = set(int(e) - 1 for e in edges[1:]) - {chain - 1}  # TAD ends-1
    boundaries = [int(b) for b in edges[1:-1]]                  # internal CTCF
    return chain, num_chains, ctcf_left, ctcf_right, boundaries


def gene_body_mask(h5_path, chain):
    """Chain-local boolean mask (length `chain`) of monomers inside any gene
    body. Topology is replicated across chains, so one mask serves all."""
    with h5py.File(h5_path, "r") as f:
        g = f["genes"][:]
    lo = np.minimum(g["tss"], g["tes"])
    hi = np.maximum(g["tss"], g["tes"])
    gch = lo // chain
    mask = np.zeros(chain, bool)
    c0 = gch == 0  # chain-0 genes; identical pattern on every chain
    for a, b in zip(lo[c0] % chain, hi[c0] % chain):
        mask[a:b + 1] = True
    return mask


def per_chain_metrics(h5_path, chain, num_chains, ctcf_left, ctcf_right,
                      boundaries, gbody):
    """Return arrays (len num_chains) of the three normalized metrics."""
    with h5py.File(h5_path, "r") as f:
        pos = f["positions"][:]            # (F, L, 2)
    F, L, _ = pos.shape
    lo = np.minimum(pos[..., 0], pos[..., 1]).astype(np.int64)
    hi = np.maximum(pos[..., 0], pos[..., 1]).astype(np.int64)
    cl = lo // chain                       # chain of left leg
    cr = hi // chain                       # chain of right leg
    valid = cl == cr                       # same-chain loop
    ll = lo % chain                        # chain-local left-leg position
    lr = hi % chain                        # chain-local right-leg position

    ctcf_l = np.array(sorted(ctcf_left))
    ctcf_r = np.array(sorted(ctcf_right))
    in_ctcf_l = np.isin(ll, ctcf_l)        # left leg parked at TAD start
    in_ctcf_r = np.isin(lr, ctcf_r)        # right leg parked at TAD end-1
    in_gb_l = gbody[ll]                     # left leg inside a gene body
    in_gb_r = gbody[lr]                     # right leg inside a gene body

    # boundary crossings per loop (count over all internal boundaries)
    bcount = np.zeros((F, L), np.int32)
    for b in boundaries:
        bcount += (valid & (ll < b) & (lr > b)).astype(np.int32)

    gene = np.empty(num_chains)
    ctcf = np.empty(num_chains)
    cross = np.empty(num_chains)
    for c in range(num_chains):
        sel = cl == c                       # cohesin-frames belonging to chain c
        n_cf = sel.sum()                    # cohesin-frames
        n_legs = 2 * n_cf                   # two legs each
        gene[c] = (in_gb_l[sel].sum() + in_gb_r[sel].sum()) / n_legs
        ctcf[c] = (in_ctcf_l[sel].sum() + in_ctcf_r[sel].sum()) / n_legs
        vsel = sel & valid
        cross[c] = bcount[vsel].sum() / n_cf  # crossings per cohesin per frame
    return gene, ctcf, cross


def main():
    chain, num_chains, ctcf_left, ctcf_right, boundaries = topology(C1_CFG)
    gbody = gene_body_mask(C1_H5, chain)  # same topology -> reuse for both

    m1 = per_chain_metrics(C1_H5, chain, num_chains, ctcf_left, ctcf_right,
                           boundaries, gbody)
    m2 = per_chain_metrics(C2_H5, chain, num_chains, ctcf_left, ctcf_right,
                           boundaries, gbody)

    baseline = [arr.mean() for arr in m1]            # mean over config1 chains
    folds = [m2[i] / baseline[i] for i in range(3)]  # per-chain config2 fold

    # ---- report ----
    names = ["gene_body_occ", "ctcf_occ", "crossings_per_cohesin_frame"]
    print(f"{'metric':28s} {'c1 mean':>10s} {'c2 mean':>10s} {'fold mean':>10s}")
    for i, nm in enumerate(names):
        print(f"{nm:28s} {baseline[i]:10.4f} {m2[i].mean():10.4f} "
              f"{folds[i].mean():10.3f}")
    print("\nper-chain config2/baseline folds:")
    for i, nm in enumerate(names):
        print(f"  {nm:28s} " + " ".join(f"{v:.3f}" for v in folds[i]))

    # ---- swarm plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(0)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, num_chains))
    for i, fold in enumerate(folds):
        jitter = rng.uniform(-0.12, 0.12, size=fold.size)
        ax.scatter(np.full(fold.size, i) + jitter, fold, s=90,
                   c=colors, edgecolor="k", linewidth=0.6, zorder=3)
        ax.hlines(fold.mean(), i - 0.25, i + 0.25, color="k", lw=2, zorder=2)
    ax.axhline(1.0, color="grey", ls="--", lw=1,
               label="config1 baseline (=1)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("config2 / config1 fold change\n(per replicate chain)")
    ax.set_title("config2 (transcription OFF) vs config1 (transcription ON)\n"
                 "per-replicate fold change vs baseline mean")
    ax.legend(frameon=False, loc="best")
    # per-chain color legend
    handles = [plt.Line2D([0], [0], marker="o", ls="", mec="k",
                          mfc=colors[c], ms=8, label=f"chain {c}")
               for c in range(num_chains)]
    ax.legend(handles=handles + [plt.Line2D([0], [0], color="grey", ls="--",
              label="baseline =1")], frameon=False, fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nsaved {OUT_PNG}")


if __name__ == "__main__":
    main()
