#!/usr/bin/env python
"""Genome-browser-style Pol II ChIP and nascent RNA-seq tracks for the locus.

Builds two per-site signals over the whole locus from a `lef`-stage
``LEFPositions.h5`` and the YAML config, then plots them with TAD boundaries
marked:

* **Pol II ChIP-seq**  -- time-mean occupancy of *all* chromatin-bound Pol II at
  each site (POISED+PAUSED+ELONGATING+TERMINATING+STALLED). This is the ChIP
  observable: everything cross-linked to DNA, so promoter-proximal pausing shows
  up as sharp TSS peaks.
* **Nascent RNA-seq**   -- time-mean occupancy of *engaged* Pol II carrying a
  growing transcript (ELONGATING+STALLED), the GRO/PRO-seq / 4sU observable.

The chain copies are folded onto the single ``chain_length`` locus
(``site % chain_length``) and **averaged over alleles** (divide by
``num_chains``), so the signal is per-allele mean occupancy -- the same
convention as ``nascent_allele`` in ``nascent_rna_abundance.py``.

Usage::

    python scripts/polii_chip_nascent_track.py CONFIG.yaml [RUN_DIR_OR_H5] [OUT.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polychrom.pipelines.loop_extrusion.config import load_config  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POISED, PAUSED, ELONGATING, TERMINATING, STALLED = 0, 1, 2, 3, 4
ENGAGED = (ELONGATING, STALLED)   # carries a nascent transcript


def resolve_h5(arg: str | None) -> Path:
    if arg is None:
        return Path("trajectory/LEFPositions.h5")
    p = Path(arg)
    return p if p.is_file() else p / "LEFPositions.h5"


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling mean (a la ChIP fragment smoothing); w sites wide."""
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def build_tracks(h5_path: Path, locus: int):
    """Return (chip, nascent, T, num_chains) folded onto a single `locus` track.

    Per-allele mean occupancy (averaged over chains):
    chip[i]    = mean number of any present Pol II at site i per tick, per allele
    nascent[i] = mean number of ELONGATING|STALLED Pol II at site i per tick, per allele
    """
    with h5py.File(h5_path, "r") as fh:
        if not fh.attrs.get("rnapii_enabled", False):
            raise ValueError(f"{h5_path}: RNAPII not enabled in this run.")
        pos = fh["rnapii_positions"][:]            # (T, R, 2): [global_site, gene_id]
        states = fh["rnapii_states"][:].astype(int)
        num_chains = int(fh.attrs.get("num_chains", 1))
    T = pos.shape[0]
    site = pos[:, :, 0]
    present = site >= 0
    local = np.where(present, site % locus, -1)    # fold chains onto one locus

    chip_counts = np.zeros(locus, dtype=np.float64)
    nasc_counts = np.zeros(locus, dtype=np.float64)
    np.add.at(chip_counts, local[present], 1.0)
    eng = present & np.isin(states, ENGAGED)
    np.add.at(nasc_counts, local[eng], 1.0)
    # average over alleles: divide by ticks AND number of chain copies
    denom = T * max(1, num_chains)
    return chip_counts / denom, nasc_counts / denom, T, num_chains


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cfg = load_config(argv[1])
    h5_path = resolve_h5(argv[2] if len(argv) > 2 and not argv[2].endswith(".png") else None)
    out = next((a for a in argv[2:] if a.endswith(".png")), "polii_chip_nascent_track.png")

    locus = int(cfg.lef.chain_length)
    tk = cfg.lef.topology_kwargs
    boundaries = list(tk.get("tad_positions", []))
    strengths = tk.get("boundary_strength", {}) or {}

    chip, nascent, T, num_chains = build_tracks(h5_path, locus)
    sm = max(1, locus // 1000)                      # ~10-site fragment smoothing
    chip_s, nasc_s = smooth(chip, sm), smooth(nascent, sm)
    x = np.arange(locus)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    specs = [
        (axes[0], chip_s, "#2c5f8a", "Pol II ChIP-seq",
         "Pol II occupancy\n(per-allele mean / tick)"),
        (axes[1], nasc_s, "#b3322a", "Nascent RNA-seq",
         "engaged Pol II\n(ELONG+STALL, per allele)"),
    ]
    for ax, sig, color, label, ylab in specs:
        ax.fill_between(x, sig, color=color, alpha=0.85, linewidth=0)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_ylim(0, sig.max() * 1.12 if sig.max() > 0 else 1)
        ax.margins(x=0)
        ax.text(0.004, 0.92, label, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", color=color)
        for b in boundaries:
            s = strengths.get(b, strengths.get(int(b)))
            lw = 0.8 + 1.6 * float(s) if s is not None else 1.0
            ax.axvline(b, color="0.35", ls="--", lw=lw, alpha=0.7, zorder=3)

    # boundary labels along the top axis
    for b in boundaries:
        axes[0].text(b, 1.02, str(b), transform=axes[0].get_xaxis_transform(),
                     rotation=90, ha="center", va="bottom", fontsize=7, color="0.3")
    axes[0].text(0.5, 1.13, "dashed = TAD boundary (line width ~ boundary strength)",
                 transform=axes[0].transAxes, ha="center", fontsize=8, color="0.4")

    axes[1].set_xlabel(f"Locus position (site = 1 kb)   |   {locus} sites, "
                       f"averaged over {num_chains} alleles, T={T} ticks",
                       fontsize=10)
    fig.suptitle("Pol II ChIP & nascent RNA-seq across the locus", fontsize=13,
                 y=0.97)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"# wrote {out}")
    print(f"# locus={locus} sites | boundaries={boundaries}")
    print(f"# Pol II ChIP   total per-allele occupancy (sum over sites) = {chip.sum():.2f}")
    print(f"# nascent RNA   total per-allele occupancy (sum over sites) = {nascent.sum():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
