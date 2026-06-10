#!/usr/bin/env python
"""Run 1D (lef-stage) sims for N configs into one output folder and write a
loop-length comparison report. The FIRST config is the baseline; every metric
is folded against it.

Usage
-----
    micromamba run -n polychrom python scripts/compare4_loops_1d.py \
        OUT_DIR  config1.yaml config2.yaml config3.yaml config4.yaml \
        [--labels c1,c2,c3,c4] [--bin 25] [--max 1000]
        [--block-size 500] [--bootstrap 1000] [--smooth-bins 5]
        [--force] [--quiet]

What it does
------------
1. For each config, uses OUT_DIR/<label>/LEFPositions.h5 if it already exists;
   otherwise runs the lef stage there. Use --force to rerun and overwrite.
2. Computes, per config:
     * loop-length distribution (mean/median/std + percentile ladder), in kb
       (1 monomer = 1 kb in this pipeline);
     * loop-length frequency in coarse, stable bands (headline) plus a fine,
       fixed-width grid (--bin / --max) kept as a diagnostic only;
     * topology-aware loop classes (short-abs, local-short, TAD-level, near-full,
       within/between-TAD);
     * CTCF / corner anchoring (inward-facing convergent geometry: left leg
       captured at a TAD start, right leg at end-1);
     * block-bootstrap confidence intervals for phenotype folds;
     * lesion state-space, when the run has lesion datasets.
3. Writes into OUT_DIR/report/ :
     report.json                 -- all metrics, machine-readable
     report.md                   -- formatted tables with folds vs baseline
     phenotype_summary.tsv       -- headline config4-vs-baseline evaluator
     length_band_summary.tsv     -- stable coarse loop-length bands
     loop_length_frequency.tsv   -- fine raw/smoothed per-bin diagnostic table
"""
from __future__ import annotations
import sys, json, argparse, csv
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from polychrom.pipelines.loop_extrusion.config import load_config
from polychrom.pipelines.loop_extrusion import lef as lef_stage

PCTS = [5, 10, 25, 50, 75, 90, 95, 99]
STABLE_BAND_EDGES = [0, 50, 100, 200, 500, 1000]


def _band_edges(chain):
    edges = [e for e in STABLE_BAND_EDGES if e < chain]
    if not edges or edges[0] != 0:
        edges.insert(0, 0)
    if edges[-1] != chain:
        edges.append(chain)
    return edges


def _band_label(lo, hi, chain):
    return f">={lo}" if hi == chain and lo != 0 else f"{lo}-{hi}"


def _smooth_freq(freq, window):
    """Centered moving-average smoothing for diagnostic fine-bin plots/tables."""
    arr = np.asarray(freq, dtype=float)
    window = int(window)
    if window <= 1 or arr.size <= 2:
        return arr.tolist()
    # Do not smooth the final overflow bin with the finite-width bins.
    body = arr[:-1]
    overflow = arr[-1:]
    if body.size < 2:
        return arr.tolist()
    window = min(window, body.size)
    kernel = np.ones(window, dtype=float) / float(window)
    pad_l = window // 2
    pad_r = window - 1 - pad_l
    padded = np.pad(body, (pad_l, pad_r), mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    return np.concatenate([smooth, overflow]).tolist()


def _log_histogram(flat, chain, n_bins=80):
    """Log-spaced all-distance histogram for log-log plotting."""
    arr = np.asarray(flat, dtype=float)
    total = int(arr.size)
    positive = arr[np.isfinite(arr) & (arr > 0)]
    upper = max(2.0, float(chain), float(positive.max()) if positive.size else 2.0)
    edges = np.unique(np.geomspace(1.0, upper, int(n_bins) + 1))
    if edges.size < 2:
        edges = np.array([1.0, upper], dtype=float)
    counts, _ = np.histogram(positive, bins=edges)
    widths = np.diff(edges)
    denom = max(total, 1)
    freq = counts.astype(float) / denom
    density = freq / widths
    centers = np.sqrt(edges[:-1] * edges[1:])
    return dict(
        edges=[float(x) for x in edges],
        centers=[float(x) for x in centers],
        frequency=[float(x) for x in freq],
        density=[float(x) for x in density],
        counts=[int(x) for x in counts],
        total_count=total,
        positive_count=int(positive.size),
        zero_count=int(total - positive.size),
    )


def _block_slices(n_frames, block_size):
    block_size = max(1, int(block_size))
    for start in range(0, n_frames, block_size):
        end = min(n_frames, start + block_size)
        if end > start:
            yield start, end


def _ci_from_samples(samples):
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(low=float("nan"), high=float("nan"))
    return dict(low=float(np.percentile(arr, 2.5)),
                high=float(np.percentile(arr, 97.5)))


# --------------------------------------------------------------------------- #
# topology + per-config analysis
# --------------------------------------------------------------------------- #
def topology(cfg):
    lc = cfg.lef
    chain = int(lc.chain_length)
    num_chains = int(getattr(lc, "num_chains", 1))
    inner = [int(p) for p in lc.topology_kwargs["tad_positions"]]
    edges = np.array([0, *inner, chain])
    sizes = np.diff(edges)
    left_sites = {int(e) for e in edges[:-1]}
    right_sites = {int(e) - 1 for e in edges[1:]}
    return dict(N=chain * num_chains, chain=chain, num_chains=num_chains,
                tick_seconds=float(getattr(lc, "tick_seconds", 1.0)),
                edges=edges, sizes=sizes,
                ctcf_left=left_sites - {0}, ctcf_right=right_sites - {chain - 1},
                corner_pairs={int(edges[i]): int(edges[i + 1]) - 1
                              for i in range(len(sizes))})


def _isin(a, sites):
    s = np.array(sorted(sites))
    return np.isin(a, s) if s.size else np.zeros_like(a, bool)


def _near(a, sites, slack=1):
    """Vectorized membership in a +/- slack window around a small site set."""
    out = np.zeros_like(a, bool)
    for s in sites:
        out |= np.abs(a - int(s)) <= int(slack)
    return out


def _dist(s):
    if not s.size:
        return dict(n=0, mean=0.0, median=0.0, std=0.0, min=0.0, max=0.0,
                    **{f"p{p}": 0.0 for p in PCTS})
    d = dict(n=int(s.size), mean=float(s.mean()), median=float(np.median(s)),
             std=float(s.std()), min=float(s.min()), max=float(s.max()))
    for p in PCTS:
        d[f"p{p}"] = float(np.percentile(s, p))
    return d


def analyze(h5, topo, bin_kb, max_kb, cfg=None, block_size=500, smooth_bins=5):
    chain, edges, sizes = topo["chain"], topo["edges"], topo["sizes"]
    median_tad = float(np.median(sizes))
    with h5py.File(h5, "r") as f:
        pos = f["positions"][:]
        rnap_data = None
        if "rnapii_positions" in f and "rnapii_states" in f:
            rnap_data = dict(
                positions=f["rnapii_positions"][:],
                states=f["rnapii_states"][:],
                ids=f["rnapii_ids"][:] if "rnapii_ids" in f else None,
            )
        les = None
        if "lesions" in f:
            les = dict(lz=f["lesions"][:], st=f["lesion_states"][:],
                       ty=f["lesion_types"][:], attrs=dict(f.attrs))
    F, L, _ = pos.shape
    lo = np.minimum(pos[..., 0], pos[..., 1]).astype(int)
    hi = np.maximum(pos[..., 0], pos[..., 1]).astype(int)
    span = (hi - lo).astype(float)
    ll, lr = lo % chain, hi % chain
    same_chain = (lo // chain) == (hi // chain)
    valid = same_chain & (ll <= lr)

    def tad_of(x):
        return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(sizes) - 1)
    tl, tr = tad_of(ll), tad_of(lr)
    same = valid & (tl == tr)
    between = valid & ~same
    own = sizes[tl].astype(float)

    flat = span[valid].ravel()
    qc_edges = [e for e in [0, 50, 100, 150, 200, 300, 500] if e < chain] + [chain]
    qh, _ = np.histogram(flat, bins=qc_edges)
    qfreq = (qh / max(qh.sum(), 1)).tolist()
    fine_cap = min(int(max_kb), chain)
    fine_edges = list(range(0, fine_cap + 1, int(bin_kb)))
    if not fine_edges or fine_edges[-1] != fine_cap:
        fine_edges.append(fine_cap)
    if fine_edges[-1] < chain:
        fine_edges.append(chain)
    fine_edges = sorted(set(fine_edges))
    fh, _ = np.histogram(flat, bins=fine_edges)
    fine_freq = (fh / max(fh.sum(), 1)).tolist()
    fine_freq_smooth = _smooth_freq(fine_freq, smooth_bins)
    log_hist = _log_histogram(flat, chain)

    stable_edges = _band_edges(chain)
    bh, _ = np.histogram(flat, bins=stable_edges)
    band_labels = [
        _band_label(int(stable_edges[i]), int(stable_edges[i + 1]), chain)
        for i in range(len(stable_edges) - 1)
    ]
    band_freq = bh / max(bh.sum(), 1)
    band_per_frame = bh / max(F, 1)

    short_abs = valid & (span < 0.25 * median_tad)
    local_short = same & (span < 0.25 * own)
    tad_level = same & (span >= 0.5 * own)
    near_full = same & (span >= 0.75 * own)

    left_ctcf = _isin(ll, topo["ctcf_left"])
    right_ctcf = _isin(lr, topo["ctcf_right"])
    left_ctcf_slack1 = _near(ll, topo["ctcf_left"], slack=1)
    right_ctcf_slack1 = _near(lr, topo["ctcf_right"], slack=1)
    any_ctcf = valid & (left_ctcf | right_ctcf)
    any_ctcf_slack1 = valid & (left_ctcf_slack1 | right_ctcf_slack1)
    corner_strict = np.zeros((F, L), bool)
    corner_slack1 = np.zeros((F, L), bool)
    for st, en1 in topo["corner_pairs"].items():
        if st == 0 or en1 == chain - 1:
            continue
        corner_strict |= valid & (ll == st) & (lr == en1)
        corner_slack1 |= valid & (np.abs(ll - st) <= 1) & (np.abs(lr - en1) <= 1)

    boundary_any = np.zeros((F, L), bool)
    boundary_count = np.zeros((F, L), dtype=np.int16)
    for b in edges[1:-1]:
        cross = valid & (ll < b) & (lr > b)
        boundary_any |= cross
        boundary_count += cross.astype(np.int16)

    def cond(mask):
        s = span[mask]
        return dict(frac=float(mask.mean()), n=int(mask.sum()),
                    per_frame=float(mask.sum() / max(F, 1)),
                    mean=float(s.mean()) if s.size else 0.0,
                    median=float(np.median(s)) if s.size else 0.0)

    boundary = cond(boundary_any)
    boundary["crossings_per_frame"] = float(boundary_count.sum() / max(F, 1))
    boundary["mean_boundaries_per_valid_loop"] = (
        float(boundary_count[valid].mean()) if valid.any() else 0.0
    )
    corner_strict_c = cond(corner_strict)
    corner_slack1_c = cond(corner_slack1)

    def block_per_frame(mask):
        vals = []
        for start, end in _block_slices(F, block_size):
            vals.append(float(mask[start:end].sum() / max(end - start, 1)))
        return vals

    def block_mean_loop():
        vals = []
        for start, end in _block_slices(F, block_size):
            m = valid[start:end]
            vals.append(float(span[start:end][m].mean()) if m.any() else 0.0)
        return vals

    corner_slack1_blocks = block_per_frame(corner_slack1)
    phenotype_blocks = dict(
        short_abs_per_frame=block_per_frame(short_abs),
        local_short_per_frame=block_per_frame(local_short),
        tad_level_per_frame=block_per_frame(tad_level),
        near_full_per_frame=block_per_frame(near_full),
        within_tad_per_frame=block_per_frame(same),
        between_tad_per_frame=block_per_frame(between),
        boundary_crossing_per_frame=block_per_frame(boundary_any),
        corner_slack1_per_frame=corner_slack1_blocks,
        corner_slack1_pct=[100.0 * x / max(L, 1) for x in corner_slack1_blocks],
        mean_loop_kb=block_mean_loop(),
    )

    out = dict(
        frames=F, lefs=L, median_tad=median_tad,
        tad_sizes=[int(x) for x in sizes],
        block_size=int(block_size),
        sanity=dict(valid_loop_frac=float(valid.mean()),
                    invalid_cross_chain_frac=float((~same_chain).mean())),
        distribution=_dist(flat),
        qc_histogram=dict(edges=qc_edges,
                          frequency=qfreq,
                          counts=[int(x) for x in qh]),
        length_bands=dict(edges=stable_edges,
                          labels=band_labels,
                          frequency=[float(x) for x in band_freq],
                          per_frame=[float(x) for x in band_per_frame],
                          counts=[int(x) for x in bh]),
        fine_histogram=dict(edges=fine_edges, frequency=fine_freq,
                            smoothed_frequency=fine_freq_smooth,
                            smooth_bins=int(smooth_bins),
                            note="Fine bins are diagnostic; use length_bands and phenotype for decisions."),
        log_histogram=log_hist,
        classes=dict(short_abs=cond(short_abs), local_short=cond(local_short),
                     tad_level=cond(tad_level), near_full=cond(near_full),
                     within_tad=cond(same), between_tad=cond(between)),
        boundary_crossing=boundary,
        anchoring=dict(ctcf_anchored=cond(any_ctcf),
                       ctcf_anchored_slack1=cond(any_ctcf_slack1),
                       free=cond(valid & ~any_ctcf_slack1),
                       corner=corner_strict_c,
                       corner_strict=corner_strict_c,
                       corner_slack1=corner_slack1_c,
                       ctcf_leg_pct=float((left_ctcf.sum() + right_ctcf.sum())
                                          / (2 * F * L) * 100),
                       ctcf_leg_slack1_pct=float(
                           (left_ctcf_slack1.sum() + right_ctcf_slack1.sum())
                           / (2 * F * L) * 100)),
    )
    out["phenotype"] = dict(
        short_abs_per_frame=out["classes"]["short_abs"]["per_frame"],
        local_short_per_frame=out["classes"]["local_short"]["per_frame"],
        tad_level_per_frame=out["classes"]["tad_level"]["per_frame"],
        near_full_per_frame=out["classes"]["near_full"]["per_frame"],
        within_tad_per_frame=out["classes"]["within_tad"]["per_frame"],
        between_tad_per_frame=out["classes"]["between_tad"]["per_frame"],
        boundary_crossing_per_frame=out["boundary_crossing"]["per_frame"],
        boundary_crossings_per_frame=out["boundary_crossing"]["crossings_per_frame"],
        corner_strict_per_frame=out["anchoring"]["corner_strict"]["per_frame"],
        corner_strict_pct=out["anchoring"]["corner_strict"]["frac"] * 100.0,
        corner_slack1_per_frame=out["anchoring"]["corner_slack1"]["per_frame"],
        corner_slack1_pct=out["anchoring"]["corner_slack1"]["frac"] * 100.0,
        mean_loop_kb=out["distribution"]["mean"],
    )
    out["phenotype_blocks"] = {
        key: {
            "block_size": int(block_size),
            "n_blocks": int(len(vals)),
            "values": [float(v) for v in vals],
            "mean": float(np.mean(vals)) if vals else 0.0,
            "ci95": _ci_from_samples(vals),
        }
        for key, vals in phenotype_blocks.items()
    }
    if les is not None:
        a, lz, st, ty = les["attrs"], les["lz"], les["st"], les["ty"]
        valid = lz >= 0
        Nl = valid.sum()
        pre = (st == 0) & valid; rep = (st == 1) & valid; tA = (ty == 0) & valid
        rnapii_enabled = bool(a.get("rnapii_enabled", False))
        stall = rep | (pre & tA) if not rnapii_enabled else rep
        tk = cfg.lef.topology_kwargs if cfg is not None else {}
        out["lesion"] = dict(mean_count_pf=float(valid.sum(1).mean()),
                             pre_frac=float(pre.sum() / max(Nl, 1)),
                             repair_frac=float(rep.sum() / max(Nl, 1)),
                             typeA_frac=float(tA.sum() / max(Nl, 1)),
                             stalling_pf=float(stall.sum(1).mean()),
                             stalling_frac=float(stall.sum() / max(Nl, 1)),
                             spacing=float(tk.get("lesion_spacing", float("nan"))),
                             type_a_prob=float(tk.get("lesion_type_a_prob", float("nan"))),
                             prerecognition_ticks=float(tk.get("lesion_prerecognition_ticks", float("nan"))),
                             repair_ticks=float(tk.get("lesion_repair_ticks", float("nan"))),
                             block_prob=float(tk.get("lesion_block_prob",
                                                     a.get("lesion_block_prob", float("nan")))))
    if rnap_data is not None:
        rp, rs, ri = rnap_data["positions"], rnap_data["states"], rnap_data["ids"]
        present = rp[..., 0] >= 0
        states = rs.astype(int)
        total = int(present.sum())
        state_names = {0: "poised", 1: "paused", 2: "elongating",
                       3: "terminating", 4: "stalled"}
        state_mix = {
            name: float(100.0 * ((states == code) & present).sum() / max(total, 1))
            for code, name in state_names.items()
        }
        adv = 0
        eticks = 0
        sites = rp[..., 0]
        if ri is not None:
            for k in range(sites.shape[1]):
                for t in range(1, sites.shape[0]):
                    if states[t, k] != 2 or states[t - 1, k] != 2:
                        continue
                    if ri[t, k] < 0 or ri[t, k] != ri[t - 1, k]:
                        continue
                    adv += abs(int(sites[t, k]) - int(sites[t - 1, k]))
                    eticks += 1
        sites_per_tick = adv / max(eticks, 1)
        out["rnapii"] = dict(
            mean_count_all_frames=float(present.sum(1).mean()),
            max_simultaneous=int(present.sum(1).max()),
            capacity=int(present.shape[1]),
            capacity_hit_frames_pct=float(100.0 * (present.sum(1) == present.shape[1]).mean()),
            state_mix_pct=state_mix,
            elongation_speed_sites_per_tick=float(sites_per_tick),
            elongation_speed_kb_per_min=float(sites_per_tick / max(topo.get("tick_seconds", 1.0), 1e-12) * 60.0),
        )
    return out


# --------------------------------------------------------------------------- #
# sim driver + report writers
# --------------------------------------------------------------------------- #
class H5BusyError(RuntimeError):
    """Raised when an existing H5 file appears to be open for writing."""


def _h5_summary(h5):
    try:
        with h5py.File(h5, "r") as fh:
            if "positions" not in fh:
                raise ValueError("missing required dataset 'positions'")
            pos = fh["positions"]
            if len(pos.shape) != 3 or pos.shape[-1] != 2:
                raise ValueError(f"unexpected positions shape {pos.shape}")
            rnapii = "rnapii_positions" in fh and "rnapii_states" in fh
            lesions = (
                "lesions" in fh
                and "lesion_types" in fh
                and "lesion_states" in fh
            )
            info = {
                "frames": int(pos.shape[0]),
                "lefs": int(pos.shape[1]),
                "size_mb": h5.stat().st_size / (1024.0 * 1024.0),
                "rnapii": bool(rnapii),
                "lesions": bool(lesions),
            }
            if rnapii:
                info["rnapii_capacity"] = int(fh["rnapii_positions"].shape[1])
            if lesions:
                info["lesion_capacity"] = int(fh["lesions"].shape[1])
            return info
    except BlockingIOError as exc:
        raise H5BusyError(str(exc)) from exc
    except OSError as exc:
        raise ValueError(str(exc)) from exc


def _format_h5_summary(info):
    parts = [
        f"frames={info['frames']}",
        f"lefs/frame={info['lefs']}",
        f"size={info['size_mb']:.1f} MB",
        f"rnapii={'yes' if info['rnapii'] else 'no'}",
        f"lesions={'yes' if info['lesions'] else 'no'}",
    ]
    if "rnapii_capacity" in info:
        parts.append(f"rnapii_capacity={info['rnapii_capacity']}")
    if "lesion_capacity" in info:
        parts.append(f"lesion_capacity={info['lesion_capacity']}")
    return ", ".join(parts)


def run_sim(cfg_path, cfg, run_dir, force=False, quiet=False):
    h5 = run_dir / "LEFPositions.h5"
    if h5.exists() and not force:
        try:
            info = _h5_summary(h5)
        except H5BusyError as exc:
            raise RuntimeError(
                f"Existing H5 is locked/in progress: {h5}\n"
                f"Reason: {exc}\n"
                "Wait for the writing process to finish, then rerun the comparison."
            ) from exc
        except ValueError as exc:
            raise RuntimeError(
                f"Existing H5 is not usable: {h5}\n"
                f"Reason: {exc}\n"
                "Remove it or rerun with --force to overwrite it once no process is writing it."
            ) from exc
        if not quiet:
            print(f"[reuse]  {cfg_path.name} -> {h5}")
            print(f"         {_format_h5_summary(info)}")
        return h5
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.lef.output_path = str(h5)
    if not quiet:
        if h5.exists() and force:
            print(f"[force]  rerunning {cfg_path.name}; overwriting {h5}")
        else:
            print(f"[run]    {cfg_path.name} -> {h5}")
    lef_stage.run(cfg.lef)
    if not quiet:
        print(f"[done]   {h5}")
        print(f"         {_format_h5_summary(_h5_summary(h5))}")
    return h5


def _fold(x, base):
    return f"{x / base:.2f}" if base else "-"


def plot_histogram(res, labels, base, out):
    """Overlaid loop-length frequency curves (density = fraction per kb), one per
    config. Top: smoothed fine-bin close-range view. Middle: all-distance
    log-log view with log-spaced bins. Bottom: all-distance fold change versus
    baseline with log-x and linear-y axes. Saved as loop_length_histogram.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_lin, ax_log, ax_fold) = plt.subplots(3, 1, figsize=(9, 11), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    base_log_centers = np.array(res[base]["log_histogram"]["centers"], dtype=float)
    base_log_density = np.array(res[base]["log_histogram"]["density"], dtype=float)
    fold_values = []
    for i, c in enumerate(labels):
        color = colors[i % 10]
        lw = 2.5 if c == base else 1.8
        edges = np.array(res[c]["fine_histogram"]["edges"], dtype=float)
        freq = np.array(res[c]["fine_histogram"]["smoothed_frequency"], dtype=float)
        # drop the wide overflow bin (last) so density isn't distorted; note its mass
        widths = np.diff(edges)[:-1]
        centers = (edges[:-2] + edges[1:-1]) / 2.0
        density = freq[:-1] / widths            # fraction per kb
        overflow = freq[-1] * 100
        lab = f"{c}" + (" (baseline)" if c == base else "")
        ax_lin.step(
            centers, density, where="mid",
            label=lab + f"  [>{int(edges[-2])}kb: {overflow:.1f}%]",
            lw=lw, color=color,
        )

        log_centers = np.array(res[c]["log_histogram"]["centers"], dtype=float)
        log_density = np.array(res[c]["log_histogram"]["density"], dtype=float)
        keep = (log_centers > 0) & (log_density > 0)
        if keep.any():
            ax_log.plot(
                log_centers[keep], log_density[keep],
                drawstyle="steps-mid", label=lab, lw=lw, color=color,
            )
        if c != base and log_density.shape == base_log_density.shape:
            ratio = np.full_like(log_density, np.nan, dtype=float)
            valid_ratio = (
                (base_log_centers > 0)
                & (base_log_density > 0)
                & (log_density > 0)
            )
            ratio[valid_ratio] = log_density[valid_ratio] / base_log_density[valid_ratio]
            fold_values.extend(ratio[np.isfinite(ratio) & (ratio > 0)].tolist())
            if valid_ratio.any():
                ax_fold.plot(
                    base_log_centers, ratio,
                    drawstyle="steps-mid", label=f"{c} / {base}",
                    lw=lw, color=color,
                )
    med_tad = res[base]["median_tad"]
    for ax in (ax_lin, ax_log, ax_fold):
        ax.axvline(0.25 * med_tad, ls="--", c="grey", alpha=0.6, lw=1)
        ax.axvline(med_tad, ls=":", c="grey", alpha=0.6, lw=1)
    for ax in (ax_lin, ax_log):
        ax.set_ylabel("frequency (fraction of loops per kb)")
        ax.legend(fontsize=8, frameon=False)
    ax_lin.set_title(f"Smoothed diagnostic loop-length frequency — baseline {base}\n"
                     f"(dashed = 0.25×median TAD {0.25*med_tad:.0f}kb, dotted = median TAD {med_tad:.0f}kb)")
    ax_lin.set_xlim(0, res[base]["fine_histogram"]["edges"][-2])
    ax_lin.set_xlabel("loop length (kb)")
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_xlim(1, res[base]["log_histogram"]["edges"][-1])
    ax_log.set_xlabel("loop length (kb)")
    ax_log.set_title("All-distance loop-length frequency, log-log (log-spaced bins)")
    ax_fold.axhline(1.0, ls="-", c="black", alpha=0.45, lw=1)
    ax_fold.set_xscale("log")
    ax_fold.set_xlim(1, res[base]["log_histogram"]["edges"][-1])
    if fold_values:
        vals = np.asarray(fold_values, dtype=float)
        ymin = min(float(vals.min()), 1.0)
        ymax = max(float(vals.max()), 1.0)
        pad = max(0.05, 0.08 * (ymax - ymin))
        ax_fold.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax_fold.set_xlabel("loop length (kb)")
    ax_fold.set_ylabel(f"fold change vs {base}")
    ax_fold.set_title(f"All-distance fold change versus {base}, log-x")
    ax_fold.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    png = out / "loop_length_histogram.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


PHENOTYPE_METRICS = [
    ("short_abs_per_frame", "UP"),
    ("local_short_per_frame", "UP"),
    ("tad_level_per_frame", "UP_OR_SAME"),
    ("near_full_per_frame", "UP_OR_SAME"),
    ("within_tad_per_frame", "UP_OR_SAME"),
    ("between_tad_per_frame", "DOWN"),
    ("boundary_crossing_per_frame", "DOWN"),
    ("corner_slack1_per_frame", "UP"),
    ("corner_slack1_pct", "UP"),
    ("mean_loop_kb", "INFO"),
]


def _bootstrap_fold_ci(comp_vals, base_vals, n_bootstrap, rng):
    comp = np.asarray(comp_vals, dtype=float)
    base = np.asarray(base_vals, dtype=float)
    comp = comp[np.isfinite(comp)]
    base = base[np.isfinite(base)]
    if n_bootstrap <= 0 or comp.size == 0 or base.size == 0:
        return dict(low=float("nan"), high=float("nan"))
    samples = []
    for _ in range(int(n_bootstrap)):
        c = comp[rng.integers(0, comp.size, size=comp.size)].mean()
        b = base[rng.integers(0, base.size, size=base.size)].mean()
        samples.append(c / b if b else float("nan"))
    return _ci_from_samples(samples)


def _add_phenotype_folds(res, labels, base, n_bootstrap=1000, seed=12345):
    """Attach fold-vs-baseline and desired-phenotype pass/fail to report JSON."""
    b = res[base]["phenotype"]
    rng = np.random.default_rng(int(seed))
    for c in labels:
        p = res[c]["phenotype"]
        folds = {}
        fold_ci95 = {}
        for key, _direction in PHENOTYPE_METRICS:
            base_v = b.get(key, 0.0)
            folds[key] = float(p.get(key, 0.0) / base_v) if base_v else float("nan")
            base_blocks = res[base].get("phenotype_blocks", {}).get(key, {}).get("values")
            comp_blocks = res[c].get("phenotype_blocks", {}).get(key, {}).get("values")
            if c == base:
                fold_ci95[key] = dict(low=1.0, high=1.0)
            elif base_blocks is not None and comp_blocks is not None:
                fold_ci95[key] = _bootstrap_fold_ci(comp_blocks, base_blocks, n_bootstrap, rng)
            else:
                fold_ci95[key] = dict(low=float("nan"), high=float("nan"))
        p["fold_vs_baseline"] = folds
        p["fold_vs_baseline_ci95"] = fold_ci95
        if c == base:
            p["desired_vs_baseline"] = {"pass": None, "checks": {}}
            continue
        checks = {
            "short_abs_up": folds["short_abs_per_frame"] > 1.0,
            "local_short_up": folds["local_short_per_frame"] > 1.0,
            "tad_level_preserved_or_up": folds["tad_level_per_frame"] >= 1.0,
            "near_full_preserved_or_up": folds["near_full_per_frame"] >= 1.0,
            "corner_slack1_up": folds["corner_slack1_per_frame"] > 1.0,
            "boundary_crossing_down": folds["boundary_crossing_per_frame"] < 1.0,
            "between_tad_down": folds["between_tad_per_frame"] < 1.0,
        }
        p["desired_vs_baseline"] = {
            "pass": bool(all(checks.values())),
            "checks": checks,
        }


def write_phenotype_tsv(res, labels, base, out):
    rows = []
    fields = ["config", "desired_pass"]
    for key, _direction in PHENOTYPE_METRICS:
        fields.extend([key, f"{key}_fold", f"{key}_fold_ci95_low", f"{key}_fold_ci95_high"])
    for c in labels:
        p = res[c]["phenotype"]
        row = {
            "config": c,
            "desired_pass": p["desired_vs_baseline"]["pass"],
        }
        folds = p["fold_vs_baseline"]
        for key, _direction in PHENOTYPE_METRICS:
            row[key] = p.get(key, 0.0)
            row[f"{key}_fold"] = 1.0 if c == base else folds.get(key, float("nan"))
            ci = p.get("fold_vs_baseline_ci95", {}).get(key, {})
            row[f"{key}_fold_ci95_low"] = ci.get("low", float("nan"))
            row[f"{key}_fold_ci95_high"] = ci.get("high", float("nan"))
        rows.append(row)
    with (out / "phenotype_summary.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def write_length_band_tsv(res, labels, base, out):
    labels0 = res[base]["length_bands"]["labels"]
    fields = ["band_kb"]
    for c in labels:
        fields.extend([f"{c}_per_frame", f"{c}_frequency"])
        if c != base:
            fields.extend([f"{c}_per_frame_fold", f"{c}_frequency_fold"])
    rows = []
    for i, band in enumerate(labels0):
        row = {"band_kb": band}
        base_pf = res[base]["length_bands"]["per_frame"][i]
        base_freq = res[base]["length_bands"]["frequency"][i]
        for c in labels:
            pf = res[c]["length_bands"]["per_frame"][i]
            freq = res[c]["length_bands"]["frequency"][i]
            row[f"{c}_per_frame"] = pf
            row[f"{c}_frequency"] = freq
            if c != base:
                row[f"{c}_per_frame_fold"] = pf / base_pf if base_pf else float("nan")
                row[f"{c}_frequency_fold"] = freq / base_freq if base_freq else float("nan")
        rows.append(row)
    with (out / "length_band_summary.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def write_reports(res, labels, base, out, n_bootstrap=1000, bootstrap_seed=12345,
                  quiet=False):
    out.mkdir(parents=True, exist_ok=True)
    _add_phenotype_folds(res, labels, base,
                         n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    (out / "report.json").write_text(json.dumps(res, indent=2))
    write_phenotype_tsv(res, labels, base, out)
    write_length_band_tsv(res, labels, base, out)

    # ---- fine loop-length frequency TSV (diagnostic; noisy by design) ----
    fe = res[base]["fine_histogram"]["edges"]
    rows = ["bin_kb\t" + "\t".join(
        [f"{c}_raw\t{c}_smoothed" for c in labels]
    )]
    for i in range(len(fe) - 1):
        lab = f"{fe[i]}-{fe[i + 1]}"
        cells = []
        for c in labels:
            cells.append(f"{res[c]['fine_histogram']['frequency'][i]:.5f}")
            cells.append(f"{res[c]['fine_histogram']['smoothed_frequency'][i]:.5f}")
        rows.append(lab + "\t" + "\t".join(cells))
    (out / "loop_length_frequency.tsv").write_text("\n".join(rows) + "\n")

    # ---- markdown report ----
    L = labels
    md = [f"# 1D loop-length report — baseline `{base}`", ""]
    md.append(f"{res[base]['lefs']} cohesins/frame, {res[base]['frames']} frames. "
              f"1 monomer = 1 kb. median TAD = {res[base]['median_tad']:.0f} kb.")
    md.append("")
    md.append("## Loop-length distribution (kb)  — value (fold vs baseline)")
    md.append("| stat | " + " | ".join(L) + " |")
    md.append("|" + "---|" * (len(L) + 1))
    b = res[base]["distribution"]
    for k in ["mean", "median", "std", "p25", "p75", "p90", "p95", "p99", "max"]:
        cells = []
        for c in L:
            v = res[c]["distribution"][k]
            cells.append(f"{v:.0f}" if c == base else f"{v:.0f} ({_fold(v, b[k])})")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Stable loop-length bands")
    md.append("These coarse bands are the length-distribution table to interpret; "
              "the fine per-bin table is diagnostic and expected to be noisy.")
    be = res[base]["length_bands"]["labels"]
    hdr = be
    md.append("| config | " + " | ".join(hdr) + " |")
    md.append("|" + "---|" * (len(hdr) + 1))
    for c in L:
        pf = res[c]["length_bands"]["per_frame"]
        if c == base:
            md.append(f"| {c} | " + " | ".join(f"{x:.3f}" for x in pf) + " |")
        else:
            base_pf = res[base]["length_bands"]["per_frame"]
            cells = [
                f"{x:.3f} ({(x / b if b else float('nan')):.2f})"
                for x, b in zip(pf, base_pf)
            ]
            md.append(f"| {c} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("Full stable-band table: length_band_summary.tsv")
    md.append("Fine raw/smoothed diagnostic table: loop_length_frequency.tsv")
    md.append("")
    md.append("## Desired config4 phenotype checkpoint")
    md.append("| config | pass | short | local short | TAD-level | near-full | boundary cross | between-TAD | corner ±1 |")
    md.append("|" + "---|" * 9)
    for c in L:
        p = res[c]["phenotype"]
        folds = p["fold_vs_baseline"]
        status = "baseline" if c == base else ("PASS" if p["desired_vs_baseline"]["pass"] else "FAIL")
        def cell(key):
            v = p[key]
            if c == base:
                return f"{v:.3f}"
            ci = p.get("fold_vs_baseline_ci95", {}).get(key, {})
            lo, hi = ci.get("low", float("nan")), ci.get("high", float("nan"))
            return f"{v:.3f} ({folds[key]:.2f}; {lo:.2f}-{hi:.2f})"
        md.append(
            f"| {c} | {status} | "
            f"{cell('short_abs_per_frame')} | "
            f"{cell('local_short_per_frame')} | "
            f"{cell('tad_level_per_frame')} | "
            f"{cell('near_full_per_frame')} | "
            f"{cell('boundary_crossing_per_frame')} | "
            f"{cell('between_tad_per_frame')} | "
            f"{cell('corner_slack1_per_frame')} |"
        )
    md.append("")
    md.append("Desired pass rule for non-baseline configs: short/local-short/corner up, "
              "TAD-level and near-full preserved or up, boundary-crossing and between-TAD down.")
    md.append("Fold confidence intervals are block-bootstrap CIs over trajectory blocks.")
    md.append("Full machine-readable table: phenotype_summary.tsv")
    md.append("")
    md.append("## Loop classes & anchoring (% of loops; fold vs baseline)")
    md.append("| metric | " + " | ".join(L) + " |")
    md.append("|" + "---|" * (len(L) + 1))
    def classrow(label, getter):
        base_v = getter(res[base])
        cells = []
        for c in L:
            v = getter(res[c])
            cells.append(f"{v:.2f}" if c == base else f"{v:.2f} ({_fold(v, base_v)})")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    for k in ["within_tad", "between_tad", "short_abs", "tad_level", "near_full"]:
        classrow(k, lambda r, k=k: r["classes"][k]["frac"] * 100)
    classrow("boundary_crossing%", lambda r: r["boundary_crossing"]["frac"] * 100)
    classrow("ctcf_leg_pct", lambda r: r["anchoring"]["ctcf_leg_pct"])
    classrow("ctcf_leg_slack1_pct", lambda r: r["anchoring"]["ctcf_leg_slack1_pct"])
    classrow("ctcf_anchored%", lambda r: r["anchoring"]["ctcf_anchored"]["frac"] * 100)
    classrow("corner_frac%", lambda r: r["anchoring"]["corner"]["frac"] * 100)
    classrow("corner_slack1%", lambda r: r["anchoring"]["corner_slack1"]["frac"] * 100)
    classrow("free_meanlen_kb", lambda r: r["anchoring"]["free"]["mean"])
    md.append("")
    les = [c for c in L if "lesion" in res[c]]
    if les:
        md.append("## Lesions")
        md.append("| metric | " + " | ".join(les) + " |")
        md.append("|" + "---|" * (len(les) + 1))
        for k in ["spacing", "block_prob", "type_a_prob",
                  "prerecognition_ticks", "repair_ticks",
                  "mean_count_pf", "pre_frac", "repair_frac",
                  "typeA_frac", "stalling_pf", "stalling_frac"]:
            md.append(f"| {k} | " + " | ".join(f"{res[c]['lesion'][k]:.3f}" for c in les) + " |")
        md.append("")
    rnaps = [c for c in L if "rnapii" in res[c]]
    if rnaps:
        md.append("## RNAPII")
        md.append("| metric | " + " | ".join(rnaps) + " |")
        md.append("|" + "---|" * (len(rnaps) + 1))
        for k in ["mean_count_all_frames", "max_simultaneous",
                  "capacity_hit_frames_pct", "elongation_speed_kb_per_min"]:
            md.append(f"| {k} | " + " | ".join(f"{res[c]['rnapii'][k]:.3f}" for c in rnaps) + " |")
        for state in ["poised", "paused", "elongating", "terminating", "stalled"]:
            md.append(f"| state_{state}_pct | " + " | ".join(
                f"{res[c]['rnapii']['state_mix_pct'].get(state, 0.0):.3f}" for c in rnaps
            ) + " |")
        md.append("")
    md.append("## Loop-length histogram")
    md.append("Top panel: smoothed fine-bin diagnostic view. Middle panel: all-distance log-log view. "
              "Bottom panel: all-distance fold change versus baseline with log-x and linear-y axes.")
    md.append("![loop-length frequency](loop_length_histogram.png)")
    md.append("")
    (out / "report.md").write_text("\n".join(md) + "\n")
    png = plot_histogram(res, labels, base, out)
    if not quiet:
        print(f"[report] wrote {out}/report.md , report.json , "
              f"phenotype_summary.tsv , length_band_summary.tsv , "
              f"loop_length_frequency.tsv , {png.name}")


def main():
    ap = argparse.ArgumentParser(description="1D loop-length comparison for N configs (first = baseline)")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("configs", type=Path, nargs="+", help="config YAMLs; first is baseline")
    ap.add_argument("--labels", type=str, default=None, help="comma-separated labels (default: file stems)")
    ap.add_argument("--bin", type=int, default=25, help="fine histogram bin width in kb (default 25)")
    ap.add_argument("--max", type=int, default=1000, help="fine histogram max edge in kb before overflow bin (default 1000)")
    ap.add_argument("--smooth-bins", type=int, default=5,
                    help="moving-average window for diagnostic fine histogram (default 5)")
    ap.add_argument("--block-size", type=int, default=500,
                    help="trajectory frames per block for phenotype CI estimates (default 500)")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples for fold CIs; 0 disables CIs (default 1000)")
    ap.add_argument("--bootstrap-seed", type=int, default=12345,
                    help="random seed for block-bootstrap CIs")
    ap.add_argument("--reuse", action="store_true",
                    help="deprecated compatibility flag; reuse is now the default")
    ap.add_argument("--force", action="store_true",
                    help="rerun LEF stage even if LEFPositions.h5 already exists")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s", force=True)

    labels = (args.labels.split(",") if args.labels
              else [c.stem.replace("_", "-") for c in args.configs])
    if len(labels) != len(args.configs):
        ap.error("--labels count must match number of configs")
    if args.reuse and not args.quiet:
        print("[note]   --reuse is now default; existing H5 files are reused unless --force is set")

    res = {}
    for label, cfgp in zip(labels, args.configs):
        cfg = load_config(cfgp)
        h5 = run_sim(cfgp, cfg, args.out_dir / label,
                     force=args.force, quiet=args.quiet)
        res[label] = analyze(h5, topology(cfg), args.bin, args.max, cfg=cfg,
                             block_size=args.block_size,
                             smooth_bins=args.smooth_bins)

    write_reports(res, labels, labels[0], args.out_dir / "report",
                  n_bootstrap=args.bootstrap,
                  bootstrap_seed=args.bootstrap_seed,
                  quiet=args.quiet)


if __name__ == "__main__":
    main()
