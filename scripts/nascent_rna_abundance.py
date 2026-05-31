#!/usr/bin/env python
"""Nascent-RNA abundance estimation from a loop-extrusion `lef`-stage run.

Reads ``LEFPositions.h5`` (the 1D stage output, which records RNAPII positions
and states) plus the YAML config, and reports per-gene nascent-RNA abundance.

Real-time calibration
---------------------
1 LEF tick == 1 MD block == ``polymer.md_steps_per_block`` integrator steps.
With the Hansen-lab MSD calibration (1 MD step ~= 0.0063 s; same error_tol /
collision_rate / density), the wall-clock duration of one tick is::

    tick_seconds = md_steps_per_block * 0.0063

For the calibrated config (md_steps_per_block=2540) that is 16.0 s/tick.

Nascent-RNA abundance
---------------------
Each ELONGATING Pol II carries one growing nascent transcript. The nascent-RNA
abundance of a gene is therefore the time-mean number of ELONGATING Pol II on
its body -- exactly the GRO/PRO-seq / 4sU observable (Pol II density). We also
report completed transcripts (TES arrivals) as transcripts/hour, and the
realized elongation speed in kb/min as a calibration cross-check.

The per-gene measurement is exposed as :func:`measure_per_gene` /
:func:`aggregate_by_base` for reuse (e.g. scripts/topology_layout_svg.py).

Usage::

    python scripts/nascent_rna_abundance.py CONFIG.yaml [RUN_DIR_OR_H5]

If the second argument is omitted, ``trajectory/LEFPositions.h5`` is used.
When the target ``LEFPositions.h5`` does not exist, the 1D ``lef`` stage is run
first (pure numpy, no GPU) to generate it, then measured.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polychrom.pipelines.loop_extrusion import lef as lef_stage  # noqa: E402
from polychrom.pipelines.loop_extrusion.config import load_config  # noqa: E402

# Hansen & Yang 2024 Suppl. Box 1: MSD-calibrated MD integrator step duration
# for error_tol=0.01, collision_rate=1, density=0.2 (the polychrom setup used
# throughout this pipeline).
MD_STEP_SECONDS = 0.0063
STATE_NAMES = {0: "POISED", 1: "PAUSED", 2: "ELONGATING", 3: "TERMINATING",
               4: "STALLED"}
PAUSED = 1
ELONGATING = 2
STALLED = 4


def resolve_h5(arg: str | None) -> Path:
    if arg is None:
        return Path("trajectory/LEFPositions.h5")
    p = Path(arg)
    return p if p.is_file() else p / "LEFPositions.h5"


def ensure_h5(cfg, h5_path: Path) -> None:
    """Run the (numpy-only, no-GPU) lef stage to create ``h5_path`` if missing."""
    if h5_path.exists():
        return
    print(f"# {h5_path} not found; running lef stage to generate it...")
    cfg.lef.output_path = str(h5_path)
    lef_stage.run(cfg.lef)


def tick_seconds(cfg) -> float:
    return cfg.polymer.md_steps_per_block * MD_STEP_SECONDS


def measure_per_gene(h5_path: Path, cfg) -> Tuple[List[dict], Dict]:
    """Per-gene-id transcription metrics measured from the trajectory.

    Returns ``(gid_rows, meta)``. ``gid_rows`` is one dict per recorded gene_id
    (chain copies are distinct gene_ids); ``meta`` carries tick/timebase and the
    global Pol II state mix. Use :func:`aggregate_by_base` to collapse chain
    copies into per-biological-gene (per-allele) numbers.
    """
    ts = tick_seconds(cfg)
    with h5py.File(h5_path, "r") as fh:
        if not fh.attrs.get("rnapii_enabled", False):
            raise ValueError(f"{h5_path}: RNAPII not enabled in this run.")
        pos = fh["rnapii_positions"][:]          # (T, R, 2): [site, gene_id]
        states = fh["rnapii_states"][:].astype(int)
        ids = fh["rnapii_ids"][:]                 # (T, R): stable per-Pol uid
        genes = fh["genes"][:]
        num_chains = int(fh.attrs.get("num_chains", 1))

    T = pos.shape[0]
    total_seconds = T * ts
    site_pos = pos[:, :, 0]
    gid = pos[:, :, 1]
    present = site_pos >= 0
    n_base = max(1, len({int(r["gene_id"]) for r in genes}) // max(1, num_chains))

    gid_rows: List[dict] = []
    for row in sorted(genes, key=lambda r: int(r["gene_id"])):
        g = int(row["gene_id"]); tss = int(row["tss"]); tes = int(row["tes"])
        glen = abs(tes - tss)
        sel = (gid == g) & present
        n_present = int(sel.sum())
        # Engaged Pol II carries a nascent transcript whether it is advancing
        # (ELONGATING) or obstacle-blocked (STALLED); both count toward nascent.
        engaged = sel & ((states == ELONGATING) | (states == STALLED))
        nascent = float(engaged.sum(axis=1).mean())
        paused = sel & (states == PAUSED)
        # %paused is measured among Pol II in the promoter-proximal kinetic pool,
        # i.e. excluding obstacle-STALLED ticks -- otherwise long stalls in the
        # gene body dilute the metric toward 0 (see STATE_STALLED).
        n_stalled = int((sel & (states == STALLED)).sum())
        denom = n_present - n_stalled
        pct_paused = (100.0 * paused.sum() / denom) if denom else 0.0

        # Recording columns are the live, compacted RNAPII list, so a column is
        # not a stable Pol II across ticks. Use the per-Pol uid: a completion is
        # one distinct uid of this gene that reaches the TES.
        tes_uids = ids[(site_pos == tes) & (gid == g)]
        completions = int(len({int(u) for u in tes_uids if u >= 0}))
        per_hour = completions / total_seconds * 3600.0

        # Elongation speed for THIS gene: single-site steps over consecutive
        # ELONGATING ticks of the same uid in the same column. Steps where the Pol
        # shifts columns (an earlier Pol unloads -> the live list compacts) are
        # dropped; the estimate stays unbiased but uses fewer samples than the raw
        # tick count. Filter by gid so each gene reports its own speed, not the
        # lattice-wide mean.
        s_cur, s_prev = states[1:], states[:-1]
        g_cur, g_prev = gid[1:], gid[:-1]
        i_cur, i_prev = ids[1:], ids[:-1]
        step = site_pos[1:] - site_pos[:-1]
        adv_mask = (
            (s_cur == ELONGATING) & (s_prev == ELONGATING)
            & (g_cur == g) & (g_prev == g)
            & (i_cur >= 0) & (i_cur == i_prev)
            & (np.abs(step) <= 1)
        )
        eticks = int(adv_mask.sum())
        adv = int(np.abs(step)[adv_mask].sum())
        kb_min = (adv / eticks / ts * 60.0) if eticks else float("nan")

        gid_rows.append({
            "gid": g, "base": g % n_base, "len_kb": glen, "nascent": nascent,
            "completed": completions, "per_hour": per_hour, "kb_min": kb_min,
            "pct_paused": pct_paused,
        })

    tot = present.sum()
    state_mix = {}
    if tot:
        state_mix = {STATE_NAMES[s]: 100.0 * ((states == s) & present).sum() / tot
                     for s in STATE_NAMES}
    meta = {"T": T, "tick_seconds": ts, "total_seconds": total_seconds,
            "num_chains": num_chains, "state_mix": state_mix}
    return gid_rows, meta


def aggregate_by_base(gid_rows: List[dict], num_chains: int) -> Dict[int, dict]:
    """Collapse chain copies -> per-biological-gene. Per-allele = mean over copies."""
    out: Dict[int, dict] = {}
    for r in gid_rows:
        a = out.setdefault(r["base"], {
            "len_kb": r["len_kb"], "n": 0, "nascent_sum": 0.0,
            "per_hour_sum": 0.0, "kb_min": [], "pct_paused": [], "completed": 0,
        })
        a["n"] += 1
        a["nascent_sum"] += r["nascent"]
        a["per_hour_sum"] += r["per_hour"]
        a["completed"] += r["completed"]
        if r["kb_min"] == r["kb_min"]:
            a["kb_min"].append(r["kb_min"])
        a["pct_paused"].append(r["pct_paused"])
    for a in out.values():
        a["nascent_allele"] = a["nascent_sum"] / max(1, a["n"])
        a["per_hour_allele"] = a["per_hour_sum"] / max(1, a["n"])
        a["kb_min_mean"] = float(np.mean(a["kb_min"])) if a["kb_min"] else float("nan")
        a["pct_paused_mean"] = float(np.mean(a["pct_paused"]))
    return out


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cfg = load_config(argv[1])
    h5_path = resolve_h5(argv[2] if len(argv) > 2 else None)
    ensure_h5(cfg, h5_path)
    gid_rows, meta = measure_per_gene(h5_path, cfg)
    agg = aggregate_by_base(gid_rows, meta["num_chains"])

    ts = meta["tick_seconds"]; tot_s = meta["total_seconds"]
    print(f"# run: {h5_path}")
    print(f"# config: {argv[1]}")
    print(f"# T = {meta['T']} ticks | tick = {ts:.2f} s "
          f"(md_steps_per_block={cfg.polymer.md_steps_per_block} x {MD_STEP_SECONDS} s)")
    print(f"# total simulated time = {tot_s/3600:.2f} h "
          f"({tot_s/3600/24:.2f} days) | chains = {meta['num_chains']}")
    print()
    hdr = ("gene", "len_kb", "nascent_PolII", "completed", "transcripts/h",
           "elong_kb/min", "%paused")
    fmt = "{:>5} {:>7} {:>14} {:>10} {:>14} {:>13} {:>8}"
    print(fmt.format(*hdr))
    for r in gid_rows:
        print(fmt.format(r["gid"], r["len_kb"], f"{r['nascent']:.3f}", r["completed"],
                         f"{r['per_hour']:.3f}", f"{r['kb_min']:.2f}",
                         f"{r['pct_paused']:.1f}"))

    print("\n# aggregated per biological gene (summed over chain copies):")
    print(fmt.format(*hdr))
    all_nascent = []
    for base in sorted(agg):
        a = agg[base]; all_nascent.append(a["nascent_sum"])
        print(fmt.format(base, a["len_kb"], f"{a['nascent_sum']:.3f}", a["completed"],
                         f"{a['per_hour_sum']:.3f}", f"{a['kb_min_mean']:.2f}",
                         f"{a['pct_paused_mean']:.1f}"))

    if meta["state_mix"]:
        print("\n# global Pol II state mix (% of present-Pol-II-ticks):")
        print("  " + "  ".join(f"{k}={v:.1f}%" for k, v in meta["state_mix"].items()))
    print(f"\n# total nascent-RNA abundance (sum mean elongating Pol II) = "
          f"{sum(all_nascent):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
