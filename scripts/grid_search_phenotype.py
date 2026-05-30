#!/usr/bin/env python
"""Biology-gated grid search to maximize the CONFIG_EXPLANATIONS phenotype.

For each parameter combo it runs the matched pair at the 1D stage --
  WT  = config1 (transcription ON)  + overrides + boundary = --c1-boundary
  OFF = same overrides + RNAPII disabled + boundary = --c2-boundary
-- then scores the config2/config1 (OFF/WT) phenotype folds and evaluates
biological correctness on the WT run.

DESIRED PHENOTYPE (configs/CONFIG_EXPLANATIONS.md, OFF/WT folds):
    larger loops, fewer short loops, more TAD-scale loops, higher corner-dot
    score, more boundary-crossing stripes. (boundary_occ is monitored only -- at
    a weak config2 boundary the single-leg occupancy proxy is flux-confounded and
    not part of the takeaway, so it is NOT in the objective.)

BIOLOGY GATES (measured on WT; Kim 2026 / mammalian calibration):
    * displacement > 0   cohesin loops follow transcription (Kim Fig 2)
    * rho > 0            cohesin E-P loop drives output (Spearman)
    * 1.5 <= median elongation kb/min <= 2.5   mammalian ~2 kb/min
    * ELONGATING >= 40% and PAUSED <= 70%      sane Pol II state mix

GATE MODES (--gate-mode):
    valid   keep only biology-valid combos
    overall ignore gates (chase raw phenotype)
    both    report BOTH rankings side by side (default)

STAGES (--stage):
    A   coarse: 4 primary knobs, short traj, 1 seed
    B   refine: top-K combos from A's best.json + secondary knobs, long traj, N seeds

Usage:
    python scripts/grid_search_phenotype.py --stage A --traj 20000 --seeds 1
    python scripts/grid_search_phenotype.py --stage B --traj 50000 --seeds 3 --top 8
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
from polychrom.pipelines.loop_extrusion import lef as lef_stage
from polychrom.pipelines.loop_extrusion.config import load_config
from tune_depletion_1d import (_boundaries, metrics_1d, _disable_rnapii,
                               _scale_boundary)
from nascent_rna_abundance import measure_per_gene, aggregate_by_base
from cohesin_rnapii_coupling import measure_coupling

BASE = "configs/config1.yaml"

# --------------------------------------------------------------------------- #
# parameter grids
# --------------------------------------------------------------------------- #
# Stage A: 4 primary RNAPII knobs (the moving-barrier / eviction physics).
GRID_A = {
    "lifetime_rnapii_stalled":      [18, 24, 30, 40, 55],
    "rnapii_stall_prob":            [0.20, 0.30, 0.40],
    "rnapii_elongating_block_prob": [0.80, 0.90],
    "rnapii_push_prob":             [0.40, 0.50, 0.60],
}
# Stage B: refine each carried-over combo with these secondary knobs.
GRID_B_SECONDARY = {
    "rnapii_headon_push_prob":       [0.02, 0.05],
    "rnapii_pause_cohesin_restraint": [0.20, 0.30],
}

# phenotype objective: (metric, want_up?, default weight)
PHENO = [
    ("mean_loop", True, 1.0),
    ("frac_short", False, 1.0),
    ("frac_tad", True, 1.0),
    ("corner_both_pct", True, 2.0),     # corner score = headline (CONFIG_EXPLANATIONS)
    ("cross_boundary_pct", True, 1.0),
]
MONITOR = ["boundary_occ_pct"]          # printed, not scored

# biology gates
ELONG_LO, ELONG_HI = 1.5, 2.5
ELONG_MIN_FRAC, PAUSE_MAX_FRAC = 40.0, 70.0


# --------------------------------------------------------------------------- #
def _key(overrides: dict, boundary: float, seed: int, traj: int) -> str:
    payload = json.dumps([sorted(overrides.items()), boundary, seed, traj])
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _run(overrides: dict, *, deplete: bool, boundary: float, seed: int,
         traj: int, out_dir: Path) -> Path:
    """Run (or reuse cached) one lef trajectory; return its h5 path.

    The OFF (depletion) run has RNAPII disabled, so the RNAPII overrides are
    inert -- it depends only on (boundary, seed, traj). Key it with empty
    overrides so every combo shares ONE OFF trajectory per seed (big speedup).
    """
    ckey = _key({} if deplete else overrides, boundary, seed, traj)
    h5 = out_dir / f"{'off' if deplete else 'wt'}_{ckey}.h5"
    if h5.exists():
        return h5
    cfg = load_config(BASE)
    lc = cfg.lef
    lc.output_path = str(h5)
    lc.trajectory_length = traj
    lc.warmup_steps = min(int(lc.warmup_steps), max(2000, traj // 5))
    lc.seed = int(seed)
    for k, v in overrides.items():
        lc.topology_kwargs[k] = v
    if deplete:
        _disable_rnapii(cfg)
        _scale_boundary(lc.topology_kwargs, boundary)
    else:
        _scale_boundary(lc.topology_kwargs, boundary)
    lef_stage.run(lc)
    return h5


def _biology(wt_h5: Path) -> dict:
    """Biology readouts + gate verdict on a WT (txn ON) run."""
    cfg = load_config(BASE)
    gid_rows, meta = measure_per_gene(wt_h5, cfg)
    kb = [r["kb_min"] for r in gid_rows if r["kb_min"] == r["kb_min"]]
    elong_med = float(np.median(kb)) if kb else float("nan")
    mix = meta["state_mix"]
    elong_pct = mix.get("ELONGATING", 0.0)
    pause_pct = mix.get("PAUSED", 0.0)
    coup = measure_coupling(wt_h5, cfg)
    # soft: constitutive (bases 0,2) vs long cohesin-dep (bases 5,6) output
    agg = aggregate_by_base(gid_rows, int(cfg.lef.num_chains))
    def _out(b): return agg[b]["nascent_allele"] if b in agg else float("nan")
    class_ok = (np.nanmin([_out(0), _out(2)]) > np.nanmax([_out(5), _out(6)]))
    gates = {
        "kim_dir":  coup["displacement"] > 0,
        "ep_coop":  coup["rho"] > 0,
        "elong":    ELONG_LO <= elong_med <= ELONG_HI,
        "polmix":   (elong_pct >= ELONG_MIN_FRAC) and (pause_pct <= PAUSE_MAX_FRAC),
    }
    return {
        "displacement": coup["displacement"], "rho": coup["rho"],
        "elong_kb_min": elong_med, "elong_pct": elong_pct, "pause_pct": pause_pct,
        "class_ok": bool(class_ok), "gates": gates, "bio_valid": all(gates.values()),
    }


def evaluate(overrides: dict, *, c1: float, c2: float, seeds: int, traj: int,
             out_dir: Path, weights: dict) -> dict:
    """Run all seeds, average folds + biology, score phenotype."""
    fold_acc = {m: [] for m, _, _ in PHENO + [(x, True, 0) for x in MONITOR]}
    bios = []
    for si in range(seeds):
        seed = 1001 + si
        wt = _run(overrides, deplete=False, boundary=c1, seed=seed, traj=traj, out_dir=out_dir)
        off = _run(overrides, deplete=True, boundary=c2, seed=seed, traj=traj, out_dir=out_dir)
        cfg = load_config(BASE); bnd = _boundaries(cfg)
        mw = metrics_1d(wt, bnd); md = metrics_1d(off, bnd)
        for m in fold_acc:
            fold_acc[m].append(md[m] / mw[m] if mw[m] else float("nan"))
        bios.append(_biology(wt))
    folds = {m: float(np.nanmean(v)) for m, v in fold_acc.items()}
    # phenotype score = weighted SIGNED log-fold (correct direction -> positive).
    # NOT hard-gated on all-correct: at short traj the corner/cross counts are
    # noisy and a single spurious sign flip should penalize, not disqualify. The
    # all-correct flag is reported separately and confirmed at the longer Stage B.
    dir_ok, score = True, 0.0
    for m, up, _w in PHENO:
        w = weights.get(m, _w)
        f = folds[m]
        correct = (f > 1) if up else (f < 1)
        dir_ok = dir_ok and correct
        score += w * (np.log(f) if up else -np.log(f))
    # biology: average gate pass-rate over seeds; valid if all gates hold on the mean
    bio_mean = {
        "displacement": float(np.nanmean([b["displacement"] for b in bios])),
        "rho": float(np.nanmean([b["rho"] for b in bios])),
        "elong_kb_min": float(np.nanmean([b["elong_kb_min"] for b in bios])),
        "elong_pct": float(np.nanmean([b["elong_pct"] for b in bios])),
        "pause_pct": float(np.nanmean([b["pause_pct"] for b in bios])),
        "class_ok": all(b["class_ok"] for b in bios),
    }
    bio_valid = (bio_mean["displacement"] > 0 and bio_mean["rho"] > 0 and
                 ELONG_LO <= bio_mean["elong_kb_min"] <= ELONG_HI and
                 bio_mean["elong_pct"] >= ELONG_MIN_FRAC and
                 bio_mean["pause_pct"] <= PAUSE_MAX_FRAC)
    return {
        "overrides": overrides, "folds": folds, "pheno_dir_ok": dir_ok,
        "score": score, "bio": bio_mean, "bio_valid": bio_valid,
    }


# --------------------------------------------------------------------------- #
def _combos_A():
    keys = list(GRID_A)
    for vals in itertools.product(*[GRID_A[k] for k in keys]):
        yield dict(zip(keys, vals))


def _combos_B(top_overrides):
    keys = list(GRID_B_SECONDARY)
    for base in top_overrides:
        for vals in itertools.product(*[GRID_B_SECONDARY[k] for k in keys]):
            ov = dict(base); ov.update(dict(zip(keys, vals)))
            yield ov


def _print_table(title, rows):
    print(f"\n#### {title}  (top {len(rows)}) ####")
    cols = [m for m, _, _ in PHENO] + MONITOR
    hdr = (f"{'score':>7} {'bioOK':>5} " +
           " ".join(f"{c.split('_')[0][:5]:>6}" for c in cols) +
           f" {'disp':>7} {'rho':>5} {'kbmin':>6}  params")
    print(hdr)
    for r in rows:
        f = r["folds"]; b = r["bio"]
        fs = " ".join(f"{f[c]:>6.3f}" for c in cols)
        ov = ",".join(f"{k.split('_')[1] if '_' in k else k}={v}"
                      for k, v in sorted(r["overrides"].items()))
        sc = r["score"]
        print(f"{sc:>7.3f} {('Y' if r['bio_valid'] else 'n'):>5} {fs} "
              f"{b['displacement']:>+7.3f} {b['rho']:>5.2f} {b['elong_kb_min']:>6.2f}  {ov}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], default="A")
    ap.add_argument("--traj", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="runs/grid")
    ap.add_argument("--c1-boundary", type=float, default=0.60)
    ap.add_argument("--c2-boundary", type=float, default=0.50)
    ap.add_argument("--gate-mode", choices=["valid", "overall", "both"], default="both")
    ap.add_argument("--weights", default="", help="m=w,... override score weights")
    args = ap.parse_args(argv[1:])

    weights = {}
    for kv in filter(None, args.weights.split(",")):
        k, v = kv.split("="); weights[k] = float(v)

    out_dir = Path(args.out) / f"stage{args.stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "A":
        combos = list(_combos_A())
    else:
        prev = json.loads((Path(args.out) / "stageA" / "best.json").read_text())
        # refine the biology-VALID leaders (fall back to overall if none valid)
        pool = prev.get("ranked_valid") or prev["ranked"]
        top_ovs = [r["overrides"] for r in pool[:args.top]]
        combos = list(_combos_B(top_ovs))
    print(f"# stage={args.stage}  combos={len(combos)}  traj={args.traj} "
          f"seeds={args.seeds}  c1={args.c1_boundary} c2={args.c2_boundary}")

    results, jl = [], (out_dir / "results.jsonl").open("w")
    for i, ov in enumerate(combos, 1):
        r = evaluate(ov, c1=args.c1_boundary, c2=args.c2_boundary,
                     seeds=args.seeds, traj=args.traj, out_dir=out_dir,
                     weights=weights)
        results.append(r); jl.write(json.dumps(r) + "\n"); jl.flush()
        print(f"  [{i}/{len(combos)}] score={r['score']:>7.3f} "
              f"bio={'Y' if r['bio_valid'] else 'n'} {ov}")
    jl.close()

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    valid = [r for r in ranked if r["bio_valid"]]
    (out_dir / "best.json").write_text(json.dumps(
        {"ranked": ranked[:args.top], "ranked_valid": valid[:args.top]}, indent=2))

    if args.gate_mode in ("overall", "both"):
        _print_table("BEST OVERALL (biology gates ignored)", ranked[:args.top])
    if args.gate_mode in ("valid", "both"):
        _print_table("BEST BIOLOGY-VALID", valid[:args.top] or ranked[:args.top])
    if valid:
        b = valid[0]
        print("\n# WINNER (biology-valid):", json.dumps(b["overrides"]))
        print("# folds:", {k: round(v, 3) for k, v in b["folds"].items()})
        print("# bio:  ", {k: round(v, 3) if isinstance(v, float) else v
                           for k, v in b["bio"].items()})
    else:
        print("\n# NO biology-valid combo with correct phenotype direction.")
    print(f"\n# wrote {out_dir/'best.json'} , {out_dir/'results.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
