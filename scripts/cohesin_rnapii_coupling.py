#!/usr/bin/env python
"""Measure cohesin <-> RNAPII coupling in a `lef`-stage trajectory.

Tests whether the simulated dynamics reproduce the published cohesin-RNAPII
interplay:

  1. INTERFERENCE / "moving barrier" (Banigan & Mirny PNAS 2022; Kim 2026):
     elongating Pol II pushes cohesin in the direction of transcription, so
     cohesin legs are relocalized DOWNSTREAM (TES side) of active genes.
     -> reported as downstream/upstream cohesin-occupancy ratio.

  2. DIRECTIONALITY (Kim et al. 2026 Fig 2): cohesin coordinates with RNAPII in
     the direction of gene transcription. -> per-gene leg displacement bias.

  3. COOPERATION (Yang & Hansen 2024; Tei 2026): cohesin E-P loop contact drives
     transcriptional output. -> correlation(E-P contact frequency, nascent Pol II).

Usage::  python scripts/cohesin_rnapii_coupling.py CONFIG.yaml [RUN_DIR_OR_H5]
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polychrom.pipelines.loop_extrusion import lef as lef_stage  # noqa: E402
from polychrom.pipelines.loop_extrusion.config import load_config  # noqa: E402

ELONGATING = 2


def measure_coupling(h5_path: Path, cfg) -> dict:
    """Importable core for the two Kim-2026 gates used by the grid search.

    Returns ``{"displacement", "rho", "median_down_up", "n_active"}``:
      * ``displacement`` -- mean signed cohesin-leg step (sites/tick) inside ACTIVE
        gene bodies in the transcription-direction frame. >0 == "cohesin loops
        follow transcription" (Kim 2026 Fig 2). Same computation as ``main``.
      * ``rho`` -- Spearman corr(E-P contact freq, nascent Pol II); >0 == cohesin
        loop drives output.
    NaN for a quantity when it cannot be computed (no active genes / <3 E-P genes).
    """
    with h5py.File(h5_path, "r") as fh:
        pos = fh["positions"][:]                  # (T, L, 2)
        rpos = fh["rnapii_positions"][:]          # (T, R, 2): [site, gene_id]
        rstate = fh["rnapii_states"][:].astype(int)
        genes = fh["genes"][:]
        tol = int(cfg.lef.topology_kwargs.get("ep_contact_tolerance", 2))
    T = pos.shape[0]
    chain_length = int(cfg.lef.chain_length)
    num_chains = int(cfg.lef.num_chains)
    base_specs = cfg.lef.topology_kwargs.get("genes", [])
    n_base = max(1, len(genes) // num_chains)

    nascent_list, contact_list = [], []
    for row in genes:
        g = int(row["gene_id"]); tss = int(row["tss"])
        sel = (rpos[:, :, 1] == g) & (rpos[:, :, 0] >= 0) & (rstate == ELONGATING)
        nascent_list.append(float(sel.sum(1).mean()))
        spec = base_specs[g % n_base] if base_specs else {}
        offset = (g // n_base) * chain_length
        raw_enh = spec.get("enhancers") or (
            [spec["enhancer_pos"]] if spec.get("enhancer_pos") is not None else [])
        if not raw_enh:
            continue
        ehits = 0
        for t in range(T):
            ivals = [(min(a, b), max(a, b)) for a, b in pos[t] if a >= 0]
            for e in raw_enh:
                ea = int(e) + offset
                lo = min(tss, ea) + tol; hi = max(tss, ea) - tol
                if any(L <= lo and R >= hi for L, R in ivals):
                    ehits += 1; break
        contact_list.append((ehits / T, nascent_list[-1]))

    active = {int(r["gene_id"]): nas for r, nas in zip(genes, nascent_list) if nas >= 1.0}
    gdir = {int(r["gene_id"]): int(r["direction"]) for r in genes}
    gbody = {int(r["gene_id"]): tuple(sorted((int(r["tss"]), int(r["tes"]))))
             for r in genes}
    signed = []
    for j in range(pos.shape[1]):
        for side in (0, 1):
            s = pos[:, j, side]
            for t in range(T - 1):
                a, b = s[t], s[t + 1]
                if a < 0 or b < 0 or abs(int(b) - int(a)) > 1:
                    continue
                for g in active:
                    lo, hi = gbody[g]
                    if lo <= a <= hi:
                        signed.append((int(b) - int(a)) * gdir[g]); break

    rho = float("nan")
    if len(contact_list) >= 3:
        c = np.array(contact_list)
        rc = np.argsort(np.argsort(c[:, 0])); rn = np.argsort(np.argsort(c[:, 1]))
        rho = float(np.corrcoef(rc, rn)[0, 1])
    return {
        "displacement": float(np.mean(signed)) if signed else float("nan"),
        "rho": rho,
        "n_active": len(active),
    }


def _resolve_h5(arg: str | None) -> Path:
    if arg is None:
        return Path("trajectory/LEFPositions.h5")
    p = Path(arg)
    return p if p.is_file() else p / "LEFPositions.h5"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cfg = load_config(argv[1])
    h5_path = _resolve_h5(argv[2] if len(argv) > 2 else None)
    if not h5_path.exists():
        print(f"# {h5_path} not found; running lef stage...")
        cfg.lef.output_path = str(h5_path)
        lef_stage.run(cfg.lef)

    with h5py.File(h5_path, "r") as fh:
        pos = fh["positions"][:]                 # (T, L, 2) cohesin leg sites
        rpos = fh["rnapii_positions"][:]         # (T, R, 2): [site, gene_id]
        rstate = fh["rnapii_states"][:].astype(int)
        genes = fh["genes"][:]
        N = int(fh.attrs["N"])
        tol = int(cfg.lef.topology_kwargs.get("ep_contact_tolerance", 2))

    T = pos.shape[0]
    # ---- cohesin leg occupancy profile (fraction of ticks a leg sits here) ---
    occ = np.zeros(N)
    legs = pos.reshape(T, -1)                     # all leg coords per tick
    for t in range(T):
        valid = legs[t][legs[t] >= 0]
        occ[valid] += 1
    occ /= T
    global_occ = occ.mean()

    # ---- per-gene metrics ----------------------------------------------------
    enh = {}  # gene_id -> list of enhancer sites, from config ep_pairs (chain0)
    # Build enhancer map from gene specs (chain-relative) + chain offsets.
    chain_length = int(cfg.lef.chain_length)
    num_chains = int(cfg.lef.num_chains)
    base_specs = cfg.lef.topology_kwargs.get("genes", [])

    W = 25                                        # flank window (sites = kb)
    print(f"# run: {h5_path} | T={T} ticks | global cohesin leg occ={global_occ:.4f}")
    print("\n## 1. Cohesin spatial distribution flanking genes (DESCRIPTIVE).")
    print("#   up_occ/down_occ = cohesin leg occupancy just upstream of TSS vs just")
    print("#   downstream of TES. NOTE: dominated by NIPBL loading + E-P anchoring at")
    print("#   enhancers (Kim 2026 Fig 2d), NOT a clean pushing probe -- when a gene's")
    print("#   enhancer is upstream, cohesin enriches upstream (ratio<1) by design.")
    print("#   The clean pushing/directionality test is the leg-displacement metric below.")
    print("{:>5} {:>4} {:>9} {:>9} {:>11} {:>9}".format(
        "gene", "dir", "up_occ", "down_occ", "down/up", "nascentE"))

    nascent_list, contact_list, ratio_list = [], [], []
    for row in genes:
        g = int(row["gene_id"]); tss = int(row["tss"]); tes = int(row["tes"])
        d = int(row["direction"])
        chain_lo = (g // (len(genes)//num_chains)) * 0  # not used; keep absolute
        # upstream = behind TSS, downstream = beyond TES (transcription frame)
        up = (tss - d * W, tss) if d > 0 else (tss, tss - d * W)
        dn = (tes, tes + d * W) if d > 0 else (tes + d * W, tes)
        up_lo, up_hi = sorted(up); dn_lo, dn_hi = sorted(dn)
        up_occ = occ[max(0, up_lo):up_hi].mean() if up_hi > up_lo else 0.0
        dn_occ = occ[dn_lo:min(N, dn_hi)].mean() if dn_hi > dn_lo else 0.0
        ratio = dn_occ / up_occ if up_occ > 0 else float("nan")

        # nascent (elongating Pol II for this gene, time-mean)
        sel = (rpos[:, :, 1] == g) & (rpos[:, :, 0] >= 0) & (rstate == ELONGATING)
        nascent = float(sel.sum(1).mean())

        # E-P contact frequency: fraction of ticks a cohesin loop brackets (TSS,enh)
        # enhancer sites for this gene, derived from base spec replicated per chain
        n_base = max(1, len(genes) // num_chains)
        spec = base_specs[g % n_base] if base_specs else {}
        offset = (g // n_base) * chain_length
        raw_enh = spec.get("enhancers") or ([spec["enhancer_pos"]] if spec.get("enhancer_pos") is not None else [])
        contact_freq = float("nan")
        if raw_enh:
            ehits = 0
            for t in range(T):
                lt = pos[t]
                ivals = [(min(a, b), max(a, b)) for a, b in lt if a >= 0]
                hit = False
                for e in raw_enh:
                    ea = int(e) + offset
                    lo = min(tss, ea) + tol; hi = max(tss, ea) - tol
                    if any(L <= lo and R >= hi for L, R in ivals):
                        hit = True; break
                ehits += hit
            contact_freq = ehits / T

        nascent_list.append(nascent)
        if contact_freq == contact_freq:
            contact_list.append((contact_freq, nascent))
        if ratio == ratio:
            ratio_list.append(ratio)
        print(f"{g:>5} {d:>+4d} {up_occ:>9.4f} {dn_occ:>9.4f} {ratio:>11.2f} {nascent:>9.3f}")

    # ---- direct pushing test: signed leg displacement inside ACTIVE gene bodies
    # Per-slot leg identity is stable across ticks, so pos[t+1,j]-pos[t,j] is a
    # real displacement of the same leg. Moving-barrier model => legs inside an
    # active (transcribed) gene body drift in the transcription direction.
    active = {int(r["gene_id"]): nas for r, nas in zip(genes, nascent_list) if nas >= 1.0}
    gdir = {int(r["gene_id"]): int(r["direction"]) for r in genes}
    gbody = {int(r["gene_id"]): tuple(sorted((int(r["tss"]), int(r["tes"]))))
             for r in genes}
    signed = []
    for j in range(pos.shape[1]):
        for side in (0, 1):
            s = pos[:, j, side]
            for t in range(T - 1):
                a, b = s[t], s[t + 1]
                if a < 0 or b < 0 or abs(int(b) - int(a)) > 1:
                    continue
                for g, nas in active.items():
                    lo, hi = gbody[g]
                    if lo <= a <= hi:
                        signed.append((int(b) - int(a)) * gdir[g])
                        break
    if ratio_list:
        med = float(np.median(ratio_list))
        print(f"\n# median downstream/upstream cohesin occupancy = {med:.2f} "
              f"(<1 => cohesin enriched on the enhancer/loading side, "
              f"per Kim 2026 Fig 2d; this is loading+anchoring, not pushing)")

    # ---- summary: directional pushing (the clean directionality verdict) -----
    if signed:
        ms = float(np.mean(signed))
        verdict = ("MATCH: legs pushed in transcription direction (moving barrier, "
                   "Banigan&Mirny 2022 / Kim 2026 Fig 2)"
                   if ms > 0 else "no transcription-directed push")
        print(f"\n## 2. Directional pushing (clean test): mean signed cohesin-leg "
              f"displacement\n#   inside ACTIVE gene bodies = {ms:+.4f} sites/tick "
              f"(txn-dir frame)  [{verdict}]")

    if len(contact_list) >= 3:
        c = np.array(contact_list)
        # Spearman via rank correlation
        rc = np.argsort(np.argsort(c[:, 0])); rn = np.argsort(np.argsort(c[:, 1]))
        rho = float(np.corrcoef(rc, rn)[0, 1])
        print("\n## 3. Cooperation: E-P contact frequency vs nascent output")
        print("{:>5} {:>14} {:>10}".format("gene", "EP_contact_frq", "nascent"))
        for (cf, na), row in zip(contact_list, genes):
            print(f"{int(row['gene_id']):>5} {cf:>14.3f} {na:>10.3f}")
        print(f"\n# Spearman rho(E-P contact, nascent) = {rho:.2f} "
              f"({'positive => cohesin loop drives output (MATCH)' if rho > 0 else 'non-positive'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
