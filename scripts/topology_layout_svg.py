"""Render a publication-quality 1D topology layout from a pipeline config to SVG.

Reads a loop-extrusion YAML config, re-derives the same CTCF / gene /
enhancer / promoter / TAD / E-P annotations the interactive viewer uses
(:func:`viewer.derive_annotations`), and draws them as a still, vector
figure -- one horizontal track per chain -- a print-ready version of the
viewer's "1D Lattice" panel.

No trajectory is needed: annotations come straight from the ``lef`` stage
topology plugin, so this works on a bare config without any simulation
output. Cohesin arcs (per-frame, dynamic) are not drawn; the static
counterpart shown here is the enhancer->promoter pairing as an arc.

The output is clean, layered, Illustrator-editable SVG: a colourblind-safe
(Okabe-Ito) palette, gene-direction arrowheads, round-number genomic axis in
kb, and greedy label-collision avoidance.

Usage::

    PYTHONPATH=. micromamba run -n openmm \\
        python scripts/topology_layout_svg.py configs/config1.yaml [out.svg]
        [--site-start S] [--site-end E] [--width W] [--bp-per-site BP]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from polychrom.pipelines.loop_extrusion.config import load_config
from polychrom.pipelines.loop_extrusion.viewer import derive_annotations

# Reuse the trajectory measurement (single source of truth).
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from nascent_rna_abundance import (  # noqa: E402
    aggregate_by_base, ensure_h5, measure_per_gene, resolve_h5,
)

# --------------------------------------------------------------------------- #
# Palette (Okabe-Ito, colourblind-safe) + semantic colours.
# --------------------------------------------------------------------------- #
C_PROMOTER = "#009E73"      # bluish green
C_ENH_FILL = "#E69F00"      # orange
C_ENH_STROKE = "#B87A00"
C_CTCF = "#3f3f46"          # dark slate
C_BACKBONE = "#9aa3af"
C_TAD_BAND = ["#f6f8fa", "#eceff3"]   # alternating TAD fill
C_TAD_EDGE = "#c3cbd6"
C_AXIS = "#6b7280"
C_TEXT = "#1f2933"
C_TEXT_MUTE = "#6b7280"
PALETTE = [                  # Okabe-Ito categorical (skip black)
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#999999",
]

# Typography.
FONT = "Helvetica, Arial, sans-serif"
CHAR_W = 5.4                 # approx advance @ 9px, for label-collision width

# Hansen & Yang 2024 MSD calibration: 1 MD integrator step ~= 0.0063 s, and
# 1 LEF tick == 1 MD block == md_steps_per_block steps (see config header /
# scripts/nascent_rna_abundance.py). Lets us print per-gene kinetics in real time.
MD_STEP_SECONDS = 0.0063


def _palette(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _gene_stats(cfg, bp_per_site: int) -> Tuple[List[dict], float]:
    """Derive per-gene kinetics + regulatory class straight from the config.

    Returns ``(rows, tick_seconds)``. Each row is a dict of display strings.
    No trajectory needed: rates come from the YAML gene specs converted to real
    time via the tick calibration (``md_steps_per_block * 0.0063 s``).
    """
    tick_s = cfg.polymer.md_steps_per_block * MD_STEP_SECONDS
    kb_per_site = bp_per_site / 1000.0
    specs = cfg.lef.topology_kwargs.get("genes", []) or []
    rows: List[dict] = []
    for gid, s in enumerate(specs):
        tss, tes = int(s["tss"]), int(s["tes"])
        length_kb = abs(tes - tss) * kb_per_site
        strand = "+" if tes > tss else "−"
        enh = s.get("enhancers") or (
            [s["enhancer_pos"]] if s.get("enhancer_pos") is not None else []
        )
        ep_kb = sorted(abs(int(e) - tss) * kb_per_site for e in enh)
        if not ep_kb:
            ep_str = "—"
        elif len(ep_kb) == 1:
            ep_str = f"{ep_kb[0]:.0f}"
        else:
            ep_str = f"{ep_kb[0]:.0f}–{ep_kb[-1]:.0f}"

        requires = bool(s.get("requires_enhancer", False))
        esp = float(s.get("elongation_step_prob", 1.0))
        kb_min = esp * kb_per_site / tick_s * 60.0
        transit_min = length_kb / kb_min if kb_min > 0 else float("nan")
        prp = float(s.get("pause_release_prob", 1.0))
        pause_min = (tick_s / prp / 60.0) if prp > 0 else float("nan")
        tp = float(s.get("termination_prob", 1.0))
        dwell_min = (tick_s / tp / 60.0) if tp > 0 else float("nan")

        if not requires:
            cls = "constitutive / short-range"
        elif ep_kb and max(ep_kb) >= 100:
            cls = "long-range cohesin-dep"
        else:
            cls = "short-range cohesin-dep"
        logic = str(s.get("enhancer_logic", "")) if len(enh) > 1 else ""
        rows.append({
            "gid": gid, "strand": strand, "len": f"{length_kb:.0f}",
            "ep": ep_str, "n_enh": str(len(enh)) + (f" {logic}" if logic else ""),
            "cls": cls, "kbmin": f"{kb_min:.1f}", "transit": f"{transit_min:.0f}",
            "pause": f"{pause_min:.1f}", "dwell": f"{dwell_min:.1f}",
        })
    return rows, tick_s


def _arc_height(span_px: float) -> float:
    return min(88.0, 12.0 + (span_px ** 0.5) * 2.8)


def _nice_step(span: float, target_ticks: int = 8) -> float:
    """Round 'nice' tick step (1/2/2.5/5 x 10^k) for ~target_ticks ticks."""
    if span <= 0:
        return 1.0
    raw = span / max(1, target_ticks)
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 10 ** -2
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _fmt_kb(bp: float) -> str:
    """Format a base-pair coordinate compactly in kb / Mb."""
    if bp >= 1_000_000:
        v = bp / 1_000_000
        return (f"{v:.1f}".rstrip("0").rstrip(".")) + " Mb"
    v = bp / 1000
    return (f"{v:.1f}".rstrip("0").rstrip(".")) + " kb"


def _esc(s) -> str:
    return escape(str(s))


# --------------------------------------------------------------------------- #
# Collision avoidance: assign each labelled point to a stacking level so that
# adjacent labels never overlap horizontally.
# --------------------------------------------------------------------------- #
def _assign_levels(items: List[Tuple[float, float]], n_levels: int = 3) -> List[int]:
    """``items`` = [(center_x, half_width)] sorted by x; return level per item."""
    last_right = [-1e9] * n_levels
    levels: List[int] = []
    for cx, hw in items:
        placed = 0
        for lvl in range(n_levels):
            if cx - hw >= last_right[lvl] + 3:
                placed = lvl
                break
        else:
            placed = n_levels - 1
        last_right[placed] = cx + hw
        levels.append(placed)
    return levels


# --------------------------------------------------------------------------- #
# Per-chain grouping of absolute-coordinate annotations into chain-local rows.
# --------------------------------------------------------------------------- #
def _group_by_chain(
    chain_length: int,
    ctcf_sites: List[int],
    elements: List[dict],
    eps: List[dict],
    genes: List[dict],
    tads: List[dict],
    chains: List[int],
) -> Dict[int, dict]:
    """Bucket every annotation into chain-local coordinates per chain index."""
    out: Dict[int, dict] = {
        c: {"ctcf": [], "elements": [], "eps": [], "genes": [], "tads": []}
        for c in chains
    }

    def loc(p: int) -> Tuple[int, int]:
        return p // chain_length, p % chain_length

    for s in ctcf_sites:
        c, l = loc(s)
        if c in out:
            out[c]["ctcf"].append(l)
    for el in elements:
        c, l = loc(int(el["position"]))
        if c in out:
            out[c]["elements"].append({**el, "position": l})
    for g in genes:
        c, tss = loc(int(g["tss"]))
        if c in out:
            out[c]["genes"].append({**g, "tss": tss, "tes": int(g["tes"]) % chain_length})
    for pair in eps:
        c, e = loc(int(pair["e"]))
        c2, p = loc(int(pair["p"]))
        if c == c2 and c in out:
            out[c]["eps"].append({**pair, "e": e, "p": p})
    for t in tads:
        c = int(t["chain"])
        if c in out:
            out[c]["tads"].append({
                **t,
                "start": int(t["start"]) - c * chain_length,
                "end": int(t["end"]) - c * chain_length,
            })
    return out


# --------------------------------------------------------------------------- #
# SVG assembly.
# --------------------------------------------------------------------------- #
def build_svg(
    config_path: str,
    site_start: int | None = None,
    site_end: int | None = None,
    width: int = 1480,
    bp_per_site: int = 1000,
    run_path: str | None = None,
) -> str:
    """Build the SVG document string for one config's topology layout.

    When ``run_path`` is given (a run dir or ``LEFPositions.h5``) -- or a default
    ``trajectory/LEFPositions.h5`` exists -- per-gene transcription outputs are
    MEASURED from the trajectory and added to the stats table. A missing h5 is
    generated on the fly by running the (numpy-only) lef stage.
    """
    cfg = load_config(config_path)
    lef_cfg = cfg.lef
    chain_length = lef_cfg.chain_length

    ctcf_sites, elements, eps, genes, tads = derive_annotations(lef_cfg)
    stat_rows, tick_s = _gene_stats(cfg, bp_per_site)

    # Measure actual transcription outputs from the 1D trajectory (generate it
    # if absent). Merged into stat_rows by base gene id.
    measured: Dict[int, dict] = {}
    h5_path = resolve_h5(run_path)
    try:
        ensure_h5(cfg, h5_path)
        gid_rows, meta = measure_per_gene(h5_path, cfg)
        measured = aggregate_by_base(gid_rows, meta["num_chains"])
    except Exception as exc:  # measurement is best-effort; layout still renders
        print(f"[topology-svg] WARNING: could not measure trajectory ({exc}); "
              f"rendering config-derived stats only")
    for row in stat_rows:
        m = measured.get(row["gid"])
        if m:
            row["m_nascent"] = f"{m['nascent_allele']:.2f}"
            row["m_per_h"] = f"{m['per_hour_allele']:.1f}"
            row["m_pct_pause"] = f"{m['pct_paused_mean']:.0f}"
            row["m_kbmin"] = f"{m['kb_min_mean']:.2f}"
        else:
            row["m_nascent"] = row["m_per_h"] = row["m_pct_pause"] = row["m_kbmin"] = "—"

    # Display window -> set of chains to render.
    full_n = lef_cfg.num_sites
    start = (
        site_start if site_start is not None
        else (cfg.viewer.site_start if cfg.viewer.site_start is not None else 0)
    )
    end = (
        site_end if site_end is not None
        else (cfg.viewer.site_end if cfg.viewer.site_end is not None else full_n)
    )
    start, end = int(start), int(end)
    chains = list(range(start // chain_length, (max(end - 1, start)) // chain_length + 1))
    chains = [c for c in chains if 0 <= c < lef_cfg.num_chains] or [0]

    grouped = _group_by_chain(
        chain_length, ctcf_sites, elements, eps, genes, tads, chains
    )

    # --- geometry ---------------------------------------------------------- #
    pad_l, pad_r = 96, 40
    head_h = 86                  # title + legend band
    row_h = 168                  # per-chain row
    axis_h = 60
    plot_w = width - pad_l - pad_r
    # Per-gene stats table band below the axis (title + header + one row/gene).
    table_h = (44 + 18 + 16 * len(stat_rows) + 12) if stat_rows else 0
    height = head_h + row_h * len(chains) + axis_h + table_h
    base_off = 96                # backbone y within a row (room for arcs above)

    def xs(local: int) -> float:
        return pad_l + local * plot_w / max(1, chain_length)

    P: List[str] = []
    P.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" font-family="{FONT}">'
    )
    P.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    # Reusable gene-direction arrowhead.
    P.append(
        '<defs><marker id="ah" markerWidth="7" markerHeight="7" refX="5.5" '
        'refY="3" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="context-stroke"/></marker></defs>'
    )

    # --- title ------------------------------------------------------------- #
    P.append(
        f'<text x="{pad_l}" y="34" font-size="17" font-weight="700" '
        f'fill="{C_TEXT}">Genome topology &#8212; {_esc(Path(config_path).stem)}</text>'
    )
    sub = (
        f"{lef_cfg.num_chains} chain(s) &#215; {chain_length} sites "
        f"({_fmt_kb(chain_length * bp_per_site)} @ {bp_per_site} bp/site) "
        f"&#183; {len(genes)} genes &#183; {len(tads)} TADs "
        f"&#183; 1 tick = {tick_s:.0f} s ({cfg.polymer.md_steps_per_block}"
        f"&#215;{MD_STEP_SECONDS}s)"
    )
    P.append(
        f'<text x="{pad_l}" y="53" font-size="11" fill="{C_TEXT_MUTE}">{sub}</text>'
    )

    # --- legend ------------------------------------------------------------ #
    lx, ly = pad_l, 74
    P.append(
        f'<line x1="{lx}" y1="{ly - 3}" x2="{lx + 18}" y2="{ly - 3}" '
        f'stroke="{C_CTCF}" stroke-width="2"/>'
        f'<text x="{lx + 24}" y="{ly}" font-size="10.5" fill="{C_TEXT}">CTCF</text>'
    )
    lx += 78
    P.append(
        f'<path d="M {lx} {ly - 8} L {lx + 7} {ly - 3} L {lx} {ly + 2}" '
        f'fill="none" stroke="{C_PROMOTER}" stroke-width="2"/>'
        f'<text x="{lx + 14}" y="{ly}" font-size="10.5" fill="{C_TEXT}">promoter (TSS)</text>'
    )
    lx += 130
    P.append(
        f'<ellipse cx="{lx + 7}" cy="{ly - 3}" rx="7" ry="4.5" fill="{C_ENH_FILL}" '
        f'stroke="{C_ENH_STROKE}" stroke-width="1.3"/>'
        f'<text x="{lx + 20}" y="{ly}" font-size="10.5" fill="{C_TEXT}">enhancer</text>'
    )
    lx += 92
    P.append(
        f'<line x1="{lx}" y1="{ly - 3}" x2="{lx + 18}" y2="{ly - 3}" '
        f'stroke="{_palette(0)}" stroke-width="4" stroke-linecap="round" '
        f'marker-end="url(#ah)"/>'
        f'<text x="{lx + 26}" y="{ly}" font-size="10.5" fill="{C_TEXT}">gene body</text>'
    )
    lx += 96
    P.append(
        f'<path d="M {lx} {ly - 1} Q {lx + 11} {ly - 13} {lx + 22} {ly - 1}" '
        f'fill="none" stroke="{_palette(0)}" stroke-width="1.6" stroke-dasharray="4 3"/>'
        f'<text x="{lx + 28}" y="{ly}" font-size="10.5" fill="{C_TEXT}">E&#8211;P pair</text>'
    )

    # --- per-chain rows ---------------------------------------------------- #
    for ri, c in enumerate(chains):
        row_top = head_h + ri * row_h
        base = row_top + base_off
        g = grouped[c]

        # chain label
        P.append(
            f'<text x="{pad_l - 12}" y="{base + 4}" text-anchor="end" '
            f'font-size="11" font-weight="600" fill="{C_TEXT_MUTE}">chain {c}</text>'
        )

        # TAD bands
        band_top, band_bot = base - 52, base + 52
        for bi, t in enumerate(g["tads"]):
            x0, x1 = xs(int(t["start"])), xs(int(t["end"]))
            P.append(
                f'<rect x="{x0:.1f}" y="{band_top}" width="{max(0.0, x1 - x0):.1f}" '
                f'height="{band_bot - band_top}" fill="{C_TAD_BAND[bi % 2]}" '
                f'stroke="{C_TAD_EDGE}" stroke-width="0.6"/>'
            )
            P.append(
                f'<text x="{(x0 + x1) / 2:.1f}" y="{band_top - 5}" text-anchor="middle" '
                f'font-size="9" fill="{C_TEXT_MUTE}">{_esc(t["label"])}</text>'
            )
            # TAD edge positions in kb.
            P.append(
                f'<text x="{x0:.1f}" y="{band_bot + 11}" text-anchor="middle" '
                f'font-size="8" fill="{C_TAD_EDGE}">'
                f'{_fmt_kb(int(t["start"]) * bp_per_site)}</text>'
            )
            P.append(
                f'<text x="{x1:.1f}" y="{band_bot + 11}" text-anchor="middle" '
                f'font-size="8" fill="{C_TAD_EDGE}">'
                f'{_fmt_kb(int(t["end"]) * bp_per_site)}</text>'
            )

        # backbone
        P.append(
            f'<line x1="{xs(0):.1f}" y1="{base}" x2="{xs(chain_length):.1f}" '
            f'y2="{base}" stroke="{C_BACKBONE}" stroke-width="2.2"/>'
        )

        # E-P arcs (above backbone)
        for i, pair in enumerate(g["eps"]):
            e, p = int(pair["e"]), int(pair["p"])
            x0, x1 = xs(min(e, p)), xs(max(e, p))
            mid = (x0 + x1) / 2
            h = _arc_height(abs(x1 - x0))
            # Colour by gene so shadow/super-enhancers of one gene share a hue.
            arc_color = _palette(int(pair.get("geneId", i)))
            span_bp = int(pair.get("genomic", abs(p - e))) * bp_per_site
            P.append(
                f'<path d="M {x0:.1f} {base - 1} Q {mid:.1f} {base - h:.1f} '
                f'{x1:.1f} {base - 1}" fill="none" stroke="{arc_color}" '
                f'stroke-width="1.6" stroke-dasharray="4 3" opacity="0.85">'
                f'<title>{_esc(pair.get("label", "E-P"))} '
                f'(genomic {_fmt_kb(span_bp)})</title></path>'
            )
            # Arc-length label at the apex.
            P.append(
                f'<text x="{mid:.1f}" y="{base - h - 4:.1f}" text-anchor="middle" '
                f'font-size="9" fill="{arc_color}">{_fmt_kb(span_bp)}</text>'
            )

        # gene bodies as directional arrows (below backbone)
        gy = base + 16
        for gi, gene in enumerate(g["genes"]):
            tss, tes = int(gene["tss"]), int(gene["tes"])
            color = _palette(int(gene.get("geneId", gi)))
            P.append(
                f'<line x1="{xs(tss):.1f}" y1="{gy}" x2="{xs(tes):.1f}" y2="{gy}" '
                f'stroke="{color}" stroke-width="4" stroke-linecap="round" '
                f'opacity="0.85" marker-end="url(#ah)">'
                f'<title>{_esc(gene.get("label", "gene"))} '
                f'TSS {tss} &#8594; TES {tes}</title></line>'
            )

        # CTCF ticks
        for s in g["ctcf"]:
            x = xs(int(s))
            P.append(
                f'<line x1="{x:.1f}" y1="{base - 8}" x2="{x:.1f}" y2="{base + 8}" '
                f'stroke="{C_CTCF}" stroke-width="2"><title>CTCF @ site {s} '
                f'({_fmt_kb(int(s) * bp_per_site)})</title></line>'
            )
            # CTCF boundary position in kb, above the tick.
            P.append(
                f'<text x="{x:.1f}" y="{base - 12}" text-anchor="middle" '
                f'font-size="8" fill="{C_CTCF}">{_fmt_kb(int(s) * bp_per_site)}</text>'
            )

        # promoters / enhancers + collision-avoided labels
        label_items: List[Tuple[float, dict]] = []
        for el in g["elements"]:
            x = xs(int(el["position"]))
            if el["type"] == "enhancer":
                P.append(
                    f'<ellipse cx="{x:.1f}" cy="{base}" rx="7" ry="4.5" '
                    f'fill="{C_ENH_FILL}" stroke="{C_ENH_STROKE}" stroke-width="1.4"/>'
                )
                fill = C_ENH_STROKE
            else:
                d = -1 if el.get("direction") == -1 else 1
                P.append(
                    f'<path d="M {x:.1f} {base - 9} L {x + 6 * d:.1f} {base} '
                    f'L {x:.1f} {base + 9}" fill="none" stroke="{C_PROMOTER}" '
                    f'stroke-width="2.2" stroke-linejoin="round"/>'
                )
                fill = C_PROMOTER
            if el.get("label"):
                label_items.append((x, {"text": str(el["label"]), "fill": fill}))

        label_items.sort(key=lambda it: it[0])
        levels = _assign_levels(
            [(x, len(d["text"]) * CHAR_W / 2) for x, d in label_items]
        )
        for (x, d), lvl in zip(label_items, levels):
            ly2 = base + 30 + lvl * 13
            P.append(
                f'<text x="{x:.1f}" y="{ly2}" text-anchor="middle" font-size="9" '
                f'fill="{d["fill"]}">{_esc(d["text"])}</text>'
            )

    # --- shared genomic axis (chain-local, in bp -> kb) -------------------- #
    axis_y = head_h + row_h * len(chains) + 14
    P.append(
        f'<line x1="{xs(0):.1f}" y1="{axis_y}" x2="{xs(chain_length):.1f}" '
        f'y2="{axis_y}" stroke="{C_AXIS}" stroke-width="1.1"/>'
    )
    step_bp = _nice_step(chain_length * bp_per_site)
    step_sites = max(1, int(round(step_bp / bp_per_site)))
    site = 0
    while site <= chain_length:
        x = xs(site)
        P.append(
            f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 5}" '
            f'stroke="{C_AXIS}" stroke-width="1"/>'
        )
        P.append(
            f'<text x="{x:.1f}" y="{axis_y + 18}" text-anchor="middle" '
            f'font-size="9.5" fill="{C_TEXT_MUTE}">{_fmt_kb(site * bp_per_site)}</text>'
        )
        site += step_sites
    P.append(
        f'<text x="{xs(chain_length):.1f}" y="{axis_y + 33}" text-anchor="end" '
        f'font-size="9.5" fill="{C_TEXT_MUTE}">genomic position</text>'
    )

    # --- per-gene stats table (config kinetics + MEASURED transcription) --- #
    if stat_rows:
        tbl_top = head_h + row_h * len(chains) + axis_h
        any_measured = any(row.get("m_nascent", "—") != "—" for row in stat_rows)
        # (header, x-offset-from-pad_l, key, is_measured)
        cols = [
            ("gene", 0, "gid", False), ("str", 70, "strand", False),
            ("len kb", 104, "len", False), ("E&#8211;P kb", 156, "ep", False),
            ("enh", 214, "n_enh", False), ("regulatory class", 272, "cls", False),
            # measured from trajectory
            ("Pol&#8201;II", 512, "m_nascent", True),
            ("mRNA/h", 576, "m_per_h", True),
            ("%paused", 648, "m_pct_pause", True),
            ("kb/min", 724, "m_kbmin", True),
            # config-predicted kinetics
            ("transit&#8201;m", 794, "transit", False),
            ("pause&#8201;m", 880, "pause", False),
            ("TES&#8201;m", 956, "dwell", False),
        ]
        src = ("measured from 1D trajectory" if any_measured
               else "config-derived only (no trajectory)")
        P.append(
            f'<text x="{pad_l}" y="{tbl_top + 16}" font-size="12" font-weight="700" '
            f'fill="{C_TEXT}">Per-gene transcription output &#38; regulatory class '
            f'<tspan font-weight="400" fill="{C_TEXT_MUTE}">'
            f'(1 tick = {tick_s:.0f} s; Pol&#8201;II/mRNA&#8201;h/%paused/kb&#8201;min '
            f'{_esc(src)}, per allele; transit/pause/TES = config prediction)'
            f'</tspan></text>'
        )
        hy = tbl_top + 38
        for hdr, dx, _, is_meas in cols:
            P.append(
                f'<text x="{pad_l + dx}" y="{hy}" font-size="9.5" font-weight="700" '
                f'fill="{C_PROMOTER if is_meas else C_TEXT_MUTE}">{hdr}</text>'
            )
        P.append(
            f'<line x1="{pad_l}" y1="{hy + 4}" x2="{width - pad_r}" y2="{hy + 4}" '
            f'stroke="{C_TAD_EDGE}" stroke-width="0.8"/>'
        )
        for ri, row in enumerate(stat_rows):
            ry = hy + 18 + ri * 16
            color = _palette(int(row["gid"]))
            P.append(
                f'<rect x="{pad_l}" y="{ry - 8}" width="10" height="10" rx="2" '
                f'fill="{color}" opacity="0.85"/>'
                f'<text x="{pad_l + 15}" y="{ry}" font-size="9.5" fill="{C_TEXT}">'
                f'{row["gid"]}</text>'
            )
            for hdr, dx, key, is_meas in cols[1:]:
                fill = color if key == "cls" else (C_PROMOTER if is_meas else C_TEXT)
                weight = ' font-weight="600"' if is_meas else ""
                P.append(
                    f'<text x="{pad_l + dx}" y="{ry}" font-size="9.5"{weight} '
                    f'fill="{fill}">{_esc(row[key])}</text>'
                )

    P.append("</svg>")
    return "\n".join(P)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="pipeline YAML config")
    ap.add_argument(
        "output", nargs="?", default=None,
        help="output SVG path (default: <config-stem>_topology.svg)",
    )
    ap.add_argument("--site-start", type=int, default=None,
                    help="override display window start site")
    ap.add_argument("--site-end", type=int, default=None,
                    help="override display window end site")
    ap.add_argument("--width", type=int, default=1480, help="SVG width in px")
    ap.add_argument("--bp-per-site", type=int, default=1000,
                    help="base pairs per lattice site for the genomic axis")
    ap.add_argument("--run", default=None,
                    help="run dir or LEFPositions.h5 to MEASURE transcription "
                         "outputs from (generated via the lef stage if missing; "
                         "default: trajectory/LEFPositions.h5)")
    args = ap.parse_args(argv)

    svg = build_svg(
        args.config,
        site_start=args.site_start,
        site_end=args.site_end,
        width=args.width,
        bp_per_site=args.bp_per_site,
        run_path=args.run,
    )

    out = (
        Path(args.output) if args.output
        else Path(args.config).with_name(Path(args.config).stem + "_topology.svg")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"[topology-svg] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
