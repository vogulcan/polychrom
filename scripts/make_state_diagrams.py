#!/usr/bin/env python3
"""Generate publication-quality SVG state-machine diagrams for the loop-extrusion
RNAPII / cohesin model. Pure stdlib, no Mermaid, no external deps.

Outputs (docs/figs/):
  rnapii_state_machine.svg        - RNAPII POISED->PAUSED->ELONGATING<->STALLED->TERMINATING
  cohesin_state_machine.svg       - cohesin leg EXTRUDING<->STALLED / CTCF-captured
  rnapii_cohesin_interaction.svg  - collision resolution + pause-release coupling
  lesion_state_machine.svg        - DNA-lesion PRE-RECOGNITION->REPAIR->repaired,
                                    type A/B assignment + cohesin/RNAPII interaction matrix

The transition labels and probabilities mirror the engine
(polychrom/pipelines/loop_extrusion/plugins/{rnapii,lef_dynamics,lesions}.py).
RNAPII state fills match the trajectory viewer palette (_viewer_template.py).

Run:  python scripts/make_state_diagrams.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "figs"

# Filled by load_params() when a config is passed; otherwise labels fall back to
# the symbolic parameter name so the figures stay valid without a config.
PARAMS: dict | None = None


def val(key, symbol, fmt="{:g}"):
    """Numeric value for ``key`` from the loaded config, else ``symbol``."""
    if PARAMS and PARAMS.get(key) is not None:
        return fmt.format(PARAMS[key])
    return symbol


def load_params(path: str) -> dict:
    """Pull the RNAPII/cohesin mechanic values from a pipeline config."""
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from polychrom.pipelines.loop_extrusion.config import load_config

    cfg = load_config(path)
    tk = dict(cfg.lef.topology_kwargs or {})
    g = tk.get
    blk = g("rnapii_block_prob")
    ls = getattr(cfg.lef, "lifetime_stalled", None)
    lc = getattr(cfg.lef, "lifetime_ctcf", None)
    return {
        "stall": g("rnapii_stall_prob"),
        "push": g("rnapii_push_prob"),
        "headon": g("rnapii_headon_push_prob"),
        "b_poised": g("rnapii_poised_block_prob"),
        "b_paused": g("rnapii_paused_block_prob", blk),
        "b_elong": g("rnapii_elongating_block_prob", blk),
        "b_term": g("rnapii_terminating_block_prob"),
        "restraint": g("rnapii_pause_cohesin_restraint"),
        "window": g("rnapii_pause_restraint_window"),
        "lifetime": getattr(cfg.lef, "lifetime", None),
        "lifetime_stalled": ls,
        "lifetime_rnapii_stalled": g("lifetime_rnapii_stalled", ls),
        "lifetime_ctcf": lc if lc is not None else getattr(cfg.lef, "lifetime", None),
        # Lesion (UV-damage) machine knobs (config4); absent (None) for config1/2/3.
        "lesion_spacing": g("lesion_spacing"),
        "lesion_type_a_prob": g("lesion_type_a_prob"),
        "lesion_prerecognition_ticks": g("lesion_prerecognition_ticks"),
        "lesion_repair_ticks": g("lesion_repair_ticks"),
        "lesion_block_prob": g("lesion_block_prob"),
        "lesion_tad_size_exponent": g("lesion_tad_size_exponent"),
        "lesion_tad_repair_exponent": g("lesion_tad_repair_exponent"),
        "src": Path(path).name,
    }

# --- palette -----------------------------------------------------------------
# RNAPII state fills (viewer RNAPII_COLORS) + matched darker strokes.
RNAPII = {
    "POISED":      ("#64748b", "#475569"),
    "PAUSED":      ("#f59e0b", "#b45309"),
    "ELONGATING":  ("#be185d", "#831843"),
    "TERMINATING": ("#7c3aed", "#5b21b6"),
    "STALLED":     ("#dc2626", "#991b1b"),
}
COHESIN = {
    "EXTRUDING":     ("#0e7490", "#155e75"),
    "STALLED":       ("#b45309", "#7c2d12"),
    "CTCF_CAPTURED": ("#6d28d9", "#4c1d95"),
}
# Lesion state fills: PRE-RECOGNITION (amber) -> REPAIR (teal).
LESION = {
    "PRE":    ("#d97706", "#92400e"),
    "REPAIR": ("#0891b2", "#155e75"),
}
INK = "#1f2937"        # primary text / generic stroke
MUTED = "#6b7280"      # captions
FONT = ("font-family=\"Helvetica Neue, Helvetica, Arial, sans-serif\"")

# --- low-level SVG helpers ---------------------------------------------------


class SVG:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h = w, h
        self.title = title
        self.body: list[str] = []

    def add(self, s: str) -> None:
        self.body.append(s)

    def text(self, x, y, s, size=13, anchor="middle", weight="normal",
             fill=INK, italic=False, spacing=None):
        st = f' letter-spacing="{spacing}"' if spacing else ""
        it = ' font-style="italic"' if italic else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"'
            f'{it}{st}>{escape(s)}</text>'
        )

    def node(self, cx, cy, w, h, label, fill, stroke, sub=None, code=None):
        x, y = cx - w / 2, cy - h / 2
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="12" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        ty = cy + (4 if not sub else -3)
        self.text(cx, ty, label, size=16, weight="700", fill="#ffffff",
                  spacing="0.4")
        if sub:
            self.text(cx, cy + 15, sub, size=11, fill="#f8fafc")
        if code is not None:
            self.add(
                f'<circle cx="{x + w - 16:.1f}" cy="{y + 16:.1f}" r="11" '
                f'fill="#ffffff" fill-opacity="0.9" stroke="{stroke}"/>'
            )
            self.text(x + w - 16, y + 20, str(code), size=12, weight="700",
                      fill=stroke)
        return (cx, cy, w, h)

    def _pill(self, x, y, s, size=12, fill=INK):
        w = len(s) * size * 0.58 + 12
        self.add(
            f'<rect x="{x - w / 2:.1f}" y="{y - size / 2 - 4:.1f}" '
            f'width="{w:.1f}" height="{size + 8}" rx="{(size + 8) / 2:.1f}" '
            f'fill="#ffffff" stroke="#e5e7eb" stroke-width="1"/>'
        )
        self.text(x, y + size * 0.35, s, size=size, fill=fill)

    def edge(self, p0, p1, label=None, curve=0.0, dash=False, label_size=12,
             label_fill=INK, label_at=None):
        """Arrow from point p0 to p1. ``curve`` bows the path (quadratic)."""
        x0, y0 = p0
        x1, y1 = p1
        d = ' stroke-dasharray="5,4"' if dash else ""
        if curve:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy) or 1
            nx, ny = -dy / ln, dx / ln
            qx, qy = mx + nx * curve, my + ny * curve
            self.add(
                f'<path d="M{x0:.1f},{y0:.1f} Q{qx:.1f},{qy:.1f} {x1:.1f},{y1:.1f}" '
                f'fill="none" stroke="{INK}" stroke-width="1.8" '
                f'marker-end="url(#arrow)"{d}/>'
            )
            lx, ly = qx, qy
        else:
            self.add(
                f'<path d="M{x0:.1f},{y0:.1f} L{x1:.1f},{y1:.1f}" fill="none" '
                f'stroke="{INK}" stroke-width="1.8" marker-end="url(#arrow)"{d}/>'
            )
            lx, ly = (x0 + x1) / 2, (y0 + y1) / 2
        if label:
            if label_at is not None:
                lx, ly = label_at
            self._pill(lx, ly, label, size=label_size, fill=label_fill)

    def self_loop(self, node, side, label, label_size=11):
        cx, cy, w, h = node
        if side == "left":
            ax, ay = cx - w / 2, cy - 12
            bx, by = cx - w / 2, cy + 12
            qx, qy = cx - w / 2 - 64, cy
            lx, ly = qx - 4, cy
            anch = "end"
        else:  # right
            ax, ay = cx + w / 2, cy - 12
            bx, by = cx + w / 2, cy + 12
            qx, qy = cx + w / 2 + 64, cy
            lx, ly = qx + 4, cy
            anch = "start"
        self.add(
            f'<path d="M{ax:.1f},{ay:.1f} Q{qx:.1f},{qy:.1f} {bx:.1f},{by:.1f}" '
            f'fill="none" stroke="{INK}" stroke-width="1.6" '
            f'marker-end="url(#arrow)"/>'
        )
        self.text(lx, ly + 3, label, size=label_size, anchor=anch, fill=MUTED)

    def start_dot(self, x, y):
        self.add(f'<circle cx="{x}" cy="{y}" r="8" fill="{INK}"/>')

    def end_dot(self, x, y):
        self.add(f'<circle cx="{x}" cy="{y}" r="10" fill="none" '
                 f'stroke="{INK}" stroke-width="2"/>')
        self.add(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{INK}"/>')

    def render(self) -> str:
        defs = (
            '<defs>'
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
            '</defs>'
        )
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
            f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
        )
        bg = f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>'
        title = (f'<text x="{self.w/2:.0f}" y="34" {FONT} font-size="19" '
                 f'font-weight="700" text-anchor="middle" fill="{INK}">'
                 f'{escape(self.title)}</text>')
        return head + defs + bg + title + "".join(self.body) + "</svg>"

    def save(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(self.render())
        print(f"wrote {OUT / name}")


def edge_pt(node, where):
    cx, cy, w, h = node
    return {
        "top": (cx, cy - h / 2), "bottom": (cx, cy + h / 2),
        "left": (cx - w / 2, cy), "right": (cx + w / 2, cy),
    }[where]


# --- Figure 1: RNAPII --------------------------------------------------------

def fig_rnapii():
    s = SVG(940, 1000, "RNA Polymerase II state machine")
    NW, NH = 220, 64
    cx = 330
    ys = {"P": 130, "PA": 290, "EL": 480, "TE": 680}
    s.start_dot(cx, 70)
    poised = s.node(cx, ys["P"], NW, NH, "POISED", *RNAPII["POISED"],
                    sub="at TSS, pre-initiation", code=0)
    paused = s.node(cx, ys["PA"], NW, NH, "PAUSED", *RNAPII["PAUSED"],
                    sub="promoter-proximal", code=1)
    elong = s.node(cx, ys["EL"], NW, NH, "ELONGATING", *RNAPII["ELONGATING"],
                   sub="productive", code=2)
    stall = s.node(cx + 360, ys["EL"], NW, NH, "STALLED", *RNAPII["STALLED"],
                   sub="blocked this tick", code=4)
    term = s.node(cx, ys["TE"], NW, NH, "TERMINATING", *RNAPII["TERMINATING"],
                  sub="dwell at TES", code=3)
    s.end_dot(cx, 800)

    s.edge((cx, 78), edge_pt(poised, "top"), "load_prob × enh_factor*")
    s.edge(edge_pt(poised, "bottom"), edge_pt(paused, "top"), "initiation_prob")
    s.self_loop(paused, "left", "gate closed / dice fail")
    s.edge(edge_pt(paused, "bottom"), edge_pt(elong, "top"), "release_prob ‡")
    s.self_loop(elong, "left", "step ✓ / no dice")
    # ELONGATING <-> STALLED (offset the two horizontal arrows)
    s.edge((elong[0] + NW / 2, ys["EL"] - 12), (stall[0] - NW / 2, ys["EL"] - 12),
           "rolled dice + blocked", label_size=11)
    s.edge((stall[0] - NW / 2, ys["EL"] + 12), (elong[0] + NW / 2, ys["EL"] + 12),
           "step ✓", label_size=11)
    s.self_loop(stall, "right", "still blocked")
    s.edge(edge_pt(elong, "bottom"), edge_pt(term, "top"), "pos = TES")
    s.edge((stall[0], stall[1] + NH / 2), (term[0] + NW / 2 - 20, term[1] - NH / 2),
           "pos = TES", curve=70, label_size=11, label_at=(600, 690))
    s.edge(edge_pt(term, "bottom"), (cx, 790), "termination_prob")

    # A lesion-stalled Pol II is evicted (not resumed) once its lesion is repaired.
    s.end_dot(stall[0], 800)
    s.edge((stall[0], stall[1] + NH / 2), (stall[0], 790),
           "lesion repaired → evicted", label_size=11)

    # footnotes
    fy = 860
    s.text(40, fy, "*  load_prob × enh_factor   (enh_factor applied only if "
           "load_requires_enhancer)", size=12, anchor="start", fill=MUTED)
    s.text(40, fy + 22, "‡  release_prob = pause_release_prob × enh_factor "
           "(if requires_enhancer; 0 ⇒ stay PAUSED) × restraint "
           "(if a cohesin leg sits within rnapii_pause_restraint_window)",
           size=12, anchor="start", fill=MUTED)
    s.text(40, fy + 44, "Non-ELONGATING states are stationary blocks to cohesin; "
           "only ELONGATING can push a cohesin leg.", size=12, anchor="start",
           fill=MUTED, italic=True)
    s.text(40, fy + 66, "A lesion blocks RNAPII (Type-A pre-recognition, or any "
           "repair-state lesion) -> STALLED; the Pol II is then evicted when that "
           "lesion is repaired.", size=12, anchor="start", fill=MUTED, italic=True)
    s.save("rnapii_state_machine.svg")


# --- Figure 2: cohesin -------------------------------------------------------

def fig_cohesin():
    s = SVG(1080, 664, "Cohesin (per-leg) state machine")
    NW, NH = 184, 64
    yc = 300
    ext = s.node(540, yc, NW, NH, "EXTRUDING", *COHESIN["EXTRUDING"],
                 sub="steps out when target free")
    ctcf = s.node(170, yc, 196, NH, "CTCF-CAPTURED", *COHESIN["CTCF_CAPTURED"],
                  sub="anchored")
    stall = s.node(910, yc, NW, NH, "STALLED", *COHESIN["STALLED"],
                   sub="obstacle ahead")
    s.start_dot(540, 96)
    s.end_dot(540, 524)
    off = 18  # vertical offset between the two arrows of a bidirectional pair

    s.edge((540, 104), edge_pt(ext, "top"), "load_one / load_targeted")

    # CTCF-CAPTURED <-> EXTRUDING (capture left, release back right)
    cap_mid = (ctcf[0] + 196 / 2 + (540 - NW / 2 - (ctcf[0] + 196 / 2)) / 2)
    s.edge((540 - NW / 2, yc - off), (ctcf[0] + 196 / 2, yc - off))
    s.text(cap_mid, yc - off - 12, "ctcfCapture prob", size=11, fill=MUTED)
    s.edge((ctcf[0] + 196 / 2, yc + off), (540 - NW / 2, yc + off))
    s.text(cap_mid, yc + off + 18, "ctcfRelease prob", size=11, fill=MUTED)

    # EXTRUDING <-> STALLED (occupied right, frees back left)
    st_mid = (540 + NW / 2 + (910 - NW / 2 - (540 + NW / 2)) / 2)
    s.edge((540 + NW / 2, yc - off), (910 - NW / 2, yc - off))
    s.text(st_mid, yc - off - 12, "target occupied", size=11, fill=MUTED)
    s.edge((910 - NW / 2, yc + off), (540 + NW / 2, yc + off))
    s.text(st_mid, yc + off + 18, "target frees", size=11, fill=MUTED)

    # unload edges to the terminal node
    lt = (lambda k: f" = {PARAMS[k]:g}t" if (PARAMS and PARAMS.get(k) is not None)
          else "")
    s.edge((ctcf[0], ctcf[1] + NH / 2), (540 - 46, 516),
           f"1/LIFETIME_CTCF{lt('lifetime_ctcf')}", curve=70, label_size=11)
    s.edge(edge_pt(ext, "bottom"), (540, 516), f"1/LIFETIME{lt('lifetime')}")
    s.edge((stall[0], stall[1] + NH / 2), (540 + 46, 516),
           f"1/LIFETIME_STALLED{lt('lifetime_stalled')} †", curve=-70,
           label_size=11)

    fy = 562
    s.text(40, fy, "†  uses 1/LIFETIME_RNAPII_STALLED instead when the cohesin "
           "was stalled/pushed by RNAPII (rnapii_stalled flag).", size=12,
           anchor="start", fill=MUTED)
    s.text(40, fy + 20, "Each leg extrudes independently; on unload a fresh "
           "cohesin reloads to keep the count constant.", size=12,
           anchor="start", fill=MUTED, italic=True)
    s.text(40, fy + 44, "A DNA lesion is another obstacle: an eligible lesion "
           "(Type-A pre-recognition with no Pol II present, or any repair-state "
           "lesion) stalls the leg with prob lesion_block_prob.", size=12,
           anchor="start", fill=MUTED)
    s.text(40, fy + 62, "A Type-A pre-recognition intrinsic stall sets the "
           "rnapii_stalled flag (fast eviction, †); with RNAPII present the stalled "
           "Pol II does the blocking, not the lesion directly.", size=12,
           anchor="start", fill=MUTED)
    s.save("cohesin_state_machine.svg")


# --- Figure 3: interaction ---------------------------------------------------

def fig_interaction():
    s = SVG(1020, 1000, "RNAPII × cohesin coupling")
    s.text(510, 56,
           "Each tick resolves in two phases: ① RNAPII step (B), then "
           "② cohesin step (A). Blocking, pushing and bypass are all "
           "probabilistic.", size=12.5, fill=MUTED, italic=True)

    def panel(x, y, w, h, tag, title):
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
              f'fill="#f9fafb" stroke="#e5e7eb" stroke-width="1.5"/>')
        s.add(f'<rect x="{x}" y="{y}" width="34" height="34" rx="10" '
              f'fill="{INK}"/>')
        s.text(x + 17, y + 23, tag, size=17, weight="700", fill="#ffffff")
        s.text(x + 46, y + 23, title, size=14, weight="700", anchor="start",
               fill=INK)

    def small(cx, cy, w, h, label, fill, stroke, sub=None):
        x, y = cx - w / 2, cy - h / 2
        s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        s.text(cx, cy + (3 if not sub else -3), label, size=12.5, weight="700",
               fill="#ffffff")
        if sub:
            s.text(cx, cy + 13, sub, size=10, fill="#f8fafc")
        return (cx, cy, w, h)

    def diamond(cx, cy, w, h, label):
        pts = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
        s.add(f'<polygon points="{pts}" fill="#fff7ed" stroke="#9a3412" '
              f'stroke-width="1.8"/>')
        parts = label.split("|")
        for i, ln in enumerate(parts):
            dy = (i - (len(parts) - 1) / 2) * 14 + 4
            s.text(cx, cy + dy, ln, size=12, weight="700", fill="#9a3412")
        return (cx, cy, w, h)

    def box(cx, cy, w, h, label, size=11.5, fill="#eef2ff", stroke="#3730a3"):
        x, y = cx - w / 2, cy - h / 2
        s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        for i, ln in enumerate(label.split("|")):
            n = len(label.split("|"))
            dy = (i - (n - 1) / 2) * 15 + 4
            s.text(cx, cy + dy, ln, size=size, fill=INK)
        return (cx, cy, w, h)

    def vlist(x, y, w, title, rows, rowh=21):
        h = 30 + rowh * len(rows)
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
              f'fill="#ffffff" stroke="#e5e7eb" stroke-width="1.4"/>')
        s.text(x + 12, y + 20, title, size=11.5, weight="700", anchor="start",
               fill=INK)
        for i, (k, v) in enumerate(rows):
            yy = y + 40 + i * rowh
            s.text(x + 12, yy, k, size=11, anchor="start", fill=MUTED)
            s.text(x + w - 12, yy, v, size=11, anchor="end", weight="700",
                   fill=INK)
        return (x, y, w, h)

    pp = f"{1 - PARAMS['stall']:g}" if (PARAMS and PARAMS.get('stall') is not None) \
        else "1−stall"

    # Panel A -- cohesin step: leg moves toward an RNAPII body
    panel(20, 74, 488, 300, "A", "Cohesin step:  leg → RNAPII body")
    da = diamond(108, 250, 132, 78, "blocked?")
    vlist(322, 100, 176, "block prob by Pol state", [
        ("POISED", val("b_poised", "b_poised")),
        ("PAUSED", val("b_paused", "b_paused")),
        ("ELONGATING", val("b_elong", "b_elong")),
        ("TERMINATING", val("b_term", "b_term")),
    ])
    a_bypass = box(360, 240, 280, 48,
                   "no → BYPASS|leg tunnels onto Pol site (coexist)",
                   fill="#ecfeff", stroke="#155e75", size=11)
    a_stall = box(360, 330, 280, 54,
                  "yes → leg STALLS|rnapii_stalled → fast eviction",
                  fill="#fef2f2", stroke="#991b1b", size=11)
    s.edge(edge_pt(da, "right"), edge_pt(a_bypass, "left"), "no", label_size=10,
           curve=-18)
    s.edge(edge_pt(da, "right"), edge_pt(a_stall, "left"), "yes", label_size=10,
           curve=22)

    # Panel B -- RNAPII step: Pol moves toward a cohesin leg. Linear ladder; each
    # test falls to the shared STALL node or continues down; pass all -> PUSH.
    panel(528, 74, 472, 600, "B", "RNAPII step:  → cohesin leg")
    xc, xs = 706, 894
    d1 = diamond(xc, 168, 178, 70, "Pol ELONGATING?")
    d2 = diamond(xc, 314, 178, 70, "intrinsic stall?")
    d3 = diamond(xc, 460, 178, 70, "pushes leg?")
    push = box(xc, 598, 268, 58,
               "PUSH — displace leg|leg.rnapii_stalled = True → evict",
               fill="#ecfdf5", stroke="#065f46")
    stall = small(xs, 314, 150, 58, "STALL", *RNAPII["STALLED"],
                  sub="Pol stays put")
    s.edge(edge_pt(d1, "bottom"), edge_pt(d2, "top"), "yes", label_size=10)
    s.edge(edge_pt(d2, "bottom"), edge_pt(d3, "top"), f"no ({pp})",
           label_size=10)
    s.edge(edge_pt(d3, "bottom"), edge_pt(push, "top"),
           f"yes · p = {val('push','push')} co-dir / "
           f"{val('headon','headon')} head-on", label_size=10)
    s.edge(edge_pt(d1, "right"), (xs, stall[1] - 22), "no", curve=-30,
           label_size=10)
    s.edge(edge_pt(d2, "right"), edge_pt(stall, "left"),
           f"yes ({val('stall','stall')})", label_size=10)
    s.edge(edge_pt(d3, "right"), (xs, stall[1] + 22), "no", curve=30,
           label_size=10)

    # Panel C -- how cohesin sets the PAUSED -> ELONGATING release probability
    panel(20, 394, 488, 280, "C", "Cohesin sets PAUSE-release probability")
    c_pa = small(92, 470, 96, 50, "PAUSED", *RNAPII["PAUSED"])
    c_gate = diamond(258, 470, 150, 74, "draw <|release_prob?")
    c_el = small(440, 470, 118, 48, "ELONGATING", *RNAPII["ELONGATING"])
    s.edge(edge_pt(c_pa, "right"), edge_pt(c_gate, "left"))
    s.edge(edge_pt(c_gate, "right"), edge_pt(c_el, "left"), "yes", label_size=10)
    # no -> stay PAUSED (loop back underneath)
    s.edge((c_gate[0], c_gate[1] + 37), (c_pa[0], c_pa[1] + 25), "no → stay",
           curve=44, label_size=10)
    # the release-probability formula, fed by two cohesin modulators
    s.text(264, 566, "release_prob  =  pause_release_prob  ×  enh_factor  ×  restraint",
           size=12, weight="700", fill=INK)
    s.add(f'<rect x="44" y="588" width="13" height="13" rx="3" fill="#ecfdf5" '
          f'stroke="#065f46" stroke-width="1.5"/>')
    s.text(64, 599, "enh_factor ↑  E-P loop brackets (TSS, enhancer); "
           "factor = 0 ⇒ gate shut → stay PAUSED", size=11, anchor="start",
           fill=INK)
    s.add(f'<rect x="44" y="616" width="13" height="13" rx="3" fill="#fef2f2" '
          f'stroke="#991b1b" stroke-width="1.5"/>')
    s.text(64, 627, f"× restraint ({val('restraint','restraint')}) ↓  cohesin leg "
           f"within {val('window','window')} site(s) of the paused Pol → slower release",
           size=11, anchor="start", fill=INK)
    s.text(264, 652, "activation (E-P) and restraint are the two opposing cohesin "
           "arms (Kim 2026; Tei 2026)", size=10.5, fill=MUTED, italic=True)

    # Panel D -- the actual parameter values this figure is drawn from
    src = (PARAMS or {}).get("src")
    panel(20, 694, 980, 286, "D",
          f"Parameters" + (f"  (from {src})" if src else
                           "  (symbolic — pass a config for values)"))
    vlist(40, 740, 300, "RNAPII → cohesin (push)", [
        ("rnapii_stall_prob", val("stall", "—")),
        ("rnapii_push_prob (co-dir)", val("push", "—")),
        ("rnapii_headon_push_prob", val("headon", "—")),
    ])
    vlist(360, 740, 300, "cohesin → RNAPII (block)", [
        ("rnapii_poised_block_prob", val("b_poised", "—")),
        ("rnapii_paused_block_prob", val("b_paused", "—")),
        ("rnapii_elongating_block_prob", val("b_elong", "—")),
        ("rnapii_terminating_block_prob", val("b_term", "—")),
    ])
    vlist(680, 740, 300, "pause restraint / cohesin lifetimes", [
        ("rnapii_pause_cohesin_restraint", val("restraint", "—")),
        ("rnapii_pause_restraint_window", val("window", "—")),
        ("lifetime", val("lifetime", "—")),
        ("lifetime_stalled", val("lifetime_stalled", "—")),
        ("lifetime_rnapii_stalled", val("lifetime_rnapii_stalled", "—")),
        ("lifetime_ctcf", val("lifetime_ctcf", "—")),
    ])

    s.save("rnapii_cohesin_interaction.svg")


# --- Figure 4: lesions -------------------------------------------------------

def fig_lesion():
    s = SVG(1060, 720, "DNA-lesion (UV-damage) state machine")
    s.text(530, 58,
           "Each lesion spawns with a fixed type, runs PRE-RECOGNITION -> REPAIR "
           "-> repaired, and the population is held at N // lesion_spacing by refill.",
           size=12.5, fill=MUTED, italic=True)

    # local helpers (kept figure-local, mirroring fig_interaction)
    def box(cx, cy, w, h, label, size=12, fill="#eef2ff", stroke="#3730a3"):
        x, y = cx - w / 2, cy - h / 2
        s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        lines = label.split("|")
        for i, ln in enumerate(lines):
            dy = (i - (len(lines) - 1) / 2) * 17 + 4
            s.text(cx, cy + dy, ln, size=size, fill=INK)
        return (cx, cy, w, h)

    def vlist(x, y, w, title, rows, rowh=21):
        h = 30 + rowh * len(rows)
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
              f'fill="#ffffff" stroke="#e5e7eb" stroke-width="1.4"/>')
        s.text(x + 12, y + 20, title, size=11.5, weight="700", anchor="start", fill=INK)
        for i, (k, v) in enumerate(rows):
            yy = y + 40 + i * rowh
            s.text(x + 12, yy, k, size=11, anchor="start", fill=MUTED)
            s.text(x + w - 12, yy, v, size=11, anchor="end", weight="700", fill=INK)
        return (x, y, w, h)

    def itable(x, y, w, rows, rowh=27):
        head = ("type · state", "blocks RNAPII", "stalls cohesin")
        c0, c1, c2 = x + 14, x + w * 0.42, x + w * 0.70
        h = 34 + rowh * len(rows)
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
              f'fill="#ffffff" stroke="#e5e7eb" stroke-width="1.4"/>')
        for cx_, t in ((c0, head[0]), (c1, head[1]), (c2, head[2])):
            s.text(cx_, y + 21, t, size=11, weight="700", anchor="start", fill=INK)
        s.add(f'<line x1="{x + 10:.1f}" y1="{y + 30}" x2="{x + w - 10:.1f}" '
              f'y2="{y + 30}" stroke="#e5e7eb" stroke-width="1"/>')

        def clr(v):
            return "#065f46" if v.startswith("yes") else (MUTED if v == "no" else INK)
        for i, (a, b, c) in enumerate(rows):
            yy = y + 34 + i * rowh + 14
            s.text(c0, yy, a, size=11, anchor="start", weight="700", fill=INK)
            s.text(c1, yy, b, size=11, anchor="start", fill=clr(b))
            s.text(c2, yy, c, size=11, anchor="start", fill=clr(c))
        return (x, y, w, h)

    # --- state machine (left column) ---
    NW, NH = 210, 64
    cx = 250
    pre = s.node(cx, 240, NW, NH, "PRE-RECOGNITION", *LESION["PRE"],
                 sub="damage unmarked", code=0)
    rep = s.node(cx, 470, NW, NH, "REPAIR", *LESION["REPAIR"],
                 sub="machinery bound", code=1)
    s.start_dot(cx, 100)
    s.end_dot(cx, 640)

    tau_pre = val("lesion_prerecognition_ticks", "τ_pre")
    tau_rep = val("lesion_repair_ticks", "τ_rep")
    s.edge((cx, 108), edge_pt(pre, "top"), "spawn")
    s.edge(edge_pt(pre, "bottom"), edge_pt(rep, "top"),
           f"recognise · 1/{tau_pre}", label_size=11)
    s.edge(edge_pt(rep, "bottom"), (cx, 632),
           f"repair · 1/{tau_rep}", label_size=11)
    s.self_loop(pre, "right", "stay")
    s.self_loop(rep, "right", "stay")
    s.text(cx, 662, "repaired → removed", size=11, fill=MUTED)

    # homeostatic refill feedback (far left, dashed)
    s.add(f'<path d="M235,646 C95,560 95,200 235,106" fill="none" '
          f'stroke="{MUTED}" stroke-width="1.6" stroke-dasharray="5,4" '
          f'marker-end="url(#arrow)"/>')
    s.text(96, 374, "homeostatic", size=10.5, anchor="middle", fill=MUTED)
    s.text(96, 388, "refill", size=10.5, anchor="middle", fill=MUTED)

    # --- type assignment + interaction matrix (right column) ---
    box(800, 150, 470, 86,
        "Type assigned at spawn (fixed for life):|"
        "in a gene body → Type A (p = type_a_prob) else Type B|"
        "off any gene body → always Type B",
        size=12, fill="#fdf4ff", stroke="#86198f")
    s.text(568, 226, "What each (type · state) does:", size=12, weight="700",
           anchor="start", fill=INK)
    itable(568, 236, 470, [
        ("Type A · pre", "yes", "via Pol II / intrinsic †"),
        ("Type B · pre", "no", "no"),
        ("Type A · repair", "yes", "yes ‡"),
        ("Type B · repair", "yes", "yes ‡"),
    ])
    src = (PARAMS or {}).get("src")
    vlist(568, 392, 470,
          "parameters" + (f"  (from {src})" if src else "  (symbolic — pass a config)"),
          [
              ("lesion_spacing  (hold N // spacing)", val("lesion_spacing", "—")),
              ("lesion_type_a_prob", val("lesion_type_a_prob", "—")),
              ("lesion_prerecognition_ticks  (τ_pre)", val("lesion_prerecognition_ticks", "—")),
              ("lesion_repair_ticks  (τ_rep)", val("lesion_repair_ticks", "—")),
              ("lesion_block_prob", val("lesion_block_prob", "—")),
              ("lesion_tad_size_exponent  (α)", val("lesion_tad_size_exponent", "—")),
              ("lesion_tad_repair_exponent  (β)", val("lesion_tad_repair_exponent", "—")),
          ])

    # footnotes
    fy = 600
    s.text(40, fy, "†  Type-A pre-recognition: a real stalled Pol II blocks "
           "cohesin when RNAPII is in the model; with no RNAPII the lesion stalls "
           "the leg itself (rnapii_stalled → fast eviction).",
           size=11.5, anchor="start", fill=MUTED)
    s.text(40, fy + 20, "‡  stalls an incoming cohesin leg with probability "
           "lesion_block_prob. A lesion-stalled Pol II is evicted when the lesion "
           "is repaired.", size=11.5, anchor="start", fill=MUTED)
    s.text(40, fy + 44, "Shorter TADs carry MORE lesions (placement ∝ "
           "L_TAD^(−α)) but recognise & repair them FASTER "
           "(both rates × (L_mean / L_TAD)^β).",
           size=12, anchor="start", fill=MUTED, italic=True)
    s.save("lesion_state_machine.svg")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate RNAPII/cohesin state-machine SVGs.")
    ap.add_argument("config", nargs="?",
                    help="pipeline YAML; pulls actual parameter values into the "
                         "interaction figure (omit for symbolic labels)")
    ap.add_argument("--out", help="output directory (default docs/figs)")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
    if args.config:
        PARAMS = load_params(args.config)
        print(f"# parameters from {PARAMS['src']}")
    fig_rnapii()
    fig_cohesin()
    fig_interaction()
    fig_lesion()
