"""HTML+JS template for the standalone bridging viewer.

Reproduces the layout/behaviour of abdenlab.org/cohesin-bridging as a single
self-contained file: a playback bar plus E-P proximity, 1D lattice + arcs,
kymograph, and bridge-map panels, all driven in the browser from an embedded
JSON trajectory. Build with :func:`render` -- it substitutes the two
placeholders (no ``str.format``: the JS/CSS is full of braces).

Embedded data contract (compact for size)::

    {
      title, latticeSize, siteOffset,
      ctcfSites: [int, ...],
      elements:  [{position, type: "promoter"|"enhancer", label, direction?}],
      genes:     [{geneId, tss, tes, start, end, label}, ...],
      tads:      [{chain, tad, label, start, end, size}, ...],
      insulation:{window, positions, frame, cumulative},
      tadSignals:{tads:[{label, size}], intra, inter, ratio},  # frame-major series

      eps:       [{e, p, label, genomic}, ...],
      frames:    [{c: [[id, left, right], ...], r: [[pos, gene_id, state], ...],
                   s: [sEff|null, ...], l: [site, ...]}, ...]
    }
"""

from __future__ import annotations

import json

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root { --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --bg:#f8fafc; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--fg); background:#fff; }
  main { max-width:1100px; margin:0 auto; padding:24px 28px 60px; }
  h1 { font-size:22px; font-weight:700; margin:0 0 16px; }
  h2 { font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
       color:var(--muted); margin:22px 0 8px; }
  .controls { display:flex; align-items:center; gap:14px; padding:10px 14px; background:var(--bg);
              border:1px solid var(--line); border-radius:10px; }
  .play-btn { width:34px; height:34px; border:none; border-radius:8px; background:#2563eb; color:#fff;
              font-size:14px; cursor:pointer; flex:0 0 auto; }
  .play-btn:hover { background:#1d4ed8; }
  .slider { flex:1 1 auto; }
  .time-label { font-variant-numeric:tabular-nums; color:var(--muted); font-size:13px; min-width:90px;
                text-align:right; }
  .speed-control { font-size:13px; color:var(--muted); }
  .panel-svg { width:100%; display:block; }
  .bottom-panels { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
  .metric-panels { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
  .panel { min-width:0; }
  canvas { width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }
  .legend { display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:8px; font-size:11px; color:var(--muted); }
  .legend-item { display:inline-flex; align-items:center; gap:6px; }
  .swatch { width:11px; height:11px; border-radius:2px; display:inline-block; }
  .swatch.instant { background:#2563eb; }
  .swatch.trace { background:#cbd5e1; }
  .swatch.target { background:#fff; border:1.5px dashed #2563eb; border-radius:999px; }
  .swatch.promoter { background:#16a34a; }
  .swatch.enhancer { background:#f59e0b; border:1px solid #d97706; }
  .swatch.ctcf { background:#6b7280; }
  .swatch.cohesin { width:16px; height:3px; border-radius:999px; background:#2563eb; }
  .swatch.gene-body { width:16px; height:4px; border-radius:999px;
                      background:linear-gradient(90deg,#2563eb,#dc2626,#16a34a); }
  .swatch.rnapii-poised { background:#64748b; border-radius:999px; }
  .swatch.rnapii-paused { background:#f59e0b; border-radius:999px; }
  .swatch.rnapii-elongating { background:#be185d; border-radius:999px; }
  .swatch.lesion { background:#dc2626; transform:rotate(45deg); }
  .epval { font-size:13px; color:var(--fg); font-variant-numeric:tabular-nums; }
  .empty { color:var(--muted); font-size:13px; font-style:italic; padding:6px 0; }
</style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>

  <div class="controls">
    <button id="play" class="play-btn">&#9654;</button>
    <input id="slider" class="slider" type="range" min="0" value="0"/>
    <span id="time" class="time-label">0 / 0</span>
    <label class="speed-control">Speed:
      <select id="speed">
        <option>0.25x</option><option>0.5x</option><option selected>1x</option>
        <option>2x</option><option>4x</option><option>8x</option>
      </select>
    </label>
  </div>

  <h2>E-P Proximity</h2>
  <div id="ep-wrap"><svg id="ep" class="panel-svg" height="80"></svg></div>

  <h2>1D Lattice</h2>
  <div id="lat-wrap"><svg id="lattice" class="panel-svg" height="140"></svg></div>
  <div class="legend">
    <span class="legend-item"><span class="swatch ctcf"></span> CTCF</span>
    <span class="legend-item"><span class="swatch promoter"></span> promoter</span>
    <span class="legend-item"><span class="swatch enhancer"></span> enhancer</span>
    <span class="legend-item"><span class="swatch gene-body"></span> gene body by G#</span>
    <span class="legend-item"><span class="swatch rnapii-poised"></span> RNAPII poised</span>
    <span class="legend-item"><span class="swatch rnapii-paused"></span> RNAPII paused</span>
    <span class="legend-item"><span class="swatch rnapii-elongating"></span> RNAPII elongating</span>
    <span class="legend-item"><span class="swatch lesion"></span> DNA lesion</span>
  </div>

  <div class="bottom-panels">
    <div class="panel">
      <h2>Kymograph</h2>
      <canvas id="kymo" width="460" height="460"></canvas>
    </div>
    <div class="panel">
      <h2>Bridge Map</h2>
      <canvas id="bridge" width="460" height="460"></canvas>
      <div class="legend">
        <span class="legend-item"><span class="swatch instant"></span> upper triangle: current bridges</span>
        <span class="legend-item"><span class="swatch trace"></span> lower triangle: visited trace so far</span>
        <span class="legend-item"><span class="swatch target"></span> dashed rings: E-P targets</span>
        <span class="legend-item"><span class="swatch promoter"></span> promoter sites (both axes)</span>
        <span class="legend-item"><span class="swatch enhancer"></span> enhancer sites (both axes)</span>
        <span class="legend-item"><span class="swatch gene-body"></span> marginal bands: genic regions</span>
        <span class="legend-item">⌗ dashed crosshair: TAD boundaries B# site (both axes)</span>
        <span class="legend-item">T# on right y-axis: TAD region midpoint</span>
      </div>
    </div>
  </div>

  <div class="metric-panels">
    <div class="panel">
      <h2>Cumulative Insulation Score</h2>
      <canvas id="insulation-cumulative" width="560" height="260"></canvas>
    </div>
    <div class="panel">
      <h2>Per-Frame Insulation Score</h2>
      <canvas id="insulation-frame" width="560" height="260"></canvas>
    </div>
  </div>

  <h2>TAD Inter / Intra Signals (cumulative)</h2>
  <div id="tad-wrap"><canvas id="tad-signals" width="1040" height="300"></canvas></div>
  <div class="legend" id="tad-legend"></div>
</main>

<script>
const DATA = __VIEWER_DATA__;
const SVGNS = "http://www.w3.org/2000/svg";
const PALETTE = ["#2563eb","#dc2626","#16a34a","#9333ea","#ea580c","#0891b2",
                 "#c026d3","#65a30d","#d97706","#4f46e5","#e11d48","#059669"];
const C_PROMOTER = "#16a34a", C_ENH_FILL = "#f59e0b", C_ENH_STROKE = "#d97706", C_CTCF = "#6b7280";
const RNAPII_COLORS = ["#64748b", "#f59e0b", "#be185d", "#7c3aed"];
const RNAPII_STATES = ["poised", "paused", "elongating", "terminating"];
const cohColor = id => PALETTE[((id % PALETTE.length) + PALETTE.length) % PALETTE.length];

const N = DATA.latticeSize;
const T = DATA.frames.length;
const lin = (d0,d1,r0,r1) => v => r0 + (v - d0) * (r1 - r0) / ((d1 - d0) || 1);

let t = 0, playing = false, speed = 1, acc = 0, last = 0;

// ---- DOM ----
const $play = document.getElementById("play");
const $slider = document.getElementById("slider");
const $time = document.getElementById("time");
const $speed = document.getElementById("speed");
$slider.max = Math.max(0, T - 1);

// ---------------------------------------------------------------- E-P proximity
// One row per E-P pair: a linear track from contact (0) to genomic separation,
// with a coloured marker whose position is the current effective distance --
// it slides toward "contact" as cohesin loops bridge the pair.
const epSvg = document.getElementById("ep");
const EP_ROW = 30, EP_PAD = 10, EP_LABEL_W = 96, EP_VAL_W = 92;
const mkText = (x, y, s, opt = {}) => {
  const e = document.createElementNS(SVGNS, "text");
  e.setAttribute("x", x); e.setAttribute("y", y);
  e.setAttribute("font-size", opt.size || "11"); e.textContent = s;
  if (opt.anchor) e.setAttribute("text-anchor", opt.anchor);
  if (opt.fill) e.setAttribute("fill", opt.fill);
  if (opt.weight) e.setAttribute("font-weight", opt.weight);
  return e;
};
function drawProximity() {
  epSvg.innerHTML = "";
  const eps = DATA.eps || [];
  const w = epSvg.clientWidth || 1000;
  if (eps.length === 0) {
    epSvg.setAttribute("height", 40);
    epSvg.appendChild(mkText(10, 24, "no enhancer-promoter pair in this trajectory",
                             { fill: "#9ca3af" }));
    return;
  }
  epSvg.setAttribute("height", EP_PAD * 2 + eps.length * EP_ROW);
  const x0 = EP_LABEL_W, x1 = w - EP_VAL_W;
  eps.forEach((pair, i) => {
    const color = cohColor(i);
    const cy = EP_PAD + i * EP_ROW + EP_ROW / 2;
    const genomic = pair.genomic || Math.abs(pair.p - pair.e) || 1;
    const eff = DATA.frames[t].s[i];
    const frac = eff == null ? 1 : Math.max(0, Math.min(1, eff / genomic));
    epSvg.appendChild(mkText(8, cy + 4, pair.label, { fill: color, weight: "600" }));
    // track
    const track = document.createElementNS(SVGNS, "line");
    track.setAttribute("x1", x0); track.setAttribute("x2", x1);
    track.setAttribute("y1", cy); track.setAttribute("y2", cy);
    track.setAttribute("stroke", "#e5e7eb"); track.setAttribute("stroke-width", "3");
    track.setAttribute("stroke-linecap", "round"); epSvg.appendChild(track);
    // filled portion 0..eff (contact side on the left)
    const mx = x0 + frac * (x1 - x0);
    const close = frac < 0.18;
    const fill = document.createElementNS(SVGNS, "line");
    fill.setAttribute("x1", x0); fill.setAttribute("x2", mx);
    fill.setAttribute("y1", cy); fill.setAttribute("y2", cy);
    fill.setAttribute("stroke", close ? "#16a34a" : color);
    fill.setAttribute("stroke-width", "3"); fill.setAttribute("stroke-linecap", "round");
    epSvg.appendChild(fill);
    const dot = document.createElementNS(SVGNS, "circle");
    dot.setAttribute("cx", mx); dot.setAttribute("cy", cy); dot.setAttribute("r", 5);
    dot.setAttribute("fill", close ? "#16a34a" : color);
    dot.setAttribute("stroke", "#fff"); dot.setAttribute("stroke-width", "1.5");
    epSvg.appendChild(dot);
    epSvg.appendChild(mkText(8, cy - 9, "", {}));   // spacer keeps label baseline tidy
    epSvg.appendChild(mkText(x0 - 4, cy - 8, "contact", { anchor: "end", size: "8", fill: "#9ca3af" }));
    epSvg.appendChild(mkText(x1 + 4, cy - 8, "apart", { size: "8", fill: "#9ca3af" }));
    epSvg.appendChild(mkText(w - 4, cy + 4,
      `${eff == null ? "–" : Math.round(eff)} / ${genomic}`,
      { anchor: "end", fill: "#374151", weight: "600" }));
  });
}

// ---------------------------------------------------------------- 1D lattice
const latSvg = document.getElementById("lattice");
function arcHeight(span) { return Math.min(70, 8 + Math.sqrt(span) * 2.6); }
function drawGeneBodies(svg, xs, base) {
  const y = base + 10;
  for (const gene of (DATA.genes || [])) {
    const x0 = xs(gene.start), x1 = xs(gene.end);
    const color = cohColor(gene.geneId || 0);
    const seg = document.createElementNS(SVGNS, "line");
    seg.setAttribute("x1", x0); seg.setAttribute("x2", x1);
    seg.setAttribute("y1", y); seg.setAttribute("y2", y);
    seg.setAttribute("stroke", color); seg.setAttribute("stroke-width", "5");
    seg.setAttribute("stroke-linecap", "round"); seg.setAttribute("opacity", "0.75");
    const title = document.createElementNS(SVGNS, "title");
    title.textContent = `${gene.label || "gene"} TSS ${gene.tss}, TES ${gene.tes}`;
    seg.appendChild(title);
    svg.appendChild(seg);
  }
}
function drawRNAPII(svg, xs, base) {
  const rnap = (DATA.frames[t].r || []);
  for (const [pos, geneId, state] of rnap) {
    const x = xs(pos);
    const y = base - 16;
    const color = RNAPII_COLORS[state] || "#475569";
    const halo = document.createElementNS(SVGNS, "circle");
    halo.setAttribute("cx", x); halo.setAttribute("cy", y); halo.setAttribute("r", 7.5);
    halo.setAttribute("fill", "#fff"); halo.setAttribute("opacity", "0.95");
    svg.appendChild(halo);
    const body = document.createElementNS(SVGNS, "circle");
    body.setAttribute("cx", x); body.setAttribute("cy", y); body.setAttribute("r", 5.5);
    body.setAttribute("fill", color); body.setAttribute("stroke", "#fff"); body.setAttribute("stroke-width", "1.5");
    const title = document.createElementNS(SVGNS, "title");
    const geneLabel = geneId >= 0 ? `gene ${geneId}` : "unknown gene";
    title.textContent = `RNAPII ${geneLabel}, ${RNAPII_STATES[state] || "unknown state"}`;
    body.appendChild(title);
    svg.appendChild(body);
    const label = document.createElementNS(SVGNS, "text");
    label.setAttribute("x", x); label.setAttribute("y", y + 3);
    label.setAttribute("text-anchor", "middle"); label.setAttribute("font-size", "6.5");
    label.setAttribute("font-weight", "700"); label.setAttribute("fill", "#fff");
    label.textContent = "II";
    svg.appendChild(label);
  }
}
function drawLesions(svg, xs, base) {
  // DNA-damage sites on the backbone (current frame): red star burst.
  const r = 4.5;
  for (const site of (DATA.frames[t].l || [])) {
    const x = xs(site);
    const g = document.createElementNS(SVGNS, "g");
    for (const ang of [0, 45, 90, 135]) {
      const rad = ang * Math.PI / 180;
      const dx = r * Math.cos(rad), dy = r * Math.sin(rad);
      const ray = document.createElementNS(SVGNS, "line");
      ray.setAttribute("x1", x - dx); ray.setAttribute("y1", base - dy);
      ray.setAttribute("x2", x + dx); ray.setAttribute("y2", base + dy);
      ray.setAttribute("stroke", "#dc2626"); ray.setAttribute("stroke-width", "1.6");
      ray.setAttribute("stroke-linecap", "round");
      g.appendChild(ray);
    }
    const title = document.createElementNS(SVGNS, "title");
    title.textContent = `DNA lesion @ site ${site + (DATA.siteOffset || 0)}`;
    g.appendChild(title);
    svg.appendChild(g);
  }
}
function drawLattice() {
  latSvg.innerHTML = "";
  const w = latSvg.clientWidth || 1000, pad = 30, base = 118;
  const xs = lin(0, N - 1, pad, w - pad);
  // backbone
  const line = document.createElementNS(SVGNS, "line");
  line.setAttribute("x1", pad); line.setAttribute("y1", base);
  line.setAttribute("x2", w - pad); line.setAttribute("y2", base);
  line.setAttribute("stroke", "#cbd5e1"); line.setAttribute("stroke-width", "2");
  latSvg.appendChild(line);
  drawGeneBodies(latSvg, xs, base);
  // CTCF ticks
  for (const s of DATA.ctcfSites) {
    const tk = document.createElementNS(SVGNS, "line");
    tk.setAttribute("x1", xs(s)); tk.setAttribute("y1", base - 6);
    tk.setAttribute("x2", xs(s)); tk.setAttribute("y2", base + 6);
    tk.setAttribute("stroke", C_CTCF); tk.setAttribute("stroke-width", "2");
    latSvg.appendChild(tk);
  }
  // gene elements
  for (const el of DATA.elements) {
    const x = xs(el.position);
    if (el.type === "enhancer") {
      const e = document.createElementNS(SVGNS, "ellipse");
      e.setAttribute("cx", x); e.setAttribute("cy", base); e.setAttribute("rx", 7); e.setAttribute("ry", 4.5);
      e.setAttribute("fill", C_ENH_FILL); e.setAttribute("opacity", "0.85");
      e.setAttribute("stroke", C_ENH_STROKE); e.setAttribute("stroke-width", "1.5");
      latSvg.appendChild(e);
    } else {
      const m = document.createElementNS(SVGNS, "path");  // promoter chevron
      const dir = el.direction === -1 ? -1 : 1;            // points toward TES
      m.setAttribute("d", `M ${x} ${base - 8} L ${x + 5 * dir} ${base} L ${x} ${base + 8}`);
      m.setAttribute("fill", "none"); m.setAttribute("stroke", C_PROMOTER); m.setAttribute("stroke-width", "2");
      latSvg.appendChild(m);
    }
    if (el.label) {
      const lab = document.createElementNS(SVGNS, "text");
      lab.setAttribute("x", x); lab.setAttribute("y", base + 22); lab.setAttribute("text-anchor", "middle");
      lab.setAttribute("font-size", "9"); lab.setAttribute("fill", el.type === "enhancer" ? C_ENH_STROKE : C_PROMOTER);
      lab.textContent = el.label; latSvg.appendChild(lab);
    }
  }
  // cohesin arcs (current frame)
  for (const [id, l, r] of DATA.frames[t].c) {
    const lo = Math.min(l, r), hi = Math.max(l, r);
    if (lo === hi) continue;
    const x0 = xs(lo), x1 = xs(hi), mid = (x0 + x1) / 2;
    const p = document.createElementNS(SVGNS, "path");
    p.setAttribute("d", `M ${x0} ${base} Q ${mid} ${base - arcHeight(hi - lo)} ${x1} ${base}`);
    p.setAttribute("fill", "none"); p.setAttribute("stroke", cohColor(id));
    p.setAttribute("stroke-width", "2"); p.setAttribute("stroke-linecap", "round");
    p.setAttribute("opacity", "0.85");
    latSvg.appendChild(p);
  }
  drawLesions(latSvg, xs, base);
  drawRNAPII(latSvg, xs, base);
}

// ---------------------------------------------------------------- kymograph (canvas)
const kymo = document.getElementById("kymo");
const kctx = kymo.getContext("2d");
const kymoBg = document.createElement("canvas");      // cached static layer
kymoBg.width = kymo.width; kymoBg.height = kymo.height;
function buildKymoBg() {
  const W = kymo.width, H = kymo.height, ctx = kymoBg.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const xs = lin(0, N - 1, 0, W), ys = lin(0, T - 1, 0, H);
  for (let f = 0; f < T; f++) {
    const y = ys(f);
    for (const [id, l, r] of DATA.frames[f].c) {
      ctx.fillStyle = cohColor(id);
      ctx.fillRect(xs(l), y, 1.4, 1.4);
      ctx.fillRect(xs(r), y, 1.4, 1.4);
    }
  }
}
function drawKymo() {
  const W = kymo.width, H = kymo.height;
  kctx.clearRect(0, 0, W, H);
  kctx.drawImage(kymoBg, 0, 0);
  const y = lin(0, T - 1, 0, H)(t);          // time cursor
  kctx.strokeStyle = "rgba(220,38,38,0.9)"; kctx.lineWidth = 1.5;
  kctx.beginPath(); kctx.moveTo(0, y); kctx.lineTo(W, y); kctx.stroke();
}

// ---------------------------------------------------------------- dynamic insulation line plots
const insulationCum = document.getElementById("insulation-cumulative");
const insulationFrame = document.getElementById("insulation-frame");
const insulationCumCtx = insulationCum.getContext("2d");
const insulationFrameCtx = insulationFrame.getContext("2d");
const INSULATION = DATA.insulation || { window: 0, positions: [], frame: [], cumulative: [] };

function insulationRowAbsMax(row) {                  // log2 scores are signed
  let m = 0;
  for (const v of row || []) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  return m || 1;
}
function drawInsulationGeneBands(ctx, x, padL, padT, plotW, plotH) {
  const yBand = padT + plotH + 29;
  ctx.save();
  ctx.lineWidth = 3.5;
  ctx.font = "8px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  for (const gene of DATA.genes || []) {
    const a = Math.max(0, Math.min(N - 1, gene.start));
    const b = Math.max(0, Math.min(N - 1, gene.end));
    const lo = Math.min(a, b), hi = Math.max(a, b);
    const x0 = Math.max(padL, x(lo)), x1 = Math.min(padL + plotW, x(hi));
    if (x1 < padL || x0 > padL + plotW) continue;
    const color = cohColor(gene.geneId || 0);
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.72;
    ctx.beginPath(); ctx.moveTo(x0, yBand); ctx.lineTo(x1, yBand); ctx.stroke();
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = color;
    ctx.fillText(gene.label || `G${gene.geneId}`, (x0 + x1) / 2, yBand - 3);
  }
  ctx.restore();
}
function drawInsulationElementAxis(ctx, x, padL, padT, plotW, plotH) {
  const y0 = padT + plotH + 5, y1 = padT + plotH + 14, yText = padT + plotH + 17;
  const occupied = [];
  const els = [
    ...bridgeAxisElements("promoter").map(e => ({ ...e, color: C_PROMOTER, type: "promoter" })),
    ...bridgeAxisElements("enhancer").map(e => ({ ...e, color: C_ENH_STROKE, type: "enhancer" })),
  ].sort((a, b) => a.position - b.position);
  ctx.save();
  ctx.font = "8px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  for (const el of els) {
    const px = x(el.position);
    if (px < padL || px > padL + plotW) continue;
    ctx.strokeStyle = el.color; ctx.fillStyle = el.color; ctx.lineWidth = 1.2;
    if (el.type === "promoter") {
      const dir = el.direction === -1 ? -1 : 1;
      ctx.beginPath();
      ctx.moveTo(px - 3 * dir, y0);
      ctx.lineTo(px + 3 * dir, (y0 + y1) / 2);
      ctx.lineTo(px - 3 * dir, y1);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.ellipse(px, (y0 + y1) / 2, 3.8, 2.5, 0, 0, Math.PI * 2);
      ctx.fill();
    }
    const dirText = el.type === "promoter" ? (el.direction === -1 ? "<" : ">") : "";
    const text = el.type === "promoter" && el.direction === -1
      ? `${dirText}${el.label}`
      : `${el.label}${dirText}`;
    const tw = ctx.measureText(text).width;
    const left = px - tw / 2, right = px + tw / 2;
    if (!occupied.some(([a, b]) => left < b + 3 && right > a - 3)) {
      occupied.push([left, right]);
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(text, px, yText);
    }
  }
  ctx.restore();
}
function drawInsulationUnavailableMargins(ctx, x, positions, padL, padT, plotW, plotH) {
  if (!positions.length) return;
  const supportMin = positions[0], supportMax = positions[positions.length - 1];
  const leftW = Math.max(0, x(supportMin) - padL);
  const rightX = x(supportMax);
  const rightW = Math.max(0, padL + plotW - rightX);
  ctx.save();
  ctx.fillStyle = "rgba(148,163,184,0.10)";
  if (leftW > 0.5) ctx.fillRect(padL, padT, leftW, plotH);
  if (rightW > 0.5) ctx.fillRect(rightX, padT, rightW, plotH);
  ctx.restore();
}
function drawInsulationBoundaries(ctx, x, minP, maxP, padL, padT, plotW, plotH) {
  const occupied = [];
  ctx.save();
  ctx.font = "8px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  for (const b of bridgeBoundaries()) {
    if (b.pos <= minP || b.pos >= maxP) continue;
    const bx = x(b.pos);
    ctx.strokeStyle = "rgba(15,23,42,0.22)";
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(bx, padT); ctx.lineTo(bx, padT + plotH); ctx.stroke();
    ctx.setLineDash([]);
    const text = `${b.label} ${b.siteLabel}`;
    const tw = ctx.measureText(text).width;
    const left = bx - tw / 2, right = bx + tw / 2;
    if (!occupied.some(([a, c]) => left < c + 4 && right > a - 4)) {
      occupied.push([left, right]);
      ctx.fillStyle = "rgba(15,23,42,0.68)";
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(text, bx, padT - 3);
    }
  }
  ctx.restore();
}
function drawInsulationAxes(ctx, x, minP, maxP, padL, padT, plotW, plotH) {
  const yAxis = padT + plotH + 5;
  ctx.save();
  ctx.strokeStyle = "#cbd5e1";
  ctx.beginPath(); ctx.moveTo(padL, yAxis); ctx.lineTo(padL + plotW, yAxis); ctx.stroke();
  drawInsulationElementAxis(ctx, x, padL, padT, plotW, plotH);
  drawInsulationGeneBands(ctx, x, padL, padT, plotW, plotH);
  ctx.fillStyle = "#6b7280";
  ctx.font = "8px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText(bridgeSiteLabel(minP), padL, padT + plotH + 43);
  ctx.fillText(bridgeSiteLabel(maxP), padL + plotW, padT + plotH + 43);
  ctx.fillStyle = "#374151";
  ctx.font = "600 9px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  ctx.fillText("boundary / P direction / enhancer / gene", padL + plotW / 2, padT + plotH + 52);
  ctx.restore();
}
function drawInsulationPanel(canvas, ctx, matrix, color, label) {
  const W = canvas.width, H = canvas.height;
  const padL = 46, padR = 16, padT = 34, padB = 64;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const positions = INSULATION.positions || [];
  const row = (matrix || [])[t] || [];
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
  ctx.fillStyle = "#6b7280";
  ctx.font = "10px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText(`${label} | window ${INSULATION.window || 0} | frame ${t}/${T - 1}`, padL, 6);
  if (!positions.length || !row.length) {
    ctx.fillStyle = "#9ca3af"; ctx.font = "13px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
    ctx.fillText("no insulation scores available", padL, padT + 18);
    return;
  }
  const minP = 0, maxP = Math.max(1, N - 1);
  const yAbs = insulationRowAbsMax(row);              // symmetric around 0 (log2)
  const x = lin(minP, maxP, padL, padL + plotW);
  const y = lin(-yAbs, yAbs, padT + plotH, padT);
  drawInsulationUnavailableMargins(ctx, x, positions, padL, padT, plotW, plotH);
  ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 1;     // log2 zero baseline
  ctx.beginPath(); ctx.moveTo(padL, y(0)); ctx.lineTo(padL + plotW, y(0)); ctx.stroke();
  drawInsulationBoundaries(ctx, x, minP, maxP, padL, padT, plotW, plotH);
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath();
  let hasPoint = false;
  for (let i = 0; i < positions.length; i++) {
    const v = row[i];
    if (!Number.isFinite(v)) {
      if (hasPoint) ctx.stroke();
      ctx.beginPath();
      hasPoint = false;
      continue;
    }
    const px = x(positions[i]), py = y(v);
    if (!hasPoint) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
    hasPoint = true;
  }
  if (hasPoint) ctx.stroke();
  ctx.fillStyle = "#6b7280";
  ctx.font = "9px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  ctx.fillText(`+${yAbs.toPrecision(3)}`, padL - 5, padT);
  ctx.fillText("0", padL - 5, y(0));
  ctx.fillText(`-${yAbs.toPrecision(3)}`, padL - 5, padT + plotH);
  drawInsulationAxes(ctx, x, minP, maxP, padL, padT, plotW, plotH);
}
function drawInsulation() {
  drawInsulationPanel(
    insulationCum, insulationCumCtx, INSULATION.cumulative || [], "#2563eb",
    "visited trace · log2(obs/mean)"
  );
  drawInsulationPanel(
    insulationFrame, insulationFrameCtx, INSULATION.frame || [], "#dc2626",
    "current frame · log2(obs/mean)"
  );
}

// ---------------------------------------------------------------- TAD inter/intra signals
// One coloured line per displayed TAD per metric, plotted against frame and
// drawn up to the current frame t (cumulative), with a red time cursor. intra =
// both cohesin legs inside the TAD; inter = one leg in TAD, one on an equal-sized
// flank; ratio = cumulative intra / inter (gaps where inter is still 0).
const tadCanvas = document.getElementById("tad-signals");
const tadCtx = tadCanvas.getContext("2d");
const TADSIG = DATA.tadSignals || { tads: [], intra: [], inter: [], ratio: [] };
const tadColor = k => PALETTE[((k % PALETTE.length) + PALETTE.length) % PALETTE.length];

function tadSeriesMax(series) {
  let m = 0;
  for (const row of series || []) for (const v of row) if (Number.isFinite(v)) m = Math.max(m, v);
  return m || 1;
}
function buildTadLegend() {
  const $leg = document.getElementById("tad-legend");
  if (!TADSIG.tads.length) { $leg.textContent = ""; return; }
  $leg.innerHTML = TADSIG.tads.map((tad, k) =>
    `<span class="legend-item"><span class="swatch" style="background:${tadColor(k)}"></span>`
    + `${tad.label} (size ${tad.size})</span>`
  ).join("");
}
function drawTadSubplot(x0, y0, plotW, plotH, series, title, ymax) {
  const ctx = tadCtx;
  ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, plotW, plotH);
  ctx.fillStyle = "#374151";
  ctx.font = "600 10px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.fillText(title, x0, y0 - 4);
  // y-axis labels (0 .. ymax)
  ctx.fillStyle = "#6b7280"; ctx.font = "9px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  ctx.fillText(ymax.toPrecision(3), x0 - 4, y0);
  ctx.fillText("0", x0 - 4, y0 + plotH);
  // x-axis labels (frame range)
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText("0", x0, y0 + plotH + 4);
  ctx.fillText(`${T - 1}`, x0 + plotW, y0 + plotH + 4);
  ctx.fillText("frame", x0 + plotW / 2, y0 + plotH + 16);
  const xpix = lin(0, Math.max(1, T - 1), x0, x0 + plotW);
  const ypix = lin(0, ymax, y0 + plotH, y0);
  TADSIG.tads.forEach((tad, k) => {
    ctx.strokeStyle = tadColor(k); ctx.lineWidth = 1.5;
    ctx.beginPath();
    let has = false;
    for (let f = 0; f <= t; f++) {
      const v = (series[f] || [])[k];
      if (!Number.isFinite(v)) { if (has) ctx.stroke(); ctx.beginPath(); has = false; continue; }
      const px = xpix(f), py = ypix(v);
      if (!has) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      has = true;
    }
    if (has) ctx.stroke();
  });
  // time cursor
  ctx.strokeStyle = "rgba(220,38,38,0.75)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(xpix(t), y0); ctx.lineTo(xpix(t), y0 + plotH); ctx.stroke();
}
function drawTadSignals() {
  const W = tadCanvas.width, H = tadCanvas.height;
  tadCtx.clearRect(0, 0, W, H);
  tadCtx.fillStyle = "#fff"; tadCtx.fillRect(0, 0, W, H);
  if (!TADSIG.tads.length) {
    tadCtx.fillStyle = "#9ca3af"; tadCtx.font = "13px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
    tadCtx.textAlign = "left"; tadCtx.textBaseline = "top";
    tadCtx.fillText("no TADs in this trajectory", 12, 12);
    return;
  }
  const padL = 46, padR = 16, gap = 40, padT = 22, padB = 40;
  const plotW = (W - padL - padR - 2 * gap) / 3;
  const plotH = H - padT - padB;
  const subs = [
    [TADSIG.intra, "intra-TAD contacts", tadSeriesMax(TADSIG.intra)],
    [TADSIG.inter, "inter-TAD contacts", tadSeriesMax(TADSIG.inter)],
    [TADSIG.ratio, "intra : inter ratio", tadSeriesMax(TADSIG.ratio)],
  ];
  subs.forEach(([series, title, ymax], i) => {
    const x0 = padL + i * (plotW + gap);
    drawTadSubplot(x0, padT, plotW, plotH, series || [], title, ymax);
  });
}

// ---------------------------------------------------------------- bridge map (canvas)
// Visited trace accumulates over time: the lower triangle fills in as playback
// advances (frames 0..t), so it shows where bridges have been visited *so far*,
// not the whole run. Current bridges sit in the upper triangle.
const bridge = document.getElementById("bridge");
const bctx = bridge.getContext("2d");
const traceCv = document.createElement("canvas");     // accumulated visited trace
traceCv.width = bridge.width; traceCv.height = bridge.height;
const tctx = traceCv.getContext("2d");
let traceUpto = -1;                                   // last frame drawn into trace
function bridgeLayout() {
  const W = bridge.width, H = bridge.height;
  const left = 72, top = 46, right = 46, bottom = 72;
  const size = Math.max(1, Math.min(W - left - right, H - top - bottom));
  return { W, H, left, top, size, right, bottom,
           x0: left, y0: top, x1: left + size, y1: top + size,
           w: lin(0, N - 1, 0, size) };
}
function bridgeTicks() {                                 // ~6 even intervals
  const n = 6, out = [];
  for (let i = 0; i <= n; i++) out.push(Math.round(i * (N - 1) / n));
  return [...new Set(out)];
}
// TAD boundaries (inner edges shared by adjacent TADs), numbered along the axis.
function bridgeBoundaries() {
  const seen = new Set();
  for (const tad of DATA.tads || []) {
    for (const edge of [tad.start, tad.end]) {
      const p = +edge;
      if (p > 0 && p < N) seen.add(p);
    }
  }
  return [...seen].sort((a, b) => a - b).map((pos, i) => ({
    pos, label: `B${i}`, siteLabel: bridgeSiteLabel(pos)
  }));
}
function bridgeSiteLabel(site) {
  return `${Math.round(site + (DATA.siteOffset || 0))}`;
}
function bridgeAxisElements(type) {
  const seen = new Set(), out = [];
  for (const el of DATA.elements || []) {
    if (el.type !== type || el.position == null) continue;
    const p = +el.position;
    if (p < 0 || p >= N || seen.has(p)) continue;
    seen.add(p);
    out.push({
      position: p,
      label: el.label || (type === "promoter" ? "P" : "E"),
      direction: el.direction,
    });
  }
  return out;
}
function drawBridgeGeneBands(layout) {
  const { x0, y0, x1, y1, w } = layout;
  bctx.save();
  bctx.lineWidth = 4;
  bctx.font = "8px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  for (const gene of DATA.genes || []) {
    const a = Math.max(0, Math.min(N - 1, gene.start));
    const b = Math.max(0, Math.min(N - 1, gene.end));
    const color = cohColor(gene.geneId || 0);
    const xA = x0 + w(Math.min(a, b)), xB = x0 + w(Math.max(a, b));
    const yA = y0 + w(Math.min(a, b)), yB = y0 + w(Math.max(a, b));

    bctx.strokeStyle = color;
    bctx.globalAlpha = 0.72;
    bctx.beginPath(); bctx.moveTo(xA, y1 + 24); bctx.lineTo(xB, y1 + 24); bctx.stroke();
    bctx.beginPath(); bctx.moveTo(x0 - 25, yA); bctx.lineTo(x0 - 25, yB); bctx.stroke();

    bctx.globalAlpha = 0.9;
    bctx.fillStyle = color;
    bctx.textAlign = "center";
    bctx.textBaseline = "bottom";
    bctx.fillText(gene.label || `G${gene.geneId}`, (xA + xB) / 2, y1 + 21);
    bctx.save();
    bctx.translate(x0 - 30, (yA + yB) / 2);
    bctx.rotate(-Math.PI / 2);
    bctx.fillText(gene.label || `G${gene.geneId}`, 0, 0);
    bctx.restore();
  }
  bctx.restore();
}
// Element ticks (promoters + enhancers) drawn identically on both axes, so the
// x- and y-axes carry the same information for any element.
function drawBridgeAxisElements(layout) {
  const { x0, y0, x1, y1, w } = layout;
  const els = [
    ...bridgeAxisElements("promoter").map(e => ({ ...e, color: C_PROMOTER })),
    ...bridgeAxisElements("enhancer").map(e => ({ ...e, color: C_ENH_STROKE })),
  ];
  bctx.font = "9px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  for (const el of els) {
    const c = w(el.position);
    bctx.strokeStyle = el.color; bctx.fillStyle = el.color;
    // x-axis (bottom)
    bctx.beginPath(); bctx.moveTo(x0 + c, y1 + 2); bctx.lineTo(x0 + c, y1 + 12); bctx.stroke();
    bctx.textAlign = "center"; bctx.textBaseline = "top";
    bctx.fillText(el.label, x0 + c, y1 + 15);
    // y-axis (left)
    bctx.beginPath(); bctx.moveTo(x0 - 13, y0 + c); bctx.lineTo(x0 - 3, y0 + c); bctx.stroke();
    bctx.textAlign = "right"; bctx.textBaseline = "middle";
    bctx.fillText(el.label, x0 - 16, y0 + c);
  }
}
// TAD boundaries: dashed crosshair across the plot + "Boundary #" on both axes.
function drawBridgeBoundaries(layout) {
  const { x0, y0, x1, y1, w } = layout;
  bctx.save();
  bctx.font = "8px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  for (const b of bridgeBoundaries()) {
    const c = w(b.pos);
    bctx.strokeStyle = "rgba(15,23,42,0.30)"; bctx.setLineDash([4, 3]); bctx.lineWidth = 1;
    bctx.beginPath(); bctx.moveTo(x0 + c, y0); bctx.lineTo(x0 + c, y1); bctx.stroke();   // x boundary
    bctx.beginPath(); bctx.moveTo(x0, y0 + c); bctx.lineTo(x1, y0 + c); bctx.stroke();   // y boundary
    bctx.setLineDash([]);
    bctx.fillStyle = "rgba(15,23,42,0.7)";
    bctx.textAlign = "center"; bctx.textBaseline = "bottom";                              // x label (top)
    bctx.fillText(`${b.label} ${b.siteLabel}`, x0 + c, y0 - 3);
    bctx.save();                                                                          // y label (left, rotated)
    bctx.translate(x0 - 50, y0 + c); bctx.rotate(-Math.PI / 2);
    bctx.textAlign = "center"; bctx.textBaseline = "middle";
    bctx.fillText(`${b.label} ${b.siteLabel}`, 0, 0);
    bctx.restore();
  }
  bctx.restore();
}
// TAD region labels on the right-most y-axis: each TAD label sits at the vertical
// midpoint of its [start,end] span (not on the map itself).
function drawBridgeTadLabels(layout) {
  const { W, x1, y0, w } = layout;
  bctx.save();
  bctx.font = "600 9px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  bctx.textAlign = "right"; bctx.textBaseline = "middle";
  for (const tad of DATA.tads || []) {
    const a = Math.max(0, Math.min(N - 1, +tad.start));
    const b = Math.max(0, Math.min(N - 1, +tad.end));
    const cy = y0 + w((a + b) / 2);
    const label = tad.label || "T";
    const tw = bctx.measureText(label).width;
    bctx.fillStyle = "rgba(255,255,255,0.85)";       // pill keeps label clear of site ticks
    bctx.fillRect(W - 3 - tw - 2, cy - 7, tw + 4, 14);
    bctx.fillStyle = "rgba(37,99,235,0.9)";
    bctx.fillText(label, W - 3, cy);
  }
  bctx.restore();
}
function drawBridgeAxes(layout) {
  const { W, H, x0, y0, x1, y1, w } = layout;
  const ticks = bridgeTicks();

  bctx.save();
  bctx.font = "9px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  bctx.lineWidth = 1;

  bctx.strokeStyle = "#f1f5f9";                      // plot frame
  bctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

  bctx.strokeStyle = "#cbd5e1";                      // axis baselines
  bctx.beginPath();
  bctx.moveTo(x0, y1 + 5); bctx.lineTo(x1, y1 + 5);
  bctx.moveTo(x0 - 5, y0); bctx.lineTo(x0 - 5, y1);
  bctx.stroke();

  drawBridgeGeneBands(layout);

  bctx.strokeStyle = "rgba(203,213,225,0.35)";       // numeric grid/ticks
  bctx.fillStyle = "#94a3b8";
  bctx.textAlign = "center";
  bctx.textBaseline = "top";
  for (const site of ticks) {
    const x = x0 + w(site);
    bctx.beginPath(); bctx.moveTo(x, y0); bctx.lineTo(x, y1); bctx.stroke();
    bctx.fillText(bridgeSiteLabel(site), x, y1 + 37);
  }
  bctx.textAlign = "left";
  bctx.textBaseline = "middle";
  for (const site of ticks) {
    const y = y0 + w(site);
    bctx.beginPath(); bctx.moveTo(x0, y); bctx.lineTo(x1, y); bctx.stroke();
    bctx.fillText(bridgeSiteLabel(site), x1 + 5, y);
  }

  drawBridgeBoundaries(layout);
  drawBridgeTadLabels(layout);
  drawBridgeAxisElements(layout);

  bctx.fillStyle = "#374151";
  bctx.font = "600 10px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  bctx.textAlign = "center";
  bctx.textBaseline = "bottom";
  bctx.fillText("Lattice sites (x)", (x0 + x1) / 2, H - 4);
  bctx.save();
  bctx.translate(11, (y0 + y1) / 2);
  bctx.rotate(-Math.PI / 2);
  bctx.fillText("Lattice sites (y)", 0, 0);
  bctx.restore();

  bctx.fillStyle = "rgba(55,65,81,0.72)";
  bctx.font = "600 10px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
  bctx.textAlign = "center";
  bctx.textBaseline = "middle";
  bctx.fillText(`frame ${t} / ${T - 1} | ${DATA.frames[t].c.length} current bridges`,
                (x0 + x1) / 2, y0 - 16);
  bctx.fillStyle = "rgba(37,99,235,0.65)";
  bctx.fillText("current bridges", x1 - 72, y0 + 18);
  bctx.fillStyle = "rgba(100,116,139,0.72)";
  bctx.fillText(`visited trace 0..${t}`, x0 + 70, y1 - 18);
  bctx.restore();
}
function drawTraceFrame(f) {
  const { x0, y0, w } = bridgeLayout();
  tctx.fillStyle = "rgba(148,163,184,0.5)";
  for (const [id, l, r] of DATA.frames[f].c) {
    const lo = Math.min(l, r), hi = Math.max(l, r);
    tctx.fillRect(x0 + w(lo), y0 + w(hi), 1.2, 1.2);  // lower triangle
  }
}
function accumulateTrace(upTo) {
  if (upTo < traceUpto) { tctx.clearRect(0, 0, traceCv.width, traceCv.height); traceUpto = -1; }
  for (let f = traceUpto + 1; f <= upTo; f++) drawTraceFrame(f);
  traceUpto = upTo;
}
function drawBridge() {
  const { W, H, x0, y0, x1, y1, w } = bridgeLayout();
  accumulateTrace(t);
  bctx.clearRect(0, 0, W, H);
  bctx.drawImage(traceCv, 0, 0);
  bctx.strokeStyle = "#e5e7eb"; bctx.lineWidth = 1;   // diagonal
  bctx.beginPath(); bctx.moveTo(x0, y0); bctx.lineTo(x1, y1); bctx.stroke();
  for (const [id, l, r] of DATA.frames[t].c) {        // current bridges, upper triangle
    const lo = Math.min(l, r), hi = Math.max(l, r);
    bctx.fillStyle = cohColor(id); bctx.strokeStyle = "#fff"; bctx.lineWidth = 0.75;
    bctx.beginPath(); bctx.arc(x0 + w(hi), y0 + w(lo), 3.5, 0, Math.PI * 2); bctx.fill(); bctx.stroke();
  }
  (DATA.eps || []).forEach((pair, i) => {             // E-P target rings (upper triangle)
    const lo = Math.min(pair.e, pair.p), hi = Math.max(pair.e, pair.p);
    const px = x0 + w(hi), py = y0 + w(lo);
    bctx.strokeStyle = cohColor(i); bctx.lineWidth = 1.5; bctx.setLineDash([3, 2]);
    bctx.beginPath(); bctx.arc(px, py, 6, 0, Math.PI * 2); bctx.stroke();
    bctx.setLineDash([]);
    bctx.fillStyle = cohColor(i);
    bctx.font = "9px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
    bctx.textAlign = px > x1 - 54 ? "right" : "left";
    bctx.textBaseline = "middle";
    bctx.fillText(pair.label || `E-P${i}`, px + (px > x1 - 54 ? -8 : 8),
                  Math.max(y0 + 8, py - 8));
  });
  drawBridgeAxes({ W, H, x0, y0, x1, y1, w });
}

// ---------------------------------------------------------------- driver
function renderDynamic() { drawProximity(); drawLattice(); drawKymo(); drawBridge(); drawInsulation(); drawTadSignals(); }
function setT(v) {
  t = Math.max(0, Math.min(T - 1, v | 0));
  $slider.value = t;
  $time.textContent = `${t} / ${T - 1}`;
  renderDynamic();
}
function tick(ts) {
  if (!playing) return;
  if (!last) last = ts;
  acc += (ts - last) / 1000 * speed * 30;            // ~30 frames/sec at 1x
  last = ts;
  if (acc >= 1) { setT((t + Math.floor(acc)) % T); acc = 0; }
  requestAnimationFrame(tick);
}
$play.onclick = () => {
  playing = !playing;
  $play.innerHTML = playing ? "&#10073;&#10073;" : "&#9654;";
  if (playing) { last = 0; requestAnimationFrame(tick); }
};
$slider.oninput = e => { playing = false; $play.innerHTML = "&#9654;"; setT(+e.target.value); };
$speed.onchange = e => { speed = parseFloat(e.target.value); };

function rebuildBackgrounds() {
  buildKymoBg();
  buildTadLegend();
}
const ro = new ResizeObserver(() => renderDynamic());
ro.observe(document.getElementById("lat-wrap"));
ro.observe(document.getElementById("ep-wrap"));
rebuildBackgrounds();
setT(0);
</script>
</body>
</html>
"""


def render(data: dict, title: str = "Understanding Cohesin Bridging") -> str:
    """Substitute trajectory JSON + title into the standalone HTML template."""
    payload = json.dumps(data, separators=(",", ":"))
    return _TEMPLATE.replace("__VIEWER_DATA__", payload).replace("__TITLE__", title)
