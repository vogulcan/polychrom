"""Stage 4 driver: standalone interactive bridging viewer.

Reads ``LEFPositions.h5`` plus the ``lef`` stage topology (for CTCF / gene /
enhancer annotations) and emits a single self-contained HTML file that
animates the 1D loop-extrusion trajectory: playback bar, E-P proximity,
1D lattice + arcs + optional RNAPII, kymograph, and a bridge map. Reproduces
the layout of abdenlab.org/cohesin-bridging for our own 1D pipeline output.

The per-frame "effective E-P distance" is the graph shortest path on a line
graph (backbone edge = 1 per site) augmented with one chord edge per cohesin
loop (cost ``bridge_cost``): a loop bracketing the E-P pair short-circuits the
backbone, so the distance drops as loops form.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from . import _viewer_template
from .config import LEFConfig, ViewerConfig, resolve_plugin
from .progress import ProgressMeter, log


# --------------------------------------------------------------------------- #
# Effective E-P distance: shortest path on (backbone line + cohesin chords).
# --------------------------------------------------------------------------- #
def effective_distance(
    arcs: List[Tuple[int, int]],
    src: int,
    dst: int,
    chain_length: int,
    bridge_cost: float = 1.0,
) -> Optional[float]:
    """Shortest path from ``src`` to ``dst`` over the reduced graph.

    Nodes are ``src``, ``dst`` and every cohesin leg position. Consecutive
    nodes on the same chain are joined by a backbone edge equal to their site
    gap; each cohesin contributes a chord edge of weight ``bridge_cost``.
    Returns ``None`` when unreachable (e.g. the pair sits on different chains
    with no bridging path).
    """
    if src == dst:
        return 0.0
    nodes = sorted({src, dst, *(p for a in arcs for p in a)})
    idx = {p: i for i, p in enumerate(nodes)}
    adj: List[List[Tuple[int, float]]] = [[] for _ in nodes]
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if a // chain_length == b // chain_length:       # same chain backbone
            w = float(b - a)
            adj[i].append((i + 1, w))
            adj[i + 1].append((i, w))
    for l, r in arcs:
        if l == r:
            continue
        il, ir = idx[l], idx[r]
        adj[il].append((ir, float(bridge_cost)))
        adj[ir].append((il, float(bridge_cost)))

    s, d = idx[src], idx[dst]
    dist = [float("inf")] * len(nodes)
    dist[s] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, s)]
    while pq:
        du, u = heapq.heappop(pq)
        if u == d:
            return du
        if du > dist[u]:
            continue
        for v, w in adj[u]:
            nv = du + w
            if nv < dist[v]:
                dist[v] = nv
                heapq.heappush(pq, (nv, v))
    return None if dist[d] == float("inf") else dist[d]


# --------------------------------------------------------------------------- #
# Annotations re-derived from the lef-stage topology.
# --------------------------------------------------------------------------- #
def derive_annotations(
    lef_cfg: LEFConfig,
) -> Tuple[List[int], List[dict], List[dict], List[dict], List[dict]]:
    """Return ``(ctcf_sites, elements, eps, genes, tads)`` from topology/config."""
    topology_fn = resolve_plugin(lef_cfg.plugins.topology)
    args = topology_fn(lef_cfg, **lef_cfg.topology_kwargs)

    ctcf = set()
    capture = args.get("ctcfCapture", {})
    for side in capture.values():
        ctcf.update(int(s) for s in side.keys())
    ctcf_sites = sorted(ctcf)

    elements: List[dict] = []
    eps: List[dict] = []
    genes: List[dict] = []
    enhancer_ids: Dict[int, int] = {}
    for g in args.get("genes", []):
        gene_id = int(g.gene_id)
        tss = int(g.tss)
        tes = int(g.tes)
        genes.append({"geneId": gene_id, "tss": tss, "tes": tes, "label": f"G{gene_id}"})
        # direction (+1 if TSS<TES, -1 otherwise) drives the promoter chevron
        # so it points the way transcription runs in the 1D lattice.
        elements.append({
            "position": tss, "type": "promoter", "label": f"P{gene_id}",
            "direction": int(g.direction),
        })
        if getattr(g, "enhancer_pos", None) is not None:
            enhancer_pos = int(g.enhancer_pos)
            if enhancer_pos not in enhancer_ids:
                enhancer_ids[enhancer_pos] = len(enhancer_ids)
                elements.append({
                    "position": enhancer_pos,
                    "type": "enhancer",
                    "label": f"E{enhancer_ids[enhancer_pos]}",
                })
            enhancer_id = enhancer_ids[enhancer_pos]
            eps.append({
                "e": enhancer_pos,
                "p": tss,
                "label": f"E{enhancer_id}-P{gene_id}",
                "genomic": abs(tss - enhancer_pos),
            })
    tads = derive_tads(lef_cfg)
    return ctcf_sites, elements, eps, genes, tads


def derive_tads(lef_cfg: LEFConfig) -> List[dict]:
    """Build original TAD intervals from ``tad_positions`` for every chain."""
    inner = sorted({
        int(p) for p in lef_cfg.topology_kwargs.get("tad_positions", [])
        if 0 < int(p) < lef_cfg.chain_length
    })
    rel_bounds = [0, *inner, lef_cfg.chain_length]
    tads: List[dict] = []
    for chain_idx in range(lef_cfg.num_chains):
        offset = chain_idx * lef_cfg.chain_length
        for tad_idx, (start, end) in enumerate(zip(rel_bounds[:-1], rel_bounds[1:])):
            label = f"T{tad_idx}" if lef_cfg.num_chains == 1 else f"C{chain_idx}:T{tad_idx}"
            tads.append({
                "chain": chain_idx,
                "tad": tad_idx,
                "label": label,
                "start": offset + int(start),
                "end": offset + int(end),
                "size": int(end - start),
            })
    return tads


# --------------------------------------------------------------------------- #
# Frame assembly.
# --------------------------------------------------------------------------- #
def _frame_indices(traj_len: int, stride: int, max_frames: int) -> np.ndarray:
    idx = np.arange(0, traj_len, max(1, int(stride)))
    if max_frames and len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, int(max_frames)).astype(int)]
    return idx


def _in_interval(pos: int, start: int, end: int) -> bool:
    return start <= pos < end


def _display_tads(tads: List[dict], start: int, end: int) -> List[dict]:
    return [t for t in tads if int(t["end"]) > start and int(t["start"]) < end]


def _insulation_window(cfg: ViewerConfig) -> int:
    return max(1, int(cfg.insulation_score_window))


def _cross_window_count(arcs: List[Tuple[int, int]], boundary: int, window: int) -> int:
    left_start = boundary - window
    left_end = boundary
    right_start = boundary
    right_end = boundary + window
    return sum(
        (
            _in_interval(l, left_start, left_end) and _in_interval(r, right_start, right_end)
        ) or (
            _in_interval(r, left_start, left_end) and _in_interval(l, right_start, right_end)
        )
        for l, r in arcs
    )


def _log2_ratio_row(scores: np.ndarray) -> List[Optional[float]]:
    """log2(score / mean_positive) per position; non-positive -> None (gap).

    Mean is taken over the positive entries of the row, so a boundary with the
    typical crossing rate sits at 0, stronger boundaries (fewer crossings) dip
    negative. The window normalisation cancels in the ratio, so raw counts work.
    """
    arr = np.asarray(scores, dtype=float)
    positive = arr[arr > 0]
    if positive.size == 0:
        return [None] * len(arr)
    mean = float(positive.mean())
    return [
        round(float(np.log2(v / mean)), 6) if v > 0 else None
        for v in arr
    ]


def build_insulation_scores(
    sampled_arcs: List[List[Tuple[int, int]]],
    start: int,
    end: int,
    window: int,
) -> dict:
    """Compute dynamic insulation lineplots from bridge-map trace contacts.

    Per-frame scores use only current cohesin bridge contacts. Cumulative
    scores use the visited trace, i.e. the summed bridge contacts up to frame t.
    Each row is reported on a log2 scale as ``log2(score / mean)`` (mean over
    positive positions in that row); positions with no crossings are ``None``.
    """
    lattice_size = end - start
    if lattice_size < 2 * window or not sampled_arcs:
        return {"window": int(window), "positions": [], "frame": [], "cumulative": []}

    rel_positions = list(range(window, lattice_size - window + 1))
    boundaries = [start + p for p in rel_positions]
    cumulative_counts = np.zeros(len(boundaries), dtype=np.int32)
    frame_scores: List[List[Optional[float]]] = []
    cumulative_scores: List[List[Optional[float]]] = []

    for arcs in sampled_arcs:
        counts = np.array([
            _cross_window_count(arcs, boundary, window) for boundary in boundaries
        ], dtype=np.int32)
        cumulative_counts += counts
        frame_scores.append(_log2_ratio_row(counts))
        cumulative_scores.append(_log2_ratio_row(cumulative_counts))

    return {
        "window": int(window),
        "positions": rel_positions,
        "frame": frame_scores,
        "cumulative": cumulative_scores,
    }


def _count_intra_inter(
    arcs: List[Tuple[int, int]], start: int, end: int, flank: int
) -> Tuple[int, int]:
    """Intra- and inter-TAD bridge counts for one TAD interval ``[start, end)``.

    Intra contacts have both cohesin legs inside the TAD. Inter contacts bridge
    the TAD to one of its equal-sized flanks ``[start-flank, start)`` or
    ``[end, end+flank)`` -- one leg in the TAD, the other on a flank.
    """
    left_lo, left_hi = start - flank, start
    right_lo, right_hi = end, end + flank
    intra = inter = 0
    for l, r in arcs:
        l_in = _in_interval(l, start, end)
        r_in = _in_interval(r, start, end)
        if l_in and r_in:
            intra += 1
            continue
        l_flank = _in_interval(l, left_lo, left_hi) or _in_interval(l, right_lo, right_hi)
        r_flank = _in_interval(r, left_lo, left_hi) or _in_interval(r, right_lo, right_hi)
        if (l_in and r_flank) or (r_in and l_flank):
            inter += 1
    return intra, inter


def build_tad_signals(
    sampled_arcs: List[List[Tuple[int, int]]], tads: List[dict]
) -> dict:
    """Per-TAD cumulative intra/inter contact signals + intra:inter ratio.

    Counts accumulate over sampled frames, so each frame's value is the running
    total up to that frame; ``ratio`` is cumulative intra / cumulative inter
    (``None`` until inter > 0). Flank size equals the original TAD size. All
    series are frame-major: ``series[frame][tad_idx]``.
    """
    meta = [
        {"label": t["label"], "size": int(t["size"])}
        for t in tads
    ]
    intra_series: List[List[int]] = []
    inter_series: List[List[int]] = []
    ratio_series: List[List[Optional[float]]] = []
    cum_intra = np.zeros(len(tads), dtype=np.int64)
    cum_inter = np.zeros(len(tads), dtype=np.int64)
    for arcs in sampled_arcs:
        for k, t in enumerate(tads):
            intra, inter = _count_intra_inter(
                arcs, int(t["start"]), int(t["end"]), int(t["size"])
            )
            cum_intra[k] += intra
            cum_inter[k] += inter
        intra_series.append([int(v) for v in cum_intra])
        inter_series.append([int(v) for v in cum_inter])
        ratio_series.append([
            round(float(cum_intra[k] / cum_inter[k]), 4) if cum_inter[k] > 0 else None
            for k in range(len(tads))
        ])
    return {
        "tads": meta,
        "intra": intra_series,
        "inter": inter_series,
        "ratio": ratio_series,
    }


def build_payload(
    positions: np.ndarray,
    cfg: ViewerConfig,
    lef_cfg: LEFConfig,
    rnapii_positions: Optional[np.ndarray] = None,
    rnapii_states: Optional[np.ndarray] = None,
) -> dict:
    """Assemble the JSON payload embedded into the viewer HTML."""
    traj_len, n_lefs, _ = positions.shape
    chain_length = lef_cfg.chain_length
    full_n = lef_cfg.num_sites

    ctcf_sites, elements, eps, genes, tads = derive_annotations(lef_cfg)

    # Display window (defaults to whole lattice).
    start = 0 if cfg.site_start is None else int(cfg.site_start)
    end = full_n if cfg.site_end is None else int(cfg.site_end)
    lattice_size = end - start

    def in_win(p: int) -> bool:
        return start <= p < end

    # Explicit config pairs are appended when topology supplies none.
    if not eps and cfg.ep_pairs:
        for p0 in cfg.ep_pairs:
            e, p = int(p0["e"]), int(p0["p"])
            eps.append({"e": e, "p": p, "label": p0.get("label", "E-P"),
                        "genomic": abs(p - e)})
    display_eps = [pair for pair in eps if in_win(pair["e"]) and in_win(pair["p"])]

    frame_idx = _frame_indices(traj_len, cfg.stride, cfg.max_frames)

    # Cohesin id reassignment: a leg moves <=1 site per dynamics step, so between
    # two sampled frames it travels at most their index gap. A jump beyond that
    # means the cohesin unloaded and reloaded -> assign a fresh colour (= a new
    # contiguous extrusion track in the kymograph).
    ids = list(range(n_lefs))
    next_id = n_lefs
    prev: Optional[np.ndarray] = None
    prev_fi: Optional[int] = None

    frames: List[dict] = []
    sampled_arcs: List[List[Tuple[int, int]]] = []
    log.info(
        "[viewer] building 1D payload: %d/%d frames sampled, %d E-P pairs "
        "(bridge shortest-paths per frame)",
        len(frame_idx), traj_len, len(display_eps),
    )
    frame_meter = ProgressMeter(len(frame_idx), "viewer:frames")
    for fi in frame_idx:
        fi = int(fi)
        legs = positions[fi]                      # (n_lefs, 2)
        if prev is not None:
            jump_thr = (fi - prev_fi) + 2
            moved = np.abs(legs - prev).max(axis=1)
            for j in np.flatnonzero(moved > jump_thr):
                ids[j] = next_id
                next_id += 1
        prev = legs
        prev_fi = fi

        arcs = [(int(l), int(r)) for l, r in legs]            # original coords
        sampled_arcs.append(arcs)
        # Per-pair effective distance (None entry when a pair is unreachable).
        s_eff = []
        for pair in display_eps:
            d = effective_distance(arcs, pair["e"], pair["p"], chain_length, cfg.bridge_cost)
            s_eff.append(None if d is None else round(float(d), 1))

        coh = [
            [int(ids[j]), int(l) - start, int(r) - start]
            for j, (l, r) in enumerate(legs)
            if in_win(int(l)) and in_win(int(r))
        ]
        rnap = []
        if rnapii_positions is not None and fi < rnapii_positions.shape[0]:
            states = (
                rnapii_states[fi]
                if rnapii_states is not None and fi < rnapii_states.shape[0]
                else None
            )
            for j, (pos, gene_id) in enumerate(rnapii_positions[fi]):
                pos = int(pos)
                if pos < 0 or not in_win(pos):
                    continue
                state = int(states[j]) if states is not None and j < len(states) else -1
                rnap.append([pos - start, int(gene_id), state])
        frames.append({"c": coh, "s": s_eff, "r": rnap})
        frame_meter.update()
    frame_meter.done()
    log.info("[viewer] computing insulation + TAD signals + heatmap")

    gene_spans = []
    for gene in genes:
        lo, hi = sorted((int(gene["tss"]), int(gene["tes"])))
        if hi < start or lo >= end:
            continue
        gene_spans.append({
            **gene,
            "start": max(lo, start) - start,
            "end": min(hi, end - 1) - start,
        })

    display_tads = _display_tads(tads, start, end)
    tad_spans = [
        {
            **tad,
            "start": max(int(tad["start"]), start) - start,
            "end": min(int(tad["end"]), end) - start,
        }
        for tad in display_tads
    ]
    insulation = build_insulation_scores(
        sampled_arcs,
        start,
        end,
        min(_insulation_window(cfg), max(1, lattice_size // 2)),
    )
    tad_signals = build_tad_signals(sampled_arcs, display_tads)

    payload = {
        "title": "Understanding Cohesin Bridging",
        "latticeSize": int(lattice_size),
        "siteOffset": int(start),
        "ctcfSites": [s - start for s in ctcf_sites if in_win(s)],
        "elements": [
            {**e, "position": e["position"] - start} for e in elements if in_win(e["position"])
        ],
        "genes": gene_spans,
        "tads": tad_spans,
        "insulation": insulation,
        "tadSignals": tad_signals,
        "eps": [{**p, "e": p["e"] - start, "p": p["p"] - start} for p in display_eps],
        "frames": frames,
    }
    return payload


# --------------------------------------------------------------------------- #
# Companion exports: numeric cumulative heatmap + annotation positions.
# --------------------------------------------------------------------------- #
def build_visited_heatmap(payload: dict) -> np.ndarray:
    """Cumulative bridge-contact matrix over every sampled frame.

    Symmetric ``(latticeSize, latticeSize)`` integer array: for each cohesin in
    each frame, increment ``[lo, hi]`` and ``[hi, lo]`` in display coordinates.
    The numeric form of the viewer's accumulated lower-triangle "visited trace".
    """
    n = int(payload["latticeSize"])
    mat = np.zeros((n, n), dtype=np.int64)
    for frame in payload["frames"]:
        for _id, l, r in frame["c"]:
            lo, hi = (l, r) if l <= r else (r, l)
            if 0 <= lo < n and 0 <= hi < n:
                mat[lo, hi] += 1
                if lo != hi:
                    mat[hi, lo] += 1
    return mat


def build_elements_export(payload: dict) -> dict:
    """Annotation positions (CTCF / elements / genes / TADs / E-P) for the heatmap.

    Coordinates are display-window relative; ``siteOffset`` recovers absolute
    lattice sites (``absolute = relative + siteOffset``).
    """
    keys = ("title", "latticeSize", "siteOffset",
            "ctcfSites", "elements", "genes", "tads", "eps")
    return {k: payload[k] for k in keys}


def run(cfg: ViewerConfig, lef_cfg: LEFConfig) -> Path:
    """Build the interactive viewer HTML from ``LEFPositions.h5``.

    If the positions file is missing, the 1D ``lef`` stage is run first from
    ``lef_cfg`` to generate it -- so ``viewer config.yaml`` works on a fresh
    config without a separate ``lef`` invocation.
    """
    from . import lef as lef_stage

    h5_path = Path(cfg.lef_positions_path)
    if not h5_path.exists():
        print(f"[viewer]   {h5_path} not found -> running lef stage to build it")
        h5_path = lef_stage.run(lef_cfg)          # writes lef_cfg.output_path
    log.info("[viewer] loading positions from %s", h5_path)
    with h5py.File(h5_path, "r") as fh:
        positions = fh["positions"][:]            # (T, L, 2)
        rnapii_positions = fh["rnapii_positions"][:] if "rnapii_positions" in fh else None
        rnapii_states = fh["rnapii_states"][:] if "rnapii_states" in fh else None

    payload = build_payload(positions, cfg, lef_cfg, rnapii_positions, rnapii_states)
    html = _viewer_template.render(payload, title=payload["title"])

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    # Companion exports alongside the HTML: numeric cumulative visited heatmap
    # (.npy) and annotation positions (.json), for downstream analysis.
    heatmap_path = (
        Path(cfg.heatmap_output_path) if cfg.heatmap_output_path
        else out_path.with_name(out_path.stem + "_visited_heatmap.npy")
    )
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(heatmap_path, build_visited_heatmap(payload))
    print(f"[viewer]   wrote {heatmap_path}")

    elements_path = (
        Path(cfg.elements_output_path) if cfg.elements_output_path
        else out_path.with_name(out_path.stem + "_elements.json")
    )
    elements_path.parent.mkdir(parents=True, exist_ok=True)
    elements_path.write_text(
        json.dumps(build_elements_export(payload), indent=2), encoding="utf-8"
    )
    print(f"[viewer]   wrote {elements_path}")
    return out_path
