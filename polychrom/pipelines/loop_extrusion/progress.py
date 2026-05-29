"""Lightweight progress + ETA logging shared across pipeline stages.

All stages log through the ``polychrom.loopext`` logger; the CLI configures
the root logger to INFO by default (``--quiet`` raises it to WARNING), so
progress is high-verbosity out of the box.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

log = logging.getLogger("polychrom.loopext")


def fmt_hms(seconds: float) -> str:
    """Format a duration as ``H:MM:SS`` (or ``M:SS``); ``?`` for NaN/negative."""
    if seconds != seconds or seconds < 0:        # NaN or negative
        return "?"
    s = int(round(seconds))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


class ProgressMeter:
    """Throttled ``[stage] i/total (pct) elapsed=.. ETA=.. rate`` logger.

    Call :meth:`update` every iteration (cheap); output is rate-limited to one
    line per ``min_interval`` seconds, plus a guaranteed final line.
    """

    def __init__(self, total: int, stage: str, *, min_interval: float = 2.0):
        self.total = max(1, int(total))
        self.stage = stage
        self.min_interval = float(min_interval)
        self._start = time.time()
        self._last = 0.0
        self.n = 0

    def update(self, n: Optional[int] = None, extra: str = "") -> None:
        self.n = self.n + 1 if n is None else int(n)
        now = time.time()
        if self.n < self.total and (now - self._last) < self.min_interval:
            return
        self._last = now
        el = now - self._start
        rate = self.n / el if el > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else float("nan")
        rate_str = f"{rate:.0f}/s" if rate >= 1 else f"{rate * 60:.1f}/min"
        log.info(
            "[%s] %d/%d (%.1f%%) elapsed=%s ETA=%s %s%s",
            self.stage, self.n, self.total, 100.0 * self.n / self.total,
            fmt_hms(el), fmt_hms(eta), rate_str,
            (" " + extra) if extra else "",
        )

    def done(self, extra: str = "") -> None:
        el = time.time() - self._start
        log.info("[%s] done in %s%s", self.stage, fmt_hms(el),
                 (" " + extra) if extra else "")


def stepped_run(
    step_fn: Callable[[int], object],
    total_steps: int,
    stage: str,
    *,
    chunks: int = 20,
    min_interval: float = 2.0,
) -> None:
    """Run a long integrator burn-in in chunks, logging ETA between them.

    ``step_fn(n)`` advances ``n`` steps (e.g. ``sim.integrator.step``). Splits
    ``total_steps`` into ~``chunks`` pieces so silent multi-million-step
    relaxations report progress instead of hanging quietly.
    """
    total = int(total_steps)
    if total <= 0:
        return
    chunk = max(1, total // max(1, chunks))
    meter = ProgressMeter(total, stage, min_interval=min_interval)
    done = 0
    while done < total:
        n = min(chunk, total - done)
        step_fn(n)
        done += n
        meter.update(done)
    meter.done()
