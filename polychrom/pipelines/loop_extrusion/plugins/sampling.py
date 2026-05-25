"""Default contact-map samplers + post-processors.

Sampler signature::

    sampler(uris, *, cfg, **kwargs) -> np.ndarray

Post-process signature::

    post_process(contact_map, **kwargs) -> np.ndarray
"""

from __future__ import annotations

import numpy as np

from .... import contactmaps


def monomer_resolution_sampler(uris, *, cfg, **kwargs) -> np.ndarray:
    """Per-monomer contact map averaged over multiple subchain starts."""
    return contactmaps.monomerResolutionContactMapSubchains(
        filenames=uris,
        mapStarts=list(cfg.map_starts),
        mapN=cfg.map_size,
        cutoff=cfg.cutoff,
        n=cfg.num_processes,
        verbose=cfg.verbose,
        **kwargs,
    )


def binned_sampler(uris, *, cfg, bin_size: int = 5, **kwargs) -> np.ndarray:
    """Binned (coarse-grained) contact map."""
    return contactmaps.binnedContactMap(
        filenames=uris,
        chains=[(0, cfg.map_size, False)],
        binSize=bin_size,
        cutoff=cfg.cutoff,
        n=cfg.num_processes,
        **kwargs,
    )


def observed_over_expected(contact_map: np.ndarray, **_: object) -> np.ndarray:
    """Diagonal-wise normalise a contact map."""
    cm = np.asarray(contact_map, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("contact_map must be a square matrix")
    n = cm.shape[0]
    result = np.full_like(cm, np.nan, dtype=float)
    for offset in range(n):
        diag = np.diagonal(cm, offset=offset)
        expected = diag.mean()
        if expected <= 0:
            continue
        normalized = diag / expected
        rows = np.arange(n - offset)
        cols = rows + offset
        result[rows, cols] = normalized
        if offset:
            result[cols, rows] = normalized
    return result


def iterative_correction(
    contact_map: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1.0e-6,
    ignore_diagonals: int = 1,
    min_row_sum: float = 0.0,
    **_: object,
) -> np.ndarray:
    """Balance a square contact map by symmetric iterative correction.

    The correction estimates one multiplicative bias per row/column and
    returns ``contact_map * outer(bias, bias)``. Near-diagonal pixels can be
    ignored while fitting the bias, but the final bias is applied to the
    original matrix so downstream O/E still has full-diagonal support.
    """
    cm = np.asarray(contact_map, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("contact_map must be a square matrix")
    if np.any(cm[np.isfinite(cm)] < 0):
        raise ValueError("contact_map must be non-negative")

    n = cm.shape[0]
    fit = np.where(np.isfinite(cm), cm, 0.0).astype(float, copy=True)
    if ignore_diagonals >= 0:
        idx = np.arange(n)
        for offset in range(int(ignore_diagonals) + 1):
            rows = idx[: n - offset]
            cols = rows + offset
            fit[rows, cols] = 0.0
            fit[cols, rows] = 0.0

    row_sum = fit.sum(axis=1)
    active = row_sum > float(min_row_sum)
    if active.sum() == 0:
        out = cm.copy()
        out[~np.isfinite(cm)] = np.nan
        return out

    sub = fit[np.ix_(active, active)]
    bias = np.ones(sub.shape[0], dtype=float)
    for _ in range(int(max_iter)):
        balanced = sub * np.outer(bias, bias)
        sums = balanced.sum(axis=1)
        valid = sums > 0
        if not np.all(valid):
            bias[~valid] = 0.0
            sums = np.where(valid, sums, np.nan)
        target = np.nanmedian(sums)
        if not np.isfinite(target) or target <= 0:
            break
        rel = np.nanmax(np.abs(sums / target - 1.0))
        if rel <= tol:
            break
        scale = np.ones_like(bias)
        scale[valid] = np.sqrt(target / sums[valid])
        bias *= scale

    full_bias = np.zeros(n, dtype=float)
    full_bias[active] = bias
    corrected = cm * np.outer(full_bias, full_bias)
    corrected[~np.isfinite(cm)] = np.nan
    corrected[~active, :] = np.nan
    corrected[:, ~active] = np.nan
    return corrected


def balanced_observed_over_expected(
    contact_map: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1.0e-6,
    ignore_diagonals: int = 1,
    min_row_sum: float = 0.0,
    **kwargs: object,
) -> np.ndarray:
    """Iterative-correction balancing followed by diagonal O/E normalisation."""
    balanced = iterative_correction(
        contact_map,
        max_iter=max_iter,
        tol=tol,
        ignore_diagonals=ignore_diagonals,
        min_row_sum=min_row_sum,
        **kwargs,
    )
    return observed_over_expected(balanced)


def default_oe_heatmap(
    matrix: np.ndarray,
    *,
    output_path: str,
    log: bool = True,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: tuple[float, float] = (10.0, 10.0),
    dpi: int = 150,
    title: str = "Observed / expected contact map",
    **_: object,
) -> str:
    """Render an O/E matrix to a PNG.

    By default plots log(O/E) on a diverging colormap. Set ``log=False`` to
    plot the raw matrix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.asarray(matrix, dtype=float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.log(data)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("monomer index")
    ax.set_ylabel("monomer index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
