"""1D LEF visualisation plugins.

Renderer signature mirrors the contact-map viz plugin::

    kymograph(colors, *, output_path, **kwargs) -> str

``colors`` is a ``(steps, N)`` array of per-site leg states produced by
:func:`...plugins.lef_dynamics.color` (0 empty, 1 free, 2 stalled, 3 CTCF).
The point is to eyeball CTCF stalls / TAD boundaries / occupancy in 1D
*before* paying for the 3D MD stage -- the pipeline equivalent of the
preview cell in ``extrusion_1D_newCode.ipynb``.
"""

from __future__ import annotations

import numpy as np


def default_kymograph(
    colors: np.ndarray,
    *,
    output_path: str,
    site_slice: tuple[int, int] | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10.0, 15.0),
    dpi: int = 150,
    title: str = "1D LEF kymograph (1 free, 2 stalled, 3 CTCF)",
    **_: object,
) -> str:
    """Render a (steps, N) leg-state array to a PNG kymograph.

    ``site_slice`` zooms onto a ``[start, end)`` window of sites (handy for
    multi-Mb lattices). Time runs down the y-axis, sites across x.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.asarray(colors)
    if data.ndim != 2:
        raise ValueError("colors must be a (steps, N) array")
    if site_slice is not None:
        data = data[:, site_slice[0] : site_slice[1]]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("site")
    ax.set_ylabel("step")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
