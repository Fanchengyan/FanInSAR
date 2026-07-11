"""xarray-aware InSAR plotting for Frame products.

Four plot functions consuming :class:`xarray.DataArray` from a
:class:`~faninsar.datasets.frame.Frame`:

- :func:`plot_interferogram` — unwrapped phase (y, x) with π-formatted axes
- :func:`plot_coherence` — coherence (y, x) with a 0-1 colorbar
- :func:`plot_displacement_timeseries` — displacement (time, y, x) at a point
- :func:`plot_velocity` — velocity map (y, x)

All functions accept ``ax=None`` (create a new figure) or an existing Axes
and return the Axes for further customisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    import matplotlib.axes
    import xarray as xr

logger = setup_logger(__name__)


def plot_interferogram(
    da: xr.DataArray,
    pair: str | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "RdBu",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot an unwrapped-phase interferogram with π-formatted axes.

    Parameters
    ----------
    da : xarray.DataArray
        Phase array with dims ``(y, x)`` or ``(pair, y, x)``. When the pair
        dimension is present, *pair* selects which slice to plot.
    pair : str, optional
        Pair name to select when *da* has a ``pair`` dimension.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new figure is created when *None*.
    cmap : str
        Colormap name. Default ``"RdBu"`` (diverging, natural for phase).
    vmin, vmax : float, optional
        Color scale bounds. Default ``(-π, π)``.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    """
    import matplotlib.pyplot as plt

    from faninsar.plots.formatters import PiFormatter, PiLocator

    data = _select_pair(da, pair)
    data = _squeeze_to_2d(data)

    if ax is None:
        _, ax = plt.subplots()
    if vmin is None:
        vmin = -np.pi
    if vmax is None:
        vmax = np.pi

    im = ax.imshow(
        np.ma.masked_invalid(np.asarray(data.values)),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        **kwargs,
    )
    ax.set_title(_title(data, "Interferogram"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Phase (rad)")
    cb.ax.yaxis.set_major_formatter(PiFormatter())
    cb.ax.yaxis.set_major_locator(PiLocator())
    return ax


def plot_coherence(
    da: xr.DataArray,
    pair: str | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot a coherence map with a 0-1 colorbar.

    Parameters
    ----------
    da : xarray.DataArray
        Coherence array with dims ``(y, x)`` or ``(pair, y, x)``.
    pair : str, optional
        Pair name to select when *da* has a ``pair`` dimension.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created when *None*.
    cmap : str
        Colormap name. Default ``"viridis"``.
    vmin, vmax : float, optional
        Color scale bounds. Default ``(0, 1)``.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    data = _select_pair(da, pair)
    data = _squeeze_to_2d(data)

    if ax is None:
        _, ax = plt.subplots()
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = 1.0

    im = ax.imshow(
        np.ma.masked_invalid(np.asarray(data.values)),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        **kwargs,
    )
    ax.set_title(_title(data, "Coherence"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Coherence")
    return ax


def plot_displacement_timeseries(
    da: xr.DataArray,
    point: tuple[float, float] | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the displacement time series at a point.

    Parameters
    ----------
    da : xarray.DataArray
        Displacement array with dims ``(time, y, x)``.
    point : tuple (y, x), optional
        The ``(y, x)`` coordinate to sample. Defaults to the centre pixel.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created when *None*.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    if "time" not in da.dims:
        msg = "displacement array must have a 'time' dimension"
        raise ValueError(msg)

    ny, nx = da.sizes["y"], da.sizes["x"]
    if point is None:
        y_val = da["y"].values[ny // 2]
        x_val = da["x"].values[nx // 2]
    else:
        y_val, x_val = point

    series = da.sel(y=y_val, x=x_val, method="nearest")
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(
        np.asarray(da["time"].values),
        np.asarray(series.values),
        marker="o",
        **kwargs,
    )
    ax.set_title(f"Displacement at (y={y_val}, x={x_val})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Displacement")
    return ax


def plot_velocity(
    da: xr.DataArray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot a velocity map with a histogram-based colorbar.

    Parameters
    ----------
    da : xarray.DataArray
        Velocity array with dims ``(y, x)``.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new figure is created when *None*.
    cmap : str
        Colormap name. Default ``"RdBu_r"``.
    vmin, vmax : float, optional
        Color scale bounds. Auto-derived from percentiles when *None*.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    data = _squeeze_to_2d(da)
    arr = np.ma.masked_invalid(np.asarray(data.values))

    if ax is None:
        _, ax = plt.subplots()
    if vmin is None or vmax is None:
        finite = arr.compressed()
        if finite.size > 0:
            lo, hi = np.nanpercentile(finite, [2, 98])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        else:  # pragma: no cover - defensive
            vmin = -1.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", **kwargs)
    ax.set_title("Velocity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Velocity")
    return ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_pair(
    da: xr.DataArray, pair: str | None
) -> xr.DataArray:
    """Select a pair slice if the pair dimension is present."""
    if "pair" in da.dims:
        if pair is None:
            return da.isel(pair=0)
        return da.sel(pair=pair)
    return da


def _squeeze_to_2d(da: xr.DataArray) -> xr.DataArray:
    """Drop any singleton non-spatial dimensions (e.g. band)."""
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    return da


def _title(da: xr.DataArray, default: str) -> str:
    """Build a plot title from the dataarray name and optional pair coord."""
    name = da.name or default
    if "pair" in da.coords:
        return f"{name}: {da['pair'].item()}"
    return str(name)
