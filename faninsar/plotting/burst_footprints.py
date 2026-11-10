"""Render scene-burst source-coverage footprints (PROPOSAL-0084)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = setup_logger(__name__)

_EDGECOLOR = "#4a4a4a"
_DEFAULT_FONTSIZE = 8.0
_MIN_OWNED_FONTSIZE = 7.0
_MIN_EXTERNAL_FONTSIZE = 4.0
_FIGURE_WIDTH_IN = 6.0
_GAP_PX = 6.0
_LEADER_PX = 12.0
_SIDE_PAD_PX = 8.0
_FIGURE_LABEL_X = 0.62


def _official_label(value: object) -> str:
    """Render an official burst ID as an integer or the literal ``None``."""
    if value is None:
        return "None"
    try:
        if pd.isna(value):
            return "None"
    except (TypeError, ValueError):
        return "None"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "None"


def _identity_text(burst: object, official: object) -> str:
    """Return the mandatory two-line identity annotation."""
    return f"Burst: {burst}\nAbsolute burst ID: {_official_label(official)}"


class _DisplayBurst(NamedTuple):
    """One unique on-figure identity with a representative centroid."""

    text: str
    cx: float
    cy: float


def _unique_display_bursts(plottable: Any) -> list[_DisplayBurst]:
    """Collapse identical two-line annotations to one display burst.

    Scientific rows stay untouched; only the drawn label set is reduced.
    Differing official IDs keep separate blocks.
    """
    groups: dict[str, list[tuple[float, float]]] = {}
    order: list[str] = []
    for _, row in plottable.iterrows():
        text = _identity_text(row.burst, row.absolute_burst_id)
        centroid = row.geometry.centroid
        point = (float(centroid.x), float(centroid.y))
        if text not in groups:
            groups[text] = []
            order.append(text)
        groups[text].append(point)
    bursts: list[_DisplayBurst] = []
    for text in order:
        points = groups[text]
        count = len(points)
        cx = sum(point[0] for point in points) / count
        cy = sum(point[1] for point in points) / count
        bursts.append(_DisplayBurst(text=text, cx=cx, cy=cy))
    bursts.sort(key=lambda item: (-item.cy, item.text))
    return bursts


def _unique_geometry_frame(plottable: Any) -> Any:
    """Return a frame with one row per distinct display geometry."""
    keep: list[Any] = []
    seen: set[bytes] = set()
    for index, geom in zip(plottable.index, plottable.geometry, strict=True):
        key = geom.wkb
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)
    return plottable.loc[keep]


def _owned_figsize(n_labels: int) -> tuple[float, float]:
    """Return a 6-inch-wide figure tall enough for *n_labels* two-line blocks."""
    row_in = (2.0 * _MIN_OWNED_FONTSIZE * 1.25 + 8.0) / 72.0
    height = max(4.5, 2.0 + n_labels * row_in)
    return (_FIGURE_WIDTH_IN, height)


def _measure_block(
    ax: Axes, text: str, fontsize: float, renderer: Any
) -> tuple[float, float]:
    """Return display-pixel width and height of a two-line identity block."""
    artist = ax.text(
        0.0,
        0.0,
        text,
        fontsize=fontsize,
        ha="left",
        va="center",
        alpha=0.0,
        zorder=-1,
    )
    bbox = artist.get_window_extent(renderer=renderer)
    artist.remove()
    return float(bbox.width), float(bbox.height)


def _max_block_size(
    ax: Axes, bursts: list[_DisplayBurst], fontsize: float, renderer: Any
) -> tuple[float, float]:
    """Return the largest identity-block width and height in pixels."""
    widths: list[float] = []
    heights: list[float] = []
    for item in bursts:
        width, height = _measure_block(ax, item.text, fontsize, renderer)
        widths.append(width)
        heights.append(height)
    return max(widths, default=0.0), max(heights, default=0.0)


def _annotate_column(
    ax: Axes,
    bursts: list[_DisplayBurst],
    *,
    x_frac: float,
    y_fracs: list[float],
    fontsize: float,
    transform: Any,
) -> None:
    """Draw leadered identity blocks in a single callout column."""
    inverse = ax.transData.inverted()
    for item, y_frac in zip(bursts, y_fracs, strict=True):
        artist = ax.text(
            x_frac,
            y_frac,
            item.text,
            transform=transform,
            ha="left",
            va="center",
            color=_EDGECOLOR,
            fontsize=fontsize,
            zorder=4,
            clip_on=False,
        )
        x_disp, y_disp = artist.get_transform().transform((x_frac, y_frac))
        x_data, y_data = inverse.transform((x_disp, y_disp))
        ax.plot(
            [item.cx, float(x_data)],
            [item.cy, float(y_data)],
            color=_EDGECOLOR,
            linewidth=0.6,
            linestyle=":",
            zorder=3,
            clip_on=False,
        )


def _place_callouts(
    ax: Axes,
    bursts: list[_DisplayBurst],
    bounds: tuple[float, float, float, float],
    *,
    owns_figure: bool,
) -> None:
    """Place a right-side callout column with non-overlapping text blocks."""
    n_labels = len(bursts)
    if n_labels == 0:
        return
    fontsize = _DEFAULT_FONTSIZE
    minx, miny, maxx, maxy = bounds
    span = max(maxx - minx, maxy - miny, 1e-6)
    pad = 0.04 * span
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_autoscale_on(False)

    fig = ax.figure
    if owns_figure:
        fig.subplots_adjust(left=0.10, right=0.60, bottom=0.12, top=0.96)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_w, label_h = _max_block_size(ax, bursts, fontsize, renderer)
    block_h = label_h + _GAP_PX

    if owns_figure:
        needed_px = n_labels * block_h + 0.20 * fig.bbox.height
        if needed_px > fig.bbox.height:
            width_in, _height_in = fig.get_size_inches()
            fig.set_size_inches(width_in, needed_px / fig.dpi, forward=True)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            _, label_h = _max_block_size(ax, bursts, fontsize, renderer)
            block_h = label_h + _GAP_PX
        block_frac = block_h / max(float(fig.bbox.height), 1.0)
        y_fracs = [0.94 - (index + 0.5) * block_frac for index in range(n_labels)]
        _annotate_column(
            ax,
            bursts,
            x_frac=_FIGURE_LABEL_X,
            y_fracs=y_fracs,
            fontsize=fontsize,
            transform=fig.transFigure,
        )
        return

    ax_bbox = ax.get_window_extent(renderer=renderer)
    while (
        n_labels * block_h + 2.0 * _SIDE_PAD_PX > ax_bbox.height
        and fontsize > _MIN_EXTERNAL_FONTSIZE
    ):
        fontsize = max(_MIN_EXTERNAL_FONTSIZE, fontsize - 0.5)
        label_w, label_h = _max_block_size(ax, bursts, fontsize, renderer)
        block_h = label_h + _GAP_PX
    x0, x1 = ax.get_xlim()
    gutter = (_LEADER_PX + label_w + _SIDE_PAD_PX) * (x1 - x0) / max(ax_bbox.width, 1.0)
    ax.set_xlim(x0, x1 + gutter)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    if n_labels == 1:
        y_fracs = [0.5]
    else:
        step = max(block_h / max(ax_bbox.height, 1.0), 1e-6)
        y_fracs = [0.96 - index * step for index in range(n_labels)]
    _annotate_column(
        ax,
        bursts,
        x_frac=1.02,
        y_fracs=y_fracs,
        fontsize=fontsize,
        transform=ax.transAxes,
    )


def plot_burst_footprints(
    footprints: Any, ax: Axes | None = None, **kwargs: Any
) -> Axes:
    """Plot source-coverage burst polygons with the P0084 two-line labels.

    Each distinct two-line identity is drawn once, with a leader to a
    representative centroid. Identical display geometries are stroked once.
    Differing official IDs or geometries stay separate. Date/track legends
    and per-date color cycles are not used.

    Parameters
    ----------
    footprints : geopandas.GeoDataFrame
        EPSG:4326 table with ``burst``, ``absolute_burst_id``, and ``geometry``.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new 6-inch-wide figure is created when omitted.
    **kwargs
        Forwarded to :meth:`geopandas.GeoDataFrame.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the drawn polygons, identity labels, and leaders.

    Raises
    ------
    ValueError
        If every row is missing plottable geographic geometry.

    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    geometry = footprints.geometry
    plottable = footprints.loc[geometry.notna() & ~geometry.is_empty].copy()
    omitted = int(len(footprints) - len(plottable))
    if omitted:
        logger.warning(
            "omitting %s burst footprint(s) with null geometry",
            omitted,
        )
    if plottable.empty:
        message = "no plottable geographic burst coverage"
        logger.error(message)
        raise ValueError(message)

    bursts = _unique_display_bursts(plottable)
    owns_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=_owned_figsize(len(bursts)))

    plot_kwargs = dict(kwargs)
    plot_kwargs.setdefault("facecolor", "none")
    plot_kwargs.pop("edgecolor", None)
    plot_kwargs.pop("color", None)
    plot_kwargs.pop("linestyle", None)
    plot_kwargs.pop("label", None)
    _unique_geometry_frame(plottable).plot(
        ax=ax,
        edgecolor=_EDGECOLOR,
        **plot_kwargs,
    )

    minx, miny, maxx, maxy = (float(value) for value in plottable.total_bounds)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    _place_callouts(
        ax,
        bursts,
        (minx, miny, maxx, maxy),
        owns_figure=owns_figure,
    )
    return ax
