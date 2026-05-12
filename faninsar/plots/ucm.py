"""UCM panel plotting utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap

from faninsar.logging import setup_logger

from .cm import cmaps
from .hist_colorbar import HistColorbar

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    class _UcmAxes(TypedDict, total=False):
        """Axes created for named UCM panels."""

        heatmap: Axes
        spatial: Axes
        temporal: Axes
        surface_3d: Axes3D


logger = setup_logger(__name__)

_PanelName: TypeAlias = Literal["heatmap", "spatial", "temporal", "surface_3d"]
_MosaicLayout: TypeAlias = list[list[_PanelName]]
_ColorMapLike: TypeAlias = str | Colormap
_LabelName: TypeAlias = Literal[
    "xlabel",
    "ylabel",
    "zlabel",
    "legend_title",
]
_PanelText: TypeAlias = str | Mapping[_PanelName, str | None] | None

DEFAULT_UCM_MOSAIC: _MosaicLayout = [
    ["heatmap", "spatial"],
    ["temporal", "surface_3d"],
]
DEFAULT_UCM_VALUE_LABEL = "Velocity (mm/yr)"

_UCM_PANEL_NAMES: frozenset[_PanelName] = frozenset({
    "heatmap",
    "spatial",
    "temporal",
    "surface_3d",
})


@overload
def _resolve_ucm_axes(
    ds_ucm: xr.DataArray,
    ax: Axes | None = None,
    *,
    projection: Literal["rectilinear"] = "rectilinear",
    figure_kwargs: dict[str, object] | None = None,
) -> tuple[xr.DataArray, Axes]: ...


@overload
def _resolve_ucm_axes(
    ds_ucm: xr.DataArray,
    ax: Axes3D | None = None,
    *,
    projection: Literal["3d"],
    figure_kwargs: dict[str, object] | None = None,
) -> tuple[xr.DataArray, Axes3D]: ...


def _resolve_ucm_axes(
    ds_ucm: xr.DataArray,
    ax: Axes | Axes3D | None = None,
    *,
    projection: Literal["rectilinear", "3d"] = "rectilinear",
    figure_kwargs: dict[str, object] | None = None,
) -> tuple[xr.DataArray, Axes | Axes3D]:
    """Resolve UCM data and Matplotlib axes.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        UCM values.
    ax : matplotlib.axes.Axes, Axes3D, or None
        Target axes. If None, a new axes is created.
    projection : {"rectilinear", "3d"}, default "rectilinear"
        Projection used when a new axes is created.
    figure_kwargs : dict or None, optional
        Keyword arguments passed to :func:`matplotlib.pyplot.subplots` when
        ``ax`` is None. If ``dpi`` is not provided, it defaults to 300.

    Returns
    -------
    tuple[xarray.DataArray, matplotlib.axes.Axes or Axes3D]
        Resolved UCM data and target axes.

    Raises
    ------
    ValueError
        If ``projection`` is not supported.

    """
    if ax is not None:
        return ds_ucm, ax

    subplot_kwargs = {} if figure_kwargs is None else figure_kwargs.copy()
    subplot_kwargs.setdefault("dpi", 300)
    if projection == "rectilinear":
        _, created_ax = plt.subplots(**subplot_kwargs)
        return ds_ucm, created_ax
    if projection == "3d":
        subplot_kw = dict(
            cast("Mapping[str, object]", subplot_kwargs.pop("subplot_kw", {}))
        )
        subplot_kw["projection"] = "3d"
        _, created_ax = plt.subplots(subplot_kw=subplot_kw, **subplot_kwargs)
        return ds_ucm, cast("Axes3D", created_ax)

    logger.error("Unsupported UCM axes projection: %s.", projection)
    msg = f"projection must be 'rectilinear' or '3d', got {projection!r}."
    raise ValueError(msg)


def _default_ucm_labels(
    quantity_label: str,
) -> dict[_PanelName, dict[_LabelName, str | None]]:
    """Create default panel labels for a UCM summary plot.

    Parameters
    ----------
    quantity_label : str
        Value label used for velocity axes.

    Returns
    -------
    dict[str, dict[str, str or None]]
        Default labels keyed by panel name and label field.

    """
    temporal_baseline_label = "Maximum Temporal Baseline (days)"
    resolution_label = "Resolution (m)"
    return {
        "heatmap": {
            "xlabel": temporal_baseline_label,
            "ylabel": resolution_label,
            "zlabel": None,
            "legend_title": None,
        },
        "spatial": {
            "xlabel": quantity_label,
            "ylabel": resolution_label,
            "zlabel": None,
            "legend_title": "Days",
        },
        "temporal": {
            "xlabel": temporal_baseline_label,
            "ylabel": quantity_label,
            "zlabel": None,
            "legend_title": resolution_label,
        },
        "surface_3d": {
            "xlabel": temporal_baseline_label,
            "ylabel": resolution_label,
            "zlabel": quantity_label,
            "legend_title": None,
        },
    }


def _resolve_panel_text(
    panel_name: _PanelName,
    label_name: _LabelName,
    defaults: Mapping[_PanelName, Mapping[_LabelName, str | None]],
    override: _PanelText = None,
) -> str | None:
    """Resolve a label value from defaults and optional user overrides.

    Parameters
    ----------
    panel_name : {"heatmap", "spatial", "temporal", "surface_3d"}
        Panel whose label is resolved.
    label_name : {"xlabel", "ylabel", "zlabel", "legend_title"}
        Label field to resolve.
    defaults : mapping
        Default labels keyed by panel name and label field.
    override : str, mapping, or None, optional
        User override. A string is applied to all panels, a mapping is applied by
        panel name, and None keeps the default value.

    Returns
    -------
    str or None
        Resolved label value.

    """
    default_value = defaults[panel_name][label_name]
    if isinstance(override, Mapping):
        return override.get(panel_name, default_value)
    if override is not None:
        return override
    return default_value


def _resolve_panel_bool(
    panel_name: _PanelName,
    default_value: bool,
    override: bool | Mapping[_PanelName, bool] | None = None,
) -> bool:
    """Resolve a boolean panel option from defaults and optional overrides.

    Parameters
    ----------
    panel_name : {"heatmap", "spatial", "temporal", "surface_3d"}
        Panel whose option is resolved.
    default_value : bool
        Default option value.
    override : bool, mapping, or None, optional
        User override. A boolean is applied to all panels, a mapping is applied
        by panel name, and None keeps the default value.

    Returns
    -------
    bool
        Resolved boolean value.

    """
    if isinstance(override, Mapping):
        return override.get(panel_name, default_value)
    if override is not None:
        return override
    return default_value


def _prepare_ucm_data(ds_ucm: xr.DataArray) -> xr.DataArray:
    """Validate and transpose UCM data for plotting.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        UCM values with ``res`` and ``day`` dimensions.

    Returns
    -------
    xarray.DataArray
        Data transposed to ``("res", "day")``.

    Raises
    ------
    ValueError
        If the input data does not contain the required dimensions.

    """
    missing_dimensions = {"res", "day"}.difference(ds_ucm.dims)
    if missing_dimensions:
        logger.error(
            "UCM data is missing required dimensions: %s.",
            sorted(missing_dimensions),
        )
        msg = "ds_ucm must contain 'res' and 'day' dimensions."
        raise ValueError(msg)
    return ds_ucm.transpose("res", "day")


def _resolve_color_limits(
    ds_ucm: xr.DataArray,
    vmin: float | None = None,
    vmax: float | None = None,
    padding_fraction: float = 0.1,
) -> tuple[float, float]:
    """Resolve padded UCM color limits.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        UCM values.
    vmin : float or None, optional
        Optional lower limit. If omitted, the data minimum is used.
    vmax : float or None, optional
        Optional upper limit. If omitted, the data maximum is used.
    padding_fraction : float, default 0.1
        Fraction of the value range added to both sides of the limits.

    Returns
    -------
    tuple[float, float]
        Resolved lower and upper color limits.

    Raises
    ------
    ValueError
        If no finite values are available or if limits are invalid.

    """
    finite_values = np.asarray(ds_ucm.values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0 and (vmin is None or vmax is None):
        logger.error("Cannot resolve UCM color limits because no finite values exist.")
        msg = "Cannot resolve color limits from all-NaN or non-finite UCM values."
        raise ValueError(msg)

    lower_limit = float(np.nanmin(finite_values)) if vmin is None else float(vmin)
    upper_limit = float(np.nanmax(finite_values)) if vmax is None else float(vmax)
    if not np.isfinite(lower_limit) or not np.isfinite(upper_limit):
        logger.error(
            "Invalid non-finite UCM color limits received: vmin=%s, vmax=%s.",
            lower_limit,
            upper_limit,
        )
        msg = "vmin and vmax must be finite values."
        raise ValueError(msg)
    if lower_limit > upper_limit:
        logger.error(
            "Invalid UCM color limits received: vmin=%s is greater than vmax=%s.",
            lower_limit,
            upper_limit,
        )
        msg = "vmin must be less than or equal to vmax."
        raise ValueError(msg)

    value_range = upper_limit - lower_limit
    if value_range == 0.0:
        value_range = abs(upper_limit) if upper_limit != 0.0 else 1.0
    padding = value_range * padding_fraction
    return lower_limit - padding, upper_limit + padding


def _validate_mosaic(mosaic: _MosaicLayout) -> None:
    """Validate UCM panel labels in a subplot mosaic.

    Parameters
    ----------
    mosaic : list[list[_PanelName]] or tuple[tuple[_PanelName, ...], ...]
        Matplotlib subplot mosaic layout.

    Raises
    ------
        If the mosaic contains an unknown UCM panel name.

    """
    invalid_panel_names = {
        panel_name
        for row in mosaic
        for panel_name in row
        if panel_name != "." and panel_name not in _UCM_PANEL_NAMES
    }
    if invalid_panel_names:
        logger.error("Unknown UCM mosaic panel names: %s.", sorted(invalid_panel_names))
        msg = (
            "mosaic contains unknown panel names. Expected labels are "
            f"{sorted(_UCM_PANEL_NAMES)} or '.'."
        )
        raise ValueError(msg)


def _profile_palette(color_count: int) -> list[tuple[float, float, float, float]]:
    """Create profile line colors from the default UCM profile colormap.

    Parameters
    ----------
    color_count : int
        Number of colors to create.

    Returns
    -------
    list[tuple[float, float, float, float]]
        RGBA colors sampled from ``RdYlGn_r``.

    """
    if color_count <= 0:
        return []
    palette_cmap = plt.get_cmap("RdYlGn_r", color_count)
    return [palette_cmap(index) for index in range(color_count)]


def create_ucm_mosaic(
    mosaic: _MosaicLayout | None = None,
    figsize: tuple[float, float] = (10.0, 10.0),
    dpi: int = 300,
    constrained_layout: bool = True,
) -> tuple[Figure, _UcmAxes]:
    """Create a named UCM subplot mosaic.

    Parameters
    ----------
    mosaic : list[list[str]] or None, optional
        Named subplot mosaic. Supported labels are ``"heatmap"``, ``"spatial"``,
        ``"temporal"``, and ``"surface_3d"``. Repeated labels create spanning
        axes according to Matplotlib ``subplot_mosaic`` rules.
    figsize : tuple[float, float], default (10.0, 10.0)
        Figure size in inches.
    dpi : int, default 300
        Figure resolution.
    constrained_layout : bool, default True
        Whether to enable Matplotlib constrained layout.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Created figure and axes keyed by mosaic label.

    """
    resolved_mosaic = DEFAULT_UCM_MOSAIC if mosaic is None else mosaic
    _validate_mosaic(resolved_mosaic)
    return cast(
        "tuple[Figure, _UcmAxes]",
        plt.subplot_mosaic(
            resolved_mosaic,
            figsize=figsize,
            dpi=dpi,
            constrained_layout=constrained_layout,
            per_subplot_kw={"surface_3d": {"projection": "3d"}},
        ),
    )


def plot_ucm_heatmap(
    ds_ucm: xr.DataArray,
    ax: Axes | None = None,
    cmap: _ColorMapLike = cmaps.bam,
    vmin: float | None = None,
    vmax: float | None = None,
    show_hist_colorbar: bool = True,
    colorbar_kwargs: dict[str, object] | None = None,
    xlabel: str | None = "Maximum Temporal Baseline (days)",
    ylabel: str | None = "Resolution (m)",
    figure_kwargs: dict[str, object] | None = None,
) -> AxesImage:
    """Plot the two-dimensional UCM heatmap.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        UCM values with ``res`` and ``day`` dimensions.
    ax : matplotlib.axes.Axes or None, optional
        Axis that receives the heatmap. If None, a new axis is created.
    figure_kwargs : dict or None, optional
        Keyword arguments passed to :func:`matplotlib.pyplot.subplots` when
        ``ax`` is None. If ``dpi`` is not provided, it defaults to 300.
    cmap : str or matplotlib.colors.Colormap, default cmaps.bam
        Matplotlib colormap.
    vmin : float or None, optional
        Optional lower value limit.
    vmax : float or None, optional
        Optional upper value limit.
    show_hist_colorbar : bool, default True
        Whether to draw a histogram colorbar.
    colorbar_kwargs : dict[str, object] or None, optional
        Additional keyword arguments passed to :class:`HistColorbar`. Use
        ``cax`` to provide a target colorbar axis. The default ``location`` is
        ``"right"``.
    xlabel : str or None, default "Maximum Temporal Baseline (days)"
        Label for the x-axis. If None, the existing label is left unchanged.
    ylabel : str or None, default "Resolution (m)"
        Label for the y-axis. If None, the existing label is left unchanged.

    Returns
    -------
    AxesImage: matplotlib.image.AxesImage
        Rendered heatmap image artist.

    """
    ds_ucm, ax = _resolve_ucm_axes(ds_ucm, ax, figure_kwargs=figure_kwargs)
    prepared_ucm = _prepare_ucm_data(ds_ucm)
    vmin, vmax = _resolve_color_limits(prepared_ucm, vmin=vmin, vmax=vmax)
    days = prepared_ucm.day.values
    resolutions = prepared_ucm.res.values

    image = ax.imshow(
        prepared_ucm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    if show_hist_colorbar:
        resolved_colorbar_kwargs = {"location": "right"}
        if colorbar_kwargs is not None:
            resolved_colorbar_kwargs.update(colorbar_kwargs)
        HistColorbar(
            prepared_ucm.values,
            image,
            **resolved_colorbar_kwargs,
        )
    ax.set_xticks(range(len(days)), days)
    ax.set_yticks(range(len(resolutions)), resolutions)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return image


def plot_ucm_spatial_profile(
    ds_ucm: xr.DataArray,
    ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str | None = DEFAULT_UCM_VALUE_LABEL,
    ylabel: str | None = "Resolution (m)",
    legend_title: str | None = "Days",
    legend_kwargs: dict[str, object] | None = None,
    figure_kwargs: dict[str, object] | None = None,
) -> list[list[Line2D]]:
    """Plot UCM variation across spatial resolution for each temporal baseline.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        UCM values with ``res`` and ``day`` dimensions.
    ax : matplotlib.axes.Axes or None, optional
        Axis that receives the spatial profile plot. If None, a new axis is
        created.
    figure_kwargs : dict or None, optional
        Keyword arguments passed to :func:`matplotlib.pyplot.subplots` when
        ``ax`` is None. If ``dpi`` is not provided, it defaults to 300.
    vmin : float or None, optional
        Optional lower x-axis limit.
    vmax : float or None, optional
        Optional upper x-axis limit.
    xlabel : str or None, default "Velocity (mm/yr)"
        Label for the x-axis. If None, the existing label is left unchanged.
    ylabel : str or None, default "Resolution (m)"
        Label for the y-axis. If None, the existing label is left unchanged.
    legend_title : str or None, default "Days"
        Legend title. If None, no title is set.
    legend_kwargs : dict[str, object] or None, optional
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.legend`.

    Returns
    -------
    list[list[Line2D]]
        List of line objects for each hue in the plot.

    """
    ds_ucm, ax = _resolve_ucm_axes(ds_ucm, ax, figure_kwargs=figure_kwargs)
    prepared_ucm = _prepare_ucm_data(ds_ucm)
    vmin, vmax = _resolve_color_limits(prepared_ucm, vmin=vmin, vmax=vmax)
    resolutions = prepared_ucm.res.values
    dataframe = prepared_ucm.to_pandas().stack().reset_index(name="velocity")

    hues = dataframe["day"].unique()
    palette = _profile_palette(len(hues))
    lines = []
    for index, hue in enumerate(hues):
        subset = dataframe[dataframe["day"] == hue]
        lines.append(
            ax.plot(
                subset["velocity"],
                subset["res"],
                label=hue,
                color=palette[index],
                marker="o",
                linewidth=2,
            )
        )
    ax.invert_yaxis()
    ax.set_yticks(resolutions, resolutions)
    ax.legend(title=legend_title, **({} if legend_kwargs is None else legend_kwargs))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim(vmin, vmax)
    ax.axvline(0, color="k", lw=2, ls="-.")
    return lines


def plot_ucm_temporal_profile(
    ds_ucm: xr.DataArray,
    ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str | None = "Maximum Temporal Baseline (day)",
    ylabel: str | None = DEFAULT_UCM_VALUE_LABEL,
    legend_title: str | None = "Resolution (m)",
    legend_kwargs: dict[str, object] | None = None,
    figure_kwargs: dict[str, object] | None = None,
) -> list[list[Line2D]]:
    """Plot UCM variation across temporal baseline for each resolution.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        UCM values with ``res`` and ``day`` dimensions.
    ax : matplotlib.axes.Axes or None, optional
        Axis that receives the temporal profile plot. If None, a new axis is
        created.
    figure_kwargs : dict or None, optional
        Keyword arguments passed to :func:`matplotlib.pyplot.subplots` when
        ``ax`` is None. If ``dpi`` is not provided, it defaults to 300.
    vmin : float or None, optional
        Optional lower y-axis limit.
    vmax : float or None, optional
        Optional upper y-axis limit.
    xlabel : str or None, default "Maximum Temporal Baseline (day)"
        Label for the x-axis. If None, the existing label is left unchanged.
    ylabel : str or None, default "Velocity (mm/yr)"
        Label for the y-axis. If None, the existing label is left unchanged.
    legend_title : str or None, default "Resolution (m)"
        Legend title. If None, no title is set.
    legend_kwargs : dict[str, object] or None, optional
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.legend`.

    Returns
    -------
    list[list[Line2D]]
        List of line objects for each hue in the plot.

    """
    ds_ucm, ax = _resolve_ucm_axes(ds_ucm, ax, figure_kwargs=figure_kwargs)
    prepared_ucm = _prepare_ucm_data(ds_ucm)
    vmin, vmax = _resolve_color_limits(prepared_ucm, vmin=vmin, vmax=vmax)
    dataframe = prepared_ucm.to_pandas().stack().reset_index(name="velocity")

    hues = dataframe["res"].unique()
    palette = _profile_palette(len(hues))
    lines = []
    for index, hue in enumerate(hues):
        subset = dataframe[dataframe["res"] == hue]
        lines.append(
            ax.plot(
                subset["day"],
                subset["velocity"],
                label=hue,
                color=palette[index],
                marker="o",
                linewidth=2,
            )
        )
    ax.axhline(0, color="k", lw=2, ls="-.")
    ax.set_xticks(dataframe["day"].unique())
    ax.legend(title=legend_title, **({} if legend_kwargs is None else legend_kwargs))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_ylim(vmin, vmax)
    return lines


def plot_ucm_surface_3d(
    ds_ucm: xr.DataArray,
    ax: Axes3D | None = None,
    cmap: _ColorMapLike = cmaps.bam,
    vmin: float | None = None,
    vmax: float | None = None,
    show_hist_colorbar: bool = True,
    xlabel: str | None = "Maximum Temporal Baseline (days)",
    ylabel: str | None = "Resolution (m)",
    zlabel: str | None = DEFAULT_UCM_VALUE_LABEL,
    colorbar_kwargs: dict[str, object] | None = None,
    figure_kwargs: dict[str, object] | None = None,
) -> Poly3DCollection:
    """Plot the three-dimensional UCM surface.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        UCM values with ``res`` and ``day`` dimensions.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None, optional
        Three-dimensional axis that receives the surface plot. If None, a new
        3D axis is created.
    figure_kwargs : dict or None, optional
        Keyword arguments passed to :func:`matplotlib.pyplot.subplots` when
        ``ax`` is None. Any ``subplot_kw`` values are merged with the required
        3D projection. If ``dpi`` is not provided, it defaults to 300.
    cmap : str or matplotlib.colors.Colormap, default cmaps.bam
        Matplotlib colormap.
    vmin : float or None, optional
        Optional lower value limit.
    vmax : float or None, optional
        Optional upper value limit.
    show_hist_colorbar : bool, default True
        Whether to add a histogram colorbar for the 3D surface.
    xlabel : str or None, default "Maximum Temporal Baseline (days)"
        Label for the x-axis. If None, the existing label is left unchanged.
    ylabel : str or None, default "Resolution (m)"
        Label for the y-axis. If None, the existing label is left unchanged.
    zlabel : str or None, default "Velocity (mm/yr)"
        Label for the z-axis. If None, the existing label is left unchanged.
    colorbar_kwargs : dict[str, object] or None, optional
        Additional keyword arguments passed to :class:`HistColorbar`. The
        default ``location`` is ``"right"``.

    Returns
    -------
    mpl_toolkits.mplot3d.art3d.Poly3DCollection
        Rendered surface artist.

    """
    ds_ucm, ax = _resolve_ucm_axes(
        ds_ucm,
        ax,
        projection="3d",
        figure_kwargs=figure_kwargs,
    )
    prepared_ucm = _prepare_ucm_data(ds_ucm)
    vmin, vmax = _resolve_color_limits(prepared_ucm, vmin=vmin, vmax=vmax)
    days = prepared_ucm.day.values
    resolutions = prepared_ucm.res.values
    day_grid, resolution_grid = np.meshgrid(days, resolutions)

    surface = ax.plot_surface(
        day_grid,
        resolution_grid,
        prepared_ucm.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        antialiased=True,
    )
    if show_hist_colorbar:
        resolved_colorbar_kwargs = {"ax": ax, "location": "right", "fraction": 0.15}
        if colorbar_kwargs is not None:
            resolved_colorbar_kwargs.update(colorbar_kwargs)
        HistColorbar(
            prepared_ucm.values,
            surface,
            **resolved_colorbar_kwargs,
        )
    ax.invert_yaxis()
    ax.set_xticks(days)
    ax.set_yticks(resolutions)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    return surface


def plot_ucm(
    ds_ucm: xr.DataArray,
    cmap: _ColorMapLike = cmaps.bam,
    vmin: float | None = None,
    vmax: float | None = None,
    mosaic: _MosaicLayout | None = None,
    figsize: tuple[float, float] = (10.0, 10.0),
    dpi: int = 300,
    constrained_layout: bool = True,
    show_hist_colorbar: bool | Mapping[_PanelName, bool] | None = None,
    quantity_label: str = DEFAULT_UCM_VALUE_LABEL,
    xlabel: _PanelText = None,
    ylabel: _PanelText = None,
    zlabel: _PanelText = None,
    legend_title: _PanelText = None,
    legend_kwargs: Mapping[_PanelName, dict[str, object]] | None = None,
    colorbar_kwargs: Mapping[_PanelName, dict[str, object]] | None = None,
) -> tuple[Figure, _UcmAxes]:
    """Plot a UCM summary figure from a named subplot mosaic.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        UCM values with ``res`` and ``day`` dimensions.
    cmap : str or matplotlib.colors.Colormap, default cmaps.bam
        Matplotlib colormap used by the heatmap and 3D surface.
    quantity_label : str, optional
        Quantity label describing the input data. Used for default value-axis
        labels. Defaults to "Velocity (mm/yr)".
    vmin : float or None, optional
        Optional lower value limit shared by all panels.
    vmax : float or None, optional
        Optional upper value limit shared by all panels.
    mosaic : list[list[str]] or None, optional
        Named subplot mosaic. Supported labels are ``"heatmap"``, ``"spatial"``,
        ``"temporal"``, and ``"surface_3d"``. Defaults to a two-by-two layout
        with the 3D surface in the lower-right panel.
    figsize : tuple[float, float], default (10.0, 10.0)
        Figure size in inches.
    dpi : int, default 300
        Figure resolution.
    constrained_layout : bool, default True
        Whether to enable Matplotlib constrained layout.
    show_hist_colorbar : bool, mapping, or None, optional
        Histogram colorbar visibility. A boolean is applied to all supported
        panels. A mapping can override individual panels, for example
        ``{"heatmap": False, "surface_3d": True}``. Defaults to False for
        ``"heatmap"`` and True for ``"surface_3d"``. Only ``"heatmap"`` and
        ``"surface_3d"`` use this option.
    xlabel, ylabel, zlabel, legend_title : str, mapping, or None
        Label overrides. A string is applied to all panels that use the field.
        A mapping can override labels by panel name. For example,
        ``xlabel={"heatmap": "Temporal baseline", "spatial": "Velocity"}``
        changes only the selected panels. None keeps defaults derived from
        ``quantity_label``. ``zlabel`` is used only by ``"surface_3d"``, and
        ``legend_title`` is used by ``"spatial"`` and ``"temporal"``.
    legend_kwargs : mapping or None, optional
        Additional legend keyword arguments keyed by panel name. Supported keys
        are ``"spatial"`` and ``"temporal"``.
    colorbar_kwargs : mapping or None, optional
        Additional :class:`HistColorbar` keyword arguments keyed by panel name.
        Supported keys are ``"heatmap"`` and ``"surface_3d"``. The default
        ``"surface_3d"`` colorbar ``location`` is ``"bottom"``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Figure and named axes.

    Notes
    -----
    Panel-specific mappings use the same names as the subplot mosaic:
    ``"heatmap"``, ``"spatial"``, ``"temporal"``, and ``"surface_3d"``. Missing
    keys keep the default value for that panel.

    Examples
    --------
    Draw the default four-panel layout.

    >>> fig, axes = plot_ucm(ds_ucm)

    Draw a custom layout where the 3D surface spans the right column.

    >>> fig, axes = plot_ucm(
    ...     ds_ucm,
    ...     mosaic=[["heatmap", "surface_3d"], ["temporal", "surface_3d"]],
    ... )

    Override selected labels by panel name.

    >>> fig, axes = plot_ucm(
    ...     ds_ucm,
    ...     quantity_label="Velocity (cm/yr)",
    ...     xlabel={"heatmap": "Temporal baseline", "surface_3d": "Days"},
    ...     legend_title={"spatial": "Temporal baseline"},
    ...     show_hist_colorbar={"heatmap": False, "surface_3d": True},
    ... )

    """
    labels = _default_ucm_labels(quantity_label)
    figure, axes = create_ucm_mosaic(
        mosaic=mosaic,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=constrained_layout,
    )
    if "heatmap" in axes:
        plot_ucm_heatmap(
            ds_ucm,
            ax=axes["heatmap"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            show_hist_colorbar=_resolve_panel_bool(
                "heatmap", False, show_hist_colorbar
            ),
            xlabel=_resolve_panel_text("heatmap", "xlabel", labels, xlabel),
            ylabel=_resolve_panel_text("heatmap", "ylabel", labels, ylabel),
            colorbar_kwargs=None
            if colorbar_kwargs is None
            else colorbar_kwargs.get("heatmap"),
        )
    if "spatial" in axes:
        plot_ucm_spatial_profile(
            ds_ucm,
            ax=axes["spatial"],
            vmin=vmin,
            vmax=vmax,
            xlabel=_resolve_panel_text("spatial", "xlabel", labels, xlabel),
            ylabel=_resolve_panel_text("spatial", "ylabel", labels, ylabel),
            legend_title=_resolve_panel_text(
                "spatial", "legend_title", labels, legend_title
            ),
            legend_kwargs={} if legend_kwargs is None else legend_kwargs.get("spatial"),
        )
    if "temporal" in axes:
        plot_ucm_temporal_profile(
            ds_ucm,
            ax=axes["temporal"],
            vmin=vmin,
            vmax=vmax,
            xlabel=_resolve_panel_text("temporal", "xlabel", labels, xlabel),
            ylabel=_resolve_panel_text("temporal", "ylabel", labels, ylabel),
            legend_title=_resolve_panel_text(
                "temporal", "legend_title", labels, legend_title
            ),
            legend_kwargs={}
            if legend_kwargs is None
            else legend_kwargs.get("temporal"),
        )
    if "surface_3d" in axes:
        surface_colorbar_kwargs: dict[str, object] = {"location": "bottom"}
        if colorbar_kwargs is not None:
            panel_colorbar_kwargs = colorbar_kwargs.get("surface_3d")
            if panel_colorbar_kwargs is not None:
                surface_colorbar_kwargs.update(panel_colorbar_kwargs)
        plot_ucm_surface_3d(
            ds_ucm,
            ax=axes["surface_3d"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            show_hist_colorbar=_resolve_panel_bool(
                "surface_3d", True, show_hist_colorbar
            ),
            xlabel=_resolve_panel_text("surface_3d", "xlabel", labels, xlabel),
            ylabel=_resolve_panel_text("surface_3d", "ylabel", labels, ylabel),
            zlabel=_resolve_panel_text("surface_3d", "zlabel", labels, zlabel),
            colorbar_kwargs=surface_colorbar_kwargs,
        )
    return figure, axes
