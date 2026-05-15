"""UCM panel plotting utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    TypeAlias,
    TypedDict,
    cast,
    overload,
)

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
_UcmArrayDims: TypeAlias = Literal["st", "ts"]
_ProfilePanelName: TypeAlias = Literal["spatial", "temporal"]
_ProfileXAxis: TypeAlias = Literal["variable", "quantity"]
_ProfileXAxisSetting: TypeAlias = (
    _ProfileXAxis | Mapping[_ProfilePanelName, _ProfileXAxis]
)
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
    figure_kwargs: dict[str, Any] | None = None,
) -> tuple[xr.DataArray, Axes]: ...


@overload
def _resolve_ucm_axes(
    ds_ucm: xr.DataArray,
    ax: Axes3D | None = None,
    *,
    projection: Literal["3d"],
    figure_kwargs: dict[str, Any] | None = None,
) -> tuple[xr.DataArray, Axes3D]: ...


def _resolve_ucm_axes(
    ds_ucm: xr.DataArray,
    ax: Axes | Axes3D | None = None,
    *,
    projection: Literal["rectilinear", "3d"] = "rectilinear",
    figure_kwargs: dict[str, Any] | None = None,
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
            cast("Mapping[str, Any]", subplot_kwargs.pop("subplot_kw", {}))
        )
        subplot_kw["projection"] = "3d"
        _, created_ax = plt.subplots(subplot_kw=subplot_kw, **subplot_kwargs)
        return ds_ucm, cast("Axes3D", created_ax)

    logger.error("Unsupported UCM axes projection: %s.", projection)
    msg = f"projection must be 'rectilinear' or '3d', got {projection!r}."
    raise ValueError(msg)


def _default_ucm_labels(
    quantity_label: str,
    spatial_label: str = "Resolution (m)",
    temporal_label: str = "Maximum Temporal Baseline (days)",
    spatial_legend_label: str = "Days",
    spatial_profile_x_axis: _ProfileXAxis = "variable",
    temporal_profile_x_axis: _ProfileXAxis = "variable",
) -> dict[_PanelName, dict[_LabelName, str | None]]:
    """Create default panel labels for a UCM summary plot.

    Parameters
    ----------
    quantity_label : str
        Value label used for velocity axes.
    spatial_label : str, default "Resolution (m)"
        Label used for spatial resolution axes and legends.
    temporal_label : str, default "Maximum Temporal Baseline (days)"
        Label used for temporal baseline axes and legends.
    spatial_legend_label : str, default "Days"
        Legend title used by the spatial profile panel.
    spatial_profile_x_axis : {"variable", "quantity"}, default "variable"
        X-axis mode for the spatial profile panel.
    temporal_profile_x_axis : {"variable", "quantity"}, default "variable"
        X-axis mode for the temporal profile panel.

    Returns
    -------
    dict[str, dict[str, str or None]]
        Default labels keyed by panel name and label field.

    """
    spatial_profile_labels = (
        {"xlabel": spatial_label, "ylabel": quantity_label}
        if spatial_profile_x_axis == "variable"
        else {"xlabel": quantity_label, "ylabel": spatial_label}
    )
    temporal_profile_labels = (
        {"xlabel": temporal_label, "ylabel": quantity_label}
        if temporal_profile_x_axis == "variable"
        else {"xlabel": quantity_label, "ylabel": temporal_label}
    )
    return {
        "heatmap": {
            "xlabel": temporal_label,
            "ylabel": spatial_label,
            "zlabel": None,
            "legend_title": None,
        },
        "spatial": {
            "xlabel": spatial_profile_labels["xlabel"],
            "ylabel": spatial_profile_labels["ylabel"],
            "zlabel": None,
            "legend_title": spatial_legend_label,
        },
        "temporal": {
            "xlabel": temporal_profile_labels["xlabel"],
            "ylabel": temporal_profile_labels["ylabel"],
            "zlabel": None,
            "legend_title": spatial_label,
        },
        "surface_3d": {
            "xlabel": temporal_label,
            "ylabel": spatial_label,
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


def _validate_profile_x_axis(x_axis: str) -> _ProfileXAxis:
    """Validate a UCM profile x-axis mode.

    Parameters
    ----------
    x_axis : str
        Requested x-axis mode.

    Returns
    -------
    {"variable", "quantity"}
        Validated x-axis mode.

    Raises
    ------
    ValueError
        If ``x_axis`` is not supported.

    """
    if x_axis in {"variable", "quantity"}:
        return cast("_ProfileXAxis", x_axis)

    logger.error("Invalid UCM profile x_axis value: %s.", x_axis)
    msg = "x_axis must be 'variable' or 'quantity'."
    raise ValueError(msg)


def _resolve_profile_x_axis(
    panel_name: _ProfilePanelName,
    profile_x_axis: _ProfileXAxisSetting,
) -> _ProfileXAxis:
    """Resolve a profile x-axis mode for a named panel.

    Parameters
    ----------
    panel_name : {"spatial", "temporal"}
        Profile panel whose x-axis mode is resolved.
    profile_x_axis : str or mapping
        X-axis mode applied to both profile panels, or a mapping keyed by panel
        name.

    Returns
    -------
    {"variable", "quantity"}
        Resolved and validated x-axis mode.

    """
    if isinstance(profile_x_axis, Mapping):
        return _validate_profile_x_axis(profile_x_axis.get(panel_name, "variable"))
    return _validate_profile_x_axis(profile_x_axis)


def _prepare_ucm_data(
    ds_ucm: xr.DataArray,
    spatial_dim: str = "res",
    temporal_dim: str = "day",
) -> xr.DataArray:
    """Validate and standardize UCM data for plotting.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        Two-dimensional UCM values.
    spatial_dim : str, default "res"
        Name of the spatial resolution dimension in ``ds_ucm``.
    temporal_dim : str, default "day"
        Name of the temporal baseline dimension in ``ds_ucm``.

    Returns
    -------
    xarray.DataArray
        Data renamed and transposed to ``("res", "day")``.

    Raises
    ------
    ValueError
        If dimensions are invalid, missing, or not two-dimensional.

    """
    if spatial_dim == temporal_dim:
        logger.error(
            "UCM spatial and temporal dimensions must differ: %s.",
            spatial_dim,
        )
        msg = "spatial_dim and temporal_dim must be different."
        raise ValueError(msg)

    required_dimensions = {spatial_dim, temporal_dim}
    missing_dimensions = required_dimensions.difference(ds_ucm.dims)
    if missing_dimensions:
        logger.error(
            "UCM data is missing required dimensions: %s.",
            sorted(missing_dimensions),
        )
        msg = (
            "ds_ucm must contain the spatial and temporal dimensions "
            f"{spatial_dim!r} and {temporal_dim!r}."
        )
        raise ValueError(msg)

    if len(ds_ucm.dims) != 2:
        logger.error(
            "UCM data must be two-dimensional, got dimensions: %s.",
            ds_ucm.dims,
        )
        msg = "ds_ucm must be two-dimensional."
        raise ValueError(msg)

    dimension_mapping = {
        old_dimension: new_dimension
        for old_dimension, new_dimension in (
            (spatial_dim, "res"),
            (temporal_dim, "day"),
        )
        if old_dimension != new_dimension
    }
    return ds_ucm.rename(dimension_mapping).transpose("res", "day")


def _resolve_array_coords(
    coords: Any | None,
    length: int,
    coord_name: str,
) -> np.ndarray:
    """Resolve user-provided or default coordinates for UCM array input.

    Parameters
    ----------
    coords : object or None
        Coordinate values provided by the caller. If None, integer index
        coordinates are generated.
    length : int
        Expected coordinate length.
    coord_name : str
        Coordinate name used in validation messages.

    Returns
    -------
    numpy.ndarray
        Coordinate values with the expected length.

    Raises
    ------
    ValueError
        If provided coordinate values do not match ``length``.

    """
    if coords is None:
        return np.arange(length)

    resolved_coords = np.asarray(coords)
    if resolved_coords.ndim == 0 or len(resolved_coords) != length:
        logger.error(
            "Invalid %s coordinate length for UCM array input: expected %s, got %s.",
            coord_name,
            length,
            resolved_coords.shape,
        )
        msg = f"{coord_name}_coords must have length {length}."
        raise ValueError(msg)
    return resolved_coords


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


def _profile_palette(
    color_count: int,
    cmap: _ColorMapLike = "RdYlGn_r",
) -> list[tuple[float, float, float, float]]:
    """Create profile line colors from a UCM profile colormap.

    Parameters
    ----------
    color_count : int
        Number of colors to create.
    cmap : str or matplotlib.colors.Colormap, default "RdYlGn_r"
        Matplotlib colormap sampled for profile line colors.

    Returns
    -------
    list[tuple[float, float, float, float]]
        RGBA colors sampled from the selected colormap.

    """
    if color_count <= 0:
        return []
    palette_cmap = (
        cmap.resampled(color_count)
        if isinstance(cmap, Colormap)
        else plt.get_cmap(cmap, color_count)
    )
    return [palette_cmap(index) for index in range(color_count)]


def create_ucm_mosaic(
    mosaic: _MosaicLayout | None = None,
    figsize: tuple[float, float] = (10.0, 10.0),
    dpi: int = 300,
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
            per_subplot_kw={"surface_3d": {"projection": "3d"}},
        ),
    )


class UCM:
    """Stateful wrapper for UCM plotting utilities.

    Parameters
    ----------
    ds_ucm : xarray.DataArray
        Two-dimensional UCM values.
    quantity_label : str, default "Velocity (mm/yr)"
        Label for the plotted UCM quantity.
    spatial_label : str, default "Resolution (m)"
        Label for spatial resolution axes and legends.
    temporal_label : str, default "Maximum Temporal Baseline (days)"
        Label for temporal baseline axes and legends.
    spatial_dim : str, default "res"
        Name of the spatial resolution dimension in ``ds_ucm``.
    temporal_dim : str, default "day"
        Name of the temporal baseline dimension in ``ds_ucm``.
    cmap : str or matplotlib.colors.Colormap, default cmaps.bam
        Matplotlib colormap used by heatmap and 3D surface plots.
    vmin : float or None, optional
        Default lower value limit.
    vmax : float or None, optional
        Default upper value limit.

    Attributes
    ----------
    ds_ucm : xarray.DataArray
        Prepared UCM values transposed to ``("res", "day")``.
    quantity_label : str
        Label for value axes.
    spatial_label : str
        Label for spatial resolution axes and legends.
    temporal_label : str
        Label for temporal baseline axes and legends.
    cmap : str or matplotlib.colors.Colormap
        Default colormap.
    vmin, vmax : float or None
        Default value limits.

    Examples
    --------
    >>> ucm = UCM(ds_ucm, quantity_label="Velocity (cm/yr)")
    >>> image = ucm.plot_heatmap(show_hist_colorbar=False)
    >>> fig, axes = ucm.plot()

    """

    def __init__(
        self,
        ds_ucm: xr.DataArray,
        *,
        quantity_label: str = "Velocity (mm/yr)",
        spatial_label: str = "Resolution (m)",
        temporal_label: str = "Maximum Temporal Baseline (days)",
        spatial_dim: str = "res",
        temporal_dim: str = "day",
        cmap: _ColorMapLike = cmaps.bam,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Initialize a UCM plot wrapper.

        Parameters
        ----------
        ds_ucm : xarray.DataArray
            Two-dimensional UCM values.
        quantity_label : str, default "Velocity (mm/yr)"
            Label for the plotted UCM quantity.
        spatial_label : str, default "Resolution (m)"
            Label for spatial resolution axes and legends.
        temporal_label : str, default "Maximum Temporal Baseline (days)"
            Label for temporal baseline axes and legends.
        spatial_dim : str, default "res"
            Name of the spatial resolution dimension in ``ds_ucm``.
        temporal_dim : str, default "day"
            Name of the temporal baseline dimension in ``ds_ucm``.
        cmap : str or matplotlib.colors.Colormap, default cmaps.bam
            Matplotlib colormap used by heatmap and 3D surface plots.
        vmin : float or None, optional
            Default lower value limit.
        vmax : float or None, optional
            Default upper value limit.

        """
        self.ds_ucm = _prepare_ucm_data(
            ds_ucm,
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
        )
        self.quantity_label = quantity_label
        self.spatial_label = spatial_label
        self.temporal_label = temporal_label
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        spatial_coords: Any | None = None,
        temporal_coords: Any | None = None,
        input_dims: _UcmArrayDims = "st",
        name: str | None = None,
        attrs: Mapping[str, Any] | None = None,
        quantity_label: str = "Velocity (mm/yr)",
        spatial_label: str = "Resolution (m)",
        temporal_label: str = "Maximum Temporal Baseline (days)",
        cmap: _ColorMapLike = cmaps.bam,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Self:
        """Create a UCM plot wrapper from a NumPy array.

        Parameters
        ----------
        array : numpy.ndarray
            Two-dimensional UCM values.
        spatial_coords : array-like or None, optional
            Spatial resolution coordinate values. If None, integer coordinates
            from 0 to ``n - 1`` are generated.
        temporal_coords : array-like or None, optional
            Temporal baseline coordinate values. If None, integer coordinates
            from 0 to ``n - 1`` are generated.
        input_dims : {"st", "ts"}, default "st"
            Axis order of ``array``. ``"st"`` means spatial then temporal, so
            ``array.shape == (len(spatial_coords), len(temporal_coords))``.
            ``"ts"`` means temporal then spatial, so
            ``array.shape == (len(temporal_coords), len(spatial_coords))``.
            The created instance always stores data internally as
            ``("res", "day")``.
        name : str or None, optional
            Name assigned to the intermediate :class:`xarray.DataArray`.
        attrs : mapping or None, optional
            Attributes assigned to the intermediate :class:`xarray.DataArray`.
        quantity_label : str, default "Velocity (mm/yr)"
            Label for the plotted UCM quantity.
        spatial_label : str, default "Resolution (m)"
            Label for spatial resolution axes and legends.
        temporal_label : str, default "Maximum Temporal Baseline (days)"
            Label for temporal baseline axes and legends.
        cmap : str or matplotlib.colors.Colormap, default cmaps.bam
            Matplotlib colormap used by heatmap and 3D surface plots.
        vmin : float or None, optional
            Default lower value limit.
        vmax : float or None, optional
            Default upper value limit.

        Returns
        -------
        UCM
            UCM plot wrapper initialized from ``array``.

        Raises
        ------
        ValueError
            If ``array`` is not two-dimensional, ``input_dims`` is invalid, or
            coordinate lengths do not match the declared array axes.

        Examples
        --------
        >>> ucm = UCM.from_array(
        ...     values,
        ...     spatial_coords=[30, 60],
        ...     temporal_coords=[12, 24, 36],
        ...     input_dims="st",
        ... )

        """
        if array.ndim != 2:
            logger.error(
                "UCM array input must be two-dimensional, got shape: %s.",
                array.shape,
            )
            msg = "array must be two-dimensional."
            raise ValueError(msg)

        if input_dims not in {"st", "ts"}:
            logger.error("Invalid UCM array input_dims value: %s.", input_dims)
            msg = "input_dims must be 'st' or 'ts'."
            raise ValueError(msg)

        if input_dims == "st":
            dims = ("spatial", "temporal")
            spatial_length, temporal_length = array.shape
        else:
            dims = ("temporal", "spatial")
            temporal_length, spatial_length = array.shape

        resolved_spatial_coords = _resolve_array_coords(
            spatial_coords,
            spatial_length,
            "spatial",
        )
        resolved_temporal_coords = _resolve_array_coords(
            temporal_coords,
            temporal_length,
            "temporal",
        )
        coords = {
            "spatial": resolved_spatial_coords,
            "temporal": resolved_temporal_coords,
        }

        import xarray as xr

        data_array = xr.DataArray(
            array,
            dims=dims,
            coords=coords,
            name=name,
            attrs={} if attrs is None else dict(attrs),
        )
        return cls(
            data_array,
            quantity_label=quantity_label,
            spatial_label=spatial_label,
            temporal_label=temporal_label,
            spatial_dim="spatial",
            temporal_dim="temporal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    def plot_heatmap(
        self,
        ax: Axes | None = None,
        cmap: _ColorMapLike | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        show_hist_colorbar: bool = True,
        colorbar_kwargs: dict[str, Any] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        figure_kwargs: dict[str, Any] | None = None,
    ) -> AxesImage:
        """Plot the two-dimensional UCM heatmap.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axis that receives the heatmap. If None, a new axis is created.
        cmap : str, matplotlib.colors.Colormap, or None, optional
            Colormap override. If None, the instance colormap is used.
        vmin : float or None, optional
            Lower value limit override. If None, the instance value is used.
        vmax : float or None, optional
            Upper value limit override. If None, the instance value is used.
        show_hist_colorbar : bool, default True
            Whether to draw a histogram colorbar.
        colorbar_kwargs : dict[str, Any] or None, optional
            Additional keyword arguments passed to :class:`HistColorbar`.
        xlabel : str or None, optional
            X-axis label override. If None, ``temporal_label`` is used.
        ylabel : str or None, optional
            Y-axis label override. If None, ``spatial_label`` is used.
        figure_kwargs : dict[str, Any] or None, optional
            Keyword arguments passed to :func:`matplotlib.pyplot.subplots` when
            ``ax`` is None.

        Returns
        -------
        matplotlib.image.AxesImage
            Rendered heatmap image artist.

        """
        prepared_ucm, ax = _resolve_ucm_axes(
            self.ds_ucm,
            ax,
            figure_kwargs=figure_kwargs,
        )
        vmin, vmax = _resolve_color_limits(
            prepared_ucm,
            vmin=self.vmin if vmin is None else vmin,
            vmax=self.vmax if vmax is None else vmax,
        )
        days = prepared_ucm.day.values
        resolutions = prepared_ucm.res.values

        image = ax.imshow(
            prepared_ucm,
            cmap=self.cmap if cmap is None else cmap,
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
        resolved_xlabel = self.temporal_label if xlabel is None else xlabel
        resolved_ylabel = self.spatial_label if ylabel is None else ylabel
        if resolved_xlabel is not None:
            ax.set_xlabel(resolved_xlabel)
        if resolved_ylabel is not None:
            ax.set_ylabel(resolved_ylabel)
        return image

    def plot_sprofile(
        self,
        ax: Axes | None = None,
        cmap: _ColorMapLike = "RdYlGn_r",
        x_axis: _ProfileXAxis = "variable",
        vmin: float | None = None,
        vmax: float | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        legend_title: str | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        figure_kwargs: dict[str, Any] | None = None,
    ) -> list[list[Line2D]]:
        """Plot UCM spatial profiles.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axis that receives the spatial profile plot.
        cmap : str or matplotlib.colors.Colormap, default "RdYlGn_r"
            Colormap sampled for profile line colors.
        x_axis : {"variable", "quantity"}, default "variable"
            Whether the x-axis shows spatial resolution values or UCM quantity
            values.
        vmin : float or None, optional
            Lower quantity-axis limit override. If None, the instance value is
            used.
        vmax : float or None, optional
            Upper quantity-axis limit override. If None, the instance value is
            used.
        xlabel : str or None, optional
            X-axis label override. If None, a label is selected from
            ``x_axis``.
        ylabel : str or None, optional
            Y-axis label override. If None, a label is selected from
            ``x_axis``.
        legend_title : str or None, optional
            Legend title override. If None, ``temporal_label`` is used.
        legend_kwargs : dict[str, Any] or None, optional
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.legend`.
        figure_kwargs : dict[str, Any] or None, optional
            Keyword arguments passed to :func:`matplotlib.pyplot.subplots` when
            ``ax`` is None.

        Returns
        -------
        list[list[matplotlib.lines.Line2D]]
            List of line objects for each hue in the plot.

        """
        resolved_x_axis = _validate_profile_x_axis(x_axis)
        prepared_ucm, ax = _resolve_ucm_axes(
            self.ds_ucm,
            ax,
            figure_kwargs=figure_kwargs,
        )
        vmin, vmax = _resolve_color_limits(
            prepared_ucm,
            vmin=self.vmin if vmin is None else vmin,
            vmax=self.vmax if vmax is None else vmax,
        )
        resolutions = prepared_ucm.res.values
        dataframe = prepared_ucm.to_pandas().stack().reset_index(name="velocity")

        hues = dataframe["day"].unique()
        palette = _profile_palette(len(hues), cmap=cmap)
        lines = []
        for index, hue in enumerate(hues):
            subset = dataframe[dataframe["day"] == hue]
            if resolved_x_axis == "variable":
                lines.append(
                    ax.plot(
                        subset["res"],
                        subset["velocity"],
                        label=hue,
                        color=palette[index],
                        marker="o",
                        linewidth=2,
                    )
                )
            else:
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
        ax.legend(
            title=self.temporal_label if legend_title is None else legend_title,
            **({} if legend_kwargs is None else legend_kwargs),
        )
        default_xlabel = (
            self.spatial_label if resolved_x_axis == "variable" else self.quantity_label
        )
        default_ylabel = (
            self.quantity_label if resolved_x_axis == "variable" else self.spatial_label
        )
        resolved_xlabel = default_xlabel if xlabel is None else xlabel
        resolved_ylabel = default_ylabel if ylabel is None else ylabel
        if resolved_xlabel is not None:
            ax.set_xlabel(resolved_xlabel)
        if resolved_ylabel is not None:
            ax.set_ylabel(resolved_ylabel)
        if resolved_x_axis == "variable":
            ax.set_xticks(resolutions, resolutions)
            ax.set_ylim(vmin, vmax)
            ax.axhline(0, color="k", lw=2, ls="-.")
        else:
            ax.invert_yaxis()
            ax.set_yticks(resolutions, resolutions)
            ax.set_xlim(vmin, vmax)
            ax.axvline(0, color="k", lw=2, ls="-.")
        return lines

    def plot_tprofile(
        self,
        ax: Axes | None = None,
        cmap: _ColorMapLike = "RdYlGn_r",
        x_axis: _ProfileXAxis = "variable",
        vmin: float | None = None,
        vmax: float | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        legend_title: str | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        figure_kwargs: dict[str, Any] | None = None,
    ) -> list[list[Line2D]]:
        """Plot UCM temporal profiles.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axis that receives the temporal profile plot.
        cmap : str or matplotlib.colors.Colormap, default "RdYlGn_r"
            Colormap sampled for profile line colors.
        x_axis : {"variable", "quantity"}, default "variable"
            Whether the x-axis shows temporal baseline values or UCM quantity
            values.
        vmin : float or None, optional
            Lower quantity-axis limit override. If None, the instance value is
            used.
        vmax : float or None, optional
            Upper quantity-axis limit override. If None, the instance value is
            used.
        xlabel : str or None, optional
            X-axis label override. If None, a label is selected from
            ``x_axis``.
        ylabel : str or None, optional
            Y-axis label override. If None, a label is selected from
            ``x_axis``.
        legend_title : str or None, optional
            Legend title override. If None, ``spatial_label`` is used.
        legend_kwargs : dict[str, Any] or None, optional
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.legend`.
        figure_kwargs : dict[str, Any] or None, optional
            Keyword arguments passed to :func:`matplotlib.pyplot.subplots` when
            ``ax`` is None.

        Returns
        -------
        list[list[matplotlib.lines.Line2D]]
            List of line objects for each hue in the plot.

        """
        resolved_x_axis = _validate_profile_x_axis(x_axis)
        prepared_ucm, ax = _resolve_ucm_axes(
            self.ds_ucm,
            ax,
            figure_kwargs=figure_kwargs,
        )
        vmin, vmax = _resolve_color_limits(
            prepared_ucm,
            vmin=self.vmin if vmin is None else vmin,
            vmax=self.vmax if vmax is None else vmax,
        )
        dataframe = prepared_ucm.to_pandas().stack().reset_index(name="velocity")
        days = prepared_ucm.day.values

        hues = dataframe["res"].unique()
        palette = _profile_palette(len(hues), cmap=cmap)
        lines = []
        for index, hue in enumerate(hues):
            subset = dataframe[dataframe["res"] == hue]
            if resolved_x_axis == "variable":
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
            else:
                lines.append(
                    ax.plot(
                        subset["velocity"],
                        subset["day"],
                        label=hue,
                        color=palette[index],
                        marker="o",
                        linewidth=2,
                    )
                )
        ax.legend(
            title=self.spatial_label if legend_title is None else legend_title,
            **({} if legend_kwargs is None else legend_kwargs),
        )
        default_xlabel = (
            self.temporal_label
            if resolved_x_axis == "variable"
            else self.quantity_label
        )
        default_ylabel = (
            self.quantity_label
            if resolved_x_axis == "variable"
            else self.temporal_label
        )
        resolved_xlabel = default_xlabel if xlabel is None else xlabel
        resolved_ylabel = default_ylabel if ylabel is None else ylabel
        if resolved_xlabel is not None:
            ax.set_xlabel(resolved_xlabel)
        if resolved_ylabel is not None:
            ax.set_ylabel(resolved_ylabel)
        if resolved_x_axis == "variable":
            ax.axhline(0, color="k", lw=2, ls="-.")
            ax.set_xticks(days)
            ax.set_ylim(vmin, vmax)
        else:
            ax.axvline(0, color="k", lw=2, ls="-.")
            ax.set_yticks(days)
            ax.set_xlim(vmin, vmax)
        return lines

    def plot_3d_surface(
        self,
        ax: Axes3D | None = None,
        cmap: _ColorMapLike | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        show_hist_colorbar: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        colorbar_kwargs: dict[str, Any] | None = None,
        figure_kwargs: dict[str, Any] | None = None,
    ) -> Poly3DCollection:
        """Plot the three-dimensional UCM surface.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None, optional
            Three-dimensional axis that receives the surface plot.
        cmap : str, matplotlib.colors.Colormap, or None, optional
            Colormap override. If None, the instance colormap is used.
        vmin : float or None, optional
            Lower value limit override. If None, the instance value is used.
        vmax : float or None, optional
            Upper value limit override. If None, the instance value is used.
        show_hist_colorbar : bool, default True
            Whether to draw a histogram colorbar.
        xlabel : str or None, optional
            X-axis label override. If None, ``temporal_label`` is used.
        ylabel : str or None, optional
            Y-axis label override. If None, ``spatial_label`` is used.
        zlabel : str or None, optional
            Z-axis label override. If None, ``quantity_label`` is used.
        colorbar_kwargs : dict[str, Any] or None, optional
            Additional keyword arguments passed to :class:`HistColorbar`.
        figure_kwargs : dict[str, Any] or None, optional
            Keyword arguments passed to :func:`matplotlib.pyplot.subplots` when
            ``ax`` is None.

        Returns
        -------
        mpl_toolkits.mplot3d.art3d.Poly3DCollection
            Rendered surface artist.

        """
        prepared_ucm, ax = _resolve_ucm_axes(
            self.ds_ucm,
            ax=ax,
            projection="3d",
            figure_kwargs=figure_kwargs,
        )
        vmin, vmax = _resolve_color_limits(
            prepared_ucm,
            vmin=self.vmin if vmin is None else vmin,
            vmax=self.vmax if vmax is None else vmax,
        )
        days = prepared_ucm.day.values
        resolutions = prepared_ucm.res.values
        day_grid, resolution_grid = np.meshgrid(days, resolutions)

        surface = ax.plot_surface(
            day_grid,
            resolution_grid,
            prepared_ucm.values,
            cmap=self.cmap if cmap is None else cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            antialiased=True,
        )
        if show_hist_colorbar:
            resolved_colorbar_kwargs = {
                "ax": ax,
                "location": "right",
                "fraction": 0.15,
                "label": self.quantity_label,
            }
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
        resolved_xlabel = self.temporal_label if xlabel is None else xlabel
        resolved_ylabel = self.spatial_label if ylabel is None else ylabel
        resolved_zlabel = self.quantity_label if zlabel is None else zlabel
        if resolved_xlabel is not None:
            ax.set_xlabel(resolved_xlabel)
        if resolved_ylabel is not None:
            ax.set_ylabel(resolved_ylabel)
        if resolved_zlabel is not None:
            ax.set_zlabel(resolved_zlabel)
        return surface

    def plot(
        self,
        cmap: _ColorMapLike | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        profile_cmap: _ColorMapLike = "RdYlGn_r",
        profile_x_axis: _ProfileXAxisSetting = "variable",
        mosaic: _MosaicLayout | None = None,
        figsize: tuple[float, float] = (10.0, 10.0),
        dpi: int = 300,
        show_hist_colorbar: bool | Mapping[_PanelName, bool] | None = None,
        xlabel: _PanelText = None,
        ylabel: _PanelText = None,
        zlabel: _PanelText = None,
        legend_title: _PanelText = None,
        legend_kwargs: Mapping[_PanelName, dict[str, Any]] | None = None,
        colorbar_kwargs: Mapping[_PanelName, dict[str, Any]] | None = None,
    ) -> tuple[Figure, _UcmAxes]:
        """Plot a UCM summary figure.

        Parameters
        ----------
        cmap : str, matplotlib.colors.Colormap, or None, optional
            Colormap override. If None, the instance colormap is used.
        vmin : float or None, optional
            Lower value limit override. If None, the instance value is used.
        vmax : float or None, optional
            Upper value limit override. If None, the instance value is used.
        profile_cmap : str or matplotlib.colors.Colormap, default
            "RdYlGn_r"
            Colormap sampled for spatial and temporal profile line colors.
        profile_x_axis : {"variable", "quantity"} or mapping, default "variable"
            X-axis mode for profile panels. A string applies to both profile
            panels. A mapping can configure ``"spatial"`` and ``"temporal"``
            separately.
        mosaic : list[list[str]] or None, optional
            Named subplot mosaic.
        figsize : tuple[float, float], default (10.0, 10.0)
            Figure size in inches.
        dpi : int, default 300
            Figure resolution.
        show_hist_colorbar : bool, mapping, or None, optional
            Histogram colorbar visibility.
        xlabel, ylabel, zlabel, legend_title : str, mapping, or None
            Label overrides. Mappings are merged with instance label defaults.
        legend_kwargs : mapping or None, optional
            Additional legend keyword arguments keyed by panel name.
        colorbar_kwargs : mapping or None, optional
            Additional :class:`HistColorbar` keyword arguments keyed by panel
            name.

        Returns
        -------
        tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
            Figure and named axes.

        """
        spatial_profile_x_axis = _resolve_profile_x_axis("spatial", profile_x_axis)
        temporal_profile_x_axis = _resolve_profile_x_axis("temporal", profile_x_axis)
        labels = _default_ucm_labels(
            self.quantity_label,
            spatial_label=self.spatial_label,
            temporal_label=self.temporal_label,
            spatial_legend_label=self.temporal_label,
            spatial_profile_x_axis=spatial_profile_x_axis,
            temporal_profile_x_axis=temporal_profile_x_axis,
        )
        figure, axes = create_ucm_mosaic(
            mosaic=mosaic,
            figsize=figsize,
            dpi=dpi,
        )
        if "heatmap" in axes:
            self.plot_heatmap(
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
            self.plot_sprofile(
                ax=axes["spatial"],
                cmap=profile_cmap,
                x_axis=spatial_profile_x_axis,
                vmin=vmin,
                vmax=vmax,
                xlabel=_resolve_panel_text("spatial", "xlabel", labels, xlabel),
                ylabel=_resolve_panel_text("spatial", "ylabel", labels, ylabel),
                legend_title=_resolve_panel_text(
                    "spatial", "legend_title", labels, legend_title
                ),
                legend_kwargs={}
                if legend_kwargs is None
                else legend_kwargs.get("spatial"),
            )
        if "temporal" in axes:
            self.plot_tprofile(
                ax=axes["temporal"],
                cmap=profile_cmap,
                x_axis=temporal_profile_x_axis,
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
            surface_colorbar_kwargs: dict[str, Any] = {
                "location": "bottom",
                "label": self.quantity_label,
            }
            if colorbar_kwargs is not None:
                panel_colorbar_kwargs = colorbar_kwargs.get("surface_3d")
                if panel_colorbar_kwargs is not None:
                    surface_colorbar_kwargs.update(panel_colorbar_kwargs)
            self.plot_3d_surface(
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
