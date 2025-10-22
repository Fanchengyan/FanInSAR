"""Histogram-embedded Colorbar for Matplotlib.

This module provides a customizable colorbar with an integrated histogram
showing the distribution of data values across the color gradient. This is
inspired by the colorbar in the EOMaps plotting package:
https://github.com/raphaelquast/EOmaps
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable, cast

import matplotlib as mpl
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Rectangle
from matplotlib.ticker import (
    AutoLocator,
    Formatter,
    Locator,
    LogFormatterSciNotation,
    LogLocator,
    NullFormatter,
    NullLocator,
    ScalarFormatter,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing_extensions import Literal

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure, SubFigure
    from numpy.typing import ArrayLike, NDArray

NON_COLORBAR_KEYS = [  # remove kws that cannot be passed to Colorbar
    "fraction",
    "pad",
    "shrink",
    "aspect",
    "anchor",
    "panchor",
]


class _Histogram(Colorbar):
    """Draw a histogram on axes with same positioning as colorbar.

    This class inherits from Colorbar to leverage its extend triangle behavior,
    ensuring perfect alignment with the main colorbar. Instead of drawing a
    colorbar gradient, it draws a histogram.

    Parameters
    ----------
    ax : Axes
        The axes to draw the histogram in.
    data : ArrayLike
        The data values to create the histogram from.
    mappable : ScalarMappable
        The mappable object for norm and cmap.
    orientation : str
        The orientation of the histogram ('horizontal' or 'vertical').
    hist_bins : int | NDArray
        Number of histogram bins or bin edges.
    hist_kwargs : dict
        Additional kwargs for matplotlib's hist function.
    min_count : float
        Minimum count value for the histogram axis.
    scale : str
        Scale of the histogram axis ('linear' or 'log').
    divider_style : dict
        Style for the divider line.
    location : str
        Location of the histogram ('left', 'right', 'top', 'bottom').
    **kwargs
        Additional kwargs passed to Colorbar.__init__.

    """

    def __init__(
        self,
        ax: Axes,
        data: ArrayLike,
        mappable: ScalarMappable,
        orientation: Literal["vertical", "horizontal"],
        hist_bins: int | NDArray,
        hist_kwargs: dict,
        min_count: float,
        scale: Literal["linear", "log"],
        divider_style: dict[str, Any],
        location: Literal["left", "right", "top", "bottom"],
        **kwargs: Any,
    ) -> None:
        # Store histogram-specific parameters before calling super().__init__
        self._data = np.asanyarray(data).flatten()
        self._hist_bins = hist_bins
        self._hist_kwargs = hist_kwargs
        self._min_count = min_count
        self._scale = scale
        self._divider_style = divider_style
        self._location = location
        self._hist_orientation = (
            "horizontal" if orientation == "vertical" else "vertical"
        )

        # Get levels and facecolors from parent HistColorbar for color matching
        self._levels = kwargs.pop("_levels", None)
        self._level_facecolors = kwargs.pop("_level_facecolors", None)
        self._effective_alpha = kwargs.pop("_effective_alpha", None)

        # Initialize Colorbar - this sets up extend triangles and axes locator
        super().__init__(ax, mappable=mappable, orientation=orientation, **kwargs)

    def _draw_all(self) -> None:
        """Override Colorbar._draw_all to preserve histogram count axis.

        This is a modified copy of matplotlib's Colorbar._draw_all() method.
        The main change is that we DON'T call set_xlim(0,1) or set_ylim(0,1)
        for the histogram count axis, preserving the actual histogram range.
        """
        # Set self._boundaries and self._values, including extensions
        self._process_values()

        # Set self.vmin and self.vmax to first and last boundary, excluding extensions
        self.vmin, self.vmax = self._boundaries[self._inside][[0, -1]]

        # Compute the X/Y mesh
        self._mesh()

        # Draw the extend triangles, and shrink the inner Axes to accommodate
        self._do_extends()
        self.hide_triangles()

        lower, upper = self.vmin, self.vmax
        if self.long_axis.get_inverted():
            # If the axis is inverted, we need to swap the vmin/vmax
            lower, upper = upper, lower

        # CRITICAL CHANGE: Only set limits for the DATA axis (long_axis),
        # NOT for the count axis (short_axis for histogram)
        if self.orientation == "vertical":
            self.ax.set_ylim(lower, upper)
        else:
            self.ax.set_xlim(lower, upper)

        # Set up the tick locators and formatters
        # self.update_ticks()

        # Draw the histogram (instead of calling _add_solids with mesh)
        self._add_solids()

        # Apply histogram-specific styling
        self._apply_histogram_styling()

    def hide_triangles(self) -> None:
        """Hide the extend triangles."""
        for patch in self._extend_patches:
            patch.set_fill(False)
            patch.set_edgecolor("none")

    def _add_solids(self) -> None:
        """Override to draw histogram instead of colorbar gradient.

        This method draws the histogram bars instead of the colorbar's
        pcolormesh gradient. Since we control _draw_all(), we don't need
        the X, Y, C parameters that the parent Colorbar passes.
        """
        # Clean up any previous histogram patches
        if self.solids is not None:
            self.solids.remove()
            self.solids = None
        for solid in self.solids_patches:
            solid.remove()
        self.solids_patches = []

        # Draw the histogram
        data_finite = self._data[np.isfinite(self._data)]

        if len(data_finite) == 0:
            msg = (
                "No finite values in data. Unable to draw histogram. "
                "Check your data for NaNs and infinities."
            )
            warnings.warn(msg, stacklevel=2)
            logger.warning(msg)
            return

        # Determine histogram range from norm boundaries
        boundaries = getattr(self.norm, "boundaries", None)
        levels = self._levels
        if levels is not None and len(levels) >= 2:
            vmin, vmax = float(np.min(levels)), float(np.max(levels))
        elif boundaries is not None and len(boundaries) >= 2:
            vmin, vmax = float(boundaries[0]), float(boundaries[-1])
        else:
            vmin = getattr(self.norm, "vmin", None)
            vmax = getattr(self.norm, "vmax", None)
            if vmin is None or vmax is None:
                vmin, vmax = float(np.min(data_finite)), float(np.max(data_finite))

        # Create histogram using matplotlib's hist
        hist_kwargs = {
            "bins": self._hist_bins,
            "range": (vmin, vmax),
            "align": "mid",
        }
        hist_kwargs.update(self._hist_kwargs)

        if self._hist_orientation == "horizontal":
            self.ax.hist(data_finite, orientation="horizontal", **hist_kwargs)
        else:
            self.ax.hist(data_finite, orientation="vertical", **hist_kwargs)

        # Compute color split positions for recoloring patches
        if levels is not None and len(levels) >= 2:
            splitpos = np.asarray(levels, dtype=float)
        elif boundaries is not None and len(boundaries) >= 2:
            splitpos = np.asarray(boundaries, dtype=float)
        else:
            n = int(getattr(self.cmap, "N", 256))
            t = np.linspace(0.0, 1.0, n + 1)
            inv = getattr(self.norm, "inverse", None)
            if callable(inv):
                try:
                    splitpos = np.asarray(inv(t), dtype=float)
                except Exception:
                    splitpos = np.linspace(vmin, vmax, n + 1)
            else:
                splitpos = np.linspace(vmin, vmax, n + 1)

        # Recolor histogram patches to match colorbar
        self._recolor_histogram_patches(splitpos)

    def _recolor_histogram_patches(self, splitpos: NDArray) -> None:
        """Recolor histogram patches to match colorbar colors.

        Parameters
        ----------
        splitpos : NDArray
            Positions where colors change in the colorbar.

        """

        def value_to_facecolor(val: float) -> tuple[float, float, float, float]:
            """Map a data value to a facecolor."""
            # Prefer exact per-level facecolors from contourf if available
            if self._level_facecolors is not None and self._levels is not None:
                levels = np.asarray(self._levels, dtype=float)
                idx = int(
                    np.clip(
                        np.searchsorted(levels, val, side="right") - 1,
                        0,
                        len(levels) - 2,
                    )
                )
                rgba = self._level_facecolors[idx]
                if self._effective_alpha is not None:
                    return (
                        float(rgba[0]),
                        float(rgba[1]),
                        float(rgba[2]),
                        float(self._effective_alpha),
                    )
                return (
                    float(rgba[0]),
                    float(rgba[1]),
                    float(rgba[2]),
                    float(rgba[3] if len(rgba) == 4 else 1.0),
                )
            # Fallback: use the mappable's scalar mapping
            return tuple(self.mappable.to_rgba(val, alpha=self._effective_alpha))  # type: ignore[return-value]

        # Iterate over all patches (histogram bars)
        for patch in list(self.ax.patches):
            # Skip non-Rectangle patches (e.g., extend triangles)
            if not isinstance(patch, Rectangle):
                continue

            patch = cast("Rectangle", patch)
            if self.orientation == "vertical":
                # For vertical colorbar, histogram bars are horizontal
                minval = np.atleast_1d(patch.get_y())[0]
                width = patch.get_width()
                height = patch.get_height()
                maxval = minval + height
            else:
                # For horizontal colorbar, histogram bars are vertical
                minval = np.atleast_1d(patch.get_x())[0]
                width = patch.get_width()
                height = patch.get_height()
                maxval = minval + width

            # Find split positions within this bar
            splitbins = [
                minval,
                *splitpos[(splitpos > minval) & (maxval > splitpos)],
                maxval,
            ]

            # If bar spans multiple colors, split it
            if len(splitbins) > 2:
                patch.remove()
                # Create sub-patches for each color segment
                for b0, b1 in zip(splitbins[:-1], splitbins[1:]):
                    center_val = (b0 + b1) / 2
                    color = value_to_facecolor(center_val)

                    if self.orientation == "vertical":
                        # Horizontal bars
                        pi = Rectangle(
                            (0, b0),
                            width,
                            (b1 - b0),
                            facecolor=color,
                            linewidth=0,
                            alpha=None,
                        )
                    else:
                        # Vertical bars
                        pi = Rectangle(
                            (b0, 0),
                            (b1 - b0),
                            height,
                            facecolor=color,
                            linewidth=0,
                            alpha=None,
                        )

                    self.ax.add_patch(pi)
            else:  # Bar is within a single color
                center_val = (minval + maxval) / 2
                color = value_to_facecolor(center_val)
                patch.set_facecolor(color)
                patch.set_linewidth(0)

    def _apply_histogram_styling(self) -> None:
        """Apply histogram-specific styling."""
        # Set histogram scale
        if self._hist_orientation == "horizontal":
            self.ax.set_xscale(self._scale)
        else:
            self.ax.set_yscale(self._scale)

        hide_axis_elements(self.ax)
        # Draw grid on histogram axis
        if self._hist_orientation == "horizontal":
            self.ax.grid(axis="x", which="major", **self._divider_style)
            spine = (
                self.ax.spines["left"]
                if self._location == "left"
                else self.ax.spines["right"]
            )
        else:
            self.ax.grid(axis="y", which="major", **self._divider_style)
            spine = (
                self.ax.spines["bottom"]
                if self._location == "bottom"
                else self.ax.spines["top"]
            )

        # Make the spine visible and apply divider style
        spine.set_visible(True)
        plt.setp(spine, **self._divider_style)

        # Apply min_count and axis inversion
        if self._hist_orientation == "horizontal":
            xlim = self.ax.get_xlim()
            self.ax.set_xlim(xlim[1], self._min_count)
            if self._location == "left":
                self.ax.invert_xaxis()
        else:
            ylim = self.ax.get_ylim()
            self.ax.set_ylim(ylim[1], self._min_count)
            if self._location == "bottom":
                self.ax.invert_yaxis()


class HistColorbar:
    """A colorbar with an embedded histogram showing data distribution.

    This class creates a colorbar that includes a histogram visualization
    adjacent to the color gradient, providing insight into how data values
    are distributed across the colormap range.

    Parameters
    ----------
    data : array-like
        The data values to create the histogram from. The infinite and NaN
        values are automatically removed.
    mappable : matplotlib.cm.ScalarMappable or None, optional
        A mappable object (e.g., from imshow, contourf) containing cmap and norm.
        If None, cmap and norm must be provided explicitly.
    cax : matplotlib.axes.Axes or None, optional
        Axes into which the colorbar will be drawn. If None, space will be
        stolen from the parent axes.
    ax : matplotlib.axes.Axes or array of Axes or None, optional
        Parent axes from which space for a new colorbar axes will be stolen.
        If None, uses the axes from the mappable.
    use_gridspec : bool, default True
        If True and the parent axes is a Subplot, use gridspec to create the
        colorbar axes. Otherwise use make_axes.
    location : None or {'left', 'right', 'top', 'bottom'}, optional
        The location, relative to the parent Axes, where the colorbar Axes is
        created. It also determines the *orientation* of the colorbar (colorbars
        on the left and right are vertical, colorbars at the top and bottom are
        horizontal). If None, the location will come from the *orientation*
        (vertical colorbars on the right, horizontal ones at the bottom).
    orientation : {'vertical', 'horizontal'} or None, optional
        Orientation of the colorbar. If None, determined from location.
    aspect : float, default 5
        Ratio of long to short dimensions of colorbar.
    fraction : float, default 0.2
        Fraction of original Axes to use for colorbar.
    hist_fraction : float, default 0.85
        Fraction of the colorbar axes allocated to the histogram (0-1).
        The remaining fraction is allocated to the colorbar gradient.
    pad : float or None, optional
        Fraction of original Axes between colorbar and new image Axes.
        If None, defaults to 0.05 for vertical, 0.15 for horizontal.
    alpha : float or None, optional
        Alpha transparency value for the histogram patches (0-1).
    shrink : float, default 1.0
        Fraction by which to multiply the size of the colorbar relative to the
        parent axes. Similar to matplotlib's colorbar shrink parameter.
    extend : {'neither', 'both', 'min', 'max'}, default 'neither'
        Make pointed end(s) for out-of-range values (unless 'neither'). These
        are set for a given colormap using the colormap set_under and set_over
        methods.
    extendfrac : float, default 0.025
        Fraction of the colorbar length to use for the extension triangles.
    cmap : str or Colormap or None, optional
        Colormap to use. Required if mappable is None.
    norm : Normalize or BoundaryNorm or None, optional
        Normalization instance. If None, creates a Normalize instance from
        the data range.
    log : bool, default False
        If True, use logarithmic scale for the histogram count axis.
    min_count : float or 'auto', default 'auto'
        Minimum count value for the histogram axis. If 'auto', uses 0.5 for
        log scale and 0 for linear scale.
    label : str or None, optional
        Label for the colorbar axis.
    hist_label : str or None, optional
        Label for the histogram axis. For vertical orientation, this appears
        on the x-axis (count axis). For horizontal orientation, this appears
        on the y-axis (count axis).
    hist_bins : int or array-like or 'auto', default 'auto'
        Number of histogram bins or bin edges. If 'auto', will match the number
        of levels in the colorbar if using BoundaryNorm, otherwise uses 100 bins
        for continuous colormaps.
    divider_style : dict or None, optional
        Style for the divider line between the colorbar and histogram. If None,
        uses a gray dashed line
        (e.g. `{"color": "0.35", "linestyle": (0, (5, 5)), "linewidth": 1}`).
    cbar_kwargs : dict or None, optional
        Additional keyword arguments to pass to :class:`matplotlib.colorbar.Colorbar`.
    hist_kwargs : dict or None, optional
        Additional keyword arguments to pass to :func:`matplotlib.pyplot.hist`.

    Notes
    -----
    To customize ticks and labels after creation, use the following methods:

    - :meth:`set_cbar_ticks` : Set colorbar tick positions and labels
    - :meth:`set_hist_ticks` : Set histogram tick positions and labels
    - :meth:`cbar_tick_params` : Customize colorbar tick appearance
    - :meth:`hist_tick_params` : Customize histogram tick appearance
    - :meth:`set_cbar_label` : Set colorbar label with custom styling
    - :meth:`set_hist_label` : Set histogram label with custom styling

    Attributes
    ----------
    ax : Axes
        The parent axes containing the colorbar and histogram.
    ax_cbar : Axes
        The axes containing the colorbar.
    ax_hist : Axes
        The axes containing the histogram.
    cbar : Colorbar
        The colorbar object.

    """

    fig: Figure | SubFigure
    ax: Axes
    ax_cbar: Axes
    ax_hist: Axes
    cbar: Colorbar

    _hist_locater: Locator
    _hist_formatter: Formatter
    _min_count: float
    _scale: Literal["linear", "log"]
    _has_extend: bool
    _parent_ax: Axes | None

    def __init__(
        self,
        data: ArrayLike,
        mappable: ScalarMappable | None = None,
        cax: Axes | None = None,
        ax: Axes | NDArray | Iterable[Axes] | None = None,
        use_gridspec: bool = True,
        location: Literal["left", "right", "top", "bottom"] | None = None,
        orientation: Literal["vertical", "horizontal"] | None = None,
        aspect: float = 5,
        fraction: float = 0.2,
        hist_fraction: float = 0.85,
        pad: float | None = None,
        alpha: float | None = None,
        shrink: float = 1.0,
        extend: Literal["neither", "both", "min", "max"] = "neither",
        extendfrac: float = 0.025,
        cmap: str | colors.Colormap | None = None,
        norm: colors.Normalize | colors.BoundaryNorm | None = None,
        log: bool = False,
        min_count: float | Literal["auto"] = "auto",
        label: str | None = None,
        hist_label: str | None = None,
        hist_bins: int | NDArray | Literal["auto"] = "auto",
        divider_style: dict[str, Any] | None = None,
        cbar_kwargs: dict | None = None,
        hist_kwargs: dict | None = None,
    ) -> None:
        """Initialize the HistColorbar object."""
        # Save current axes to restore later
        current_ax = plt.gca() if plt.get_fignums() else None

        # Initialize internal state variables
        self._has_extend = False
        self._parent_ax = None

        self.data = np.asanyarray(data).flatten()
        self.location, self.orientation = _determine_location_orientation(
            location, orientation
        )
        self.aspect = aspect
        self.fraction = fraction
        self.hist_fraction = hist_fraction
        # Set default pad based on orientation
        if pad is None:
            self.pad = 0.15 if self.orientation == "horizontal" else 0.05
        else:
            self.pad = pad
        self.alpha = alpha
        self.shrink = shrink
        self.extend = extend
        self.extendfrac = extendfrac
        self.log = log
        self._min_count_origin = min_count
        # Initialize scale and min_count
        self._scale: Literal["linear", "log"] = "log" if log else "linear"
        self._min_count = 0.5 if log else 0
        if min_count != "auto":
            self._min_count = min_count
        self.label = label
        self.hist_label = hist_label
        self.divider_style = (
            divider_style
            if divider_style is not None
            else {
                "color": "0.35",
                "linestyle": (0, (5, 5)),
                "linewidth": 1,
            }
        )
        self.hist_kwargs = hist_kwargs if hist_kwargs is not None else {}
        self.cbar_kwargs = cbar_kwargs if cbar_kwargs is not None else {}

        # parse colormap and normalization FIRST (before hist_bins)
        if mappable is not None:
            # Support both ScalarMappable and ContourSet-like objects
            get_cmap = getattr(mappable, "get_cmap", None)
            self.cmap = get_cmap() if callable(get_cmap) else mappable.cmap
            self.norm = mappable.norm
            # Capture levels if available (e.g., contourf/contour)
            self._levels = getattr(mappable, "levels", None)
            self._mappable = mappable

            # If the mappable comes from contourf with explicit colors,
            # capture per-level facecolors so we can exactly match them.
            self._level_facecolors = None
            try:
                # QuadContourSet itself IS a Collection with get_facecolors()
                # returning an (N, 4) array for N contour regions
                if self._levels is not None and hasattr(mappable, "get_facecolors"):
                    fcs_array = mappable.get_facecolors()
                    if fcs_array is not None and len(fcs_array) > 0:
                        # fcs_array should be shape (n_levels-1, 4) for RGBA
                        expected_count = len(self._levels) - 1
                        if len(fcs_array) >= expected_count:
                            # Take the first expected_count colors
                            fcs_array = fcs_array[:expected_count]
                            self._level_facecolors = np.asarray(fcs_array, dtype=float)
            except Exception:
                # Best-effort only; fall back to colormap mapping
                self._level_facecolors = None
        else:
            if cmap is None:
                msg = "Either mappable or cmap must be provided"
                logger.error(msg)
                raise ValueError(msg)
            self.cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

            # Create normalization
            if norm is None:
                data_finite = data[np.isfinite(data)]
                vmin, vmax = data_finite.min(), data_finite.max()
                self.norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                self.norm = norm
            self._levels = None

            # ScalarMappable for consistent RGBA mapping
            self._mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

        # Resolve effective alpha: prefer explicit alpha, else artist's scalar alpha
        self._effective_alpha = self._resolve_effective_alpha()

        # Auto-detect histogram bins from norm if 'auto' (AFTER norm is set)
        self.hist_bins = self._determine_hist_bins(hist_bins)

        # parse cax and ax
        if ax is None:
            ax = getattr(mappable, "axes", None)

        # Store parent ax for position updates
        if ax is not None:
            self._parent_ax = (
                ax if not isinstance(ax, (list, np.ndarray)) else ax.flat[0]
            )

        cax, kwargs = self._create_hcb_axes(cax, ax, use_gridspec)
        self.fig = cax.get_figure(root=False)
        self.fig.stale = True

        # Store the container axes (cax) for position updates
        self.ax = cax

        # Create axes for colorbar and histogram
        self.ax_cbar, self.ax_hist = self._create_hist_and_cbar_axes(cax)

        # Draw colorbar and histogram
        self._draw_colorbar(
            **{k: v for k, v in kwargs.items() if k not in NON_COLORBAR_KEYS}
        )
        self._draw_histogram()

        # Apply customizations
        self._apply_default_customizations()

        # Create ghost tick labels on container axes for constrained_layout
        self._create_ghost_ticklabels()

        # Restore original current axes
        if current_ax is not None and current_ax in self.fig.axes:
            plt.sca(current_ax)

    @property
    def hist_orientation(self) -> Literal["vertical", "horizontal"]:
        """Orientation of the histogram."""
        return "vertical" if self.orientation == "horizontal" else "horizontal"

    def _resolve_effective_alpha(self) -> float | None:
        """Resolve the alpha to use for both histogram patches and colorbar.

        Preference order:
        1) Explicit ``alpha`` passed to HistColorbar
        2) Scalar alpha on the provided mappable/artist (e.g., imshow(alpha=0.5))
        3) None (fall back to colormap's own alpha per color)

        Array-like alpha on the artist is not supported and will be ignored.
        """
        # 1) explicit alpha
        if self.alpha is not None:
            try:
                return float(np.clip(float(self.alpha), 0.0, 1.0))
            except Exception:  # pragma: no cover - defensive
                return self.alpha  # type: ignore[return-value]

        # 2) artist alpha if available
        artist_alpha: Any | None = None
        get_alpha = getattr(self._mappable, "get_alpha", None)
        if callable(get_alpha):
            try:
                artist_alpha = get_alpha()
            except Exception:  # pragma: no cover - defensive
                artist_alpha = None
        if artist_alpha is None:
            artist_alpha = getattr(self._mappable, "alpha", None)

        if artist_alpha is None:
            return None

        # If array-like alpha, we cannot represent it uniformly; ignore
        try:
            if np.isscalar(artist_alpha):
                return float(np.clip(float(artist_alpha), 0.0, 1.0))
        except Exception:  # pragma: no cover - defensive
            pass

        warnings.warn(
            (
                "Array-like alpha on the mappable is not supported by HistColorbar; "
                "using colormap alpha."
            ),
            stacklevel=2,
        )
        return None

    def _determine_hist_bins(
        self, hist_bins: int | NDArray | Literal["auto"]
    ) -> int | NDArray:
        """Determine the number of histogram bins.

        If hist_bins is 'auto', will match the number of levels in the colorbar
        if using BoundaryNorm, otherwise uses 100 bins for continuous colormaps.

        Parameters
        ----------
        hist_bins : BinsType
            The histogram bins specification.

        Returns
        -------
        int | NDArray
            The determined number of bins or bin edges.

        """
        if isinstance(hist_bins, str) and hist_bins == "auto":
            # Prefer explicit contour levels if available
            levels = getattr(self, "_levels", None)
            if levels is not None:
                return max(1, len(levels) - 1)
            # Check if norm is BoundaryNorm (discrete levels)
            if hasattr(self.norm, "boundaries"):
                # BoundaryNorm has discrete levels
                return max(1, len(self.norm.boundaries) - 1)
            # Continuous colormap, use fine binning
            return 100
        return hist_bins

    def _create_hcb_axes(
        self,
        cax: Axes | None,
        ax: Axes | NDArray | Iterable[Axes] | None,
        use_gridspec: bool = True,
    ) -> tuple[Axes, dict]:
        """Create a parent axes for colorbar and histogram.

        The Axes is placed in the figure of the *parent* Axes, by resizing and
        repositioning *parent*.

        References
        ----------
        This is a modified version of the :meth:`matplotlib.figure.FigureBase.colorbar`
        method: `https://github.com/matplotlib/matplotlib/blob/v3.10.7/lib/matplotlib/figure.py#L1193-L1311`_

        """
        # create kwargs for HistColorbar axes
        kwargs = self.cbar_kwargs.copy()
        kwargs.update(
            mappable=self._mappable,
            location=self.location,
            orientation=None,  # location is enough
            aspect=self.aspect,
            fraction=self.fraction,
            pad=self.pad,
            # Do NOT pass alpha to Colorbar; handle alpha explicitly on solids
            shrink=self.shrink,
            extend=self.extend,
            extendfrac=self.extendfrac,
        )
        if cax is None:
            if ax is None:
                msg = (
                    "Unable to determine Axes to steal space for HistColorbar."
                    "Either provide the *cax* argument to use as the Axes for "
                    "the Colorbar, provide the *ax* argument to steal space "
                    "from it, or add *mappable* to an Axes."
                )
                logger.error(msg)
                raise ValueError(msg)
            fig = (  # Figure of first Axes; logic copied from make_axes.
                [*ax.flat]
                if isinstance(ax, np.ndarray)
                else [*ax]
                if np.iterable(ax)
                else [ax]
            )[0].get_figure(root=False)
            current_ax = fig.gca()
            if (
                fig.get_layout_engine() is not None
                and not fig.get_layout_engine().colorbar_gridspec
            ):
                use_gridspec = False
            if (
                use_gridspec
                and isinstance(ax, mpl.axes._base._AxesBase)
                and ax.get_subplotspec()
            ):
                cax, kwargs = cbar.make_axes_gridspec(ax, **kwargs)
            else:
                cax, kwargs = cbar.make_axes(ax, **kwargs)
            # make_axes calls add_{axes,subplot} which changes gca; undo that.
            fig.sca(current_ax)
            cax.grid(visible=False, which="both", axis="both")

        return cax, kwargs

    def _create_hist_and_cbar_axes(self, parent_ax: Axes) -> tuple[Axes, Axes]:
        """Create separate axes for colorbar and histogram.

        Parameters
        ----------
        parent_ax : Axes
            The parent axes to split space from for the colorbar and histogram.

        Returns
        -------
        ax_cbar, ax_hist: tuple[Axes, Axes]
            The colorbar and histogram axes.

        """
        hide_axis_elements(parent_ax)

        # Create axes using appropriate method based on parent_ax type
        parent_subplotspec = parent_ax.get_subplotspec()
        if parent_subplotspec is not None:
            # Use SubplotSpec-based gridspec (for tight_layout)
            ax_cbar, ax_hist = _create_hcb_gridspec_axes(
                parent_ax, self.hist_fraction, self.location
            )
        else:
            # Use inset_axes (for constrained_layout or no layout)
            ax_cbar, ax_hist = _create_hcb_inset_axes(
                parent_ax, self.hist_fraction, self.location
            )

        ax_cbar.set_zorder(10)
        ax_hist.set_zorder(11)  # histogram above colorbar for line visibility

        # Join axes - histogram shares the data axis with colorbar
        # This ensures they stay aligned even when limits change
        if self.orientation == "horizontal":
            ax_hist.sharex(ax_cbar)
        else:
            ax_hist.sharey(ax_cbar)

        return ax_cbar, ax_hist

    def _draw_colorbar(self, **kwargs) -> None:
        """Draw the colorbar."""
        # Ensure we don't accidentally forward unsupported args like 'alpha'
        if "alpha" in kwargs:
            kwargs.pop("alpha", None)
        self.cbar = Colorbar(self.ax_cbar, **kwargs)
        if self._effective_alpha is not None and hasattr(self.cbar, "solids"):
            self.cbar.solids.set_alpha(self._effective_alpha)
        hide_spines(self.ax_cbar)

    def _draw_histogram(self) -> None:
        """Draw the histogram using _Histogram class."""
        # Create _Histogram instance which inherits from Colorbar
        # This automatically handles extend triangles and axes positioning
        self.hist = _Histogram(
            ax=self.ax_hist,
            data=self.data,
            mappable=self._mappable,
            orientation=self.orientation,
            hist_bins=self.hist_bins,
            hist_kwargs=self.hist_kwargs,
            min_count=self.min_count,
            scale="log" if self.log else "linear",
            divider_style=self.divider_style,
            location=self.location,
            extend=self.extend,
            extendfrac=self.extendfrac,
            _levels=self._levels,
            _level_facecolors=self._level_facecolors,
            _effective_alpha=self._effective_alpha,
        )

    @property
    def scale(self) -> Literal["linear", "log"]:
        """Get the scale of the histogram axis."""
        return self._scale

    @property
    def min_count(self) -> float:
        """Get the minimum count for the histogram."""
        return self._min_count

    def set_scale(self, scale: Literal["linear", "log"]) -> None:
        """Set the scale of the histogram axis."""
        if self.hist_orientation == "horizontal":
            self.ax_hist.set_xscale(scale)
        else:
            self.ax_hist.set_yscale(scale)

        self._scale = scale
        self.set_min_count(self._min_count_origin)

    def set_min_count(self, min_count: float | Literal["auto"] = "auto") -> None:
        """Set the minimum count for the histogram."""
        if min_count == "auto":
            min_count = 0.5 if self.scale == "log" else 0
        if min_count < 0:
            msg = "min_count must be positive"
            logger.error(msg)
            raise ValueError(msg)
        self._min_count = min_count

    def set_hist_locator(
        self, locator: Locator, which: Literal["major", "minor"] = "major"
    ) -> None:
        """Set the locator for the histogram axis.

        Parameters
        ----------
        locator : Locator
            The locator to use.
        which : {'major', 'minor'}, default: 'major'
            Which ticks to apply the locator to.

        """
        if which == "major":
            if self.hist_orientation == "horizontal":
                self.ax_hist.xaxis.set_major_locator(locator)
            else:
                self.ax_hist.yaxis.set_major_locator(locator)
        elif which == "minor":
            if self.hist_orientation == "horizontal":
                self.ax_hist.xaxis.set_minor_locator(locator)
            else:
                self.ax_hist.yaxis.set_minor_locator(locator)
        else:
            msg = f"which must be 'major' or 'minor', got {which}"
            logger.error(msg)
            raise ValueError(msg)

    def set_hist_formatter(
        self, formatter: Formatter, which: Literal["major", "minor"] = "major"
    ) -> None:
        """Set the formatter for the histogram axis.

        Parameters
        ----------
        formatter : Formatter
            The formatter to use.
        which : {'major', 'minor'}, default: 'major'
            Which ticks to apply the formatter to.

        """
        if which == "major":
            if self.hist_orientation == "horizontal":
                self.ax_hist.xaxis.set_major_formatter(formatter)
            else:
                self.ax_hist.yaxis.set_major_formatter(formatter)
        elif which == "minor":
            if self.hist_orientation == "horizontal":
                self.ax_hist.xaxis.set_minor_formatter(formatter)
            else:
                self.ax_hist.yaxis.set_minor_formatter(formatter)
        else:
            msg = f"which must be 'major' or 'minor', got {which}"
            logger.error(msg)
            raise ValueError(msg)

    def set_cbar_locator(
        self, locator: Locator, which: Literal["major", "minor"] = "major"
    ) -> None:
        """Set the locator for the colorbar axis.

        Parameters
        ----------
        locator : Locator
            The locator to use.
        which : {'major', 'minor'}, default: 'major'
            Which ticks to apply the locator to.

        """
        if which == "major":
            if self.orientation == "horizontal":
                self.ax_cbar.xaxis.set_major_locator(locator)
            else:
                self.ax_cbar.yaxis.set_major_locator(locator)
        elif which == "minor":
            if self.orientation == "horizontal":
                self.ax_cbar.xaxis.set_minor_locator(locator)
            else:
                self.ax_cbar.yaxis.set_minor_locator(locator)
        else:
            msg = f"which must be 'major' or 'minor', got {which}"
            logger.error(msg)
            raise ValueError(msg)

    def set_cbar_formatter(
        self, formatter: Formatter, which: Literal["major", "minor"] = "major"
    ) -> None:
        """Set the formatter for the colorbar axis.

        Parameters
        ----------
        formatter : Formatter
            The formatter to use.
        which : {'major', 'minor'}, default: 'major'
            Which ticks to apply the formatter to.

        """
        if which == "major":
            if self.orientation == "horizontal":
                self.ax_cbar.xaxis.set_major_formatter(formatter)
            else:
                self.ax_cbar.yaxis.set_major_formatter(formatter)
        elif which == "minor":
            if self.orientation == "horizontal":
                self.ax_cbar.xaxis.set_minor_formatter(formatter)
            else:
                self.ax_cbar.yaxis.set_minor_formatter(formatter)
        else:
            msg = f"which must be 'major' or 'minor', got {which}"
            logger.error(msg)
            raise ValueError(msg)

    def _apply_default_customizations(self) -> None:
        """Apply default custom ticks, labels, and formatting."""
        # Set colorbar and histogram labels
        if self.label is not None:
            self.set_cbar_label(self.label)
        if self.hist_label is not None:
            self.set_hist_label(self.hist_label)

        # Set tick formatting of histogram axis
        if self.log:
            self.set_hist_locator(LogLocator())
            self.set_hist_formatter(LogFormatterSciNotation())
        else:
            self.set_hist_locator(AutoLocator())
            self.set_hist_formatter(ScalarFormatter())
        # Turn off minor ticks
        self.set_hist_locator(NullLocator(), "minor")
        self.set_hist_formatter(NullFormatter(), "minor")

        # ticks of colorbar axis are handled by Colorbar

    def _create_ghost_ticklabels(self) -> None:
        """Create invisible 'ghost' tick labels on container axes.

        This is the KEY solution for constrained_layout compatibility:
        1. Configure container axes to have same tick locations as ax_cbar
        2. Set tick label location (left/right/top/bottom) to match ax_cbar
        3. Make ghost labels invisible after layout calculation
        4. Constrained_layout sees and accounts for these labels
        5. The actual visible labels remain on ax_cbar where they belong

        This elegant approach lets constrained_layout calculate proper spacing
        while keeping our actual rendering unchanged.
        """
        # Configure container axes to match ax_cbar's tick system
        # This is about tick "location" not physical position

        if self.orientation == "vertical":
            # Get tick information from ax_cbar
            tick_locs = self.ax_cbar.yaxis.get_ticklocs()
            tick_labels = [
                label.get_text() for label in self.ax_cbar.yaxis.get_ticklabels()
            ]

            # Set container axes limits to match colorbar data range
            ylim = self.ax_cbar.get_ylim()
            self.ax.set_ylim(ylim[0], ylim[1])
            self.ax.set_xlim(0, 1)  # Dummy x-axis

            # Set ticks at same data values as ax_cbar
            self.ax.yaxis.set_ticks(tick_locs, tick_labels)

            # Configure tick location to match ax_cbar
            if self.location == "left":
                self.ax.yaxis.tick_left()
                self.ax.yaxis.set_label_position("left")
                # Re-enable tick labels after hide_tick_labels was called
                # But hide the tick marks themselves (length=0)
                self.ax.tick_params(axis="y", labelleft=True, left=False, length=0)
            else:  # right
                self.ax.yaxis.tick_right()
                self.ax.yaxis.set_label_position("right")
                # Re-enable tick labels after hide_tick_labels was called
                # But hide the tick marks themselves (length=0)
                self.ax.tick_params(axis="y", labelright=True, right=False, length=0)

            # Hide x-axis completely
            self.ax.xaxis.set_label_position("bottom")

        else:  # horizontal
            # Get tick information from ax_cbar
            tick_locs = self.ax_cbar.xaxis.get_ticklocs()
            tick_labels = [
                label.get_text() for label in self.ax_cbar.xaxis.get_ticklabels()
            ]

            # Set container axes limits to match colorbar data range
            xlim = self.ax_cbar.get_xlim()
            self.ax.set_xlim(xlim[0], xlim[1])
            self.ax.set_ylim(0, 1)  # Dummy y-axis

            # Set ticks at same data values as ax_cbar
            self.ax.xaxis.set_ticks(tick_locs)
            self.ax.xaxis.set_ticklabels(tick_labels)

            # Configure tick location to match ax_cbar
            if self.location == "bottom":
                self.ax.xaxis.tick_bottom()
                self.ax.xaxis.set_label_position("bottom")
                # Re-enable tick labels after hide_tick_labels was called
                # But hide the tick marks themselves (length=0)
                self.ax.tick_params(axis="x", labelbottom=True, bottom=False, length=0)
            else:  # top
                self.ax.xaxis.tick_top()
                self.ax.xaxis.set_label_position("top")
                # Re-enable tick labels after hide_tick_labels was called
                # But hide the tick marks themselves (length=0)
                self.ax.tick_params(axis="x", labeltop=True, top=False, length=0)

            # Hide y-axis completely
            self.ax.yaxis.set_label_position("left")

        # Setup callback to hide ghost labels after layout
        self._setup_ghost_label_hiding()

    def _setup_ghost_label_hiding(self) -> None:
        """Set up callback to hide ghost labels after layout calculation.

        The ghost labels need to be visible during constrained_layout's calculation
        phase, but must be hidden during actual rendering to avoid duplication.

        Strategy: Use a draw_event callback that runs AFTER layout but BEFORE render.
        The callback should only hide labels after the FIRST draw (when layout is done).
        """
        # Track if we've already hidden the labels
        self._ghost_labels_hidden = False

        def hide_ghost_labels(event) -> None:  # noqa: ANN001, ARG001
            """Hide the ghost tick labels on container axes after first draw."""
            # Only hide after the first draw (when constrained_layout has calculated)
            if self._ghost_labels_hidden:
                return

            self._ghost_labels_hidden = True

            # Hide tick labels on container axes
            for label in self.ax.yaxis.get_ticklabels():
                label.set_alpha(0)
            for label in self.ax.xaxis.get_ticklabels():
                label.set_alpha(0)

        # Connect to draw event - this runs after layout but before render
        if hasattr(self.fig.canvas, "mpl_connect"):
            self.fig.canvas.mpl_connect("draw_event", hide_ghost_labels)

    def cbar_tick_params(
        self,
        axis: Literal["x", "y", "both"] = "both",
        which: Literal["major", "minor", "both"] = "major",
        **kwargs,
    ) -> None:
        """Set appearance of ticks (labels), and gridlines on colorbar or histogram.

        Tick properties that are not explicitly set using the keyword
        arguments remain unchanged unless *reset* is True. For the current
        style settings, see `.Axis.get_tick_params`.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}, default: 'both'
            The axis to which the parameters are applied.
        which : {'major', 'minor', 'both'}, default: 'major'
            The group of ticks to which the parameters are applied.
        **kwargs
            Keyword arguments to be passed to :meth:`~.Axis.tick_params`.

        Other Parameters
        ----------------
        direction : {'in', 'out', 'inout'}
            Puts ticks inside the Axes, outside the Axes, or both.
        length : float
            Tick length in points.
        width : float
            Tick width in points.
        color : :mpltype:`color`
            Tick color.
        pad : float
            Distance in points between tick and label.
        labelsize : float or str
            Tick label font size in points or as a string (e.g., 'large').
        labelcolor : :mpltype:`color`
            Tick label color.
        labelfontfamily : str
            Tick label font.
        colors : :mpltype:`color`
            Tick color and label color.
        zorder : float
            Tick and label zorder.
        bottom, top, left, right : bool
            Whether to draw the respective ticks.
        labelbottom, labeltop, labelleft, labelright : bool
            Whether to draw the respective tick labels.
        labelrotation : float
            Tick label rotation
        grid_color : :mpltype:`color`
            Gridline color.
        grid_alpha : float
            Transparency of gridlines: 0 (transparent) to 1 (opaque).
        grid_linewidth : float
            Width of gridlines in points.
        grid_linestyle : str
            Any valid `.Line2D` line style spec.

        """
        self.ax_hist.tick_params(axis, which=which, **kwargs)

    def hist_tick_params(
        self,
        axis: Literal["x", "y", "both"] = "both",
        which: Literal["major", "minor", "both"] = "major",
        **kwargs,
    ) -> None:
        """Set appearance of ticks (labels), and gridlines on colorbar or histogram.

        Tick properties that are not explicitly set using the keyword
        arguments remain unchanged unless *reset* is True. For the current
        style settings, see `.Axis.get_tick_params`.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}, default: 'both'
            The axis to which the parameters are applied.
        which : {'major', 'minor', 'both'}, default: 'major'
            The group of ticks to which the parameters are applied.
        **kwargs
            Keyword arguments to be passed to :meth:`~.Axis.tick_params`.

        Other Parameters
        ----------------
        direction : {'in', 'out', 'inout'}
            Puts ticks inside the Axes, outside the Axes, or both.
        length : float
            Tick length in points.
        width : float
            Tick width in points.
        color : :mpltype:`color`
            Tick color.
        pad : float
            Distance in points between tick and label.
        labelsize : float or str
            Tick label font size in points or as a string (e.g., 'large').
        labelcolor : :mpltype:`color`
            Tick label color.
        labelfontfamily : str
            Tick label font.
        colors : :mpltype:`color`
            Tick color and label color.
        zorder : float
            Tick and label zorder.
        bottom, top, left, right : bool
            Whether to draw the respective ticks.
        labelbottom, labeltop, labelleft, labelright : bool
            Whether to draw the respective tick labels.
        labelrotation : float
            Tick label rotation
        grid_color : :mpltype:`color`
            Gridline color.
        grid_alpha : float
            Transparency of gridlines: 0 (transparent) to 1 (opaque).
        grid_linewidth : float
            Width of gridlines in points.
        grid_linestyle : str
            Any valid `.Line2D` line style spec.

        """
        self.ax_hist.tick_params(axis, which=which, **kwargs)

    def set_cbar_label(
        self,
        label: str,
        **kwargs: Any,
    ) -> None:
        """Set labels for the colorbar.

        This method allows you to set labels for the colorbar and histogram
        with different styling options. You can call it multiple times with
        different parameters to apply different styles to each label.

        Parameters
        ----------
        label : str, optional
            Label for the colorbar axis. If None, colorbar label is not changed.
        **kwargs
            Additional keyword arguments passed to the label setting methods.
            Common options include: fontsize, color, fontweight, labelpad, etc.

        """
        self.cbar.set_label(label, **kwargs)

    def set_hist_label(
        self,
        label: str,
        **kwargs: Any,
    ) -> None:
        """Set labels for the histogram.

        This method allows you to set labels for the histogram
        with different styling options. You can call it multiple times with
        different parameters to apply different styles to each label.

        Parameters
        ----------
        label : str
            Label for the histogram axis.
        **kwargs
            Additional keyword arguments passed to the label setting methods.
            Common options include: fontsize, color, fontweight, labelpad, etc.

        """
        if self.orientation == "vertical":
            # For vertical colorbar, histogram bars are horizontal
            # Count is on x-axis
            self.ax_hist.set_xlabel(label, **kwargs)
        else:
            # For horizontal colorbar, histogram bars are vertical
            # Count is on y-axis
            self.ax_hist.set_ylabel(label, **kwargs)

    def set_label(
        self,
        label: str,
        which: Literal["cbar", "hist"] = "cbar",
        **kwargs: Any,
    ) -> None:
        """Set label for the colorbar or histogram.

        This method allows you to set labels for the colorbar and histogram
        with different styling options. You can call it multiple times with
        different parameters to apply different styles to each label.

        Parameters
        ----------
        label : str
            Label for the colorbar or histogram axis.
        which : {'cbar', 'hist'}, default: 'cbar'
            Which axis to set the label for.
        **kwargs
            Additional keyword arguments passed to the label setting methods.
            Common options include: fontsize, color, fontweight, labelpad, etc.

        """
        if which == "cbar":
            self.set_cbar_label(label, **kwargs)
        elif which == "hist":
            self.set_hist_label(label, **kwargs)
        else:
            msg = f"which must be 'cbar' or 'hist', got {which}"
            logger.error(msg)
            raise ValueError(msg)

    def set_cbar_ticks(
        self,
        ticks: ArrayLike,
        labels: Iterable[str] | None = None,
        *,
        minor: bool = False,
        **kwargs,
    ) -> None:
        """Set tick locations and labels for the colorbar axis.

        Parameters
        ----------
        ticks : 1D array-like
            List of tick locations.
        labels : list of str, optional
            List of tick labels. If not set, the labels show the data value.
        minor : bool, default: False
            If ``False``, set the major ticks; if ``True``, the minor ticks.
        **kwargs
            `.Text` properties for the labels. These take effect only if you
            pass *labels*. In other cases, please use `~.Axes.tick_params`.

        """
        self.cbar.set_ticks(
            np.asarray(ticks).tolist(),
            labels=np.asarray(labels).tolist(),
            minor=minor,
            **kwargs,
        )

    def set_hist_ticks(
        self,
        ticks: ArrayLike,
        labels: Iterable[str] | None = None,
        *,
        minor: bool = False,
        **kwargs,
    ) -> None:
        """Set tick locations for the histogram axis.

        Parameters
        ----------
        ticks : 1D array-like
            List of tick locations.
        labels : list of str, optional
            List of tick labels. If not set, the labels show the data value.
        minor : bool, default: False
            If ``False``, set the major ticks; if ``True``, the minor ticks.
        **kwargs
            `.Text` properties for the labels. These take effect only if you
            pass *labels*. In other cases, please use `~.Axes.tick_params`.

        """
        if self.orientation == "vertical":
            self.ax_hist.set_xticks(ticks, labels, minor=minor, **kwargs)
        else:
            self.ax_hist.set_yticks(ticks, labels, minor=minor, **kwargs)

    def set_ticks(
        self,
        ticks: ArrayLike,
        labels: Iterable[str] | None = None,
        *,
        which: Literal["cbar", "hist"] = "cbar",
        minor: bool = False,
        **kwargs,
    ) -> None:
        """Set tick locations for the colorbar and histogram axes.

        Parameters
        ----------
        ticks : 1D array-like
            List of tick locations.
        labels : list of str, optional
            List of tick labels. If not set, the labels show the data value.
        which : {'cbar', 'hist'}, default: 'cbar'
            Which axis to set the tick locations for.
        minor : bool, default: False
            If ``False``, set the major ticks; if ``True``, the minor ticks.
        **kwargs
            `.Text` properties for the labels. These take effect only if you
            pass *labels*. In other cases, please use `~.Axes.tick_params`.

        """
        if which == "cbar":
            self.set_cbar_ticks(ticks, labels, minor=minor, **kwargs)
        elif which == "hist":
            self.set_hist_ticks(ticks, labels, minor=minor, **kwargs)
        else:
            msg = f"which must be 'cbar' or 'hist', got {which}"
            logger.error(msg)
            raise ValueError(msg)

    def set_cbar_ticklabels(self, labels: Iterable[str], **kwargs: Any) -> None:
        """Set tick labels for the colorbar axis.

        Parameters
        ----------
        labels : list of str
            List of tick labels.
        **kwargs
            `.Text` properties for the labels.

        """
        self.cbar.set_ticklabels(labels, **kwargs)

    def set_hist_ticklabels(self, labels: Iterable[str], **kwargs: Any) -> None:
        """Set tick labels for the histogram axis.

        Parameters
        ----------
        labels : list of str
            List of tick labels.
        **kwargs
            `.Text` properties for the labels.

        """
        if self.orientation == "vertical":
            self.ax_hist.set_xticklabels(labels, **kwargs)
        else:
            self.ax_hist.set_yticklabels(labels, **kwargs)

    def set_ticklabels(
        self,
        labels: Iterable[str],
        which: Literal["cbar", "hist"] = "cbar",
        **kwargs: Any,
    ) -> None:
        """Set tick labels for the colorbar and histogram axes.

        Parameters
        ----------
        labels : list of str
            List of tick labels.
        which : {'cbar', 'hist'}, default: 'cbar'
            Which axis to set the tick labels for.
        **kwargs
            `.Text` properties for the labels.

        """
        if which == "cbar":
            self.set_cbar_ticklabels(labels, **kwargs)
        elif which == "hist":
            self.set_hist_ticklabels(labels, **kwargs)
        else:
            msg = f"which must be 'cbar' or 'hist', got {which}"
            logger.error(msg)
            raise ValueError(msg)

    def minorticks_on(self) -> None:
        """Turn on colorbar minor ticks."""
        self.cbar.minorticks_on()

    def minorticks_off(self) -> None:
        """Turn the minor ticks of the colorbar off."""
        self.cbar.minorticks_off()

    def remove(self) -> None:
        """Remove the HistColorbar and clean up.

        This method should be called when removing a HistColorbar to properly
        clean up resources.
        """
        # Remove axes
        if hasattr(self, "ax_cbar") and self.ax_cbar is not None:
            self.ax_cbar.remove()
        if hasattr(self, "ax_hist") and self.ax_hist is not None:
            self.ax_hist.remove()
        if hasattr(self, "ax") and self.ax is not None:
            self.ax.remove()


def _hist_colorbar(  # noqa: D417
    self: Figure | SubFigure,  # noqa: ARG001
    data: ArrayLike,
    mappable: ScalarMappable | None = None,
    cax: Axes | None = None,
    ax: Axes | NDArray | Iterable[Axes] | None = None,
    use_gridspec: bool = True,
    location: Literal["left", "right", "top", "bottom"] | None = None,
    orientation: Literal["vertical", "horizontal"] | None = None,
    aspect: float = 5,
    fraction: float = 0.2,
    hist_fraction: float = 0.85,
    pad: float | None = None,
    alpha: float | None = None,
    shrink: float = 1.0,
    extend: Literal["neither", "both", "min", "max"] = "neither",
    extendfrac: float = 0.025,
    cmap: str | colors.Colormap | None = None,
    norm: colors.Normalize | colors.BoundaryNorm | None = None,
    log: bool = False,
    min_count: float | Literal["auto"] = "auto",
    label: str | None = None,
    hist_label: str | None = None,
    hist_bins: int | NDArray | Literal["auto"] = "auto",
    divider_style: dict[str, Any] | None = None,
    cbar_kwargs: dict | None = None,
    hist_kwargs: dict | None = None,
) -> HistColorbar:
    """Create a histogram-embedded colorbar.

    This is a convenience function that creates a HistColorbar instance.
    All parameters are passed directly to the HistColorbar class constructor.

    Parameters
    ----------
    data : array-like
        The data values to create the histogram from. The infinite and NaN
        values are automatically removed.
    mappable : matplotlib.cm.ScalarMappable or None, optional
        A mappable object (e.g., from imshow, contourf) containing cmap and norm.
        If None, cmap and norm must be provided explicitly.
    cax : matplotlib.axes.Axes or None, optional
        Axes into which the colorbar will be drawn. If None, space will be
        stolen from the parent axes.
    ax : matplotlib.axes.Axes or array of Axes or None, optional
        Parent axes from which space for a new colorbar axes will be stolen.
        If None, uses the axes from the mappable.
    use_gridspec : bool, default True
        If True and the parent axes is a Subplot, use gridspec to create the
        colorbar axes. Otherwise use make_axes.
    location : None or {'left', 'right', 'top', 'bottom'}, optional
        The location, relative to the parent Axes, where the colorbar Axes is
        created. It also determines the *orientation* of the colorbar (colorbars
        on the left and right are vertical, colorbars at the top and bottom are
        horizontal). If None, the location will come from the *orientation*
        (vertical colorbars on the right, horizontal ones at the bottom).
    orientation : {'vertical', 'horizontal'} or None, optional
        Orientation of the colorbar. If None, determined from location.
    aspect : float, default 5
        Ratio of long to short dimensions of colorbar.
    fraction : float, default 0.2
        Fraction of original Axes to use for colorbar.
    hist_fraction : float, default 0.85
        Fraction of the colorbar axes allocated to the histogram (0-1).
        The remaining fraction is allocated to the colorbar gradient.
    pad : float or None, optional
        Fraction of original Axes between colorbar and new image Axes.
        If None, defaults to 0.05 for vertical, 0.15 for horizontal.
    alpha : float or None, optional
        Alpha transparency value for the histogram patches (0-1).
    shrink : float, default 1.0
        Fraction by which to multiply the size of the colorbar relative to the
        parent axes. Similar to matplotlib's colorbar shrink parameter.
    extend : {'neither', 'both', 'min', 'max'}, default 'neither'
        Make pointed end(s) for out-of-range values (unless 'neither'). These
        are set for a given colormap using the colormap set_under and set_over
        methods.
    extendfrac : float, default 0.025
        Fraction of the colorbar length to use for the extension triangles.
    cmap : str or Colormap or None, optional
        Colormap to use. Required if mappable is None.
    norm : Normalize or BoundaryNorm or None, optional
        Normalization instance. If None, creates a Normalize instance from
        the data range.
    log : bool, default False
        If True, use logarithmic scale for the histogram count axis.
    min_count : float or 'auto', default 'auto'
        Minimum count value for the histogram axis. If 'auto', uses 0.5 for
        log scale and 0 for linear scale.
    label : str or None, optional
        Label for the colorbar axis.
    hist_label : str or None, optional
        Label for the histogram axis. For vertical orientation, this appears
        on the x-axis (count axis). For horizontal orientation, this appears
        on the y-axis (count axis).
    hist_bins : int or array-like or 'auto', default 'auto'
        Number of histogram bins or bin edges. If 'auto', will match the number
        of levels in the colorbar if using BoundaryNorm, otherwise uses 100 bins
        for continuous colormaps.
    divider_style : dict or None, optional
        Style for the divider line between the colorbar and histogram. If None,
        uses a gray dashed line
        (e.g. `{"color": "0.35", "linestyle": (0, (5, 5)), "linewidth": 1}`).
    cbar_kwargs : dict or None, optional
        Additional keyword arguments to pass to :class:`matplotlib.colorbar.Colorbar`.
    hist_kwargs : dict or None, optional
        Additional keyword arguments to pass to :func:`matplotlib.pyplot.hist`.

    Returns
    -------
    HistColorbar
        The histogram-embedded colorbar instance with the following key attributes:

        - `ax_cbar` : Axes containing the colorbar
        - `ax_hist` : Axes containing the histogram
        - `cbar` : The colorbar object
        - `fig` : The figure containing the colorbar

    See Also
    --------
    HistColorbar : The main class for creating histogram-embedded colorbars.

    Notes
    -----
    This function is a convenience wrapper around the HistColorbar class.
    For more control and access to additional methods, create a HistColorbar
    instance directly.

    Examples
    --------
    Basic usage with explicit colormap and normalization:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import Normalize
    >>>
    >>> # Generate sample data
    >>> data = np.random.randn(1000)
    >>>
    >>> # Create a simple plot
    >>> fig, ax = plt.subplots()
    >>> norm = Normalize(vmin=-3, vmax=3)
    >>> hcb = fig.hist_colorbar(data=data, cmap="viridis", norm=norm, ax=ax)
    >>> plt.show()

    Using with a mappable object from imshow:

    >>> data_2d = np.random.randn(50, 50)
    >>> fig, ax = plt.subplots()
    >>> im = ax.imshow(data_2d, cmap="coolwarm")
    >>> hcb = fig.hist_colorbar(data=data_2d.flatten(), mappable=im)
    >>> plt.show()

    Horizontal orientation with custom histogram bins:

    >>> data = np.random.randn(5000)
    >>> fig, ax = plt.subplots()
    >>> hcb = fig.hist_colorbar(
    ...     data=data, cmap="plasma", orientation="horizontal", hist_bins=50, ax=ax
    ... )
    >>> plt.show()

    Using logarithmic scale for histogram counts:

    >>> data = np.random.exponential(2, 10000)
    >>> fig, ax = plt.subplots()
    >>> hcb = fig.hist_colorbar(
    ...     data=data,
    ...     cmap="inferno",
    ...     log=True,
    ...     ax=ax,
    ...     label="Intensity",
    ...     hist_label="Count",
    ... )
    >>> plt.show()

    """
    return HistColorbar(
        data=data,
        mappable=mappable,
        cax=cax,
        ax=ax,
        use_gridspec=use_gridspec,
        location=location,
        orientation=orientation,
        aspect=aspect,
        fraction=fraction,
        hist_fraction=hist_fraction,
        pad=pad,
        alpha=alpha,
        shrink=shrink,
        extend=extend,
        extendfrac=extendfrac,
        cmap=cmap,
        norm=norm,
        log=log,
        min_count=min_count,
        label=label,
        hist_label=hist_label,
        hist_bins=hist_bins,
        divider_style=divider_style,
        cbar_kwargs=cbar_kwargs,
        hist_kwargs=hist_kwargs,
    )


def hide_axis_elements(ax: Axes) -> None:
    """Hide all spines, ticks, and tick labels on the given Axes."""
    hide_spines(ax)
    hide_tick_labels(ax)


def hide_spines(ax: Axes) -> None:
    """Hide all spines on the given Axes, but keep ticks and tick labels visible."""
    for spine in ax.spines.values():
        spine.set_visible(False)


def hide_tick_labels(ax: Axes) -> None:
    """Hide tick labels on the given Axes, but keep ticks and spines visible."""
    ax.tick_params(
        which="both",
        left=False,
        right=False,
        top=False,
        bottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        labelbottom=False,
    )


def _determine_location_orientation(
    location: Literal["left", "right", "top", "bottom"] | None,
    orientation: Literal["vertical", "horizontal"] | None,
) -> tuple[
    Literal["left", "right", "top", "bottom"], Literal["vertical", "horizontal"]
]:
    """Determine the location of the colorbar and histogram."""
    # validate location and orientation
    if location in {"left", "right"} and orientation == "horizontal":
        msg = "Horizontal colorbar cannot be on left or right side."
        logger.error(msg)
        raise ValueError(msg)
    if location in {"top", "bottom"} and orientation == "vertical":
        msg = "Vertical colorbar cannot be on top or bottom side."
        logger.error(msg)
        raise ValueError(msg)

    # default location and orientation
    if location is None and orientation is None:
        return "right", "vertical"

    # determine location and orientation from each other
    if location is None:
        location = "right" if orientation == "vertical" else "bottom"
    if orientation is None:
        orientation = "vertical" if location in {"left", "right"} else "horizontal"
    return location, orientation


def _create_hcb_gridspec_axes(
    parent_ax: Axes,
    hist_fraction: float,
    location: Literal["left", "right", "top", "bottom"],
) -> tuple[Axes, Axes]:
    """Create gridspec axes for the colorbar and histogram using SubplotSpec.

    This method is used when parent_ax has a SubplotSpec (i.e., it's a subplot).
    It creates child axes using subgridspec for relative positioning.

    Parameters
    ----------
    parent_ax : Axes
        The container axes that will hold the colorbar and histogram.
    hist_fraction : float
        Fraction of space allocated to histogram.
    location : str
        Location of the colorbar.

    Returns
    -------
    tuple[Axes, Axes]
        The colorbar and histogram axes.

    """
    location, orientation = _determine_location_orientation(location, None)

    # Get the figure
    fig = parent_ax.get_figure()

    # Get parent's SubplotSpec
    parent_subplotspec = parent_ax.get_subplotspec()

    # Use SubplotSpec.subgridspec for relative positioning
    # This ensures the child axes follow parent_ax automatically
    if orientation == "vertical":
        # For vertical colorbar, split horizontally based on hist_fraction
        if location == "left":
            # Colorbar on left, histogram on right
            axes_ratios = [1 - hist_fraction, hist_fraction]
            subgs = parent_subplotspec.subgridspec(
                1, 2, width_ratios=axes_ratios, wspace=0
            )
            ax_cbar = fig.add_subplot(subgs[0, 0])
            ax_hist = fig.add_subplot(subgs[0, 1])
            hide_axis_elements(ax_cbar)
            ax_cbar.tick_params(left=True, labelleft=True)
        else:  # right
            # Histogram on left, colorbar on right
            axes_ratios = [hist_fraction, 1 - hist_fraction]
            subgs = parent_subplotspec.subgridspec(
                1, 2, width_ratios=axes_ratios, wspace=0
            )
            ax_hist = fig.add_subplot(subgs[0, 0])
            ax_cbar = fig.add_subplot(subgs[0, 1])
            hide_axis_elements(ax_cbar)
            ax_cbar.tick_params(right=True, labelright=True)

        hide_axis_elements(ax_hist)
        ax_hist.tick_params(labelbottom=True)
    else:
        # Horizontal colorbar: split vertically based on hist_fraction
        if location == "bottom":
            # Histogram on top, colorbar on bottom
            axes_ratios = [hist_fraction, 1 - hist_fraction]
            subgs = parent_subplotspec.subgridspec(
                2, 1, height_ratios=axes_ratios, hspace=0
            )
            ax_hist = fig.add_subplot(subgs[0, 0])
            ax_cbar = fig.add_subplot(subgs[1, 0])
            hide_axis_elements(ax_cbar)
            ax_cbar.tick_params(bottom=True, labelbottom=True)
        else:  # top
            # Colorbar on top, histogram on bottom
            axes_ratios = [1 - hist_fraction, hist_fraction]
            subgs = parent_subplotspec.subgridspec(
                2, 1, height_ratios=axes_ratios, hspace=0
            )
            ax_cbar = fig.add_subplot(subgs[0, 0])
            ax_hist = fig.add_subplot(subgs[1, 0])
            hide_axis_elements(ax_cbar)
            ax_cbar.tick_params(top=True, labeltop=True)

        hide_axis_elements(ax_hist)
        ax_hist.tick_params(labelleft=True)

    return ax_cbar, ax_hist


def _create_hcb_inset_axes(
    parent_ax: Axes,
    hist_fraction: float,
    location: Literal["left", "right", "top", "bottom"],
) -> tuple[Axes, Axes]:
    """Create colorbar and histogram axes using inset_axes for relative positioning.

    This method is used when parent_ax doesn't have a SubplotSpec (e.g., under
    constrained_layout). It uses inset_axes with parent_ax.transAxes to ensure
    the child axes follow parent_ax automatically.

    Parameters
    ----------
    parent_ax : Axes
        The container axes that will hold the colorbar and histogram.
    hist_fraction : float
        Fraction of space allocated to histogram.
    location : str
        Location of the colorbar.

    Returns
    -------
    tuple[Axes, Axes]
        The colorbar and histogram axes.

    """
    location, orientation = _determine_location_orientation(location, None)

    if orientation == "vertical":
        # Vertical colorbar: split horizontally
        if location == "left":
            # Colorbar on left, histogram on right
            cbar_x0 = 0
            cbar_width = 1 - hist_fraction
            hist_x0 = cbar_width
            hist_width = hist_fraction
        else:  # location == "right"
            # Histogram on left, colorbar on right
            hist_x0 = 0
            hist_width = hist_fraction
            cbar_x0 = hist_width
            cbar_width = 1 - hist_fraction

        # Fill entire height
        cbar_y0 = 0
        cbar_height = 1
        hist_y0 = 0
        hist_height = 1

        # Create axes using inset_axes (relative to parent_ax)
        # Use width="100%", height="100%" to fill the bbox_to_anchor area
        ax_cbar = inset_axes(
            parent_ax,
            width="100%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(cbar_x0, cbar_y0, cbar_width, cbar_height),
            bbox_transform=parent_ax.transAxes,
            borderpad=0,
        )
        ax_hist = inset_axes(
            parent_ax,
            width="100%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(hist_x0, hist_y0, hist_width, hist_height),
            bbox_transform=parent_ax.transAxes,
            borderpad=0,
        )

        hide_axis_elements(ax_cbar)
        hide_axis_elements(ax_hist)

        if location == "left":
            ax_cbar.tick_params(left=True, labelleft=True)
        else:
            ax_cbar.tick_params(right=True, labelright=True)

        ax_hist.tick_params(labelbottom=True)

    else:
        # Horizontal colorbar: split vertically
        if location == "bottom":
            # Colorbar on top, histogram on bottom
            cbar_y0 = 0
            cbar_height = 1 - hist_fraction
            hist_y0 = cbar_height
            hist_height = hist_fraction
        else:  # location == "top"
            # Histogram on top, colorbar on bottom
            hist_y0 = 0
            hist_height = hist_fraction
            cbar_y0 = hist_height
            cbar_height = 1 - hist_fraction

        # Fill entire width
        cbar_x0 = 0
        cbar_width = 1
        hist_x0 = 0
        hist_width = 1

        # Create axes using inset_axes
        # Use width="100%", height="100%" to fill the bbox_to_anchor area
        ax_cbar = inset_axes(
            parent_ax,
            width="100%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(cbar_x0, cbar_y0, cbar_width, cbar_height),
            bbox_transform=parent_ax.transAxes,
            borderpad=0,
        )
        ax_hist = inset_axes(
            parent_ax,
            width="100%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(hist_x0, hist_y0, hist_width, hist_height),
            bbox_transform=parent_ax.transAxes,
            borderpad=0,
        )

        hide_axis_elements(ax_cbar)
        hide_axis_elements(ax_hist)

        if location == "bottom":
            ax_cbar.tick_params(bottom=True, labelbottom=True)
        else:
            ax_cbar.tick_params(top=True, labeltop=True)

        ax_hist.tick_params(labelleft=True)

    return ax_cbar, ax_hist
