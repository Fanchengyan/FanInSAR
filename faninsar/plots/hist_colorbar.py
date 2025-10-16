"""
Histogram-embedded Colorbar for Matplotlib

This module provides a customizable colorbar with an integrated histogram
showing the distribution of data values across the color gradient. This is
inspired by the colorbar in the EOMaps plotting package:
https://github.com/raphaelquast/EOmaps
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import matplotlib.colorbar as cbar
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from typing_extensions import Iterable, Literal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure, SubFigure
    from numpy.typing import ArrayLike


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
    aspect : float, default 3
        Ratio of long to short dimensions of colorbar.
    fraction : float, default 0.25
        Fraction of original Axes to use for colorbar.
    hist_fraction : float, default 0.9
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
        uses a gray dashed line (e.g. `{"color": "0.35", "linestyle": (0, (5, 5)), "linewidth": 1}`).
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

    def __init__(
        self,
        data: ArrayLike,
        mappable: ScalarMappable | None = None,
        cax: Axes | None = None,
        ax: Axes | NDArray | Iterable[Axes] | None = None,
        use_gridspec: bool = True,
        location: Literal["left", "right", "top", "bottom"] | None = None,
        orientation: Literal["vertical", "horizontal"] | None = None,
        aspect: float = 3,
        fraction: float = 0.25,
        hist_fraction: float = 0.9,
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
        self.data = np.asanyarray(data).flatten()
        self.location, self.orientation = self._determine_location_orientation(
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
            self.cmap = mappable.get_cmap()
            self.norm = mappable.norm
        else:
            if cmap is None:
                raise ValueError("Either mappable or cmap must be provided")
            self.cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

            # Create normalization
            if norm is None:
                data_finite = data[np.isfinite(data)]
                vmin, vmax = data_finite.min(), data_finite.max()
                self.norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                self.norm = norm

        # Auto-detect histogram bins from norm if 'auto' (AFTER norm is set)
        self.hist_bins = self._determine_hist_bins(hist_bins)

        # parse cax and ax
        if ax is None:
            ax = getattr(mappable, "axes", None)
        cax, kwargs = self._create_hcb_axes(cax, ax, use_gridspec)
        self.fig = cax.get_figure(root=False)
        self.fig.stale = True

        # Create axes for colorbar and histogram
        self.ax_cbar, self.ax_hist = self._create_hist_and_cbar_axes(cax)

        # Draw colorbar and histogram
        NON_COLORBAR_KEYS = [  # remove kws that cannot be passed to Colorbar
            "fraction",
            "pad",
            "shrink",
            "aspect",
            "anchor",
            "panchor",
        ]
        self._draw_colorbar(**{
            k: v for k, v in kwargs.items() if k not in NON_COLORBAR_KEYS
        })
        self._draw_histogram()

        # Apply customizations
        self._apply_default_customizations()

    @property
    def hist_orientation(self) -> Literal["vertical", "horizontal"]:
        """Orientation of the histogram."""
        return "vertical" if self.orientation == "horizontal" else "horizontal"

    def _determine_location_orientation(
        self,
        location: Literal["left", "right", "top", "bottom"] | None,
        orientation: Literal["vertical", "horizontal"] | None,
    ) -> tuple[
        Literal["left", "right", "top", "bottom"], Literal["vertical", "horizontal"]
    ]:
        """Determine the location of the colorbar and histogram."""
        # validate location and orientation
        if location in ["left", "right"] and orientation == "horizontal":
            msg = "Horizontal colorbar cannot be on left or right side."
            raise ValueError(msg)
        if location in ["top", "bottom"] and orientation == "vertical":
            msg = "Vertical colorbar cannot be on top or bottom side."
            raise ValueError(msg)

        # default location and orientation
        if location is None and orientation is None:
            return "right", "vertical"

        # determine location and orientation from each other
        if location is None:
            location = "right" if orientation == "vertical" else "bottom"
        if orientation is None:
            orientation = "vertical" if location in ["left", "right"] else "horizontal"
        return location, orientation

    def _determine_hist_bins(
        self, hist_bins: int | NDArray | Literal["auto"]
    ) -> int | NDArray:
        """
        Determine the number of histogram bins.

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
            # Check if norm is BoundaryNorm (discrete levels)
            if hasattr(self.norm, "boundaries"):
                # BoundaryNorm has discrete levels
                return len(self.norm.boundaries) - 1
            else:
                # Continuous colormap, use fine binning
                return 100
        else:
            return hist_bins

    def _trim_hist_axes(self):
        """trim histogram axes to remove extra space caused by extend triangles."""
        if self.extend != "neither":
            divider = make_axes_locatable(self.ax_hist)
            kwargs = {"size": f"{self.extendfrac * 100}%", "pad": 0}
            if self.orientation == "vertical":
                if self.extend in ["min", "both"]:
                    ax = divider.append_axes("bottom", **kwargs)
                    hide_axis_elements(ax)
                    ax.set_zorder(-1)  # send to back
                if self.extend in ["max", "both"]:
                    ax = divider.append_axes("top", **kwargs)
                    hide_axis_elements(ax)
                    ax.set_zorder(-1)  # send to back
            else:
                if self.extend in ["min", "both"]:
                    ax = divider.append_axes("left", **kwargs)
                    hide_axis_elements(ax)
                    ax.set_zorder(-1)  # send to back
                if self.extend in ["max", "both"]:
                    ax = divider.append_axes("right", **kwargs)
                    hide_axis_elements(ax)
                    ax.set_zorder(-1)  # send to back

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
            location=self.location,
            orientation=None,  # location is enough
            cmap=self.cmap,
            norm=self.norm,
            aspect=self.aspect,
            fraction=self.fraction,
            pad=self.pad,
            alpha=self.alpha,
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
        # Get position of parent axes
        pos = parent_ax.get_position()
        x0, y0, width, height = pos.x0, pos.y0, pos.width, pos.height
        parent_ax.remove()

        # Apply shrink factor
        if self.orientation == "vertical":
            # Shrink height and center vertically
            new_height = height * self.shrink
            y0 = y0 + (height - new_height) / 2
            height = new_height
        else:
            # Shrink width and center horizontally
            new_width = width * self.shrink
            x0 = x0 + (width - new_width) / 2
            width = new_width

        x1, y1 = x0 + width, y0 + height

        gs_kwargs = {
            "figure": self.fig,
            "left": x0,
            "right": x1,
            "bottom": y0,
            "top": y1,
            "wspace": 0,
            "hspace": 0,
        }
        # Calculate dimensions based on orientation
        if self.orientation == "vertical":
            if self.location == "left":
                # colorbar on left, histogram on right
                ratios = [1 - self.hist_fraction, self.hist_fraction]
                gs = gridspec.GridSpec(1, 2, width_ratios=ratios, **gs_kwargs)
                ax_cbar = self.fig.add_subplot(gs[0])
                ax_hist = self.fig.add_subplot(gs[1])
                hide_axis_elements(ax_cbar)
                ax_cbar.tick_params(left=True, labelleft=True)
            else:
                # histogram on left, colorbar on right
                ratios = [self.hist_fraction, 1 - self.hist_fraction]
                gs = gridspec.GridSpec(1, 2, width_ratios=ratios, **gs_kwargs)
                ax_hist = self.fig.add_subplot(gs[0])
                ax_cbar = self.fig.add_subplot(gs[1])
                hide_axis_elements(ax_cbar)
                ax_cbar.tick_params(right=True, labelright=True)
            hide_axis_elements(ax_hist)
            ax_hist.tick_params(labelbottom=True)
        else:  # horizontal
            if self.location == "bottom":
                # colorbar on bottom, histogram on top
                ratios = [self.hist_fraction, 1 - self.hist_fraction]
                gs = gridspec.GridSpec(2, 1, height_ratios=ratios, **gs_kwargs)
                ax_hist = self.fig.add_subplot(gs[0])
                ax_cbar = self.fig.add_subplot(gs[1])
                hide_axis_elements(ax_cbar)
                ax_cbar.tick_params(bottom=True, labelbottom=True)
            else:
                # histogram on bottom, colorbar on top
                ratios = [1 - self.hist_fraction, self.hist_fraction]
                gs = gridspec.GridSpec(2, 1, height_ratios=ratios, **gs_kwargs)
                ax_cbar = self.fig.add_subplot(gs[0])
                ax_hist = self.fig.add_subplot(gs[1])
                hide_axis_elements(ax_cbar)
                ax_cbar.tick_params(top=True, labeltop=True)
            hide_axis_elements(ax_hist)
            ax_hist.tick_params(labelleft=True)

        ax_cbar.set_zorder(10)
        ax_hist.set_zorder(11)  # set histogram on top of colorbar for line visibility

        # Join axes - histogram shares the data axis with colorbar
        # This ensures they stay aligned even when limits change
        if self.orientation == "horizontal":
            ax_hist.sharex(ax_cbar)
        else:
            ax_hist.sharey(ax_cbar)

        return ax_cbar, ax_hist

    def _draw_colorbar(self, **kwargs) -> None:
        """Draw the colorbar."""
        self.cbar = Colorbar(self.ax_cbar, **kwargs)
        hide_spines(self.ax_cbar)

    def _draw_histogram(self) -> None:
        """Draw the histogram using matplotlib's hist and recolor patches."""
        # Remove NaN and infinite values
        data_finite = self.data[np.isfinite(self.data)]

        if len(data_finite) == 0:
            msg = (
                "No finite values in data. Unable to draw histogram. "
                "Check your data for NaNs and infinities."
            )
            warnings.warn(msg)
            return

        # Trim histogram axes to remove extra space caused by extend triangles.
        self._trim_hist_axes()

        # Use matplotlib's hist to create histogram
        hist_kwargs = {
            "bins": self.hist_bins,
            "range": (self.norm.vmin, self.norm.vmax),
            "align": "mid",
        }
        self.hist_kwargs.update(hist_kwargs)

        if self.hist_orientation == "horizontal":
            # Vertical: histogram bars go horizontal
            h = self.ax_hist.hist(
                data_finite, orientation="horizontal", **self.hist_kwargs
            )
        else:
            # Horizontal: histogram bars go vertical
            h = self.ax_hist.hist(
                data_finite, orientation="vertical", **self.hist_kwargs
            )

        # Get color split positions from colormap
        bins = getattr(self.norm, "boundaries", None)
        if bins is None:
            # For continuous colormaps
            if isinstance(self.cmap, LinearSegmentedColormap):
                splitpos = np.linspace(self.norm.vmin, self.norm.vmax, self.cmap.N)
            else:
                splitpos = np.linspace(self.norm.vmin, self.norm.vmax, self.cmap.N + 1)
        else:
            # For discrete colormaps (BoundaryNorm)
            splitpos = np.asanyarray(bins)

        # Recolor histogram patches to match colorbar
        self._recolor_histogram_patches(splitpos)

        self._draw_hist_grid()

        self.set_scale("log" if self.log else "linear")

        if self.hist_orientation == "horizontal":
            # Invert x-axis for mirroring
            xlim = self.ax_hist.get_xlim()
            self.ax_hist.set_xlim(xlim[1], self.min_count)
            if self.location == "left":
                self.ax_hist.invert_xaxis()
        else:
            # Invert y-axis for mirroring
            ylim = self.ax_hist.get_ylim()
            self.ax_hist.set_ylim(ylim[1], self.min_count)
            if self.location == "bottom":
                self.ax_hist.invert_yaxis()

    def _draw_hist_grid(self) -> None:
        """Draw grid on histogram axis."""
        if self.hist_orientation == "horizontal":
            # remove 0 from ticks of grid
            self.ax_hist.grid(axis="x", which="major", **self.divider_style)

            if self.location == "left":
                spine = self.ax_hist.spines["left"]
            else:
                spine = self.ax_hist.spines["right"]
        else:
            self.ax_hist.grid(axis="y", which="major", **self.divider_style)
            if self.location == "bottom":
                spine = self.ax_hist.spines["bottom"]
            else:
                spine = self.ax_hist.spines["top"]
        # Make the spine visible and apply style in the intersection of the spines
        spine.set_visible(True)
        plt.setp(spine, **self.divider_style)

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
            raise ValueError(msg)

    def _adjust_limits_for_extends(self) -> None:
        """Adjust axis limits to include extension triangles.

        When extend is set, the colorbar draws extension triangles outside
        the data range. We need to adjust the histogram axis limits to match
        the full range including these extensions.
        """
        if not hasattr(self.cbar, "extend") or self.cbar.extend == "neither":
            return

        # Calculate the extension length
        vmin = float(self.norm.vmin) if self.norm.vmin is not None else 0.0
        vmax = float(self.norm.vmax) if self.norm.vmax is not None else 1.0
        data_range = vmax - vmin

        # Get extendfrac (default to 0.05 if not set or invalid)
        extendfrac_raw = self.cbar.extendfrac
        extendfrac: float
        if extendfrac_raw is None or extendfrac_raw == "auto":
            extendfrac = 0.05
        elif isinstance(extendfrac_raw, (list, tuple)):
            # If it's a sequence, use the first value
            extendfrac = float(extendfrac_raw[0])
        else:
            extendfrac = float(extendfrac_raw)  # type: ignore[arg-type]

        extend_length = data_range * extendfrac

        # Calculate new limits based on extend direction
        if self.cbar.extend in ("min", "both"):
            new_min = vmin - extend_length
        else:
            new_min = vmin

        if self.cbar.extend in ("max", "both"):
            new_max = vmax + extend_length
        else:
            new_max = vmax

        # Set the new limits
        if self.orientation == "horizontal":
            self.ax_cbar.set_xlim(new_min, new_max)
            # ax_hist shares x-axis, so it will update automatically
        else:
            self.ax_cbar.set_ylim(new_min, new_max)
            # ax_hist shares y-axis, so it will update automatically

    def _recolor_histogram_patches(self, splitpos: NDArray) -> None:
        """Recolor histogram patches to match colorbar colors.

        This method splits histogram bars that span multiple colors in the colorbar
        and assigns the correct color to each segment.

        Parameters
        ----------
        splitpos : NDArray
            Positions where colors change in the colorbar.
        """

        # Iterate over all patches (histogram bars)
        for patch in list(self.ax_hist.patches):
            patch = cast(Rectangle, patch)
            if self.orientation == "vertical":
                # For vertical orientation, bars are horizontal
                minval = np.atleast_1d(patch.get_y())[0]
                width = patch.get_width()
                height = patch.get_height()
                maxval = minval + height
            else:
                # For horizontal orientation, bars are vertical
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
                    # Use colormap color - sample at the center of the segment
                    # Normalize the value to [0, 1] range for colormap
                    center_val = (b0 + b1) / 2
                    normalized_val = self.norm(center_val)
                    color = self.cmap(normalized_val)

                    if self.orientation == "vertical":
                        # Horizontal bars
                        pi = Rectangle(
                            (0, b0),
                            width,
                            (b1 - b0),
                            facecolor=color,
                            linewidth=0,
                            alpha=self.alpha,
                            # clip_on=False,
                        )
                    else:
                        # Vertical bars
                        pi = Rectangle(
                            (b0, 0),
                            (b1 - b0),
                            height,
                            facecolor=color,
                            linewidth=0,
                            alpha=self.alpha,
                            # clip_on=False,
                        )

                    self.ax_hist.add_patch(pi)
            else:  # Bar is within a single color
                # Use colormap color - sample at the center of the bar
                center_val = (minval + maxval) / 2
                normalized_val = self.norm(center_val)
                color = self.cmap(normalized_val)

                patch.set_facecolor(color)
                patch.set_alpha(self.alpha)
                patch.set_linewidth(0)
                # patch.set_clip_on(False)

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

        # set tick formatting of colorbar axis
        self.set_cbar_locator(AutoLocator())
        self.set_cbar_formatter(ScalarFormatter())

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
        reset : bool, default: False
            Whether to reset the ticks to defaults before updating them.

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
        which_ax : {'cbar', 'hist'}, default: 'cbar'
            The axes to which the parameters are applied.
        reset : bool, default: False
            Whether to reset the ticks to defaults before updating them.

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
        """
        Set labels for the colorbar.

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

    def set_cbar_ticks(
        self,
        ticks: ArrayLike,
        labels: Iterable[str] | None = None,
        *,
        minor=False,
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
        minor=False,
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

    def minorticks_on(self):
        """Turn on colorbar minor ticks."""
        self.cbar.minorticks_on()

    def minorticks_off(self):
        """Turn the minor ticks of the colorbar off."""
        self.cbar.minorticks_off()


def _hist_colorbar(
    self: Figure| SubFigure,
    data: ArrayLike,
    mappable: ScalarMappable | None = None,
    cax: Axes | None = None,
    ax: Axes | NDArray | Iterable[Axes] | None = None,
    use_gridspec: bool = True,
    location: Literal["left", "right", "top", "bottom"] | None = None,
    orientation: Literal["vertical", "horizontal"] | None = None,
    aspect: float = 3,
    fraction: float = 0.25,
    hist_fraction: float = 0.9,
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
    aspect : float, default 3
        Ratio of long to short dimensions of colorbar.
    fraction : float, default 0.25
        Fraction of original Axes to use for colorbar.
    hist_fraction : float, default 0.9
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
        uses a gray dashed line (e.g. `{"color": "0.35", "linestyle": (0, (5, 5)), "linewidth": 1}`).
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
    >>> from hist_colorbar import hist_colorbar
    >>> from matplotlib.colors import Normalize
    >>>
    >>> # Generate sample data
    >>> data = np.random.randn(1000)
    >>>
    >>> # Create a simple plot
    >>> fig, ax = plt.subplots()
    >>> norm = Normalize(vmin=-3, vmax=3)
    >>> hcb = hist_colorbar(data=data, cmap='viridis', norm=norm,
    ...                     ax=ax, label='Value')
    >>> plt.show()

    Using with a mappable object from imshow:

    >>> data_2d = np.random.randn(50, 50)
    >>> fig, ax = plt.subplots()
    >>> im = ax.imshow(data_2d, cmap='coolwarm')
    >>> hcb = hist_colorbar(data=data_2d.flatten(), mappable=im)
    >>> plt.show()

    Horizontal orientation with custom histogram bins:

    >>> data = np.random.randn(5000)
    >>> fig, ax = plt.subplots()
    >>> hcb = hist_colorbar(data=data, cmap='plasma',
    ...                     orientation='horizontal',
    ...                     hist_bins=50, ax=ax)
    >>> plt.show()

    Using logarithmic scale for histogram counts:

    >>> data = np.random.exponential(2, 10000)
    >>> fig, ax = plt.subplots()
    >>> hcb = hist_colorbar(data=data, cmap='inferno',
    ...                     log=True, ax=ax,
    ...                     label='Intensity',
    ...                     hist_label='Count')
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
