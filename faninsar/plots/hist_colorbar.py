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
from matplotlib import colors
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.layout_engine import LayoutEngine
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
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing_extensions import Literal

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.transforms import Bbox
    from numpy.typing import ArrayLike, NDArray

NON_COLORBAR_KEYS = [  # remove kws that cannot be passed to Colorbar
    "fraction",
    "pad",
    "shrink",
    "aspect",
    "anchor",
    "panchor",
]


class HistColorbarLayoutEngine(LayoutEngine):
    """Custom layout engine for HistColorbar compatibility with constrained_layout.

    This layout engine works in conjunction with matplotlib's constrained_layout
    to properly position HistColorbar axes after the main layout has been computed.
    It ensures that histogram and colorbar axes maintain correct positions even
    when the parent figure uses constrained_layout.

    The engine is automatically registered when a HistColorbar is created in a
    figure with constrained_layout enabled.
    """

    def __init__(self) -> None:
        """Initialize the layout engine."""
        super().__init__()
        # Track all HistColorbar instances in this figure
        self._hist_colorbars: list[HistColorbar] = []

    def execute(self, fig: Figure) -> None:
        """Execute layout adjustments for all HistColorbar instances.

        This method is called by matplotlib during the layout phase,
        after constrained_layout has computed positions for regular axes.

        Parameters
        ----------
        fig : Figure
            The figure to layout.

        """
        # First, execute the underlying constrained layout if available
        parent_engine = getattr(fig, "_original_layout_engine", None)
        if parent_engine is not None:
            parent_engine.execute(fig)

        # Then adjust HistColorbar positions
        for hcb in self._hist_colorbars:
            if fig in {hcb.fig, hcb.fig.figure}:
                hcb._update_positions()

    def set(self, **kwargs) -> None:
        """Set layout engine parameters."""
        # Delegate to parent engine if available

    @property
    def colorbar_gridspec(self) -> bool:
        """Whether to use gridspec for colorbars.

        Returns False to force matplotlib to use make_axes instead of
        make_axes_gridspec, which has better constrained_layout compatibility.
        """
        return False

    @property
    def adjust_compatible(self) -> bool:
        """Whether this layout engine is compatible with layout adjustments.

        Returns True to allow matplotlib to temporarily switch layout engines
        during operations like savefig.
        """
        return True

    def register_hist_colorbar(self, hcb: HistColorbar) -> None:
        """Register a HistColorbar instance for layout management.

        Parameters
        ----------
        hcb : HistColorbar
            The HistColorbar instance to manage.

        """
        if hcb not in self._hist_colorbars:
            self._hist_colorbars.append(hcb)

    def unregister_hist_colorbar(self, hcb: HistColorbar) -> None:
        """Unregister a HistColorbar instance.

        Parameters
        ----------
        hcb : HistColorbar
            The HistColorbar instance to remove.

        """
        if hcb in self._hist_colorbars:
            self._hist_colorbars.remove(hcb)


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
    _draw_callback_id: int | None
    _position_cache: dict[str, Any]

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
        """Initialize the HistColorbar object."""
        # Save current axes to restore later
        current_ax = plt.gca() if plt.get_fignums() else None

        # Initialize internal state variables
        self._has_extend = False
        self._parent_ax = None
        self._draw_callback_id = None
        self._position_cache = {}

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

        # Setup layout integration for constrained_layout compatibility
        self._setup_layout_integration()

        # Restore original current axes
        if current_ax is not None and current_ax in self.fig.axes:
            plt.sca(current_ax)

    @property
    def hist_orientation(self) -> Literal["vertical", "horizontal"]:
        """Orientation of the histogram."""
        return "vertical" if self.orientation == "horizontal" else "horizontal"

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
            # Check if norm is BoundaryNorm (discrete levels)
            if hasattr(self.norm, "boundaries"):
                # BoundaryNorm has discrete levels
                return len(self.norm.boundaries) - 1
            # Continuous colormap, use fine binning
            return 100
        return hist_bins

    def _trim_hist_axes(self) -> None:
        """Adjust histogram axes to account for extend triangles.

        Instead of using AxesDivider (which conflicts with constrained_layout),
        we adjust the histogram's position to leave space for extend triangles.
        This is handled by reducing the histogram's data limits rather than
        creating additional axes.
        """
        if self.extend == "neither":
            return

        # Store extend information for later position updates
        self._has_extend = True

        # Calculate extend fraction relative to the axis position
        extend_frac = self.extendfrac

        # Adjust histogram position by shrinking it to leave space for extends
        # This will be applied in _update_positions() method
        pos = self.ax_hist.get_position()

        if self.orientation == "vertical":
            # Vertical: extends at top/bottom
            total_height = pos.height

            if self.extend == "both":
                # Shrink from both sides
                new_height = total_height * (1 - 2 * extend_frac)
                new_y0 = pos.y0 + total_height * extend_frac
                self.ax_hist.set_position((pos.x0, new_y0, pos.width, new_height))
            elif self.extend == "min":
                # Shrink from bottom
                new_height = total_height * (1 - extend_frac)
                new_y0 = pos.y0 + total_height * extend_frac
                self.ax_hist.set_position((pos.x0, new_y0, pos.width, new_height))
            elif self.extend == "max":
                # Shrink from top
                new_height = total_height * (1 - extend_frac)
                self.ax_hist.set_position((pos.x0, pos.y0, pos.width, new_height))
        else:
            # Horizontal: extends at left/right
            total_width = pos.width

            if self.extend == "both":
                # Shrink from both sides
                new_width = total_width * (1 - 2 * extend_frac)
                new_x0 = pos.x0 + total_width * extend_frac
                self.ax_hist.set_position((new_x0, pos.y0, new_width, pos.height))
            elif self.extend == "min":
                # Shrink from left
                new_width = total_width * (1 - extend_frac)
                new_x0 = pos.x0 + total_width * extend_frac
                self.ax_hist.set_position((new_x0, pos.y0, new_width, pos.height))
            elif self.extend == "max":
                # Shrink from right
                new_width = total_width * (1 - extend_frac)
                self.ax_hist.set_position((pos.x0, pos.y0, new_width, pos.height))

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
                parent_ax, self.shrink, self.hist_fraction, self.location, self.pad
            )
        else:
            # Use inset_axes (for constrained_layout or no layout)
            ax_cbar, ax_hist = _create_hcb_inset_axes(
                parent_ax, self.shrink, self.hist_fraction, self.location, self.pad
            )

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
            warnings.warn(msg, stacklevel=2)
            logger.warning(msg)
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
            self.ax_hist.hist(data_finite, orientation="horizontal", **self.hist_kwargs)
        else:
            # Horizontal: histogram bars go vertical
            self.ax_hist.hist(data_finite, orientation="vertical", **self.hist_kwargs)

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
            patch = cast("Rectangle", patch)
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

    def minorticks_on(self) -> None:
        """Turn on colorbar minor ticks."""
        self.cbar.minorticks_on()

    def minorticks_off(self) -> None:
        """Turn the minor ticks of the colorbar off."""
        self.cbar.minorticks_off()

    def _update_positions(self) -> None:
        """Update positions of colorbar and histogram axes.

        This method is called during draw events when using constrained_layout
        to ensure the histogram and colorbar maintain correct positions after
        the layout engine has adjusted axes positions.
        """
        if not hasattr(self, "ax") or self.ax is None or not self.ax.get_visible():
            return

        # Get the container axes position (updated by constrained_layout)
        pos = self.ax.get_position()

        # Check if position has changed since last update (avoid unnecessary work)
        cache_key = f"{pos.x0:.6f},{pos.y0:.6f},{pos.width:.6f},{pos.height:.6f}"
        if self._position_cache.get("container_pos") == cache_key:
            return
        self._position_cache["container_pos"] = cache_key

        # Calculate hcb position within container
        if self.orientation == "vertical":
            if self.location == "left":
                pos_hcb, _ = pos.splitx(1 - self.pad)
            else:
                _, pos_hcb = pos.splitx(self.pad)
        elif self.location == "bottom":
            pos_hcb, _ = pos.splity(1 - self.pad)
        else:
            _, pos_hcb = pos.splity(self.pad)

        # Recalculate and apply positions for colorbar and histogram
        self._apply_hcb_positions(pos_hcb)

        # Reapply extend adjustments if needed
        if getattr(self, "_has_extend", False):
            self._trim_hist_axes()

    def _apply_hcb_positions(self, pos_hcb: Bbox) -> None:
        """Apply calculated positions to colorbar and histogram axes.

        Parameters
        ----------
        pos_hcb : Bbox
            The bounding box for the histogram-colorbar container.

        """
        # Calculate space for shrink
        space = max((1 - self.shrink) / 2, 1e-6)

        if self.orientation == "vertical":
            # Calculate vertical positions with shrink
            total_height = pos_hcb.height
            shrunk_height = total_height * self.shrink
            y_start = pos_hcb.y0 + space * total_height

            # Split width between histogram and colorbar
            if self.location == "left":
                # Colorbar on left, histogram on right
                cbar_width = pos_hcb.width * (1 - self.hist_fraction)
                hist_width = pos_hcb.width * self.hist_fraction
                self.ax_cbar.set_position(
                    (
                        pos_hcb.x0,
                        y_start,
                        cbar_width,
                        shrunk_height,
                    )
                )
                self.ax_hist.set_position(
                    (
                        pos_hcb.x0 + cbar_width,
                        y_start,
                        hist_width,
                        shrunk_height,
                    )
                )
            else:
                # Histogram on left, colorbar on right
                hist_width = pos_hcb.width * self.hist_fraction
                cbar_width = pos_hcb.width * (1 - self.hist_fraction)
                self.ax_hist.set_position(
                    (
                        pos_hcb.x0,
                        y_start,
                        hist_width,
                        shrunk_height,
                    )
                )
                self.ax_cbar.set_position(
                    (
                        pos_hcb.x0 + hist_width,
                        y_start,
                        cbar_width,
                        shrunk_height,
                    )
                )
        else:
            # Calculate horizontal positions with shrink
            total_width = pos_hcb.width
            shrunk_width = total_width * self.shrink
            x_start = pos_hcb.x0 + space * total_width

            # Split height between histogram and colorbar
            if self.location == "bottom":
                # Histogram on top, colorbar on bottom
                hist_height = pos_hcb.height * self.hist_fraction
                cbar_height = pos_hcb.height * (1 - self.hist_fraction)
                self.ax_hist.set_position(
                    (
                        x_start,
                        pos_hcb.y0 + cbar_height,
                        shrunk_width,
                        hist_height,
                    )
                )
                self.ax_cbar.set_position(
                    (
                        x_start,
                        pos_hcb.y0,
                        shrunk_width,
                        cbar_height,
                    )
                )
            else:
                # Colorbar on top, histogram on bottom
                cbar_height = pos_hcb.height * (1 - self.hist_fraction)
                hist_height = pos_hcb.height * self.hist_fraction
                self.ax_cbar.set_position(
                    (
                        x_start,
                        pos_hcb.y0 + hist_height,
                        shrunk_width,
                        cbar_height,
                    )
                )
                self.ax_hist.set_position(
                    (
                        x_start,
                        pos_hcb.y0,
                        shrunk_width,
                        hist_height,
                    )
                )

    def _on_draw(self, event) -> None:  # noqa: ANN001
        """Call back for draw events to update positions when needed.

        Parameters
        ----------
        event : DrawEvent
            The draw event from matplotlib.

        """
        # Disabled: Position updates on draw cause issues
        # Let constrained_layout handle positioning naturally

    def _setup_layout_integration(self) -> None:
        """Configure integration with figure's layout engine.

        This method configures the HistColorbar to work with constrained_layout
        by either registering with a custom layout engine or setting up draw callbacks.
        """
        layout_engine = self.fig.get_layout_engine()

        if layout_engine is None:
            # No layout engine, nothing to do
            logger.debug("No layout engine detected, using standard positioning.")
            return

        # Check if it's a constrained layout
        engine_type = str(type(layout_engine))
        if "constrained" not in engine_type.lower():
            # Not constrained layout, nothing special needed
            msg = f"Layout engine '{engine_type}' detected, no special handling needed."
            logger.debug(msg)
            return

        # Constrained layout detected - setup compatibility
        logger.info(
            "Constrained layout detected. HistColorbar will automatically adjust "
            "positions during rendering to maintain proper alignment."
        )

        # Setup for constrained layout compatibility
        if isinstance(layout_engine, HistColorbarLayoutEngine):
            # Already using our custom engine, just register
            logger.debug("Registering with existing HistColorbarLayoutEngine.")
            layout_engine.register_hist_colorbar(self)
        else:
            # Need to wrap the existing layout engine
            logger.debug(
                "Wrapping existing layout engine with HistColorbarLayoutEngine."
            )
            self._wrap_layout_engine()

        # Register draw callback as fallback/supplement
        if hasattr(self.fig.canvas, "mpl_connect"):
            self._draw_callback_id = self.fig.canvas.mpl_connect(
                "draw_event", self._on_draw
            )
            logger.debug("Draw event callback registered for position updates.")

    def _wrap_layout_engine(self) -> None:
        """Wrap the figure's existing layout engine with our custom one."""
        current_engine = self.fig.get_layout_engine()

        if not isinstance(current_engine, HistColorbarLayoutEngine):
            # Store the original engine
            self.fig._original_layout_engine = current_engine

            # Create and install our custom engine
            new_engine = HistColorbarLayoutEngine()
            new_engine.register_hist_colorbar(self)
            self.fig.set_layout_engine(new_engine)
        else:
            # Already wrapped, just register
            current_engine.register_hist_colorbar(self)

    def remove(self) -> None:
        """Remove the HistColorbar and clean up.

        This method should be called when removing a HistColorbar to properly
        unregister callbacks and layout engine references.
        """
        # Unregister from layout engine
        layout_engine = self.fig.get_layout_engine()
        if isinstance(layout_engine, HistColorbarLayoutEngine):
            layout_engine.unregister_hist_colorbar(self)

        # Disconnect draw callback
        if self._draw_callback_id is not None and hasattr(
            self.fig.canvas, "mpl_disconnect"
        ):
            self.fig.canvas.mpl_disconnect(self._draw_callback_id)
            self._draw_callback_id = None

        # Remove axes
        if hasattr(self, "ax_cbar") and self.ax_cbar is not None:
            self.ax_cbar.remove()
        if hasattr(self, "ax_hist") and self.ax_hist is not None:
            self.ax_hist.remove()


def _hist_colorbar(  # noqa: D417
    self: Figure | SubFigure,  # noqa: ARG001
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
    shrink: float,
    hist_fraction: float,
    location: Literal["left", "right", "top", "bottom"],
    pad: float,
) -> tuple[Axes, Axes]:
    """Create gridspec axes for the colorbar and histogram using SubplotSpec.

    This method is used when parent_ax has a SubplotSpec (i.e., it's a subplot).
    It creates child axes using subgridspec for relative positioning.

    Parameters
    ----------
    parent_ax : Axes
        The container axes that will hold the colorbar and histogram.
    shrink : float
        Fraction by which to shrink the colorbar.
    hist_fraction : float
        Fraction of space allocated to histogram.
    location : str
        Location of the colorbar.
    pad : float
        Padding between colorbar sections.

    Returns
    -------
    tuple[Axes, Axes]
        The colorbar and histogram axes.

    """
    location, orientation = _determine_location_orientation(location, None)

    # Calculate space for shrink
    space = max((1 - shrink) / 2, 1e-6)
    shrink_ratio = [space, shrink, space]

    # Get the figure
    fig = parent_ax.get_figure()

    # Get parent's SubplotSpec
    parent_subplotspec = parent_ax.get_subplotspec()

    # Use SubplotSpec.subgridspec for relative positioning
    # This ensures the child axes follow parent_ax automatically
    if orientation == "vertical":
        # For vertical colorbar, create 3 rows (for shrink) and 2 columns
        subgs = parent_subplotspec.subgridspec(
            3,
            2,
            width_ratios=[1 - pad, pad] if location == "left" else [pad, 1 - pad],
            height_ratios=shrink_ratio,
            wspace=0,
            hspace=0,
        )

        if location == "left":
            # Colorbar on left: subdivide left column
            cbar_col = 0
            axes_ratios = [1 - hist_fraction, hist_fraction]
            cbar_hist_gs = subgs[1, cbar_col].subgridspec(
                1, 2, width_ratios=axes_ratios, wspace=0
            )
            ax_cbar = fig.add_subplot(cbar_hist_gs[0, 0])
            ax_hist = fig.add_subplot(cbar_hist_gs[0, 1])
            hide_axis_elements(ax_cbar)
            ax_cbar.tick_params(left=True, labelleft=True)
        else:  # right
            # Colorbar on right: subdivide right column
            cbar_col = 1
            axes_ratios = [hist_fraction, 1 - hist_fraction]
            cbar_hist_gs = subgs[1, cbar_col].subgridspec(
                1, 2, width_ratios=axes_ratios, wspace=0
            )
            ax_hist = fig.add_subplot(cbar_hist_gs[0, 0])
            ax_cbar = fig.add_subplot(cbar_hist_gs[0, 1])
            hide_axis_elements(ax_cbar)
            ax_cbar.tick_params(right=True, labelright=True)

        hide_axis_elements(ax_hist)
        ax_hist.tick_params(labelbottom=True)
    else:
        # Horizontal colorbar: create 2 rows and 3 columns
        subgs = parent_subplotspec.subgridspec(
            2,
            3,
            width_ratios=shrink_ratio,
            height_ratios=[1 - pad, pad] if location == "top" else [pad, 1 - pad],
            wspace=0,
            hspace=0,
        )

        if location == "bottom":
            cbar_row = 0
            axes_ratios = [hist_fraction, 1 - hist_fraction]
            cbar_hist_gs = subgs[cbar_row, 1].subgridspec(
                2, 1, height_ratios=axes_ratios, hspace=0
            )
            ax_hist = fig.add_subplot(cbar_hist_gs[0, 0])
            ax_cbar = fig.add_subplot(cbar_hist_gs[1, 0])
            hide_axis_elements(ax_cbar)
            ax_cbar.tick_params(bottom=True, labelbottom=True)
        else:  # top
            cbar_row = 1
            axes_ratios = [1 - hist_fraction, hist_fraction]
            cbar_hist_gs = subgs[cbar_row, 1].subgridspec(
                2, 1, height_ratios=axes_ratios, hspace=0
            )
            ax_cbar = fig.add_subplot(cbar_hist_gs[0, 0])
            ax_hist = fig.add_subplot(cbar_hist_gs[1, 0])
            hide_axis_elements(ax_cbar)
            ax_cbar.tick_params(top=True, labeltop=True)

        hide_axis_elements(ax_hist)
        ax_hist.tick_params(labelleft=True)

    return ax_cbar, ax_hist


def _create_hcb_inset_axes(
    parent_ax: Axes,
    shrink: float,
    hist_fraction: float,
    location: Literal["left", "right", "top", "bottom"],
    pad: float,
) -> tuple[Axes, Axes]:
    """Create colorbar and histogram axes using inset_axes for relative positioning.

    This method is used when parent_ax doesn't have a SubplotSpec (e.g., under
    constrained_layout). It uses inset_axes with parent_ax.transAxes to ensure
    the child axes follow parent_ax automatically.

    Parameters
    ----------
    parent_ax : Axes
        The container axes that will hold the colorbar and histogram.
    shrink : float
        Fraction by which to shrink the colorbar.
    hist_fraction : float
        Fraction of space allocated to histogram.
    location : str
        Location of the colorbar.
    pad : float
        Padding between colorbar and parent axes.

    Returns
    -------
    tuple[Axes, Axes]
        The colorbar and histogram axes.

    """
    location, orientation = _determine_location_orientation(location, None)

    # Fallback: parent_ax is not a subplot (e.g., under constrained_layout)
    # Use inset_axes for relative positioning instead of absolute GridSpec
    # This ensures axes follow parent_ax automatically

    # Calculate bounds and width/height in axes coordinates (relative)
    # Format: [x0, y0, width, height] in axes fraction

    # Reserve extra space for tick labels
    # This prevents overlap with adjacent subplots and ensures labels are visible
    tick_label_space = 0.05  # 5% of parent_ax for tick labels

    if orientation == "vertical":
        # Vertical colorbar
        # Total available width after pad and tick label space
        if location == "left":
            # Reserve space on the left for tick labels
            available_width = 1 - pad - tick_label_space
            cbar_x0 = tick_label_space  # Start after tick label space
            cbar_width = available_width * (1 - hist_fraction)
            hist_width = available_width * hist_fraction
            hist_x0 = cbar_x0 + cbar_width  # Starts right after colorbar
        else:  # location == "right"
            # Reserve space on the right for tick labels
            available_width = 1 - pad - tick_label_space
            hist_x0 = pad
            hist_width = available_width * hist_fraction
            cbar_width = available_width * (1 - hist_fraction)
            cbar_x0 = hist_x0 + hist_width  # Starts right after histogram

        cbar_height = shrink
        hist_height = shrink
        cbar_y0 = (1 - shrink) / 2  # Center vertically
        hist_y0 = cbar_y0

        # Create axes using inset_axes (relative to parent_ax)
        # Use width="100%", height="100%" to fill the bbox_to_anchor area
        # loc=3 means lower left corner, which with 100% size fills the entire bbox
        ax_cbar = inset_axes(
            parent_ax,
            width="100%",
            height="100%",
            loc="lower left",  # lower left
            bbox_to_anchor=(cbar_x0, cbar_y0, cbar_width, cbar_height),
            bbox_transform=parent_ax.transAxes,
            borderpad=0,
        )
        ax_hist = inset_axes(
            parent_ax,
            width="100%",
            height="100%",
            loc="lower left",  # lower left
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
        # Horizontal colorbar
        # Total available height after pad and tick label space
        if location == "bottom":
            # Reserve space on the bottom for tick labels
            available_height = 1 - pad - tick_label_space
            hist_y0 = tick_label_space  # Start after tick label space
            hist_height = available_height * hist_fraction
            cbar_height = available_height * (1 - hist_fraction)
            cbar_y0 = hist_y0 + hist_height  # Starts right after histogram
        else:  # location == "top"
            # Reserve space on the top for tick labels
            available_height = 1 - pad - tick_label_space
            hist_y0 = pad
            hist_height = available_height * hist_fraction
            cbar_height = available_height * (1 - hist_fraction)
            cbar_y0 = hist_y0 + hist_height  # Starts right after histogram

        cbar_width = shrink
        hist_width = shrink
        cbar_x0 = (1 - shrink) / 2  # Center horizontally
        hist_x0 = cbar_x0

        # Create axes using inset_axes
        # Use width="100%", height="100%" to fill the bbox_to_anchor area
        ax_cbar = inset_axes(
            parent_ax,
            width="100%",
            height="100%",
            loc="lower left",  # lower left
            bbox_to_anchor=(cbar_x0, cbar_y0, cbar_width, cbar_height),
            bbox_transform=parent_ax.transAxes,
            borderpad=0,
        )
        ax_hist = inset_axes(
            parent_ax,
            width="100%",
            height="100%",
            loc="lower left",  # lower left
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
