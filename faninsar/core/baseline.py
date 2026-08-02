from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from faninsar.plots.utils import create_discrete_colormap

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from matplotlib.axes import Axes
    from matplotlib.collections import Collection, LineCollection, PathCollection
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.legend import Legend

    from faninsar.core.pairs import Pairs
    from faninsar.plots.hist_colorbar import HistColorbar


class Baselines:
    """A class manage the baselines of the interferograms."""

    def __init__(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        values: np.ndarray,
    ) -> None:
        """Initialize the Baselines object.

        Parameters
        ----------
        dates : pd.DatetimeIndex | Sequence[datetime]
            The dates of the SAR acquisitions.
        values : np.ndarray
            The cumulative values of the baselines relative to the first
            acquisition.

        """
        dates = pd.to_datetime(dates)
        values = np.asarray(values, dtype=np.float32).flatten()

        if len(dates) != len(values):
            msg = "The length of dates and values should be the same."
            raise ValueError(msg)

        self._dates = dates
        self._values = np.asarray(values, dtype=np.float32)

    def __repr__(self) -> str:
        """Return the representation of the Baselines object."""
        return f"Baselines(num={len(self)})"

    def __str__(self) -> str:
        """Return the string representation of the Baselines object."""
        return f"Baselines(num={len(self)})"

    def __len__(self) -> int:
        """Return the number of the baselines."""
        return len(self.values)

    @property
    def series(self) -> pd.Series:
        """Return the Series of the baselines."""
        return pd.Series(self.values, index=self.dates)

    @property
    def values(self) -> np.ndarray:
        """Return the values of the baselines."""
        return self._values

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Return the dates of the SAR acquisitions."""
        return self._dates

    @classmethod
    def from_pair_wise(cls, pairs: Pairs, values: np.ndarray) -> Baselines:
        """Generate the Baselines object from the pair-wise baseline.

        Parameters
        ----------
        pairs : Pairs
            The pairs instance of the interferograms.
        values : np.ndarray
            The values of spatial baselines of the pairs.

        Returns
        -------
        baselines : Baselines
            The Baselines object.

        """
        from faninsar.timeseries import LinearModel, NSBASSolver

        solver = NSBASSolver(
            values[:, None],
            pairs,
            LinearModel(pairs.dates),
            device="cpu",
            verbose=False,
        )
        incs, *_ = solver.inverse()

        cum = np.cumsum(incs, axis=0)
        cum = np.insert(cum, 0, 0, axis=0)
        return cls(pairs.dates, cum.flatten())

    def to_pair_wise(self, pairs: Pairs) -> pd.Series:
        """Generate the pair-wise baseline from the Baselines object.

        Parameters
        ----------
        pairs : Pairs
            The pairs of the interferograms.

        Returns
        -------
        values : np.ndarray
            The values of the baselines.

        """
        baselines = self.series[pairs.secondary] - self.series[pairs.primary]
        bs = pd.Series(baselines, index=pairs.to_names())
        bs.index.name = "pairs"
        bs.name = "baseline"
        return bs

    def _create_pairs_collection(
        self, pairs: Pairs, cmap: str | None, ax: Axes
    ) -> tuple[LineCollection, HistColorbar | None]:
        """Create LineCollection for pairs and optional colorbar."""
        from matplotlib.collections import LineCollection
        from matplotlib.dates import date2num

        # Create line segments for pairs
        pair_segments = [
            [
                [date2num(p.primary), self.series[p.primary]],
                [date2num(p.secondary), self.series[p.secondary]],
            ]
            for p in pairs
        ]

        if cmap is not None:
            days_array = pairs.days.data
            unique_days = np.unique(days_array)
            discrete_cmap, norm = create_discrete_colormap(unique_days, cmap)

            lc = LineCollection(
                pair_segments, cmap=discrete_cmap, norm=norm, linestyles="-"
            )
            lc.set_array(days_array)
            pairs_collection = ax.add_collection(lc)

            from faninsar.plots.hist_colorbar import HistColorbar

            colorbar = HistColorbar(
                data=days_array,
                mappable=lc,
                ax=ax,
                label="Temporal baseline (days)",
                ticks=unique_days,
            )
        else:
            lc = LineCollection(pair_segments, colors="tab:blue", linestyles="-")
            pairs_collection = ax.add_collection(lc)
            colorbar = None

        return pairs_collection, colorbar

    def _plot_gaps(
        self, pairs: Pairs, pairs_removed: Pairs | None, ax: Axes
    ) -> LineCollection | None:
        """Plot gap lines."""
        if len(pairs) == 0:
            return None

        gaps = pairs.parse_gaps(pairs_removed)
        if len(gaps) == 0:
            return None

        dates_valid = np.setdiff1d(pairs.dates.data, gaps)
        vals = self.series[dates_valid]
        margin = vals.std() / 3
        ymin, ymax = vals.min() - margin, vals.max() + margin

        return ax.vlines(gaps, ymin=ymin, ymax=ymax, color="k", ls="--", alpha=0.5)

    def _create_legend(
        self,
        ax: Axes,
        pairs_collection: Collection | None,
        pairs_removed_lines: list,
        acq_collection: PathCollection,
        gaps_lines: LineCollection | None,
        cmap: str | None,
    ) -> tuple[Legend, list[str]]:
        """Create legend for the plot.

        Returns
        -------
        tuple[Legend, list[str]]
            A tuple of (legend, legend_order) where legend_order is a list
            of keys like ["pairs", "pairs_removed", "acquisitions", "gaps"].

        """
        from faninsar.plots.utils import HandlerGradientLine

        handles = []
        labels = []
        legend_order = []
        handler_map = {}

        # Add acquisitions
        handles.append(acq_collection)
        labels.append("Acquisitions")
        legend_order.append("acquisitions")

        # Add pairs
        if pairs_collection is not None:
            if cmap:
                # Use a dummy handle and custom handler for gradient
                proxy = Line2D([], [], color="gray")  # dummy, not used directly
                handler_map[proxy] = HandlerGradientLine(
                    pairs_collection.get_cmap(), pairs_collection.norm
                )
            else:
                proxy = Line2D(
                    [],
                    [],
                    color="tab:blue",
                    linestyle="-",
                )
            handles.append(proxy)
            pairs_valid = "Valid pairs" if pairs_removed_lines else "Pairs"
            labels.append(pairs_valid)
            legend_order.append("pairs")

        # Add removed pairs
        if len(pairs_removed_lines) > 0:
            handles.append(pairs_removed_lines[0])
            labels.append("Removed pairs")
            legend_order.append("pairs_removed")

        # Add gaps
        if gaps_lines is not None:
            handles.append(gaps_lines)
            labels.append("Gaps")
            legend_order.append("gaps")

        return ax.legend(handles, labels, handler_map=handler_map), legend_order

    def plot(
        self,
        pairs: Pairs,
        pairs_removed: Pairs | None = None,
        ax: Axes | None = None,
        cmap: str | None = None,
        plot_gaps: bool = True,
        legend: bool = True,
        figsize: tuple[float, float] = (12, 4),
    ) -> BaselinePlotResult:
        """Plot the baselines of the interferograms.

        Parameters
        ----------
        pairs : Pairs
            All pairs used (temporal baseline).
        pairs_removed : Pairs, optional
            The pairs removed. Default is None.
        ax : Axes, optional
            The axes of the plot. If None, a new plot will be created.
        cmap : str, optional
            Colormap name to color-code pairs by temporal baseline (days).
            If None, pairs are plotted with a single color.
        plot_gaps : bool
            Whether to plot the gaps between acquisitions. Default is True.
        legend : bool
            Whether to show the legend. Default is True.
        figsize : tuple[float, float], optional
            Figure size if a new figure is created. Default is (10, 4).

        Returns
        -------
        result : BaselinePlotResult
            Result object containing all plot elements. Use convenience
            methods to customize the plot after creation.

        Examples
        --------
        Basic usage:

        >>> result = baselines.plot(pairs)
        >>> result.set_xlabel("Date", fontsize=12)
        >>> result.set_ylabel("Baseline (m)", fontsize=12)

        With colormap:

        >>> result = baselines.plot(pairs, cmap="viridis")
        >>> result.set_colorbar_label("Days", fontsize=14)

        Advanced customization:

        >>> result = baselines.plot(pairs, cmap="plasma")
        >>> result.set_pairs_style(linewidths=2, alpha=0.8)
        >>> result.set_acq_style(markersize=10, color="red")
        >>> result.set_legend_labels(pairs="Valid", acquisitions="Acq")
        >>> result.savefig("baseline.png", dpi=300)

        Or using chain:

        >>> result = (
        ...     baselines
        ...     .plot(pairs, cmap="plasma")
        ...     .set_pairs_style(linewidths=2, alpha=0.8)
        ...     .set_acq_style(markersize=10, color="red")
        ...     .set_legend_labels(pairs="Valid", acquisitions="Acq")
        ...     .savefig("baseline.png", dpi=300)
        ... )

        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Compute valid pairs
        pairs_valid = pairs if pairs_removed is None else pairs - pairs_removed

        # Plot valid pairs using LineCollection
        pairs_collection = None
        colorbar = None
        if len(pairs_valid) > 0:
            pairs_collection, colorbar = self._create_pairs_collection(
                pairs_valid, cmap, ax
            )

        # Plot removed pairs
        pairs_removed_lines = []
        if pairs_removed is not None:
            for pair in pairs_removed:
                line = ax.plot(
                    [pair.primary, pair.secondary],
                    [self.series[pair.primary], self.series[pair.secondary]],
                    c="r",
                    ls="--",
                )[0]
                pairs_removed_lines.append(line)

        # Plot acquisitions
        acq_collection = ax.scatter(
            self.dates,
            self.values,
            c="tab:blue",
            marker="o",
            ls="",
            alpha=0.5,
            zorder=2,
        )

        # Plot gaps
        gaps_lines = None
        if plot_gaps and pairs_removed is not None:
            gaps_lines = self._plot_gaps(pairs, pairs_removed, ax)

        # Set default labels, can be changed by returned BaselinePlotResult
        ax.set_xlabel("Acquisition date")
        ax.set_ylabel("Perpendicular baseline (m)")
        ax.autoscale()

        # Create legend
        legend_order = []
        if legend:
            _, legend_order = self._create_legend(
                ax,
                pairs_collection,
                pairs_removed_lines,
                acq_collection,
                gaps_lines,
                cmap,
            )

        return BaselinePlotResult(
            ax=ax,
            pairs_collection=pairs_collection,
            pairs_removed_lines=pairs_removed_lines,
            acq_collection=acq_collection,
            gaps_lines=gaps_lines,
            colorbar=colorbar,
            _legend_order=legend_order,
        )


@dataclass
class BaselinePlotResult:
    """Result object from Baselines.plot() containing all plot elements.

    This class encapsulates all the visual elements created by the
    Baselines.plot() method, allowing users to customize the plot after
    creation.

    Attributes
    ----------
    ax : Axes
        The matplotlib axes object.
    pairs_collection : LineCollection | None
        The LineCollection for valid pairs.
    pairs_removed_lines : list[Line2D]
        List of Line2D objects for removed pairs.
    acq_collection : PathCollection | None
        The PathCollection object for acquisition points.
    gaps_lines : LineCollection | None
        The LineCollection for gap vertical lines.
    colorbar : HistColorbar | None
        The colorbar object (if cmap was used).

    Examples
    --------
    Create a baseline plot and get the result object:

    >>> result = baselines.plot(pairs, cmap="viridis")

    Modify pairs style:

    >>> result.set_pairs_style(linewidths=2, alpha=0.8)

    Modify colorbar:

    >>> result.set_colorbar_label("Time Span (days)", fontsize=12)

    Modify acquisition points:

    >>> result.set_acq_style(markersize=10, color="red")

    Update the figure:

    >>> result.update_canvas()

    Save the figure:

    >>> result.savefig("baseline_plot.png", dpi=300)

    """

    ax: Axes
    pairs_collection: LineCollection | None = None
    pairs_removed_lines: list[Line2D] = field(default_factory=list)
    acq_collection: PathCollection | None = None
    gaps_lines: object | None = None
    colorbar: HistColorbar | None = None
    _legend_order: list[str] = field(default_factory=list)

    @property
    def fig(self) -> Figure:
        """Return the root figure containing the baseline axes."""
        return self.ax.get_figure(root=True)

    @property
    def ax_figure(self) -> Figure | SubFigure:
        """Return the parent figure containing the baseline axes."""
        return self.ax.get_figure(root=False)

    def savefig(self, *args, **kwargs) -> None:
        """Save the figure to a file.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to :meth:`matplotlib.figure.Figure.savefig`.

        Returns
        -------
        result
            The result from Figure.savefig().

        """
        return self.fig.savefig(*args, **kwargs)

    def set_pairs_style(self, **kwargs) -> BaselinePlotResult:
        """Set style for valid pairs LineCollection.

        Parameters
        ----------
        **kwargs
            Keyword arguments for LineCollection setters. Common options:
            linewidths, linestyles, colors, alpha, etc.

        Returns
        -------
        self : BaselinePlotResult
            Returns self for method chaining.

        Examples
        --------
        >>> result.set_pairs_style(linewidths=2, alpha=0.8)

        """
        if self.pairs_collection is not None:
            for key, value in kwargs.items():
                setter = getattr(self.pairs_collection, f"set_{key}", None)
                if setter:
                    setter(value)
        return self

    def set_acq_style(self, **kwargs) -> BaselinePlotResult:
        """Set style for acquisition points.

        Parameters
        ----------
        **kwargs
            Keyword arguments for Line2D setters. Common options:
            markersize, markerfacecolor, markeredgecolor, color, alpha, etc.

        Returns
        -------
        self : BaselinePlotResult
            Returns self for method chaining.

        Examples
        --------
        >>> result.set_acq_style(markersize=10, color="red")

        """
        if self.acq_collection is not None:
            for key, value in kwargs.items():
                setter = getattr(self.acq_collection, f"set_{key}", None)
                if setter:
                    setter(value)
        return self

    def set_colorbar_label(self, label: str, **kwargs) -> BaselinePlotResult:
        """Set colorbar label.

        Parameters
        ----------
        label : str
            The label text for the colorbar.
        **kwargs
            Additional keyword arguments passed to colorbar.set_label().

        Returns
        -------
        self : BaselinePlotResult
            Returns self for method chaining.

        Examples
        --------
        >>> result.set_colorbar_label("Days", fontsize=14)

        """
        if self.colorbar is not None:
            self.colorbar.set_label(label, **kwargs)
        return self

    def update_canvas(self) -> BaselinePlotResult:
        """Redraw the canvas to reflect changes.

        Returns
        -------
        self : BaselinePlotResult
            Returns self for method chaining.

        """
        if hasattr(self.fig.canvas, "draw_idle"):
            self.fig.canvas.draw_idle()
        return self

    def set_xlabel(self, label: str, **kwargs) -> BaselinePlotResult:
        """Set x-axis label.

        Parameters
        ----------
        label : str
            The label text for the x-axis.
        **kwargs
            Additional keyword arguments passed to ax.set_xlabel().

        Returns
        -------
        self : BaselinePlotResult
            Returns self for method chaining.

        """
        self.ax.set_xlabel(label, **kwargs)
        return self

    def set_ylabel(self, label: str, **kwargs) -> BaselinePlotResult:
        """Set y-axis label.

        Parameters
        ----------
        label : str
            The label text for the y-axis.
        **kwargs
            Additional keyword arguments passed to ax.set_ylabel().

        Returns
        -------
        self : BaselinePlotResult
            Returns self for method chaining.

        """
        self.ax.set_ylabel(label, **kwargs)
        return self

    def set_legend_labels(
        self,
        pairs: str | None = None,
        pairs_removed: str | None = None,
        acquisitions: str | None = None,
        gaps: str | None = None,
    ) -> BaselinePlotResult:
        """Update legend labels.

        Parameters
        ----------
        pairs : str, optional
            Label for valid pairs.
        pairs_removed : str, optional
            Label for removed pairs.
        acquisitions : str, optional
            Label for acquisition points.
        gaps : str, optional
            Label for gap lines.

        Returns
        -------
        self : BaselinePlotResult
            Returns self for method chaining.

        """
        legend = self.ax.get_legend()
        if legend is not None and self._legend_order:
            # Map legend keys to new labels
            label_map = {
                "pairs": pairs,
                "pairs_removed": pairs_removed,
                "acquisitions": acquisitions,
                "gaps": gaps,
            }
            texts = legend.get_texts()
            for i, key in enumerate(self._legend_order):
                if i < len(texts) and key in label_map and label_map[key] is not None:
                    texts[i].set_text(label_map[key])
        return self
