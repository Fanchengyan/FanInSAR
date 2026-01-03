from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.lines import Line2D


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
    colorbar : Colorbar | None
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
    colorbar: Colorbar | None = None
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
