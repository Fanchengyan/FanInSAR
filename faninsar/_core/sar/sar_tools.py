"""A module for SAR data processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from faninsar.plots.results import BaselinePlotResult

from .sar_property import Frequency, FrequencyUnit, Wavelength, WavelengthUnit

if TYPE_CHECKING:
    from datetime import datetime

    from matplotlib.axes import Axes
    from matplotlib.collections import Collection, LineCollection, PathCollection
    from matplotlib.colorbar import Colorbar
    from matplotlib.legend import Legend
    from numpy.typing import NDArray

    from .pairs import Pairs


def multi_look(
    arr_in: np.ndarray,
    azimuth_looks: int,
    range_looks: int,
) -> np.ndarray:
    """Multi-look an array by averaging blocks.

    Parameters
    ----------
    arr_in : numpy.ndarray
        Input array to be multi-looked. Can be 2D (rows, cols) or
        nD (..., rows, cols).
    azimuth_looks : int
        Number of looks in azimuth (rows).
    range_looks : int
        Number of looks in range (cols).

    Returns
    -------
    numpy.ndarray
        Multi-looked array.

    Notes
    -----
    The input array is cropped to the nearest multiple of `azimuth_looks`
    and `range_looks` before multi-looking.

    Examples
    --------
    >>> data = np.ones((4, 4))
    >>> multi_look(data, 2, 2)
    array([[1.]])

    """
    if azimuth_looks == 1 and range_looks == 1:
        return arr_in.copy()

    rows, cols = arr_in.shape[-2:]
    out_rows = rows // azimuth_looks
    out_cols = cols // range_looks

    # Slice to handle non-divisible dimensions
    arr = arr_in[..., : out_rows * azimuth_looks, : out_cols * range_looks]

    # Reshape and compute mean
    new_shape = arr.shape[:-2] + (out_rows, azimuth_looks, out_cols, range_looks)
    return arr.reshape(new_shape).mean(axis=(-3, -1))


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
        from faninsar.NSBAS import (
            LinearModel,
            NSBASInversion,
            NSBASMatrixFactory,
        )

        model_bs = LinearModel(pairs.dates)
        mf = NSBASMatrixFactory(values[:, None], pairs, model_bs)
        incs, *_ = NSBASInversion(mf, verbose=False, device="cpu").inverse()

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
    ) -> tuple[LineCollection, Colorbar | None]:
        """Create LineCollection for pairs and optional colorbar."""
        from matplotlib.collections import LineCollection
        from matplotlib.dates import date2num

        from faninsar.plots.utils import create_discrete_colormap

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

            colorbar = plt.colorbar(
                lc, ax=ax, label="Temporal baseline (days)", ticks=unique_days
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

    def _create_legend(  # noqa: PLR6301
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
        ...     baselines.plot(pairs, cmap="plasma")
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


class PhaseDeformationConverter:
    """Convert between phase and deformation (mm) for SAR interferometry.

    .. note::

        In FanInSAR, deformation/displacement is referenced to Earth,
        resulting in inverted signs when referring to radar measurements.
        Specifically, negative values indicate movement away from the radar
        (e.g., subsidence), while positive values signify movement towards
        the radar (e.g., uplift).

    """

    def __init__(
        self,
        freq_or_wl: Frequency | Wavelength,
    ) -> None:
        """Initialize the converter.

        Parameters
        ----------
        freq_or_wl : Frequency or Wavelength
            Either a Frequency or Wavelength object for the SAR mission.

        """
        if isinstance(freq_or_wl, Frequency):
            self._frequency = freq_or_wl.to_GHz()
            self._wavelength = freq_or_wl.to_wavelength(unit="mm")
        elif isinstance(freq_or_wl, Wavelength):
            self._wavelength = freq_or_wl.to_mm()
            self._frequency = freq_or_wl.to_frequency(unit="GHz")
        else:
            msg = "freq_or_wl must be a Frequency or Wavelength object, "
            msg += f"got {type(freq_or_wl)}"
            raise TypeError(msg)

        # convert radian to mm
        self.coef_rd2mm = -self.wavelength.data / 4 / np.pi

    @property
    def wavelength(self) -> Wavelength:
        """Get the wavelength of the SAR mission."""
        return self._wavelength

    @property
    def frequency(self) -> Frequency:
        """Get the frequency of the SAR mission."""
        return self._frequency

    @classmethod
    def from_frequency(
        cls,
        frequency: float,
        unit: FrequencyUnit = "GHz",
    ) -> PhaseDeformationConverter:
        """Create a PhaseDeformationConverter from frequency value.

        Parameters
        ----------
        frequency : float
            The frequency value.
        unit : Literal["GHz", "MHz", "kHz", "Hz"], optional
            The unit of frequency, by default "GHz".

        Returns
        -------
        PhaseDeformationConverter
            The converter instance.

        """
        freq = Frequency(frequency, unit)
        return cls(freq)

    @classmethod
    def from_wavelength(
        cls,
        wavelength: float,
        unit: WavelengthUnit = "m",
    ) -> PhaseDeformationConverter:
        """Create a PhaseDeformationConverter from wavelength value.

        Parameters
        ----------
        wavelength : float
            The wavelength value.
        unit : Literal["m", "cm", "dm", "mm"], optional
            The unit of wavelength, by default "m".

        Returns
        -------
        PhaseDeformationConverter
            The converter instance.

        """
        wl = Wavelength(wavelength, unit)
        return cls(wl)

    def __str__(self) -> str:
        """Return string representation of PhaseDeformationConverter object."""
        return f"PhaseDeformationConverter(wavelength={self.wavelength})"

    def __repr__(self) -> str:
        """Return string representation of PhaseDeformationConverter object."""
        return str(self)

    def phase2deformation(
        self,
        phase: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Convert phase to deformation (mm)."""
        return phase * self.coef_rd2mm

    def deformation2phase(
        self,
        deformation: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Convert deformation (mm) to phase (radian)."""
        return deformation / self.coef_rd2mm

    @staticmethod
    def wrap_phase(
        phase: np.ndarray, min_val: float = 0, max_val: float = 2 * np.pi
    ) -> np.ndarray:
        """Wrap phase to [min_val, max_val], by default [0, 2π].

        Parameters
        ----------
        phase : np.ndarray
            The phase to be wrapped.
        min_val : float, optional
            The minimum value of the wrapped phase, by default 0.
        max_val : float, optional
            The maximum value of the wrapped phase, by default 2π.

        Returns
        -------
        np.ndarray
            The wrapped phase.

        """
        return np.mod(phase - min_val, max_val - min_val) + min_val
