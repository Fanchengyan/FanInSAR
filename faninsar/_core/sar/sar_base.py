"""A module for SAR data processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

    from .pairs import Pairs


def multi_look(
    arr_in: np.ndarray,
    azimuth_looks: int,
    range_looks: int,
) -> np.ndarray:
    """Multi_look array with alks and rlks.

    Parameters
    ----------
    arr_in: numpy.ndarray
        input array to be multi-looked
    azimuth_looks: int
        number of looks in azimuth
    range_looks: int
        number of looks in range

    Returns
    -------
    numpy.ndarray
        multi-looked array

    """
    # TODO: 1. make it work 2. support for complex data
    # Get the shape of the input array
    rows, cols = arr_in.shape

    # Calculate the shape of the output array
    out_rows = rows // azimuth_looks
    out_cols = cols // range_looks

    # Initialize the output array
    arr_out = np.zeros((out_rows, out_cols), dtype=arr_in.dtype)

    # Perform multi-looking
    for i in range(out_rows):
        for j in range(out_cols):
            # Calculate the block to average
            block = arr_in[
                i * azimuth_looks : (i + 1) * azimuth_looks,
                j * range_looks : (j + 1) * range_looks,
            ]
            # Compute the mean of the block
            arr_out[i, j] = block.mean()

    return arr_out


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
            The cumulative values of the baselines relative to the first acquisition.

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
    def frame(self) -> pd.Series:
        """Return the DataFrame of the baselines."""
        return pd.Series(self.values, index=self.dates, name="baseline")

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
        from faninsar.NSBAS import LinearModel, NSBASInversion, NSBASMatrixFactory

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
        baselines = (
            self.frame[pairs.secondary].values - self.frame[pairs.primary].values
        )
        bs = pd.Series(baselines, index=pairs.to_names())
        bs.index.name = "pairs"
        bs.name = "baseline"
        return bs

    def plot(
        self,
        pairs: Pairs,
        pairs_removed: Pairs | None = None,
        plot_gaps: bool = True,
        ax: plt.Axes | None = None,
        xlabel: str = "Acquisition date",
        ylabel: str = "Perpendicular baseline (m)",
        legend: bool = True,
        legend_labels: list[str] | None = None,
        pairs_kwargs: dict | None = None,
        pairs_removed_kwargs: dict | None = None,
        acq_kwargs: dict | None = None,
        gaps_kwargs: dict | None = None,
    ) -> plt.Axes:
        """Plot the baselines of the interferograms.

        Parameters
        ----------
        pairs : Pairs
            All pairs used (temporal baseline).
        pairs_removed : Pairs, optional
            The pairs of the interferograms which are removed. Default is None.
        plot_gaps : bool
            Whether to plot the gaps between the acquisitions. Default is True.
        ax : plt.Axes
            The axes of the plot. Default is None, which means a new plot will
            be created.
        xlabel : str
            The label of the x-axis. Default is "Acquisition date".
        ylabel : str
            The label of the y-axis. Default is "Perpendicular baseline (m)".
        legend : bool
            Whether to show the legend. Default is True.
        legend_labels : list[str] | None
            The labels of the legend in the order of [valid pairs, removed pairs,
            acquisitions, gaps]. Default is None,
        pairs_kwargs : dict
            The keyword arguments for the pairs to :meth:`plt.plot`. Default is
            None.
        pairs_removed_kwargs : dict
            The keyword arguments for the pairs remove to :meth:`plt.plot`.
            Default is None.
        acq_kwargs : dict
            The keyword arguments for the acquisitions to :meth:`plt.plot`.
            Default is {}.
        gaps_kwargs : dict
            The keyword arguments for the gaps to :meth:`plt.vlines`. Default is
            None.

        Returns
        -------
        ax : plt.Axes
            The axes of the plot.

        """
        if gaps_kwargs is None:
            gaps_kwargs = {}
        if acq_kwargs is None:
            acq_kwargs = {}
        if pairs_removed_kwargs is None:
            pairs_removed_kwargs = {}
        if pairs_kwargs is None:
            pairs_kwargs = {}
        if legend_labels is None:
            legend_labels = ["Remained pairs", "Removed pairs", "Acquisitions", "Gaps"]
        if ax is None:
            ax = plt.gca()

        _pairs_kwargs = {"c": "tab:blue", "alpha": 0.5, "ls": "-"}
        _pairs_removed_kwargs = {"c": "r", "alpha": 0.3, "ls": "--"}

        _pairs_kwargs.update(pairs_kwargs)
        _pairs_removed_kwargs.update(pairs_removed_kwargs)

        pairs_valid = pairs
        if pairs_removed is not None:
            pairs_valid = pairs - pairs_removed

        # plot valid pairs
        for pair in pairs_valid:
            start, end = pair.primary, pair.secondary
            line_valid = ax.plot(
                [start, end],
                [self.frame[start], self.frame[end]],
                **_pairs_kwargs,
            )[0]
        # plot removed pairs
        if pairs_removed is not None:
            for pair in pairs_removed:
                start, end = pair.primary, pair.secondary
                line_removed = ax.plot(
                    [start, end],
                    [self.frame[start], self.frame[end]],
                    **_pairs_removed_kwargs,
                )[0]
        # plot acquisitions
        _pairs_kwargs = {"c": "tab:blue", "marker": "o", "ls": "", "alpha": 0.5}
        _pairs_kwargs.update(acq_kwargs)
        acq = ax.plot(self.dates, self.values, **_pairs_kwargs)[0]

        # plot gaps
        if plot_gaps:
            gaps = pairs.parse_gaps(pairs_removed)
            offset = pairs.days.min() / 2
            gaps = gaps - pd.Timedelta(offset, "D")

            dates_valid = np.setdiff1d(pairs.dates, gaps)
            vals = self.frame[dates_valid]
            margin = vals.std() / 3
            ymin, ymax = vals.min() - margin, vals.max() + margin
            _gaps_kwargs = {"color": "k", "ls": "--", "alpha": 0.5}
            _gaps_kwargs.update(gaps_kwargs)
            line_gaps = ax.vlines(gaps, ymin=ymin, ymax=ymax, **_gaps_kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(
                [line_valid, line_removed, acq, line_gaps],
                legend_labels,
            )

        return ax


class PhaseDeformationConverter:
    """A class to convert between phase and deformation (mm) for SAR interferometry."""

    # TODO: rewrite using new frequency and wavelength classes
    def __init__(
        self,
        frequency: float | None = None,
        wavelength: float | None = None,
    ) -> None:
        """Initialize the converter.

        Either wavelength or frequency should be provided. If both are provided,
        wavelength will be recalculated by frequency.

        Parameters
        ----------
        frequency : float
            The frequency of the radar signal. Unit: GHz.
        wavelength : float
            The wavelength of the radar signal. Unit: meter.
            this parameter will be ignored if frequency is provided.

        """
        speed_of_light = 299792458

        if frequency is not None:
            frequency = frequency * 1e9  # GHz to Hz
            self.wavelength = speed_of_light / frequency  # meter
            self.frequency = frequency
        elif wavelength is not None:
            self.wavelength = wavelength
            self.frequency = speed_of_light / wavelength
        else:
            msg = "Either wavelength or frequency should be provided."
            raise ValueError(msg)

        # convert radian to mm
        self.coef_rd2mm = -self.wavelength / 4 / np.pi * 1000

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

    def wrap_phase(self, phase: NDArray[np.floating]) -> NDArray[np.floating]:
        """Wrap phase to [0, 2π]."""
        # TODO: add user defined range
        return np.mod(phase, 2 * np.pi)
