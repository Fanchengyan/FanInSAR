"""A module for SAR data processing."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.core.sar_property import Frequency, Wavelength

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray

    from faninsar.core.types import FrequencyUnit, WavelengthUnit


@dataclass(frozen=True, slots=True)
class Looks:
    """Range-first multilook factors for one radar product.

    A look factor is an integer block size in one radar axis. Range looks
    apply to columns and azimuth looks apply to rows. The two factors are
    independent, so a product may be non-square.

    Parameters
    ----------
    range : int
        Number of range (column) looks, at least one.
    azimuth : int
        Number of azimuth (row) looks, at least one.

    Examples
    --------
    >>> looks = Looks(10, 2)
    >>> looks.range, looks.azimuth, looks.nlooks
    (10, 2, 20)
    >>> tuple(looks)
    (10, 2)

    """

    range: int
    azimuth: int

    def __post_init__(self) -> None:
        """Reject non-integer and non-positive look factors."""
        for name, value in (("range", self.range), ("azimuth", self.azimuth)):
            if isinstance(value, bool) or not isinstance(value, int):
                message = f"Looks.{name} must be an integer; got {value!r}"
                raise TypeError(message)
            if value < 1:
                message = f"Looks.{name} must be at least 1; got {value!r}"
                raise ValueError(message)

    @property
    def nlooks(self) -> int:
        """Return the nominal number of independent looks."""
        return self.range * self.azimuth

    def __iter__(self) -> Iterator[int]:
        """Iterate range then azimuth."""
        return iter((self.range, self.azimuth))


def coerce_looks(value: Looks | Sequence[int]) -> Looks:
    """Return ``value`` as a range-first :class:`Looks`.

    Parameters
    ----------
    value : Looks or sequence of int
        A ready value or a two-element ``(range, azimuth)`` sequence.

    Returns
    -------
    Looks
        The normalized range-first look factors.

    Raises
    ------
    TypeError
        If ``value`` is not a two-element sequence of integers.

    """
    if isinstance(value, Looks):
        return value
    try:
        range_looks, azimuth_looks = value
    except (TypeError, ValueError) as error:
        message = f"multilook must be a Looks or a (range, azimuth) pair; got {value!r}"
        raise TypeError(message) from error
    factors = []
    for name, factor in (
        ("range", range_looks),
        ("azimuth", azimuth_looks),
    ):
        if isinstance(factor, bool):
            message = f"multilook {name} factor must be an integer; got {factor!r}"
            raise TypeError(message)
        try:
            factors.append(operator.index(factor))
        except TypeError as error:
            message = f"multilook {name} factor must be an integer; got {factor!r}"
            raise TypeError(message) from error
    return Looks(factors[0], factors[1])


def multi_look(
    arr_in: np.ndarray,
    looks: Looks | Sequence[int],
) -> np.ndarray:
    """Multi-look an array by averaging blocks.

    Parameters
    ----------
    arr_in : numpy.ndarray
        Input array to be multi-looked. Can be 2D (rows, cols) or
        nD (..., rows, cols).
    looks : Looks or sequence of int
        Range-first ``(range_looks, azimuth_looks)`` factors. Range looks
        apply to columns and azimuth looks apply to rows.

    Returns
    -------
    numpy.ndarray
        Multi-looked array.

    Notes
    -----
    The input array is cropped to the nearest multiple of the azimuth and
    range factors before multi-looking.

    Examples
    --------
    >>> data = np.ones((4, 4))
    >>> multi_look(data, Looks(2, 2))
    array([[1.]])

    """
    resolved = coerce_looks(looks)
    azimuth_looks = resolved.azimuth
    range_looks = resolved.range
    if azimuth_looks == 1 and range_looks == 1:
        return arr_in.copy()

    rows, cols = arr_in.shape[-2:]
    out_rows = rows // azimuth_looks
    out_cols = cols // range_looks

    # Slice to handle non-divisible dimensions
    arr = arr_in[..., : out_rows * azimuth_looks, : out_cols * range_looks]

    # Reshape and compute mean
    new_shape = (*arr.shape[:-2], out_rows, azimuth_looks, out_cols, range_looks)
    return arr.reshape(new_shape).mean(axis=(-3, -1))


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
