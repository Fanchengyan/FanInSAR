"""A module for SAR data processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .sar_property import Frequency, Wavelength

if TYPE_CHECKING:
    from faninsar.typing import FrequencyUnit, WavelengthUnit

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
