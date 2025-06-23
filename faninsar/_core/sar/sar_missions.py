"""sar module contains classes for SAR missions."""

from __future__ import annotations

from .sar_property import Frequency, Wavelength


class SAR:
    """Abstract base class for SAR missions."""

    _frequency: Frequency
    _wavelength: Wavelength

    def __repr__(self) -> str:
        """Return a string representation of the SAR mission."""
        frequency = None
        wavelength = None
        if hasattr(self, "_frequency") or hasattr(self, "_wavelength"):
            frequency = self.frequency
            wavelength = self.wavelength
        return (
            f"{self.__class__.__name__}(frequency={frequency}, wavelength={wavelength})"
        )

    def __str__(self) -> str:
        """Return a string representation of the SAR mission."""
        return self.__repr__()

    @property
    def frequency(self) -> Frequency:
        """Get the frequency of the SAR mission."""
        if hasattr(self, "_frequency"):
            return self._frequency
        if hasattr(self, "_wavelength"):
            return self._wavelength.to_frequency()
        msg = (
            "Neither frequency nor wavelength is defined for "
            f"{self.__class__.__name__}."
        )
        raise ValueError(msg)

    @property
    def wavelength(self) -> Wavelength:
        """Get the wavelength of the SAR mission."""
        if hasattr(self, "_wavelength"):
            return self._wavelength
        if hasattr(self, "_frequency"):
            return self._frequency.to_wavelength("mm")
        msg = (
            "Neither frequency nor wavelength is defined for "
            f"{self.__class__.__name__}."
        )
        raise ValueError(msg)


class Sentinel1(SAR):
    """class for Sentinel-1 mission."""

    _frequency = Frequency(5.405, "GHz")
