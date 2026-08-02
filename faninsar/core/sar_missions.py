"""sar module contains classes for SAR missions."""

from __future__ import annotations

from faninsar.core.sar_property import Frequency, Wavelength


class SAR:
    """Base class for SAR missions with immutable frequency/wavelength properties.

    Examples
    --------
    >>> # Define a SAR mission by subclassing
    >>> class Sentinel1(SAR):
    ...     _frequency = Frequency(5.405, "GHz")

    >>> # Access frequency and wavelength directly from class
    >>> print(Sentinel1._frequency)
    5.405 GHz

    >>> # Or use with instantiation
    >>> s1 = Sentinel1()
    >>> print(s1.frequency)
    5.405 GHz
    >>> print(s1.wavelength)
    55.46 mm

    """

    _frequency: Frequency
    _wavelength: Wavelength

    def __setattr__(self, name: str, value: object) -> None:  # type: ignore[override]
        """Prevent setting computed properties to enforce read-only behavior."""
        if name in {"frequency", "wavelength"}:
            msg = f"{name} is read-only"
            raise AttributeError(msg)
        super().__setattr__(name, value)

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
        """The frequency of the SAR mission."""
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
        """The wavelength of the SAR mission."""
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
    """Sentinel-1 SAR mission (C-band, 5.405 GHz).

    Sentinel-1 is a European radar imaging mission with a C-band SAR
    operating at 5.405 GHz frequency (approximately 55.46 mm wavelength).

    Examples
    --------
    >>> # Or use with instantiation
    >>> s1 = Sentinel1()
    >>> print(s1.frequency)
    5.405 GHz
    >>> print(s1.wavelength)
    55.46 mm

    """

    _frequency = Frequency(5.405, "GHz")
