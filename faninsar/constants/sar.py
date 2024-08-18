"""sar module contains classes for SAR missions."""

from dataclasses import dataclass
from typing import Literal

SPEED_OF_LIGHT = 299792458


@dataclass
class Wavelength:
    """Dataclass for wavelength."""

    #: The value of the wavelength.
    value: float

    #: The unit of the wavelength. Default: m.
    unit: Literal["m", "cm", "dm", "mm"] = "m"

    def to_unit(self, unit: Literal["m", "cm", "dm", "mm"]) -> "Wavelength":
        """Convert wavelength to the specified unit."""
        if unit == "m":
            return self.to_m()
        if unit == "cm":
            return self.to_cm()
        if unit == "dm":
            return self.to_dm()
        if unit == "mm":  # noqa: RET503
            return self.to_mm()

    def to_mm(self) -> "Wavelength":
        """Convert wavelength to mm."""
        if self.unit == "m":
            return Wavelength(self.value * 1000, "mm")
        if self.unit == "dm":
            return Wavelength(self.value * 100, "mm")
        if self.unit == "cm":
            return Wavelength(self.value * 10, "mm")
        if self.unit == "mm":  # noqa: RET503
            return self

    def to_cm(self) -> "Wavelength":
        """Convert wavelength to cm."""
        if self.unit == "m":
            return Wavelength(self.value * 100, "cm")
        if self.unit == "dm":
            return Wavelength(self.value * 10, "cm")
        if self.unit == "cm":
            return self
        if self.unit == "mm":  # noqa: RET503
            return Wavelength(self.value / 10, "cm")

    def to_dm(self) -> "Wavelength":
        """Convert wavelength to dm."""
        if self.unit == "m":
            return Wavelength(self.value * 10, "dm")
        if self.unit == "dm":
            return self
        if self.unit == "cm":
            return Wavelength(self.value / 10, "dm")
        if self.unit == "mm":  # noqa: RET503
            return Wavelength(self.value / 100, "dm")

    def to_m(self) -> "Wavelength":
        """Convert wavelength to m."""
        if self.unit == "m":
            return self
        if self.unit == "dm":
            return Wavelength(self.value / 10, "m")
        if self.unit == "cm":
            return Wavelength(self.value / 100, "m")
        if self.unit == "mm":  # noqa: RET503
            return Wavelength(self.value / 1000, "m")

    def to_frequency(
        self,
        unit: Literal["GHz", "MHz", "kHz", "Hz"] = "GHz",
    ) -> "Frequency":
        """Convert wavelength to frequency.

        Parameters
        ----------
        unit : str
            The unit of the frequency. Default: GHz.

        """
        return Frequency(SPEED_OF_LIGHT / self.to_m().value, "Hz").to_unit(unit)


@dataclass
class Frequency:
    """Dataclass for frequency."""

    #: The value of the frequency.
    value: float

    #: The unit of the frequency. Default: GHz.
    unit: Literal["GHz", "MHz", "kHz", "Hz"] = "GHz"

    def to_unit(self, unit: Literal["GHz", "MHz", "kHz", "Hz"]) -> "Frequency":
        """Convert frequency to the specified unit."""
        if unit == "GHz":
            return self.to_GHz()
        if unit == "MHz":
            return self.to_MHz()
        if unit == "kHz":
            return self.to_kHz()
        if unit == "Hz":  # noqa: RET503
            return self.to_Hz()

    def to_GHz(self) -> "Frequency":
        """Convert frequency to GHz."""
        if self.unit == "Hz":
            return Frequency(self.value / 1e9, "GHz")
        if self.unit == "kHz":
            return Frequency(self.value / 1e6, "GHz")
        if self.unit == "MHz":
            return Frequency(self.value / 1e3, "GHz")
        if self.unit == "GHz":  # noqa: RET503
            return self

    def to_MHz(self) -> "Frequency":
        """Convert frequency to MHz."""
        if self.unit == "Hz":
            return Frequency(self.value / 1e6, "MHz")
        if self.unit == "kHz":
            return Frequency(self.value / 1e3, "MHz")
        if self.unit == "MHz":
            return self
        if self.unit == "GHz":  # noqa: RET503
            return Frequency(self.value * 1e3, "MHz")

    def to_kHz(self) -> "Frequency":
        """Convert frequency to kHz."""
        if self.unit == "Hz":
            return Frequency(self.value / 1e3, "kHz")
        if self.unit == "kHz":
            return self
        if self.unit == "MHz":
            return Frequency(self.value * 1e3, "kHz")
        if self.unit == "GHz":  # noqa: RET503
            return Frequency(self.value * 1e6, "kHz")

    def to_Hz(self) -> "Frequency":
        """Convert frequency to Hz."""
        if self.unit == "Hz":
            return self
        if self.unit == "kHz":
            return Frequency(self.value * 1e3, "Hz")
        if self.unit == "MHz":
            return Frequency(self.value * 1e6, "Hz")
        if self.unit == "GHz":  # noqa: RET503
            return Frequency(self.value * 1e9, "Hz")

    def to_wavelength(
        self,
        unit: Literal["m", "cm", "dm", "mm"] = "m",
    ) -> Wavelength:
        """Convert frequency to wavelength.

        Parameters
        ----------
        unit : str
            The unit of the wavelength. Default: m.

        """
        return Wavelength(SPEED_OF_LIGHT / self.to_Hz().value).to_unit(unit)


class SAR:
    """Abstract base class for SAR missions."""

    _frequency: Frequency
    _wavelength: Wavelength

    @property
    def frequency(self) -> Frequency:
        """Get the frequency of the SAR mission."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: Frequency) -> None:  # noqa: ARG002
        msg = "frequency for SAR mission is read-only."
        raise AttributeError(msg)

    @property
    def wavelength(self) -> Wavelength:
        """Get the wavelength of the SAR mission."""
        return self.frequency.to_wavelength("mm")

    @wavelength.setter
    def wavelength(self, value: Wavelength) -> None:  # noqa: ARG002
        msg = "wavelength for SAR mission is read-only."
        raise AttributeError(msg)


class Sentinel1(SAR):
    """class for Sentinel-1 mission."""

    _frequency = Frequency(5.405, "GHz")
