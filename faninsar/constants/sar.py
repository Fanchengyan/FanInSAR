from dataclasses import dataclass
from typing import Literal

SPEED_OF_LIGHT = 299792458


@dataclass
class Wavelength:
    """
    Dataclass for wavelength.
    """

    #: The value of the wavelength.
    value: float

    #: The unit of the wavelength. Default: m.
    unit: Literal["m", "cm", "dm", "mm"] = "m"

    def to_unit(self, unit: Literal["m", "cm", "dm", "mm"]) -> "Wavelength":
        """
        Convert wavelength to the specified unit.
        """
        if unit == "m":
            return self.to_m()
        elif unit == "cm":
            return self.to_cm()
        elif unit == "dm":
            return self.to_dm()
        elif unit == "mm":
            return self.to_mm()

    def to_mm(self) -> "Wavelength":
        """
        Convert wavelength to mm.
        """
        if self.unit == "m":
            return Wavelength(self.value * 1000, "mm")
        elif self.unit == "dm":
            return Wavelength(self.value * 100, "mm")
        elif self.unit == "cm":
            return Wavelength(self.value * 10, "mm")
        elif self.unit == "mm":
            return self

    def to_cm(self) -> "Wavelength":
        """
        Convert wavelength to cm.
        """
        if self.unit == "m":
            return Wavelength(self.value * 100, "cm")
        elif self.unit == "dm":
            return Wavelength(self.value * 10, "cm")
        elif self.unit == "cm":
            return self
        elif self.unit == "mm":
            return Wavelength(self.value / 10, "cm")

    def to_dm(self) -> "Wavelength":
        """
        Convert wavelength to dm.
        """
        if self.unit == "m":
            return Wavelength(self.value * 10, "dm")
        elif self.unit == "dm":
            return self
        elif self.unit == "cm":
            return Wavelength(self.value / 10, "dm")
        elif self.unit == "mm":
            return Wavelength(self.value / 100, "dm")

    def to_m(self) -> "Wavelength":
        """
        Convert wavelength to m.
        """
        if self.unit == "m":
            return self
        elif self.unit == "dm":
            return Wavelength(self.value / 10, "m")
        elif self.unit == "cm":
            return Wavelength(self.value / 100, "m")
        elif self.unit == "mm":
            return Wavelength(self.value / 1000, "m")

    def to_frequency(
        self,
        unit: Literal["GHz", "MHz", "kHz", "Hz"] = "GHz",
    ) -> "Frequency":
        """
        Convert wavelength to frequency.

        Parameters
        ----------
        unit : str
            The unit of the frequency. Default: GHz.
        """
        return Frequency(SPEED_OF_LIGHT / self.to_m().value, "Hz").to_unit(unit)


@dataclass
class Frequency:
    """
    Dataclass for frequency.
    """

    #: The value of the frequency.
    value: float

    #: The unit of the frequency. Default: GHz.
    unit: Literal["GHz", "MHz", "kHz", "Hz"] = "GHz"

    def to_unit(self, unit: Literal["GHz", "MHz", "kHz", "Hz"]) -> "Frequency":
        """
        Convert frequency to the specified unit.
        """
        if unit == "GHz":
            return self.to_GHz()
        elif unit == "MHz":
            return self.to_MHz()
        elif unit == "kHz":
            return self.to_kHz()
        elif unit == "Hz":
            return self.to_Hz()

    def to_GHz(self) -> "Frequency":
        """
        Convert frequency to GHz.
        """
        if self.unit == "Hz":
            return Frequency(self.value / 1e9, "GHz")
        if self.unit == "kHz":
            return Frequency(self.value / 1e6, "GHz")
        elif self.unit == "MHz":
            return Frequency(self.value / 1e3, "GHz")
        elif self.unit == "GHz":
            return self

    def to_MHz(self) -> "Frequency":
        """
        Convert frequency to MHz.
        """
        if self.unit == "Hz":
            return Frequency(self.value / 1e6, "MHz")
        elif self.unit == "kHz":
            return Frequency(self.value / 1e3, "MHz")
        elif self.unit == "MHz":
            return self
        elif self.unit == "GHz":
            return Frequency(self.value * 1e3, "MHz")

    def to_kHz(self) -> "Frequency":
        """
        Convert frequency to kHz.
        """
        if self.unit == "Hz":
            return Frequency(self.value / 1e3, "kHz")
        elif self.unit == "kHz":
            return self
        elif self.unit == "MHz":
            return Frequency(self.value * 1e3, "kHz")
        elif self.unit == "GHz":
            return Frequency(self.value * 1e6, "kHz")

    def to_Hz(self) -> "Frequency":
        """
        Convert frequency to Hz.
        """
        if self.unit == "Hz":
            return self
        elif self.unit == "kHz":
            return Frequency(self.value * 1e3, "Hz")
        elif self.unit == "MHz":
            return Frequency(self.value * 1e6, "Hz")
        elif self.unit == "GHz":
            return Frequency(self.value * 1e9, "Hz")

    def to_wavelength(
        self,
        unit: Literal["m", "cm", "dm", "mm"] = "m",
    ) -> Wavelength:
        """
        Convert frequency to wavelength.

        Parameters
        ----------
        unit : str
            The unit of the wavelength. Default: m.
        """
        return Wavelength(SPEED_OF_LIGHT / self.to_Hz().value).to_unit(unit)


class SAR:
    """
    Abstract base class for SAR missions.
    """

    _frequency: Frequency
    _wavelength: Wavelength

    @property
    def frequency(self) -> Frequency:
        return self._frequency

    @frequency.setter
    def frequency(self, value: Frequency) -> None:
        raise AttributeError("frequency for SAR mission is read-only.")

    @property
    def wavelength(self) -> Wavelength:
        return self.frequency.to_wavelength("mm")

    @wavelength.setter
    def wavelength(self, value: Wavelength) -> None:
        raise AttributeError("wavelength for SAR mission is read-only.")


class Sentinel1(SAR):
    """
    class for Sentinel-1 mission.
    """

    _frequency = Frequency(5.405, "GHz")
