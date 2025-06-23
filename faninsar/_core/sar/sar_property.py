from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SPEED_OF_LIGHT = 299792458

# Conversion factors to meters (base unit)
_WAVELENGTH_UNIT_TO_METERS = {
    "m": 1.0,
    "dm": 0.1,
    "cm": 0.01,
    "mm": 0.001,
}

# Conversion factors to Hz (base unit)
_FREQUENCY_UNIT_TO_HZ = {
    "Hz": 1.0,
    "kHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
}

WavelengthUnit = Literal["m", "cm", "dm", "mm"]
FrequencyUnit = Literal["GHz", "MHz", "kHz", "Hz"]


@dataclass
class Wavelength:
    """Dataclass for wavelength."""

    data: float
    """The data of the wavelength."""

    unit: WavelengthUnit = "m"
    """The unit of the wavelength."""

    def to_unit(self, unit: WavelengthUnit) -> Wavelength:
        """Convert wavelength to the specified unit."""
        if unit not in _WAVELENGTH_UNIT_TO_METERS:
            msg = f"Invalid unit: {unit}. Must be one of 'm', 'cm', 'dm', 'mm'."
            raise ValueError(msg)

        if self.unit == unit:
            return self

        # Convert current value to meters, then to target unit
        value_in_meters = self.data * _WAVELENGTH_UNIT_TO_METERS[self.unit]
        target_value = value_in_meters / _WAVELENGTH_UNIT_TO_METERS[unit]

        return Wavelength(target_value, unit)

    def to_mm(self) -> Wavelength:
        """Convert wavelength to mm."""
        return self.to_unit("mm")

    def to_cm(self) -> Wavelength:
        """Convert wavelength to cm."""
        return self.to_unit("cm")

    def to_dm(self) -> Wavelength:
        """Convert wavelength to dm."""
        return self.to_unit("dm")

    def to_m(self) -> Wavelength:
        """Convert wavelength to m."""
        return self.to_unit("m")

    def to_frequency(
        self,
        unit: FrequencyUnit = "GHz",
    ) -> Frequency:
        """Convert wavelength to frequency.

        Parameters
        ----------
        unit : str
            The unit of the frequency. Default: GHz.

        """
        return Frequency(SPEED_OF_LIGHT / self.to_m().data, "Hz").to_unit(unit)


@dataclass
class Frequency:
    """Dataclass for frequency."""

    data: float
    """The data of the frequency."""

    unit: FrequencyUnit = "GHz"
    """The unit of the frequency."""

    def to_unit(self, unit: FrequencyUnit) -> Frequency:
        """Convert frequency to the specified unit."""
        if unit not in _FREQUENCY_UNIT_TO_HZ:
            msg = f"Invalid unit: {unit}. Must be one of 'GHz', 'MHz', 'kHz', 'Hz'."
            raise ValueError(msg)

        if self.unit == unit:
            return self

        # Convert current value to Hz, then to target unit
        value_in_hz = self.data * _FREQUENCY_UNIT_TO_HZ[self.unit]
        target_value = value_in_hz / _FREQUENCY_UNIT_TO_HZ[unit]

        return Frequency(target_value, unit)

    def to_GHz(self) -> Frequency:
        """Convert frequency to GHz."""
        return self.to_unit("GHz")

    def to_MHz(self) -> Frequency:
        """Convert frequency to MHz."""
        return self.to_unit("MHz")

    def to_kHz(self) -> Frequency:
        """Convert frequency to kHz."""
        return self.to_unit("kHz")

    def to_Hz(self) -> Frequency:
        """Convert frequency to Hz."""
        return self.to_unit("Hz")

    def to_wavelength(
        self,
        unit: WavelengthUnit = "m",
    ) -> Wavelength:
        """Convert frequency to wavelength.

        Parameters
        ----------
        unit : str
            The unit of the wavelength. Default: m.

        """
        return Wavelength(SPEED_OF_LIGHT / self.to_Hz().data).to_unit(unit)
