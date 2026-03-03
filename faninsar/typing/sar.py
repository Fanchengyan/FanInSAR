"""Typing for SAR wavelength and frequency units.

This module defines literal types for wavelength and frequency units used
throughout the FanInSAR library. These types provide IDE autocompletion
and type checking for unit-related parameters.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, get_args

# Wavelength units
WavelengthUnit: TypeAlias = Literal["m", "cm", "dm", "mm", "nm", "km", "um"]

# Frequency units
FrequencyUnit: TypeAlias = Literal["GHz", "MHz", "kHz", "Hz", "THz"]

# Runtime-accessible unit tuples (derived from the Literal types above)
WAVELENGTH_UNITS: tuple[str, ...] = get_args(WavelengthUnit)
FREQUENCY_UNITS: tuple[str, ...] = get_args(FrequencyUnit)
