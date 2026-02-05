"""Typing for SAR wavelength and frequency units.

This module defines literal types for wavelength and frequency units used
throughout the FanInSAR library. These types provide IDE autocompletion
and type checking for unit-related parameters.
"""

from __future__ import annotations

from typing import Literal

# Wavelength units
WavelengthUnit = Literal["m", "cm", "dm", "mm", "nm", "km", "um"]

# Frequency units
FrequencyUnit = Literal["GHz", "MHz", "kHz", "Hz", "THz"]