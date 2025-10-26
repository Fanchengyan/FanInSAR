"""Compatibility exports for SAR constants and base mission class.

This module re-exports the SAR base class and unit classes used in tests and
user extensions. It provides a stable import path under faninsar.constants.
"""

from __future__ import annotations

from faninsar._core.sar.sar_missions import SAR
from faninsar._core.sar.sar_property import Frequency, Wavelength

__all__ = ["SAR", "Frequency", "Wavelength"]
