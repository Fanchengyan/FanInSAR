"""Sentinel-1 Level-1 SAFE/EOPF adapters."""

from __future__ import annotations

from .errors import Sentinel1ProductError, UnsupportedPolarizationError
from .io import (
    BurstArray,
    read_burst_window,
    read_full_burst,
    read_swath_bursts,
    stitch_bursts,
)
from .orbit import read_eof_orbit
from .safe import open_safe_product
from .types import S1Burst, S1Product, S1Swath

__all__ = [
    "BurstArray",
    "S1Burst",
    "S1Product",
    "S1Swath",
    "Sentinel1ProductError",
    "UnsupportedPolarizationError",
    "open_safe_product",
    "read_burst_window",
    "read_eof_orbit",
    "read_full_burst",
    "read_swath_bursts",
    "stitch_bursts",
]
