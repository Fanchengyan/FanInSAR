"""Sentinel-1 mission package: constants + SAFE reader (same package)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from faninsar.logging import setup_logger
from faninsar.missions.base import Sensor, register
from faninsar.missions.s1.annotation import (
    parse_annotation_xml,
)
from faninsar.missions.s1.errors import (
    Sentinel1ProductError,
    UnsupportedPolarizationError,
    reject_product,
)
from faninsar.missions.s1.io import (
    BurstArray,
    RemoteSafe,
    estimate_burst_bytes,
    export_swath_bursts,
    extract_burst,
    extract_bursts,
    extract_remote_burst,
    read_burst_window,
    read_full_burst,
    read_swath_bursts,
    stitch_bursts,
)
from faninsar.missions.s1.orbit import read_eof_orbit
from faninsar.missions.s1.safe import open_safe_product
from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath

SPEED_OF_LIGHT_M_S: float = 299_792_458.0
S1_C_BAND_WAVELENGTH_M: float = 0.05546576

logger = setup_logger(__name__)


def __getattr__(name: str) -> Any:
    """Load the concrete Sentinel-1 Stack adapter lazily."""
    if name == "S1Stack":
        from faninsar.missions.s1.stack import S1Stack

        return S1Stack
    raise AttributeError(name)


@register(name="sentinel1")
class Sentinel1Sensor(Sensor):
    """Sentinel-1 SAFE / ZIP adapter implementing SensorAdapter."""

    name = "sentinel1"
    wavelength_m: float = S1_C_BAND_WAVELENGTH_M

    def open_product(self, uri: str, **kwargs: Any) -> Any:
        """Open a SAFE directory or ZIP."""
        return open_safe_product(Path(uri), **kwargs)

    def to_slc_product(self, handle: Any, **_kwargs: Any) -> Any:
        """Return the open product handle for production loaders."""
        return handle

    def read_slc_window(self, handle: Any, window: Any, **kwargs: Any) -> Any:
        """Read a complex window when the handle supports it."""
        if hasattr(handle, "read_window"):
            return handle.read_window(window, **kwargs)
        message = "Sentinel1 handle does not expose read_window; use production loaders"
        logger.error(message)
        raise NotImplementedError(message)


__all__ = [
    "S1_C_BAND_WAVELENGTH_M",
    "SPEED_OF_LIGHT_M_S",
    "BurstArray",
    "RemoteSafe",
    "S1Burst",
    "S1Product",
    "S1Stack",
    "S1Swath",
    "Sentinel1ProductError",
    "Sentinel1Sensor",
    "UnsupportedPolarizationError",
    "estimate_burst_bytes",
    "export_swath_bursts",
    "extract_burst",
    "extract_bursts",
    "extract_remote_burst",
    "open_safe_product",
    "parse_annotation_xml",
    "read_burst_window",
    "read_eof_orbit",
    "read_full_burst",
    "read_swath_bursts",
    "reject_product",
    "stitch_bursts",
]
