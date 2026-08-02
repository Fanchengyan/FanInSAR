"""ALOS-2 mission adapter stub."""

from __future__ import annotations

from typing import Any

from faninsar.missions.base import Sensor, register


@register(name="alos2")
class Alos2Sensor(Sensor):
    """ALOS-2 stub registered for discovery; readers land later."""

    name = "alos2"

    def open_product(self, uri: str, **kwargs: Any) -> Any:
        """Open an ALOS-2 product (stub)."""
        raise NotImplementedError(f"ALOS-2 open_product not implemented: {uri}")

    def to_slc_product(self, handle: Any, **kwargs: Any) -> Any:
        """Convert ALOS-2 handle to SLCProduct (stub)."""
        raise NotImplementedError("ALOS-2 to_slc_product not implemented")

    def read_slc_window(self, handle: Any, window: Any, **kwargs: Any) -> Any:
        """Read ALOS-2 SLC window (stub)."""
        raise NotImplementedError("ALOS-2 read_slc_window not implemented")


__all__ = ["Alos2Sensor"]
