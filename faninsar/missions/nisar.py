"""NISAR mission adapter (WRAP ISCE3 — stub until fixtures exist)."""

from __future__ import annotations

from typing import Any

from faninsar.missions.base import Sensor, register


@register(name="nisar")
class NisarSensor(Sensor):
    """NISAR product adapter. Full ISCE3 wrap lands with real fixtures."""

    name = "nisar"

    def open_product(self, uri: str, **kwargs: Any) -> Any:
        """Open a NISAR product URI (not yet implemented)."""
        raise NotImplementedError(
            "NISAR open_product requires ISCE3 fixtures; URI=" + uri
        )

    def to_slc_product(self, handle: Any, **kwargs: Any) -> Any:
        """Convert NISAR handle to SLCProduct (not yet implemented)."""
        raise NotImplementedError("NISAR to_slc_product not yet implemented")

    def read_slc_window(self, handle: Any, window: Any, **kwargs: Any) -> Any:
        """Read NISAR SLC window (not yet implemented)."""
        raise NotImplementedError("NISAR read_slc_window not yet implemented")


__all__ = ["NisarSensor"]
