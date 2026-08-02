"""Sensor adapter port — mission I/O boundary."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SensorAdapter(Protocol):
    """Mission-neutral adapter for opening and reading SLC products.

    Implementations live under ``faninsar.missions``. Processing stages never
    import missions directly; they consume ``SLCProduct`` handles only.
    """

    name: str

    def open_product(self, uri: str, **kwargs: Any) -> Any:
        """Open a mission product at *uri* and return a handle."""
        ...

    def to_slc_product(self, handle: Any, **kwargs: Any) -> Any:
        """Convert an open handle into a mission-neutral SLCProduct."""
        ...

    def read_slc_window(self, handle: Any, window: Any, **kwargs: Any) -> Any:
        """Read a spatial window from *handle* as a complex ndarray."""
        ...


__all__ = ["SensorAdapter"]
