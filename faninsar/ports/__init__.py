"""Internal ports: SensorAdapter, ComputeBackend, IOBackend.

These Protocols are **not** part of the public root ``faninsar.__all__``.
"""

from __future__ import annotations

from faninsar.ports.compute import ComputeBackend
from faninsar.ports.io import Catalog, Format, IOBackend, Store
from faninsar.ports.sensor import SensorAdapter

__all__ = [
    "Catalog",
    "ComputeBackend",
    "Format",
    "IOBackend",
    "SensorAdapter",
    "Store",
]
