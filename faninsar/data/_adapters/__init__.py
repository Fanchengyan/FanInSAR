"""FanInSAR accessor registrations."""

from __future__ import annotations

from typing import Any

from .xarray import FanInSARDataArrayAccessor

__all__ = ["ARIA", "FanInSARDataArrayAccessor", "HyP3S1", "LiCSAR"]


def __getattr__(name: str) -> Any:
    """Load provider adapters only when requested."""
    if name == "ARIA":
        from .aria import ARIA

        return ARIA
    if name == "HyP3S1":
        from .hyp3 import HyP3S1

        return HyP3S1
    if name == "LiCSAR":
        from .licsar import LiCSAR

        return LiCSAR
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
