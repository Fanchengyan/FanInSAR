"""Mission adapters and supported concrete Stack entry points."""

from __future__ import annotations

from typing import Any

from faninsar.missions.base import Sensor, get_mission, list_missions, register
from faninsar.missions.protocols import SensorAdapter


def __getattr__(name: str) -> Any:
    """Load a supported concrete Stack without creating import cycles."""
    if name == "S1Stack":
        from faninsar.missions.s1.stack import S1Stack

        return S1Stack
    if name == "NISARStack":
        from faninsar.missions.nisar.stack import NISARStack

        return NISARStack
    raise AttributeError(name)


__all__ = [
    "NISARStack",
    "S1Stack",
    "Sensor",
    "SensorAdapter",
    "get_mission",
    "list_missions",
    "register",
]
