"""Mission registry: Sensor base class and @register decorator."""

from __future__ import annotations

from typing import Any, ClassVar

_REGISTRY: dict[str, type] = {}


class Sensor:
    """Base mission adapter registered via :func:`register`."""

    name: ClassVar[str] = ""

    def open_product(self, uri: str, **kwargs: Any) -> Any:
        """Open a mission product. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.open_product")

    def to_slc_product(self, handle: Any, **kwargs: Any) -> Any:
        """Convert an open handle to SLCProduct. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.to_slc_product")

    def read_slc_window(self, handle: Any, window: Any, **kwargs: Any) -> Any:
        """Read a complex SLC window. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.read_slc_window")


def register(cls: type | None = None, *, name: str | None = None):
    """Register a Sensor subclass in the global mission registry."""

    def decorator(sensor_cls: type) -> type:
        key = name or getattr(sensor_cls, "name", None) or sensor_cls.__name__.lower()
        if not key:
            raise ValueError("mission register requires a non-empty name")
        sensor_cls.name = key  # type: ignore[attr-defined]
        _REGISTRY[key] = sensor_cls
        return sensor_cls

    if cls is not None:
        return decorator(cls)
    return decorator


def list_missions() -> list[str]:
    """Return sorted registered mission names."""
    import faninsar.missions.alos2
    import faninsar.missions.nisar
    import faninsar.missions.sentinel1  # noqa: F401

    return sorted(_REGISTRY)


def get_mission(name: str) -> type:
    """Return the registered Sensor class for *name*."""
    list_missions()
    if name not in _REGISTRY:
        raise KeyError(f"unknown mission {name!r}; known={list(_REGISTRY)}")
    return _REGISTRY[name]


__all__ = ["Sensor", "get_mission", "list_missions", "register"]
