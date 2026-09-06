"""Mission registry: Sensor base class and @register decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

_REGISTRY: dict[str, type] = {}


class Sensor:
    """Base mission adapter registered via :func:`register`."""

    name: ClassVar[str] = ""

    def open_product(self, _uri: str, **_kwargs: Any) -> Any:
        """Open a mission product. Override in subclasses."""
        message = f"{type(self).__name__}.open_product"
        logger.error(message)
        raise NotImplementedError(message)

    def to_slc_product(self, _handle: Any, **_kwargs: Any) -> Any:
        """Convert an open handle to SLCProduct. Override in subclasses."""
        message = f"{type(self).__name__}.to_slc_product"
        logger.error(message)
        raise NotImplementedError(message)

    def read_slc_window(
        self,
        _handle: Any,
        _window: Any,
        **_kwargs: Any,
    ) -> Any:
        """Read a complex SLC window. Override in subclasses."""
        message = f"{type(self).__name__}.read_slc_window"
        logger.error(message)
        raise NotImplementedError(message)


def register(
    cls: type | None = None,
    *,
    name: str | None = None,
) -> type | Callable[[type], type]:
    """Register a Sensor subclass in the global mission registry."""

    def decorator(sensor_cls: type) -> type:
        key = name or getattr(sensor_cls, "name", None) or sensor_cls.__name__.lower()
        if not key:
            message = "mission register requires a non-empty name"
            logger.error(message)
            raise ValueError(message)
        sensor_cls.name = key  # type: ignore[attr-defined]
        _REGISTRY[key] = sensor_cls
        return sensor_cls

    if cls is not None:
        return decorator(cls)
    return decorator


def list_missions() -> list[str]:
    """Return sorted registered mission names."""
    import faninsar.missions.nisar
    import faninsar.missions.s1  # noqa: F401

    return sorted(_REGISTRY)


def get_mission(name: str) -> type:
    """Return the registered Sensor class for *name*."""
    list_missions()
    if name not in _REGISTRY:
        message = f"unknown mission {name!r}; known={list(_REGISTRY)}"
        logger.error(message)
        raise KeyError(message)
    return _REGISTRY[name]


__all__ = ["Sensor", "get_mission", "list_missions", "register"]
