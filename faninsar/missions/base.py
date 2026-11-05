"""Mission registry: Sensor base class, @register decorator, and bootstrap.

The registry is also the composition point that maps persisted Stack class
names (``stack.json`` ``mission`` values) to concrete mission Stack classes,
so shared Stack code never imports a concrete mission.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar

from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from faninsar.stack.selection import AcquisitionFacts, SelectionDecoder

logger = setup_logger(__name__)

_REGISTRY: dict[str, type] = {}

# Persisted ``stack.json`` mission values are Stack *class names*; they are a
# different namespace from sensor registry keys.  Each entry names the
# module owning that class explicitly; nothing is derived from the name.
_MISSION_STACKS: dict[str, tuple[str, str]] = {
    "S1Stack": ("faninsar.missions.s1.stack", "S1Stack"),
    "NISARStack": ("faninsar.missions.nisar.stack", "NISARStack"),
}


class Sensor:
    """Base mission adapter registered via :func:`register`."""

    name: ClassVar[str] = ""
    # Optional: decode the persisted mission selection dictionary into
    # normalized acquisition facts (PROPOSAL-0072 section 7).  ``None`` means
    # the mission does not support constrained selection.
    decode_selection: ClassVar[SelectionDecoder | None] = None

    @staticmethod
    def select_acquisitions(
        facts: tuple[AcquisitionFacts, ...],
        *,
        track: str | None = None,
        direction: str | None = None,
        pol: str | Sequence[str] | None = None,
        burst: int | Sequence[int] | None = None,
    ) -> tuple[str, ...]:
        """Apply exact group/channel selectors to normalized acquisition facts.

        Missions override this bounded seam only for existing selectors whose
        aliases or meaning cannot be represented by the facts, such as S1
        direction and absolute burst identifiers. Matching stays in the shared
        :func:`faninsar.stack.selection.select_acquisitions` operation.
        """
        from faninsar.stack.selection import select_acquisitions

        if direction is not None or burst is not None:
            message = "this mission does not support direction or burst selection"
            reject(message, NotImplementedError, logger=logger)
        values = (pol,) if isinstance(pol, str) else pol
        return select_acquisitions(facts, group=track, polarizations=values)

    def open_product(self, _uri: str, **_kwargs: Any) -> Any:
        """Open a mission product. Override in subclasses."""
        message = f"{type(self).__name__}.open_product"
        reject(message, NotImplementedError)

    def to_slc_product(self, _handle: Any, **_kwargs: Any) -> Any:
        """Convert an open handle to SLCProduct. Override in subclasses."""
        message = f"{type(self).__name__}.to_slc_product"
        reject(message, NotImplementedError)

    def read_slc_window(
        self,
        _handle: Any,
        _window: Any,
        **_kwargs: Any,
    ) -> Any:
        """Read a complex SLC window. Override in subclasses."""
        message = f"{type(self).__name__}.read_slc_window"
        reject(message, NotImplementedError)


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
            reject(message, ValueError)
        sensor_cls.name = key  # type: ignore[attr-defined]
        _REGISTRY[key] = sensor_cls
        return sensor_cls

    if cls is not None:
        return decorator(cls)
    return decorator


def list_missions() -> list[str]:
    """Return sorted registered mission names."""
    return sorted(_REGISTRY)


def get_mission(name: str) -> type:
    """Return the registered Sensor class for *name*."""
    list_missions()
    if name not in _REGISTRY:
        message = f"unknown mission {name!r}; known={list(_REGISTRY)}"
        reject(message, KeyError)
    return _REGISTRY[name]


def get_mission_stack(class_name: str) -> type:
    """Return the concrete mission Stack class recorded as *class_name*.

    Parameters
    ----------
    class_name : str
        Persisted ``stack.json`` ``mission`` value, e.g. ``"S1Stack"``.

    Returns
    -------
    type
        The mission Stack subclass.

    Raises
    ------
    KeyError
        If *class_name* is not a registered mission Stack.

    """
    entry = _MISSION_STACKS.get(class_name)
    if entry is None:
        message = (
            f"unknown mission Stack {class_name!r}; known={sorted(_MISSION_STACKS)}"
        )
        reject(message, KeyError, logger=logger)
    module_name, attribute = entry
    return getattr(importlib.import_module(module_name), attribute)


__all__ = [
    "Sensor",
    "get_mission",
    "get_mission_stack",
    "list_missions",
    "register",
]
