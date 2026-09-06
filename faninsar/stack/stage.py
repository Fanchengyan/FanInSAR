"""Stage Protocol — plain callables with optional metadata."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Stage(Protocol):
    """Callable stage: ``(state, **kwargs) -> state``."""

    def __call__(self, state: Any, /, **kwargs: Any) -> Any:
        """Transform one processing state into the next state."""
        ...


__all__ = ["Stage"]
