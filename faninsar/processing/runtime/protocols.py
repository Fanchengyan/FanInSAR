"""Compute backend port — device/array execution boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


@runtime_checkable
class ComputeBackend(Protocol):
    """Three-method compute port used by processing stages.

    Day-1 surface intentionally excludes session/GPU context managers.
    """

    name: str

    def to_device_array(self, x: np.ndarray) -> Any:
        """Move or wrap a NumPy array for backend-local computation."""
        ...

    def map_blocks(
        self,
        fn: Callable[..., Any],
        *args: Any,
        chunks: tuple[int, ...] | None = None,
        dtype: np.dtype | type | None = None,
        resources: dict[str, int] | None = None,
    ) -> Any:
        """Apply *fn* over blocks of *args*, optionally with resource tags."""
        ...

    def compute(
        self,
        *arrays: Any,
        sync: bool = True,
    ) -> tuple[np.ndarray, ...]:
        """Materialize backend arrays to NumPy (sync by default)."""
        ...


__all__ = ["ComputeBackend"]
