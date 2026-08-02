"""Stage Protocol — plain callables with optional metadata."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from faninsar.core.physical import PhysicalType


@runtime_checkable
class Stage(Protocol):
    """Callable stage: ``(state, **kwargs) -> state``."""

    def __call__(self, state: Any, /, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class StageNode:
    """Optional metadata wrapper for Workflow lattice checks."""

    fn: Callable[..., Any]
    name: str
    input_type: PhysicalType | None = None
    output_type: PhysicalType | None = None
    op: str = "seq"  # seq | par | map_over | reduce


__all__ = ["Stage", "StageNode"]
