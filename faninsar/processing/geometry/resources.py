# ruff: noqa: D105, EM101, EM102, TRY003
"""Small, deterministic resource admission helpers for DEM materialization."""

from __future__ import annotations

from dataclasses import dataclass

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


class ResourcePreflightError(ValueError):
    """Raised when a DEM request exceeds its finite admission budget."""


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Finite cell and byte limits checked before provider I/O."""

    max_cells: int = 2**28
    max_output_bytes: int = 2**31
    max_fetch_bytes: int = 2**33
    max_temporary_bytes: int = 2**33

    def __post_init__(self) -> None:
        for name in (
            "max_cells",
            "max_output_bytes",
            "max_fetch_bytes",
            "max_temporary_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ResourcePreflightError(f"{name} must be a positive integer")


def preflight_grid(
    height: int,
    width: int,
    *,
    bytes_per_cell: int = 4,
    budget: ResourceBudget | None = None,
) -> int:
    """Validate output cell/byte counts before network or allocation."""
    active = budget or ResourceBudget()
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ResourcePreflightError("DEM dimensions must be positive integers")
    if type(bytes_per_cell) is not int or bytes_per_cell <= 0:
        raise ResourcePreflightError("bytes_per_cell must be a positive integer")
    cells = height * width
    output = cells * bytes_per_cell
    if cells > active.max_cells:
        raise ResourcePreflightError("DEM cell count exceeds the resource budget")
    if output > active.max_output_bytes:
        raise ResourcePreflightError("DEM output bytes exceed the resource budget")
    return cells


def preflight_transfer(
    size: int | None,
    *,
    budget: ResourceBudget | None = None,
) -> None:
    """Validate a known transfer length, while bounding unknown streams later."""
    active = budget or ResourceBudget()
    if size is not None and (type(size) is not int or size < 0):
        raise ResourcePreflightError("transfer size must be a non-negative integer")
    if size is not None and size > active.max_fetch_bytes:
        raise ResourcePreflightError("DEM transfer bytes exceed the resource budget")


__all__ = [
    "ResourceBudget",
    "ResourcePreflightError",
    "preflight_grid",
    "preflight_transfer",
]
