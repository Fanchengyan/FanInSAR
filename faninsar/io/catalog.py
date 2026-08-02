"""STAC catalog helpers at product boundaries."""

from __future__ import annotations

from typing import Any


class MemoryCatalog:
    """In-memory STAC item collection for tests and boundary emission."""

    def __init__(self) -> None:
        self.items: list[Any] = []

    def add_item(self, item: Any) -> None:
        """Register a STAC Item."""
        self.items.append(item)


__all__ = ["MemoryCatalog"]
