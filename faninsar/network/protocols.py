"""Protocols for explicit Network readers.

The reader boundary deliberately contains one operation.  Canonical Network
products and external formats can therefore share dispatch without sharing
their storage or scientific interpretation code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from .network import Network


@runtime_checkable
class NetworkReader(Protocol):
    """Read one Network from a path or reader-specific source."""

    def read(self, path: str | Path) -> Network:
        """Read and return a Network view."""


__all__ = ["NetworkReader"]
