"""Stack session API (PROPOSAL-0017)."""

from __future__ import annotations

from faninsar.processing.stack.catalog import SceneCatalog
from faninsar.processing.stack.config import CoregMode, EsdMethod, StackConfig
from faninsar.processing.stack.session import Stack

__all__ = [
    "CoregMode",
    "EsdMethod",
    "SceneCatalog",
    "Stack",
    "StackConfig",
]
