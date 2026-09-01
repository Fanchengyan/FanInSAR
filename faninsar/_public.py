"""Curated public surface for ``import faninsar as fis``.

Hard cap: ``len(__all__) <= 20``. Ports/compute internals are never exported.
"""

from __future__ import annotations

# Domain nouns from core / _core.sar
from faninsar.core import (
    Acquisition,
    Acquisitions,
    Interferogram,
    Pair,
    Pairs,
)

# Stack and Network are exposed lazily from ``faninsar.__init__`` because their
# Dataset-backed implementations import optional data-layer modules during
# package initialization.
__all__ = [
    "Acquisition",
    "Acquisitions",
    "Interferogram",
    "Pair",
    "Pairs",
]
