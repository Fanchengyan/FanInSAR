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
from faninsar.io.readers import open_stac, open_zarr
from faninsar.missions.base import list_missions, register
from faninsar.processing.contracts.stage import Stage
from faninsar.timeseries.invert import NSBAS, SBAS, invert

__all__ = [
    "NSBAS",
    "SBAS",
    "Acquisition",
    "Acquisitions",
    "Interferogram",
    "Pair",
    "Pairs",
    "Stage",
    "invert",
    "list_missions",
    "open_stac",
    "open_zarr",
    "register",
]
