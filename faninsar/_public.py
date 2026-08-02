"""Curated public surface for ``import faninsar as fis``.

Hard cap: ``len(__all__) <= 20``. Ports/compute internals are never exported.
"""

from __future__ import annotations

# Domain nouns from core / _core.sar
from faninsar.core import (
    Acquisition,
    Baselines,
    Loop,
    Loops,
    Pair,
    Pairs,
)

# Frame + Pipeline/Stage aliases
from faninsar.core.frame import Frame
from faninsar.core.physical import PhysicalType
from faninsar.io.readers import open_stac, open_zarr
from faninsar.missions.base import list_missions, register
from faninsar.processing.contracts.stage import Stage
from faninsar.processing.workflow import Workflow as Pipeline
from faninsar.run import run
from faninsar.timeseries.invert import NSBAS, SBAS, invert

__all__ = [
    "NSBAS",
    "SBAS",
    "Acquisition",
    "Baselines",
    "Frame",
    "Loop",
    "Loops",
    "Pair",
    "Pairs",
    "Pipeline",
    "Stage",
    "invert",
    "list_missions",
    "open_stac",
    "open_zarr",
    "register",
    "run",
]

# PhysicalType is available but not counted in the Artisan root 14 — keep under 20.
# Intentionally omitted from __all__ to leave room; import from faninsar.core if needed.
del PhysicalType
