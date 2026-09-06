"""Public DEM grid contract.

The canonical implementation lives in :mod:`faninsar.processing.geometry.grids`; this
module is a convenient public import location for DEM callers.
"""

from __future__ import annotations

from faninsar.processing.geometry.grids import GridSpec

__all__ = ["GridSpec"]
