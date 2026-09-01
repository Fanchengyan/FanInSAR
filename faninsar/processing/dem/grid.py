"""Public DEM grid contract.

The canonical implementation lives in :mod:`faninsar._core.geo.grids`; this
module is a convenient public import location for DEM callers.
"""

from __future__ import annotations

from faninsar._core.geo.grids import GridSpec

__all__ = ["GridSpec"]
