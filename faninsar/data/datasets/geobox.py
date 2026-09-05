"""Deprecated: use :mod:`faninsar.data.datasets.geogrid` instead.

This module is kept as a backwards-compatibility shim. It re-exports
``GeoGrid`` and related names from :mod:`faninsar.data.datasets.geogrid`. Importing
``GeoBox`` from here emits a :class:`DeprecationWarning`.

See Also
--------
faninsar.data.datasets.geogrid : Canonical module.

"""

from __future__ import annotations

import warnings

from faninsar.data.datasets.geogrid import (
    GeoBoxTileGrid,
    GeoGrid,
    TileIndex,
)

warnings.warn(
    "`faninsar.data.datasets.geobox` is deprecated; "
    "import from `faninsar.data.datasets.geogrid` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Deprecated alias — keep type-hint compatibility for existing callers.
GeoBox = GeoGrid

__all__ = ["GeoBox", "GeoBoxTileGrid", "GeoGrid", "TileIndex"]
