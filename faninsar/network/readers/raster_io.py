"""Backwards-compatible imports for the raster helpers.

The implementations moved to :mod:`faninsar.io.raster`, which owns raster
writing, COG validation, grid reprojection, and alignment validation. This
module is a re-export shim so existing imports keep working; new code should
import from ``faninsar.io.raster`` or ``faninsar.io``.

"""

from __future__ import annotations

from faninsar.io.raster import (
    RasterGridSpec,
    read_geobox,
    read_geogrid,
    read_profile,
    reproject_phase_to_geogrid,
    reproject_to_geogrid,
    validate_alignment,
    validate_cog,
    write_cog,
)

__all__ = [
    "RasterGridSpec",
    "read_geobox",
    "read_geogrid",
    "read_profile",
    "reproject_phase_to_geogrid",
    "reproject_to_geogrid",
    "validate_alignment",
    "validate_cog",
    "write_cog",
]
