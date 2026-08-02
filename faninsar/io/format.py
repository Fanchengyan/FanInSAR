"""Serialization format tags."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """Named serialization format."""

    name: str


ZARR = FormatSpec("zarr")
COG = FormatSpec("cog")
GEOTIFF = FormatSpec("geotiff")


__all__ = ["COG", "GEOTIFF", "ZARR", "FormatSpec"]
