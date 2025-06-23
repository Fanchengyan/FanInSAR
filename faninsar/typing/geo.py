"""Typing for geospatial data."""

from __future__ import annotations

from pyproj.crs.crs import CRS as PyprojCRS  # noqa: N811
from rasterio.crs import CRS as RasterioCRS  # noqa: N811

CrsLike = PyprojCRS | RasterioCRS | tuple[str, str] | dict[str, str] | str | int
