"""Typing for geospatial data."""

from __future__ import annotations

from typing import Union

from pyproj.crs.crs import CRS as PyprojCRS  # noqa: N811
from rasterio.crs import CRS as RasterioCRS  # noqa: N811

CrsLike = Union[
    PyprojCRS,
    RasterioCRS,
    tuple[str, str],  # ("auth_name": "auth_code") [i.e ('epsg', '4326')]
    dict[str, str],
    str,
    int,
]
