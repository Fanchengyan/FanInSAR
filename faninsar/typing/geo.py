"""Typing for geospatial data."""

from __future__ import annotations

from typing import Literal

from pyproj.crs.crs import CRS as PyprojCRS  # noqa: N811
from rasterio.crs import CRS as RasterioCRS  # noqa: N811
from rasterio.enums import Resampling as RasterioResampling

CrsLike = PyprojCRS | RasterioCRS | tuple[str, str] | dict[str, str] | str | int
ResamplingLike = (
    Literal[
        "nearest",
        "average",
        "bilinear",
        "cubic",
        "cubic_spline",
        "lanczos",
        "mode",
        "gauss",
        "max",
        "min",
        "med",
        "q1",
        "q3",
        "sum",
        "rms",
    ]
    | RasterioResampling
)
