from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from pyproj import CRS

from faninsar._core.geo.geo_property import ResolutionLike, normalize_resolution
from faninsar.query import BoundingBox


def download_dem(
    path: Path,
    bbox: BoundingBox | Iterable[float],
    res: ResolutionLike = None,
    crs: str | None = None,
    dem_type: Literal["SRTM", "GLO"] = "GLO",
    source: Literal["MPC", "NASA"] = "MPC",
    **kwargs,
) -> None:
    """Download DEM data for the bounds of the stack.

    Parameters
    ----------
    path : Path
        Path of the DEM data to be downloaded
    bbox : BoundingBox | Iterable[float]
        Bounding box for the DEM
    res : ResolutionLike, optional
        Resolution of the DEM. Can be:
        - None: use stack's default resolution
        - float/int: symmetric resolution in CRS units
        - (float, float): (x_res, y_res) in CRS units
        - Resolution: explicit resolution with units
        - pint.Quantity: resolution with units from pint
    crs : str | None, optional
        Coordinate reference system. If None, use stack's CRS
    dem_type : Literal["SRTM", "GLO"], optional
        Type of DEM to fetch
    source : Literal["MPC", "NASA"], optional
        Source of DEM data. Default is "MPC"

        - "MPC": Microsoft Planetary Computer
        - "NASA": NASA EarthDATA
    **kwargs
        Additional keyword arguments
    """
    crs_dst = CRS.from_user_input(crs)
    res_normalized = normalize_resolution(res, crs_dst)
    if not isinstance(bbox, BoundingBox):
        bbox = BoundingBox(*bbox, crs=crs_dst)

    # TODO: Implement DEM fetching logic
    # This would use res_normalized for fetching the DEM
    raise NotImplementedError("DEM fetching logic not yet implemented")


def download_orbit(dates: pd.DatetimeIndex | None = None) -> None:
    """Download orbit data for the given dates."""
    pass
