from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from pyproj import CRS

from faninsar._core.geo.geo_property import ResolutionLike, normalize_resolution
from faninsar.query import BoundingBox


class InSARStackBase:
    """Base class for all InSAR stack types."""

    _crs: CRS
    _res: tuple[float, float]
    _bounds: BoundingBox
    _roi: BoundingBox
    _dates: pd.DatetimeIndex
    _workspace: Path

    def __init__(
        self,
        workspace: Path | str,
        crs: CRS,
        res: tuple[float, float],
        bounds: BoundingBox,
        dates: pd.DatetimeIndex,
        roi: BoundingBox | None = None,
    ) -> None:
        """Initialize the InSAR stack.

        Parameters
        ----------
        crs : CRS
            Coordinate reference system of the stack
        res : tuple[float, float]
            Resolution of the stack in (x, y) direction
        bounds : BoundingBox
            bounds of the stack
        dates : pd.DatetimeIndex
            Dates of the stack
        roi : BoundingBox | None
            Region of interest (ROI) of the stack. If specified, only the data
            within the ROI will be processed. Default is None.
        """
        self._workspace = Path(workspace)
        self._crs = crs
        self._res = res
        self._bounds = bounds
        self._roi = roi
        self._dates = dates

    @property
    def workspace(self) -> Path:
        """The workspace of the stack."""
        return self._workspace

    @property
    def crs(self) -> CRS:
        """The coordinate reference system of the stack."""
        return self._crs

    @property
    def res(self) -> tuple[float, float]:
        """The resolution of the stack in (x, y) direction."""
        return self._res

    @property
    def bounds(self) -> BoundingBox:
        """The bounding box of the stack."""
        return self._bounds

    @property
    def roi(self) -> BoundingBox | None:
        """The region of interest of the stack."""
        return self._roi

    @property
    def dates(self) -> pd.DatetimeIndex:
        """The dates of the stack."""
        return self._dates

    def download_dem(
        self,
        crop: bool = False,
        res: ResolutionLike = None,
        crs: str | None = None,
        dem_type: Literal["SRTM", "GLO"] = "GLO",
        source: Literal["MPC", "NASA"] = "MPC",
        **kwargs,
    ) -> None:
        """Download DEM data for the bounds of the stack.

        Parameters
        ----------
        crop : bool, optional
            Whether to crop the DEM to the roi/bounds of the stack. Default is False.
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
        # Determine the CRS to use
        crs_used = self.crs if crs is None else crs
        crs_dst = CRS.from_user_input(crs_used)

        # Normalize resolution
        if res is None:
            # Use stack's resolution as default
            res_normalized = self.res
        else:
            res_normalized = normalize_resolution(res, crs_dst)

        # Normalize bbox
        if not isinstance(bbox, BoundingBox):
            bbox = BoundingBox(*bbox, crs=crs_dst)

        # TODO: Implement DEM fetching logic
        # This would use res_normalized for fetching the DEM
        raise NotImplementedError("DEM fetching logic not yet implemented")
