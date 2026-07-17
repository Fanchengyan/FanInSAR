"""DEM sampling protocol and constant-height / raster DEM adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rasterio.io import DatasetReader

logger = setup_logger(__name__)


@runtime_checkable
class DEMSampler(Protocol):
    """Sample ellipsoidal height at geodetic coordinates."""

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return height in metres for each latitude/longitude sample."""


@dataclass(frozen=True, slots=True)
class ConstantHeightDEM:
    """DEM that returns a constant ellipsoidal height everywhere."""

    height_m: float = 0.0

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return a constant height array matching the coordinate shape."""
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        lat_b, _ = np.broadcast_arrays(lat, lon)
        return np.full(lat_b.shape, self.height_m, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GeoidAdjustedDEM:
    """Convert orthometric DEM samples to ellipsoidal heights."""

    orthometric_dem: DEMSampler
    geoid: DEMSampler

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return orthometric height plus geoid undulation."""
        orthometric = self.orthometric_dem.sample(latitude_deg, longitude_deg)
        undulation = self.geoid.sample(latitude_deg, longitude_deg)
        return np.asarray(orthometric) + np.asarray(undulation)


@dataclass(slots=True)
class NetCDFGeoid:
    """Sample geoid undulation from a regular NetCDF longitude/latitude grid."""

    path: Path
    _interpolator: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Validate that the geoid grid exists."""
        if not self.path.exists():
            message = f"geoid grid does not exist: {self.path}"
            logger.error(message)
            raise FileNotFoundError(message)

    def _open(self) -> Callable[[np.ndarray], np.ndarray]:
        if self._interpolator is None:
            import xarray as xr
            from scipy.interpolate import RegularGridInterpolator

            grid = xr.open_dataarray(self.path)
            latitude_name = "lat" if "lat" in grid.coords else "y"
            longitude_name = "lon" if "lon" in grid.coords else "x"
            latitude = np.asarray(grid[latitude_name], dtype=np.float64)
            longitude = np.asarray(grid[longitude_name], dtype=np.float64)
            values = np.asarray(grid, dtype=np.float64)
            if latitude[0] > latitude[-1]:
                latitude = latitude[::-1]
                values = values[::-1]
            self._interpolator = RegularGridInterpolator(
                (latitude, longitude),
                values,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        return self._interpolator

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Bilinear-sample geoid undulation in metres."""
        latitude, longitude = np.broadcast_arrays(
            np.asarray(latitude_deg, dtype=np.float64),
            np.asarray(longitude_deg, dtype=np.float64),
        )
        interpolator = self._open()
        points = np.column_stack([latitude.ravel(), longitude.ravel()])
        values = interpolator(points)
        return np.asarray(values, dtype=np.float64).reshape(latitude.shape)


@dataclass(slots=True)
class RasterDEM:
    """Sample heights from a single-band georeferenced DEM raster."""

    path: Path
    nodata: float | None = None
    _dataset: DatasetReader | None = None

    def __post_init__(self) -> None:
        """Validate that the DEM path exists."""
        if not self.path.exists():
            message = f"DEM path does not exist: {self.path}"
            logger.error(message)
            raise FileNotFoundError(message)

    def _open(self) -> DatasetReader:
        if self._dataset is None:
            import rasterio

            dataset = rasterio.open(self.path)
            self._dataset = dataset
            if self.nodata is None:
                self.nodata = dataset.nodata
        dataset = self._dataset
        if dataset is None:
            message = f"failed to open DEM dataset: {self.path}"
            logger.error(message)
            raise RuntimeError(message)
        return dataset

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Bilinear-sample DEM heights at lon/lat coordinates.

        DEM height is a real-valued smooth field, so bilinear interpolation
        is the appropriate kernel. The previous implementation used
        :meth:`rasterio.DatasetReader.sample`, which is nearest-neighbour —
        a doc/behaviour mismatch that also needlessly quantised the height
        field. This implementation converts the requested lon/lat to source
        pixel coordinates via the raster transform and performs an explicit
        bilinear interpolation, returning NaN for out-of-bounds samples.

        Parameters
        ----------
        latitude_deg, longitude_deg : numpy.ndarray
            Geodetic sample coordinates in degrees.

        Returns
        -------
        numpy.ndarray
            Sampled heights in metres. Out-of-bounds samples are NaN.

        """
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        dataset = self._open()
        band = dataset.read(1)
        nodata = self.nodata
        if nodata is None:
            nodata = dataset.nodata

        # Convert source nodata to NaN for the interpolation arithmetic.
        h_arr = band.astype(np.float64, copy=True)
        if nodata is not None:
            h_arr = np.where(np.isclose(h_arr, nodata), np.nan, h_arr)

        transform = dataset.transform
        inv = ~transform
        # lon/lat -> CRS pixel coordinates. We assume the DEM is already in a
        # CRS matching the requested lon/lat (geographic). If the dataset is
        # projected, callers should reproject their coordinates first.
        xs = lon_b.ravel()
        ys = lat_b.ravel()
        # transform.invert expects (col, row) order via (x, y) -> (col, row).
        cols, rows = inv * (xs, ys)
        cols = np.asarray(cols, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)

        r0 = np.floor(rows).astype(np.int64)
        c0 = np.floor(cols).astype(np.int64)
        dr = rows - r0
        dc = cols - c0
        h, w = h_arr.shape
        in_bounds = (r0 >= 0) & (r0 <= h - 2) & (c0 >= 0) & (c0 <= w - 2)

        heights = np.full(xs.shape, np.nan, dtype=np.float64)
        if in_bounds.any():
            idx = np.where(in_bounds)[0]
            r0i = r0[idx]
            c0i = c0[idx]
            dri = dr[idx]
            dci = dc[idx]
            h00 = h_arr[r0i, c0i]
            h01 = h_arr[r0i, c0i + 1]
            h10 = h_arr[r0i + 1, c0i]
            h11 = h_arr[r0i + 1, c0i + 1]
            top = h00 * (1.0 - dci) + h01 * dci
            bot = h10 * (1.0 - dci) + h11 * dci
            vals = top * (1.0 - dri) + bot * dri
            heights[idx] = vals
        return heights.reshape(lat_b.shape)

    def close(self) -> None:
        """Close the underlying raster dataset if open."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
