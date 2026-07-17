"""Build geographic-to-radar lookup tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry import geo2rdr

if TYPE_CHECKING:
    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.merge.grid import GeoGridSpec

logger = setup_logger(__name__)

__all__ = ["Geo2RdrLUT", "build_geo2rdr_lut"]


@dataclass(frozen=True, slots=True)
class Geo2RdrLUT:
    """Full-resolution radar indices sampled on a geographic grid.

    Attributes
    ----------
    az_full, rg_full : numpy.ndarray
        Full-resolution azimuth and range indices.
    valid : numpy.ndarray
        Mask of converged indices inside the radar image.
    full_radar_shape : tuple[int, int]
        Shape of the full-resolution radar image.
    height_m : float
        Mean ellipsoidal height used to build the lookup table.

    """

    az_full: np.ndarray
    rg_full: np.ndarray
    valid: np.ndarray
    full_radar_shape: tuple[int, int]
    height_m: float

    @property
    def shape(self) -> tuple[int, int]:
        """Return the geographic grid shape."""
        return self.valid.shape


def grid_lonlat(grid: GeoGridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return geographic coordinates at destination pixel centers."""
    from pyproj import Transformer

    x, y = grid.xy_pixel_centers()
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(x.ravel(), y.ravel())
    return (
        np.asarray(latitude, dtype=np.float64).reshape(x.shape),
        np.asarray(longitude, dtype=np.float64).reshape(x.shape),
    )


def build_geo2rdr_lut(
    *,
    geometry: RadarGeometryModel,
    grid: GeoGridSpec,
    full_radar_shape: tuple[int, int],
    height_m: float | np.ndarray = 0.0,
    dem: DEMSampler | None = None,
    chunk_size: int = 40,
) -> Geo2RdrLUT:
    """Build a reusable geographic-to-radar lookup table.

    Parameters
    ----------
    geometry : RadarGeometryModel
        Reference-scene radar geometry.
    grid : GeoGridSpec
        Destination geographic grid.
    full_radar_shape : tuple[int, int]
        Full-resolution radar image shape.
    height_m : float or numpy.ndarray, optional
        Fallback ellipsoidal height.
    dem : DEMSampler, optional
        Per-pixel ellipsoidal height source.
    chunk_size : int, optional
        Destination rows processed per geometry call.

    Returns
    -------
    Geo2RdrLUT
        Full-resolution radar coordinates on the destination grid.

    """
    full_height, full_width = full_radar_shape
    latitude, longitude = grid_lonlat(grid)
    azimuth = np.full(grid.shape, np.nan, dtype=np.float64)
    range_index = np.full(grid.shape, np.nan, dtype=np.float64)
    valid = np.zeros(grid.shape, dtype=bool)

    fallback_height = (
        float(height_m) if np.isscalar(height_m) else float(np.nanmean(height_m))
    )
    height_array = None if np.isscalar(height_m) else np.asarray(height_m)
    if height_array is not None and height_array.shape != grid.shape:
        reject_invalid_state(
            f"height_m shape {height_array.shape} does not match grid {grid.shape}"
        )

    mean_heights: list[float] = []
    for row_start in range(0, grid.height, chunk_size):
        row_stop = min(row_start + chunk_size, grid.height)
        latitude_chunk = np.ascontiguousarray(latitude[row_start:row_stop])
        longitude_chunk = np.ascontiguousarray(longitude[row_start:row_stop])
        finite_geo = np.isfinite(latitude_chunk) & np.isfinite(longitude_chunk)
        safe_latitude = np.where(finite_geo, latitude_chunk, 0.0)
        safe_longitude = np.where(finite_geo, longitude_chunk, 0.0)
        if dem is not None:
            sampled_height = np.asarray(
                dem.sample(safe_latitude, safe_longitude),
                dtype=np.float64,
            )
            height_chunk = np.where(
                np.isfinite(sampled_height),
                sampled_height,
                fallback_height,
            )
        elif height_array is not None:
            selected_height = np.asarray(
                height_array[row_start:row_stop],
                dtype=np.float64,
            )
            height_chunk = np.where(
                np.isfinite(selected_height),
                selected_height,
                fallback_height,
            )
        else:
            height_chunk = fallback_height
        if not np.isscalar(height_chunk):
            mean_heights.append(float(np.nanmean(height_chunk)))

        result = geo2rdr(
            geometry,
            safe_latitude,
            safe_longitude,
            height_chunk,
        )
        chunk_valid = (
            finite_geo
            & result.converged
            & np.isfinite(result.azimuth_index)
            & np.isfinite(result.range_index)
            & (result.azimuth_index >= 0.0)
            & (result.azimuth_index <= full_height - 1.0)
            & (result.range_index >= 0.0)
            & (result.range_index <= full_width - 1.0)
        )
        azimuth[row_start:row_stop] = np.where(
            chunk_valid,
            result.azimuth_index,
            np.nan,
        )
        range_index[row_start:row_stop] = np.where(
            chunk_valid,
            result.range_index,
            np.nan,
        )
        valid[row_start:row_stop] = chunk_valid

    mean_height = float(np.mean(mean_heights)) if mean_heights else fallback_height
    logger.info(
        "Built geo2rdr LUT: %d/%d valid, radar_shape=%s, mean_height=%.1f m",
        int(valid.sum()),
        valid.size,
        full_radar_shape,
        mean_height,
    )
    return Geo2RdrLUT(
        az_full=azimuth,
        rg_full=range_index,
        valid=valid,
        full_radar_shape=(int(full_height), int(full_width)),
        height_m=mean_height,
    )
