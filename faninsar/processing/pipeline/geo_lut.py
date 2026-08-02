"""Build geographic-to-radar lookup tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry import geo2rdr
from faninsar.processing.memory import release_memmap_pages

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.merge.grid import GeoGridSpec

logger = setup_logger(__name__)

__all__ = ["Geo2RdrLUT", "build_geo2rdr_lut", "grid_lonlat", "grid_lonlat_rows"]


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
    height_full : numpy.ndarray, optional
        Per-pixel DEM samples reused by exact geometric flattening.

    """

    az_full: np.ndarray
    rg_full: np.ndarray
    valid: np.ndarray
    full_radar_shape: tuple[int, int]
    height_m: float
    height_full: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the geographic grid shape."""
        return self.valid.shape


def grid_lonlat(grid: GeoGridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return geographic coordinates at destination pixel centers."""
    return grid_lonlat_rows(grid, 0, grid.height)


def grid_lonlat_rows(
    grid: GeoGridSpec,
    row_start: int,
    row_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return geographic pixel-centre coordinates for selected grid rows.

    Parameters
    ----------
    grid : GeoGridSpec
        Destination geographic grid.
    row_start, row_stop : int
        Half-open destination row interval.

    Returns
    -------
    latitude, longitude : tuple[numpy.ndarray, numpy.ndarray]
        Coordinate arrays with shape ``(row_stop - row_start, grid.width)``.

    """
    from pyproj import Transformer

    if row_start < 0 or row_stop > grid.height or row_start >= row_stop:
        reject_invalid_state("invalid geographic grid row interval")
    x0, dx, _, y0, _, dy = grid.transform
    x_coordinates = x0 + dx * (0.5 + np.arange(grid.width, dtype=np.float64))
    y_coordinates = y0 + dy * (0.5 + np.arange(row_start, row_stop, dtype=np.float64))
    x, y = np.meshgrid(x_coordinates, y_coordinates)
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
    chunk_size: int = 128,
    storage_dir: str | Path | None = None,
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
    storage_dir : str or pathlib.Path, optional
        Directory for disk-backed LUT arrays. In-memory arrays are used when
        omitted.

    Returns
    -------
    Geo2RdrLUT
        Full-resolution radar coordinates on the destination grid.

    """
    from pathlib import Path

    full_height, full_width = full_radar_shape
    if storage_dir is None:
        azimuth = np.full(grid.shape, np.nan, dtype=np.float64)
        range_index = np.full(grid.shape, np.nan, dtype=np.float64)
        valid = np.zeros(grid.shape, dtype=bool)
        height_lookup = np.full(grid.shape, np.nan, dtype=np.float64)
    else:
        directory = Path(storage_dir)
        directory.mkdir(parents=True, exist_ok=True)
        azimuth = np.memmap(
            directory / "reference_azimuth.float64",
            mode="w+",
            dtype=np.float64,
            shape=grid.shape,
        )
        range_index = np.memmap(
            directory / "reference_range.float64",
            mode="w+",
            dtype=np.float64,
            shape=grid.shape,
        )
        valid = np.memmap(
            directory / "reference_valid.bool",
            mode="w+",
            dtype=np.bool_,
            shape=grid.shape,
        )
        height_lookup = np.memmap(
            directory / "height.float64",
            mode="w+",
            dtype=np.float64,
            shape=grid.shape,
        )

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
        latitude_chunk, longitude_chunk = grid_lonlat_rows(
            grid,
            row_start,
            row_stop,
        )
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
        height_lookup[row_start:row_stop] = height_chunk
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
        for array in (azimuth, range_index, valid, height_lookup):
            if isinstance(array, np.memmap):
                release_memmap_pages(array)

    if isinstance(azimuth, np.memmap):
        azimuth.flush()
    if isinstance(range_index, np.memmap):
        range_index.flush()
    if isinstance(valid, np.memmap):
        valid.flush()
    if isinstance(height_lookup, np.memmap):
        height_lookup.flush()
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
        height_full=height_lookup,
    )
