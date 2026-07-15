"""Geocode radar-coordinate layers with rdr2geo + DEM sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry import (
    ConstantHeightDEM,
    RadarGeometryModel,
    rdr2geo_with_dem,
)

if TYPE_CHECKING:
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.sentinel1.types import S1Burst, S1Swath

logger = setup_logger(__name__)

SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True, slots=True)
class GeocodedLayer:
    """Geocoded scalar field with geographic coordinates."""

    values: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    height_m: np.ndarray
    converged: np.ndarray


def radar_geometry_from_swath(
    swath: S1Swath,
    burst: S1Burst,
    *,
    window_shape: tuple[int, int],
    row0: int,
    col0: int,
) -> RadarGeometryModel:
    """Build a radar geometry model for a burst window.

    Parameters
    ----------
    swath : S1Swath
        Annotation swath providing orbit and spacings.
    burst : S1Burst
        Burst providing the sensing start for line 0 of the burst.
    window_shape : tuple[int, int]
        ``(height, width)`` of the window being geocoded.
    row0, col0 : int
        Absolute measurement-raster origin of the window.

    Returns
    -------
    RadarGeometryModel
        Geometry model whose index (0, 0) is the window origin.

    """
    height, width = window_shape
    # Absolute azimuth index of window origin within the full swath raster.
    local_line = row0 - burst.index * swath.lines_per_burst
    sensing_start = burst.azimuth_time
    # Shift sensing_start so that window row 0 corresponds to absolute line row0.
    from datetime import timedelta

    sensing_start = burst.azimuth_time + timedelta(
        seconds=float(local_line) * swath.azimuth_time_interval_s
    )
    starting_slant_range_m = (
        swath.slant_range_time_s * SPEED_OF_LIGHT_M_S / 2.0
        + float(col0) * swath.range_pixel_spacing_m
    )
    wavelength_m = SPEED_OF_LIGHT_M_S / swath.radar_frequency_hz
    grid = RadarGrid(
        shape=(height, width),
        starting_slant_range_m=starting_slant_range_m,
        range_spacing_m=swath.range_pixel_spacing_m,
        sensing_start=sensing_start,
        azimuth_time_interval_s=swath.azimuth_time_interval_s,
        wavelength_m=wavelength_m,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(grid, swath.orbit)


def geocode_layer(
    values: np.ndarray,
    *,
    swath: S1Swath,
    burst: S1Burst,
    row0: int,
    col0: int,
    dem: DEMSampler | None = None,
    stride: int = 1,
) -> GeocodedLayer:
    """Geocode a radar-coordinate layer to lon/lat samples.

    Parameters
    ----------
    values : numpy.ndarray
        2-D radar-coordinate field (phase, coherence, ...).
    swath, burst : S1Swath, S1Burst
        Geometry metadata for the window.
    row0, col0 : int
        Window origin in the measurement raster.
    dem : DEMSampler, optional
        Height sampler. Defaults to zero-height ellipsoid.
    stride : int, optional
        Subsample factor for the rdr2geo solve (1 = every pixel).

    Returns
    -------
    GeocodedLayer
        Values with geographic coordinates and convergence mask.

    """
    if values.ndim != 2:
        reject_invalid_state("geocode_layer requires a 2-D array")
    if stride < 1:
        reject_invalid_state("stride must be >= 1")

    height, width = values.shape
    model = radar_geometry_from_swath(
        swath,
        burst,
        window_shape=(height, width),
        row0=row0,
        col0=col0,
    )
    az = np.arange(0, height, stride, dtype=np.float64)
    rg = np.arange(0, width, stride, dtype=np.float64)
    az_grid, rg_grid = np.meshgrid(az, rg, indexing="ij")
    dem_sampler = dem if dem is not None else ConstantHeightDEM(0.0)
    transform = rdr2geo_with_dem(model, az_grid, rg_grid, dem_sampler)

    # Nearest-neighbour gather of values at solved samples.
    src_az = np.clip(np.rint(az_grid).astype(np.int64), 0, height - 1)
    src_rg = np.clip(np.rint(rg_grid).astype(np.int64), 0, width - 1)
    sampled = values[src_az, src_rg].astype(np.float32, copy=False)
    sampled = np.where(transform.converged, sampled, np.nan)

    n_conv = int(np.count_nonzero(transform.converged))
    logger.info(
        "Geocoded layer %s with stride=%s (%s/%s converged)",
        values.shape,
        stride,
        n_conv,
        transform.converged.size,
    )
    return GeocodedLayer(
        values=sampled,
        latitude_deg=transform.latitude_deg,
        longitude_deg=transform.longitude_deg,
        height_m=transform.height_m,
        converged=transform.converged,
    )
