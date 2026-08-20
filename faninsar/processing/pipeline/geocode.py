"""Geocode radar-coordinate layers with rdr2geo + DEM sampling.

This is **Path A** (forward geocode) in the terminology of the
``sar-resampling-kernels`` skill: for each output pixel we invert the radar
geometry with ``run_rdr2geo`` to get fractional radar ``(az, rg)``
coordinates, then resample the radar image at those coordinates.

Resampling kernel is chosen from the array's physical type — **never** by
nearest-gather. ``radar[round(az), round(rg)]`` is a bug for every continuous
quantity (complex, real smooth, phase): it aliases and staircases edges and
drops ~30% of the spatial information. Only hard integer-class labels may use
nearest. See the skill's "Geocoding — the two implementation paths" section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import map_coordinates

from faninsar.logging import setup_logger
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry import ConstantHeightDEM, RadarGeometryModel
from faninsar.processing.geometry.prepare_production import run_rdr2geo

if TYPE_CHECKING:
    from faninsar.missions.sentinel1.types import S1Burst, S1Swath
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.typing import DeviceLike

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


def _resample_at_radar_coords(
    values: np.ndarray,
    az_grid: np.ndarray,
    rg_grid: np.ndarray,
    converged: np.ndarray,
) -> np.ndarray:
    """Resample ``values`` at fractional radar coordinates from ``geo2rdr``.

    Kernel is selected from the array's physical type so that the forward
    geocode preserves phase / smoothness instead of nearest-gather aliasing:

    * **complex** (SLC, wrapped ifg) → Lanczos a=4 (windowed sinc). Bilinear on
      real/imag attenuates in-band signal and leaks aliasing, producing a
      sub-pixel-offset-dependent phase bias.
    * **real, smooth** (coherence, unwrapped phase, DEM, amplitude) →
      ``scipy.map_coordinates(order=1)`` (bilinear). These are already smooth
      or already-estimated fields; bilinear is appropriate and cheap.
    * **integer / hard labels** (conncomp, layover, water masks) →
      ``map_coordinates(order=0)`` (nearest). Integer classes must not be
      interpolated — bilinear invents invalid classes.

    Out-of-bounds and non-converged pixels are set to NaN (or 0 for integer
    labels).

    Parameters
    ----------
    values : numpy.ndarray
        2-D radar-coordinate field.
    az_grid, rg_grid : numpy.ndarray
        Fractional radar coordinates on the output grid (same shape).
    converged : numpy.ndarray
        Bool mask of geo2rdr convergence, same shape as the grids.

    Returns
    -------
    numpy.ndarray
        Resampled values on the output grid, NaN where not converged
        (0 for integer labels).

    """
    height, width = values.shape
    out_shape = az_grid.shape
    is_complex = np.iscomplexobj(values)
    is_integer = np.issubdtype(values.dtype, np.integer)

    # Clip to valid radar bounds so map_coordinates does not index outside.
    az_clamped = np.clip(az_grid, 0.0, height - 1.0)
    rg_clamped = np.clip(rg_grid, 0.0, width - 1.0)

    if is_integer:
        # Hard labels — nearest, never interpolate classes.
        out = np.zeros(out_shape, dtype=values.dtype)
        coords = np.array([az_clamped.ravel(), rg_clamped.ravel()])
        sampled = map_coordinates(
            values.astype(np.float32), coords, order=0,
            mode="constant", cval=0.0,
        ).reshape(out_shape).astype(values.dtype)
        out[converged] = sampled[converged]
        return out

    if is_complex:
        # Complex SLC / wrapped ifg → Lanczos a=4 to preserve phase.
        # Import lazily to avoid a torch dependency for the real-field path.
        from faninsar.processing.resampling import lanczos_resample

        out = np.full(out_shape, np.nan + 1j * np.nan, dtype=np.complex64)
        valid = converged & np.isfinite(az_clamped) & np.isfinite(rg_clamped)
        if not np.any(valid):
            return out
        coords = np.array(
            [az_clamped[valid], rg_clamped[valid]], dtype=np.float64
        )
        sampled = lanczos_resample(
            values, coords, a=4, mode="constant", cval=0.0,
        )
        out[valid] = np.asarray(sampled, dtype=np.complex64)
        return out

    # Real, smooth field (coherence, unwrapped phase, DEM, amplitude) → bilinear.
    out = np.full(out_shape, np.nan, dtype=np.float32)
    valid = converged & np.isfinite(az_clamped) & np.isfinite(rg_clamped)
    if not np.any(valid):
        return out
    coords = np.array([az_clamped[valid], rg_clamped[valid]], dtype=np.float64)
    sampled = map_coordinates(
        np.asarray(values, dtype=np.float32), coords, order=1,
        mode="constant", cval=0.0,
    )
    out[valid] = sampled.astype(np.float32)
    return out


def geocode_layer(
    values: np.ndarray,
    *,
    swath: S1Swath,
    burst: S1Burst,
    row0: int,
    col0: int,
    dem: DEMSampler | None = None,
    device: DeviceLike,
    stride: int = 1,
) -> GeocodedLayer:
    """Geocode a radar-coordinate layer to lon/lat samples.

    Forward geocode (Path A): for each output pixel, invert the radar geometry
    with :func:`~faninsar.processing.geometry.prepare_production.run_rdr2geo`
    to get fractional radar ``(az, rg)``
    coordinates, then resample ``values`` at those coordinates with a kernel
    matched to the array's physical type — Lanczos a=4 for complex,
    bilinear for real smooth, nearest for hard integer labels. Nearest-gather
    is intentionally **not** used: it aliases every continuous quantity.

    Parameters
    ----------
    values : numpy.ndarray
        2-D radar-coordinate field (complex ifg, phase, coherence, ...).
    swath, burst : S1Swath, S1Burst
        Geometry metadata for the window.
    row0, col0 : int
        Window origin in the measurement raster.
    dem : DEMSampler, optional
        Height sampler. Defaults to zero-height ellipsoid.
    device : DeviceLike
        Required production device (``auto`` resolves to cpu or cuda).
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
    transform = run_rdr2geo(model, az_grid, rg_grid, dem_sampler, device=device)

    # Resample values at the fractional radar coordinates returned by geo2rdr.
    # The kernel is chosen inside _resample_at_radar_coords by physical type:
    # Lanczos for complex, bilinear for real smooth, nearest for hard labels.
    # Never nearest-gather continuous fields — it aliases and staircases.
    sampled = _resample_at_radar_coords(
        values, az_grid, rg_grid, transform.converged,
    )

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
