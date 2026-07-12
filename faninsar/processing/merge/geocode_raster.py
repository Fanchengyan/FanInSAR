"""Geocode radar-coordinate complex arrays onto a common merge grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import map_coordinates

from faninsar.logging import setup_logger
from faninsar.processing.geometry import RadarGeometryModel, geo2rdr

if TYPE_CHECKING:
    from faninsar.processing.merge.grid import GeoGridSpec

logger = setup_logger(__name__)

__all__ = ["GeocodedComplex", "geocode_complex_to_grid"]


@dataclass(frozen=True, slots=True)
class GeocodedComplex:
    """A complex array resampled onto a :class:`GeoGridSpec`.

    Attributes
    ----------
    complex : numpy.ndarray
        Complex64 array of shape ``(height, width)`` on the target grid.
    valid_mask : numpy.ndarray
        Boolean mask of pixels with a valid radar-domain sample.
    weight : numpy.ndarray
        Float32 per-pixel weight (defaults to the valid mask as 0/1).
    coherence : numpy.ndarray or None
        Optional coherence layer carried through from the radar product.

    """

    complex: np.ndarray
    valid_mask: np.ndarray
    weight: np.ndarray
    coherence: np.ndarray | None = None


def _grid_pixel_centers_lonlat(
    grid: GeoGridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lon/lat of pixel centers for a projected grid.

    For geographic CRS the transform already yields lon/lat. For projected
    CRS we inverse-project via pyproj.

    """
    xs, ys = grid.xy_pixel_centers()
    crs = grid.crs
    if crs.upper().startswith("EPSG:4326") or crs == "EPSG:4326":
        return ys, xs  # lat=ys, lon=xs for a geographic plate carrée grid

    from pyproj import Transformer

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xs.ravel(), ys.ravel(), direction="INVERSE")
    return (
        np.asarray(lat, dtype=np.float64).reshape(xs.shape),
        np.asarray(lon, dtype=np.float64).reshape(xs.shape),
    )


def geocode_complex_to_grid(
    complex_radar: np.ndarray,
    *,
    geometry: RadarGeometryModel,
    grid: GeoGridSpec,
    height_m: np.ndarray | float = 0.0,
    coherence: np.ndarray | None = None,
    chunk_size: int | None = None,
    radar_shape: tuple[int, int] | None = None,
) -> GeocodedComplex:
    """Resample a radar complex array onto a common geographic grid.

    The mapping uses :func:`geo2rdr` to obtain radar ``(az, rg)`` indices
    for each target pixel center, then bilinearly resamples the real and
    imaginary parts of ``complex_radar``.

    Parameters
    ----------
    complex_radar : numpy.ndarray
        Complex radar array of shape ``(height, width)`` matching
        ``geometry``.
    geometry : RadarGeometryModel
        Radar timing/geometry model for the source burst.
    grid : GeoGridSpec
        Target common grid.
    height_m : array or float, optional
        Terrain/ellipsoid height used by ``geo2rdr``. Defaults to 0.
    coherence : numpy.ndarray, optional
        Optional coherence layer to resample alongside the complex field.
    chunk_size : int, optional
        Row-chunk size for chunked geo2rdr evaluation. ``None`` processes
        the whole grid in one call.
    radar_shape : tuple of int, optional
        Expected radar array shape ``(height, width)``. When provided the
        input array is checked against it.

    Returns
    -------
    GeocodedComplex
        Resampled complex field, valid mask, and weight on the target grid.

    Raises
    ------
    ValueError
        If ``complex_radar`` is empty or does not match ``radar_shape``.

    """
    radar_h, radar_w = complex_radar.shape
    if radar_h == 0 or radar_w == 0:
        msg = "complex_radar must be non-empty"
        raise ValueError(msg)
    if radar_shape is not None and complex_radar.shape != tuple(radar_shape):
        msg = (
            f"complex_radar shape {complex_radar.shape} does not match "
            f"radar_shape {tuple(radar_shape)}"
        )
        raise ValueError(msg)

    target_shape = grid.shape
    lat, lon = _grid_pixel_centers_lonlat(grid)

    if chunk_size is None:
        chunks = [np.arange(target_shape[0])]
    else:
        chunks = [
            slice(i, min(i + chunk_size, target_shape[0]))
            for i in range(0, target_shape[0], chunk_size)
        ]

    complex_out = np.zeros(target_shape, dtype=np.complex64)
    valid = np.zeros(target_shape, dtype=bool)
    coh_out = (
        np.zeros(target_shape, dtype=np.float32) if coherence is not None else None
    )

    for chunk in chunks:
        lat_c = np.ascontiguousarray(lat[chunk])
        lon_c = np.ascontiguousarray(lon[chunk])
        geo_finite = np.isfinite(lat_c) & np.isfinite(lon_c)
        # Replace non-finite coordinates with safe values so geo2rdr does not
        # raise; we will mask these pixels out afterwards.
        safe_lat = np.where(geo_finite, lat_c, 0.0)
        safe_lon = np.where(geo_finite, lon_c, 0.0)
        result = geo2rdr(geometry, safe_lat, safe_lon, height_m)
        az = result.azimuth_index
        rg = result.range_index
        conv = result.converged
        in_range = (
            geo_finite
            & conv
            & np.isfinite(az)
            & np.isfinite(rg)
            & (az >= 0.0)
            & (az <= radar_h - 1.0)
            & (rg >= 0.0)
            & (rg <= radar_w - 1.0)
        )
        coords = np.array([az[in_range], rg[in_range]])
        re = map_coordinates(
            complex_radar.real, coords, order=1, mode="constant", cval=0.0
        )
        im = map_coordinates(
            complex_radar.imag, coords, order=1, mode="constant", cval=0.0
        )
        block = np.zeros(lat_c.shape, dtype=np.complex64)
        block[in_range] = (re + 1j * im).astype(np.complex64)
        complex_out[chunk] = block
        valid[chunk] = in_range
        if coh_out is not None and coherence is not None:
            coh_block = np.zeros(lat_c.shape, dtype=np.float32)
            coh_vals = map_coordinates(
                coherence.astype(np.float32),
                coords,
                order=1,
                mode="constant",
                cval=0.0,
            )
            coh_block[in_range] = coh_vals.astype(np.float32)
            coh_out[chunk] = coh_block

    weight = valid.astype(np.float32)
    logger.info(
        "geocode_complex_to_grid: %d/%d valid pixels",
        int(valid.sum()),
        valid.size,
    )
    return GeocodedComplex(
        complex=complex_out,
        valid_mask=valid,
        weight=weight,
        coherence=coh_out,
    )
