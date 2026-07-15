"""Dense geometry-driven offset fields from dual orbits and DEM sampling.

Estimates spatially varying range/azimuth offsets by evaluating the
reference-to-secondary mapping on a coarse control-point grid and
interpolating to full resolution.  Coverage and uncertainty are derived
from actual geometry convergence rather than constant placeholders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from faninsar.logging import setup_logger
from faninsar.processing.coreg.offsets import OffsetFieldResult
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry.transforms import (
    RadarGeometryModel,
    geo2rdr,
    rdr2geo_with_dem,
)

if TYPE_CHECKING:
    from faninsar.processing.geometry.dem import DEMSampler

logger = setup_logger(__name__)


def _build_control_grid(
    shape: tuple[int, int],
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a regular control-point grid with the given stride.

    Parameters
    ----------
    shape : tuple[int, int]
        Full array shape ``(height, width)``.
    stride : int
        Sampling step in both dimensions.

    Returns
    -------
    az_idx : numpy.ndarray
        1-D azimuth indices of control points.
    rg_idx : numpy.ndarray
        1-D range indices of control points.

    """
    height, width = shape
    az_idx = np.arange(0, height, stride, dtype=np.float64)
    rg_idx = np.arange(0, width, stride, dtype=np.float64)
    # Ensure the last pixel is included so the interpolator covers the edge
    if az_idx[-1] < height - 1:
        az_idx = np.append(az_idx, float(height - 1))
    if rg_idx[-1] < width - 1:
        rg_idx = np.append(rg_idx, float(width - 1))
    return az_idx, rg_idx


def _interpolate_field(
    az_ctrl: np.ndarray,
    rg_ctrl: np.ndarray,
    values_ctrl: np.ndarray,
    shape: tuple[int, int],
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Interpolate a scalar field from a control grid to full resolution.

    Uses bilinear interpolation via ``RegularGridInterpolator``.

    Parameters
    ----------
    az_ctrl, rg_ctrl : numpy.ndarray
        1-D control-point coordinates.
    values_ctrl : numpy.ndarray
        2-D array of shape ``(len(az_ctrl), len(rg_ctrl))``.
    shape : tuple[int, int]
        Target full-resolution shape.
    fill_value : float, optional
        Value for out-of-bounds samples (should not occur for control grids
        that span the array).

    Returns
    -------
    numpy.ndarray
        Interpolated full-resolution array.

    """
    interpolator = RegularGridInterpolator(
        (az_ctrl, rg_ctrl),
        values_ctrl,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )
    height, width = shape
    # Tile azimuth so the (N, 2) query buffer never holds a full S1 burst
    # (~30M points × 16 B ≈ 0.5 GB) at once alongside other offset fields.
    out = np.empty(shape, dtype=np.float64)
    row_chunk = 128
    col_idx = np.arange(width, dtype=np.float64)
    for row0 in range(0, height, row_chunk):
        row1 = min(row0 + row_chunk, height)
        n_rows = row1 - row0
        az = np.arange(row0, row1, dtype=np.float64)
        az_grid, rg_grid = np.meshgrid(az, col_idx, indexing="ij")
        points = np.column_stack([az_grid.ravel(), rg_grid.ravel()])
        out[row0:row1] = interpolator(points).reshape(n_rows, width)
    return out


def dense_geometry_offsets(
    shape: tuple[int, int],
    *,
    reference_model: RadarGeometryModel,
    secondary_model: RadarGeometryModel,
    dem: DEMSampler | None = None,
    stride: int = 32,
    max_iter: int = 20,
    range_tol_m: float = 0.01,
    doppler_tol_hz: float = 0.1,
) -> OffsetFieldResult:
    """Estimate dense range/azimuth offsets from dual-orbit geometry + DEM.

    The algorithm places control points on a coarse grid, maps each point
    to geodetic coordinates via ``rdr2geo`` (with DEM if provided), then
    back-projects those coordinates into the secondary radar geometry via
    ``geo2rdr``.  The difference between secondary and reference indices
    gives the local offset.  The sparse offset field is interpolated to
    full resolution.

    Parameters
    ----------
    shape : tuple[int, int]
        Output offset-field shape ``(azimuth, range)``.
    reference_model, secondary_model : RadarGeometryModel
        Geometry models for the reference and secondary images.
    dem : DEMSampler | None, optional
        DEM height sampler.  If ``None``, a constant zero-height ellipsoid
        is used.
    stride : int, optional
        Control-point spacing in pixels (default 32).  Larger values are
        faster but may miss rapid geometric variation.
    max_iter, range_tol_m, doppler_tol_hz : optional
        Newton-solver tolerances passed to the rdr2geo/geo2rdr transforms.

    Returns
    -------
    OffsetFieldResult
        Dense offset field with a real coverage mask (``True`` where the
        control-point geometry converged and interpolated) and an
        uncertainty map derived from transform residuals.

    Raises
    ------
    ValueError
        If ``shape`` is not positive or ``stride`` < 1.

    Notes
    -----
    The coverage mask is ``True`` only where both the reference
    ``rdr2geo`` and the secondary ``geo2rdr`` converged.  Uncertainty
    is a heuristic combining range and Doppler residuals, scaled by
    the pixel spacing so that it is expressed in pixels.

    """
    if min(shape) <= 0:
        reject_invalid_state("offset field shape must be positive")
    if stride < 1:
        reject_invalid_state("stride must be >= 1")

    az_ctrl, rg_ctrl = _build_control_grid(shape, stride)
    az_grid, rg_grid = np.meshgrid(az_ctrl, rg_ctrl, indexing="ij")

    # Reference: radar -> geo
    if dem is not None:
        ref_geo = rdr2geo_with_dem(
            reference_model,
            az_grid,
            rg_grid,
            dem,
            max_iter=max_iter,
            range_tol_m=range_tol_m,
            doppler_tol_hz=doppler_tol_hz,
        )
    else:
        from faninsar.processing.geometry.transforms import rdr2geo_ellipsoid

        ref_geo = rdr2geo_ellipsoid(
            reference_model,
            az_grid,
            rg_grid,
            height_m=0.0,
            max_iter=max_iter,
            range_tol_m=range_tol_m,
            doppler_tol_hz=doppler_tol_hz,
        )

    # Secondary: geo -> radar
    # Mask out non-finite geodetic coordinates to avoid NaN propagation in geo2rdr
    geo_valid = (
        ref_geo.converged
        & np.isfinite(ref_geo.latitude_deg)
        & np.isfinite(ref_geo.longitude_deg)
        & np.isfinite(ref_geo.height_m)
    )
    if not np.any(geo_valid):
        logger.warning("No valid geodetic coordinates for secondary geo2rdr")
        return OffsetFieldResult(
            range_offset_px=np.full(shape, np.nan, dtype=np.float64),
            azimuth_offset_px=np.full(shape, np.nan, dtype=np.float64),
            coverage=np.zeros(shape, dtype=bool),
            uncertainty_px=np.full(shape, np.nan, dtype=np.float64),
        )
    sec_rdr = geo2rdr(
        secondary_model,
        np.where(geo_valid, ref_geo.latitude_deg, 0.0),
        np.where(geo_valid, ref_geo.longitude_deg, 0.0),
        np.where(geo_valid, ref_geo.height_m, 0.0),
        max_iter=max_iter,
    )

    # Valid only where both transforms converged
    valid = geo_valid & sec_rdr.converged

    # Compute offsets: secondary_index - reference_index
    # Positive offset means secondary is at larger index => shift secondary backward
    rg_offset_ctrl = sec_rdr.range_index - rg_grid
    az_offset_ctrl = sec_rdr.azimuth_index - az_grid

    # Mask invalid control points before interpolation
    rg_offset_ctrl = np.where(valid, rg_offset_ctrl, np.nan)
    az_offset_ctrl = np.where(valid, az_offset_ctrl, np.nan)

    # Simple nearest-neighbour fill for NaN holes so interpolation is stable
    # (small holes from occasional non-convergence)
    def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
        """Fill NaN values with the nearest valid value."""
        if not np.any(np.isnan(arr)):
            return arr
        filled = arr.copy()
        mask = np.isnan(filled)
        if mask.all():
            return np.zeros_like(filled)
        for axis in range(filled.ndim):
            # Build an index array with the same shape as filled for broadcasting
            shape_idx = [1] * filled.ndim
            shape_idx[axis] = filled.shape[axis]
            arange = np.arange(filled.shape[axis]).reshape(shape_idx)
            # Forward fill: replace NaN with the index of the last valid element
            idx = np.minimum.accumulate(
                np.where(~mask, arange, filled.shape[axis] - 1),
                axis=axis,
            )
            filled = np.take_along_axis(filled, idx, axis=axis)
            # Backward fill: replace remaining NaN with the next valid element
            mask = np.isnan(filled)
            idx = np.maximum.accumulate(
                np.where(~mask, arange, 0),
                axis=axis,
            )
            filled = np.take_along_axis(filled, idx, axis=axis)
        return filled

    rg_offset_ctrl = _fill_nan_nearest(rg_offset_ctrl)
    az_offset_ctrl = _fill_nan_nearest(az_offset_ctrl)

    # Interpolate to full resolution
    rg_offset = _interpolate_field(az_ctrl, rg_ctrl, rg_offset_ctrl, shape)
    az_offset = _interpolate_field(az_ctrl, rg_ctrl, az_offset_ctrl, shape)

    # Coverage: interpolate the valid mask as float then threshold
    coverage = _interpolate_field(
        az_ctrl, rg_ctrl, valid.astype(np.float64), shape, fill_value=0.0
    )
    coverage = coverage > 0.5

    # Uncertainty heuristic: sum of absolute residuals scaled to pixels
    # Range residual -> range pixels; Doppler residual -> azimuth pixels
    # Doppler residual (Hz) -> azimuth time -> azimuth pixels:
    #   Δaz = Δf_doppler * λ / (2 * |acc|) ... complicated.
    # Simpler heuristic: use the raw residual magnitudes, scaled by pixel spacing.
    range_res_px = np.abs(ref_geo.residual_range_m) / max(
        reference_model.range_spacing_m, 1e-6
    ) + np.abs(sec_rdr.residual_range_m) / max(secondary_model.range_spacing_m, 1e-6)
    # For azimuth, convert Doppler residual (Hz) to approximate pixels.
    # Doppler = 2 v·u / λ  [Hz].  d(doppler)/dt ~ 2 a·u / λ.
    # Approximate azimuth uncertainty from Doppler residual:
    #   Δt ≈ Δf_doppler / (2 |a| / λ)   but we don't have acceleration easily.
    # Fallback: scale by PRF-like factor = 1 / azimuth_time_interval_s
    prf_like = 1.0 / max(reference_model.azimuth_time_interval_s, 1e-6)
    az_res_px = np.abs(ref_geo.residual_doppler_hz) / max(prf_like, 1e-6) + np.abs(
        sec_rdr.residual_doppler_hz
    ) / max(prf_like, 1e-6)
    uncertainty_ctrl = range_res_px + az_res_px
    uncertainty_ctrl = np.where(valid, uncertainty_ctrl, np.nan)
    uncertainty_ctrl = _fill_nan_nearest(uncertainty_ctrl)
    uncertainty = _interpolate_field(
        az_ctrl, rg_ctrl, uncertainty_ctrl, shape, fill_value=0.0
    )
    # Cap uncertainty at a reasonable maximum for display / weighting
    uncertainty = np.clip(uncertainty, 0.0, 10.0)

    logger.info(
        "Dense geometry offsets shape=%s stride=%d valid_ctrl=%d/%d",
        shape,
        stride,
        int(np.sum(valid)),
        valid.size,
    )

    # float32 is ample for sub-pixel geometry offsets and halves the full-burst
    # footprint of the four dense maps (~1 GB instead of ~2 GB at S1 IW size).
    return OffsetFieldResult(
        range_offset_px=rg_offset.astype(np.float32, copy=False),
        azimuth_offset_px=az_offset.astype(np.float32, copy=False),
        coverage=coverage,
        uncertainty_px=uncertainty.astype(np.float32, copy=False),
    )
