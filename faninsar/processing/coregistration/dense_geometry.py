"""Dense geometry-driven offset fields from dual orbits and DEM sampling.

Estimates spatially varying range/azimuth offsets by evaluating the
reference-to-secondary mapping on a coarse control-point grid and
interpolating to full resolution.  Coverage and uncertainty are derived
from actual geometry convergence rather than constant placeholders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coregistration.offsets import OffsetFieldResult
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry.prepare_production import (
    interpolate_control_field,
    run_geo2rdr,
    run_rdr2geo,
)

if TYPE_CHECKING:
    from faninsar.processing.geometry import DEM
    from faninsar.processing.geometry.transforms import RadarGeometryModel
    from faninsar.processing.runtime.types import DeviceLike

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
    device: DeviceLike,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Interpolate a scalar field from a control grid to full resolution.

    Uses device-resident bilinear interpolation (PROPOSAL-0031).

    Parameters
    ----------
    az_ctrl, rg_ctrl : numpy.ndarray
        1-D control-point coordinates.
    values_ctrl : numpy.ndarray
        2-D array of shape ``(len(az_ctrl), len(rg_ctrl))``.
    shape : tuple[int, int]
        Target full-resolution shape.
    device : DeviceLike
        Required production device (cpu or cuda after resolution).
    fill_value : float, optional
        Value for out-of-bounds samples (should not occur for control grids
        that span the array).

    Returns
    -------
    numpy.ndarray
        Interpolated full-resolution array.

    """
    return interpolate_control_field(
        az_ctrl,
        rg_ctrl,
        values_ctrl,
        shape,
        device=device,
        fill_value=fill_value,
    )


def _control_point_geometry_offsets(
    shape: tuple[int, int],
    *,
    reference_model: RadarGeometryModel,
    secondary_model: RadarGeometryModel,
    device: DeviceLike,
    dem: DEM | None = None,
    stride: int = 32,
    max_iter: int = 30,
    range_tol_m: float = 0.001,
    doppler_tol_hz: float = 0.1,
    row0: int = 0,
    col0: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate exact geometry offsets on the control-point grid.

    Parameters
    ----------
    shape : tuple[int, int]
        Control-grid array shape ``(azimuth, range)``.
    reference_model, secondary_model : RadarGeometryModel
        Geometry models for the reference and secondary images.
    device : DeviceLike
        Required production device (``auto`` resolves to cpu or cuda).
    dem : DEM | None, optional
        DEM height sampler.  If ``None``, a constant zero-height ellipsoid
        is used.
    stride : int, optional
        Control-point spacing in pixels (default 32).
    max_iter, range_tol_m, doppler_tol_hz : optional
        Newton-solver tolerances passed to the rdr2geo/geo2rdr transforms.
    row0, col0 : int, optional
        Offset of the window inside the native burst for control-point
        coordinates.

    Returns
    -------
    az_ctrl, rg_ctrl : numpy.ndarray
        1-D control-point coordinates relative to ``(row0, col0)``.
    rg_offset_ctrl, az_offset_ctrl : numpy.ndarray
        Control-point range/azimuth offsets in pixels (NaN where invalid).
    uncertainty_ctrl : numpy.ndarray
        Heuristic control-point uncertainty in pixels (NaN where invalid).
    valid : numpy.ndarray
        Boolean mask of converged control points.

    """
    az_ctrl, rg_ctrl = _build_control_grid(shape, stride)
    az_grid = az_ctrl[:, None] + float(row0)
    rg_grid = rg_ctrl[None, :] + float(col0)

    ref_geo = run_rdr2geo(
        reference_model,
        az_grid,
        rg_grid,
        dem,
        device=device,
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
        invalid = np.full(shape, np.nan, dtype=np.float64)
        return (
            az_ctrl,
            rg_ctrl,
            invalid,
            invalid,
            invalid,
            np.zeros(shape, dtype=bool),
        )
    sec_rdr = run_geo2rdr(
        secondary_model,
        np.where(geo_valid, ref_geo.latitude_deg, 0.0),
        np.where(geo_valid, ref_geo.longitude_deg, 0.0),
        np.where(geo_valid, ref_geo.height_m, 0.0),
        device=device,
        max_iter=max_iter,
        range_tol_m=range_tol_m,
        doppler_tol_hz=doppler_tol_hz,
    )

    # Valid only where both transforms converged
    valid = geo_valid & sec_rdr.converged

    # Offsets for :func:`resample_complex`, which samples
    # ``source = output_index - offset`` on the secondary.
    # Same ground point is at ``ref_index`` on the reference and
    # ``sec_index`` on the secondary, so we need
    # ``source = sec_index`` when ``output = ref_index``:
    # ``offset = ref_index - sec_index``.
    # (Previously the opposite sign was used, which mis-registered the
    # secondary by about twice the geometric shift and destroyed interferogram
    # coherence on real Sentinel-1 pairs.)
    rg_offset_ctrl = np.where(valid, rg_grid - sec_rdr.range_index, np.nan)
    az_offset_ctrl = np.where(valid, az_grid - sec_rdr.azimuth_index, np.nan)

    # Uncertainty heuristic: sum of absolute residuals scaled to pixels.
    # Range residual -> range pixels; Doppler residual (Hz) -> azimuth pixels
    # via a PRF-like factor (1 / azimuth_time_interval_s).
    range_res_px = np.abs(ref_geo.residual_range_m) / max(
        reference_model.range_spacing_m, 1e-6
    ) + np.abs(sec_rdr.residual_range_m) / max(secondary_model.range_spacing_m, 1e-6)
    prf_like = 1.0 / max(reference_model.azimuth_time_interval_s, 1e-6)
    az_res_px = np.abs(ref_geo.residual_doppler_hz) / max(prf_like, 1e-6) + np.abs(
        sec_rdr.residual_doppler_hz
    ) / max(prf_like, 1e-6)
    uncertainty_ctrl = np.where(valid, range_res_px + az_res_px, np.nan)
    return az_ctrl, rg_ctrl, rg_offset_ctrl, az_offset_ctrl, uncertainty_ctrl, valid


def dense_geometry_offsets(
    shape: tuple[int, int],
    *,
    reference_model: RadarGeometryModel,
    secondary_model: RadarGeometryModel,
    device: DeviceLike,
    dem: DEM | None = None,
    stride: int = 32,
    max_iter: int = 30,
    range_tol_m: float = 0.001,
    doppler_tol_hz: float = 0.1,
    row0: int = 0,
    col0: int = 0,
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
    device : DeviceLike
        Required production device (``auto`` resolves to cpu or cuda).
    dem : DEM | None, optional
        DEM height sampler.  If ``None``, a constant zero-height ellipsoid
        is used.
    stride : int, optional
        Control-point spacing in pixels (default 32).  Larger values are
        faster but may miss rapid geometric variation.
    max_iter, range_tol_m, doppler_tol_hz : optional
        Newton-solver tolerances passed to the rdr2geo/geo2rdr transforms.
    row0, col0 : int, optional
        Offset of the window inside the native burst for control-point
        coordinates.

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

    (
        az_ctrl,
        rg_ctrl,
        rg_offset_ctrl,
        az_offset_ctrl,
        uncertainty_ctrl,
        valid,
    ) = _control_point_geometry_offsets(
        shape,
        reference_model=reference_model,
        secondary_model=secondary_model,
        device=device,
        dem=dem,
        stride=stride,
        max_iter=max_iter,
        range_tol_m=range_tol_m,
        doppler_tol_hz=doppler_tol_hz,
        row0=row0,
        col0=col0,
    )
    if not np.any(valid):
        logger.warning("No valid control points; returning NaN offset field")
        return OffsetFieldResult(
            range_offset_px=np.full(shape, np.nan, dtype=np.float64),
            azimuth_offset_px=np.full(shape, np.nan, dtype=np.float64),
            coverage=np.zeros(shape, dtype=bool),
            uncertainty_px=np.full(shape, np.nan, dtype=np.float64),
        )

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
            # Forward fill: replace NaN with the index of the last valid
            # element.  ``minimum`` was wrong here: over the valid indices
            # (0, 1, 2, ...) the accumulated minimum stays 0, collapsing the
            # whole axis to the first row/column whenever a NaN hole exists.
            idx = np.maximum.accumulate(
                np.where(~mask, arange, -1),
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
    rg_offset = _interpolate_field(
        az_ctrl, rg_ctrl, rg_offset_ctrl, shape, device=device
    )
    az_offset = _interpolate_field(
        az_ctrl, rg_ctrl, az_offset_ctrl, shape, device=device
    )

    # Coverage: interpolate the valid mask as float then threshold
    coverage = _interpolate_field(
        az_ctrl,
        rg_ctrl,
        valid.astype(np.float64),
        shape,
        device=device,
        fill_value=0.0,
    )
    coverage = coverage > 0.5

    uncertainty_ctrl = _fill_nan_nearest(uncertainty_ctrl)
    uncertainty = _interpolate_field(
        az_ctrl,
        rg_ctrl,
        uncertainty_ctrl,
        shape,
        device=device,
        fill_value=0.0,
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


def geometry_offset_window_extent(
    window: tuple[int, int, int, int],
    *,
    burst_shape: tuple[int, int],
    reference_model: RadarGeometryModel,
    secondary_model: RadarGeometryModel,
    device: DeviceLike,
    dem: DEM | None = None,
    probe_stride: int = 64,
    max_iter: int = 30,
    range_tol_m: float = 0.001,
    doppler_tol_hz: float = 0.1,
) -> float:
    """Return the largest geometric offset magnitude near a radar window.

    The dense offset field is bilinear between control points, so its
    maximum over the window is attained at a control point of the grid
    covering the window.  This function evaluates the exact control-point
    offsets on a coarse grid spanning the window plus one probe cell on
    every side and returns the largest offset magnitude in pixels.  The
    caller uses the value to size the resampling halo; the fine field is
    re-checked after estimation and the crop is grown when the coarse
    bound turns out too tight.

    Parameters
    ----------
    window : tuple[int, int, int, int]
        Radar window ``(row0, row1, col0, col1)`` inside the burst.
    burst_shape : tuple[int, int]
        Full burst shape ``(azimuth, range)``.
    reference_model, secondary_model : RadarGeometryModel
        Geometry models for the reference and secondary images.
    device : DeviceLike
        Required production device (``auto`` resolves to cpu or cuda).
    dem : DEM | None, optional
        DEM height sampler.  If ``None``, a constant zero-height ellipsoid
        is used.
    probe_stride : int, optional
        Control-point spacing of the coarse probe (default 64).
    max_iter, range_tol_m, doppler_tol_hz : optional
        Newton-solver tolerances passed to the rdr2geo/geo2rdr transforms.

    Returns
    -------
    float
        Maximum absolute offset magnitude (pixels), or 0.0 when no control
        point converges.

    """
    if len(window) != 4:
        reject_invalid_state("window must be a (row0, row1, col0, col1) tuple")
    wr0, wr1, wc0, wc1 = window
    height, width = burst_shape
    probe_row0 = max(0, wr0 - probe_stride)
    probe_row1 = min(height, wr1 + probe_stride)
    probe_col0 = max(0, wc0 - probe_stride)
    probe_col1 = min(width, wc1 + probe_stride)
    if probe_row1 <= probe_row0 or probe_col1 <= probe_col0:
        return 0.0
    _, _, rg_offset, az_offset, _, valid = _control_point_geometry_offsets(
        (probe_row1 - probe_row0, probe_col1 - probe_col0),
        reference_model=reference_model,
        secondary_model=secondary_model,
        device=device,
        dem=dem,
        stride=probe_stride,
        max_iter=max_iter,
        range_tol_m=range_tol_m,
        doppler_tol_hz=doppler_tol_hz,
        row0=probe_row0,
        col0=probe_col0,
    )
    magnitude = np.hypot(rg_offset, az_offset)
    ok = valid & np.isfinite(magnitude)
    if not np.any(ok):
        return 0.0
    return float(np.nanmax(magnitude[ok]))
