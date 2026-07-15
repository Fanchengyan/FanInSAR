"""Reversible radar ↔ geographic coordinate transforms on WGS84."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.ellipsoid import ecef_to_llh, llh_to_ecef
from faninsar.processing.geometry.orbit import OrbitInterpolator

if TYPE_CHECKING:
    from faninsar.processing.contracts import OrbitMetadata
    from faninsar.processing.coordinates import RadarGrid

logger = setup_logger(__name__)

SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Dense transform arrays plus convergence diagnostics."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    height_m: np.ndarray
    range_index: np.ndarray
    azimuth_index: np.ndarray
    converged: np.ndarray
    residual_range_m: np.ndarray
    residual_doppler_hz: np.ndarray


@dataclass(frozen=True, slots=True)
class RadarGeometryModel:
    """Radar timing/geometry model used by rdr2geo and geo2rdr."""

    orbit: OrbitInterpolator
    sensing_start: object
    azimuth_time_interval_s: float
    starting_slant_range_m: float
    range_spacing_m: float
    wavelength_m: float
    look_direction: str

    @classmethod
    def from_radar_grid(
        cls,
        grid: RadarGrid,
        orbit: OrbitMetadata,
    ) -> RadarGeometryModel:
        """Build a geometry model from a radar grid and orbit metadata."""
        return cls(
            orbit=OrbitInterpolator.from_orbit(orbit),
            sensing_start=grid.sensing_start,
            azimuth_time_interval_s=grid.azimuth_time_interval_s,
            starting_slant_range_m=grid.starting_slant_range_m,
            range_spacing_m=grid.range_spacing_m,
            wavelength_m=grid.wavelength_m,
            look_direction=grid.look_direction,
        )

    def azimuth_time(self, azimuth_index: float) -> datetime:
        """Return absolute sensing time for an azimuth sample index."""
        start = self.sensing_start
        if not isinstance(start, datetime):
            message = "sensing_start must be a datetime"
            logger.error(message)
            raise TypeError(message)
        return start + timedelta(
            seconds=float(azimuth_index) * self.azimuth_time_interval_s
        )

    def azimuth_time_seconds(self, azimuth_index: np.ndarray) -> np.ndarray:
        """Return seconds from orbit epoch for an array of azimuth indices."""
        start = self.sensing_start
        if not isinstance(start, datetime):
            message = "sensing_start must be a datetime"
            logger.error(message)
            raise TypeError(message)
        sensing_offset_s = (start - self.orbit.epoch).total_seconds()
        return (
            sensing_offset_s
            + np.asarray(azimuth_index, dtype=np.float64) * self.azimuth_time_interval_s
        )

    def slant_range_m(self, range_index: float) -> float:
        """Return two-way-consistent one-way slant range for a range index."""
        return self.starting_slant_range_m + float(range_index) * self.range_spacing_m


def _rdr2geo_ellipsoid_scalar(
    model: RadarGeometryModel,
    azimuth_index: float,
    range_index: float,
    height_m: float,
    max_iter: int,
    range_tol_m: float,
    doppler_tol_hz: float,
) -> tuple[float, float, float, bool, float, float]:
    """Scalar Newton solve for one pixel (kept for testing and small arrays)."""
    try:
        time = model.azimuth_time(float(azimuth_index))
        state = model.orbit.evaluate(time)
    except Exception:
        return np.nan, np.nan, np.nan, False, np.nan, np.nan

    sat = np.asarray(state.position_m, dtype=np.float64)
    vel = np.asarray(state.velocity_m_s, dtype=np.float64)
    target_range = model.slant_range_m(float(range_index))

    # Initial guess: project look direction (cross product of vel and radial)
    # onto the Earth's surface at the target slant range.
    vel_u = vel / np.linalg.norm(vel)
    radial = sat / np.linalg.norm(sat)
    if model.look_direction == "right":
        look_dir = np.cross(vel_u, radial)
    else:
        look_dir = np.cross(radial, vel_u)
    look_dir = look_dir / np.linalg.norm(look_dir)
    target_approx = sat + look_dir * target_range
    lat0, lon0, _ = ecef_to_llh(target_approx[0], target_approx[1], target_approx[2])
    lat0 = float(lat0)
    lon0 = float(lon0)
    h0 = height_m
    success = False
    range_res = np.nan
    doppler_res = np.nan

    for _ in range(max_iter):
        tx, ty, tz = llh_to_ecef(lat0, lon0, h0)
        target = np.array([float(tx), float(ty), float(tz)], dtype=np.float64)
        look = target - sat
        range_m = float(np.linalg.norm(look))
        if range_m <= 0:
            break
        unit = look / range_m
        range_res = range_m - target_range
        doppler_res = 2.0 * float(np.dot(vel, unit)) / model.wavelength_m
        if abs(range_res) < range_tol_m and abs(doppler_res) < doppler_tol_hz:
            success = True
            break
        delta = 1e-5
        tx2, ty2, tz2 = llh_to_ecef(lat0 + delta, lon0, h0)
        look2 = np.array([float(tx2), float(ty2), float(tz2)]) - sat
        r2 = float(np.linalg.norm(look2))
        u2 = look2 / r2
        d_range_dlat = (r2 - range_m) / delta
        d_dop_dlat = (
            2.0
            * (float(np.dot(vel, u2)) - float(np.dot(vel, unit)))
            / model.wavelength_m
            / delta
        )
        tx3, ty3, tz3 = llh_to_ecef(lat0, lon0 + delta, h0)
        look3 = np.array([float(tx3), float(ty3), float(tz3)]) - sat
        r3 = float(np.linalg.norm(look3))
        u3 = look3 / r3
        d_range_dlon = (r3 - range_m) / delta
        d_dop_dlon = (
            2.0
            * (float(np.dot(vel, u3)) - float(np.dot(vel, unit)))
            / model.wavelength_m
            / delta
        )
        jacobian = np.array(
            [[d_range_dlat, d_range_dlon], [d_dop_dlat, d_dop_dlon]],
            dtype=np.float64,
        )
        try:
            step = np.linalg.solve(jacobian, np.array([-range_res, -doppler_res]))
        except np.linalg.LinAlgError:
            break
        lat0 += float(step[0])
        lon0 += float(step[1])

    return lat0, lon0, h0, success, range_res, doppler_res


def rdr2geo_ellipsoid(
    model: RadarGeometryModel,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    *,
    height_m: float | np.ndarray = 0.0,
    max_iter: int = 20,
    range_tol_m: float = 0.01,
    doppler_tol_hz: float = 0.1,
) -> TransformResult:
    """Map radar indices to geodetic coordinates on a constant-height ellipsoid.

    Parameters
    ----------
    model : RadarGeometryModel
        Orbit and radar timing model.
    azimuth_index, range_index : numpy.ndarray
        Broadcastable radar sample coordinates.
    height_m : float or numpy.ndarray, optional
        Ellipsoidal height(s) for the surface.
    max_iter : int, optional
        Newton iterations per sample.
    range_tol_m, doppler_tol_hz : float, optional
        Convergence tolerances.

    Returns
    -------
    TransformResult
        Geodetic arrays, radar indices, convergence masks, and residuals.

    Notes
    -----
    This is a phase-safe geometry solution, not a lossless inverse of image
    resampling. Residuals remain available for downstream quality control.
    The implementation is fully vectorised with per-pixel Newton iteration
    masks; for very large arrays use :func:`rdr2geo_with_dem_chunked`.

    """
    az = np.asarray(azimuth_index, dtype=np.float64)
    rg = np.asarray(range_index, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    az_b, rg_b, h_b = np.broadcast_arrays(az, rg, height)
    shape = az_b.shape
    n_pixels = int(np.prod(shape))

    # For very small arrays (<= 4 pixels) the scalar path avoids temporary
    # large-array overhead and provides an exact reference for tests.
    if n_pixels <= 4:
        lat = np.full(shape, np.nan, dtype=np.float64)
        lon = np.full(shape, np.nan, dtype=np.float64)
        h_out = np.full(shape, np.nan, dtype=np.float64)
        converged = np.zeros(shape, dtype=bool)
        residual_range = np.full(shape, np.nan, dtype=np.float64)
        residual_doppler = np.full(shape, np.nan, dtype=np.float64)
        for flat_index, (az_i, rg_i, h_i) in enumerate(
            zip(az_b.ravel(), rg_b.ravel(), h_b.ravel(), strict=True)
        ):
            lat0, lon0, h0, success, range_res, doppler_res = _rdr2geo_ellipsoid_scalar(
                model,
                float(az_i),
                float(rg_i),
                float(h_i),
                max_iter,
                range_tol_m,
                doppler_tol_hz,
            )
            idx = np.unravel_index(flat_index, shape)
            residual_range[idx] = range_res
            residual_doppler[idx] = doppler_res
            if success:
                lat[idx] = lat0
                lon[idx] = lon0
                h_out[idx] = h0
                converged[idx] = True
        return TransformResult(
            latitude_deg=lat,
            longitude_deg=lon,
            height_m=h_out,
            range_index=rg_b.copy(),
            azimuth_index=az_b.copy(),
            converged=converged,
            residual_range_m=residual_range,
            residual_doppler_hz=residual_doppler,
        )

    # Vectorised path for larger arrays
    lat = np.full(n_pixels, np.nan, dtype=np.float64)
    lon = np.full(n_pixels, np.nan, dtype=np.float64)
    h_out = np.full(n_pixels, np.nan, dtype=np.float64)
    converged = np.zeros(n_pixels, dtype=bool)
    residual_range = np.full(n_pixels, np.nan, dtype=np.float64)
    residual_doppler = np.full(n_pixels, np.nan, dtype=np.float64)

    times_s = model.azimuth_time_seconds(az_b.ravel())
    in_bounds = (times_s >= model.orbit.t_min_s) & (times_s <= model.orbit.t_max_s)

    if not np.any(in_bounds):
        return TransformResult(
            latitude_deg=lat.reshape(shape),
            longitude_deg=lon.reshape(shape),
            height_m=h_out.reshape(shape),
            range_index=rg_b.copy(),
            azimuth_index=az_b.copy(),
            converged=converged.reshape(shape),
            residual_range_m=residual_range.reshape(shape),
            residual_doppler_hz=residual_doppler.reshape(shape),
        )

    sat = np.full((n_pixels, 3), np.nan, dtype=np.float64)
    vel = np.full((n_pixels, 3), np.nan, dtype=np.float64)
    sat[in_bounds], vel[in_bounds] = model.orbit.evaluate_array(times_s[in_bounds])

    target_range = model.starting_slant_range_m + rg_b.ravel() * model.range_spacing_m

    # Initial guess: project look direction (cross product of vel and radial)
    # onto the Earth's surface at the target slant range.
    vel_norm = np.linalg.norm(vel, axis=-1)
    vel_u = vel / np.maximum(vel_norm, 1e-12)[:, None]
    sat_norm = np.linalg.norm(sat, axis=-1)
    radial = sat / np.maximum(sat_norm, 1e-12)[:, None]
    if model.look_direction == "right":
        look_dir = np.cross(vel_u, radial)
    else:
        look_dir = np.cross(radial, vel_u)
    look_dir_norm = np.linalg.norm(look_dir, axis=-1)
    look_dir = look_dir / np.maximum(look_dir_norm, 1e-12)[:, None]
    target_approx = sat + look_dir * target_range[:, None]
    lat0, lon0, _ = ecef_to_llh(
        target_approx[:, 0], target_approx[:, 1], target_approx[:, 2]
    )
    h0 = h_b.ravel().copy()

    active = in_bounds.copy()
    delta = 1e-5

    for _ in range(max_iter):
        if not np.any(active):
            break

        # Compute ECEF for active pixels
        tx, ty, tz = llh_to_ecef(lat0[active], lon0[active], h0[active])
        target = np.stack([tx, ty, tz], axis=-1)
        look = target - sat[active]
        range_m = np.linalg.norm(look, axis=-1)

        valid_range = range_m > 0
        active_idx = np.nonzero(active)[0]
        if not np.any(valid_range):
            active[active_idx] = False
            break
        invalid_idx = active_idx[~valid_range]
        active[invalid_idx] = False
        if not np.any(active):
            break

        # Recompute for valid active pixels
        active_idx = np.nonzero(active)[0]
        tx, ty, tz = llh_to_ecef(lat0[active], lon0[active], h0[active])
        target = np.stack([tx, ty, tz], axis=-1)
        look = target - sat[active]
        range_m = np.linalg.norm(look, axis=-1)
        unit = look / range_m[:, None]

        range_res = range_m - target_range[active]
        doppler_res = 2.0 * np.sum(vel[active] * unit, axis=-1) / model.wavelength_m

        residual_range[active_idx] = range_res
        residual_doppler[active_idx] = doppler_res

        converged_mask = (
            (np.abs(range_res) < range_tol_m)
            & (np.abs(doppler_res) < doppler_tol_hz)
        )
        conv_idx = active_idx[converged_mask]
        lat[conv_idx] = lat0[conv_idx]
        lon[conv_idx] = lon0[conv_idx]
        h_out[conv_idx] = h0[conv_idx]
        converged[conv_idx] = True
        active[conv_idx] = False

        if not np.any(active):
            break

        # Remaining active pixels
        remaining_idx = active_idx[~converged_mask]
        range_m_rem = range_m[~converged_mask]
        unit_rem = unit[~converged_mask]
        range_res_rem = range_res[~converged_mask]
        doppler_res_rem = doppler_res[~converged_mask]

        # Jacobian via finite differences
        tx2, ty2, tz2 = llh_to_ecef(lat0[active] + delta, lon0[active], h0[active])
        look2 = np.stack([tx2, ty2, tz2], axis=-1) - sat[active]
        r2 = np.linalg.norm(look2, axis=-1)
        u2 = look2 / np.maximum(r2, 1e-12)[:, None]
        d_range_dlat = (r2 - range_m_rem) / delta
        d_dop_dlat = (
            2.0
            * (
                np.sum(vel[active] * u2, axis=-1)
                - np.sum(vel[active] * unit_rem, axis=-1)
            )
            / model.wavelength_m
            / delta
        )

        tx3, ty3, tz3 = llh_to_ecef(lat0[active], lon0[active] + delta, h0[active])
        look3 = np.stack([tx3, ty3, tz3], axis=-1) - sat[active]
        r3 = np.linalg.norm(look3, axis=-1)
        u3 = look3 / np.maximum(r3, 1e-12)[:, None]
        d_range_dlon = (r3 - range_m_rem) / delta
        d_dop_dlon = (
            2.0
            * (
                np.sum(vel[active] * u3, axis=-1)
                - np.sum(vel[active] * unit_rem, axis=-1)
            )
            / model.wavelength_m
            / delta
        )

        # Solve 2x2 systems with Cramer's rule
        det = d_range_dlat * d_dop_dlon - d_range_dlon * d_dop_dlat
        valid_det = np.abs(det) > 1e-12

        if not np.any(valid_det):
            break

        rhs0 = -range_res_rem[valid_det]
        rhs1 = -doppler_res_rem[valid_det]
        step_lat = (
            rhs0 * d_dop_dlon[valid_det] - d_range_dlon[valid_det] * rhs1
        ) / det[valid_det]
        step_lon = (
            d_range_dlat[valid_det] * rhs1 - rhs0 * d_dop_dlat[valid_det]
        ) / det[valid_det]

        valid_idx = remaining_idx[valid_det]
        lat0[valid_idx] += step_lat
        lon0[valid_idx] += step_lon

    # Store last residuals for any pixels that never converged
    if np.any(active):
        active_idx = np.nonzero(active)[0]
        tx, ty, tz = llh_to_ecef(lat0[active], lon0[active], h0[active])
        target = np.stack([tx, ty, tz], axis=-1)
        look = target - sat[active]
        range_m = np.linalg.norm(look, axis=-1)
        valid = range_m > 0
        if np.any(valid):
            unit = look[valid] / range_m[valid][:, None]
            range_res = range_m[valid] - target_range[active_idx[valid]]
            doppler_res = (
                2.0 * np.sum(vel[active][valid] * unit, axis=-1) / model.wavelength_m
            )
            residual_range[active_idx[valid]] = range_res
            residual_doppler[active_idx[valid]] = doppler_res

    return TransformResult(
        latitude_deg=lat.reshape(shape),
        longitude_deg=lon.reshape(shape),
        height_m=h_out.reshape(shape),
        range_index=rg_b.copy(),
        azimuth_index=az_b.copy(),
        converged=converged.reshape(shape),
        residual_range_m=residual_range.reshape(shape),
        residual_doppler_hz=residual_doppler.reshape(shape),
    )


def geo2rdr(
    model: RadarGeometryModel,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    height_m: np.ndarray | float,
    *,
    max_iter: int = 20,
    time_tol_s: float = 1e-6,
) -> TransformResult:
    """Map geodetic coordinates to radar range/azimuth indices.

    Parameters
    ----------
    model : RadarGeometryModel
        Orbit and radar timing model.
    latitude_deg, longitude_deg : numpy.ndarray
        Geodetic coordinates in degrees.
    height_m : array or float
        Ellipsoidal heights in metres.
    max_iter : int, optional
        Newton iterations on azimuth time.
    time_tol_s : float, optional
        Azimuth-time convergence tolerance.

    Returns
    -------
    TransformResult
        Radar indices with residuals and convergence mask.

    """
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    lat_b, lon_b, h_b = np.broadcast_arrays(lat, lon, height)
    shape = lat_b.shape
    az = np.full(shape, np.nan, dtype=np.float64)
    rg = np.full(shape, np.nan, dtype=np.float64)
    converged = np.zeros(shape, dtype=bool)
    residual_range = np.full(shape, np.nan, dtype=np.float64)
    residual_doppler = np.full(shape, np.nan, dtype=np.float64)

    t_mid = 0.5 * (model.orbit.t_min_s + model.orbit.t_max_s)
    for flat_index, (lat_i, lon_i, h_i) in enumerate(
        zip(lat_b.ravel(), lon_b.ravel(), h_b.ravel(), strict=True)
    ):
        tx, ty, tz = llh_to_ecef(lat_i, lon_i, h_i)
        target = np.array([float(tx), float(ty), float(tz)], dtype=np.float64)
        t_s = t_mid
        success = False
        range_res = np.nan
        doppler_res = np.nan
        for _ in range(max_iter):
            time = model.orbit.epoch + timedelta(seconds=float(t_s))
            try:
                state = model.orbit.evaluate(time)
            except Exception:
                break
            sat = np.asarray(state.position_m, dtype=np.float64)
            vel = np.asarray(state.velocity_m_s, dtype=np.float64)
            look = target - sat
            range_m = float(np.linalg.norm(look))
            if range_m <= 0:
                break
            unit = look / range_m
            doppler_res = float(np.dot(vel, unit))
            # d(doppler)/dt ~ acceleration projection; use finite difference
            dt = 1e-3
            try:
                state2 = model.orbit.evaluate(
                    model.orbit.epoch + timedelta(seconds=float(t_s + dt))
                )
            except Exception:
                break
            sat2 = np.asarray(state2.position_m, dtype=np.float64)
            vel2 = np.asarray(state2.velocity_m_s, dtype=np.float64)
            unit2 = (target - sat2) / np.linalg.norm(target - sat2)
            d_dop_dt = (float(np.dot(vel2, unit2)) - doppler_res) / dt
            if abs(d_dop_dt) < 1e-12:
                break
            step = -doppler_res / d_dop_dt
            t_s += step
            if abs(step) < time_tol_s:
                success = True
                range_res = range_m - model.starting_slant_range_m
                # convert residual later via index
                break
        idx = np.unravel_index(flat_index, shape)
        if success:
            az_index = (t_s - 0.0) / model.azimuth_time_interval_s
            # t_s is relative to orbit epoch, convert relative to sensing_start
            sensing_offset = (model.sensing_start - model.orbit.epoch).total_seconds()
            az_index = (t_s - sensing_offset) / model.azimuth_time_interval_s
            # recompute range at converged time
            time = model.orbit.epoch + timedelta(seconds=float(t_s))
            state = model.orbit.evaluate(time)
            sat = np.asarray(state.position_m, dtype=np.float64)
            range_m = float(np.linalg.norm(target - sat))
            rg_index = (range_m - model.starting_slant_range_m) / model.range_spacing_m
            az[idx] = az_index
            rg[idx] = rg_index
            residual_range[idx] = 0.0
            residual_doppler[idx] = doppler_res
            converged[idx] = True
        else:
            residual_range[idx] = range_res
            residual_doppler[idx] = doppler_res

    return TransformResult(
        latitude_deg=lat_b.copy(),
        longitude_deg=lon_b.copy(),
        height_m=h_b.copy(),
        range_index=rg,
        azimuth_index=az,
        converged=converged,
        residual_range_m=residual_range,
        residual_doppler_hz=residual_doppler,
    )


def rdr2geo_with_dem(
    model: RadarGeometryModel,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    dem: object,
    *,
    height_seed_m: float = 0.0,
    max_iter: int = 20,
    range_tol_m: float = 0.01,
    doppler_tol_hz: float = 0.1,
    dem_iterations: int = 2,
) -> TransformResult:
    """Map radar indices to geodetic coordinates using a DEM height sampler.

    The solver first converges on a constant-height surface, then re-samples the
    DEM at the estimated lon/lat and re-solves with the updated height. Missing
    DEM values leave the sample marked as not converged.

    Parameters
    ----------
    model : RadarGeometryModel
        Orbit and radar timing model.
    azimuth_index, range_index : numpy.ndarray
        Broadcastable radar sample coordinates.
    dem : DEMSampler
        Height sampler implementing ``sample(lat, lon) -> height``.
    height_seed_m : float, optional
        Initial constant height before DEM refinement.
    max_iter, range_tol_m, doppler_tol_hz : optional
        Passed to the ellipsoid Newton solver.
    dem_iterations : int, optional
        Number of DEM re-sample / re-solve cycles.

    Returns
    -------
    TransformResult
        Geodetic arrays with DEM heights where converged.

    """
    result = rdr2geo_ellipsoid(
        model,
        azimuth_index,
        range_index,
        height_m=height_seed_m,
        max_iter=max_iter,
        range_tol_m=range_tol_m,
        doppler_tol_hz=doppler_tol_hz,
    )
    for _ in range(dem_iterations):
        if not np.any(result.converged):
            break
        heights = np.full(result.latitude_deg.shape, np.nan, dtype=np.float64)
        mask = result.converged
        if np.any(mask):
            heights[mask] = dem.sample(
                result.latitude_deg[mask], result.longitude_deg[mask]
            )
        # Re-solve all pixels with their DEM heights.
        next_result = rdr2geo_ellipsoid(
            model,
            azimuth_index,
            range_index,
            height_m=heights,
            max_iter=max_iter,
            range_tol_m=range_tol_m,
            doppler_tol_hz=doppler_tol_hz,
        )
        valid_dem = np.isfinite(heights)
        converged = next_result.converged & valid_dem
        lat = np.where(converged, next_result.latitude_deg, np.nan)
        lon = np.where(converged, next_result.longitude_deg, np.nan)
        height = np.where(converged, heights, np.nan)
        residual_range = np.where(converged, next_result.residual_range_m, np.nan)
        residual_doppler = np.where(converged, next_result.residual_doppler_hz, np.nan)
        result = TransformResult(
            latitude_deg=lat,
            longitude_deg=lon,
            height_m=height,
            range_index=next_result.range_index,
            azimuth_index=next_result.azimuth_index,
            converged=converged,
            residual_range_m=residual_range,
            residual_doppler_hz=residual_doppler,
        )
    return result


def rdr2geo_with_dem_chunked(
    model: RadarGeometryModel,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    dem: object,
    *,
    chunk_size: tuple[int, int] = (512, 512),
    **kwargs,
) -> TransformResult:
    """Chunked vectorised ``rdr2geo_with_dem`` for large arrays.

    Processes the full grid in non-overlapping tiles to keep peak memory
    bounded. Each tile is solved independently, so convergence at chunk
    boundaries is identical to the full-array solver.

    Parameters
    ----------
    model : RadarGeometryModel
        Orbit and radar timing model.
    azimuth_index, range_index : numpy.ndarray
        Broadcastable radar sample coordinates.
    dem : DEMSampler
        Height sampler implementing ``sample(lat, lon) -> height``.
    chunk_size : tuple[int, int], optional
        ``(azimuth_chunk, range_chunk)`` tile size.
    **kwargs
        Forwarded to :func:`rdr2geo_with_dem`.

    Returns
    -------
    TransformResult
        Geodetic arrays with DEM heights where converged.

    """
    az = np.asarray(azimuth_index, dtype=np.float64)
    rg = np.asarray(range_index, dtype=np.float64)
    az_b, rg_b = np.broadcast_arrays(az, rg)
    shape = az_b.shape

    lat = np.full(shape, np.nan, dtype=np.float64)
    lon = np.full(shape, np.nan, dtype=np.float64)
    height = np.full(shape, np.nan, dtype=np.float64)
    converged = np.zeros(shape, dtype=bool)
    residual_range = np.full(shape, np.nan, dtype=np.float64)
    residual_doppler = np.full(shape, np.nan, dtype=np.float64)

    az_chunk, rg_chunk = chunk_size
    for row in range(0, shape[0], az_chunk):
        for col in range(0, shape[1], rg_chunk):
            slc = (
                slice(row, min(row + az_chunk, shape[0])),
                slice(col, min(col + rg_chunk, shape[1])),
            )
            chunk_result = rdr2geo_with_dem(
                model,
                az_b[slc],
                rg_b[slc],
                dem=dem,
                **kwargs,
            )
            lat[slc] = chunk_result.latitude_deg
            lon[slc] = chunk_result.longitude_deg
            height[slc] = chunk_result.height_m
            converged[slc] = chunk_result.converged
            residual_range[slc] = chunk_result.residual_range_m
            residual_doppler[slc] = chunk_result.residual_doppler_hz

    return TransformResult(
        latitude_deg=lat,
        longitude_deg=lon,
        height_m=height,
        range_index=rg_b.copy(),
        azimuth_index=az_b.copy(),
        converged=converged,
        residual_range_m=residual_range,
        residual_doppler_hz=residual_doppler,
    )
