"""Device-resident Torch geometry kernels used by the public v2 adapter.

The kernels intentionally keep the numerical loop in Torch.  Preparation
materialises the orbit samples; execution performs interpolation, coordinate
conversion, residual evaluation, and the masked iteration loop on the target
device.  This module has no NumPy geometry fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_A = 6_378_137.0
_E2 = 0.0066943799901413165


def _llh_to_ecef(lat: Any, lon: Any, height: Any) -> Any:
    """Convert degree LLH tensors to ECEF tensors."""
    import torch

    lat_r = torch.deg2rad(lat)
    lon_r = torch.deg2rad(lon)
    sin_lat = torch.sin(lat_r)
    cos_lat = torch.cos(lat_r)
    radius = _A / torch.sqrt(1.0 - _E2 * sin_lat.square())
    return torch.stack(
        (
            (radius + height) * cos_lat * torch.cos(lon_r),
            (radius + height) * cos_lat * torch.sin(lon_r),
            (radius * (1.0 - _E2) + height) * sin_lat,
        ),
        dim=-1,
    )


def _ecef_to_llh(ecef: Any) -> tuple[Any, Any, Any]:
    """Convert ECEF tensors to degree LLH using a stable iterative formula."""
    import torch

    x, y, z = ecef.unbind(-1)
    longitude = torch.rad2deg(torch.atan2(y, x))
    latitude = torch.atan2(z, torch.hypot(x, y))
    height = torch.zeros_like(latitude)
    for _ in range(6):
        sin_lat = torch.sin(latitude)
        radius = _A / torch.sqrt(1.0 - _E2 * sin_lat.square())
        height = (
            torch.hypot(x, y) / torch.clamp(torch.cos(latitude), min=1.0e-12) - radius
        )
        latitude = torch.atan2(
            z,
            torch.hypot(x, y)
            * (1.0 - _E2 * radius / torch.clamp(radius + height, min=1.0)),
        )
    return torch.rad2deg(latitude), longitude, height


def _orbit_state(
    times: Any, orbit_times: Any, positions: Any, velocities: Any
) -> tuple[Any, Any]:
    """Linearly interpolate a prepared orbit table on the target device."""
    import torch

    safe = torch.clamp(times, orbit_times[0], orbit_times[-1])
    right = torch.bucketize(safe, orbit_times).clamp(1, orbit_times.numel() - 1)
    left = right - 1
    fraction = (safe - orbit_times[left]) / torch.clamp(
        orbit_times[right] - orbit_times[left], min=1.0e-12
    )
    fraction = fraction.unsqueeze(-1)
    return (
        positions[left] + fraction * (positions[right] - positions[left]),
        velocities[left] + fraction * (velocities[right] - velocities[left]),
    )


def _result_invalid(output: dict[str, Any], finite: Any) -> dict[str, Any]:
    """Apply v2 invalid-lane sentinels to a tensor result."""
    import torch

    for name in (
        "range_index",
        "azimuth_index",
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "residual_range_m",
        "residual_doppler_hz",
    ):
        output[name] = torch.where(
            finite, output[name], torch.full_like(output[name], torch.nan)
        )
    output["converged"] = output["converged"] & finite
    output["iterations"] = torch.where(
        finite,
        output["iterations"],
        torch.full_like(output["iterations"], -1),
    )
    output["invalid"] = ~finite
    return output


def geo2rdr_kernel(
    latitude: Any,
    longitude: Any,
    height: Any,
    orbit_times: Any,
    orbit_positions: Any,
    orbit_velocities: Any,
    *,
    sensing_offset_s: float,
    azimuth_interval_s: float,
    starting_range_m: float,
    range_spacing_m: float,
    wavelength_m: float,
    max_iter: int,
    range_tol_m: float,
    doppler_tol_hz: float,
    dynamic_iterations: bool,
) -> dict[str, Any]:
    """Solve geo2rdr with a masked, per-lane Newton loop on Torch."""
    import torch

    target = _llh_to_ecef(latitude, longitude, height)
    finite = torch.isfinite(target).all(dim=-1)
    sat0, vel0 = _orbit_state(
        torch.full_like(latitude, sensing_offset_s),
        orbit_times,
        orbit_positions,
        orbit_velocities,
    )
    look0 = target - sat0
    speed2 = torch.sum(vel0 * vel0, dim=-1)
    seed = sensing_offset_s + torch.sum(look0 * vel0, dim=-1) / torch.clamp(
        speed2, min=1.0e-12
    )
    times = torch.where(finite, seed, torch.full_like(seed, sensing_offset_s))
    solved = torch.zeros_like(finite)
    iterations = torch.zeros_like(latitude, dtype=torch.int32)
    doppler = torch.full_like(latitude, torch.nan)
    range_residual = torch.full_like(latitude, torch.nan)
    for attempt in range(1, max_iter + 1):
        sat, velocity = _orbit_state(
            times, orbit_times, orbit_positions, orbit_velocities
        )
        look = target - sat
        distance = torch.linalg.vector_norm(look, dim=-1)
        unit = look / torch.clamp(distance, min=1.0e-12).unsqueeze(-1)
        current = 2.0 * torch.sum(velocity * unit, dim=-1) / wavelength_m
        step = torch.where(
            torch.isfinite(current), current * 0.0, torch.zeros_like(current)
        )
        # The azimuth-time root is the Doppler zero.  A finite-difference slope
        # keeps the operation explicit and works for eager and compiled paths.
        delta = torch.full_like(times, 1.0e-3)
        sat_next, vel_next = _orbit_state(
            times + delta, orbit_times, orbit_positions, orbit_velocities
        )
        unit_next = (target - sat_next) / torch.clamp(
            torch.linalg.vector_norm(target - sat_next, dim=-1), min=1.0e-12
        ).unsqueeze(-1)
        doppler_next = 2.0 * torch.sum(vel_next * unit_next, dim=-1) / wavelength_m
        slope = (doppler_next - current) / delta
        step = torch.where(
            torch.isfinite(slope) & (torch.abs(slope) > 1.0e-12), -current / slope, step
        )
        active = finite & ~solved
        next_times = torch.where(active, times + step, times)
        final_sat, final_velocity = _orbit_state(
            next_times, orbit_times, orbit_positions, orbit_velocities
        )
        final_look = target - final_sat
        final_distance = torch.linalg.vector_norm(final_look, dim=-1)
        final_unit = final_look / torch.clamp(final_distance, min=1.0e-12).unsqueeze(-1)
        final_doppler = (
            2.0 * torch.sum(final_velocity * final_unit, dim=-1) / wavelength_m
        )
        final_range_index = (final_distance - starting_range_m) / range_spacing_m
        final_range = starting_range_m + final_range_index * range_spacing_m
        final_range_residual = final_distance - final_range
        metric = torch.maximum(
            torch.abs(final_range_residual) / range_tol_m,
            torch.abs(final_doppler) / doppler_tol_hz,
        )
        newly = active & torch.isfinite(metric) & (metric < 1.0)
        solved |= newly
        iterations = torch.where(
            active, torch.full_like(iterations, attempt), iterations
        )
        times = next_times
        doppler = torch.where(active, final_doppler, doppler)
        range_residual = torch.where(active, final_range_residual, range_residual)
        if dynamic_iterations and bool(torch.all(solved | ~finite).item()):
            break
    final_sat, final_velocity = _orbit_state(
        times, orbit_times, orbit_positions, orbit_velocities
    )
    final_look = target - final_sat
    final_distance = torch.linalg.vector_norm(final_look, dim=-1)
    final_unit = final_look / torch.clamp(final_distance, min=1.0e-12).unsqueeze(-1)
    doppler = 2.0 * torch.sum(final_velocity * final_unit, dim=-1) / wavelength_m
    range_index = (final_distance - starting_range_m) / range_spacing_m
    range_residual = final_distance - (starting_range_m + range_index * range_spacing_m)
    azimuth_index = (times - sensing_offset_s) / azimuth_interval_s
    output = {
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "height_m": height,
        "range_index": range_index,
        "azimuth_index": azimuth_index,
        "converged": solved,
        "iterations": iterations,
        "residual_range_m": range_residual,
        "residual_doppler_hz": doppler,
    }
    return _result_invalid(output, finite)


def rdr2geo_kernel(
    azimuth: Any,
    range_index: Any,
    height_seed: Any,
    orbit_times: Any,
    orbit_positions: Any,
    orbit_velocities: Any,
    *,
    sensing_offset_s: float,
    azimuth_interval_s: float,
    starting_range_m: float,
    range_spacing_m: float,
    wavelength_m: float,
    look_sign: float,
    max_iter: int,
    range_tol_m: float,
    doppler_tol_hz: float,
    dynamic_iterations: bool,
) -> dict[str, Any]:
    """Solve rdr2geo with a masked two-variable Newton loop on Torch."""
    import torch

    target_range = starting_range_m + range_index * range_spacing_m
    times = sensing_offset_s + azimuth * azimuth_interval_s
    sat, velocity = _orbit_state(times, orbit_times, orbit_positions, orbit_velocities)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    radial = sat / torch.clamp(
        torch.linalg.vector_norm(sat, dim=-1), min=1.0e-12
    ).unsqueeze(-1)
    velocity_unit = velocity / torch.clamp(speed, min=1.0e-12).unsqueeze(-1)
    cross = torch.linalg.cross(velocity_unit, radial, dim=-1) * look_sign
    cross = cross / torch.clamp(
        torch.linalg.vector_norm(cross, dim=-1), min=1.0e-12
    ).unsqueeze(-1)
    guess = sat + target_range.unsqueeze(-1) * cross
    lat, lon, _ = _ecef_to_llh(guess)
    height = height_seed
    finite = torch.isfinite(target_range) & torch.isfinite(azimuth)
    solved = torch.zeros_like(finite)
    iterations = torch.zeros_like(azimuth, dtype=torch.int32)
    range_residual = torch.full_like(azimuth, torch.nan)
    doppler_residual = torch.full_like(azimuth, torch.nan)
    delta = torch.full_like(lat, 1.0e-5)
    for attempt in range(1, max_iter + 1):
        point = _llh_to_ecef(lat, lon, height)
        look = point - sat
        distance = torch.linalg.vector_norm(look, dim=-1)
        unit = look / torch.clamp(distance, min=1.0e-12).unsqueeze(-1)
        rr = distance - target_range
        dd = 2.0 * torch.sum(velocity * unit, dim=-1) / wavelength_m
        lat_p = lat + delta
        lon_p = lon + delta
        range_lat = torch.linalg.vector_norm(
            _llh_to_ecef(lat_p, lon, height) - sat, dim=-1
        )
        range_lon = torch.linalg.vector_norm(
            _llh_to_ecef(lat, lon_p, height) - sat, dim=-1
        )
        unit_lat = (_llh_to_ecef(lat_p, lon, height) - sat) / torch.clamp(
            range_lat, min=1.0e-12
        ).unsqueeze(-1)
        unit_lon = (_llh_to_ecef(lat, lon_p, height) - sat) / torch.clamp(
            range_lon, min=1.0e-12
        ).unsqueeze(-1)
        dr_lat = (range_lat - distance) / delta
        dr_lon = (range_lon - distance) / delta
        dd_lat = 2.0 * torch.sum(velocity * unit_lat, dim=-1) / wavelength_m
        dd_lon = 2.0 * torch.sum(velocity * unit_lon, dim=-1) / wavelength_m
        dd_lat = (dd_lat - dd) / delta
        dd_lon = (dd_lon - dd) / delta
        determinant = dr_lat * dd_lon - dr_lon * dd_lat
        determinant = determinant + torch.where(
            determinant >= 0.0,
            torch.full_like(determinant, 1.0e-12),
            torch.full_like(determinant, -1.0e-12),
        )
        dlat = (-rr * dd_lon + dr_lon * dd) / determinant
        dlon = (-dr_lat * dd + rr * dd_lat) / determinant
        active = finite & ~solved
        newly = (
            active & (torch.abs(rr) < range_tol_m) & (torch.abs(dd) < doppler_tol_hz)
        )
        solved |= newly
        iterations = torch.where(
            active, torch.full_like(iterations, attempt), iterations
        )
        lat = torch.where(active, lat + dlat, lat)
        lon = torch.where(active, lon + dlon, lon)
        range_residual = torch.where(active, rr, range_residual)
        doppler_residual = torch.where(active, dd, doppler_residual)
        if dynamic_iterations and bool(torch.all(solved | ~finite).item()):
            break
    point = _llh_to_ecef(lat, lon, height)
    final_look = point - sat
    final_distance = torch.linalg.vector_norm(final_look, dim=-1)
    final_unit = final_look / torch.clamp(final_distance, min=1.0e-12).unsqueeze(-1)
    range_residual = final_distance - target_range
    doppler_residual = 2.0 * torch.sum(velocity * final_unit, dim=-1) / wavelength_m
    final_metric = torch.maximum(
        torch.abs(range_residual) / range_tol_m,
        torch.abs(doppler_residual) / doppler_tol_hz,
    )
    solved |= finite & torch.isfinite(final_metric) & (final_metric < 1.0)
    output = {
        "latitude_deg": lat,
        "longitude_deg": lon,
        "height_m": height,
        "range_index": range_index,
        "azimuth_index": azimuth,
        "converged": solved,
        "iterations": iterations,
        "residual_range_m": range_residual,
        "residual_doppler_hz": doppler_residual,
    }
    return _result_invalid(output, finite)


def prepared_orbit_tensors(model: Any, device: str) -> tuple[Any, Any, Any, float]:
    """Materialise the model orbit for a prepared device executable."""
    import torch

    times = np.asarray(model.orbit.times_s, dtype=np.float64)
    positions = np.stack(
        [spline(times) for spline in model.orbit.trajectory_splines], axis=-1
    )
    velocities = np.stack(
        [spline(times, 1) for spline in model.orbit.trajectory_splines], axis=-1
    )
    offset = (model.sensing_start - model.orbit.epoch).total_seconds()
    return (
        *(
            torch.as_tensor(value, dtype=torch.float64, device=device)
            for value in (times, positions, velocities)
        ),
        offset,
    )


__all__ = ["geo2rdr_kernel", "prepared_orbit_tensors", "rdr2geo_kernel"]
