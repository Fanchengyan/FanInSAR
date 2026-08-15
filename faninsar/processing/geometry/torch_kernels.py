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
    """Convert ECEF tensors to degree LLH with Vermeille's closed form.

    The native v2 rdr2geo implementation uses the Vermeille construction,
    rather than an iterative latitude update.  Keeping this expression in
    Torch is important for both eager/device execution and compiled kernels.
    """
    import torch

    x, y, z = ecef.unbind(-1)
    e4 = _E2 * _E2
    a2 = _A * _A
    lateral = (x.square() + y.square()) / a2
    polar = (1.0 - _E2) * z.square() / a2
    reduced = (lateral + polar - e4) / 6.0
    reduced_safe = torch.where(reduced > 0.0, reduced, torch.ones_like(reduced))
    cubic = e4 * lateral * polar / (4.0 * reduced_safe.pow(3))
    cubic_radical = cubic * (2.0 + cubic)
    radical_safe = torch.clamp(cubic_radical, min=0.0)
    root_argument = torch.clamp(1.0 + cubic + torch.sqrt(radical_safe), min=0.0)
    root = torch.sign(root_argument) * torch.abs(root_argument).pow(1.0 / 3.0)
    root_safe = torch.where(torch.abs(root) > 0.0, root, torch.ones_like(root))
    u = reduced_safe * (1.0 + root_safe + 1.0 / root_safe)
    radial = torch.sqrt(torch.clamp(u.square() + e4 * polar, min=0.0))
    radial_safe = torch.where(radial > 0.0, radial, torch.ones_like(radial))
    w = _E2 * (u + radial - polar) / (2.0 * radial_safe)
    k_argument = torch.clamp(u + radial + w.square(), min=0.0)
    k = torch.sqrt(k_argument) - w
    k_safe = torch.where(torch.abs(k) > 0.0, k, torch.ones_like(k))
    horizontal = torch.hypot(x, y)
    d = k * horizontal / (k + _E2)
    latitude = torch.atan2(z, d)
    longitude = torch.atan2(y, x)
    height = (k + _E2 - 1.0) * torch.hypot(d, z) / k_safe
    valid = (
        (reduced > 0.0)
        & torch.isfinite(cubic_radical)
        & (cubic_radical >= 0.0)
        & torch.isfinite(root)
        & (torch.abs(root) > 0.0)
        & torch.isfinite(latitude)
        & torch.isfinite(longitude)
        & torch.isfinite(height)
    )
    nan = torch.full_like(latitude, torch.nan)
    return (
        torch.where(valid, torch.rad2deg(latitude), nan),
        torch.where(valid, torch.rad2deg(longitude), nan),
        torch.where(valid, height, nan),
    )


def _orbit_state(
    times: Any, orbit_times: Any, positions: Any, velocities: Any
) -> tuple[Any, Any, Any]:
    """Evaluate a cubic Hermite orbit and its first two derivatives."""
    import torch

    # Native ``orbit_segment`` advances while ``times[next] <= value``.
    segment = torch.bucketize(times, orbit_times, right=True) - 1
    segment = segment.clamp(0, orbit_times.numel() - 2)
    t0 = orbit_times[segment]
    duration = torch.clamp(orbit_times[segment + 1] - t0, min=1.0e-12)
    u = (times - t0) / duration
    u2 = u.square()
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    dh00 = (6.0 * u2 - 6.0 * u) / duration
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = (-6.0 * u2 + 6.0 * u) / duration
    dh11 = 3.0 * u2 - 2.0 * u
    d2h00 = (12.0 * u - 6.0) / duration.square()
    d2h10 = (6.0 * u - 4.0) / duration
    d2h01 = (-12.0 * u + 6.0) / duration.square()
    d2h11 = (6.0 * u - 2.0) / duration
    def expand(value: Any) -> Any:
        """Add a vector axis to a per-lane scalar."""
        return value.unsqueeze(-1)
    p0 = positions[segment]
    p1 = positions[segment + 1]
    v0 = velocities[segment]
    v1 = velocities[segment + 1]
    position = expand(h00) * p0 + expand(h10 * duration) * v0
    position = position + expand(h01) * p1 + expand(h11 * duration) * v1
    velocity = expand(dh00) * p0 + expand(dh10) * v0
    velocity = velocity + expand(dh01) * p1 + expand(dh11) * v1
    acceleration = expand(d2h00) * p0 + expand(d2h10) * v0
    acceleration = acceleration + expand(d2h01) * p1 + expand(d2h11) * v1
    return position, velocity, acceleration


def _natural_spline_six(values: Any, fraction: Any) -> Any:
    """Evaluate the local six-sample natural spline on Torch tensors."""
    import torch

    second = [torch.zeros_like(fraction) for _ in range(6)]
    recurrence = [torch.zeros_like(fraction) for _ in range(6)]
    for index in range(1, 5):
        denominator = recurrence[index - 1] / 2.0 + 2.0
        recurrence[index] = -0.5 / denominator
        second[index] = (
            3.0
            * (
                values[..., index + 1]
                - 2.0 * values[..., index]
                + values[..., index - 1]
            )
            - second[index - 1] / 2.0
        ) / denominator
    for index in range(4, 0, -1):
        second[index] = recurrence[index] * second[index + 1] + second[index]
    return values[..., 1] + fraction * (
        values[..., 2]
        - values[..., 1]
        - second[1] / 3.0
        - second[2] / 6.0
        + fraction * (second[1] / 2.0 + fraction * (second[2] - second[1]) / 6.0)
    )


def _sample_dem_six(
    dem_samples: Any,
    latitude: Any,
    longitude: Any,
    latitude_start_deg: float,
    longitude_start_deg: float,
    latitude_spacing_deg: float,
    longitude_spacing_deg: float,
) -> Any:
    """Sample a device-resident DEM with the native six-point spline."""
    import torch

    row = (latitude - latitude_start_deg) / latitude_spacing_deg
    column = (longitude - longitude_start_deg) / longitude_spacing_deg
    row_base = torch.floor(row).to(torch.int64)
    column_base = torch.floor(column).to(torch.int64)
    rows, columns = dem_samples.shape[-2:]
    valid = (
        torch.isfinite(row)
        & torch.isfinite(column)
        & (row_base >= 1)
        & (row_base <= rows - 5)
        & (column_base >= 1)
        & (column_base <= columns - 5)
    )
    safe_row = row_base.clamp(1, rows - 5)
    safe_column = column_base.clamp(1, columns - 5)
    row_values = []
    column_fraction = column - safe_column
    row_fraction = row - safe_row
    for row_offset in range(-1, 5):
        window = torch.stack(
            [
                dem_samples[safe_row + row_offset, safe_column + column_offset]
                for column_offset in range(-1, 5)
            ],
            dim=-1,
        )
        row_values.append(_natural_spline_six(window, column_fraction))
    sampled = _natural_spline_six(torch.stack(row_values, dim=-1), row_fraction)
    return torch.where(
        valid & torch.isfinite(sampled), sampled, torch.full_like(sampled, torch.nan)
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
    time_tol_s: float = 1.0e-6,
) -> dict[str, Any]:
    """Solve geo2rdr with acceleration-aware Newton iterations on Torch.

    The update and stopping rules intentionally mirror the native v2 CPU and
    CUDA kernels.  In particular, orbit samples are cubic-Hermite evaluated
    with acceleration, Newton overshoots are rejected instead of clamped, and
    residuals are recomputed at the accepted final time.
    """
    import torch

    target = _llh_to_ecef(latitude, longitude, height)
    finite = torch.isfinite(target).all(dim=-1)
    orbit_start = orbit_times[0]
    orbit_end = orbit_times[-1]
    seed_time = torch.full_like(latitude, sensing_offset_s).clamp(
        min=orbit_start, max=orbit_end
    )
    sat0, vel0, _ = _orbit_state(
        seed_time,
        orbit_times,
        orbit_positions,
        orbit_velocities,
    )
    look0 = target - sat0
    speed2 = torch.sum(vel0 * vel0, dim=-1)
    seed = sensing_offset_s + torch.sum(look0 * vel0, dim=-1) / torch.clamp(
        speed2, min=1.0e-12
    )
    times = torch.where(
        finite,
        seed.clamp(min=orbit_start, max=orbit_end),
        torch.full_like(seed, sensing_offset_s),
    )
    solved = torch.zeros_like(finite)
    failed = ~finite
    attempts = torch.zeros_like(latitude, dtype=torch.int32)
    doppler = torch.full_like(latitude, torch.nan)
    range_residual = torch.full_like(latitude, torch.nan)
    budget = max_iter
    for attempt in range(1, budget + 1):
        active = finite & ~solved & ~failed
        sat, velocity, acceleration = _orbit_state(
            times, orbit_times, orbit_positions, orbit_velocities
        )
        look = target - sat
        distance = torch.linalg.vector_norm(look, dim=-1)
        unit = look / torch.clamp(distance, min=1.0e-12).unsqueeze(-1)
        radial = torch.sum(velocity * unit, dim=-1)
        velocity_squared = torch.sum(velocity * velocity, dim=-1)
        acceleration_along_look = torch.sum(acceleration * unit, dim=-1)
        derivative = acceleration_along_look + (
            radial * radial - velocity_squared
        ) / torch.clamp(distance, min=1.0)
        valid = (
            torch.isfinite(distance)
            & (distance > 0.0)
            & torch.isfinite(derivative)
            & (torch.abs(derivative) >= 1.0e-12)
        )
        attempts = torch.where(active, torch.full_like(attempts, attempt), attempts)
        step = -radial / derivative
        next_times = torch.where(active & valid, times + step, times)
        in_bounds = (next_times >= orbit_start) & (next_times <= orbit_end)
        failed |= active & (~valid | ~torch.isfinite(step) | ~in_bounds)
        step_small = active & valid & in_bounds & (torch.abs(step) <= time_tol_s)
        final_sat, final_velocity, _ = _orbit_state(
            next_times, orbit_times, orbit_positions, orbit_velocities
        )
        final_look = target - final_sat
        final_distance = torch.linalg.vector_norm(final_look, dim=-1)
        final_unit = final_look / torch.clamp(final_distance, min=1.0e-12).unsqueeze(-1)
        final_doppler = 2.0 * torch.sum(
            final_velocity * final_unit, dim=-1
        ) / wavelength_m
        final_range_index = (final_distance - starting_range_m) / range_spacing_m
        final_range = starting_range_m + final_range_index * range_spacing_m
        final_range_residual = final_distance - final_range
        metric = torch.maximum(
            torch.abs(final_range_residual) / range_tol_m,
            torch.abs(final_doppler) / doppler_tol_hz,
        )
        newly = step_small & torch.isfinite(metric) & (metric < 1.0)
        solved |= newly
        doppler = torch.where(active, final_doppler, doppler)
        range_residual = torch.where(active, final_range_residual, range_residual)
        times = next_times
        if dynamic_iterations and bool(torch.all(solved | failed | ~finite).item()):
            break
    final_sat, final_velocity, _ = _orbit_state(
        times, orbit_times, orbit_positions, orbit_velocities
    )
    final_look = target - final_sat
    final_distance = torch.linalg.vector_norm(final_look, dim=-1)
    final_unit = final_look / torch.clamp(final_distance, min=1.0e-12).unsqueeze(-1)
    doppler = 2.0 * torch.sum(final_velocity * final_unit, dim=-1) / wavelength_m
    computed_range_index = (final_distance - starting_range_m) / range_spacing_m
    range_residual = final_distance - (
        starting_range_m + computed_range_index * range_spacing_m
    )
    computed_azimuth_index = (times - sensing_offset_s) / azimuth_interval_s
    nan = torch.full_like(computed_range_index, torch.nan)
    output = {
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "height_m": height,
        "range_index": torch.where(solved, computed_range_index, nan),
        "azimuth_index": torch.where(solved, computed_azimuth_index, nan),
        "converged": solved,
        "iterations": torch.where(
            solved | (finite & ~failed), attempts, torch.full_like(attempts, -1)
        ),
        "residual_range_m": range_residual,
        "residual_doppler_hz": doppler,
    }
    return _result_invalid(output, finite)


def _rdr2geo_once(
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
    dem_samples: Any | None = None,
    dem_latitude_start_deg: float = 0.0,
    dem_longitude_start_deg: float = 0.0,
    dem_latitude_spacing_deg: float = 1.0,
    dem_longitude_spacing_deg: float = 1.0,
    dem_iterations: int = 1,
    dem_height_tol_m: float = 1.0e-3,
    dem_height_m: float | None = None,
) -> dict[str, Any]:
    """Solve rdr2geo with the native closed-form TCN construction."""
    import torch

    # Reserved for the prepared DEM adapter; the no-DEM path follows the
    # native constant-height fixed point exactly.
    _ = (
        doppler_tol_hz,
        dem_samples,
        dem_iterations,
        dem_height_tol_m,
        dem_height_m,
    )

    target_range = starting_range_m + range_index * range_spacing_m
    times = sensing_offset_s + azimuth * azimuth_interval_s
    orbit_start, orbit_end = orbit_times[0], orbit_times[-1]
    sat, velocity, _ = _orbit_state(
        times, orbit_times, orbit_positions, orbit_velocities
    )
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    satellite_norm = torch.linalg.vector_norm(sat, dim=-1)
    finite = (
        torch.isfinite(target_range) & (target_range > 0.0)
        & torch.isfinite(azimuth) & torch.isfinite(range_index)
        & torch.isfinite(height_seed) & torch.isfinite(times)
        & (times >= orbit_start) & (times <= orbit_end)
        & (speed > 0.0) & (satellite_norm > 0.0)
    )
    velocity_unit = velocity / torch.clamp(speed, min=1.0e-12).unsqueeze(-1)
    normal = -sat / torch.clamp(satellite_norm, min=1.0e-12).unsqueeze(-1)
    cross_track = torch.linalg.cross(normal, velocity, dim=-1)
    cross_track = cross_track / torch.clamp(
        torch.linalg.vector_norm(cross_track, dim=-1), min=1.0e-12
    ).unsqueeze(-1)
    along_track = torch.linalg.cross(cross_track, normal, dim=-1)
    along_track = along_track / torch.clamp(
        torch.linalg.vector_norm(along_track, dim=-1), min=1.0e-12
    ).unsqueeze(-1)
    normal_dot_velocity = torch.sum(normal * velocity_unit, dim=-1)
    velocity_dot_along = torch.sum(velocity_unit * along_track, dim=-1)
    finite &= torch.isfinite(normal_dot_velocity) & (velocity_dot_along > 1.0e-12)
    minor = _A * torch.sqrt(
        torch.as_tensor(1.0 - _E2, dtype=sat.dtype, device=sat.device)
    )
    eta = 1.0 / torch.sqrt(
        (sat[..., 0] / _A).square() + (sat[..., 1] / _A).square()
        + (sat[..., 2] / minor).square()
    )
    radius = eta * satellite_norm
    ellipsoid_height = (1.0 - eta) * satellite_norm
    height = height_seed
    latitude = torch.full_like(azimuth, torch.nan)
    longitude = torch.full_like(azimuth, torch.nan)
    solved = torch.zeros_like(finite)
    failed = ~finite
    attempts = torch.zeros_like(azimuth, dtype=torch.int32)
    range_residual = torch.full_like(azimuth, torch.nan)
    doppler_residual = torch.full_like(azimuth, torch.nan)
    for attempt in range(1, max_iter + 1):
        active = finite & ~solved & ~failed & (ellipsoid_height - height < target_range)
        failed |= finite & ~solved & ~active
        semi_minor = radius + height
        cos_theta = 0.5 * (
            satellite_norm / target_range + target_range / satellite_norm
            - (semi_minor / satellite_norm) * (semi_minor / target_range)
        )
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.square(), min=0.0))
        gamma = target_range * cos_theta
        alpha = -gamma * normal_dot_velocity / torch.clamp(
            velocity_dot_along, min=1.0e-12
        )
        beta_argument = (target_range * sin_theta).square() - alpha.square()
        valid = torch.isfinite(beta_argument) & (beta_argument >= -1.0e-6)
        beta = look_sign * torch.sqrt(torch.clamp(beta_argument, min=0.0))
        target_xyz = sat + alpha.unsqueeze(-1) * along_track
        target_xyz = target_xyz + beta.unsqueeze(-1) * cross_track
        target_xyz = target_xyz + gamma.unsqueeze(-1) * normal
        latitude_candidate, longitude_candidate, _ = _ecef_to_llh(target_xyz)
        if dem_samples is not None:
            dem_height = _sample_dem_six(
                dem_samples,
                latitude_candidate,
                longitude_candidate,
                dem_latitude_start_deg,
                dem_longitude_start_deg,
                dem_latitude_spacing_deg,
                dem_longitude_spacing_deg,
            )
        else:
            dem_height = (
                torch.full_like(height, dem_height_m)
                if dem_height_m is not None
                else height
            )
        dem_xyz = _llh_to_ecef(latitude_candidate, longitude_candidate, dem_height)
        look = dem_xyz - sat
        slant_range = torch.linalg.vector_norm(look, dim=-1)
        unit = look / torch.clamp(slant_range, min=1.0e-12).unsqueeze(-1)
        rr = slant_range - target_range
        dd = 2.0 * torch.sum(velocity * unit, dim=-1) / wavelength_m
        next_height = torch.linalg.vector_norm(dem_xyz, dim=-1) - radius
        valid &= (
            torch.isfinite(latitude_candidate)
            & torch.isfinite(longitude_candidate)
        )
        valid &= (
            torch.isfinite(slant_range)
            & (slant_range > 0.0)
            & torch.isfinite(next_height)
            & torch.isfinite(dem_height)
        )
        attempts = torch.where(active, torch.full_like(attempts, attempt), attempts)
        range_residual = torch.where(active, rr, range_residual)
        doppler_residual = torch.where(active, dd, doppler_residual)
        latitude = torch.where(active, latitude_candidate, latitude)
        longitude = torch.where(active, longitude_candidate, longitude)
        failed |= active & ~valid
        newly_solved = (
            active & valid & torch.isfinite(rr) & (torch.abs(rr) < range_tol_m)
        )
        solved |= newly_solved
        height = torch.where(
            newly_solved,
            dem_height,
            torch.where(active & valid, next_height, height),
        )
        if dynamic_iterations and bool(torch.all(solved | failed | ~finite).item()):
            break
    semi_minor = radius + height
    cos_theta = 0.5 * (
        satellite_norm / target_range + target_range / satellite_norm
        - (semi_minor / satellite_norm) * (semi_minor / target_range)
    )
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.square(), min=0.0))
    gamma = target_range * cos_theta
    alpha = -gamma * normal_dot_velocity / torch.clamp(velocity_dot_along, min=1.0e-12)
    beta_argument = (target_range * sin_theta).square() - alpha.square()
    beta = look_sign * torch.sqrt(torch.clamp(beta_argument, min=0.0))
    final_xyz = sat + alpha.unsqueeze(-1) * along_track
    final_xyz = final_xyz + beta.unsqueeze(-1) * cross_track
    final_xyz = final_xyz + gamma.unsqueeze(-1) * normal
    final_latitude, final_longitude, _ = _ecef_to_llh(final_xyz)
    final_look = final_xyz - sat
    final_distance = torch.linalg.vector_norm(final_look, dim=-1)
    final_unit = final_look / torch.clamp(final_distance, min=1.0e-12).unsqueeze(-1)
    final_range_residual = final_distance - target_range
    final_doppler = 2.0 * torch.sum(velocity * final_unit, dim=-1) / wavelength_m
    final_valid = (
        finite
        & ~failed
        & torch.isfinite(final_range_residual)
        & (final_distance > 0.0)
    )
    solved &= final_valid & (torch.abs(final_range_residual) < range_tol_m)
    range_residual = torch.where(finite, final_range_residual, range_residual)
    doppler_residual = torch.where(finite, final_doppler, doppler_residual)
    output = {
        "latitude_deg": torch.where(solved, final_latitude, latitude),
        "longitude_deg": torch.where(solved, final_longitude, longitude),
        "height_m": height,
        "range_index": range_index,
        "azimuth_index": azimuth,
        "converged": solved,
        "iterations": torch.where(
            solved | (finite & ~failed), attempts, torch.full_like(attempts, -1)
        ),
        "residual_range_m": range_residual,
        "residual_doppler_hz": doppler_residual,
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
    dem_samples: Any | None = None,
    dem_latitude_start_deg: float = 0.0,
    dem_longitude_start_deg: float = 0.0,
    dem_latitude_spacing_deg: float = 1.0,
    dem_longitude_spacing_deg: float = 1.0,
    dem_iterations: int = 1,
    dem_height_tol_m: float = 1.0e-3,
    dem_height_m: float | None = None,
) -> dict[str, Any]:
    """Solve rdr2geo with native primary-solve/DEM fixed-point semantics."""
    import torch

    if dem_samples is None:
        fixed_height = (
            torch.full_like(height_seed, dem_height_m)
            if dem_height_m is not None
            else height_seed
        )
        return _rdr2geo_once(
            azimuth, range_index, fixed_height, orbit_times, orbit_positions,
            orbit_velocities, sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=azimuth_interval_s, starting_range_m=starting_range_m,
            range_spacing_m=range_spacing_m, wavelength_m=wavelength_m,
            look_sign=look_sign, max_iter=max_iter, range_tol_m=range_tol_m,
            doppler_tol_hz=doppler_tol_hz, dynamic_iterations=dynamic_iterations,
            dem_height_m=dem_height_m,
        )

    heights = height_seed
    result: dict[str, Any] = {}
    frozen = torch.zeros((), dtype=torch.bool, device=heights.device)
    for _ in range(dem_iterations):
        result = _rdr2geo_once(
            azimuth, range_index, heights, orbit_times, orbit_positions,
            orbit_velocities, sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=azimuth_interval_s, starting_range_m=starting_range_m,
            range_spacing_m=range_spacing_m, wavelength_m=wavelength_m,
            look_sign=look_sign, max_iter=max_iter, range_tol_m=range_tol_m,
            doppler_tol_hz=doppler_tol_hz, dynamic_iterations=dynamic_iterations,
        )
        if dem_samples is not None:
            next_heights = _sample_dem_six(
                dem_samples, result["latitude_deg"], result["longitude_deg"],
                dem_latitude_start_deg, dem_longitude_start_deg,
                dem_latitude_spacing_deg, dem_longitude_spacing_deg,
            )
        else:
            next_heights = torch.full_like(heights, dem_height_m)
        valid = result["converged"] & torch.isfinite(next_heights)
        next_heights = torch.where(
            valid, next_heights, torch.full_like(next_heights, torch.nan)
        )
        update = torch.abs(next_heights - heights)
        heights = torch.where(frozen, heights, next_heights)
        global_done = torch.all(
            (~result["converged"]) | (update < dem_height_tol_m)
        )
        frozen = frozen | global_done
        if dynamic_iterations and bool(
            global_done.item()
        ):
            break
    result = _rdr2geo_once(
        azimuth, range_index, heights, orbit_times, orbit_positions,
        orbit_velocities, sensing_offset_s=sensing_offset_s,
        azimuth_interval_s=azimuth_interval_s, starting_range_m=starting_range_m,
        range_spacing_m=range_spacing_m, wavelength_m=wavelength_m,
        look_sign=look_sign, max_iter=max_iter, range_tol_m=range_tol_m,
        doppler_tol_hz=doppler_tol_hz, dynamic_iterations=dynamic_iterations,
    )
    result["height_m"] = heights
    return result


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
