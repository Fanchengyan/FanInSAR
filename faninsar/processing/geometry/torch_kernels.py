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


def _spline_six_weights(fraction: Any) -> Any:
    """Return closed-form natural-spline weights for six samples."""
    import torch

    second_one = (
        1.6076555023923444,
        -3.6459330143540667,
        2.5837320574162677,
        -0.6889952153110048,
        0.1722488038277512,
        -0.0287081339712919,
    )
    second_two = (
        -0.4306220095693780,
        2.5837320574162677,
        -4.3349282296650715,
        2.7559808612440193,
        -0.6889952153110048,
        0.1148325358851674,
    )
    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    weights = [
        fraction * (-one / 3.0 - two / 6.0)
        + fraction2 * (one / 2.0)
        + fraction3 * (two - one) / 6.0
        for one, two in zip(second_one, second_two, strict=True)
    ]
    weights[1] = weights[1] + 1.0 - fraction
    weights[2] = weights[2] + fraction
    return torch.stack(weights, dim=-1)


def _natural_spline_six(values: Any, fraction: Any) -> Any:
    """Evaluate the local six-sample natural spline on Torch tensors."""
    import torch

    return torch.sum(values * _spline_six_weights(fraction), dim=-1)


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
    # A six-point stencil has no valid base when either raster dimension is
    # smaller than six.  Return the public invalid sentinel before clamping;
    # otherwise ``clamp(1, rows - 5)`` can produce a negative gather index for
    # a degenerate tile.
    if rows < 6 or columns < 6:
        return torch.full_like(row, torch.nan)
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
    column_fraction = column - safe_column
    row_fraction = row - safe_row
    column_weights = _spline_six_weights(column_fraction)
    # ``unfold`` exposes every six-by-six stencil as a view.  Indexing that
    # view avoids constructing two broadcasted integer index tensors and a
    # flattened gather for every active lane while preserving the exact
    # row-major sample window used by the native kernel.
    windows = dem_samples.unfold(0, 6, 1).unfold(1, 6, 1)
    window = windows[safe_row - 1, safe_column - 1]
    row_values = torch.sum(window * column_weights.unsqueeze(-2), dim=-1)
    row_weights = _spline_six_weights(row_fraction)
    sampled = torch.sum(row_values * row_weights, dim=-1)
    return torch.where(
        valid & torch.isfinite(sampled), sampled, torch.full_like(sampled, torch.nan)
    )


def _guarded_aitken_height(
    previous_previous: Any,
    previous: Any,
    current: Any,
    dem_min: Any,
    dem_max: Any,
) -> tuple[Any, Any]:
    """Apply a guarded scalar Aitken delta-squared height update.

    Parameters
    ----------
    previous_previous, previous, current : torch.Tensor
        Three consecutive fixed-point height states.  ``current`` is the
        unaccelerated height for the current DEM update.
    dem_min, dem_max : torch.Tensor
        Finite materialized DEM height bounds.

    Returns
    -------
    candidate : torch.Tensor
        Aitken's candidate where every guard passes, otherwise ``current``.
    enabled : torch.Tensor
        Boolean mask identifying accepted candidates.

    Notes
    -----
    The local fixed-point slope is restricted to the conservative contracting
    interval ``(-0.95, 0.95)``.  The candidate must move in the same direction
    as the current fixed-point step, remain within one step (or the 1 mm
    convergence scale),
    and stay inside the materialized DEM range.  All guards are tensor
    predicates, so eager and fixed-shape compiled execution share exactly the
    same branch-free semantics.

    """
    import torch

    denominator = current - 2.0 * previous + previous_previous
    step = current - previous
    previous_step = previous - previous_previous
    slope = step / previous_step
    candidate = previous_previous - previous_step.square() / denominator
    candidate_step = candidate - previous
    denominator_safe = torch.abs(denominator) > 1.0e-6
    previous_step_safe = torch.abs(previous_step) > 1.0e-12
    finite = (
        torch.isfinite(previous_previous)
        & torch.isfinite(previous)
        & torch.isfinite(current)
        & torch.isfinite(dem_min)
        & torch.isfinite(dem_max)
        & torch.isfinite(slope)
        & torch.isfinite(candidate)
    )
    enabled = (
        finite
        & previous_step_safe
        & denominator_safe
        & (slope > -0.95)
        & (slope < 0.95)
        & (dem_min <= dem_max)
        & (candidate_step.abs() <= step.abs().clamp_min(1.0e-3))
        & (candidate_step * step >= 0.0)
        & (candidate >= dem_min)
        & (candidate <= dem_max)
    )
    return torch.where(enabled, candidate, current), enabled


def _result_invalid(output: dict[str, Any], finite: Any) -> dict[str, Any]:
    """Apply v2 invalid-lane sentinels to a tensor result."""
    import torch

    for name in (
        "range_index",
        "azimuth_index",
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "ecef_x_m",
        "ecef_y_m",
        "ecef_z_m",
        "residual_range_m",
        "residual_doppler_hz",
    ):
        if name not in output:
            continue
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


def _geo2rdr_initialize(
    latitude: Any,
    longitude: Any,
    height: Any,
    orbit_times: Any,
    orbit_positions: Any,
    orbit_velocities: Any,
    *,
    sensing_offset_s: float,
) -> tuple[Any, ...]:
    """Initialize the fixed-shape Geo2Rdr Newton state."""
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
    sat, velocity, acceleration = _orbit_state(
        times, orbit_times, orbit_positions, orbit_velocities
    )
    solved = torch.zeros_like(finite)
    failed = ~finite
    attempts = torch.zeros_like(latitude, dtype=torch.int32)
    doppler = torch.full_like(latitude, torch.nan)
    range_residual = torch.full_like(latitude, torch.nan)
    evaluated_distance = torch.full_like(latitude, torch.nan)
    return (
        target,
        finite,
        times,
        sat,
        velocity,
        acceleration,
        solved,
        failed,
        attempts,
        doppler,
        range_residual,
        evaluated_distance,
    )


def _geo2rdr_active(state: tuple[Any, ...]) -> Any:
    """Return valid lanes that still require a Newton transition."""
    finite, solved, failed = state[1], state[6], state[7]
    return finite & ~solved & ~failed


def _geo2rdr_step(
    state: tuple[Any, ...],
    orbit_times: Any,
    orbit_positions: Any,
    orbit_velocities: Any,
    *,
    starting_range_m: float,
    range_spacing_m: float,
    wavelength_m: float,
    range_tol_m: float,
    doppler_tol_hz: float,
) -> tuple[Any, ...]:
    """Advance one fixed-shape Geo2Rdr Newton transition."""
    import torch

    (
        target,
        finite,
        times,
        sat,
        velocity,
        acceleration,
        solved,
        failed,
        attempts,
        doppler,
        range_residual,
        evaluated_distance,
    ) = state
    active = finite & ~solved & ~failed
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
    attempts = torch.where(active, attempts + 1, attempts)
    step = -radial / derivative
    next_times = torch.where(active & valid, times + step, times)
    orbit_start, orbit_end = orbit_times[0], orbit_times[-1]
    in_bounds = (next_times >= orbit_start) & (next_times <= orbit_end)
    failed = failed | (active & (~valid | ~torch.isfinite(step) | ~in_bounds))
    final_sat, final_velocity, final_acceleration = _orbit_state(
        next_times, orbit_times, orbit_positions, orbit_velocities
    )
    final_look = target - final_sat
    final_distance = torch.linalg.vector_norm(final_look, dim=-1)
    final_unit = final_look / torch.clamp(final_distance, min=1.0e-12).unsqueeze(-1)
    final_doppler = 2.0 * torch.sum(final_velocity * final_unit, dim=-1) / wavelength_m
    final_range_index = (final_distance - starting_range_m) / range_spacing_m
    final_range = starting_range_m + final_range_index * range_spacing_m
    final_range_residual = final_distance - final_range
    metric = torch.maximum(
        torch.abs(final_range_residual) / range_tol_m,
        torch.abs(final_doppler) / doppler_tol_hz,
    )
    newly = active & valid & in_bounds & torch.isfinite(metric) & (metric < 1.0)
    solved = solved | newly
    return (
        target,
        finite,
        next_times,
        final_sat,
        final_velocity,
        final_acceleration,
        solved,
        failed,
        attempts,
        torch.where(active, final_doppler, doppler),
        torch.where(active, final_range_residual, range_residual),
        torch.where(active, final_distance, evaluated_distance),
    )


def _geo2rdr_finalize(
    state: tuple[Any, ...],
    latitude: Any,
    longitude: Any,
    height: Any,
    *,
    sensing_offset_s: float,
    azimuth_interval_s: float,
    starting_range_m: float,
    range_spacing_m: float,
) -> dict[str, Any]:
    """Publish one Geo2Rdr state with canonical invalid sentinels."""
    import torch

    finite = state[1]
    times = state[2]
    solved = state[6]
    failed = state[7]
    attempts = state[8]
    doppler = state[9]
    range_residual = state[10]
    evaluated_distance = state[11]
    # ``evaluated_distance`` comes from the last
    # post-update evaluation, which is also the state used above for stopping.
    computed_range_index = (evaluated_distance - starting_range_m) / range_spacing_m
    computed_azimuth_index = (times - sensing_offset_s) / azimuth_interval_s
    nan = torch.full_like(computed_range_index, torch.nan)
    output = {
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "height_m": height,
        "range_index": torch.where(solved, computed_range_index, nan),
        "azimuth_index": torch.where(solved, computed_azimuth_index, nan),
        "converged": solved,
        "max_iter_exhausted": finite & ~failed & ~solved,
        "iterations": torch.where(
            solved | (finite & ~failed), attempts, torch.full_like(attempts, -1)
        ),
        "residual_range_m": range_residual,
        "residual_doppler_hz": doppler,
    }
    return _result_invalid(output, finite)


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
    """Solve Geo2Rdr with acceleration-aware Newton iterations on Torch."""
    import torch

    del time_tol_s
    state = _geo2rdr_initialize(
        latitude,
        longitude,
        height,
        orbit_times,
        orbit_positions,
        orbit_velocities,
        sensing_offset_s=sensing_offset_s,
    )
    for _ in range(max_iter):
        if dynamic_iterations and not bool(torch.any(_geo2rdr_active(state)).item()):
            break
        state = _geo2rdr_step(
            state,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            starting_range_m=starting_range_m,
            range_spacing_m=range_spacing_m,
            wavelength_m=wavelength_m,
            range_tol_m=range_tol_m,
            doppler_tol_hz=doppler_tol_hz,
        )
    return _geo2rdr_finalize(
        state,
        latitude,
        longitude,
        height,
        sensing_offset_s=sensing_offset_s,
        azimuth_interval_s=azimuth_interval_s,
        starting_range_m=starting_range_m,
        range_spacing_m=range_spacing_m,
    )


def _rdr2geo_geometry_state_for_context(
    azimuth: Any,
    range_index: Any,
    orbit_times: Any,
    orbit_positions: Any,
    orbit_velocities: Any,
    *,
    sensing_offset_s: float,
    azimuth_interval_s: float,
    starting_range_m: float,
    range_spacing_m: float,
    context_azimuth: Any,
) -> tuple[Any, ...]:
    """Build geometry state from point inputs and a reusable azimuth context.

    ``context_azimuth`` may contain one value per row for a regular two
    dimensional radar grid.  The resulting context tensors are expanded to
    the point shape, so the numerical solver keeps its existing elementwise
    contract.
    """
    import torch

    target_range = starting_range_m + range_index * range_spacing_m
    times = sensing_offset_s + azimuth * azimuth_interval_s
    context_times = sensing_offset_s + context_azimuth * azimuth_interval_s
    orbit_start, orbit_end = orbit_times[0], orbit_times[-1]
    sat, velocity, _ = _orbit_state(
        context_times, orbit_times, orbit_positions, orbit_velocities
    )
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    satellite_norm = torch.linalg.vector_norm(sat, dim=-1)
    context_finite = (
        torch.isfinite(context_azimuth)
        & torch.isfinite(context_times)
        & (context_times >= orbit_start)
        & (context_times <= orbit_end)
        & (speed > 0.0)
        & (satellite_norm > 0.0)
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
    context_finite &= torch.isfinite(normal_dot_velocity) & (
        velocity_dot_along > 1.0e-12
    )
    minor = _A * torch.sqrt(
        torch.as_tensor(1.0 - _E2, dtype=sat.dtype, device=sat.device)
    )
    eta = 1.0 / torch.sqrt(
        (sat[..., 0] / _A).square()
        + (sat[..., 1] / _A).square()
        + (sat[..., 2] / minor).square()
    )
    radius = eta * satellite_norm
    ellipsoid_height = (1.0 - eta) * satellite_norm

    if context_azimuth.shape != azimuth.shape:
        expand_shape = (*azimuth.shape,)

        def expand(value: Any) -> Any:
            """Expand a row context to the point grid."""
            # ``torch.cond`` requires both branches to publish matching
            # strides.  Materialize only after the cheaper row-wise orbit and
            # frame evaluation; this preserves the fixed point-shaped state
            # contract without a host roundtrip.
            return value.expand(*expand_shape, *value.shape[2:]).contiguous()

        sat = expand(sat)
        velocity = expand(velocity)
        satellite_norm = expand(satellite_norm)
        normal = expand(normal)
        cross_track = expand(cross_track)
        along_track = expand(along_track)
        normal_dot_velocity = expand(normal_dot_velocity)
        velocity_dot_along = expand(velocity_dot_along)
        radius = expand(radius)
        ellipsoid_height = expand(ellipsoid_height)
        context_finite = expand(context_finite)

    finite = (
        torch.isfinite(target_range)
        & (target_range > 0.0)
        & torch.isfinite(azimuth)
        & torch.isfinite(range_index)
        & torch.isfinite(times)
        & (times >= orbit_start)
        & (times <= orbit_end)
        & context_finite
    )
    return (
        target_range,
        sat,
        velocity,
        satellite_norm,
        normal,
        cross_track,
        along_track,
        normal_dot_velocity,
        velocity_dot_along,
        radius,
        ellipsoid_height,
        finite,
    )


def _rdr2geo_geometry_state(
    azimuth: Any,
    range_index: Any,
    orbit_times: Any,
    orbit_positions: Any,
    orbit_velocities: Any,
    *,
    sensing_offset_s: float,
    azimuth_interval_s: float,
    starting_range_m: float,
    range_spacing_m: float,
) -> tuple[Any, ...]:
    """Build immutable device state shared by all TCN/DEM iterations.

    The old high-throughput Torch solver accepted satellite and velocity as
    device-resident inputs.  The prepared public API accepts radar indices
    instead, so this equivalent state is materialized once per invocation and
    then reused by every fixed-point iteration.  Keeping this work outside
    :func:`_rdr2geo_once` avoids rebuilding the TCN basis for each DEM update.
    """
    import torch

    if azimuth.ndim != 2 or azimuth.shape[1] <= 1:
        return _rdr2geo_geometry_state_for_context(
            azimuth,
            range_index,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=azimuth_interval_s,
            starting_range_m=starting_range_m,
            range_spacing_m=range_spacing_m,
            context_azimuth=azimuth,
        )

    row_azimuth = azimuth[:, :1]
    regular_rows = torch.all(azimuth == row_azimuth)

    def row_context(azimuth_value: Any, range_value: Any) -> tuple[Any, ...]:
        """Evaluate orbit/TCN context once per regular-grid row."""
        return _rdr2geo_geometry_state_for_context(
            azimuth_value,
            range_value,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=azimuth_interval_s,
            starting_range_m=starting_range_m,
            range_spacing_m=range_spacing_m,
            context_azimuth=azimuth_value[:, :1],
        )

    def point_context(azimuth_value: Any, range_value: Any) -> tuple[Any, ...]:
        """Evaluate the exact point-wise context for irregular rows."""
        return _rdr2geo_geometry_state_for_context(
            azimuth_value,
            range_value,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=azimuth_interval_s,
            starting_range_m=starting_range_m,
            range_spacing_m=range_spacing_m,
            context_azimuth=azimuth_value,
        )

    if torch.compiler.is_compiling():
        return point_context(azimuth, range_index)
    if bool(regular_rows.item()):
        return row_context(azimuth, range_index)
    return point_context(azimuth, range_index)


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
    extra_iter: int = 0,
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
    geometry_state: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Solve rdr2geo with the native closed-form TCN construction."""
    import torch

    # CPU benefits from compacting unfinished lanes.  On CUDA, ``nonzero`` and
    # ``index_copy_`` force a host-visible dynamic shape on every iteration;
    # keep the fixed-shape masked loop there so the device can run
    # asynchronously.  Compilation already takes the fixed-shape path below.
    if (
        dynamic_iterations
        and dem_samples is not None
        and azimuth.device.type == "cpu"
        and not torch.compiler.is_compiling()
    ):
        return _rdr2geo_once_eager_active(
            azimuth,
            range_index,
            height_seed,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=azimuth_interval_s,
            starting_range_m=starting_range_m,
            range_spacing_m=range_spacing_m,
            wavelength_m=wavelength_m,
            look_sign=look_sign,
            max_iter=max_iter,
            extra_iter=extra_iter,
            range_tol_m=range_tol_m,
            doppler_tol_hz=doppler_tol_hz,
            dynamic_iterations=dynamic_iterations,
            dem_samples=dem_samples,
            dem_latitude_start_deg=dem_latitude_start_deg,
            dem_longitude_start_deg=dem_longitude_start_deg,
            dem_latitude_spacing_deg=dem_latitude_spacing_deg,
            dem_longitude_spacing_deg=dem_longitude_spacing_deg,
            dem_iterations=dem_iterations,
            dem_height_tol_m=dem_height_tol_m,
            dem_height_m=dem_height_m,
            geometry_state=geometry_state,
        )

    # These compatibility arguments remain part of the prepared adapter
    # contract; Doppler and the legacy outer DEM budget do not control this
    # closed-form TCN loop.
    _ = (doppler_tol_hz, dem_iterations)

    if geometry_state is None:
        geometry_state = _rdr2geo_geometry_state(
            azimuth,
            range_index,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=azimuth_interval_s,
            starting_range_m=starting_range_m,
            range_spacing_m=range_spacing_m,
        )
    (
        target_range,
        sat,
        velocity,
        satellite_norm,
        normal,
        cross_track,
        along_track,
        normal_dot_velocity,
        velocity_dot_along,
        radius,
        ellipsoid_height,
        finite,
    ) = geometry_state
    finite = finite & torch.isfinite(height_seed)
    height = height_seed
    if dem_samples is not None:
        dem_min = torch.amin(dem_samples)
        dem_max = torch.amax(dem_samples)
    latitude = torch.full_like(azimuth, torch.nan)
    longitude = torch.full_like(azimuth, torch.nan)
    solved = torch.zeros_like(finite)
    failed = ~finite
    dem_invalid = torch.zeros_like(finite)
    attempts = torch.zeros_like(azimuth, dtype=torch.int32)
    range_residual = torch.full_like(azimuth, torch.nan)
    doppler_residual = torch.full_like(azimuth, torch.nan)
    old_latitude = torch.full_like(azimuth, torch.nan)
    old_longitude = torch.full_like(azimuth, torch.nan)
    old_height = torch.full_like(azimuth, torch.nan)
    if dem_samples is not None:
        previous_height = torch.full_like(height, torch.nan)
        previous_fixed_height = torch.full_like(height, torch.nan)
        aitken_restart = torch.zeros_like(finite)
    for attempt in range(1, max_iter + extra_iter + 1):
        active = finite & ~solved & ~failed & (ellipsoid_height - height < target_range)
        failed |= finite & ~solved & ~active
        semi_minor = radius + height
        cos_theta = 0.5 * (
            satellite_norm / target_range
            + target_range / satellite_norm
            - (semi_minor / satellite_norm) * (semi_minor / target_range)
        )
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.square(), min=0.0))
        gamma = target_range * cos_theta
        alpha = (
            -gamma * normal_dot_velocity / torch.clamp(velocity_dot_along, min=1.0e-12)
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
                else height_seed
            )
        dem_xyz = _llh_to_ecef(latitude_candidate, longitude_candidate, dem_height)
        look = dem_xyz - sat
        slant_range = torch.linalg.vector_norm(look, dim=-1)
        # Doppler is a publication residual; convergence uses range and DEM
        # height, so defer its normalization to the final TCN pass below.
        rr = slant_range - target_range
        new_height = torch.linalg.vector_norm(dem_xyz, dim=-1) - radius
        next_height = new_height
        valid &= torch.isfinite(latitude_candidate) & torch.isfinite(
            longitude_candidate
        )
        valid &= (
            torch.isfinite(slant_range)
            & (slant_range > 0.0)
            & torch.isfinite(next_height)
            & torch.isfinite(dem_height)
        )
        if dem_samples is not None:
            dem_invalid |= active & ~torch.isfinite(dem_height)
        attempts = torch.where(active, torch.full_like(attempts, attempt), attempts)
        range_residual = torch.where(active, rr, range_residual)
        failed |= active & ~valid
        if dem_samples is None:
            dem_height_converged = torch.ones_like(active)
        else:
            dem_height_converged = torch.isfinite(next_height) & (
                torch.abs(next_height - height) <= dem_height_tol_m
            )
        newly_solved = (
            active
            & valid
            & torch.isfinite(rr)
            & (torch.abs(rr) < range_tol_m)
            & dem_height_converged
        )
        solved |= newly_solved
        next_latitude = latitude_candidate
        next_longitude = longitude_candidate
        next_llh_height = dem_height
        converged_height = new_height if dem_samples is not None else dem_height
        if dem_samples is None or attempt < 3:
            aitken_height = new_height
            aitken_enabled = torch.zeros_like(active)
        else:
            aitken_height, aitken_enabled = _guarded_aitken_height(
                previous_height,
                height,
                new_height,
                dem_min,
                dem_max,
            )
            prior_residual = previous_fixed_height - previous_height
            current_residual = new_height - height
            sign_change = prior_residual * current_residual <= 0.0
            between = (aitken_height - previous_height) * (
                aitken_height - height
            ) <= 0.0
            aitken_enabled &= (
                active
                & valid
                & torch.isfinite(previous_fixed_height)
                & ((~sign_change) | between)
            )
            aitken_enabled &= ~aitken_restart
        damping_valid = torch.zeros_like(active)
        restart_after_damping = torch.zeros_like(active)
        if attempt - 1 >= max_iter:
            old_xyz = _llh_to_ecef(old_latitude, old_longitude, old_height)
            average_xyz = 0.5 * (old_xyz + dem_xyz)
            (
                average_latitude,
                average_longitude,
                average_llh_height,
            ) = _ecef_to_llh(average_xyz)
            average_height = torch.linalg.vector_norm(average_xyz, dim=-1) - radius
            damping_valid = (
                active
                & valid
                & ~newly_solved
                & torch.isfinite(average_latitude)
                & torch.isfinite(average_longitude)
                & torch.isfinite(average_llh_height)
                & torch.isfinite(average_height)
            )
            next_latitude = torch.where(damping_valid, average_latitude, next_latitude)
            next_longitude = torch.where(
                damping_valid, average_longitude, next_longitude
            )
            next_llh_height = torch.where(
                damping_valid, average_llh_height, next_llh_height
            )
            next_height = torch.where(damping_valid, average_height, next_height)
            aitken_enabled &= ~damping_valid
        if dem_samples is not None:
            restart_after_damping = damping_valid & ~aitken_restart
        selected_height = torch.where(aitken_enabled, aitken_height, next_height)
        state_height = torch.where(
            newly_solved,
            converged_height,
            torch.where(active & valid, selected_height, height),
        )
        old_latitude = torch.where(active & valid, next_latitude, old_latitude)
        old_longitude = torch.where(active & valid, next_longitude, old_longitude)
        old_height = torch.where(active & valid, next_llh_height, old_height)
        latitude = torch.where(active, next_latitude, latitude)
        longitude = torch.where(active, next_longitude, longitude)
        history_valid = active & valid
        if dem_samples is not None:
            previous_height = torch.where(history_valid, height, previous_height)
            previous_fixed_height = torch.where(
                history_valid, new_height, previous_fixed_height
            )
            previous_height = torch.where(
                restart_after_damping,
                torch.full_like(previous_height, torch.nan),
                previous_height,
            )
            previous_fixed_height = torch.where(
                restart_after_damping,
                torch.full_like(previous_fixed_height, torch.nan),
                previous_fixed_height,
            )
            aitken_restart = torch.where(active, restart_after_damping, aitken_restart)
        height = torch.where(
            active & valid,
            state_height,
            height,
        )
        if (
            dynamic_iterations
            and azimuth.device.type == "cpu"
            and bool(torch.all(solved | failed | ~finite).item())
        ):
            break
    semi_minor = radius + height
    cos_theta = 0.5 * (
        satellite_norm / target_range
        + target_range / satellite_norm
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
    final_latitude, final_longitude, final_height = _ecef_to_llh(final_xyz)
    final_look = final_xyz - sat
    final_distance = torch.linalg.vector_norm(final_look, dim=-1)
    final_unit = final_look / torch.clamp(final_distance, min=1.0e-12).unsqueeze(-1)
    final_range_residual = final_distance - target_range
    final_doppler = 2.0 * torch.sum(velocity * final_unit, dim=-1) / wavelength_m
    final_valid = (
        finite
        & ~failed
        & torch.isfinite(final_latitude)
        & torch.isfinite(final_longitude)
        & torch.isfinite(final_height)
        & torch.isfinite(final_range_residual)
        & (final_distance > 0.0)
    )
    if dem_samples is None:
        final_dem_valid = torch.ones_like(final_valid)
        final_dem_converged = torch.ones_like(final_valid)
    else:
        final_dem_height = _sample_dem_six(
            dem_samples,
            final_latitude,
            final_longitude,
            dem_latitude_start_deg,
            dem_longitude_start_deg,
            dem_latitude_spacing_deg,
            dem_longitude_spacing_deg,
        )
        final_dem_valid = torch.isfinite(final_dem_height)
        final_dem_converged = final_dem_valid & (
            torch.abs(final_height - final_dem_height) <= dem_height_tol_m
        )
        final_valid &= final_dem_valid
    # Re-evaluate the public convergence vote against the coordinates and DEM
    # height that are actually published.  A lane can reach both tolerances on
    # the final TCN publication after its last DEM update.
    solved = (
        final_valid
        & (torch.abs(final_range_residual) < range_tol_m)
        & final_dem_converged
    )
    # ``final_valid`` owns the public validity vote, including the final DEM
    # recheck.  A failed recheck may still leave a finite, authoritative TCN
    # coordinate triple; preserve those coordinates while publishing
    # ``converged=False`` and the corresponding exhausted status.
    result_finite = (final_valid | (~final_dem_valid & finite & ~failed)) & ~dem_invalid
    range_residual = torch.where(result_finite, final_range_residual, range_residual)
    doppler_residual = torch.where(result_finite, final_doppler, doppler_residual)
    output = {
        # Publish one final TCN state for both converged and exhausted valid
        # lanes.  Publishing the last DEM iterate for lat/lon together with
        # this final ellipsoid height creates a mixed coordinate triple.
        "latitude_deg": final_latitude,
        "longitude_deg": final_longitude,
        "height_m": final_height if dem_samples is not None else height,
        "range_index": range_index,
        "azimuth_index": azimuth,
        "converged": solved,
        "iterations": torch.where(
            solved | (finite & ~failed), attempts, torch.full_like(attempts, -1)
        ),
        "residual_range_m": range_residual,
        "residual_doppler_hz": doppler_residual,
    }
    # ``final_xyz`` is the authoritative TCN point used for the residuals and
    # is already device-resident.  Publish it directly instead of rebuilding
    # ECEF from LLH on the host (or evaluating a second trigonometric pass).
    published_ecef = final_xyz
    output.update(
        {
            "ecef_x_m": published_ecef[..., 0],
            "ecef_y_m": published_ecef[..., 1],
            "ecef_z_m": published_ecef[..., 2],
        }
    )
    return _result_invalid(output, result_finite)


def _rdr2geo_once_eager_active(
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
    extra_iter: int = 0,
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
    geometry_state: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Solve one eager TCN pass by gathering unfinished lanes.

    The native equations and publication rules are kept identical to the
    fixed-shape solver.  Only the tensors participating in unfinished lanes
    are carried through each iteration; final TCN publication is still done
    on the original shape so callers observe the same result contract.
    """
    import torch

    del dynamic_iterations
    del doppler_tol_hz, dem_iterations

    original_shape = azimuth.shape
    azimuth_ndim = azimuth.ndim

    def flatten(value: Any) -> Any:
        """Flatten lane dimensions while retaining trailing vector axes."""
        trailing_shape = value.shape[azimuth_ndim:]
        return value.reshape(-1, *trailing_shape)

    if geometry_state is None:
        # This call is intentionally made only for an unprepared direct use.
        geometry_state = _rdr2geo_geometry_state(
            azimuth,
            range_index,
            orbit_times,
            orbit_positions,
            orbit_velocities,
            sensing_offset_s=sensing_offset_s,
            azimuth_interval_s=azimuth_interval_s,
            starting_range_m=starting_range_m,
            range_spacing_m=range_spacing_m,
        )
    (
        target_range,
        sat,
        velocity,
        satellite_norm,
        normal,
        cross_track,
        along_track,
        normal_dot_velocity,
        velocity_dot_along,
        radius,
        ellipsoid_height,
        finite,
    ) = (flatten(value) for value in geometry_state)
    height_seed = flatten(height_seed)
    azimuth = flatten(azimuth)
    range_index = flatten(range_index)
    target_range = flatten(target_range)
    finite = flatten(finite) & torch.isfinite(height_seed)
    height = height_seed.clone()
    if dem_samples is not None:
        dem_min = torch.amin(dem_samples)
        dem_max = torch.amax(dem_samples)
    solved = torch.zeros_like(finite)
    failed = ~finite
    dem_invalid = torch.zeros_like(finite)
    attempts = torch.zeros_like(azimuth, dtype=torch.int32)
    range_residual = torch.full_like(azimuth, torch.nan)
    doppler_residual = torch.full_like(azimuth, torch.nan)
    old_latitude = torch.full_like(azimuth, torch.nan)
    old_longitude = torch.full_like(azimuth, torch.nan)
    old_height = torch.full_like(azimuth, torch.nan)
    if dem_samples is not None:
        previous_height = torch.full_like(height, torch.nan)
        previous_fixed_height = torch.full_like(height, torch.nan)
        aitken_restart = torch.zeros_like(finite)

    total_iterations = max_iter + extra_iter
    initial_active = finite & (ellipsoid_height - height < target_range)
    failed |= finite & ~initial_active
    active_indices = torch.nonzero(initial_active, as_tuple=False).flatten()
    for attempt in range(1, total_iterations + 1):
        if active_indices.numel() == 0:
            break

        target_range_active = target_range[active_indices]
        sat_active = sat[active_indices]
        satellite_norm_active = satellite_norm[active_indices]
        normal_active = normal[active_indices]
        cross_track_active = cross_track[active_indices]
        along_track_active = along_track[active_indices]
        normal_dot_velocity_active = normal_dot_velocity[active_indices]
        velocity_dot_along_active = velocity_dot_along[active_indices]
        radius_active = radius[active_indices]
        ellipsoid_height_active = ellipsoid_height[active_indices]
        height_active = height[active_indices]

        semi_minor = radius_active + height_active
        cos_theta = 0.5 * (
            satellite_norm_active / target_range_active
            + target_range_active / satellite_norm_active
            - (semi_minor / satellite_norm_active) * (semi_minor / target_range_active)
        )
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.square(), min=0.0))
        gamma = target_range_active * cos_theta
        alpha = (
            -gamma
            * normal_dot_velocity_active
            / torch.clamp(velocity_dot_along_active, min=1.0e-12)
        )
        beta_argument = (target_range_active * sin_theta).square() - alpha.square()
        valid = torch.isfinite(beta_argument) & (beta_argument >= -1.0e-6)
        beta = look_sign * torch.sqrt(torch.clamp(beta_argument, min=0.0))
        target_xyz = sat_active + alpha.unsqueeze(-1) * along_track_active
        target_xyz = target_xyz + beta.unsqueeze(-1) * cross_track_active
        target_xyz = target_xyz + gamma.unsqueeze(-1) * normal_active
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
                torch.full_like(height_active, dem_height_m)
                if dem_height_m is not None
                else height_seed[active_indices]
            )
        dem_xyz = _llh_to_ecef(latitude_candidate, longitude_candidate, dem_height)
        look = dem_xyz - sat_active
        slant_range = torch.linalg.vector_norm(look, dim=-1)
        # The final pass recomputes Doppler for every finite lane; avoid doing
        # the same normalization for each intermediate DEM iterate.
        rr = slant_range - target_range_active
        new_height = torch.linalg.vector_norm(dem_xyz, dim=-1) - radius_active
        next_height = new_height
        valid &= torch.isfinite(latitude_candidate) & torch.isfinite(
            longitude_candidate
        )
        valid &= (
            torch.isfinite(slant_range)
            & (slant_range > 0.0)
            & torch.isfinite(next_height)
            & torch.isfinite(dem_height)
        )
        if dem_samples is not None:
            dem_invalid.index_fill_(
                0,
                active_indices[~torch.isfinite(dem_height)],
                True,
            )
        attempts.index_copy_(
            0,
            active_indices,
            torch.full_like(attempts[active_indices], attempt),
        )
        range_residual.index_copy_(0, active_indices, rr)
        failed.index_fill_(0, active_indices[~valid], True)
        if dem_samples is None:
            dem_height_converged = torch.ones_like(valid)
        else:
            dem_height_converged = torch.isfinite(next_height) & (
                torch.abs(next_height - height_active) <= dem_height_tol_m
            )
        newly_solved = (
            valid
            & torch.isfinite(rr)
            & (torch.abs(rr) < range_tol_m)
            & dem_height_converged
        )
        solved.index_fill_(0, active_indices[newly_solved], True)

        next_latitude = latitude_candidate
        next_longitude = longitude_candidate
        next_llh_height = dem_height
        converged_height = new_height if dem_samples is not None else dem_height
        if dem_samples is None or attempt < 3:
            aitken_height = new_height
            aitken_enabled = torch.zeros_like(valid)
        else:
            aitken_height, aitken_enabled = _guarded_aitken_height(
                previous_height[active_indices],
                height_active,
                new_height,
                dem_min,
                dem_max,
            )
            previous_active = previous_height[active_indices]
            previous_fixed_active = previous_fixed_height[active_indices]
            prior_residual = previous_fixed_active - previous_active
            current_residual = new_height - height_active
            sign_change = prior_residual * current_residual <= 0.0
            between = (aitken_height - previous_active) * (
                aitken_height - height_active
            ) <= 0.0
            aitken_enabled &= (
                valid
                & torch.isfinite(previous_fixed_active)
                & ((~sign_change) | between)
            )
            aitken_enabled &= ~aitken_restart[active_indices]
        damping_valid = torch.zeros_like(valid)
        restart_after_damping = torch.zeros_like(valid)
        if attempt - 1 >= max_iter:
            old_xyz = _llh_to_ecef(
                old_latitude[active_indices],
                old_longitude[active_indices],
                old_height[active_indices],
            )
            average_xyz = 0.5 * (old_xyz + dem_xyz)
            average_latitude, average_longitude, average_llh_height = _ecef_to_llh(
                average_xyz
            )
            average_height = (
                torch.linalg.vector_norm(average_xyz, dim=-1) - radius_active
            )
            damping_valid = (
                valid
                & ~newly_solved
                & torch.isfinite(average_latitude)
                & torch.isfinite(average_longitude)
                & torch.isfinite(average_llh_height)
                & torch.isfinite(average_height)
            )
            next_latitude = torch.where(damping_valid, average_latitude, next_latitude)
            next_longitude = torch.where(
                damping_valid, average_longitude, next_longitude
            )
            next_llh_height = torch.where(
                damping_valid, average_llh_height, next_llh_height
            )
            next_height = torch.where(damping_valid, average_height, next_height)
            aitken_enabled &= ~damping_valid
        if dem_samples is not None:
            restart_active = aitken_restart[active_indices]
            restart_after_damping = damping_valid & ~restart_active
        selected_height = torch.where(aitken_enabled, aitken_height, next_height)
        state_height = torch.where(
            newly_solved,
            converged_height,
            torch.where(valid, selected_height, height_active),
        )
        old_latitude.index_copy_(
            0,
            active_indices,
            torch.where(valid, next_latitude, old_latitude[active_indices]),
        )
        old_longitude.index_copy_(
            0,
            active_indices,
            torch.where(valid, next_longitude, old_longitude[active_indices]),
        )
        old_height.index_copy_(
            0,
            active_indices,
            torch.where(valid, next_llh_height, old_height[active_indices]),
        )
        height.index_copy_(
            0,
            active_indices,
            torch.where(valid, state_height, height_active),
        )
        if dem_samples is not None:
            previous_height.index_copy_(
                0,
                active_indices,
                torch.where(valid, height_active, previous_height[active_indices]),
            )
            previous_fixed_height.index_copy_(
                0,
                active_indices,
                torch.where(valid, new_height, previous_fixed_height[active_indices]),
            )
            restart_indices = active_indices[restart_after_damping]
            previous_height.index_fill_(0, restart_indices, torch.nan)
            previous_fixed_height.index_fill_(0, restart_indices, torch.nan)
            aitken_restart.index_copy_(0, active_indices, restart_after_damping)

        if attempt == total_iterations:
            break
        continue_mask = valid & ~newly_solved
        continue_mask &= ellipsoid_height_active - state_height < target_range_active
        failed.index_fill_(
            0,
            active_indices[valid & ~newly_solved & ~continue_mask],
            True,
        )
        active_indices = active_indices[continue_mask]

    semi_minor = radius + height
    cos_theta = 0.5 * (
        satellite_norm / target_range
        + target_range / satellite_norm
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
    final_latitude, final_longitude, final_height = _ecef_to_llh(final_xyz)
    final_look = final_xyz - sat
    final_distance = torch.linalg.vector_norm(final_look, dim=-1)
    final_unit = final_look / torch.clamp(final_distance, min=1.0e-12).unsqueeze(-1)
    final_range_residual = final_distance - target_range
    final_doppler = 2.0 * torch.sum(velocity * final_unit, dim=-1) / wavelength_m
    final_valid = (
        finite
        & ~failed
        & torch.isfinite(final_latitude)
        & torch.isfinite(final_longitude)
        & torch.isfinite(final_height)
        & torch.isfinite(final_range_residual)
        & (final_distance > 0.0)
    )
    if dem_samples is None:
        final_dem_valid = torch.ones_like(final_valid)
        final_dem_converged = torch.ones_like(final_valid)
    else:
        final_dem_height = _sample_dem_six(
            dem_samples,
            final_latitude,
            final_longitude,
            dem_latitude_start_deg,
            dem_longitude_start_deg,
            dem_latitude_spacing_deg,
            dem_longitude_spacing_deg,
        )
        final_dem_valid = torch.isfinite(final_dem_height)
        final_dem_converged = final_dem_valid & (
            torch.abs(final_height - final_dem_height) <= dem_height_tol_m
        )
        final_valid &= final_dem_valid
    # The final TCN pass is the authoritative published coordinate triple.
    # Promote lanes whose range and DEM consistency residuals meet tolerance,
    # including lanes that reached them on the budget boundary.
    solved = (
        final_valid
        & (torch.abs(final_range_residual) < range_tol_m)
        & final_dem_converged
    )
    # Keep finite final TCN coordinates visible when only the final DEM
    # recheck failed.  ``solved`` remains gated by ``final_valid`` above, so
    # validity and status are not confused with coordinate publication.
    result_finite = (final_valid | (~final_dem_valid & finite & ~failed)) & ~dem_invalid
    range_residual = torch.where(result_finite, final_range_residual, range_residual)
    doppler_residual = torch.where(result_finite, final_doppler, doppler_residual)

    def restore(value: Any) -> Any:
        """Restore public lane dimensions after compact eager execution."""
        return value.reshape(original_shape)

    output = {
        "latitude_deg": restore(final_latitude),
        "longitude_deg": restore(final_longitude),
        "height_m": restore(final_height if dem_samples is not None else height),
        "range_index": restore(range_index),
        "azimuth_index": restore(azimuth),
        "converged": restore(solved),
        "iterations": restore(
            torch.where(
                solved | (finite & ~failed),
                attempts,
                torch.full_like(attempts, -1),
            )
        ),
        "residual_range_m": restore(range_residual),
        "residual_doppler_hz": restore(doppler_residual),
    }
    published_ecef = final_xyz
    output.update(
        {
            "ecef_x_m": restore(published_ecef[..., 0]),
            "ecef_y_m": restore(published_ecef[..., 1]),
            "ecef_z_m": restore(published_ecef[..., 2]),
        }
    )
    return _result_invalid(output, restore(result_finite))


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
    extra_iter: int = 0,
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
    """Solve rdr2geo with one TCN/DEM invocation.

    Eager raster execution samples the DEM inside the active-lane TCN loop,
    while compiled execution retains a single fixed-shape loop so
    TorchInductor can capture it without a dynamic Python control-flow graph.
    Raster convergence requires both the range residual and the DEM height
    fixed-point residual to satisfy their respective tolerances.
    """
    import torch

    geometry_state = _rdr2geo_geometry_state(
        azimuth,
        range_index,
        orbit_times,
        orbit_positions,
        orbit_velocities,
        sensing_offset_s=sensing_offset_s,
        azimuth_interval_s=azimuth_interval_s,
        starting_range_m=starting_range_m,
        range_spacing_m=range_spacing_m,
    )
    # Constant-height and compiled raster paths use the single prepared loop.
    initial_height = (
        torch.full_like(height_seed, dem_height_m)
        if dem_samples is None and dem_height_m is not None
        else height_seed
    )
    # Both eager raster and compiled paths use one TCN invocation.  Eager
    # raster calls select the active-lane implementation inside _rdr2geo_once;
    # compiled calls retain its fixed-shape implementation.
    return _rdr2geo_once(
        azimuth,
        range_index,
        initial_height,
        orbit_times,
        orbit_positions,
        orbit_velocities,
        sensing_offset_s=sensing_offset_s,
        azimuth_interval_s=azimuth_interval_s,
        starting_range_m=starting_range_m,
        range_spacing_m=range_spacing_m,
        wavelength_m=wavelength_m,
        look_sign=look_sign,
        max_iter=max_iter,
        extra_iter=extra_iter,
        range_tol_m=range_tol_m,
        doppler_tol_hz=doppler_tol_hz,
        dynamic_iterations=dynamic_iterations,
        dem_samples=dem_samples,
        dem_latitude_start_deg=dem_latitude_start_deg,
        dem_longitude_start_deg=dem_longitude_start_deg,
        dem_latitude_spacing_deg=dem_latitude_spacing_deg,
        dem_longitude_spacing_deg=dem_longitude_spacing_deg,
        dem_iterations=dem_iterations,
        dem_height_tol_m=dem_height_tol_m,
        dem_height_m=dem_height_m,
        geometry_state=geometry_state,
    )


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


__all__ = [
    "geo2rdr_kernel",
    "prepared_orbit_tensors",
    "rdr2geo_kernel",
]
