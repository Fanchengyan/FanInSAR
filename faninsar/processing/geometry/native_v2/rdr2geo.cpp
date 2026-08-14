#include "native_v2_abi.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace faninsar_native_v2 {
namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

void validate_inputs(const Tensor& azimuth_index, const Tensor& range_index,
                     const Tensor& height_m, const Tensor& orbit_times_s,
                     const Tensor& orbit_positions_m,
                     const Tensor& orbit_velocities_m_s,
                     double azimuth_time_interval_s, double range_spacing_m,
                     double wavelength_m, int64_t max_iter, int64_t extra_iter,
                     double range_tol_m, double doppler_tol_hz) {
  check_vector(azimuth_index, "azimuth_index");
  check_vector(range_index, "range_index");
  check_vector(height_m, "height_m");
  check_vector(orbit_times_s, "orbit_times_s");
  TORCH_CHECK(azimuth_index.numel() == range_index.numel() &&
                  azimuth_index.numel() == height_m.numel(),
              "rdr2geo input arrays must have equal lengths");
  check_orbit(orbit_positions_m, "orbit_positions_m", orbit_times_s.numel());
  check_orbit(orbit_velocities_m_s, "orbit_velocities_m_s", orbit_times_s.numel());
  check_orbit_times(orbit_times_s);
  TORCH_CHECK(std::isfinite(azimuth_time_interval_s) && azimuth_time_interval_s > 0.0,
              "azimuth_time_interval_s must be finite and positive");
  TORCH_CHECK(std::isfinite(range_spacing_m) && range_spacing_m > 0.0,
              "range_spacing_m must be finite and positive");
  TORCH_CHECK(std::isfinite(wavelength_m) && wavelength_m > 0.0,
              "wavelength_m must be finite and positive");
  TORCH_CHECK(max_iter > 0 && extra_iter >= 0, "iteration settings are invalid");
  TORCH_CHECK(max_iter + extra_iter <= std::numeric_limits<int32_t>::max(),
              "iteration budget exceeds int32 capacity");
  TORCH_CHECK(std::isfinite(range_tol_m) && range_tol_m > 0.0 &&
                  std::isfinite(doppler_tol_hz) && doppler_tol_hz > 0.0,
              "solver tolerances must be finite and positive");
}

Vec3 cross(const Vec3& left, const Vec3& right) {
  return {left[1] * right[2] - left[2] * right[1],
          left[2] * right[0] - left[0] * right[2],
          left[0] * right[1] - left[1] * right[0]};
}

}  // namespace

std::vector<Tensor> rdr2geo_cpu(
    const Tensor& azimuth_index, const Tensor& range_index, const Tensor& height_m,
    const Tensor& orbit_times_s, const Tensor& orbit_positions_m,
    const Tensor& orbit_velocities_m_s, double sensing_offset_s,
    double azimuth_time_interval_s, double starting_slant_range_m,
    double range_spacing_m, double wavelength_m, int64_t max_iter,
    int64_t extra_iter, double range_tol_m, double doppler_tol_hz,
    bool right_looking) {
  validate_inputs(azimuth_index, range_index, height_m, orbit_times_s,
                  orbit_positions_m, orbit_velocities_m_s,
                  azimuth_time_interval_s, range_spacing_m, wavelength_m,
                  max_iter, extra_iter, range_tol_m, doppler_tol_hz);
  TORCH_CHECK(std::isfinite(sensing_offset_s) && std::isfinite(starting_slant_range_m),
              "radar timing and range origins must be finite");
  const int64_t count = azimuth_index.numel();
  auto options = azimuth_index.options();
  auto latitude = torch::full({count}, kNan, options);
  auto longitude = torch::full({count}, kNan, options);
  auto heights = height_m.clone();
  auto ranges = range_index.clone();
  auto azimuths = azimuth_index.clone();
  auto converged = torch::zeros({count}, options.dtype(torch::kBool));
  auto iterations = torch::full({count}, -1, options.dtype(torch::kInt32));
  auto decision_residual = torch::full({count}, kNan, options);
  auto final_residual = torch::full({count}, kNan, options);
  auto tolerance = torch::full({count}, range_tol_m, options);
  auto exhausted = torch::zeros({count}, options.dtype(torch::kBool));
  auto boundary_rechecked = torch::zeros({count}, options.dtype(torch::kBool));
  auto residual_range = torch::full({count}, kNan, options);
  auto residual_doppler = torch::full({count}, kNan, options);

  const auto* azimuths_in = azimuth_index.data_ptr<double>();
  const auto* ranges_in = range_index.data_ptr<double>();
  const auto* heights_in = height_m.data_ptr<double>();
  const auto* times = orbit_times_s.data_ptr<double>();
  const auto* positions = orbit_positions_m.data_ptr<double>();
  const auto* velocities = orbit_velocities_m_s.data_ptr<double>();
  auto* latitudes = latitude.data_ptr<double>();
  auto* longitudes = longitude.data_ptr<double>();
  auto* solved = converged.data_ptr<bool>();
  auto* iteration_values = iterations.data_ptr<int32_t>();
  auto* decisions = decision_residual.data_ptr<double>();
  auto* finals = final_residual.data_ptr<double>();
  auto* exhausted_values = exhausted.data_ptr<bool>();
  auto* range_residuals = residual_range.data_ptr<double>();
  auto* doppler_residuals = residual_doppler.data_ptr<double>();
  const int64_t budget = max_iter + extra_iter;
  const double orbit_start = times[0];
  const double orbit_end = times[orbit_times_s.numel() - 1];
  begin_telemetry(count);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t point = 0; point < count; ++point) {
    record_visit(point);
    const double azimuth = azimuths_in[point];
    const double range = ranges_in[point];
    const double height = heights_in[point];
    if (!std::isfinite(azimuth) || !std::isfinite(range) || !std::isfinite(height)) {
      exhausted_values[point] = true;
      continue;
    }
    double time_s = sensing_offset_s + azimuth * azimuth_time_interval_s;
    if (time_s < orbit_start || time_s > orbit_end) {
      exhausted_values[point] = true;
      continue;
    }
    const OrbitState initial = interpolate_orbit(times, positions, velocities,
                                                 orbit_times_s.numel(), time_s);
    const double target_range = starting_slant_range_m + range * range_spacing_m;
    const double velocity_norm = norm(initial.velocity);
    const double satellite_norm = norm(initial.position);
    if (!(velocity_norm > 0.0) || !(satellite_norm > 0.0) ||
        !(target_range > 0.0)) {
      exhausted_values[point] = true;
      continue;
    }
    const Vec3 velocity_unit{initial.velocity[0] / velocity_norm,
                             initial.velocity[1] / velocity_norm,
                             initial.velocity[2] / velocity_norm};
    const Vec3 radial{initial.position[0] / satellite_norm,
                      initial.position[1] / satellite_norm,
                      initial.position[2] / satellite_norm};
    Vec3 look = right_looking ? cross(velocity_unit, radial)
                              : cross(radial, velocity_unit);
    const double look_norm = norm(look);
    if (!(look_norm > 0.0)) {
      exhausted_values[point] = true;
      continue;
    }
    look = {look[0] / look_norm, look[1] / look_norm, look[2] / look_norm};
    const Vec3 approximate{initial.position[0] + look[0] * target_range,
                           initial.position[1] + look[1] * target_range,
                           initial.position[2] + look[2] * target_range};
    const Vec3 initial_llh = ecef_to_llh(approximate);
    double latitude = initial_llh[0];
    double longitude = initial_llh[1];
    double last_range_residual = kNan;
    double last_doppler = kNan;
    bool lane_solved = false;
    int64_t used_iterations = 0;
    for (int64_t iteration = 0; iteration < budget; ++iteration) {
      ++used_iterations;
      const OrbitState state = interpolate_orbit(times, positions, velocities,
                                                 orbit_times_s.numel(), time_s);
      const Vec3 target = llh_to_ecef(latitude, longitude, height);
      const Vec3 look_vector{target[0] - state.position[0],
                             target[1] - state.position[1],
                             target[2] - state.position[2]};
      const double slant_range = norm(look_vector);
      if (!(slant_range > 0.0) || !std::isfinite(slant_range)) break;
      const Vec3 unit{look_vector[0] / slant_range,
                      look_vector[1] / slant_range,
                      look_vector[2] / slant_range};
      last_range_residual = slant_range - target_range;
      last_doppler = 2.0 * dot(state.velocity, unit) / wavelength_m;
      decisions[point] = last_range_residual;
      range_residuals[point] = last_range_residual;
      doppler_residuals[point] = last_doppler;
      if (std::abs(last_range_residual) <= range_tol_m &&
          std::abs(last_doppler) <= doppler_tol_hz) {
        lane_solved = true;
        break;
      }
      constexpr double delta = 1.0e-5;
      const Vec3 lat_target = llh_to_ecef(latitude + delta, longitude, height);
      const Vec3 lon_target = llh_to_ecef(latitude, longitude + delta, height);
      const Vec3 lat_look{lat_target[0] - state.position[0],
                          lat_target[1] - state.position[1],
                          lat_target[2] - state.position[2]};
      const Vec3 lon_look{lon_target[0] - state.position[0],
                          lon_target[1] - state.position[1],
                          lon_target[2] - state.position[2]};
      const double lat_range = norm(lat_look);
      const double lon_range = norm(lon_look);
      if (!(lat_range > 0.0) || !(lon_range > 0.0)) break;
      const Vec3 lat_unit{lat_look[0] / lat_range, lat_look[1] / lat_range,
                          lat_look[2] / lat_range};
      const Vec3 lon_unit{lon_look[0] / lon_range, lon_look[1] / lon_range,
                          lon_look[2] / lon_range};
      const double range_lat = (lat_range - slant_range) / delta;
      const double range_lon = (lon_range - slant_range) / delta;
      const double doppler_lat =
          (2.0 * dot(state.velocity, lat_unit) / wavelength_m - last_doppler) /
          delta;
      const double doppler_lon =
          (2.0 * dot(state.velocity, lon_unit) / wavelength_m - last_doppler) /
          delta;
      const double determinant = range_lat * doppler_lon - range_lon * doppler_lat;
      if (!(std::abs(determinant) > 1.0e-12) || !std::isfinite(determinant)) break;
      const double step_lat =
          (-last_range_residual * doppler_lon + range_lon * last_doppler) /
          determinant;
      const double step_lon =
          (-range_lat * last_doppler + last_range_residual * doppler_lat) /
          determinant;
      latitude += step_lat;
      longitude += step_lon;
      if (!std::isfinite(latitude) || !std::isfinite(longitude)) break;
    }
    if (!lane_solved) {
      exhausted_values[point] = true;
      continue;
    }
    latitudes[point] = latitude;
    longitudes[point] = longitude;
    solved[point] = true;
    iteration_values[point] = static_cast<int32_t>(used_iterations);
    finals[point] = std::max(std::abs(last_range_residual), std::abs(last_doppler));
    exhausted_values[point] = false;
  }
  return {latitude, longitude, heights, ranges, azimuths, converged, iterations,
          decision_residual, final_residual, tolerance, exhausted,
          boundary_rechecked, residual_range, residual_doppler};
}

}  // namespace faninsar_native_v2

