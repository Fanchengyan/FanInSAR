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

double natural_spline_six(const double* values, double fraction) {
  double second[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double recurrence[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int index = 1; index < 5; ++index) {
    const double denominator = recurrence[index - 1] / 2.0 + 2.0;
    recurrence[index] = -0.5 / denominator;
    second[index] =
        (3.0 * (values[index + 1] - 2.0 * values[index] + values[index - 1]) -
         second[index - 1] / 2.0) /
        denominator;
  }
  for (int index = 4; index > 0; --index) {
    second[index] = recurrence[index] * second[index + 1] + second[index];
  }
  return values[1] + fraction *
      (values[2] - values[1] - second[1] / 3.0 - second[2] / 6.0 + fraction *
       (second[1] / 2.0 + fraction * (second[2] - second[1]) / 6.0));
}

double sample_dem_six(const Tensor& dem_samples, double latitude,
                      double longitude, double latitude_start,
                      double longitude_start, double latitude_spacing,
                      double longitude_spacing) {
  if (!std::isfinite(latitude) || !std::isfinite(longitude) ||
      !(latitude_spacing > 0.0) || !(longitude_spacing > 0.0)) {
    return kNan;
  }
  const double row = (latitude - latitude_start) / latitude_spacing;
  const double column = (longitude - longitude_start) / longitude_spacing;
  const int row_base = static_cast<int>(std::floor(row));
  const int column_base = static_cast<int>(std::floor(column));
  if (row_base < 1 || row_base > 3 || column_base < 1 || column_base > 3) {
    return kNan;
  }
  const auto* samples = dem_samples.data_ptr<double>();
  double along_rows[6]{};
  double window[6]{};
  const double column_fraction = column - column_base;
  const double row_fraction = row - row_base;
  for (int row_offset = -1; row_offset <= 4; ++row_offset) {
    for (int column_offset = -1; column_offset <= 4; ++column_offset) {
      window[column_offset + 1] =
          samples[(row_base + row_offset) * 6 + column_base + column_offset];
    }
    along_rows[row_offset + 1] = natural_spline_six(window, column_fraction);
  }
  return natural_spline_six(along_rows, row_fraction);
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
  auto* heights_out = heights.data_ptr<double>();
  auto* ranges_out = ranges.data_ptr<double>();
  auto* azimuths_out = azimuths.data_ptr<double>();
  auto* solved = converged.data_ptr<bool>();
  auto* iteration_values = iterations.data_ptr<int32_t>();
  auto* decisions = decision_residual.data_ptr<double>();
  auto* finals = final_residual.data_ptr<double>();
  auto* tolerances = tolerance.data_ptr<double>();
  auto* exhausted_values = exhausted.data_ptr<bool>();
  auto* range_residuals = residual_range.data_ptr<double>();
  auto* doppler_residuals = residual_doppler.data_ptr<double>();
  const int64_t budget = max_iter + extra_iter;
  const double orbit_start = times[0];
  const double orbit_end = times[orbit_times_s.numel() - 1];
  begin_telemetry(count, "rdr2geo_cpu");
  auto invalidate_lane = [&](int64_t point) {
    latitudes[point] = kNan;
    longitudes[point] = kNan;
    heights_out[point] = kNan;
    ranges_out[point] = kNan;
    azimuths_out[point] = kNan;
    tolerances[point] = kNan;
    solved[point] = false;
    iteration_values[point] = -1;
    decisions[point] = kNan;
    finals[point] = kNan;
    exhausted_values[point] = false;
    range_residuals[point] = kNan;
    doppler_residuals[point] = kNan;
  };

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t point = 0; point < count; ++point) {
    record_visit(point);
    const double azimuth = azimuths_in[point];
    const double range = ranges_in[point];
    const double height = heights_in[point];
    if (!std::isfinite(azimuth) || !std::isfinite(range) || !std::isfinite(height)) {
      invalidate_lane(point);
      continue;
    }
    double time_s = sensing_offset_s + azimuth * azimuth_time_interval_s;
    if (time_s < orbit_start || time_s > orbit_end) {
      invalidate_lane(point);
      continue;
    }
    const OrbitState initial = interpolate_orbit(times, positions, velocities,
                                                 orbit_times_s.numel(), time_s);
    const double target_range = starting_slant_range_m + range * range_spacing_m;
    const double velocity_norm = norm(initial.velocity);
    const double satellite_norm = norm(initial.position);
    if (!(velocity_norm > 0.0) || !(satellite_norm > 0.0) ||
        !(target_range > 0.0)) {
      invalidate_lane(point);
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
      invalidate_lane(point);
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
      const double decision_metric = std::max(
          std::abs(last_range_residual) / range_tol_m,
          std::abs(last_doppler) / doppler_tol_hz);
      if (decision_metric < 1.0) {
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
    finals[point] = last_range_residual;
    exhausted_values[point] = false;
  }
  return {latitude, longitude, heights, ranges, azimuths, converged, iterations,
          decision_residual, final_residual, tolerance, exhausted,
          boundary_rechecked, residual_range, residual_doppler};
}

std::vector<Tensor> rdr2geo_cpu_dem(
    const Tensor& azimuth_index, const Tensor& range_index,
    const Tensor& height_seed_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m, double wavelength_m,
    int64_t max_iter, int64_t extra_iter, double range_tol_m,
    double doppler_tol_hz, bool right_looking, const Tensor& dem_samples,
    double dem_latitude_start_deg, double dem_longitude_start_deg,
    double dem_latitude_spacing_deg, double dem_longitude_spacing_deg,
    int64_t dem_iterations, double dem_height_tol_m) {
  TORCH_CHECK(!dem_samples.is_cuda() &&
                  dem_samples.scalar_type() == torch::kFloat64 &&
                  dem_samples.is_contiguous() && dem_samples.dim() == 2 &&
                  dem_samples.size(0) == 6 && dem_samples.size(1) == 6,
              "dem_samples must be a contiguous CPU float64 (6, 6) array");
  TORCH_CHECK(torch::isfinite(dem_samples).all().item<bool>(),
              "dem_samples must contain only finite values");
  TORCH_CHECK(dem_iterations > 0 && std::isfinite(dem_height_tol_m) &&
                  dem_height_tol_m > 0.0,
              "DEM fixed-point settings are invalid");
  auto heights = height_seed_m.clone();
  std::vector<Tensor> result;
  for (int64_t iteration = 0; iteration < dem_iterations; ++iteration) {
    result = rdr2geo_cpu(
        azimuth_index, range_index, heights, orbit_times_s, orbit_positions_m,
        orbit_velocities_m_s, sensing_offset_s, azimuth_time_interval_s,
        starting_slant_range_m, range_spacing_m, wavelength_m, max_iter,
        extra_iter, range_tol_m, doppler_tol_hz, right_looking);
    auto next_heights = torch::full_like(heights, kNan);
    const auto* latitudes = result[0].data_ptr<double>();
    const auto* longitudes = result[1].data_ptr<double>();
    const auto* converged = result[5].data_ptr<bool>();
    auto* next = next_heights.data_ptr<double>();
    const int64_t count = heights.numel();
    double maximum_update = 0.0;
    for (int64_t index = 0; index < count; ++index) {
      if (!converged[index]) continue;
      next[index] = sample_dem_six(
          dem_samples, latitudes[index], longitudes[index],
          dem_latitude_start_deg, dem_longitude_start_deg,
          dem_latitude_spacing_deg, dem_longitude_spacing_deg);
      if (std::isfinite(next[index]) && std::isfinite(heights.data_ptr<double>()[index])) {
        maximum_update = std::max(maximum_update, std::abs(next[index] -
                                                            heights.data_ptr<double>()[index]));
      }
    }
    heights = next_heights;
    if (maximum_update < dem_height_tol_m) break;
  }
  result = rdr2geo_cpu(
      azimuth_index, range_index, heights, orbit_times_s, orbit_positions_m,
      orbit_velocities_m_s, sensing_offset_s, azimuth_time_interval_s,
      starting_slant_range_m, range_spacing_m, wavelength_m, max_iter,
      extra_iter, range_tol_m, doppler_tol_hz, right_looking);
  result[2] = heights;
  return result;
}

}  // namespace faninsar_native_v2
