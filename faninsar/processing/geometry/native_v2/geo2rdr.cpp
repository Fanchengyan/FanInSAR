#include "native_v2_abi.h"

#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace faninsar_native_v2 {
namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

void validate_inputs(const Tensor& latitude_deg, const Tensor& longitude_deg,
                     const Tensor& height_m, const Tensor& orbit_times_s,
                     const Tensor& orbit_positions_m,
                     const Tensor& orbit_velocities_m_s,
                     double azimuth_time_interval_s, double range_spacing_m,
                     double wavelength_m, int64_t max_iter, int64_t extra_iter,
                     double time_tol_s, double range_tol_m,
                     double doppler_tol_hz) {
  check_vector(latitude_deg, "latitude_deg");
  check_vector(longitude_deg, "longitude_deg");
  check_vector(height_m, "height_m");
  check_vector(orbit_times_s, "orbit_times_s");
  TORCH_CHECK(latitude_deg.numel() == longitude_deg.numel() &&
                  latitude_deg.numel() == height_m.numel(),
              "geo2rdr input arrays must have equal lengths");
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
  TORCH_CHECK(std::isfinite(time_tol_s) && time_tol_s > 0.0 &&
                  std::isfinite(range_tol_m) && range_tol_m > 0.0 &&
                  std::isfinite(doppler_tol_hz) && doppler_tol_hz > 0.0,
              "solver tolerances must be finite and positive");
}

}  // namespace

std::vector<Tensor> geo2rdr_cpu(
    const Tensor& latitude_deg, const Tensor& longitude_deg, const Tensor& height_m,
    const Tensor& orbit_times_s, const Tensor& orbit_positions_m,
    const Tensor& orbit_velocities_m_s, double sensing_offset_s,
    double azimuth_time_interval_s, double starting_slant_range_m,
    double range_spacing_m, double wavelength_m, int64_t max_iter,
    int64_t extra_iter, double time_tol_s, double range_tol_m,
    double doppler_tol_hz, bool right_looking) {
  validate_inputs(latitude_deg, longitude_deg, height_m, orbit_times_s,
                  orbit_positions_m, orbit_velocities_m_s,
                  azimuth_time_interval_s, range_spacing_m, wavelength_m,
                  max_iter, extra_iter, time_tol_s, range_tol_m,
                  doppler_tol_hz);
  TORCH_CHECK(std::isfinite(sensing_offset_s) && std::isfinite(starting_slant_range_m),
              "radar timing and range origins must be finite");
  const int64_t count = latitude_deg.numel();
  auto options = latitude_deg.options();
  auto latitude = latitude_deg.clone();
  auto longitude = longitude_deg.clone();
  auto height = height_m.clone();
  auto range_index = torch::full({count}, kNan, options);
  auto azimuth_index = torch::full({count}, kNan, options);
  auto converged = torch::zeros({count}, options.dtype(torch::kBool));
  auto iterations = torch::full({count}, -1, options.dtype(torch::kInt32));
  auto decision_residual = torch::full({count}, kNan, options);
  auto final_residual = torch::full({count}, kNan, options);
  auto tolerance = torch::full({count}, 1.0, options);
  auto max_iter_exhausted = torch::zeros({count}, options.dtype(torch::kBool));
  auto boundary_rechecked = torch::zeros({count}, options.dtype(torch::kBool));
  auto residual_range = torch::full({count}, kNan, options);
  auto residual_doppler = torch::full({count}, kNan, options);

  const auto* latitudes = latitude_deg.data_ptr<double>();
  const auto* longitudes = longitude_deg.data_ptr<double>();
  const auto* heights = height_m.data_ptr<double>();
  const auto* times = orbit_times_s.data_ptr<double>();
  const auto* positions = orbit_positions_m.data_ptr<double>();
  const auto* velocities = orbit_velocities_m_s.data_ptr<double>();
  auto* ranges = range_index.data_ptr<double>();
  auto* azimuths = azimuth_index.data_ptr<double>();
  auto* solved = converged.data_ptr<bool>();
  auto* iteration_values = iterations.data_ptr<int32_t>();
  auto* decision = decision_residual.data_ptr<double>();
  auto* final = final_residual.data_ptr<double>();
  auto* exhausted = max_iter_exhausted.data_ptr<bool>();
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
    if (!std::isfinite(latitudes[point]) || !std::isfinite(longitudes[point]) ||
        !std::isfinite(heights[point])) {
      exhausted[point] = true;
      continue;
    }
    const Vec3 target = llh_to_ecef(latitudes[point], longitudes[point], heights[point]);
    const OrbitState seed = interpolate_orbit(times, positions, velocities,
                                               orbit_times_s.numel(),
                                               std::clamp(sensing_offset_s,
                                                          orbit_start, orbit_end));
    const Vec3 target_delta{target[0] - seed.position[0], target[1] - seed.position[1],
                            target[2] - seed.position[2]};
    const double speed_squared = dot(seed.velocity, seed.velocity);
    double time_s = sensing_offset_s;
    if (std::isfinite(speed_squared) && speed_squared > 0.0) {
      time_s += dot(target_delta, seed.velocity) / speed_squared;
    }
    time_s = std::clamp(time_s, orbit_start, orbit_end);
    double last_doppler = kNan;
    double last_range_residual = kNan;
    bool lane_solved = false;
    int64_t used_iterations = 0;
    for (int64_t iteration = 0; iteration < budget; ++iteration) {
      ++used_iterations;
      if (time_s < orbit_start || time_s > orbit_end) break;
      const OrbitState state = interpolate_orbit(times, positions, velocities,
                                                 orbit_times_s.numel(), time_s);
      const Vec3 look{target[0] - state.position[0], target[1] - state.position[1],
                      target[2] - state.position[2]};
      const double range_m = norm(look);
      if (!(range_m > 0.0) || !std::isfinite(range_m)) break;
      const Vec3 unit{look[0] / range_m, look[1] / range_m, look[2] / range_m};
      const double radial_velocity = dot(state.velocity, unit);
      // ``geo2rdr``'s eager reference publishes radial velocity here even
      // though the field retains the historical ``*_hz`` name.
      const double doppler_hz = radial_velocity;
      const double velocity_squared = dot(state.velocity, state.velocity);
      const double acceleration_along_look = dot(state.acceleration, unit);
      const double derivative = acceleration_along_look +
                                (radial_velocity * radial_velocity - velocity_squared) /
                                    std::max(range_m, 1.0);
      last_doppler = doppler_hz;
      last_range_residual = 0.0;
      decision[point] = std::abs(doppler_hz);
      if (!std::isfinite(derivative) || std::abs(derivative) < 1.0e-12) break;
      const double step = -radial_velocity / derivative;
      time_s += step;
      if (std::abs(step) <= time_tol_s) {
        lane_solved = time_s >= orbit_start && time_s <= orbit_end;
        break;
      }
    }
    residual_doppler[point] = last_doppler;
    residual_range[point] = last_range_residual;
    iteration_values[point] = static_cast<int32_t>(lane_solved ? used_iterations : -1);
    exhausted[point] = !lane_solved;
    if (!lane_solved) continue;
    const OrbitState state = interpolate_orbit(times, positions, velocities,
                                               orbit_times_s.numel(), time_s);
    const Vec3 look{target[0] - state.position[0], target[1] - state.position[1],
                    target[2] - state.position[2]};
    const double range_m = norm(look);
    const Vec3 unit{look[0] / range_m, look[1] / range_m, look[2] / range_m};
    const double final_doppler = dot(state.velocity, unit);
    ranges[point] = (range_m - starting_slant_range_m) / range_spacing_m;
    azimuths[point] = (time_s - sensing_offset_s) / azimuth_time_interval_s;
    solved[point] = true;
    residual_range[point] = 0.0;
    residual_doppler[point] = final_doppler;
    final[point] = std::abs(final_doppler);
  }
  return {latitude, longitude, height, range_index, azimuth_index, converged,
          iterations, decision_residual, final_residual, tolerance,
          max_iter_exhausted, boundary_rechecked, residual_range,
          residual_doppler};
}

}  // namespace faninsar_native_v2
