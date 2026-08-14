#include "geometry_cuda.cuh"

namespace faninsar::geometry::cuda_v2 {

namespace {

__global__ void geo2rdr_kernel(
    const double* latitude, const double* longitude, const double* height,
    int64_t count, const double* orbit_times, const double* positions,
    const double* velocities, int64_t orbit_count, double sensing_offset,
    double azimuth_interval, double starting_range, double range_spacing,
    int64_t budget, double time_tolerance, double range_tolerance,
    double doppler_tolerance, double wavelength,
    double* output_latitude, double* output_longitude, double* output_height,
    double* azimuth, double* range, bool* converged, double* range_residual,
    double* doppler_residual, int32_t* iterations, double* decision_residual,
    double* final_residual, double* tolerance, bool* max_iter_exhausted,
    bool* boundary_rechecked, int32_t* visit_counts) {
  const int64_t point = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (point >= count) return;
  visit_counts[point] = 1;
  if (!isfinite(latitude[point]) || !isfinite(longitude[point]) ||
      !isfinite(height[point])) return;
  iterations[point] = 0;
  max_iter_exhausted[point] = true;
  tolerance[point] = 1.0;

  double target[3];
  llh_to_ecef(latitude[point], longitude[point], height[point], target);
  const double orbit_start = orbit_times[0];
  const double orbit_end = orbit_times[orbit_count - 1];
  double satellite[3], velocity[3], acceleration[3];
  hermite(orbit_times, positions, velocities, orbit_count,
          fmin(fmax(sensing_offset, orbit_start), orbit_end), satellite,
          velocity, acceleration);
  double speed_squared = 0.0;
  double along_track_dot = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    speed_squared += velocity[axis] * velocity[axis];
    along_track_dot += (target[axis] - satellite[axis]) * velocity[axis];
  }
  double time = sensing_offset;
  if (isfinite(speed_squared) && speed_squared > 0.0)
    time += along_track_dot / speed_squared;
  time = fmin(fmax(time, orbit_start), orbit_end);

  double residual_doppler = nan("");
  double residual_range = nan("");
  double decision = nan("");
  double final = nan("");
  bool at_boundary = false;
  bool solved = false;
  int32_t attempts = 0;
  for (int64_t iteration = 0; iteration < budget; ++iteration) {
    if (time < orbit_start || time > orbit_end) break;
    hermite(orbit_times, positions, velocities, orbit_count, time, satellite,
            velocity, acceleration);
    double look[3];
    double range_squared = 0.0;
    double velocity_squared = 0.0;
    double doppler = 0.0;
    double acceleration_along_look = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      look[axis] = target[axis] - satellite[axis];
      range_squared += look[axis] * look[axis];
    }
    const double slant_range = sqrt(range_squared);
    if (!(slant_range > 0.0) || !isfinite(slant_range)) break;
    for (int axis = 0; axis < 3; ++axis) {
      const double unit_look = look[axis] / slant_range;
      doppler += velocity[axis] * unit_look;
      velocity_squared += velocity[axis] * velocity[axis];
      acceleration_along_look += acceleration[axis] * unit_look;
    }
    residual_doppler = 2.0 * doppler / wavelength;
    const double derivative = acceleration_along_look +
        (doppler * doppler - velocity_squared) / fmax(slant_range, 1.0);
    ++attempts;
    if (!isfinite(derivative) || fabs(derivative) < 1.0e-12) break;
    const double step = -doppler / derivative;
    time += step;
    if (isfinite(step) && fabs(step) < time_tolerance) {
      if (time >= orbit_start && time <= orbit_end) {
        hermite(orbit_times, positions, velocities, orbit_count, time,
                satellite, velocity, acceleration);
        double final_range_squared = 0.0;
        double final_doppler = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
          const double final_look = target[axis] - satellite[axis];
          final_range_squared += final_look * final_look;
        }
        const double final_range = sqrt(final_range_squared);
        if (final_range > 0.0 && isfinite(final_range)) {
          for (int axis = 0; axis < 3; ++axis)
            final_doppler += velocity[axis] *
                (target[axis] - satellite[axis]) / final_range;
          residual_doppler = 2.0 * final_doppler / wavelength;
          const double rate = final_doppler;
          // Geo2rdr has no independent range equation.  The Newton step
          // converted through the range-rate is the final range-equivalent
          // residual, while the Doppler residual is reported in Hz.
          residual_range = step * rate;
          decision = fmax(fabs(residual_range) / range_tolerance,
                          fabs(residual_doppler) / doppler_tolerance);
          solved = isfinite(final_doppler) && isfinite(residual_range) &&
                   fabs(residual_range) < range_tolerance &&
                   fabs(residual_doppler) < doppler_tolerance;
          final = decision;
          at_boundary = time <= orbit_start || time >= orbit_end;
        }
      }
      if (solved) break;
    }
  }
  if (!solved) {
    iterations[point] = attempts;
    doppler_residual[point] = residual_doppler;
    range_residual[point] = residual_range;
    decision_residual[point] = decision;
    final_residual[point] = decision;
    boundary_rechecked[point] = at_boundary;
    return;
  }
  iterations[point] = attempts;
  max_iter_exhausted[point] = false;
  doppler_residual[point] = residual_doppler;
  range_residual[point] = residual_range;
  decision_residual[point] = decision;
  final_residual[point] = final;
  boundary_rechecked[point] = at_boundary;
  double final_range_squared = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double look = target[axis] - satellite[axis];
    final_range_squared += look * look;
  }
  const double final_range = sqrt(final_range_squared);
  output_latitude[point] = latitude[point];
  output_longitude[point] = longitude[point];
  output_height[point] = height[point];
  azimuth[point] = (time - sensing_offset) / azimuth_interval;
  range[point] = (final_range - starting_range) / range_spacing;
  converged[point] = true;
  range_residual[point] = residual_range;
  doppler_residual[point] = residual_doppler;
}

}  // namespace

std::vector<Tensor> geo2rdr_cuda_v2(
    const Tensor& latitude_deg, const Tensor& longitude_deg,
    const Tensor& height_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m, double wavelength_m,
    int64_t max_iter, int64_t extra_iter, double time_tol_s,
    double range_tolerance_m, double doppler_tolerance_hz) {
  check_cuda_vector(latitude_deg, "latitude_deg");
  check_cuda_vector(longitude_deg, "longitude_deg");
  check_cuda_vector(height_m, "height_m");
  check_cuda_vector(orbit_times_s, "orbit_times_s");
  const int device = latitude_deg.get_device();
  for (const auto& input : {longitude_deg, height_m, orbit_times_s})
    check_same_device(latitude_deg, input, "input tensor");
  TORCH_CHECK(latitude_deg.numel() == longitude_deg.numel() &&
              latitude_deg.numel() == height_m.numel(),
              "point arrays must have equal lengths");
  TORCH_CHECK(orbit_times_s.numel() >= 2,
              "orbit_times_s must contain at least two samples");
  check_cuda_matrix(orbit_positions_m, "orbit_positions_m",
                    orbit_times_s.numel(), device);
  check_cuda_matrix(orbit_velocities_m_s, "orbit_velocities_m_s",
                    orbit_times_s.numel(), device);
  TORCH_CHECK(std::isfinite(azimuth_time_interval_s) &&
              std::isfinite(range_spacing_m) && azimuth_time_interval_s > 0.0 &&
              range_spacing_m > 0.0,
              "radar intervals and spacing must be positive");
  TORCH_CHECK(std::isfinite(sensing_offset_s) &&
              std::isfinite(starting_slant_range_m) &&
              std::isfinite(wavelength_m) && wavelength_m > 0.0,
              "radar origins must be finite");
  check_solver_scalars(max_iter, extra_iter, time_tol_s, "time_tol_s");
  TORCH_CHECK(std::isfinite(range_tolerance_m) && range_tolerance_m > 0.0,
              "range_tolerance_m must be finite and positive");
  TORCH_CHECK(std::isfinite(doppler_tolerance_hz) &&
              doppler_tolerance_hz > 0.0,
              "doppler_tol_hz must be finite and positive");
  const auto deltas = orbit_times_s.slice(0, 1) - orbit_times_s.slice(0, 0, -1);
  TORCH_CHECK(torch::isfinite(orbit_times_s).all().item<bool>() &&
              deltas.gt(0).all().item<bool>(),
              "orbit_times_s must be finite and strictly increasing");

  c10::cuda::CUDAGuard guard(latitude_deg.device());
  const auto options = latitude_deg.options();
  const int64_t count = latitude_deg.numel();
  auto latitude = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto longitude = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto output_height = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto azimuth = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto range = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto converged = torch::zeros({count}, options.dtype(torch::kBool));
  auto range_residual = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto doppler_residual = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto iterations = torch::full({count}, -1, options.dtype(torch::kInt32));
  auto decision_residual = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto final_residual = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto tolerance = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto max_iter_exhausted = torch::zeros({count}, options.dtype(torch::kBool));
  auto boundary_rechecked = torch::zeros({count}, options.dtype(torch::kBool));
  auto visit_counts = torch::zeros({count}, options.dtype(torch::kInt32));
  if (count > 0) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream(device);
    geo2rdr_kernel<<<blocks, threads, 0, stream.stream()>>>(
        latitude_deg.data_ptr<double>(), longitude_deg.data_ptr<double>(),
        height_m.data_ptr<double>(), count, orbit_times_s.data_ptr<double>(),
        orbit_positions_m.data_ptr<double>(), orbit_velocities_m_s.data_ptr<double>(),
        orbit_times_s.numel(), sensing_offset_s, azimuth_time_interval_s,
        starting_slant_range_m, range_spacing_m, max_iter + extra_iter,
        time_tol_s, range_tolerance_m, doppler_tolerance_hz, wavelength_m,
        latitude.data_ptr<double>(), longitude.data_ptr<double>(),
        output_height.data_ptr<double>(), azimuth.data_ptr<double>(),
        range.data_ptr<double>(), converged.data_ptr<bool>(),
        range_residual.data_ptr<double>(), doppler_residual.data_ptr<double>(),
        iterations.data_ptr<int32_t>(), decision_residual.data_ptr<double>(),
        final_residual.data_ptr<double>(), tolerance.data_ptr<double>(),
        max_iter_exhausted.data_ptr<bool>(),
        boundary_rechecked.data_ptr<bool>(), visit_counts.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {latitude, longitude, output_height, range, azimuth, converged,
          iterations, decision_residual, final_residual, tolerance,
          max_iter_exhausted, boundary_rechecked, range_residual,
          doppler_residual, visit_counts};
}

}  // namespace faninsar::geometry::cuda_v2
