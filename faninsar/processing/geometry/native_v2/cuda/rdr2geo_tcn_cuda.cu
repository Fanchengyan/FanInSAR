#include "geometry_cuda.cuh"

namespace faninsar::geometry::cuda_v2 {

namespace {

__device__ inline double safe_denominator(double value) {
  if (fabs(value) >= 1.0e-12) return value;
  return copysign(1.0e-12, value == 0.0 ? 1.0 : value);
}

__global__ void rdr2geo_tcn_kernel(
    const double* azimuth_index, const double* range_index,
    const double* height_seed, int64_t count, const double* orbit_times,
    const double* orbit_positions, const double* orbit_velocities,
    int64_t orbit_count, double sensing_offset, double azimuth_interval,
    double starting_range, double range_spacing, const double* dem,
    int64_t dem_rows, int64_t dem_columns, double dem_x_start_deg,
    double dem_y_start_deg, double dem_dx_deg, double dem_dy_deg,
    double reference_height, double min_height, double max_height,
    double wavelength, double range_tolerance, double doppler_tolerance,
    int64_t primary_iter, int64_t budget, double look_sign,
    double* latitude_deg, double* longitude_deg, double* output_height,
    double* output_range, double* output_azimuth,
    bool* converged, int32_t* iterations, double* decision_residual,
    double* final_residual, double* tolerance, bool* max_iter_exhausted,
    bool* boundary_rechecked, double* range_residual,
    double* doppler_residual, int32_t* visit_counts) {
  const int64_t point = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (point >= count) return;
  visit_counts[point] = 1;

  const double azimuth = azimuth_index[point];
  const double range = range_index[point];
  const double seed = height_seed[point];
  const double time = sensing_offset + azimuth * azimuth_interval;
  const double orbit_start = orbit_times[0];
  const double orbit_end = orbit_times[orbit_count - 1];
  if (!isfinite(azimuth) || !isfinite(range) || !isfinite(seed) ||
      !isfinite(time) || time < orbit_start || time > orbit_end) return;

  const double target = starting_range + range * range_spacing;
  if (!(target > 0.0) || !isfinite(target)) return;

  double sat[3], vel[3], acceleration[3];
  hermite(orbit_times, orbit_positions, orbit_velocities, orbit_count, time,
          sat, vel, acceleration);
  const double satellite_norm =
      sqrt(sat[0] * sat[0] + sat[1] * sat[1] + sat[2] * sat[2]);
  const double speed = sqrt(vel[0] * vel[0] + vel[1] * vel[1] +
                            vel[2] * vel[2]);
  if (!(satellite_norm > 0.0) || !(speed > 0.0) ||
      !isfinite(satellite_norm) || !isfinite(speed)) return;
  double velocity_unit[3] = {vel[0] / speed, vel[1] / speed, vel[2] / speed};
  double normal[3] = {-sat[0] / satellite_norm, -sat[1] / satellite_norm,
                      -sat[2] / satellite_norm};
  double cross_track[3] = {
      normal[1] * vel[2] - normal[2] * vel[1],
      normal[2] * vel[0] - normal[0] * vel[2],
      normal[0] * vel[1] - normal[1] * vel[0],
  };
  const double cross_norm = sqrt(cross_track[0] * cross_track[0] +
                                 cross_track[1] * cross_track[1] +
                                 cross_track[2] * cross_track[2]);
  if (!(cross_norm > 0.0) || !isfinite(cross_norm)) return;
  for (int axis = 0; axis < 3; ++axis) cross_track[axis] /= cross_norm;
  double along_track[3] = {
      cross_track[1] * normal[2] - cross_track[2] * normal[1],
      cross_track[2] * normal[0] - cross_track[0] * normal[2],
      cross_track[0] * normal[1] - cross_track[1] * normal[0],
  };
  const double along_norm = sqrt(along_track[0] * along_track[0] +
                                 along_track[1] * along_track[1] +
                                 along_track[2] * along_track[2]);
  if (!(along_norm > 0.0) || !isfinite(along_norm)) return;
  for (int axis = 0; axis < 3; ++axis) along_track[axis] /= along_norm;
  double normal_dot_velocity = 0.0;
  double velocity_dot_along = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    normal_dot_velocity += normal[axis] * velocity_unit[axis];
    velocity_dot_along += velocity_unit[axis] * along_track[axis];
  }

  const double minor = kWgs84A * sqrt(1.0 - kWgs84E2);
  const double eta = 1.0 / sqrt((sat[0] / kWgs84A) * (sat[0] / kWgs84A) +
      (sat[1] / kWgs84A) * (sat[1] / kWgs84A) +
      (sat[2] / minor) * (sat[2] / minor));
  const double radius = eta * satellite_norm;
  const double ellipsoid_height = (1.0 - eta) * satellite_norm;
  double height = seed;
  if (height < min_height || height > max_height) return;

  double old_llh[3] = {0.0, 0.0, height};
  double latest_range_residual = nan("");
  double latest_doppler_residual = nan("");
  double decision = nan("");
  bool solved = false;
  bool stopped_early = false;
  int32_t attempts = 0;

  for (int64_t iteration = 0; iteration < budget; ++iteration) {
    if (!isfinite(height) || height < min_height || height > max_height ||
        !(ellipsoid_height - height <= target)) {
      stopped_early = true;
      break;
    }
    const double semi_minor = radius + height;
    const double cos_theta = 0.5 *
        (satellite_norm / target + target / satellite_norm -
         (semi_minor / satellite_norm) * (semi_minor / target));
    const double sin_theta = sqrt(fmax(0.0, 1.0 - cos_theta * cos_theta));
    const double gamma = target * cos_theta;
    const double alpha = -gamma * normal_dot_velocity /
                         safe_denominator(velocity_dot_along);
    const double radicand =
        target * sin_theta * target * sin_theta - alpha * alpha;
    if (!isfinite(radicand) || radicand < -1.0e-6) {
      stopped_early = true;
      break;
    }
    const double beta = look_sign * sqrt(fmax(0.0, radicand));
    double target_xyz[3];
    for (int axis = 0; axis < 3; ++axis)
      target_xyz[axis] = sat[axis] + alpha * along_track[axis] +
                         beta * cross_track[axis] + gamma * normal[axis];
    double latitude = 0.0, longitude = 0.0, ellipsoid_point_height = 0.0;
    ecef_to_llh(target_xyz, &latitude, &longitude, &ellipsoid_point_height);
    if (iteration == 0) {
      old_llh[0] = longitude;
      old_llh[1] = latitude;
      old_llh[2] = height;
    }
    const double row = latitude / kDegreesToRadians;
    const double column = longitude / kDegreesToRadians;
    bool dem_valid = dem == nullptr;
    const double sampled_height = dem == nullptr
        ? reference_height
        : sample_dem(dem, dem_rows, dem_columns,
                     (row - dem_y_start_deg) / dem_dy_deg,
                     (column - dem_x_start_deg) / dem_dx_deg,
                     reference_height, &dem_valid);
    if (!dem_valid || !isfinite(sampled_height) ||
        sampled_height < min_height || sampled_height > max_height) {
      stopped_early = true;
      break;
    }
    double dem_llh[3] = {longitude, latitude, sampled_height};
    double dem_xyz[3];
    llh_to_ecef(dem_llh[1] / kDegreesToRadians,
                dem_llh[0] / kDegreesToRadians, dem_llh[2], dem_xyz);
    const double new_height = sqrt(dem_xyz[0] * dem_xyz[0] +
                                   dem_xyz[1] * dem_xyz[1] +
                                   dem_xyz[2] * dem_xyz[2]) - radius;
    const double look[3] = {dem_xyz[0] - sat[0], dem_xyz[1] - sat[1],
                            dem_xyz[2] - sat[2]};
    const double slant_range = sqrt(look[0] * look[0] + look[1] * look[1] +
                                    look[2] * look[2]);
    if (!(slant_range > 0.0) || !isfinite(slant_range) ||
        !isfinite(new_height) || new_height < min_height ||
        new_height > max_height) {
      stopped_early = true;
      break;
    }
    double dot = 0.0;
    for (int axis = 0; axis < 3; ++axis)
      dot += vel[axis] * look[axis] / slant_range;
    latest_range_residual = slant_range - target;
    latest_doppler_residual = 2.0 * dot / wavelength;
    ++attempts;
    const double decision_metric =
        fmax(fabs(latest_range_residual) / range_tolerance,
             fabs(latest_doppler_residual) / doppler_tolerance);
    const bool now_converged =
        isfinite(decision_metric) &&
        fabs(latest_range_residual) < range_tolerance &&
        fabs(latest_doppler_residual) < doppler_tolerance;
    decision = latest_range_residual;
    if (now_converged) {
      solved = true;
      height = sampled_height;
      break;
    }
    if (iteration >= primary_iter) {
      double old_xyz[3];
      llh_to_ecef(old_llh[1], old_llh[0], old_llh[2], old_xyz);
      for (int axis = 0; axis < 3; ++axis)
        target_xyz[axis] = 0.5 * (old_xyz[axis] + dem_xyz[axis]);
      double average_latitude = 0.0, average_longitude = 0.0;
      double average_height = 0.0;
      ecef_to_llh(target_xyz, &average_latitude, &average_longitude,
                  &average_height);
      old_llh[0] = average_longitude;
      old_llh[1] = average_latitude;
      old_llh[2] = average_height;
      height = sqrt(target_xyz[0] * target_xyz[0] +
                    target_xyz[1] * target_xyz[1] +
                    target_xyz[2] * target_xyz[2]) - radius;
    } else {
      height = new_height;
    }
  }

  const bool budget_exhausted = !solved && !stopped_early && attempts >= budget;
  if (!solved && !budget_exhausted) return;
  if (budget_exhausted) {
    iterations[point] = static_cast<int32_t>(budget);
    max_iter_exhausted[point] = true;
    tolerance[point] = range_tolerance;
    range_residual[point] = latest_range_residual;
    doppler_residual[point] = latest_doppler_residual;
    decision_residual[point] = latest_range_residual;
    final_residual[point] = latest_range_residual;
    return;
  }

  const double semi_minor = radius + height;
  const double cos_theta = 0.5 *
      (satellite_norm / target + target / satellite_norm -
       (semi_minor / satellite_norm) * (semi_minor / target));
  const double sin_theta = sqrt(fmax(0.0, 1.0 - cos_theta * cos_theta));
  const double gamma = target * cos_theta;
  const double alpha = -gamma * normal_dot_velocity /
                       safe_denominator(velocity_dot_along);
  const double radicand =
      target * sin_theta * target * sin_theta - alpha * alpha;
  if (!isfinite(radicand) || radicand < -1.0e-6) return;
  const double beta = look_sign * sqrt(fmax(0.0, radicand));
  double final_xyz[3];
  for (int axis = 0; axis < 3; ++axis)
    final_xyz[axis] = sat[axis] + alpha * along_track[axis] +
                      beta * cross_track[axis] + gamma * normal[axis];
  double final_latitude = 0.0, final_longitude = 0.0, final_height = 0.0;
  ecef_to_llh(final_xyz, &final_latitude, &final_longitude, &final_height);
  const double final_look[3] = {final_xyz[0] - sat[0], final_xyz[1] - sat[1],
                                final_xyz[2] - sat[2]};
  const double final_slant_range = sqrt(
      final_look[0] * final_look[0] + final_look[1] * final_look[1] +
      final_look[2] * final_look[2]);
  double final_doppler = 0.0;
  if (!(final_slant_range > 0.0) || !isfinite(final_slant_range)) return;
  for (int axis = 0; axis < 3; ++axis)
    final_doppler += vel[axis] * final_look[axis] / final_slant_range;
  latest_range_residual = final_slant_range - target;
  latest_doppler_residual = 2.0 * final_doppler / wavelength;
  const double final_metric =
      fmax(fabs(latest_range_residual) / range_tolerance,
           fabs(latest_doppler_residual) / doppler_tolerance);
  const bool final_converged =
      isfinite(final_metric) && fabs(latest_range_residual) < range_tolerance &&
      fabs(latest_doppler_residual) < doppler_tolerance;
  if (!final_converged) return;
  iterations[point] = attempts;
  max_iter_exhausted[point] = false;
  tolerance[point] = range_tolerance;
  range_residual[point] = latest_range_residual;
  doppler_residual[point] = latest_doppler_residual;
  decision_residual[point] = latest_range_residual;
  final_residual[point] = latest_range_residual;
  latitude_deg[point] = final_latitude / kDegreesToRadians;
  longitude_deg[point] = final_longitude / kDegreesToRadians;
  output_height[point] = dem == nullptr ? height : final_height;
  output_range[point] = range;
  output_azimuth[point] = azimuth;
  converged[point] = true;
}

}  // namespace

std::vector<Tensor> rdr2geo_tcn_cuda_v2_with_visit_counts(
    const Tensor& azimuth_index, const Tensor& range_index,
    const Tensor& height_seed_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m,
    const Tensor& dem_height_m, double dem_x_start_deg, double dem_y_start_deg,
    double dem_dx_deg, double dem_dy_deg, double reference_height_m,
    double min_height_m, double max_height_m, double wavelength_m,
    double range_tolerance_m, double doppler_tolerance_hz, int64_t max_iter,
    int64_t extra_iter, bool right_looking) {
  check_cuda_vector(azimuth_index, "azimuth_index");
  check_cuda_vector(range_index, "range_index");
  check_cuda_vector(height_seed_m, "height_seed_m");
  check_cuda_vector(orbit_times_s, "orbit_times_s");
  const int device = azimuth_index.get_device();
  for (const auto& input : {range_index, height_seed_m, orbit_times_s})
    check_same_device(azimuth_index, input, "input tensor");
  TORCH_CHECK(azimuth_index.numel() == range_index.numel() &&
              azimuth_index.numel() == height_seed_m.numel(),
              "point arrays must have equal lengths");
  TORCH_CHECK(orbit_times_s.numel() >= 2,
              "orbit_times_s must contain at least two samples");
  check_cuda_matrix(orbit_positions_m, "orbit_positions_m",
                    orbit_times_s.numel(), device);
  check_cuda_matrix(orbit_velocities_m_s, "orbit_velocities_m_s",
                    orbit_times_s.numel(), device);
  TORCH_CHECK(std::isfinite(azimuth_time_interval_s) &&
              azimuth_time_interval_s > 0.0 &&
              std::isfinite(range_spacing_m) && range_spacing_m > 0.0 &&
              std::isfinite(starting_slant_range_m),
              "radar metadata must be finite with positive intervals");
  TORCH_CHECK(std::isfinite(sensing_offset_s),
              "sensing_offset_s must be finite");
  TORCH_CHECK(dem_height_m.is_cuda() && dem_height_m.get_device() == device,
              "dem_height_m must be on the same CUDA device");
  TORCH_CHECK(dem_height_m.scalar_type() == torch::kFloat64 &&
              dem_height_m.is_contiguous() &&
              (dem_height_m.dim() == 0 || dem_height_m.dim() == 2),
              "dem_height_m must be a contiguous float64 2-D tensor or scalar");
  TORCH_CHECK(std::isfinite(dem_x_start_deg) && std::isfinite(dem_y_start_deg) &&
              std::isfinite(dem_dx_deg) && std::isfinite(dem_dy_deg) &&
              dem_dx_deg != 0.0 && dem_dy_deg != 0.0,
              "DEM coordinates must be finite with non-zero spacing");
  TORCH_CHECK(std::isfinite(reference_height_m) &&
              std::isfinite(min_height_m) && std::isfinite(max_height_m) &&
              min_height_m <= max_height_m &&
              std::isfinite(wavelength_m) && wavelength_m > 0.0,
              "height limits/reference and wavelength must be finite");
  check_solver_scalars(max_iter, extra_iter, range_tolerance_m,
                       "range_tolerance_m");
  TORCH_CHECK(std::isfinite(doppler_tolerance_hz) &&
              doppler_tolerance_hz > 0.0,
              "doppler_tolerance_hz must be finite and positive");
  TORCH_CHECK(reference_height_m >= min_height_m &&
              reference_height_m <= max_height_m,
              "reference_height_m must be within height limits");
  TORCH_CHECK(torch::isfinite(orbit_times_s).all().item<bool>() &&
              (orbit_times_s.slice(0, 1) - orbit_times_s.slice(0, 0, -1))
                      .gt(0).all().item<bool>(),
              "orbit_times_s must be finite and strictly increasing");
  if (dem_height_m.dim() == 2) {
    TORCH_CHECK(dem_height_m.size(0) >= 6 && dem_height_m.size(1) >= 6,
                "DEM bucket must have at least 6x6 samples");
    TORCH_CHECK(torch::isfinite(dem_height_m).all().item<bool>(),
                "DEM bucket must contain only finite values");
  }

  c10::cuda::CUDAGuard guard(azimuth_index.device());
  const auto options = azimuth_index.options();
  const int64_t count = azimuth_index.numel();
  auto latitude = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto longitude = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto output_height = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto output_range = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto output_azimuth = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto converged = torch::zeros({count}, options.dtype(torch::kBool));
  auto iterations = torch::full({count}, -1, options.dtype(torch::kInt32));
  auto decision_residual = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto final_residual = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto tolerance = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto max_iter_exhausted = torch::zeros({count}, options.dtype(torch::kBool));
  auto boundary_rechecked = torch::zeros({count}, options.dtype(torch::kBool));
  auto range_residual = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto doppler_residual = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto visit_counts = torch::zeros({count}, options.dtype(torch::kInt32));
  if (count > 0) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream(device);
    const double* dem_ptr = dem_height_m.dim() == 2
        ? dem_height_m.data_ptr<double>() : nullptr;
    const int64_t dem_rows = dem_height_m.dim() == 2 ? dem_height_m.size(0) : 0;
    const int64_t dem_columns = dem_height_m.dim() == 2 ? dem_height_m.size(1) : 0;
    rdr2geo_tcn_kernel<<<blocks, threads, 0, stream.stream()>>>(
        azimuth_index.data_ptr<double>(), range_index.data_ptr<double>(),
        height_seed_m.data_ptr<double>(), count,
        orbit_times_s.data_ptr<double>(), orbit_positions_m.data_ptr<double>(),
        orbit_velocities_m_s.data_ptr<double>(), orbit_times_s.numel(),
        sensing_offset_s, azimuth_time_interval_s, starting_slant_range_m,
        range_spacing_m, dem_ptr, dem_rows, dem_columns, dem_x_start_deg,
        dem_y_start_deg, dem_dx_deg, dem_dy_deg, reference_height_m,
        min_height_m, max_height_m, wavelength_m, range_tolerance_m,
        doppler_tolerance_hz, max_iter, max_iter + extra_iter,
        right_looking ? 1.0 : -1.0, latitude.data_ptr<double>(),
        longitude.data_ptr<double>(), output_height.data_ptr<double>(),
        output_range.data_ptr<double>(), output_azimuth.data_ptr<double>(),
        converged.data_ptr<bool>(), iterations.data_ptr<int32_t>(),
        decision_residual.data_ptr<double>(), final_residual.data_ptr<double>(),
        tolerance.data_ptr<double>(), max_iter_exhausted.data_ptr<bool>(),
        boundary_rechecked.data_ptr<bool>(), range_residual.data_ptr<double>(),
        doppler_residual.data_ptr<double>(), visit_counts.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {latitude, longitude, output_height, output_range, output_azimuth,
          converged, iterations, decision_residual, final_residual, tolerance,
          max_iter_exhausted, boundary_rechecked, range_residual,
          doppler_residual, visit_counts};
}

std::vector<Tensor> rdr2geo_tcn_cuda_v2(
    const Tensor& azimuth_index, const Tensor& range_index,
    const Tensor& height_seed_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m,
    const Tensor& dem_height_m, double dem_x_start_deg, double dem_y_start_deg,
    double dem_dx_deg, double dem_dy_deg, double reference_height_m,
    double min_height_m, double max_height_m, double wavelength_m,
    double range_tolerance_m, double doppler_tolerance_hz, int64_t max_iter,
    int64_t extra_iter, bool right_looking) {
  auto result = rdr2geo_tcn_cuda_v2_with_visit_counts(
      azimuth_index, range_index, height_seed_m, orbit_times_s,
      orbit_positions_m, orbit_velocities_m_s, sensing_offset_s,
      azimuth_time_interval_s, starting_slant_range_m, range_spacing_m,
      dem_height_m, dem_x_start_deg, dem_y_start_deg, dem_dx_deg, dem_dy_deg,
      reference_height_m, min_height_m, max_height_m, wavelength_m,
      range_tolerance_m, doppler_tolerance_hz, max_iter, extra_iter,
      right_looking);
  result.pop_back();
  return result;
}

Tensor rdr2geo_tcn_cuda_v2_visit_counts(
    const Tensor& azimuth_index, const Tensor& range_index,
    const Tensor& height_seed_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m,
    const Tensor& dem_height_m, double dem_x_start_deg, double dem_y_start_deg,
    double dem_dx_deg, double dem_dy_deg, double reference_height_m,
    double min_height_m, double max_height_m, double wavelength_m,
    double range_tolerance_m, double doppler_tolerance_hz, int64_t max_iter,
    int64_t extra_iter, bool right_looking) {
  return rdr2geo_tcn_cuda_v2_with_visit_counts(
             azimuth_index, range_index, height_seed_m, orbit_times_s,
             orbit_positions_m, orbit_velocities_m_s, sensing_offset_s,
             azimuth_time_interval_s, starting_slant_range_m, range_spacing_m,
             dem_height_m, dem_x_start_deg, dem_y_start_deg, dem_dx_deg,
             dem_dy_deg, reference_height_m, min_height_m, max_height_m,
             wavelength_m, range_tolerance_m, doppler_tolerance_hz, max_iter,
             extra_iter, right_looking)
      .back();
}

std::vector<Tensor> rdr2geo_cuda_v2(
    const Tensor& azimuth_index, const Tensor& range_index,
    const Tensor& height_seed_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m,
    const Tensor& dem_height_m, double dem_x_start_deg, double dem_y_start_deg,
    double dem_dx_deg, double dem_dy_deg, double reference_height_m,
    double min_height_m, double max_height_m, double wavelength_m,
    double range_tolerance_m, double doppler_tolerance_hz, int64_t max_iter,
    int64_t extra_iter, bool right_looking) {
  return rdr2geo_tcn_cuda_v2(
      azimuth_index, range_index, height_seed_m, orbit_times_s,
      orbit_positions_m, orbit_velocities_m_s, sensing_offset_s,
      azimuth_time_interval_s, starting_slant_range_m, range_spacing_m,
      dem_height_m, dem_x_start_deg, dem_y_start_deg, dem_dx_deg, dem_dy_deg,
      reference_height_m, min_height_m, max_height_m, wavelength_m,
      range_tolerance_m, doppler_tolerance_hz, max_iter, extra_iter,
      right_looking);
}

}  // namespace faninsar::geometry::cuda_v2
