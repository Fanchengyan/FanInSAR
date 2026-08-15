#pragma once

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace faninsar::geometry::cuda_v2 {

using torch::Tensor;

constexpr double kWgs84A = 6378137.0;
constexpr double kWgs84E2 = 6.6943799901413165e-3;
constexpr double kDegreesToRadians = 0.017453292519943295769;

inline void check_cuda_vector(const Tensor& value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.scalar_type() == torch::kFloat64,
              name, " must have dtype torch.float64");
  TORCH_CHECK(value.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

inline void check_cuda_matrix(const Tensor& value, const char* name,
                              int64_t rows, int device) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.get_device() == device, name,
              " must be on the same CUDA device as the point arrays");
  TORCH_CHECK(value.scalar_type() == torch::kFloat64,
              name, " must have dtype torch.float64");
  TORCH_CHECK(value.dim() == 2 && value.size(0) == rows && value.size(1) == 3,
              name, " must have shape (orbit_count, 3)");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

inline void check_same_device(const Tensor& first, const Tensor& second,
                              const char* name) {
  TORCH_CHECK(second.is_cuda() && second.get_device() == first.get_device(),
              name, " must be on the same CUDA device as the point arrays");
}

__device__ inline int64_t orbit_segment(const double* times, int64_t count,
                                        double value) {
  int64_t low = 0;
  int64_t high = count;
  while (low < high) {
    const int64_t middle = low + (high - low) / 2;
    if (times[middle] <= value) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const int64_t upper = low - 1 < count - 2 ? low - 1 : count - 2;
  return upper > 0 ? upper : 0;
}

__device__ inline void hermite(const double* times, const double* positions,
                               const double* velocities, int64_t count,
                               double value, double* position, double* velocity,
                               double* acceleration) {
  const int64_t segment = orbit_segment(times, count, value);
  const double t0 = times[segment];
  const double duration = times[segment + 1] - t0;
  const double u = (value - t0) / duration;
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
  const double h10 = u3 - 2.0 * u2 + u;
  const double h01 = -2.0 * u3 + 3.0 * u2;
  const double h11 = u3 - u2;
  const double dh00 = (6.0 * u2 - 6.0 * u) / duration;
  const double dh10 = 3.0 * u2 - 4.0 * u + 1.0;
  const double dh01 = (-6.0 * u2 + 6.0 * u) / duration;
  const double dh11 = 3.0 * u2 - 2.0 * u;
  const double d2h00 = (12.0 * u - 6.0) / (duration * duration);
  const double d2h10 = (6.0 * u - 4.0) / duration;
  const double d2h01 = (-12.0 * u + 6.0) / (duration * duration);
  const double d2h11 = (6.0 * u - 2.0) / duration;
  for (int axis = 0; axis < 3; ++axis) {
    const double p0 = positions[segment * 3 + axis];
    const double p1 = positions[(segment + 1) * 3 + axis];
    const double v0 = velocities[segment * 3 + axis];
    const double v1 = velocities[(segment + 1) * 3 + axis];
    position[axis] = h00 * p0 + h10 * duration * v0 + h01 * p1 +
                     h11 * duration * v1;
    velocity[axis] = dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1;
    acceleration[axis] = d2h00 * p0 + d2h10 * v0 + d2h01 * p1 + d2h11 * v1;
  }
}

__device__ inline void llh_to_ecef(double latitude_deg, double longitude_deg,
                                   double height, double* ecef) {
  const double latitude = latitude_deg * kDegreesToRadians;
  const double longitude = longitude_deg * kDegreesToRadians;
  const double sin_latitude = sin(latitude);
  const double cos_latitude = cos(latitude);
  const double radius = kWgs84A /
      sqrt(1.0 - kWgs84E2 * sin_latitude * sin_latitude);
  ecef[0] = (radius + height) * cos_latitude * cos(longitude);
  ecef[1] = (radius + height) * cos_latitude * sin(longitude);
  ecef[2] = (radius * (1.0 - kWgs84E2) + height) * sin_latitude;
}

__device__ inline void ecef_to_llh(const double* ecef, double* latitude,
                                   double* longitude, double* height) {
  const double x = ecef[0];
  const double y = ecef[1];
  const double z = ecef[2];
  *latitude = nan("");
  *longitude = nan("");
  *height = nan("");
  // Keep the device conversion identical to the Torch and CPU TCN paths.
  // Iterating latitude here produces millimetre-scale coordinate drift after
  // the DEM fixed-point loop, even though the underlying ECEF point matches.
  const double e4 = kWgs84E2 * kWgs84E2;
  const double a2 = kWgs84A * kWgs84A;
  const double lateral = (x * x + y * y) / a2;
  const double polar = (1.0 - kWgs84E2) * z * z / a2;
  const double reduced = (lateral + polar - e4) / 6.0;
  if (!(reduced > 0.0) || !isfinite(reduced)) {
    return;
  }
  const double cubic = e4 * lateral * polar /
                       (4.0 * reduced * reduced * reduced);
  const double cubic_radical = cubic * (2.0 + cubic);
  if (!(cubic_radical >= 0.0) || !isfinite(cubic_radical)) {
    return;
  }
  const double root_argument = 1.0 + cubic + sqrt(cubic_radical);
  if (!(root_argument > 0.0) || !isfinite(root_argument)) {
    return;
  }
  const double root = pow(root_argument, 1.0 / 3.0);
  if (!(root > 0.0) || !isfinite(root)) {
    return;
  }
  const double u = reduced * (1.0 + root + 1.0 / root);
  const double radial = sqrt(u * u + e4 * polar);
  if (!(radial > 0.0) || !isfinite(radial)) {
    return;
  }
  const double w = kWgs84E2 * (u + radial - polar) / (2.0 * radial);
  const double k_argument = u + radial + w * w;
  if (!(k_argument >= 0.0) || !isfinite(k_argument)) {
    return;
  }
  const double k = sqrt(k_argument) - w;
  if (!(k != 0.0) || !isfinite(k)) {
    return;
  }
  const double horizontal = hypot(x, y);
  *longitude = atan2(y, x);
  const double d = k * horizontal / (k + kWgs84E2);
  *latitude = atan2(z, d);
  *height = (k + kWgs84E2 - 1.0) * hypot(d, z) / k;
}

__device__ inline void spline_six_weights(double fraction, double* weights) {
  constexpr double second_one[6] = {
      1.6076555023923444, -3.6459330143540667, 2.5837320574162677,
      -0.6889952153110048, 0.1722488038277512, -0.0287081339712919};
  constexpr double second_two[6] = {
      -0.4306220095693780, 2.5837320574162677, -4.3349282296650715,
      2.7559808612440193, -0.6889952153110048, 0.1148325358851674};
  const double fraction2 = fraction * fraction;
  const double fraction3 = fraction2 * fraction;
#pragma unroll
  for (int index = 0; index < 6; ++index) {
    weights[index] = fraction * (-second_one[index] / 3.0 -
                                 second_two[index] / 6.0) +
                     fraction2 * (second_one[index] / 2.0) +
                     fraction3 * (second_two[index] - second_one[index]) / 6.0;
  }
  weights[1] += 1.0 - fraction;
  weights[2] += fraction;
}

__device__ inline double spline_six(const double* values, double fraction) {
  double weights[6];
  spline_six_weights(fraction, weights);
  double result = 0.0;
#pragma unroll
  for (int index = 0; index < 6; ++index) {
    result += weights[index] * values[index];
  }
  return result;
}

__device__ inline double sample_dem(const double* bucket, int64_t rows,
                                    int64_t columns, double row, double column,
                                    double reference, bool* valid) {
  *valid = false;
  if (!isfinite(row) || !isfinite(column)) return reference;
  const int64_t row_base = static_cast<int64_t>(floor(row));
  const int64_t column_base = static_cast<int64_t>(floor(column));
  if (row_base < 1 || row_base > rows - 5 || column_base < 1 ||
      column_base > columns - 5) return reference;
  double row_weights[6];
  double column_weights[6];
  spline_six_weights(row - row_base, row_weights);
  spline_six_weights(column - column_base, column_weights);
  double along_columns[6];
  for (int row_offset = -1; row_offset <= 4; ++row_offset) {
    double row_result = 0.0;
#pragma unroll
    for (int column_offset = -1; column_offset <= 4; ++column_offset) {
      row_result += column_weights[column_offset + 1] *
                    bucket[(row_base + row_offset) * columns +
                           column_base + column_offset];
    }
    along_columns[row_offset + 1] = row_result;
  }
  double value = 0.0;
#pragma unroll
  for (int row_offset = 0; row_offset < 6; ++row_offset) {
    value += row_weights[row_offset] * along_columns[row_offset];
  }
  *valid = isfinite(value);
  return value;
}

inline void check_solver_scalars(int64_t max_iter, int64_t extra_iter,
                                 double tolerance, const char* tolerance_name) {
  TORCH_CHECK(max_iter > 0, "max_iter must be positive");
  TORCH_CHECK(extra_iter >= 0, "extra_iter must be non-negative");
  TORCH_CHECK(max_iter <= std::numeric_limits<int32_t>::max() - extra_iter,
              "max_iter + extra_iter exceeds int32 capacity");
  TORCH_CHECK(std::isfinite(tolerance) && tolerance > 0.0,
              tolerance_name, " must be finite and positive");
}

}  // namespace faninsar::geometry::cuda_v2

namespace faninsar::geometry::cuda_v2 {

std::vector<Tensor> geo2rdr_cuda_v2_with_visit_counts(
    const Tensor& latitude_deg, const Tensor& longitude_deg,
    const Tensor& height_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m,
    double wavelength_m, int64_t max_iter, int64_t extra_iter,
    double time_tol_s, double range_tolerance_m, double doppler_tolerance_hz);

std::vector<Tensor> geo2rdr_cuda_v2(
    const Tensor& latitude_deg, const Tensor& longitude_deg,
    const Tensor& height_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m,
    double wavelength_m, int64_t max_iter, int64_t extra_iter,
    double time_tol_s, double range_tolerance_m, double doppler_tolerance_hz);

Tensor geo2rdr_cuda_v2_visit_counts(
    const Tensor& latitude_deg, const Tensor& longitude_deg,
    const Tensor& height_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m,
    double wavelength_m, int64_t max_iter, int64_t extra_iter,
    double time_tol_s, double range_tolerance_m, double doppler_tolerance_hz);

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
    int64_t extra_iter, bool right_looking);

std::vector<Tensor> rdr2geo_tcn_cuda_v2_with_visit_counts_row_width(
    const Tensor& azimuth_index, const Tensor& range_index,
    const Tensor& height_seed_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m,
    const Tensor& dem_height_m, double dem_x_start_deg, double dem_y_start_deg,
    double dem_dx_deg, double dem_dy_deg, double reference_height_m,
    double min_height_m, double max_height_m, double wavelength_m,
    double range_tolerance_m, double doppler_tolerance_hz, int64_t max_iter,
    int64_t extra_iter, bool right_looking, int64_t row_width);

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
    int64_t extra_iter, bool right_looking);

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
    int64_t extra_iter, bool right_looking);

/// Operation-level alias retained for the later native-v2 binding adapter.
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
    int64_t extra_iter, bool right_looking);

}  // namespace faninsar::geometry::cuda_v2
