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
  const double horizontal = hypot(x, y);
  *longitude = atan2(y, x);
  double latitude_estimate = atan2(z, horizontal * (1.0 - kWgs84E2));
  for (int iteration = 0; iteration < 8; ++iteration) {
    const double sine = sin(latitude_estimate);
    const double radius = kWgs84A /
        sqrt(1.0 - kWgs84E2 * sine * sine);
    const double next = atan2(z + kWgs84E2 * radius * sine, horizontal);
    if (fabs(next - latitude_estimate) < 1.0e-14) {
      latitude_estimate = next;
      break;
    }
    latitude_estimate = next;
  }
  const double sine = sin(latitude_estimate);
  const double cosine = cos(latitude_estimate);
  const double radius = kWgs84A /
      sqrt(1.0 - kWgs84E2 * sine * sine);
  *latitude = latitude_estimate;
  *height = (fabs(cosine) > 1.0e-12) ? horizontal / cosine - radius
                                     : fabs(z) - radius * (1.0 - kWgs84E2);
}

__device__ inline double spline_six(const double* values, double fraction) {
  double second[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double recurrence[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int index = 1; index < 5; ++index) {
    const double denominator = recurrence[index - 1] / 2.0 + 2.0;
    recurrence[index] = -0.5 / denominator;
    second[index] = (3.0 * (values[index + 1] - 2.0 * values[index] +
                            values[index - 1]) - second[index - 1] / 2.0) /
                    denominator;
  }
  for (int index = 4; index > 0; --index) {
    second[index] = recurrence[index] * second[index + 1] + second[index];
  }
  return values[1] + fraction *
      (values[2] - values[1] - second[1] / 3.0 - second[2] / 6.0 +
       fraction * (second[1] / 2.0 + fraction *
                   (second[2] - second[1]) / 6.0));
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
  double along_columns[6];
  for (int row_offset = -1; row_offset <= 4; ++row_offset) {
    double values[6];
    for (int column_offset = -1; column_offset <= 4; ++column_offset) {
      values[column_offset + 1] = bucket[(row_base + row_offset) * columns +
                                         column_base + column_offset];
    }
    along_columns[row_offset + 1] =
        spline_six(values, column - static_cast<double>(column_base));
  }
  const double value = spline_six(along_columns, row - static_cast<double>(row_base));
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

std::vector<Tensor> geo2rdr_cuda_v2(
    const Tensor& latitude_deg, const Tensor& longitude_deg,
    const Tensor& height_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m,
    double wavelength_m, int64_t max_iter, int64_t extra_iter,
    double time_tol_s, double range_tolerance_m, double doppler_tolerance_hz);

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
