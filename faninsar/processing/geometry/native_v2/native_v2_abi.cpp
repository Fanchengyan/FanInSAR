#include "native_v2_abi.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#include <sched.h>
#endif

namespace faninsar_native_v2 {
namespace {

std::mutex telemetry_mutex;
TelemetrySnapshot telemetry;
std::vector<int64_t> thread_slots;
std::vector<int64_t> affinity_slots;

void check_float64_cpu(const Tensor& tensor, const char* name) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat64, name,
              " must have dtype torch.float64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void check_vector(const Tensor& tensor, const char* name) {
  check_float64_cpu(tensor, name);
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
}

void check_orbit(const Tensor& tensor, const char* name, int64_t count) {
  check_float64_cpu(tensor, name);
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == count && tensor.size(1) == 3,
              name, " must have shape (orbit_vector_count, 3)");
}

void check_orbit_times(const Tensor& times) {
  TORCH_CHECK(times.numel() >= 2, "orbit_times_s must contain at least two values");
  const auto* values = times.data_ptr<double>();
  for (int64_t index = 1; index < times.numel(); ++index) {
    TORCH_CHECK(std::isfinite(values[index]) && values[index] > values[index - 1],
                "orbit_times_s must be finite and strictly increasing");
  }
  TORCH_CHECK(std::isfinite(values[0]),
              "orbit_times_s must be finite and strictly increasing");
}

int64_t orbit_segment(const double* times, int64_t count, double time_s) {
  int64_t segment = 0;
  while (segment + 1 < count && times[segment + 1] <= time_s) {
    ++segment;
  }
  return std::clamp<int64_t>(segment, 0, count - 2);
}

OrbitState interpolate_orbit(const double* times, const double* positions,
                             const double* velocities, int64_t count,
                             double time_s) {
  const int64_t segment = orbit_segment(times, count, time_s);
  const double t0 = times[segment];
  const double duration = times[segment + 1] - t0;
  const double u = (time_s - t0) / duration;
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
  OrbitState state;
  for (int axis = 0; axis < 3; ++axis) {
    const double p0 = positions[segment * 3 + axis];
    const double p1 = positions[(segment + 1) * 3 + axis];
    const double v0 = velocities[segment * 3 + axis];
    const double v1 = velocities[(segment + 1) * 3 + axis];
    state.position[axis] = h00 * p0 + h10 * duration * v0 + h01 * p1 +
                           h11 * duration * v1;
    state.velocity[axis] = dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1;
    state.acceleration[axis] = d2h00 * p0 + d2h10 * v0 + d2h01 * p1 +
                               d2h11 * v1;
  }
  return state;
}

Vec3 llh_to_ecef(double latitude_deg, double longitude_deg, double height_m) {
  constexpr double radians = 0.017453292519943295769;
  const double latitude = latitude_deg * radians;
  const double longitude = longitude_deg * radians;
  const double sin_latitude = std::sin(latitude);
  const double cos_latitude = std::cos(latitude);
  const double radius = kWgs84SemiMajorAxisM /
                        std::sqrt(1.0 - kWgs84EccentricitySquared *
                                              sin_latitude * sin_latitude);
  return {(radius + height_m) * cos_latitude * std::cos(longitude),
          (radius + height_m) * cos_latitude * std::sin(longitude),
          (radius * (1.0 - kWgs84EccentricitySquared) + height_m) *
              sin_latitude};
}

Vec3 ecef_to_llh(const Vec3& ecef) {
  constexpr double radians_to_degrees = 57.295779513082320876;
  const double longitude = std::atan2(ecef[1], ecef[0]);
  const double horizontal = std::hypot(ecef[0], ecef[1]);
  double latitude = std::atan2(ecef[2], horizontal *
                                         (1.0 - kWgs84EccentricitySquared));
  for (int iteration = 0; iteration < 12; ++iteration) {
    const double sin_latitude = std::sin(latitude);
    const double radius = kWgs84SemiMajorAxisM /
                          std::sqrt(1.0 - kWgs84EccentricitySquared *
                                                sin_latitude * sin_latitude);
    latitude = std::atan2(
        ecef[2] + kWgs84EccentricitySquared * radius * sin_latitude,
        horizontal);
  }
  const double sin_latitude = std::sin(latitude);
  const double cos_latitude = std::cos(latitude);
  const double radius = kWgs84SemiMajorAxisM /
                        std::sqrt(1.0 - kWgs84EccentricitySquared *
                                              sin_latitude * sin_latitude);
  const double height = std::abs(cos_latitude) > 1.0e-12
                            ? horizontal / cos_latitude - radius
                            : std::abs(ecef[2]) /
                                      std::max(std::abs(sin_latitude), 1.0e-16) -
                                  radius * (1.0 - kWgs84EccentricitySquared);
  return {latitude * radians_to_degrees, longitude * radians_to_degrees, height};
}

double dot(const Vec3& left, const Vec3& right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

double norm(const Vec3& value) { return std::sqrt(dot(value, value)); }

void begin_telemetry(int64_t point_count, const char* operation_symbol) {
  std::lock_guard<std::mutex> lock(telemetry_mutex);
  telemetry.openmp_defined = false;
#ifdef _OPENMP
  telemetry.openmp_defined = true;
#if defined(FANINSAR_OPENMP_RUNTIME_LIBOMP)
  telemetry.runtime_name = "libomp";
#elif defined(FANINSAR_OPENMP_RUNTIME_LIBGOMP)
  telemetry.runtime_name = "libgomp";
#else
  telemetry.runtime_name = "openmp";
#endif
#else
  telemetry.runtime_name = "serial";
#endif
  if (const char* runtime = std::getenv("FANINSAR_OPENMP_RUNTIME");
      runtime != nullptr && *runtime != '\0') {
    telemetry.runtime_name = runtime;
  }
  telemetry.operation_symbol = operation_symbol;
  telemetry.processed_point_count = 0;
  telemetry.thread_ids.clear();
  telemetry.visit_counts.assign(static_cast<size_t>(point_count), 0);
#ifdef _OPENMP
  const int64_t thread_capacity = std::max(1, omp_get_max_threads());
#else
  constexpr int64_t thread_capacity = 1;
#endif
  thread_slots.assign(static_cast<size_t>(thread_capacity), -1);
  affinity_slots.assign(static_cast<size_t>(thread_capacity), -1);
}

void record_visit(int64_t index) {
  std::lock_guard<std::mutex> lock(telemetry_mutex);
  int64_t thread_id = 0;
#ifdef _OPENMP
  thread_id = omp_get_thread_num();
#endif
  telemetry.visit_counts[static_cast<size_t>(index)] += 1;
  thread_slots[static_cast<size_t>(thread_id)] = thread_id;
#ifdef __linux__
  affinity_slots[static_cast<size_t>(thread_id)] = sched_getcpu();
#else
  // Apple does not expose a portable current-CPU query; retain the worker
  // identity so qualification can still prove deterministic coverage.
  affinity_slots[static_cast<size_t>(thread_id)] = thread_id;
#endif
}

TelemetrySnapshot telemetry_snapshot() {
  std::lock_guard<std::mutex> lock(telemetry_mutex);
  TelemetrySnapshot snapshot = telemetry;
  snapshot.processed_point_count = 0;
  for (const int64_t count : telemetry.visit_counts) {
    if (count > 0) ++snapshot.processed_point_count;
  }
  for (size_t index = 0; index < thread_slots.size(); ++index) {
    if (thread_slots[index] >= 0) {
      snapshot.thread_ids.push_back(thread_slots[index]);
      snapshot.observed_affinity.push_back(affinity_slots[index]);
    }
  }
  return snapshot;
}

std::vector<Tensor> invalid_result(int64_t count, bool geo2rdr,
                                   double tolerance) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto nan = torch::full({count}, std::numeric_limits<double>::quiet_NaN(), options);
  auto boolean = torch::zeros({count}, options.dtype(torch::kBool));
  auto iterations = torch::full({count}, -1, options.dtype(torch::kInt32));
  auto tolerance_values = torch::full({count}, geo2rdr ? 1.0 : tolerance, options);
  return {nan.clone(), nan.clone(), nan.clone(), nan.clone(), nan.clone(),
          boolean.clone(), iterations, nan.clone(), nan.clone(),
          tolerance_values, boolean.clone(), boolean.clone(), nan.clone(),
          nan.clone()};
}

}  // namespace faninsar_native_v2
