#pragma once

#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace faninsar_native_v2 {

using Vec3 = std::array<double, 3>;
using Tensor = torch::Tensor;

constexpr double kWgs84SemiMajorAxisM = 6378137.0;
constexpr double kWgs84EccentricitySquared = 6.6943799901413165e-3;

struct OrbitState {
  Vec3 position{};
  Vec3 velocity{};
  Vec3 acceleration{};
};

void check_vector(const Tensor& tensor, const char* name);
void check_orbit(const Tensor& tensor, const char* name, int64_t count);
void check_orbit_times(const Tensor& times);
int64_t orbit_segment(const double* times, int64_t count, double time_s);
OrbitState interpolate_orbit(const double* times, const double* positions,
                             const double* velocities, int64_t count,
                             double time_s);
Vec3 llh_to_ecef(double latitude_deg, double longitude_deg, double height_m);
Vec3 ecef_to_llh(const Vec3& ecef);
double dot(const Vec3& left, const Vec3& right);
double norm(const Vec3& value);

struct TelemetrySnapshot {
  bool openmp_defined;
  std::string runtime_name;
  std::vector<int64_t> thread_ids;
  std::vector<int64_t> visit_counts;
};

void begin_telemetry(int64_t point_count);
void record_visit(int64_t index);
TelemetrySnapshot telemetry_snapshot();

std::vector<Tensor> invalid_result(int64_t count, bool geo2rdr,
                                   double tolerance);

}  // namespace faninsar_native_v2

