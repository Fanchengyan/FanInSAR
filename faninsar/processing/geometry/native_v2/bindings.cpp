#include "native_v2_abi.h"

#include <pybind11/stl.h>

#ifndef FANINSAR_NATIVE_V2_CUDA
namespace faninsar_native_v2 {

std::vector<Tensor> geo2rdr_cpu(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, double, double, double, double, double, int64_t, int64_t,
    double, double, double, bool);
std::vector<Tensor> rdr2geo_cpu(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, double, double, double, double, double, int64_t, int64_t,
    double, double, bool);
std::vector<Tensor> rdr2geo_cpu_dem(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, double, double, double, double, double, int64_t, int64_t,
    double, double, bool, const Tensor&, double, double, double, double,
    int64_t, double);

}  // namespace faninsar_native_v2
#endif

#ifdef FANINSAR_NATIVE_V2_CUDA
namespace faninsar::geometry::cuda_v2 {

std::vector<torch::Tensor> geo2rdr_cuda_v2(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, double,
    double, double, double, double, int64_t, int64_t, double, double, double);
std::vector<torch::Tensor> rdr2geo_cuda_v2(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, double,
    double, double, double, const torch::Tensor&, double, double, double,
    double, double, double, double, double, double, double, int64_t, int64_t,
    bool);

std::vector<torch::Tensor> geo2rdr_cuda_public(
    const torch::Tensor& latitude, const torch::Tensor& longitude,
    const torch::Tensor& height, const torch::Tensor& orbit_times,
    const torch::Tensor& orbit_positions, const torch::Tensor& orbit_velocities,
    double sensing_offset, double azimuth_interval, double starting_range,
    double range_spacing, double wavelength, int64_t max_iter,
    int64_t extra_iter, double time_tolerance, double range_tolerance,
    double doppler_tolerance) {
  auto result = geo2rdr_cuda_v2(
      latitude, longitude, height, orbit_times, orbit_positions,
      orbit_velocities, sensing_offset, azimuth_interval, starting_range,
      range_spacing, wavelength, max_iter, extra_iter, time_tolerance,
      range_tolerance, doppler_tolerance);
  TORCH_CHECK(result.size() == 15,
              "native CUDA geo2rdr diagnostic ABI must return 15 fields");
  result.pop_back();
  return result;
}

std::vector<torch::Tensor> rdr2geo_cuda_public(
    const torch::Tensor& azimuth, const torch::Tensor& range,
    const torch::Tensor& height_seed, const torch::Tensor& orbit_times,
    const torch::Tensor& orbit_positions, const torch::Tensor& orbit_velocities,
    double sensing_offset, double azimuth_interval, double starting_range,
    double range_spacing, const torch::Tensor& dem, double dem_x_start,
    double dem_y_start, double dem_dx, double dem_dy, double reference_height,
    double min_height, double max_height, double wavelength,
    double range_tolerance, double doppler_tolerance, int64_t max_iter,
    int64_t extra_iter, bool right_looking) {
  auto result = rdr2geo_cuda_v2(
      azimuth, range, height_seed, orbit_times, orbit_positions,
      orbit_velocities, sensing_offset, azimuth_interval, starting_range,
      range_spacing, dem, dem_x_start, dem_y_start, dem_dx, dem_dy,
      reference_height, min_height, max_height, wavelength, range_tolerance,
      doppler_tolerance, max_iter, extra_iter, right_looking);
  TORCH_CHECK(result.size() == 15,
              "native CUDA rdr2geo diagnostic ABI must return 15 fields");
  result.pop_back();
  return result;
}

}  // namespace faninsar::geometry::cuda_v2
#endif

namespace {

#ifndef FANINSAR_NATIVE_V2_CUDA
pybind11::dict native_v2_telemetry() {
  const auto snapshot = faninsar_native_v2::telemetry_snapshot();
  pybind11::dict result;
  result["openmp_defined"] = snapshot.openmp_defined;
  result["runtime_name"] = snapshot.runtime_name;
  result["operation_symbol"] = snapshot.operation_symbol;
  result["processed_point_count"] = snapshot.processed_point_count;
  result["thread_ids"] = snapshot.thread_ids;
  result["visit_counts"] = snapshot.visit_counts;
  result["observed_affinity"] = snapshot.observed_affinity;
  return result;
}
#endif

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
#ifndef FANINSAR_NATIVE_V2_CUDA
  module.def("geo2rdr_cpu", &faninsar_native_v2::geo2rdr_cpu,
             "Run the OpenMP CPU geo2rdr kernel");
  module.def("rdr2geo_cpu", &faninsar_native_v2::rdr2geo_cpu,
             "Run the OpenMP CPU rdr2geo kernel");
  module.def("rdr2geo_cpu_dem", &faninsar_native_v2::rdr2geo_cpu_dem,
             "Run rdr2geo with a six-by-six DEM spline fixed point");
  module.def("native_v2_telemetry", &native_v2_telemetry,
             "Return exact geometry-loop OpenMP telemetry");
#endif
  module.def("native_source_abi", []() {
    return std::string("faninsar.geometry.native_v2.14-field.v1");
  });
#ifdef FANINSAR_NATIVE_V2_CUDA
  module.def("geo2rdr_cuda", &faninsar::geometry::cuda_v2::geo2rdr_cuda_public,
             "Run the CUDA geo2rdr kernel");
  module.def("rdr2geo_cuda", &faninsar::geometry::cuda_v2::rdr2geo_cuda_public,
             "Run the CUDA rdr2geo kernel");
  module.def("geo2rdr_cuda_diagnostic",
             &faninsar::geometry::cuda_v2::geo2rdr_cuda_v2,
             "Run CUDA geo2rdr with qualification telemetry");
  module.def("rdr2geo_cuda_diagnostic",
             &faninsar::geometry::cuda_v2::rdr2geo_cuda_v2,
             "Run CUDA rdr2geo with qualification telemetry");
#endif
}
