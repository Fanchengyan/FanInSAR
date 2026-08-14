#include "native_v2_abi.h"

#include <pybind11/stl.h>

namespace faninsar_native_v2 {

std::vector<Tensor> geo2rdr_cpu(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, double, double, double, double, double, int64_t, int64_t,
    double, double, double, bool);
std::vector<Tensor> rdr2geo_cpu(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, double, double, double, double, double, int64_t, int64_t,
    double, double, bool);

}  // namespace faninsar_native_v2

namespace {

pybind11::dict native_v2_telemetry() {
  const auto snapshot = faninsar_native_v2::telemetry_snapshot();
  pybind11::dict result;
  result["openmp_defined"] = snapshot.openmp_defined;
  result["runtime_name"] = snapshot.runtime_name;
  result["thread_ids"] = snapshot.thread_ids;
  result["visit_counts"] = snapshot.visit_counts;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("geo2rdr_cpu", &faninsar_native_v2::geo2rdr_cpu,
             "Run the OpenMP CPU geo2rdr kernel");
  module.def("rdr2geo_cpu", &faninsar_native_v2::rdr2geo_cpu,
             "Run the OpenMP CPU rdr2geo kernel");
  module.def("native_v2_telemetry", &native_v2_telemetry,
             "Return exact geometry-loop OpenMP telemetry");
  module.def("native_source_abi", []() {
    return std::string("faninsar.geometry.native_v2.14-field.v1");
  });
}

