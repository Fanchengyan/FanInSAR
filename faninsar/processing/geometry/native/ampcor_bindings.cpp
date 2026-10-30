#include <torch/extension.h>

#include <cstdint>
#include <string>

torch::Tensor ampcor_prefix_energy_cpu(const torch::Tensor&, int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("ampcor_prefix_energy_cpu", &ampcor_prefix_energy_cpu,
             "Compute float64 local secondary-search energy");
  module.def("native_source_abi", []() {
    return std::string("faninsar.ampcor_prefix_energy.v1");
  });
}
