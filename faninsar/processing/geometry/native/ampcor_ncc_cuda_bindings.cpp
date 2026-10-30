#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
ampcor_ncc_postprocess_cuda(const torch::Tensor&, const torch::Tensor&, int64_t,
                            int64_t, bool, double, double);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("ampcor_ncc_postprocess_cuda", &ampcor_ncc_postprocess_cuda,
             "Compute NCC peak, SNR, boundary, and cull masks");
  module.def("native_source_abi", []() {
    return std::string("faninsar.ampcor_ncc_postprocess.v1");
  });
}
