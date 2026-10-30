#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

namespace {

int64_t checked_product(int64_t left, int64_t right, const char* name) {
  TORCH_CHECK(left >= 0 && right >= 0, name, " dimensions must be non-negative");
  TORCH_CHECK(left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
              name, " dimension product overflows int64");
  return left * right;
}

__global__ void ampcor_prefix_energy_kernel(
    const double* secondary_searches, int64_t batch, int64_t input_height,
    int64_t input_width, int64_t window_height, int64_t window_width,
    int64_t output_height, int64_t output_width, int64_t output_elements,
    double* result) {
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t input_plane = input_height * input_width;
  const int64_t output_plane = output_height * output_width;
  for (int64_t index = thread; index < output_elements; index += stride) {
    const int64_t plane_index = index % output_plane;
    const int64_t batch_index = index / output_plane;
    const int64_t output_row = plane_index / output_width;
    const int64_t output_column = plane_index % output_width;
    const double* input_batch = secondary_searches + batch_index * input_plane;
    double energy = 0.0;
    for (int64_t row = 0; row < window_height; ++row) {
      const double* input_row =
          input_batch + (output_row + row) * input_width + output_column;
      for (int64_t column = 0; column < window_width; ++column) {
        const double value = input_row[column];
        energy += value * value;
      }
    }
    result[index] = energy;
  }
}

}  // namespace

torch::Tensor ampcor_prefix_energy_cuda(const torch::Tensor& secondary_searches,
                                         int64_t window_height,
                                         int64_t window_width) {
  TORCH_CHECK(secondary_searches.is_cuda(),
              "secondary_searches must be a CUDA tensor");
  TORCH_CHECK(secondary_searches.get_device() == c10::cuda::current_device(),
              "secondary_searches must be on the current CUDA device");
  TORCH_CHECK(secondary_searches.scalar_type() == torch::kFloat64,
              "secondary_searches must have dtype torch.float64");
  TORCH_CHECK(secondary_searches.dim() == 3,
              "secondary_searches must be three-dimensional");
  TORCH_CHECK(secondary_searches.is_contiguous(),
              "secondary_searches must be contiguous");
  TORCH_CHECK(window_height > 0 && window_width > 0,
              "window dimensions must be positive");

  const int64_t batch = secondary_searches.size(0);
  const int64_t input_height = secondary_searches.size(1);
  const int64_t input_width = secondary_searches.size(2);
  TORCH_CHECK(batch > 0 && input_height > 0 && input_width > 0,
              "secondary_searches dimensions must be positive");
  TORCH_CHECK(window_height <= input_height && window_width <= input_width,
              "window dimensions must fit inside secondary_searches");

  checked_product(window_height, window_width, "window");
  const int64_t output_height = input_height - window_height + 1;
  const int64_t output_width = input_width - window_width + 1;
  const int64_t output_plane =
      checked_product(output_height, output_width, "output");
  const int64_t output_elements = checked_product(batch, output_plane, "output");
  const int64_t input_plane = checked_product(input_height, input_width, "input");
  checked_product(batch, input_plane, "input");

  auto result = torch::empty(
      {batch, output_height, output_width}, secondary_searches.options());
  c10::cuda::CUDAGuard guard(secondary_searches.device());
  constexpr int threads = 256;
  const int64_t requested_blocks =
      output_elements / threads + (output_elements % threads != 0 ? 1 : 0);
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const int64_t max_blocks = properties->maxGridSize[0];
  const int blocks = static_cast<int>(
      requested_blocks > max_blocks ? max_blocks : requested_blocks);
  TORCH_CHECK(blocks > 0, "Ampcor CUDA launch grid must be positive");
  auto stream = at::cuda::getCurrentCUDAStream(secondary_searches.get_device());
  ampcor_prefix_energy_kernel<<<blocks, threads, 0, stream.stream()>>>(
      secondary_searches.data_ptr<double>(), batch, input_height, input_width,
      window_height, window_width, output_height, output_width, output_elements,
      result.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}
