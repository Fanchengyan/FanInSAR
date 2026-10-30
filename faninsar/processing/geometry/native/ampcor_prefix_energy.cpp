#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

int64_t checked_product(int64_t left, int64_t right, const char* name) {
  TORCH_CHECK(left >= 0 && right >= 0, name, " dimensions must be non-negative");
  TORCH_CHECK(left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
              name, " dimension product overflows int64");
  return left * right;
}

}  // namespace

torch::Tensor ampcor_prefix_energy_cpu(const torch::Tensor& secondary_searches,
                                       int64_t window_height,
                                       int64_t window_width) {
  TORCH_CHECK(!secondary_searches.is_cuda() && secondary_searches.device().is_cpu(),
              "secondary_searches must be a CPU tensor");
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
  TORCH_CHECK(input_height < std::numeric_limits<int64_t>::max(),
              "prefix height overflows int64");
  TORCH_CHECK(input_width < std::numeric_limits<int64_t>::max(),
              "prefix width overflows int64");
  const int64_t prefix_height = input_height + 1;
  const int64_t prefix_width = input_width + 1;
  const int64_t prefix_plane =
      checked_product(prefix_height, prefix_width, "prefix");
  checked_product(batch, prefix_plane, "prefix");

  auto prefix = torch::empty(
      {batch, prefix_height, prefix_width}, secondary_searches.options());
  auto result = torch::empty(
      {batch, output_height, output_width}, secondary_searches.options());
  const auto* input = secondary_searches.data_ptr<double>();
  auto* prefix_values = prefix.data_ptr<double>();
  auto* output = result.data_ptr<double>();

#pragma omp parallel for schedule(static)
  for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
    const auto* input_batch = input + batch_index * input_plane;
    auto* prefix_batch = prefix_values + batch_index * prefix_plane;
    for (int64_t column = 0; column < prefix_width; ++column) {
      prefix_batch[column] = 0.0;
    }
    for (int64_t row = 1; row < prefix_height; ++row) {
      prefix_batch[row * prefix_width] = 0.0;
      double row_energy = 0.0;
      for (int64_t column = 1; column < prefix_width; ++column) {
        const double value = input_batch[(row - 1) * input_width + column - 1];
        row_energy += value * value;
        prefix_batch[row * prefix_width + column] =
            prefix_batch[(row - 1) * prefix_width + column] + row_energy;
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int64_t index = 0; index < output_elements; ++index) {
    const int64_t plane_index = index % output_plane;
    const int64_t batch_index = index / output_plane;
    const int64_t output_row = plane_index / output_width;
    const int64_t output_column = plane_index % output_width;
    const auto* prefix_batch = prefix_values + batch_index * prefix_plane;
    const int64_t top = output_row;
    const int64_t left = output_column;
    const int64_t bottom = top + window_height;
    const int64_t right = left + window_width;
    output[index] = prefix_batch[bottom * prefix_width + right] -
                    prefix_batch[top * prefix_width + right] -
                    prefix_batch[bottom * prefix_width + left] +
                    prefix_batch[top * prefix_width + left];
  }
  return result;
}
