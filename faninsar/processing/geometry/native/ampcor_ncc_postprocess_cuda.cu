#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {

int64_t checked_product(int64_t left, int64_t right, const char* name) {
  TORCH_CHECK(left >= 0 && right >= 0, name, " dimensions must be non-negative");
  TORCH_CHECK(left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
              name, " dimension product overflows int64");
  return left * right;
}

int64_t checked_twice_plus_one(int64_t half_width, const char* name) {
  TORCH_CHECK(half_width >= 0, name, " must be non-negative");
  constexpr int64_t maximum = std::numeric_limits<int64_t>::max();
  TORCH_CHECK(half_width <= (maximum - 1) / 2,
              name, " overflows twice-plus-one");
  return 2 * half_width + 1;
}

__device__ inline double normalized_value(double corr, double energy) {
  if (isnan(corr) || isnan(energy)) {
    return NAN;
  }
  const double denominator = sqrt(energy < 1e-12 ? 1e-12 : energy);
  return corr / denominator;
}

__device__ inline bool peak_is_better(double candidate_value,
                                      int64_t candidate_index,
                                      double current_value,
                                      int64_t current_index) {
  const bool candidate_nan = isnan(candidate_value);
  const bool current_nan = isnan(current_value);
  if (candidate_nan != current_nan) {
    return candidate_nan;
  }
  if (candidate_nan) {
    return candidate_index < current_index;
  }
  return candidate_value > current_value ||
         (candidate_value == current_value && candidate_index < current_index);
}

__global__ void ampcor_ncc_postprocess_kernel(
    const double* correlation, const double* energy, int64_t batch,
    int64_t height, int64_t width, int64_t search_az, int64_t search_rg,
    bool subpixel, double snr_threshold, double max_abs_residual, double* d_rg,
    double* d_az, double* snr, bool* boundary, bool* valid) {
  constexpr int threads_per_block = 256;
  __shared__ double shared_values[threads_per_block];
  __shared__ int64_t shared_indices[threads_per_block];
  __shared__ int64_t shared_counts[threads_per_block];
  // The host boundary checks this product before launch.
  const int64_t surface_size = height * width;
  for (int64_t batch_index = blockIdx.x; batch_index < batch;
       batch_index += gridDim.x) {
    const double* corr_lane = correlation + batch_index * surface_size;
    const double* energy_lane = energy + batch_index * surface_size;
    double local_best = -INFINITY;
    int64_t local_best_flat = std::numeric_limits<int64_t>::max();
    for (int64_t flat = threadIdx.x; flat < surface_size;
         flat += blockDim.x) {
      const double value = normalized_value(corr_lane[flat], energy_lane[flat]);
      if (peak_is_better(value, flat, local_best, local_best_flat)) {
        local_best = value;
        local_best_flat = flat;
      }
    }
    shared_values[threadIdx.x] = local_best;
    shared_indices[threadIdx.x] = local_best_flat;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset &&
          peak_is_better(shared_values[threadIdx.x + offset],
                         shared_indices[threadIdx.x + offset],
                         shared_values[threadIdx.x],
                         shared_indices[threadIdx.x])) {
        shared_values[threadIdx.x] = shared_values[threadIdx.x + offset];
        shared_indices[threadIdx.x] = shared_indices[threadIdx.x + offset];
      }
      __syncthreads();
    }
    const double best_value = shared_values[0];
    const int64_t best_flat = shared_indices[0];
    const int64_t peak_az = best_flat / width;
    const int64_t peak_rg = best_flat % width;
    double local_side_sum = 0.0;
    int64_t local_side_count = 0;
    for (int64_t flat = threadIdx.x; flat < surface_size;
         flat += blockDim.x) {
      const int64_t row = flat / width;
      const int64_t column = flat % width;
      if (llabs(row - peak_az) <= 1 && llabs(column - peak_rg) <= 1) {
        continue;
      }
      const double value = normalized_value(corr_lane[flat], energy_lane[flat]);
      if (!isnan(value)) {
        local_side_sum += fabs(value);
        ++local_side_count;
      }
    }
    shared_values[threadIdx.x] = local_side_sum;
    shared_counts[threadIdx.x] = local_side_count;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        shared_values[threadIdx.x] += shared_values[threadIdx.x + offset];
        shared_counts[threadIdx.x] += shared_counts[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const double side_mean =
          shared_counts[0] > 0
              ? shared_values[0] / static_cast<double>(shared_counts[0])
              : NAN;
      const double lane_snr = isfinite(side_mean) && side_mean > 0.0
                                  ? best_value / side_mean
                                  : NAN;
      double lane_az = static_cast<double>(peak_az - search_az);
      double lane_rg = static_cast<double>(peak_rg - search_rg);
      if (subpixel && peak_az > 0 && peak_az < height - 1 && peak_rg > 0 &&
          peak_rg < width - 1) {
        const int64_t center = peak_az * width + peak_rg;
        const double az_left = normalized_value(
            corr_lane[center - width], energy_lane[center - width]);
        const double az_center =
            normalized_value(corr_lane[center], energy_lane[center]);
        const double az_right = normalized_value(
            corr_lane[center + width], energy_lane[center + width]);
        const double rg_left = normalized_value(
            corr_lane[center - 1], energy_lane[center - 1]);
        const double rg_right = normalized_value(
            corr_lane[center + 1], energy_lane[center + 1]);
        const double az_denom = az_left - 2.0 * az_center + az_right;
        const double rg_denom = rg_left - 2.0 * az_center + rg_right;
        const double az_sub = fabs(az_denom) < 1e-12
                                  ? 0.0
                                  : 0.5 * (az_left - az_right) / az_denom;
        const double rg_sub = fabs(rg_denom) < 1e-12
                                  ? 0.0
                                  : 0.5 * (rg_left - rg_right) / rg_denom;
        lane_az += az_sub;
        lane_rg += rg_sub;
      }
      const bool surface_edge = peak_az == 0 || peak_az == height - 1 ||
                                peak_rg == 0 || peak_rg == width - 1;
      const bool lane_valid = isfinite(lane_snr) && isfinite(lane_rg) &&
                              isfinite(lane_az) && lane_snr >= snr_threshold &&
                              fabs(lane_rg) <= max_abs_residual &&
                              fabs(lane_az) <= max_abs_residual;
      d_rg[batch_index] = lane_rg;
      d_az[batch_index] = lane_az;
      snr[batch_index] = lane_snr;
      boundary[batch_index] = surface_edge;
      valid[batch_index] = lane_valid;
    }
    __syncthreads();
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
ampcor_ncc_postprocess_cuda(const torch::Tensor& correlation,
                            const torch::Tensor& energy, int64_t search_az,
                            int64_t search_rg, bool subpixel,
                            double snr_threshold, double max_abs_residual) {
  TORCH_CHECK(correlation.is_cuda() && energy.is_cuda(),
              "NCC postprocess inputs must be CUDA tensors");
  TORCH_CHECK(correlation.get_device() == c10::cuda::current_device() &&
                  energy.get_device() == c10::cuda::current_device(),
              "NCC postprocess inputs must use the current CUDA device");
  TORCH_CHECK(correlation.scalar_type() == torch::kFloat64 &&
                  energy.scalar_type() == torch::kFloat64,
              "NCC postprocess inputs must be float64");
  TORCH_CHECK(correlation.dim() == 3 && energy.dim() == 3,
              "NCC postprocess inputs must be rank-3");
  TORCH_CHECK(correlation.is_contiguous() && energy.is_contiguous(),
              "NCC postprocess inputs must be contiguous");
  TORCH_CHECK(correlation.sizes() == energy.sizes(),
              "NCC postprocess inputs must have matching shapes");
  const int64_t expected_height =
      checked_twice_plus_one(search_az, "search_az");
  const int64_t expected_width =
      checked_twice_plus_one(search_rg, "search_rg");
  TORCH_CHECK(std::isfinite(snr_threshold) &&
                  std::isfinite(max_abs_residual) && max_abs_residual >= 0.0,
              "NCC cull thresholds must be finite");

  const int64_t batch = correlation.size(0);
  const int64_t height = correlation.size(1);
  const int64_t width = correlation.size(2);
  TORCH_CHECK(batch > 0 && height > 0 && width > 0,
              "NCC postprocess dimensions must be positive");
  TORCH_CHECK(height == expected_height && width == expected_width,
              "NCC surface shape does not match search half-widths");
  TORCH_CHECK(!subpixel || (height >= 3 && width >= 3),
              "subpixel NCC postprocess requires a 3x3 or larger surface");
  const int64_t surface_size = checked_product(height, width, "NCC surface");
  checked_product(batch, surface_size, "NCC batch");

  auto options = correlation.options();
  auto d_rg = torch::empty({batch}, options);
  auto d_az = torch::empty({batch}, options);
  auto snr = torch::empty({batch}, options);
  auto boundary = torch::empty({batch}, options.dtype(torch::kBool));
  auto valid = torch::empty({batch}, options.dtype(torch::kBool));
  c10::cuda::CUDAGuard guard(correlation.device());
  constexpr int threads = 256;
  const int64_t requested_blocks = batch;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const int64_t max_blocks = properties->maxGridSize[0];
  const int blocks = static_cast<int>(requested_blocks > max_blocks
                                          ? max_blocks
                                          : requested_blocks);
  TORCH_CHECK(blocks > 0, "NCC postprocess launch grid must be positive");
  auto stream = at::cuda::getCurrentCUDAStream(correlation.get_device());
  ampcor_ncc_postprocess_kernel<<<blocks, threads, 0, stream.stream()>>>(
      correlation.data_ptr<double>(), energy.data_ptr<double>(), batch, height,
      width, search_az, search_rg, subpixel, snr_threshold, max_abs_residual,
      d_rg.data_ptr<double>(), d_az.data_ptr<double>(), snr.data_ptr<double>(),
      boundary.data_ptr<bool>(), valid.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_rg, d_az, snr, boundary, valid};
}
