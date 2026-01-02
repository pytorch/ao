// MXFP8 extension for ROCm
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <string>

namespace mxfp8 {

void mxfp8_quantize_rocm(const at::Tensor &input,
                         at::Tensor &output,
                         at::Tensor &scales,
                         bool colwise,
                         int64_t block_size,
                         const std::string &scaling_mode);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mxfp8_quantize(const at::Tensor& input, bool rowwise, bool colwise,
               int64_t scale_dim_x, int64_t scale_dim_y,
               const std::string &fp8_format,
               const std::string &scaling_mode) {

  TORCH_CHECK(!rowwise, "rowwise scaling is not supported yet");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
              input.scalar_type() == at::kHalf ||
              input.scalar_type() == at::kBFloat16,
              "Input must be float32, float16, or bfloat16");
  TORCH_CHECK(rowwise || colwise, "At least one of rowwise or colwise must be true");
  TORCH_CHECK(scale_dim_x == 1 || scale_dim_x == 32, "scale_dim_x must be 1 or 32");
  TORCH_CHECK(scale_dim_y == 1 || scale_dim_y == 32, "scale_dim_y must be 1 or 32");
  TORCH_CHECK(fp8_format == "e4m3", "fp8_format must be 'e4m3'");

  const int64_t rows = input.size(0);
  const int64_t cols = input.size(1);
  TORCH_CHECK((rows >= 32) && (rows % 32 == 0), "rows must be a multiple of 32");
  TORCH_CHECK((cols >= 32) && (cols % 32 == 0), "cols must be a multiple of 32");

  c10::cuda::CUDAGuard device_guard(input.device());

  const auto options_fp8 = at::TensorOptions()
                               .dtype(at::kFloat8_e4m3fn)
                               .device(input.device());
  const auto options_scale = at::TensorOptions()
                                 .dtype(at::kFloat8_e8m0fnu)
                                 .device(input.device());

  at::Tensor output_rowwise, output_colwise;
  at::Tensor scales_rowwise, scales_colwise;

  if (rowwise) {
    const int64_t num_col_blocks = (cols + scale_dim_x - 1) / scale_dim_x;
    output_rowwise = at::empty({rows, cols}, options_fp8);
    scales_rowwise = at::empty({rows, num_col_blocks}, options_scale);
  } else {
    output_rowwise = at::empty({0}, options_fp8);
    scales_rowwise = at::empty({0}, options_scale);
  }

  if (colwise) {
    const int64_t num_row_blocks = (rows + scale_dim_y - 1) / scale_dim_y;
    output_colwise = at::empty_strided({rows, cols}, {1, rows}, options_fp8);
    scales_colwise = at::empty_strided({cols, num_row_blocks}, {1, cols}, options_scale);
  } else {
    output_colwise = at::empty({0}, options_fp8);
    scales_colwise = at::empty({0}, options_scale);
  }

  if (rowwise) {
    TORCH_CHECK(false, "rowwise scaling is not yet implemented for ROCm");
  }
  if (colwise) {
    mxfp8_quantize_rocm(input, output_colwise, scales_colwise,
                        true, scale_dim_y, scaling_mode);
  }

  return std::make_tuple(output_rowwise, output_colwise, scales_rowwise, scales_colwise);
}

} // namespace mxfp8

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("mxfp8_quantize", &mxfp8::mxfp8_quantize);
}
