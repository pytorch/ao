// MXFP8 extension using TORCH_LIBRARY (CPython ABI agnostic)
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <string>

namespace mxfp8 {

// Forward declarations
void mxfp8_quantize_cuda(const at::Tensor &input,
                         at::Tensor &output_rowwise,
                         at::Tensor &output_columnwise,
                         at::Tensor &scales_rowwise,
                         at::Tensor &scales_colwise, 
                         int64_t scale_dim_x,
                         int64_t scale_dim_y, 
                         const std::string &fp8_format,
                         const std::string &scaling_mode);

void mxfp8_quantize_3d_cuda(const at::Tensor &input,
                             at::Tensor &output_colwise,
                             at::Tensor &scales_colwise,
                             int64_t scale_dim_n,
                             const std::string &fp8_format,
                             const std::string &scaling_mode);

// Helper for tensor validation
void check_cuda_tensor(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Helper to validate FP8 format
void validate_fp8_format(const std::string &fp8_format) {
  TORCH_CHECK(fp8_format.compare("e4m3") == 0,
              "fp8_format must be 'e4m3', got: ", fp8_format);
}

// Helper to validate scale dimensions
void validate_scale_dimensions(int64_t scale_dim_x, int64_t scale_dim_y) {
  TORCH_CHECK(scale_dim_x == 1 || scale_dim_x == 32,
              "scale_dim_x must be 1 or 32, got: ", scale_dim_x);
  TORCH_CHECK(scale_dim_y == 1 || scale_dim_y == 32,
              "scale_dim_y must be 1 or 32, got: ", scale_dim_y);
}

// Main quantization function
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mxfp8_quantize(const at::Tensor& input, bool rowwise, bool colwise,
               int64_t scale_dim_x, int64_t scale_dim_y,
               const std::string &fp8_format,
               const std::string &scaling_mode) {

  // Validate inputs
  TORCH_CHECK(!rowwise, "rowwise scaling is not supported yet");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Input must be float32, float16, or bfloat16");
  TORCH_CHECK(rowwise || colwise,
              "At least one of rowwise or colwise must be true");

  validate_scale_dimensions(scale_dim_x, scale_dim_y);
  validate_fp8_format(fp8_format);

  const int64_t rows = input.size(0);
  const int64_t cols = input.size(1);
  TORCH_CHECK((rows >= 32) && (rows % 32 == 0), "rows must be a multiple of 32");
  TORCH_CHECK((cols >= 32) && (cols % 32 == 0), "cols must be a multiple of 32");

  c10::cuda::CUDAGuard device_guard(input.device());

  // Create tensor options
  const auto options_fp8 = at::TensorOptions()
                               .dtype(at::kFloat8_e4m3fn)
                               .device(input.device());

  const auto options_scale = at::TensorOptions()
                                 .dtype(at::kFloat8_e8m0fnu)
                                 .device(input.device());

  // Allocate output tensors
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
    // Need scales_colwise to be this shape so the 'col' dim stride is 1, 
    // for colwise scaling, we can avoid uncoalesced writes to global memory.
    // This is because each of the 32 threads in a warp will be computing
    // a scale for a different column of 32 input data values, then each writing
    // that scale to global memory - so the stride along this `col` dim should be 1
    // so writes can be coalesced into a single transaction.
    scales_colwise = at::empty_strided({cols, num_row_blocks}, {1, cols}, options_scale);
  } else {
    output_colwise = at::empty({0}, options_fp8);
    scales_colwise = at::empty({0}, options_scale);
  }

  // Call CUDA kernels
  mxfp8_quantize_cuda(input, 
                      output_rowwise, output_colwise, 
                      scales_rowwise, scales_colwise,
                      rowwise ? scale_dim_x : 1, // scale_dim_x
                      colwise ? scale_dim_y : 1, // scale_dim_y
                      fp8_format, scaling_mode);

  return std::make_tuple(output_rowwise, output_colwise, scales_rowwise,
                         scales_colwise);
}

// 3D tensor quantization function
std::tuple<at::Tensor, at::Tensor>
mxfp8_quantize_3d(const at::Tensor& input, int64_t scale_dim_n,
                  const std::string &fp8_format,
                  const std::string &scaling_mode) {

  // Validate inputs
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  // Note: We don't check contiguous for 3D as it may have column major strides
  TORCH_CHECK(input.dim() == 3, "input must be 3D");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Input must be float32, float16, or bfloat16");
  TORCH_CHECK(scale_dim_n == 32, "scale_dim_n must be 32 for now");

  validate_fp8_format(fp8_format);

  const int64_t E = input.size(0);
  const int64_t N = input.size(1);
  const int64_t K = input.size(2);

  // Check dimensions are valid for 3D kernel
  TORCH_CHECK((N >= 32) && (N % 32 == 0), "N must be a multiple of 32");
  TORCH_CHECK((K >= 32) && (K % 32 == 0), "K must be a multiple of 32");


  c10::cuda::CUDAGuard device_guard(input.device());

  // Create tensor options
  const auto options_fp8 = at::TensorOptions()
                               .dtype(at::kFloat8_e4m3fn)
                               .device(input.device());

  const auto options_scale = at::TensorOptions()
                                 .dtype(at::kFloat8_e8m0fnu)
                                 .device(input.device());

  // Create output tensor with column major layout (required for downstream ops)
  at::Tensor output_colwise = at::empty_strided(
      {E, N, K}, {N * K, 1, N}, options_fp8);

  // Create scales tensor with shape (E, num_n_blocks, K)
  const int64_t num_n_blocks = (N + scale_dim_n - 1) / scale_dim_n;
  at::Tensor scales_colwise = at::empty({E, num_n_blocks, K}, options_scale);

  // Call CUDA kernel
  mxfp8_quantize_3d_cuda(input, output_colwise, scales_colwise,
                         scale_dim_n, fp8_format, scaling_mode);

  return std::make_tuple(output_colwise, scales_colwise);
}

} // namespace mxfp8


// Register CUDA implementations
TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("mxfp8_quantize", &mxfp8::mxfp8_quantize);
  m.impl("mxfp8_quantize_3d", &mxfp8::mxfp8_quantize_3d);
}
