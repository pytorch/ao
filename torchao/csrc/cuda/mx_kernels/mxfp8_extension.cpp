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

void launch_mx_block_rearrange_2d_M_groups_cuda(
    const uint8_t* scales_ptr,
    int scale_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    int chunk_width,      // Template selector: 64 or 128
    int chunks_per_tb,
    cudaStream_t stream);

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

at::Tensor mx_block_rearrange_2d_M_groups(
    at::Tensor scales_tensor,
    at::Tensor input_group_end_offsets,
    int64_t chunk_width,
    int64_t chunks_per_tb) {

  // Validate inputs
  check_cuda_tensor(scales_tensor, "scales_tensor");
  check_cuda_tensor(input_group_end_offsets, "input_group_end_offsets");

  TORCH_CHECK(scales_tensor.dim() == 2, "scales_tensor must be 2D");
  TORCH_CHECK(scales_tensor.is_contiguous(), "scales_tensor must be contiguous (row-major)");
  TORCH_CHECK(scales_tensor.scalar_type() == at::kByte || // uint8
              scales_tensor.scalar_type() == at::kFloat8_e8m0fnu,
              "scales_tensor must be uint8 or e8m0");
  TORCH_CHECK(input_group_end_offsets.scalar_type() == at::kInt,
              "input_group_end_offsets must be int32");
  TORCH_CHECK(input_group_end_offsets.dim() == 1,
              "input_group_end_offsets must be 1D");
  TORCH_CHECK(chunk_width == 64 || chunk_width == 128,
              "chunk_width must be 64 or 128, got: ", chunk_width);
  TORCH_CHECK(chunks_per_tb == 1 || chunks_per_tb == 4 || chunks_per_tb == 8 || chunks_per_tb == 16,
              "chunks_per_tb must be 4, 8, or 16, got: ", chunks_per_tb);
  c10::cuda::CUDAGuard device_guard(scales_tensor.device());

  const int rows = scales_tensor.size(0);
  const int cols = scales_tensor.size(1);
  const int num_groups = input_group_end_offsets.size(0);
  TORCH_CHECK(num_groups <= 32, "num_groups must be <= 32");

  // Calculate blocks needed - uses 128-row blocks
  // For M groups, groups are along rows, so we pad each group to 128 rows
  const int BLOCK_ROWS = 128;
  const int BLOCK_COLS = 4;

  // Each group is padded to 128 rows upper bound
  const int padded_rows = rows + num_groups * BLOCK_ROWS;

  // Columns are padded to multiple of BLOCK_COLS
  const int num_col_blocks = (cols + BLOCK_COLS - 1) / BLOCK_COLS;
  const int padded_cols = num_col_blocks * BLOCK_COLS;

  // Create output tensor
  auto output = at::zeros({padded_rows, padded_cols},
                            at::TensorOptions()
                                .dtype(scales_tensor.scalar_type())
                                .device(scales_tensor.device()));

  // Get raw pointers
  const uint8_t* scales_ptr = reinterpret_cast<const uint8_t*>(scales_tensor.data_ptr());
  const int32_t* offsets_ptr = input_group_end_offsets.data_ptr<int32_t>();
  uint8_t* output_ptr = reinterpret_cast<uint8_t*>(output.data_ptr());

  // Launch pipelined M groups kernel with specified chunk_width and chunks_per_tb
  launch_mx_block_rearrange_2d_M_groups_cuda(
      scales_ptr,
      scales_tensor.stride(0),
      rows,
      cols,
      padded_rows,
      offsets_ptr,
      output_ptr,
      num_groups,
      static_cast<int>(chunk_width),
      static_cast<int>(chunks_per_tb),
      at::cuda::getCurrentCUDAStream());

  return output;
}


} // namespace mxfp8


// Register CUDA implementations
TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("mxfp8_quantize", &mxfp8::mxfp8_quantize);
  m.impl("mxfp8_quantize_3d", &mxfp8::mxfp8_quantize_3d);
  m.impl("mx_block_rearrange_2d_M_groups", &mxfp8::mx_block_rearrange_2d_M_groups);
}
