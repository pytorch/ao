// MXFP8 extension using STABLE_TORCH_LIBRARY (ABI stable)
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

using torch::stable::Tensor;
namespace tsa = torch::stable::accelerator;

namespace mxfp8 {

// Forward declarations
void mxfp8_quantize_cuda(const Tensor &input,
                         Tensor &output_rowwise,
                         Tensor &output_columnwise,
                         Tensor &scales_rowwise,
                         Tensor &scales_colwise,
                         int64_t scale_dim_x,
                         int64_t scale_dim_y,
                         const std::string &fp8_format,
                         const std::string &scaling_mode);

void mxfp8_quantize_3d_cuda(const Tensor &input,
                             Tensor &output_colwise,
                             Tensor &scales_colwise,
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

void launch_mx_block_rearrange_2d_simple_cuda(
    const uint8_t* scales_ptr,
    int scale_rows,
    int scale_cols,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    int chunk_width,
    cudaStream_t stream);

// Helper for tensor validation
void check_cuda_tensor(const Tensor &t, const char *name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Helper to validate FP8 format
void validate_fp8_format(const std::string &fp8_format) {
  STD_TORCH_CHECK(fp8_format.compare("e4m3") == 0,
              "fp8_format must be 'e4m3', got: ", fp8_format);
}

// Helper to validate scale dimensions
void validate_scale_dimensions(int64_t scale_dim_x, int64_t scale_dim_y) {
  STD_TORCH_CHECK(scale_dim_x == 1 || scale_dim_x == 32,
              "scale_dim_x must be 1 or 32, got: ", scale_dim_x);
  STD_TORCH_CHECK(scale_dim_y == 1 || scale_dim_y == 32,
              "scale_dim_y must be 1 or 32, got: ", scale_dim_y);
}

// Main quantization function
std::tuple<Tensor, Tensor, Tensor, Tensor>
mxfp8_quantize(const Tensor& input, bool rowwise, bool colwise,
               int64_t scale_dim_x, int64_t scale_dim_y,
               const std::string &fp8_format,
               const std::string &scaling_mode) {

  // Validate inputs
  STD_TORCH_CHECK(!rowwise, "rowwise scaling is not supported yet");
  STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  STD_TORCH_CHECK(input.dim() == 2, "input must be 2D");
  STD_TORCH_CHECK(input.scalar_type() == torch::headeronly::ScalarType::Float ||
                  input.scalar_type() == torch::headeronly::ScalarType::Half ||
                  input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              "Input must be float32, float16, or bfloat16");
  STD_TORCH_CHECK(rowwise || colwise,
              "At least one of rowwise or colwise must be true");

  validate_scale_dimensions(scale_dim_x, scale_dim_y);
  validate_fp8_format(fp8_format);

  const int64_t rows = input.size(0);
  const int64_t cols = input.size(1);
  STD_TORCH_CHECK((rows >= 32) && (rows % 32 == 0), "rows must be a multiple of 32");
  STD_TORCH_CHECK((cols >= 32) && (cols % 32 == 0), "cols must be a multiple of 32");

  tsa::DeviceGuard device_guard(input.get_device_index());

  // Allocate output tensors
  Tensor output_rowwise, output_colwise;
  Tensor scales_rowwise, scales_colwise;

  if (rowwise) {
    const int64_t num_col_blocks = (cols + scale_dim_x - 1) / scale_dim_x;
    output_rowwise = torch::stable::new_empty(input, {rows, cols}, torch::headeronly::ScalarType::Float8_e4m3fn);
    scales_rowwise = torch::stable::new_empty(input, {rows, num_col_blocks}, torch::headeronly::ScalarType::Float8_e8m0fnu);
  } else {
    output_rowwise = torch::stable::new_empty(input, {0}, torch::headeronly::ScalarType::Float8_e4m3fn);
    scales_rowwise = torch::stable::new_empty(input, {0}, torch::headeronly::ScalarType::Float8_e8m0fnu);
  }

  if (colwise) {
    const int64_t num_row_blocks = (rows + scale_dim_y - 1) / scale_dim_y;
    // Create column-major tensor by creating transposed shape and transposing
    // We need shape {rows, cols} with strides {1, rows}, so create {cols, rows} and transpose
    Tensor output_colwise_tmp = torch::stable::new_empty(input, {cols, rows}, torch::headeronly::ScalarType::Float8_e4m3fn);
    output_colwise = torch::stable::transpose(output_colwise_tmp, 0, 1);
    // For scales_colwise: need shape {cols, num_row_blocks} with strides {1, cols}
    // Create {num_row_blocks, cols} and transpose
    Tensor scales_colwise_tmp = torch::stable::new_empty(input, {num_row_blocks, cols}, torch::headeronly::ScalarType::Float8_e8m0fnu);
    scales_colwise = torch::stable::transpose(scales_colwise_tmp, 0, 1);
  } else {
    output_colwise = torch::stable::new_empty(input, {0}, torch::headeronly::ScalarType::Float8_e4m3fn);
    scales_colwise = torch::stable::new_empty(input, {0}, torch::headeronly::ScalarType::Float8_e8m0fnu);
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
std::tuple<Tensor, Tensor>
mxfp8_quantize_3d(const Tensor& input, int64_t scale_dim_n,
                  const std::string &fp8_format,
                  const std::string &scaling_mode) {

  // Validate inputs
  STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  // Note: We don't check contiguous for 3D as it may have column major strides
  STD_TORCH_CHECK(input.dim() == 3, "input must be 3D");
  STD_TORCH_CHECK(input.scalar_type() == torch::headeronly::ScalarType::Float ||
                  input.scalar_type() == torch::headeronly::ScalarType::Half ||
                  input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              "Input must be float32, float16, or bfloat16");
  STD_TORCH_CHECK(scale_dim_n == 32, "scale_dim_n must be 32 for now");

  validate_fp8_format(fp8_format);

  const int64_t E = input.size(0);
  const int64_t N = input.size(1);
  const int64_t K = input.size(2);

  // Check dimensions are valid for 3D kernel
  STD_TORCH_CHECK((N >= 32) && (N % 32 == 0), "N must be a multiple of 32");
  STD_TORCH_CHECK((K >= 32) && (K % 32 == 0), "K must be a multiple of 32");


  tsa::DeviceGuard device_guard(input.get_device_index());

  // Create output tensor with column major layout (required for downstream ops)
  // We need shape {E, N, K} with strides {N * K, 1, N}
  // Create {E, K, N} contiguous tensor (strides {K*N, N, 1}) then transpose dims 1 and 2
  Tensor output_colwise_tmp = torch::stable::new_empty(input, {E, K, N}, torch::headeronly::ScalarType::Float8_e4m3fn);
  Tensor output_colwise = torch::stable::transpose(output_colwise_tmp, 1, 2);

  // Create scales tensor with shape (E, num_n_blocks, K)
  const int64_t num_n_blocks = (N + scale_dim_n - 1) / scale_dim_n;
  Tensor scales_colwise = torch::stable::new_empty(input, {E, num_n_blocks, K}, torch::headeronly::ScalarType::Float8_e8m0fnu);

  // Call CUDA kernel
  mxfp8_quantize_3d_cuda(input, output_colwise, scales_colwise,
                         scale_dim_n, fp8_format, scaling_mode);

  return std::make_tuple(output_colwise, scales_colwise);
}

Tensor mx_block_rearrange_2d_M_groups(
    Tensor scales_tensor,
    Tensor input_group_end_offsets,
    int64_t chunks_per_tb) {

  // Validate inputs
  check_cuda_tensor(scales_tensor, "scales_tensor");
  check_cuda_tensor(input_group_end_offsets, "input_group_end_offsets");

  STD_TORCH_CHECK(scales_tensor.dim() == 2, "scales_tensor must be 2D");
  STD_TORCH_CHECK(scales_tensor.is_contiguous(), "scales_tensor must be contiguous (row-major)");
  STD_TORCH_CHECK(scales_tensor.scalar_type() == torch::headeronly::ScalarType::Byte || // uint8
              scales_tensor.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu,
              "scales_tensor must be uint8 or e8m0");
  STD_TORCH_CHECK(input_group_end_offsets.scalar_type() == torch::headeronly::ScalarType::Int,
              "input_group_end_offsets must be int32");
  STD_TORCH_CHECK(input_group_end_offsets.dim() == 1,
              "input_group_end_offsets must be 1D");
  STD_TORCH_CHECK(chunks_per_tb == 1 || chunks_per_tb == 4 || chunks_per_tb == 8 || chunks_per_tb == 16,
              "chunks_per_tb must be 1, 4, 8, or 16, got: ", chunks_per_tb);
  tsa::DeviceGuard device_guard(scales_tensor.get_device_index());

  const int rows = scales_tensor.size(0);
  const int cols = scales_tensor.size(1);
  const int num_groups = input_group_end_offsets.size(0);
  STD_TORCH_CHECK(num_groups <= 32, "num_groups must be <= 32");

  // Automatically select chunk_width based on scale_cols
  int chunk_width;
  if (cols >= 64) {
    chunk_width = 64;
  } else if (cols >= 32) {
    chunk_width = 32;
  } else {
    chunk_width = 16;
  }

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
  auto output = torch::stable::new_zeros(scales_tensor, {padded_rows, padded_cols}, scales_tensor.scalar_type());

  // Get raw pointers
  const uint8_t* scales_ptr = reinterpret_cast<const uint8_t*>(scales_tensor.data_ptr());
  const int32_t* offsets_ptr = input_group_end_offsets.const_data_ptr<int32_t>();
  uint8_t* output_ptr = reinterpret_cast<uint8_t*>(output.data_ptr());

  // Get CUDA stream using stable ABI
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(scales_tensor.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  // pipelined kernel will be used if input meets 2d TMA constraint (cols >= 16 and cols % 16 bytes == 0)
  // Otherwise, a fallback kernel will be used (slightly slower but supports any column count)  
  const bool can_use_pipelined_kernel = cols >= 16 && cols % 16 == 0;
  if (can_use_pipelined_kernel)
  {
    // Launch pipelined TMA kernel
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
        stream);
  }
  else 
  {
    // Launch simplified kernel (no TMA, works with any column dimension)
    launch_mx_block_rearrange_2d_simple_cuda(
        scales_ptr,
        rows,
        cols,
        offsets_ptr,
        output_ptr,
        num_groups,
        static_cast<int>(chunk_width),
        stream);
  }
  return output;
}


} // namespace mxfp8


// Register CUDA implementations using stable ABI
STABLE_TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("mxfp8_quantize", TORCH_BOX(&mxfp8::mxfp8_quantize));
  m.impl("mxfp8_quantize_3d", TORCH_BOX(&mxfp8::mxfp8_quantize_3d));
  m.impl("mx_block_rearrange_2d_M_groups", TORCH_BOX(&mxfp8::mx_block_rearrange_2d_M_groups));
}
