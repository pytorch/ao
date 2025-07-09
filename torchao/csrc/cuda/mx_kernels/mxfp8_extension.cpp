// PyBind wrapping for the mxfp8 extension
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <string>
#include <torch/extension.h>

namespace mxfp8 {

// Forward declarations
void mxfp8_quantize_cuda(const torch::Tensor &input,
                         torch::Tensor &output_rowwise,
                         torch::Tensor &output_columnwise,
                         torch::Tensor &scales_rowwise,
                         torch::Tensor &scales_colwise, 
                         int64_t scale_dim_x,
                         int64_t scale_dim_y, 
                         const std::string &fp8_format,
                         const std::string &scaling_mode);

// Helper for tensor validation
void check_cuda_tensor(const torch::Tensor &t, const char *name) {
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp8_quantize(torch::Tensor input, bool rowwise, bool colwise,
               int64_t scale_dim_x, int64_t scale_dim_y,
               const std::string &fp8_format,
               const std::string &scaling_mode) {

  // Validate inputs
  TORCH_CHECK(!rowwise, "rowwise scaling is not supported yet");
  check_cuda_tensor(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 ||
                  input.scalar_type() == torch::kFloat16 ||
                  input.scalar_type() == torch::kBFloat16,
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
  const auto options_fp8 = torch::TensorOptions()
                               .dtype(torch::kFloat8_e4m3fn) // FP8 stored as uint8
                               .device(input.device());

  const auto options_scale = torch::TensorOptions()
                                 .dtype(torch::kFloat8_e8m0fnu) // E8M0 stored as uint8
                                 .device(input.device());

  // Allocate output tensors
  torch::Tensor output_rowwise, output_colwise;
  torch::Tensor scales_rowwise, scales_colwise;

  if (rowwise) {
    const int64_t num_col_blocks = (cols + scale_dim_x - 1) / scale_dim_x;
    output_rowwise = torch::empty({rows, cols}, options_fp8);
    scales_rowwise = torch::empty({rows, num_col_blocks}, options_scale);
  } else {
    output_rowwise = torch::empty({0}, options_fp8);
    scales_rowwise = torch::empty({0}, options_scale);
  }

  if (colwise) {
    const int64_t num_row_blocks = (rows + scale_dim_y - 1) / scale_dim_y;
    output_colwise = torch::empty_strided({rows, cols}, {1, rows}, options_fp8);
    // Need scales_colwise to be this shape so the 'col' dim stride is 1, 
    // for colwise scaling, we can avoid uncoalesced writes to global memory.
    // This is because each of the 32 threads in a warp will be computing
    // a scale for a different column of 32 input data values, then each writing
    // that scale to global memory - so the stride along this `col` dim should be 1
    // so writes can be coalesced into a single transaction.
    scales_colwise = torch::empty_strided({cols, num_row_blocks}, {1, cols}, options_scale);
  } else {
    output_colwise = torch::empty({0}, options_fp8);
    scales_colwise = torch::empty({0}, options_scale);
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

} // namespace mxfp8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "MXFP8 Quantization PyTorch Extension";

  m.def("quantize", &mxfp8::mxfp8_quantize, "MXFP8 quantization",
        py::arg("input"), py::arg("rowwise") = true, py::arg("colwise") = false,
        py::arg("scale_dim_x") = 32, py::arg("scale_dim_y") = 32,
        py::arg("fp8_format") = "e4m3",
        py::arg("scaling_mode") = "floor");
}
