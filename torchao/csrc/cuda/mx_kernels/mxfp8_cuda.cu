// CUDA bridge for MXFP8 quantization

#include "mxfp8_quantize.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <string>
#include <torch/extension.h>


namespace mxfp8 {

// Convert PyTorch scalar type to our DType enum
DType get_input_dtype(const torch::Tensor &t) {
  switch (t.scalar_type()) {
  case torch::kFloat32:
    return DType::kFloat32;
  case torch::kFloat16:
    return DType::kFloat16;
  case torch::kBFloat16:
    return DType::kBFloat16;
  case torch::kUInt8:
    return DType::kByte;
  default:
    TORCH_CHECK(false, "Unsupported input tensor dtype: ", t.scalar_type());
  }
}

ScaleCalculationMode get_scaling_mode(const std::string &scaling_mode) {
  if (scaling_mode.compare("floor") == 0) {
      return ScaleCalculationMode::FLOOR;
  } else if (scaling_mode.compare("rceil") == 0) {
      return ScaleCalculationMode::RCEIL;
  } else {
      TORCH_CHECK(false, "Unsupported scaling mode: ", scaling_mode, ". Only ['floor', 'rceil'] are supported.");
  }
}

// Convert FP8 format string to DType enum
DType get_output_dtype(const std::string &fp8_format) {
  if (fp8_format.compare("e4m3") == 0) {
    return DType::kFloat8E4M3;
  } else {
    TORCH_CHECK(false, "Unsupported FP8 format: ", fp8_format,
                ". Only 'e4m3' is supported.");
  }
}

void mxfp8_quantize_cuda(const torch::Tensor &input,
                         torch::Tensor &output_rowwise,
                         torch::Tensor &output_colwise,
                         torch::Tensor &scales_rowwise,
                         torch::Tensor &scales_colwise,
                         int64_t scale_dim_x,
                         int64_t scale_dim_y,
                         const std::string &fp8_format,
                         const std::string &scaling_mode) {

  // Get tensor properties
  const int64_t rows = input.size(0);
  const int64_t cols = input.size(1);

  // Get data pointers
  const void *input_ptr = input.data_ptr();
  void *output_rowwise_ptr =
      output_rowwise.numel() > 0 ? output_rowwise.data_ptr() : nullptr;
  void *output_colwise_ptr =
      output_colwise.numel() > 0 ? output_colwise.data_ptr() : nullptr;
  e8m0_t *scales_rowwise_ptr =
      scales_rowwise.numel() > 0
          ? reinterpret_cast<e8m0_t *>(scales_rowwise.data_ptr())
          : nullptr;
  e8m0_t *scales_colwise_ptr =
      scales_colwise.numel() > 0
          ? reinterpret_cast<e8m0_t *>(scales_colwise.data_ptr())
          : nullptr;

  // Get CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Get strides of scale ptrs
  int64_t scale_rowwise_stride_dim0 = scales_rowwise.strides()[0];
  int64_t scale_rowwise_stride_dim1 = scales_rowwise.strides()[1];
  int64_t scale_colwise_stride_dim0 = scales_colwise.strides()[0];
  int64_t scale_colwise_stride_dim1 = scales_colwise.strides()[1];

#if defined(DEBUG)
  printf("mxfp8_quantize_cuda:\n");
  printf("Quantizing input tensor of size %ld x %ld\n", rows, cols);
  printf("scaling_mode: %s\n", scaling_mode.c_str());
  printf("Scale dim x: %ld\n", scale_dim_x);
  printf("Scale dim y: %ld\n", scale_dim_y);
  printf("Rowwise scale shape: %ld x %ld\n", scales_rowwise.sizes()[0], scales_rowwise.sizes()[1]);
  printf("Colwise scale shape: %ld x %ld\n", scales_colwise.sizes()[0], scales_colwise.sizes()[1]);
  printf("scale_rowwise_stride_dim0 = %ld\n", scale_rowwise_stride_dim0);
  printf("scale_rowwise_stride_dim1 = %ld\n", scale_rowwise_stride_dim1);
  printf("scale_colwise_stride_dim0 = %ld\n", scale_colwise_stride_dim0);
  printf("scale_colwise_stride_dim1 = %ld\n", scale_colwise_stride_dim1);
#endif

  // Call the quantization kernel
  MXFP8Quantizer::quantize(input_ptr,
                           output_rowwise_ptr, output_colwise_ptr,
                           scales_rowwise_ptr, scales_colwise_ptr,
                           scale_rowwise_stride_dim0, scale_rowwise_stride_dim1,
                           scale_colwise_stride_dim0, scale_colwise_stride_dim1,
                           rows, cols,
                           get_input_dtype(input), get_output_dtype(fp8_format),
                           scale_dim_x, scale_dim_y,
                           get_scaling_mode(scaling_mode),
                           stream);
}

} // namespace mxfp8
