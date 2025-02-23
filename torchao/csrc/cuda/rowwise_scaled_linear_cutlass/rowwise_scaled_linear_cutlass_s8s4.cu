#include <torch/library.h>
#include "rowwise_scaled_linear_cutlass.cuh"

namespace torchao {

at::Tensor
rowwise_scaled_linear_cutlass_s8s4(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_scale, const at::Tensor& bias) {
  // Validate input datatypes.
  TORCH_CHECK(xq.dtype() == at::kChar && wq.dtype() == at::kChar,
              __func__, " : The input datatypes combination ", xq.dtype(),
              " for xq and ", wq.dtype(), " for wq is not supported");

#if defined(BUILD_ROWWISE_SCALED_LINEAR_CUTLASS)
  // Define ElementA as int8_t since it's a standard type
  using ElementA = int8_t;
  // ElementB comes from cutlass header
  return rowwise_scaled_linear_cutlass<ElementA, cutlass::int4b_t>(
      xq, x_scale, wq, w_scale, bias);
#else
  TORCH_CHECK(false, "CUTLASS kernels not built - rowwise_scaled_linear_cutlass_s8s4 not available");
  return at::Tensor{};
#endif
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::rowwise_scaled_linear_cutlass_s8s4",
         &rowwise_scaled_linear_cutlass_s8s4);
}

}  // namespace torchao
