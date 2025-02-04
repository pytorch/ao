#include <cutlass/cutlass.h>
#include <torch/library.h>

#include "rowwise_scaled_linear_sparse_cutlass.cuh"

namespace torchao {

at::Tensor
rowwise_scaled_linear_sparse_cutlass_f8f8(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_meta, const at::Tensor& W_scale,
    const at::Tensor& bias) {
  // Validate input datatypes.
  TORCH_CHECK(
      (Xq.dtype() == at::kFloat8_e5m2 && Wq.dtype() == at::kFloat8_e4m3fn) ||
      (Xq.dtype() == at::kFloat8_e4m3fn && Wq.dtype() == at::kFloat8_e4m3fn),
      __func__, " : The input datatypes combination ", Xq.dtype(),
      " for Xq and ", Wq.dtype(), " for Wq is not supported");

  // Dispatch to appropriate kernel template.
  if (Xq.dtype() == at::kFloat8_e5m2 && Wq.dtype() == at::kFloat8_e4m3fn) {
    using DtypeXq = cutlass::float_e5m2_t;
    using DtypeWq = cutlass::float_e4m3_t;
    return rowwise_scaled_linear_sparse_cutlass<DtypeXq, DtypeWq>(
        Xq, X_scale, Wq, W_meta, W_scale, bias);
  } else if (Xq.dtype() == at::kFloat8_e4m3fn &&
             Wq.dtype() == at::kFloat8_e4m3fn) {
    using DtypeXq = cutlass::float_e4m3_t;
    using DtypeWq = cutlass::float_e4m3_t;
    return rowwise_scaled_linear_sparse_cutlass<DtypeXq, DtypeWq>(
        Xq, X_scale, Wq, W_meta, W_scale, bias);
  }

  return at::Tensor{};
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::rowwise_scaled_linear_sparse_cutlass_f8f8",
         &rowwise_scaled_linear_sparse_cutlass_f8f8);
}

}  // namespace torchao
