#include <torch/library.h>

#include "rowwise_scaled_linear_sparse_cutlass_e4m3e4m3.h"
#include "rowwise_scaled_linear_sparse_cutlass_e4m3e5m2.h"
#include "rowwise_scaled_linear_sparse_cutlass_e5m2e4m3.h"
#include "rowwise_scaled_linear_sparse_cutlass_e5m2e5m2.h"

namespace torchao {

at::Tensor
rowwise_scaled_linear_sparse_cutlass_f8f8(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_meta, const at::Tensor& W_scale,
    const at::Tensor& bias) {
  // Validate input datatypes.
  TORCH_CHECK(
      (Xq.dtype() == at::kFloat8_e4m3fn && Wq.dtype() == at::kFloat8_e4m3fn) ||
      (Xq.dtype() == at::kFloat8_e4m3fn && Wq.dtype() == at::kFloat8_e5m2) ||
      (Xq.dtype() == at::kFloat8_e5m2 && Wq.dtype() == at::kFloat8_e4m3fn) ||
      (Xq.dtype() == at::kFloat8_e5m2 && Wq.dtype() == at::kFloat8_e5m2),
      __func__, " : The input datatypes combination ", Xq.dtype(),
      " for Xq and ", Wq.dtype(), " for Wq is not supported");

  // Dispatch to appropriate kernel template.
  if (Xq.dtype() == at::kFloat8_e4m3fn && Wq.dtype() == at::kFloat8_e4m3fn) {
    return rowwise_scaled_linear_sparse_cutlass_e4m3e4m3(
      Xq, X_scale, Wq, W_meta, W_scale, bias);
  } else if (Xq.dtype() == at::kFloat8_e4m3fn &&
             Wq.dtype() == at::kFloat8_e5m2) {
    return rowwise_scaled_linear_sparse_cutlass_e4m3e5m2(
      Xq, X_scale, Wq, W_meta, W_scale, bias);
  } else if (Xq.dtype() == at::kFloat8_e5m2 &&
             Wq.dtype() == at::kFloat8_e4m3fn) {
    return rowwise_scaled_linear_sparse_cutlass_e5m2e4m3(
      Xq, X_scale, Wq, W_meta, W_scale, bias);
  } else if (Xq.dtype() == at::kFloat8_e5m2 && Wq.dtype() == at::kFloat8_e5m2) {
    return rowwise_scaled_linear_sparse_cutlass_e5m2e5m2(
      Xq, X_scale, Wq, W_meta, W_scale, bias);
  }
  return at::Tensor{};
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::rowwise_scaled_linear_sparse_cutlass_f8f8",
         &rowwise_scaled_linear_sparse_cutlass_f8f8);
}

}  // namespace torchao
