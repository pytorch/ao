#include "rowwise_scaled_linear_sparse_cutlass.cuh"
#include "rowwise_scaled_linear_sparse_cutlass_e4m3e4m3.h"

namespace torchao {

at::Tensor
rowwise_scaled_linear_sparse_cutlass_e4m3e4m3(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_meta, const at::Tensor& W_scale,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::ScalarType> out_dtype_opt) {
  // Validate input datatypes.
  TORCH_CHECK(
    Xq.dtype() == at::kFloat8_e4m3fn && Wq.dtype() == at::kFloat8_e4m3fn,
    __func__, " : The input datatypes combination ", Xq.dtype(), " for Xq and ",
    Wq.dtype(), " for Wq is not supported");

#if defined(BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS)
  using DtypeXq = cutlass::float_e4m3_t;
  using DtypeWq = cutlass::float_e4m3_t;
  return rowwise_scaled_linear_sparse_cutlass<DtypeXq, DtypeWq>(
      Xq, X_scale, Wq, W_meta, W_scale, bias_opt, out_dtype_opt);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, OPERATOR_NAME);
  return at::Tensor{};
#endif
}

}  // namespace torchao
