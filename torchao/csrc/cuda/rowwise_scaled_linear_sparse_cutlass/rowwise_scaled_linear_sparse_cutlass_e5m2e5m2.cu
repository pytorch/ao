// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#include "rowwise_scaled_linear_sparse_cutlass.cuh"
#include "rowwise_scaled_linear_sparse_cutlass_e5m2e5m2.h"

namespace torchao {

at::Tensor
rowwise_scaled_linear_sparse_cutlass_e5m2e5m2(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_meta, const at::Tensor& W_scale,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::ScalarType> out_dtype_opt) {
  // Validate input datatypes.
  TORCH_CHECK(
    Xq.dtype() == at::kFloat8_e5m2 && Wq.dtype() == at::kFloat8_e5m2,
    __func__, " : The input datatypes combination ", Xq.dtype(), " for Xq and ",
    Wq.dtype(), " for Wq is not supported");

#if defined(BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS)
  using DtypeXq = cutlass::float_e5m2_t;
  using DtypeWq = cutlass::float_e5m2_t;
  return rowwise_scaled_linear_sparse_cutlass<DtypeXq, DtypeWq>(
      Xq, X_scale, Wq, W_meta, W_scale, bias_opt, out_dtype_opt);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, OPERATOR_NAME);
  return at::Tensor{};
#endif
}

}  // namespace torchao
