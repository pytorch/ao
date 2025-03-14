// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#include <torch/library.h>

#include "rowwise_scaled_linear_cutlass.cuh"

namespace torchao {

at::Tensor
rowwise_scaled_linear_cutlass_s8s4(
    const at::Tensor& Xq, const at::Tensor& X_scale, const at::Tensor& Wq,
    const at::Tensor& W_scale,
    const std::optional<at::Tensor>& bias_opt = std::nullopt,
    const std::optional<at::ScalarType> out_dtype_opt = std::nullopt) {
  // Validate input datatypes.
  TORCH_CHECK(Xq.dtype() == at::kChar && Wq.dtype() == at::kChar,
              __func__, " : The input datatypes combination ", Xq.dtype(),
              " for Xq and ", Wq.dtype(), " for Wq is not supported");

#if defined(BUILD_ROWWISE_SCALED_LINEAR_CUTLASS)
  // Dispatch to appropriate kernel template.
  using ElementA = int8_t;
  using ElementB = cutlass::int4b_t;
  return rowwise_scaled_linear_cutlass<ElementA, ElementB>(
    Xq, X_scale, Wq, W_scale, bias_opt, out_dtype_opt);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, OPERATOR_NAME);
  return at::Tensor{};
#endif
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::rowwise_scaled_linear_cutlass_s8s4",
         &rowwise_scaled_linear_cutlass_s8s4);
}

}  // namespace torchao
