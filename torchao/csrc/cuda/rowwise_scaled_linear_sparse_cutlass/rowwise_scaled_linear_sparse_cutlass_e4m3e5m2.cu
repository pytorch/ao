// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#include "rowwise_scaled_linear_sparse_cutlass.cuh"
#include "rowwise_scaled_linear_sparse_cutlass_e4m3e5m2.h"

namespace torchao {

using torch::stable::Tensor;

Tensor
rowwise_scaled_linear_sparse_cutlass_e4m3e5m2(
    const Tensor& Xq, const Tensor& X_scale, const Tensor& Wq,
    const Tensor& W_meta, const Tensor& W_scale,
    const std::optional<Tensor>& bias_opt,
    const std::optional<torch::headeronly::ScalarType> out_dtype_opt) {
  // Validate input datatypes.
  STD_TORCH_CHECK(
    Xq.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn &&
    Wq.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2,
    __func__, " : The input datatypes combination ", Xq.scalar_type(), " for Xq and ",
    Wq.scalar_type(), " for Wq is not supported");

#if defined(BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS)
  using DtypeXq = cutlass::float_e4m3_t;
  using DtypeWq = cutlass::float_e5m2_t;
  return rowwise_scaled_linear_sparse_cutlass<DtypeXq, DtypeWq>(
      Xq, X_scale, Wq, W_meta, W_scale, bias_opt, out_dtype_opt);
#else
  STD_TORCH_CHECK(false, OPERATOR_NAME, " : Not implemented");
  return Tensor{};
#endif
}

}  // namespace torchao
