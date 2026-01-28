// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "rowwise_scaled_linear_sparse_cutlass_e4m3e4m3.h"
#include "rowwise_scaled_linear_sparse_cutlass_e4m3e5m2.h"
#include "rowwise_scaled_linear_sparse_cutlass_e5m2e4m3.h"
#include "rowwise_scaled_linear_sparse_cutlass_e5m2e5m2.h"

namespace torchao {

using torch::stable::Tensor;

Tensor
rowwise_scaled_linear_sparse_cutlass_f8f8(
    const Tensor& Xq, const Tensor& X_scale, const Tensor& Wq,
    const Tensor& W_meta, const Tensor& W_scale,
    const std::optional<Tensor>& bias_opt = std::nullopt,
    const std::optional<torch::headeronly::ScalarType> out_dtype_opt = std::nullopt) {
  // Validate input datatypes.
  STD_TORCH_CHECK(
      (Xq.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn &&
       Wq.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn) ||
      (Xq.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn &&
       Wq.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2) ||
      (Xq.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2 &&
       Wq.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn) ||
      (Xq.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2 &&
       Wq.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2),
      __func__, " : The input datatypes combination ", Xq.scalar_type(),
      " for Xq and ", Wq.scalar_type(), " for Wq is not supported");

  // Dispatch to appropriate kernel template.
  if (Xq.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn &&
      Wq.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn) {
    return rowwise_scaled_linear_sparse_cutlass_e4m3e4m3(
      Xq, X_scale, Wq, W_meta, W_scale, bias_opt, out_dtype_opt);
  } else if (Xq.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn &&
             Wq.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2) {
    return rowwise_scaled_linear_sparse_cutlass_e4m3e5m2(
      Xq, X_scale, Wq, W_meta, W_scale, bias_opt, out_dtype_opt);
  } else if (Xq.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2 &&
             Wq.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn) {
    return rowwise_scaled_linear_sparse_cutlass_e5m2e4m3(
      Xq, X_scale, Wq, W_meta, W_scale, bias_opt, out_dtype_opt);
  } else if (Xq.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2 &&
             Wq.scalar_type() == torch::headeronly::ScalarType::Float8_e5m2) {
    return rowwise_scaled_linear_sparse_cutlass_e5m2e5m2(
      Xq, X_scale, Wq, W_meta, W_scale, bias_opt, out_dtype_opt);
  }
  return Tensor{};
}

#ifdef DEF_SPARSE_CUTLASS_OPS
STABLE_TORCH_LIBRARY(torchao, m) {
  m.def(
      "torchao::rowwise_scaled_linear_sparse_cutlass_f8f8(Tensor input, Tensor input_scale, Tensor weight, Tensor weight_meta, Tensor weight_scale, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor");
}
#endif

STABLE_TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::rowwise_scaled_linear_sparse_cutlass_f8f8",
         TORCH_BOX(&rowwise_scaled_linear_sparse_cutlass_f8f8));
}

}  // namespace torchao
