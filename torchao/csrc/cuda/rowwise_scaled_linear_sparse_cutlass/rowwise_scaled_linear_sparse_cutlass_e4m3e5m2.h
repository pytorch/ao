// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <optional>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

namespace torchao {

torch::stable::Tensor
rowwise_scaled_linear_sparse_cutlass_e4m3e5m2(
    const torch::stable::Tensor& Xq, const torch::stable::Tensor& X_scale, const torch::stable::Tensor& Wq,
    const torch::stable::Tensor& W_meta, const torch::stable::Tensor& W_scale,
    const std::optional<torch::stable::Tensor>& bias_opt,
    const std::optional<torch::headeronly::ScalarType> out_dtype_opt);

}  // namespace torchao
