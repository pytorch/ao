// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Unlike ATen, ExecuTorch op registration appears to only allow on
// EXECUTORCH_LIBRARY per cpp file due to a name redefinition error, so a new
// file is needed for each variant

#include <torchao/experimental/ops/linear/linear_a8wxdq_op/linear_a8wxdq-impl.h>

namespace {
Tensor _op_out(
    RuntimeContext& ctx,
    const Tensor& packed_weights,
    const Tensor& n_tensor,
    const Tensor& k_tensor,
    const Tensor& group_size_tensor,
    const Tensor& activations,
    Tensor& out) {
  (void)ctx;
  linear_out_cpu</*weight_nbit*/ 3, /*has_weight_zeros*/ false>(
      packed_weights, n_tensor, k_tensor, group_size_tensor, activations, out);
  return out;
}
} // namespace

EXECUTORCH_LIBRARY(torchao, "_linear_a8sz_w3s.out", _op_out);
