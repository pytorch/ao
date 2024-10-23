// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Unlike ATen, ExecuTorch op registration appears to only allow on
// EXECUTORCH_LIBRARY per cpp file due to a name redefinition error, so a new
// file is needed for each variant

#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight-impl.h>

namespace {
Tensor _op_out(
    RuntimeContext& ctx,
    const Tensor& activations,
    const Tensor& packed_weights,
    const Tensor& group_size_tensor,
    const Tensor& n_tensor,
    const Tensor& k_tensor,
    Tensor& out) {
  (void)ctx;
  linear_out_cpu</*weight_nbit*/ 7, /*has_weight_zeros*/ false>(
      activations, packed_weights, group_size_tensor, n_tensor, k_tensor, out);
  return out;
}
} // namespace

EXECUTORCH_LIBRARY(torchao, "_linear_8bit_act_7bit0zp_weight.out", _op_out);
