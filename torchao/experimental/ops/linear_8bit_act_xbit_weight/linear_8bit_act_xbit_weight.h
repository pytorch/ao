// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <stddef.h>
#include <stdint.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/kernel_config.h>
#include <torchao/experimental/ops/packed_weights_header.h>
#include <array>
#include <optional>

namespace torchao::ops::linear_8bit_act_xbit_weight {

void pack_weights_operator(
    const UKernelConfig& uk,
    // Outputs
    void* packed_weights,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros,
    const float* bias);

void pack_weights_with_lut_operator(
    const UKernelConfig& uk,
    // Outputs
    void* packed_weights,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qval_idxs,
    int n_luts,
    const int8_t* luts,
    const float* weight_scales,
    const int8_t* weight_zeros,
    const float* bias);

// Linear functions
struct LinearTilingParams {
  int mc{0};
  int nc{0};

  // Returns LinearTilingParams with mc and nc chosen so that there are
  // approximately target_tiles_per_thread tiles per thread. The method
  // guarantees 1. mc = m or mc % m_step == 0, and 2. nc = n or nc % n_step == 0
  static LinearTilingParams from_target_tiles_per_thread(
      int m,
      int m_step,
      int n,
      int n_step,
      int target_tiles_per_thread);
};

void linear_operator(
    const UKernelConfig& ukernel_config,
    const std::optional<LinearTilingParams>& tiling_params,
    // Outputs
    float* output,
    // Inputs
    int m,
    int n,
    int k,
    int group_size,
    const void* packed_weights,
    const float* activations,
    bool has_clamp,
    float clamp_min,
    float clamp_max);

} // namespace
  // torchao::ops::linear_8bit_act_xbit_weight
