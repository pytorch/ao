// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <stdint.h>

namespace torchao::ops::linear_8bit_act_xbit_weight {

struct UKernelConfig {
  using activation_data_size_fn_type = int (*)(int m, int k, int group_size);
  using prepare_activation_data_fn_type = void (*)(
      void* activation_data,
      int m,
      int k,
      int group_size,
      const float* activations);
  using weight_data_size_fn_type = int (*)(int n, int k, int group_size);
  using prepare_weight_data_fn_type = void (*)(
      void* weight_data,
      int n,
      int k,
      int group_size,
      const int8_t* weight_qvals,
      const float* weight_scales,
      const int8_t* weight_zeros);
  using kernel_fn_type = void (*)(
      float* output,
      int output_m_stride,
      int m,
      int n,
      int k,
      int group_size,
      const void* weight_data,
      const void* activation_data,
      const float* bias,
      float clamp_min,
      float clamp_max);

  activation_data_size_fn_type activation_data_size_fn{nullptr};
  // activation_data_alignment is only a preferred alignment for
  // performance reasons.  Integration surfaces are not required to
  // respect this alignment, and the ukernel must behave correctly no matter
  // how the prepared_activation_data byte-array is aligned
  int activation_data_alignment{0};
  prepare_activation_data_fn_type prepare_activation_data_fn{nullptr};

  weight_data_size_fn_type weight_data_size_fn{nullptr};
  // weight_data_alignment is only a preferred alignment for
  // performance reasons.  Integration surfaces are not required to
  // respect this alignment, and the ukernel must behave correctly no matter
  // how the prepared_weight_data byte-array is aligned
  int weight_data_alignment{0};
  prepare_weight_data_fn_type prepare_weight_data_fn{nullptr};

  kernel_fn_type kernel_fn{nullptr};
  int mr{0};
  int nr{0};
};

// Pack weight functions
struct PackWeightDataTilingParams {
  int nc_by_nr{1};
};

PackWeightDataTilingParams get_default_pack_weight_data_tiling_params(
    const UKernelConfig& ukernel_config,
    int n,
    int target_panels_per_thread = 1);

inline int get_packed_weight_data_size(
    const UKernelConfig& ukernel_config,
    int n,
    int k,
    int group_size) {
  return ukernel_config.weight_data_size_fn(n, k, group_size);
}

inline int get_packed_weight_data_alignment(
    const UKernelConfig& ukernel_config) {
  return ukernel_config.weight_data_alignment;
}

void pack_weight_data_operator(
    const UKernelConfig& ukernel_config,
    const PackWeightDataTilingParams& tiling_params,
    // Outputs
    void* weight_data,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros);

// Linear functions
struct LinearTilingParams {
  int mc_by_mr{1};
  int nc_by_nr{1};
};

LinearTilingParams get_default_linear_tiling_params(
    const UKernelConfig& ukernel_config,
    int m,
    int n,
    int target_tiles_per_thread = 5);

enum class LinearTileSchedulingPolicy {
  single_mc_parallel_nc,
  parallel_mc_parallel_nc
};

int get_activation_data_buffer_size(
    const UKernelConfig& ukernel_config,
    const LinearTilingParams& tiling_params,
    LinearTileSchedulingPolicy scheduling_policy,
    int m,
    int k,
    int group_size);

inline int get_activation_data_buffer_alignment(
    const UKernelConfig& ukernel_config) {
  return ukernel_config.activation_data_alignment;
}

void linear_operator(
    const UKernelConfig& ukernel_config,
    const LinearTilingParams& tiling_params,
    LinearTileSchedulingPolicy scheduling_policy,
    char* activation_data_buffer,
    // Outputs
    float* output,
    // Inputs
    int m,
    int n,
    int k,
    int group_size,
    const void* weight_data,
    const float* activations,
    const float* bias,
    float clamp_min,
    float clamp_max);

} // namespace
  // torchao::ops::linear_8bit_act_xbit_weight
