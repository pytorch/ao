// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>
#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <torchao/experimental/ops/memory.h>
#include <torchao/experimental/ops/parallel.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

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
    const float* bias) {
  TORCHAO_CHECK(group_size % 16 == 0, "group_size must be a multiple of 16");
  TORCHAO_CHECK(k % group_size == 0, "group_size must divide k");
  TORCHAO_CHECK(
      uk.has_bias == (bias != nullptr), "bias/has_bias is inconsistent");
  TORCHAO_CHECK(
      uk.has_weight_zeros == (weight_zeros != nullptr),
      "weight_zeros/has_weight_zeros is inconsistent");

  int n_step = uk.n_step;
  int nc = std::min(n, n_step);
  int num_nc_panels = (n + nc - 1) / nc;

  torchao::parallel_1d(0, num_nc_panels, [&](int64_t idx) {
    int nc_tile_idx = idx;
    int n_idx = nc_tile_idx * nc;
    int nc_tile_size = std::min(nc, n - n_idx);

    auto packed_weights_offset = uk.packed_weights_offset(
        n_idx,
        k,
        group_size,
        uk.weight_nbit,
        uk.has_weight_zeros,
        uk.has_bias,
        uk.nr,
        uk.kr,
        uk.sr);

    int weight_qvals_offset = n_idx * k;
    int weight_scales_and_zeros_offset = (n_idx * k / group_size);
    uk.pack_weights(
        (char*)packed_weights + packed_weights_offset,
        /*n=*/nc_tile_size,
        k,
        group_size,
        weight_qvals + weight_qvals_offset,
        weight_scales + weight_scales_and_zeros_offset,
        (weight_zeros == nullptr)
            ? nullptr
            : (weight_zeros + weight_scales_and_zeros_offset),
        (bias == nullptr) ? nullptr : (bias + n_idx),
        uk.nr,
        uk.kr,
        uk.sr);
  });
}

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
    const float* bias) {
  int n_step = uk.n_step;
  int nc = std::min(n, n_step);
  int num_nc_panels = (n + nc - 1) / nc;

  torchao::parallel_1d(0, num_nc_panels, [&](int64_t idx) {
    int nc_tile_idx = idx;
    int n_idx = nc_tile_idx * nc;
    int nc_tile_size = std::min(nc, n - n_idx);

    auto packed_weights_offset = uk.packed_weights_offset(
        n_idx,
        k,
        group_size,
        uk.weight_nbit,
        uk.has_weight_zeros,
        uk.has_bias,
        uk.nr,
        uk.kr,
        uk.sr);

    int weight_qval_idxs_offset = n_idx * k;
    int weight_scales_and_zeros_offset = (n_idx * k / group_size);
    uk.pack_weights_with_lut(
        (char*)packed_weights + packed_weights_offset,
        /*n=*/nc_tile_size,
        k,
        group_size,
        weight_qval_idxs + weight_qval_idxs_offset,
        n_luts,
        luts,
        weight_scales + weight_scales_and_zeros_offset,
        (weight_zeros == nullptr)
            ? nullptr
            : (weight_zeros + weight_scales_and_zeros_offset),
        (bias == nullptr) ? nullptr : (bias + n_idx),
        uk.nr,
        uk.kr,
        uk.sr);
  });
}

LinearTilingParams LinearTilingParams::from_target_tiles_per_thread(
    int m,
    int m_step,
    int n,
    int n_step,
    int target_tiles_per_thread) {
  TORCHAO_CHECK(m >= 1, "m must be >= 1");
  TORCHAO_CHECK(m_step >= 1, "m_step must be >= 1");

  TORCHAO_CHECK(n >= 1, "n must be >= 1");
  TORCHAO_CHECK(n_step >= 1, "n_step must be >= 1");
  TORCHAO_CHECK(
      target_tiles_per_thread >= 1, "target_tiles_per_thread must be >= 1");
  auto num_threads = torchao::get_num_threads();
  TORCHAO_CHECK(num_threads >= 1, "num_threads must be >= 1");

  int mc = m_step;
  int num_mc_panels = (m + mc - 1) / mc;

  int numerator = n * num_mc_panels;
  int denominator = num_threads * target_tiles_per_thread;

  // Set nc = ceil(numerator / denominator)
  int nc = (numerator + denominator - 1) / denominator;
  assert(nc >= 1);

  // Replace nc with next number n_step divides
  nc = ((nc + n_step - 1) / n_step) * n_step;

  // Clamp mc, nc to be no larger than m, n
  mc = std::min(m, mc);
  nc = std::min(n, nc);

  assert((mc == m) || (mc % m_step == 0));
  assert((nc == n) || (nc % n_step == 0));

  LinearTilingParams tiling_params;
  tiling_params.mc = mc;
  tiling_params.nc = nc;
  return tiling_params;
}

void linear_operator(
    const UKernelConfig& uk,
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
    float clamp_max) {
  TORCHAO_CHECK(group_size % 16 == 0, "group_size must be a multiple of 16");
  TORCHAO_CHECK(k % group_size == 0, "group_size must divide k");

  // Select linear config based on m
  int linear_config_idx = uk.select_linear_config_idx(m);
  auto& linear_config = uk.linear_configs[linear_config_idx];
  int n_step = uk.n_step;
  int m_step = linear_config.m_step;

  // Choose tiling params
  int mc, nc;
  if (tiling_params.has_value()) {
    mc = tiling_params->mc;
    nc = tiling_params->nc;
  } else {
    auto params = LinearTilingParams::from_target_tiles_per_thread(
        // We process m sequentially, so m_step is the "m" for the purpose of computing tiling params
        m_step,
        m_step,
        n,
        n_step,
        /*target_tiles_per_thread=*/5);
    mc = params.mc;
    nc = params.nc;
  }
  TORCHAO_CHECK(mc >= 1, "mc must be >= 1");
  TORCHAO_CHECK(nc >= 1, "nc must be >= 1");
  TORCHAO_CHECK(
      (mc == m) || (mc % m_step == 0),
      "mc from tiling_params must be m or a multiple of m_step");
  TORCHAO_CHECK(
      (nc == n) || (nc % n_step == 0),
      "nc from tiling_params must be n or a multiple of n_step");

  int num_mc_panels = (m + mc - 1) / mc;
  int num_nc_panels = (n + nc - 1) / nc;

  auto packed_activations_size = linear_config.packed_activations_size(
      mc, k, group_size, uk.has_weight_zeros, linear_config.mr, uk.kr, uk.sr);

  auto packed_activations = torchao::make_aligned_byte_ptr(
      uk.preferred_alignment, packed_activations_size);
  for (int mc_tile_idx = 0; mc_tile_idx < num_mc_panels; mc_tile_idx++) {
    int m_idx = mc_tile_idx * mc;
    int mc_tile_size = std::min(mc, m - m_idx);
    int activations_offset = m_idx * k;

    linear_config.pack_activations(
        packed_activations.get(),
        /*m=*/mc_tile_size,
        k,
        group_size,
        activations + activations_offset,
        uk.has_weight_zeros,
        linear_config.mr,
        uk.kr,
        uk.sr);

    torchao::parallel_1d(0, num_nc_panels, [&](int64_t idx) {
      int nc_tile_idx = idx;
      int n_idx = nc_tile_idx * nc;
      int nc_tile_size = std::min(nc, n - n_idx);
      int output_offset = m_idx * n + n_idx;

      auto packed_weights_offset = uk.packed_weights_offset(
          n_idx,
          k,
          group_size,
          uk.weight_nbit,
          uk.has_weight_zeros,
          uk.has_bias,
          uk.nr,
          uk.kr,
          uk.sr);

      linear_config.kernel(
          output + output_offset,
          /*output_m_stride=*/n,
          /*m=*/mc_tile_size,
          /*n=*/nc_tile_size,
          k,
          group_size,
          /*packed_weights=*/(char*)packed_weights + packed_weights_offset,
          /*packed_activations=*/packed_activations.get(),
          clamp_min,
          clamp_max,
          uk.has_weight_zeros,
          uk.has_bias,
          has_clamp);
    });
  }
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
