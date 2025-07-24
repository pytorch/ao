// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/ops/groupwise_lowbit_weight_lut/groupwise_lowbit_weight_lut.h>

#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/memory.h>
#include <torchao/experimental/ops/parallel.h>
#include <algorithm>
#include <cassert>
#include <vector>

namespace torchao::ops::groupwise_lowbit_weight_lut {

void pack_weights_operator(
    const UKernelConfig& uk,
    // Outputs
    void* packed_weights_ptr,
    // Inputs
    int n,
    int k,
    int scale_group_size,
    int lut_group_size,
    const uint8_t* weight_qval_indices,
    const float* weight_scales,
    const float* weight_luts,
    const float* bias) {
  TORCHAO_CHECK(
      lut_group_size % scale_group_size == 0,
      "scale_group_size must devide lut_group_size");
  TORCHAO_CHECK(k % scale_group_size == 0, "scale_group_size must divide k");
  TORCHAO_CHECK(
      lut_group_size % (k * uk.nr) == 0,
      "lut_group_size must be a multiple of k*nr");
  TORCHAO_CHECK(k % uk.kr == 0, "kr must divide k");

  // 1. Define the block size for parallel work.
  int n_step = uk.n_step;
  int nc = std::min(n, n_step);
  const int num_nc_panels = (n + nc - 1) / nc;

  torchao::parallel_1d(0, num_nc_panels, [&](int64_t idx) {
    const int n_idx = idx * nc;
    const int nc_tile_size = std::min(nc, n - n_idx);

    auto packed_weights_offset = uk.packed_weights_offset(
        n_idx,
        k,
        uk.weight_nbit,
        scale_group_size,
        uk.has_scales,
        uk.has_bias,
        uk.nr,
        uk.kr,
        uk.sr);

    // Calculate offsets for all input pointers
    int weight_qval_indices_offset = n_idx * k;
    // Scales are packed in groups of nr
    int scales_offset = weight_qval_indices_offset / scale_group_size;
    int luts_offset =
        (weight_qval_indices_offset / lut_group_size) * (1 << uk.weight_nbit);

    // 2. Call pack_weights with chunk arguments
    uk.pack_weights(
        static_cast<uint8_t*>(packed_weights_ptr) + packed_weights_offset,
        weight_qval_indices + weight_qval_indices_offset,
        uk.has_scales ? weight_scales + scales_offset : nullptr,
        weight_luts + luts_offset,
        nc_tile_size,
        k,
        scale_group_size,
        lut_group_size,
        uk.has_scales,
        uk.has_bias,
        uk.has_bias ? bias + n_idx : nullptr,
        uk.nr,
        uk.kr,
        uk.sr);
  });
}

GroupwiseTilingParams GroupwiseTilingParams::from_target_tiles_per_thread(
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

  GroupwiseTilingParams tiling_params;
  tiling_params.mc = mc;
  tiling_params.nc = nc;
  return tiling_params;
}

void groupwise_lowbit_weight_lut_parallel_operator(
    const UKernelConfig& uk,
    const std::optional<GroupwiseTilingParams>& tiling_params,
    float* output,
    int m,
    int n,
    int k,
    int scale_group_size,
    int lut_group_size,
    const void* packed_weights,
    const float* activations,
    bool has_clamp,
    float clamp_min,
    float clamp_max) {
  TORCHAO_CHECK(
      lut_group_size % scale_group_size == 0,
      "scale_group_size must divide lut_group_size");
  TORCHAO_CHECK(k % scale_group_size == 0, "scale_group_size must divide k");
  TORCHAO_CHECK(
      lut_group_size % (k * uk.nr) == 0, "(k * nr) must divide lut_group_size");
  TORCHAO_CHECK(
      scale_group_size % uk.kr == 0, "kr must divide scale_group_size");
  int config_idx = uk.select_config_idx(m);
  auto& kernel_config = uk.configs[config_idx];
  int n_step = uk.n_step;
  int m_step = kernel_config.m_step;

  int mc, nc;
  if (tiling_params.has_value()) {
    mc = tiling_params->mc;
    nc = tiling_params->nc;
  } else {
    // If no params are provided, calculate them to balance the workload.
    auto params = GroupwiseTilingParams::from_target_tiles_per_thread(
        m_step, m_step, n, n_step, /*target_tiles_per_thread=*/5);
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

  const int num_mc_tiles = (m + mc - 1) / mc;
  const int num_nc_tiles = (n + nc - 1) / nc;

  const size_t packed_activations_size = kernel_config.packed_activations_size(
      mc, k, kernel_config.mr, uk.kr, uk.sr);
  auto packed_activations = torchao::make_aligned_byte_ptr(
      uk.preferred_alignment, packed_activations_size);

  // Outer loop over M blocks
  for (int mc_tile_idx = 0; mc_tile_idx < num_mc_tiles; ++mc_tile_idx) {
    const int mc_tile_start = mc_tile_idx * mc;
    const int mc_tile_size = std::min(mc, m - mc_tile_start);
    const float* activation_row_ptr = activations + mc_tile_start * k;

    kernel_config.pack_activations(
        (float*)packed_activations.get(),
        mc_tile_size,
        k,
        activation_row_ptr,
        kernel_config.mr,
        uk.kr,
        uk.sr);

    // Parallelize the work over the larger NC-tiles
    torchao::parallel_1d(0, num_nc_tiles, [&](int64_t n_tile_idx) {
      const int nc_tile_start = n_tile_idx * nc;
      const int nc_tile_size = std::min(nc, n - nc_tile_start);
      float* output_tile_ptr = output + mc_tile_start * n + nc_tile_start;

      const size_t packed_weights_offset = uk.packed_weights_offset(
          nc_tile_start,
          k,
          uk.weight_nbit,
          scale_group_size,
          uk.has_scales,
          uk.has_bias,
          uk.nr,
          uk.kr,
          uk.sr);
      const void* packed_weights_for_tile =
          static_cast<const uint8_t*>(packed_weights) + packed_weights_offset;

      kernel_config.kernel(
          output_tile_ptr,
          /*output_m_stride=*/n,
          /*m=*/mc_tile_size,
          /*n=*/nc_tile_size,
          k,
          scale_group_size,
          lut_group_size,
          packed_weights_for_tile,
          packed_activations.get(),
          clamp_min,
          clamp_max,
          uk.has_bias,
          has_clamp);
    });
  }
}

} // namespace torchao::ops::groupwise_lowbit_weight_lut
