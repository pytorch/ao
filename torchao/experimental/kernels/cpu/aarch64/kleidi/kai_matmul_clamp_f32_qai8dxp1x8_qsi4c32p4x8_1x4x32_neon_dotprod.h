// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>

#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp_qsi4c32p.h>

namespace torchao::kernels::cpu::aarch64::kleidi {
namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p {

namespace neon_dotprod_1x4x32 {
const Ukernel get_ukernel() {
  return Ukernel{
      .get_m_step =
          kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .get_n_step =
          kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .get_mr =
          kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .get_nr =
          kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .get_kr =
          kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .get_sr =
          kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .get_lhs_packed_offset =
          kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .get_rhs_packed_offset =
          kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .get_dst_offset =
          kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .get_dst_size =
          kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      .run_matmul =
          kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod};
}

size_t activation_data_size(int m, int k, int group_size) {
  (void)group_size; // unused
  return kai_matmul_clamp_f32_qai8dxp_qsi4c32p::activation_data_size(
      get_ukernel(), m, k);
}

void prepare_activation_data(
    void* activation_data,
    int m,
    int k,
    int group_size,
    const float* activations) {
  (void)group_size; // unused
  kai_matmul_clamp_f32_qai8dxp_qsi4c32p::prepare_activation_data(
      get_ukernel(), activation_data, m, k, activations);
}

size_t weight_data_size(int n, int k, int group_size) {
  return kai_matmul_clamp_f32_qai8dxp_qsi4c32p::weight_data_size(
      get_ukernel(), n, k, group_size);
}

void prepare_weight_data(
    void* weight_data,
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros) {
  kai_matmul_clamp_f32_qai8dxp_qsi4c32p::prepare_weight_data(
      get_ukernel(),
      weight_data,
      n,
      k,
      group_size,
      weight_qvals,
      weight_scales,
      weight_zeros);
}

void kernel(
    float32_t* output,
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* weight_data,
    const void* activation_data,
    const float* bias,
    float clamp_min,
    float clamp_max) {
  (void)bias; // TODO(T203756650) - unused - needs API fixing
  assert(output_m_stride == n);
  if (clamp_min == 0 && clamp_max == 0) {
    clamp_min = std::numeric_limits<float>::lowest();
    clamp_max = std::numeric_limits<float>::max();
  }

  auto ukernel = get_ukernel();
  ukernel.run_matmul(
      m,
      n,
      k,
      group_size,
      activation_data,
      weight_data,
      output,
      /*dst_stride_row=*/n * sizeof(float),
      /*dst_stride_col=*/sizeof(float),
      clamp_min,
      clamp_max);
}

size_t get_preferred_alignement() {
  return 16;
}
} // namespace neon_dotprod_1x4x32
} // namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p
} // namespace torchao::kernels::cpu::aarch64::kleidi
