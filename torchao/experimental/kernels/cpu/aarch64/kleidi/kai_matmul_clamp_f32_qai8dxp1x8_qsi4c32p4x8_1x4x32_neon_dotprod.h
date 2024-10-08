// namespace example
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cassert>
#include <cstddef>
#include <limits>
#include <vector>

#include <torchao/experimental/kernels/cpu/aarch64/kleidi/pack.h>
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp_qsi4c32p.h>

#include <kai/kai_common.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h>

namespace torchao::kernels::cpu::aarch64::kleidi {
namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p {

using ukernel = struct kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel;

namespace neon_dotprod_1x4x32 {
ukernel get_ukernel() {
  return ukernel{
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

size_t roundup(size_t a, size_t b) {
    return ((a + b - 1) / b) * b;
}

int activation_data_size(int m, int k, int group_size) {
  auto ukernel = get_ukernel();
  auto lhs_packing = get_lhs_packing();
  return lhs_packing.get_lhs_packed_size(
      m, k, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr());
}

void prepare_activation_data(
    void* activation_data,
    // Inputs
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    const float* activations) {
  auto ukernel = get_ukernel();
  auto lhs_pack = get_lhs_packing();

  lhs_pack.run_lhs_pack(
      m,
      k,
      ukernel.get_mr(),
      ukernel.get_kr(),
      ukernel.get_sr(),
      /*m_index_start=*/0,
      activations,
      /*lhs_stride=*/k * sizeof(float),
      activation_data);
}

int weight_data_size(int n, int k, int group_size) {
  auto ukernel = get_ukernel();
  auto rhs_pack = get_rhs_packing();
  return rhs_pack.get_rhs_packed_size(
      n,
      k,
      ukernel.get_nr(),
      ukernel.get_kr(),
      ukernel.get_sr(),
      group_size,
      kai_datatype::kai_dt_bf16);
}

inline uint16_t get_bf16_from_float(float f) {
  uint16_t bf16;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  memcpy(&bf16, &f, sizeof(uint16_t));
#else
  const void* fp = reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(&f) + sizeof(float) - sizeof(uint16_t));
  memcpy(&bf16, fp, sizeof(uint16_t));
#endif // __BYTE_ORDER__
  return bf16;
}

// TODO: move most of these functions in the parent namespace and take in
// ukernel as a parameter
void prepare_weight_data(
    void* weight_data,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros) {
  // TODO - remove this constraint and pad when possible
  assert(n % 2 == 0);

  assert(group_size % 32 == 0);
  assert(k % group_size == 0);

  // Convert scales to bf16
  // TODO SIMDify this
  size_t n_groups = n * k / group_size;
  auto weight_scales_bf16 = std::vector<uint16_t>(n_groups, 0);
  for (size_t i = 0; i < n_groups; i++) {
    assert(weight_zeros[i] == 0);
    weight_scales_bf16[i] = get_bf16_from_float(weight_scales[i]);
  }

  // Prepack weights before packing
  // TODO SIMDify this
  auto packed_weight_qvals = std::vector<uint8_t>(n * k / 2, 0);
  uint8_t wzp = 8;
  for (size_t i = 0; i < n * k; i += 2) {
    const uint8_t low = static_cast<uint8_t>(weight_qvals[i] + wzp);
    const uint8_t high = static_cast<uint8_t>(weight_qvals[i+1] + wzp);
    packed_weight_qvals[i / 2] = ((high << 4) | (low & 0xF));
  }

  // Parameters for packing
  rhs_packing::qparams_t qparams{
      .lhs_zero_point=1, .rhs_zero_point=wzp, .scale_dt = kai_datatype::kai_dt_bf16};

  auto ukernel = get_ukernel();
  auto rhs_pack = get_rhs_packing();

  rhs_pack.run_rhs_pack(
      /*groups=*/1,
      n,
      k,
      ukernel.get_nr(),
      ukernel.get_kr(),
      ukernel.get_sr(),
      group_size,
      /*rhs=*/reinterpret_cast<const uint8_t*>(packed_weight_qvals.data()),
      /*rhs_stride=*/roundup(k, 2) / 2,
      /*bias=*/nullptr, // TODO fix APIs to move bias here
      /*scale=*/reinterpret_cast<const uint16_t*>(weight_scales_bf16.data()),
      /*scale_stride=*/ sizeof(uint16_t) * (roundup(k, group_size) / group_size),
      /*rhs_packed=*/weight_data,
      /*extra_bytes=*/0,
      /*qparams=*/&qparams);
}

void kernel(
    // Outputs
    float32_t* output,
    // Inputs
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* weight_data,
    const void* activation_data,
    // Not applied if nullptr
    const float* bias,
    // zeros if has_clamp = false
    float clamp_min,
    float clamp_max) {
  assert(output_m_stride == n);
  if (clamp_min == clamp_max && clamp_min == 0) {
    clamp_min = std::numeric_limits<float_t>::lowest();
    clamp_max = std::numeric_limits<float_t>::max();
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

size_t get_alignement() {
  return 16;
}
} // namespace neon_dotprod_1x4x32
} // namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p
} // namespace torchao::kernels::cpu::aarch64::kleidi
