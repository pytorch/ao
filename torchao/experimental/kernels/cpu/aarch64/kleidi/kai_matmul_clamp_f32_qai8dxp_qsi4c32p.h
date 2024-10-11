// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <kai/kai_common.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h>

#include <torchao/experimental/kernels/cpu/aarch64/kleidi/pack.h>

namespace torchao::kernels::cpu::aarch64::kleidi {

// Helper functions
// TODO: find a better place for these?

size_t roundup(size_t a, size_t b) {
  return ((a + b - 1) / b) * b;
}

uint16_t get_bf16_from_float(float f) {
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

namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p {

using Ukernel = struct kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel;

size_t activation_data_size(const Ukernel ukernel, int m, int k) {
  auto lhs_packing = get_lhs_packing();
  return lhs_packing.get_lhs_packed_size(
      m, k, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr());
}

void prepare_activation_data(
    const Ukernel ukernel,
    void* activation_data,
    int m,
    int k,
    const float* activations) {
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

size_t weight_data_size(const Ukernel ukernel, int n, int k, int group_size) {
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

void prepare_weight_data(
    const Ukernel ukernel,
    void* weight_data,
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros) {
  // TODO(T204312268) - remove this constraint and pad when possible
  assert(n % 2 == 0);

  assert(group_size % 32 == 0);
  assert(k % group_size == 0);

  // TODO SIMDify this
  size_t n_groups = n * k / group_size;
  auto weight_scales_bf16 = std::vector<uint16_t>(n_groups, 0);

  // We don't support weight zeros yet
  if (weight_zeros != nullptr) {
    for (size_t i = 0; i < n_groups; i++) {
      assert(weight_zeros[i] == 0);
    }
  }

  for (size_t i = 0; i < n_groups; i++) {
    weight_scales_bf16[i] = get_bf16_from_float(weight_scales[i]);
  }

  // Prepack weights before packing
  // TODO SIMDify this
  auto packed_weight_qvals = std::vector<uint8_t>(n * k / 2, 0);
  uint8_t wzp = 8;
  for (size_t i = 0; i < n * k; i += 2) {
    const uint8_t low = static_cast<uint8_t>(weight_qvals[i] + wzp);
    const uint8_t high = static_cast<uint8_t>(weight_qvals[i + 1] + wzp);
    packed_weight_qvals[i / 2] = ((high << 4) | (low & 0xF));
  }

  // Parameters for packing
  rhs_packing::qparams_t qparams{
      .lhs_zero_point = 1,
      .rhs_zero_point = wzp,
      .scale_dt = kai_datatype::kai_dt_bf16};

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
      /*bias=*/nullptr, // TODO(T203756650) fix APIs to move bias here
      /*scale=*/reinterpret_cast<const uint16_t*>(weight_scales_bf16.data()),
      /*scale_stride=*/sizeof(uint16_t) * (roundup(k, group_size) / group_size),
      /*rhs_packed=*/weight_data,
      /*extra_bytes=*/0,
      /*qparams=*/&qparams);
}

} // namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p
} // namespace torchao::kernels::cpu::aarch64::kleidi
