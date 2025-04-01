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
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h>

#ifdef TORCHAO_ENABLE_ARM_I8MM
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h>
#endif // TORCHAO_ENABLE_ARM_I8MM

#include <torchao/experimental/kernels/cpu/aarch64/kleidi/pack.h>

namespace torchao::kernels::cpu::aarch64::kleidi {

// Helper functions
// TODO: find a better place for these?

namespace internal {

inline size_t roundup(size_t a, size_t b) {
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

// KleidiAI kernels require n is even, so we round up to next even number
// if required and pad
inline int adjust_n(int n) {
  return roundup(n, 2);
}

} // namespace internal

namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p {

using Ukernel = struct kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel;

size_t packed_activations_size(
    int m,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  (void)group_size; // unused
  (void)has_weight_zeros; // unused
  auto lhs_packing = get_lhs_packing();
  return lhs_packing.get_lhs_packed_size(m, k, mr, kr, sr);
}

size_t packed_activations_offset(
    int m_idx,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  (void)group_size; // unused
  (void)has_weight_zeros; // unused
  auto lhs_pack = get_lhs_packing();
  return lhs_pack.get_lhs_packed_offset(m_idx, k, mr, kr, sr);
}

void pack_activations(
    void* packed_activations,
    int m,
    int k,
    int group_size,
    const float* activations,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  (void)group_size; // unused
  (void)has_weight_zeros; // unused
  auto lhs_pack = get_lhs_packing();
  lhs_pack.run_lhs_pack(
      m,
      k,
      mr,
      kr,
      sr,
      /*m_index_start=*/0,
      activations,
      /*lhs_stride=*/k * sizeof(float),
      packed_activations);
}

size_t packed_weights_size(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  (void)weight_nbit; // unused
  (void)has_weight_zeros; // unused
  (void)has_bias; // unused
  auto rhs_pack = get_rhs_packing();
  return rhs_pack.get_rhs_packed_size(
      internal::adjust_n(n),
      k,
      nr,
      kr,
      sr,
      group_size,
      kai_datatype::kai_dt_bf16);
}

size_t packed_weights_offset(
    int n_idx,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  (void)has_weight_zeros; // unused
  (void)has_bias; // unused
  auto rhs_pack = get_rhs_packing();
  return rhs_pack.get_rhs_packed_offset(
      n_idx, k, nr, kr, sr, group_size, kai_datatype::kai_dt_bf16);
}

void pack_weights(
    void* packed_weights,
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros,
    const float* bias,
    int nr,
    int kr,
    int sr) {
  if (group_size % 32 != 0) {
    throw std::runtime_error(
        "Group size must be a multiple of 32, but got group_size=" +
        std::to_string(group_size));
  }
  if (k % group_size != 0) {
    throw std::runtime_error(
        "k must be a multiple of group size, but got k=" + std::to_string(k) +
        " and group_size=" + std::to_string(group_size));
  }

  // TODO SIMDify this
  size_t n_groups = n * k / group_size;

  // We don't support weight zeros yet
  if (weight_zeros != nullptr) {
    for (size_t i = 0; i < n_groups; i++) {
      assert(weight_zeros[i] == 0);
    }
  }

  auto weight_scales_bf16_padded =
      std::vector<uint16_t>(internal::adjust_n(n) * k / group_size, 0);
  for (size_t i = 0; i < n_groups; i++) {
    weight_scales_bf16_padded[i] =
        internal::get_bf16_from_float(weight_scales[i]);
  }

  // Prepack weights before packing
  // TODO SIMDify this
  auto packed_weight_qvals_padded =
      std::vector<uint8_t>(internal::adjust_n(n) * k / 2, 0);
  uint8_t wzp = 8;
  for (size_t i = 0; i < n * k; i += 2) {
    const uint8_t low = static_cast<uint8_t>(weight_qvals[i] + wzp);
    const uint8_t high = static_cast<uint8_t>(weight_qvals[i + 1] + wzp);
    packed_weight_qvals_padded[i / 2] = ((high << 4) | (low & 0xF));
  }

  auto bias_padded = std::vector<float>(internal::adjust_n(n), 0.0);
  if (bias != nullptr) {
    for (size_t i = 0; i < n; i++) {
      bias_padded[i] = bias[i];
    }
  }

  // Parameters for packing
  rhs_packing::qparams_t qparams{
      .lhs_zero_point = 1,
      .rhs_zero_point = wzp,
      .scale_dt = kai_datatype::kai_dt_bf16};

  auto rhs_pack = get_rhs_packing();

  rhs_pack.run_rhs_pack(
      /*groups=*/1,
      internal::adjust_n(n),
      k,
      nr,
      kr,
      sr,
      group_size,
      /*rhs=*/
      reinterpret_cast<const uint8_t*>(packed_weight_qvals_padded.data()),
      /*rhs_stride=*/internal::roundup(k, 2) / 2,
      /*bias=*/reinterpret_cast<const float*>(bias_padded.data()),
      /*scale=*/
      reinterpret_cast<const uint16_t*>(weight_scales_bf16_padded.data()),
      /*scale_stride=*/sizeof(uint16_t) *
          (internal::roundup(k, group_size) / group_size),
      /*rhs_packed=*/packed_weights,
      /*extra_bytes=*/0,
      /*qparams=*/&qparams);
}

size_t get_preferred_alignement() {
  return 16;
}

#define DEFINE_KERNEL_STRUCT(name)                                    \
  struct name {                                                       \
    inline static kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel       \
    get_ukernel() {                                                   \
      return kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel(           \
          {.get_m_step = kai_get_m_step_##name,                       \
           .get_n_step = kai_get_n_step_##name,                       \
           .get_mr = kai_get_mr_##name,                               \
           .get_nr = kai_get_nr_##name,                               \
           .get_kr = kai_get_kr_##name,                               \
           .get_sr = kai_get_sr_##name,                               \
           .get_lhs_packed_offset = kai_get_lhs_packed_offset_##name, \
           .get_rhs_packed_offset = kai_get_rhs_packed_offset_##name, \
           .get_dst_offset = kai_get_dst_offset_##name,               \
           .get_dst_size = kai_get_dst_size_##name,                   \
           .run_matmul = kai_run_##name});                            \
    }                                                                 \
    inline static void kernel(                                        \
        float32_t* output,                                            \
        int output_m_stride,                                          \
        int m,                                                        \
        int n,                                                        \
        int k,                                                        \
        int group_size,                                               \
        const void* packed_weights,                                   \
        const void* packed_activations,                               \
        float clamp_min,                                              \
        float clamp_max,                                              \
        bool has_weight_zeros,                                        \
        bool has_bias,                                                \
        bool has_clamp) {                                             \
      (void)has_weight_zeros;                                         \
      (void)has_bias;                                                 \
      if (!has_clamp) {                                               \
        clamp_min = std::numeric_limits<float>::lowest();             \
        clamp_max = std::numeric_limits<float>::max();                \
      }                                                               \
      get_ukernel().run_matmul(                                       \
          m,                                                          \
          n,                                                          \
          k,                                                          \
          group_size,                                                 \
          packed_activations,                                         \
          packed_weights,                                             \
          output,                                                     \
          /*dst_stride_row=*/output_m_stride * sizeof(float),         \
          /*dst_stride_col=*/sizeof(float),                           \
          /*clamp_min=*/clamp_min,                                    \
          /*clamp_max=*/clamp_max);                                   \
    }                                                                 \
  }

DEFINE_KERNEL_STRUCT(
    matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod);
DEFINE_KERNEL_STRUCT(
    matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod);

#ifdef TORCHAO_ENABLE_ARM_I8MM
DEFINE_KERNEL_STRUCT(matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm);
DEFINE_KERNEL_STRUCT(matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm);
#endif // TORCHAO_ENABLE_ARM_I8MM

#undef DEFINE_KERNEL_STRUCT

} // namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p
} // namespace torchao::kernels::cpu::aarch64::kleidi
