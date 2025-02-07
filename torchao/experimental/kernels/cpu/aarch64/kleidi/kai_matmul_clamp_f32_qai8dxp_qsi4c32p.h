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
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>

#ifdef TORCHAO_ENABLE_ARM_I8MM
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h>
#endif // TORCHAO_ENABLE_ARM_I8MM

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

size_t activation_data_size(int mr, int kr, int sr, int m, int k) {
  auto lhs_packing = get_lhs_packing();
  return lhs_packing.get_lhs_packed_size(
      m, k, mr, kr, sr);
}

void prepare_activation_data(
    int mr,
    int kr,
    int sr,
    void* activation_data,
    int m,
    int k,
    const float* activations) {
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
      activation_data);
}

size_t weight_data_size(int nr, int kr, int sr, int n, int k, int group_size) {
  auto rhs_pack = get_rhs_packing();
  return rhs_pack.get_rhs_packed_size(
      n,
      k,
      nr,
      kr,
      sr,
      group_size,
      kai_datatype::kai_dt_bf16);
}

void prepare_weight_data(
    int nr,
    int kr,
    int sr,
    void* weight_data,
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros,
    const float* bias) {
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
      nr,
      kr,
      sr,
      group_size,
      /*rhs=*/reinterpret_cast<const uint8_t*>(packed_weight_qvals.data()),
      /*rhs_stride=*/roundup(k, 2) / 2,
      /*bias=*/bias,
      /*scale=*/reinterpret_cast<const uint16_t*>(weight_scales_bf16.data()),
      /*scale_stride=*/sizeof(uint16_t) * (roundup(k, group_size) / group_size),
      /*rhs_packed=*/weight_data,
      /*extra_bytes=*/0,
      /*qparams=*/&qparams);
}


size_t get_preferred_alignement() {
  return 16;
}


#define DEFINE_WEIGHT_DATA_FNS(nr, kr, sr)                                          \
    size_t weight_data_size_nr##nr##_kr##kr##_sr##sr(int n, int k, int group_size) {  \
        return weight_data_size(nr, kr, sr, n, k, group_size);          \
    }                                                                               \
    void prepare_weight_data_nr##nr##_kr##kr##_sr##sr(                                \
        void* weight_data,                                                          \
        int n,                                                                      \
        int k,                                                                      \
        int group_size,                             \
        const int8_t* weight_qvals, \
        const float* weight_scales, \
        const int8_t* weight_zeros, \
        const float* bias) { \
        prepare_weight_data(nr, kr, sr, weight_data, n, k, group_size, weight_qvals, weight_scales, weight_zeros, bias); \
    }

#define DEFINE_ACTIVATION_DATA_FNS(mr, kr, sr) \
    size_t activation_data_size_mr##mr##_kr##kr##_sr##sr(int m, int k, int group_size) {  \
        (void)group_size; \
        return activation_data_size(mr, kr, sr, m, k);        \
    } \
    void prepare_activation_data_mr##mr##_kr##kr##_sr##sr(void* activation_data, int m, int k, int group_size, const float* activations) { \
        (void)group_size; \
        prepare_activation_data(mr, kr, sr, activation_data, m, k, activations); \
    }

// TODO: first and suffix need to be better, e.g., parametrized by mr, nr, etc
// But I don't quite follow the naming convention for KleidiAI
#define DEFINE_KERNEL_FNS(first, suffix) \
  namespace impl_##suffix { \
  const Ukernel get_ukernel() { \
  return Ukernel{ \
  .get_m_step = kai_get_m_step_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .get_n_step = kai_get_n_step_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .get_mr = kai_get_mr_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .get_nr = kai_get_nr_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .get_kr = kai_get_kr_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .get_sr = kai_get_sr_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .get_lhs_packed_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .get_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix, \
  .run_matmul = kai_run_matmul_clamp_f32_qai8dxp##first##_qsi4c32p##suffix \
  }; \
  } \
  void kernel( \
      float32_t* output, \
      int output_m_stride, \
      int m, \
      int n, \
      int k, \
      int group_size, \
      const void* weight_data, \
      const void* activation_data, \
      float clamp_min, \
      float clamp_max) { \
        get_ukernel().run_matmul( \
          m, \
          n, \
          k, \
          group_size, \
          activation_data, \
          weight_data, \
          output, \
          /*dst_stride_row=*/ output_m_stride * sizeof(float), \
          /*dst_stride_col=*/ sizeof(float), \
          /*clamp_min=*/std::numeric_limits<float>::lowest(), \
          /*clamp_max=*/std::numeric_limits<float>::max() \
          ); \
      } \
  }



DEFINE_WEIGHT_DATA_FNS(/*nr*/8, /*kr*/16, /*sr*/2)
DEFINE_ACTIVATION_DATA_FNS(/*mr*/1, /*kr*/16, /*sr*/2)
DEFINE_KERNEL_FNS(1x8, 8x8_1x8x32_neon_dotprod)
DEFINE_KERNEL_FNS(1x8, 4x8_1x4x32_neon_dotprod)

#ifdef TORCHAO_ENABLE_ARM_I8MM
DEFINE_KERNEL_FNS(4x8, 4x8_8x4x32_neon_i8mm)
DEFINE_KERNEL_FNS(4x8, 8x8_4x8x32_neon_i8mm)
#endif // TORCHAO_ENABLE_ARM_I8MM

#undef DEFINE_WEIGHT_DATA_FNS
#undef DEFINE_ACTIVATION_DATA_FNS
#undef DEFINE_KERNEL_FNS

} // namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p
} // namespace torchao::kernels::cpu::aarch64::kleidi
