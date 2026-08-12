// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/reduction.h>
#include <cassert>

int32_t torchao::kernels::cpu::aarch64::reduction::compute_sum(
    const int8_t* vals,
    int size) {
  assert(size >= 1);

  int32_t res = 0;
  int i = 0;

#pragma unroll(4)
  for (; i + 15 < size; i += 16) {
    int8x16_t vec_vals = vld1q_s8(vals + i);
#if defined(__aarch64__)
    res += (int)(vaddlvq_s8(vec_vals));
#else
    // vaddlvq_s8 (widening across-vector add) is AArch64-only. On AArch32 NEON,
    // widen pairwise (s8->s16->s32) then fold the 4 lanes to a scalar.
    int16x8_t w16 = vpaddlq_s8(vec_vals);
    int32x4_t w32 = vpaddlq_s16(w16);
    int32x2_t w2 = vadd_s32(vget_low_s32(w32), vget_high_s32(w32));
    w2 = vpadd_s32(w2, w2);
    res += vget_lane_s32(w2, 0);
#endif
  }
  for (; i < size; i += 1) {
    res += vals[i];
  }
  return res;
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
