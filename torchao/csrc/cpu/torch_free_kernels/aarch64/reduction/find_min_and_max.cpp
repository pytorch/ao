// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/reduction.h>
#include <cassert>

void torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
    float32_t& min,
    float32_t& max,
    const float32_t* vals,
    int size) {
  assert(size > 0);

  // Needed in case size < 4 so we don't compare to
  // uninitialized min/max values
  min = vals[0];
  max = min;

  int i = 0;
  if (i + 3 < size) {
    float32x4_t mins = vld1q_f32(vals + i);
    float32x4_t maxes = mins;
    i += 4;
    for (; i + 3 < size; i += 4) {
      float32x4_t v = vld1q_f32(vals + i);
      mins = vminq_f32(mins, v);
      maxes = vmaxq_f32(maxes, v);
    }
#if defined(__aarch64__)
    min = vminvq_f32(mins);
    max = vmaxvq_f32(maxes);
#else
    // vminvq_f32/vmaxvq_f32 (across-vector reductions) are AArch64-only. On
    // AArch32 NEON, fold the 128-bit vector to a scalar with pairwise reductions.
    float32x2_t min_pair = vpmin_f32(vget_low_f32(mins), vget_high_f32(mins));
    min_pair = vpmin_f32(min_pair, min_pair);
    min = vget_lane_f32(min_pair, 0);
    float32x2_t max_pair = vpmax_f32(vget_low_f32(maxes), vget_high_f32(maxes));
    max_pair = vpmax_f32(max_pair, max_pair);
    max = vget_lane_f32(max_pair, 0);
#endif
  }

  // Remainder
  while (i < size) {
    if (vals[i] < min) {
      min = vals[i];
    }
    if (vals[i] > max) {
      max = vals[i];
    }
    i += 1;
  }
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
