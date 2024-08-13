// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>

void torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
    float32_t& min,
    float32_t& max,
    const float32_t* vals,
    int size) {
  float32x4_t mins = vdupq_n_f32(0.0);
  float32x4_t maxes = vdupq_n_f32(0.0);
  int i = 0;
  for (; i < size; i += 8) {
    float32x4_t v1 = vld1q_f32(vals + i);
    float32x4_t v2 = vld1q_f32(vals + i + 4);
    mins = vminq_f32(v1, v2);
    maxes = vmaxq_f32(v1, v2);
  }
  min = vminvq_f32(mins);
  max = vmaxvq_f32(maxes);

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
