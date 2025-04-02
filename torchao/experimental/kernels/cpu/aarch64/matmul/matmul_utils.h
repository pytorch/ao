// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>

namespace torchao::kernels::cpu::aarch64::quantized_matmul {
namespace utils {

TORCHAO_ALWAYS_INLINE static void transpose_scales_and_zero_points(
    const int8_t* zero_points,
    const float* scales,
    int8_t* zero_points_transposed,
    float* scales_transposed,
    const int m,
    const int stride_m) {
  // Process 8 elements at a time using NEON
  int i = 0;
  for (; i + 8 <= m; i += 8) {
    // Load 8 zero points with stride_m
    int8x8_t zp = {
        zero_points[0 * stride_m],
        zero_points[1 * stride_m],
        zero_points[2 * stride_m],
        zero_points[3 * stride_m],
        zero_points[4 * stride_m],
        zero_points[5 * stride_m],
        zero_points[6 * stride_m],
        zero_points[7 * stride_m]};
    zero_points += 8 * stride_m;
    // Store contiguously
    vst1_s8(zero_points_transposed + i, zp);

    // Load 8 scales with stride_m
    float32x4_t scales_lo = {
        scales[0 * stride_m],
        scales[1 * stride_m],
        scales[2 * stride_m],
        scales[3 * stride_m]};
    float32x4_t scales_hi = {
        scales[4 * stride_m],
        scales[5 * stride_m],
        scales[6 * stride_m],
        scales[7 * stride_m]};
    scales += 8 * stride_m;
    // Store contiguously
    vst1q_f32(scales_transposed + i, scales_lo);
    vst1q_f32(scales_transposed + i + 4, scales_hi);
  }

  // Handle remaining elements
  for (; i < m; i++) {
    zero_points_transposed[i] = zero_points[0];
    scales_transposed[i] = scales[0];
    zero_points += stride_m;
    scales += stride_m;
  }
}

} // namespace utils
} // namespace torchao::kernels::cpu::aarch64::quantized_matmul

#endif // defined(__aarch64__) || defined(__ARM_NEON)
