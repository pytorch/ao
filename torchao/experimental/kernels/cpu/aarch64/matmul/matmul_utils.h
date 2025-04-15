// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <cassert>
#include <cstddef>

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

void transpose_4x4(
    const float32_t* a,
    const size_t lda,
    float32x4_t (&tranposed)[4]);

TORCHAO_ALWAYS_INLINE void transpose_4x4(
    const float32_t* a,
    const size_t lda,
    float32x4_t (&tranposed)[4]) {
  float32x4_t a_vec_0 = vld1q_f32(a + 0 * lda);
  float32x4_t a_vec_1 = vld1q_f32(a + 1 * lda);
  float32x4_t a_vec_2 = vld1q_f32(a + 2 * lda);
  float32x4_t a_vec_3 = vld1q_f32(a + 3 * lda);
  // Transpose the 4x4 matrix formed by a_vec_0, a_vec_1, a_vec_2, a_vec_3
  float32x4x2_t a01 = vtrnq_f32(a_vec_0, a_vec_1);
  float32x4x2_t a23 = vtrnq_f32(a_vec_2, a_vec_3);

  float32x4_t a_vec_0_t =
      vcombine_f32(vget_low_f32(a01.val[0]), vget_low_f32(a23.val[0]));
  float32x4_t a_vec_1_t =
      vcombine_f32(vget_low_f32(a01.val[1]), vget_low_f32(a23.val[1]));
  float32x4_t a_vec_2_t =
      vcombine_f32(vget_high_f32(a01.val[0]), vget_high_f32(a23.val[0]));
  float32x4_t a_vec_3_t =
      vcombine_f32(vget_high_f32(a01.val[1]), vget_high_f32(a23.val[1]));

  tranposed[0] = a_vec_0_t;
  tranposed[1] = a_vec_1_t;
  tranposed[2] = a_vec_2_t;
  tranposed[3] = a_vec_3_t;
}

void pack_kxn_b_matrix_for_mx8_dotprod_ukernel(
    const float32_t* a,
    const size_t lda,
    float32_t* b,
    const size_t n,
    const size_t k);

// Really dong what xnnpack is doing
void pack_kxn_b_matrix_for_mx8_dotprod_ukernel(
    const float32_t* a,
    const size_t lda,
    float32_t* b,
    const size_t n,
    const size_t k) {
  assert(n % 8 == 0);
  assert(k % 4 == 0);
  // Transpose the matrix in 4x4 blocks
  size_t packed_block_stride = 8 * k;
  constexpr size_t block_stride_8x4 = 8 * 4;
  for (size_t i = 0; i < n; i += 8) {
    float32_t* b_ptr = b + (i / 8) * packed_block_stride;
    for (size_t j = 0; j < k; j += 4) {
      // Get the transposed 4x4 block
      float32x4_t transposed_block0[4];
      float32x4_t transposed_block1[4];
      // This transposes the a[i: i + 4, j: j + 4]
      // Thus tranposed_block0[0] = a[j: i: i + 4]
      // Thus tranposed_block0[1] = a[j + 1: i: i + 4]
      transpose_4x4(a + (i + 0) * lda + j, lda, transposed_block0);
      // This transposes the a[i + 4: i + 8, j: j + 4]
      // Thus tranposed_block1[0] = a[j: i + 4 : i + 8]
      // Thus tranposed_block1[1] = a[j + 1: i + 4 : i + 8]
      transpose_4x4(a + (i + 4) * lda + j, lda, transposed_block1);

      // Once you have 8x4 matrix of 32bit values transposed
      // Store them by writing two adjucent 1x4 blocks so that
      // all of the 8 values from n dim are together.
      // Then pack the next set of k values.
      float32_t* b_ptr_local = b_ptr + (j / 4) * block_stride_8x4;
#pragma unroll(4)
      for (size_t ki = 0; ki < 4; ki++) {
        float32_t* b_ptr_local_k = b_ptr_local + ki * 8;
        vst1q_f32(b_ptr_local_k, transposed_block0[ki]);
        vst1q_f32(
            b_ptr_local_k + sizeof(float32x4_t) / 4, transposed_block1[ki]);
      }
    }
  }
}
} // namespace utils
} // namespace torchao::kernels::cpu::aarch64::quantized_matmul

#endif // defined(__aarch64__) || defined(__ARM_NEON)
