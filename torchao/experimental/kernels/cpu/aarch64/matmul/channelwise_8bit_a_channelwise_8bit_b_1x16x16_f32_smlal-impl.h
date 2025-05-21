// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) && defined(__ARM_NEON)

#include <algorithm>
#include <cassert>
#include <cstring>

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <torchao/experimental/kernels/cpu/aarch64/matmul/matmul_utils.h>

namespace torchao::kernels::cpu::aarch64::quantized_matmul {
namespace channelwise_8bit_a_channelwise_8bit_b_1x16x16_f32_smlal::internal {

namespace {
/*
This function loads int8x16_t value from a, and 8 int8x16_t values from b.
For each int8x16_t of b:
- subl to subtarct a_zero_point from a, to get a_low, a_high
- 4 int32x4 accumulated values
- for i in [0, 8]:
  - load b[i]
  - subl to subtarct b_zero_point from b, to get b_low, b_high
  - smlal_lane to multiply a_low[i] and b_low_low.
  - smlal_lane to multiply a_low[i] and b_low_high.
  - smlal_lane to multiply a_low[i] and b_high_low.
  - smlal_lane to multiply a_low[i] and b_high_high.
  - This produces 2 int32x4_t values
- for i in [0, 8]:
  - load b[i]
  - subl to subtarct b_zero_point from b, to get b_low, b_high
  - smlal_lane to multiply a_low[i] and b_low_low.
  - smlal_lane to multiply a_low[i] and b_low_high.
  - smlal_lane to multiply a_low[i] and b_high_low.
  - smlal_lane to multiply a_low[i] and b_high_high.
  - This produces 2 int32x4_t values
Possibly better to transpose 16x16 of b and use dotprod. Left for future.
*/

template <int lane>
TORCHAO_ALWAYS_INLINE inline void block_mul_1x16x1(
    const int16x4_t& a_vec,
    const int8x16_t& b_vec,
    const int8x16_t& b_zero_point_vec,
    int32x4_t (&partial_sums)[4]) {
  int16x8_t b_vec_low =
      vsubl_s8(vget_low_s8(b_vec), vget_low_s8(b_zero_point_vec));
  int16x8_t b_vec_high =
      vsubl_s8(vget_high_s8(b_vec), vget_high_s8(b_zero_point_vec));
  partial_sums[0] =
      vmlal_lane_s16(partial_sums[0], vget_low_s16(b_vec_low), a_vec, lane);
  partial_sums[1] =
      vmlal_lane_s16(partial_sums[1], vget_high_s16(b_vec_low), a_vec, lane);
  partial_sums[2] =
      vmlal_lane_s16(partial_sums[2], vget_low_s16(b_vec_high), a_vec, lane);
  partial_sums[3] =
      vmlal_lane_s16(partial_sums[3], vget_high_s16(b_vec_high), a_vec, lane);
}

void block_mul_1x16x16(
    const int8_t* a,
    const int8_t* b,
    const size_t ldb,
    const int8_t a_zero_point,
    const int8_t* b_zero_point,
    int32x4_t (&partial_sums)[4]) {
  int8x16_t a_vec = vld1q_s8(a);
  int8x8_t a_zero_point_vec = vdup_n_s8(a_zero_point);
  int8x16_t b_zero_point_vec = vld1q_s8(b_zero_point);
  int16x8_t a_vec_low = vsubl_s8(vget_low_s8(a_vec), a_zero_point_vec);
  int16x8_t a_vec_high = vsubl_s8(vget_high_s8(a_vec), a_zero_point_vec);

  int8x16_t b_vec = vld1q_s8(b + 0 * ldb);
  block_mul_1x16x1<0>(
      vget_low_s16(a_vec_low), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 1 * ldb);
  block_mul_1x16x1<1>(
      vget_low_s16(a_vec_low), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 2 * ldb);
  block_mul_1x16x1<2>(
      vget_low_s16(a_vec_low), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 3 * ldb);
  block_mul_1x16x1<3>(
      vget_low_s16(a_vec_low), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 4 * ldb);
  block_mul_1x16x1<0>(
      vget_high_s16(a_vec_low), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 5 * ldb);
  block_mul_1x16x1<1>(
      vget_high_s16(a_vec_low), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 6 * ldb);
  block_mul_1x16x1<2>(
      vget_high_s16(a_vec_low), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 7 * ldb);
  block_mul_1x16x1<3>(
      vget_high_s16(a_vec_low), b_vec, b_zero_point_vec, partial_sums);

  // Second set of 8 channels
  b_vec = vld1q_s8(b + 8 * ldb);
  block_mul_1x16x1<0>(
      vget_low_s16(a_vec_high), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 9 * ldb);
  block_mul_1x16x1<1>(
      vget_low_s16(a_vec_high), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 10 * ldb);
  block_mul_1x16x1<2>(
      vget_low_s16(a_vec_high), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 11 * ldb);
  block_mul_1x16x1<3>(
      vget_low_s16(a_vec_high), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 12 * ldb);
  block_mul_1x16x1<0>(
      vget_high_s16(a_vec_high), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 13 * ldb);
  block_mul_1x16x1<1>(
      vget_high_s16(a_vec_high), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 14 * ldb);
  block_mul_1x16x1<2>(
      vget_high_s16(a_vec_high), b_vec, b_zero_point_vec, partial_sums);
  b_vec = vld1q_s8(b + 15 * ldb);
  block_mul_1x16x1<3>(
      vget_high_s16(a_vec_high), b_vec, b_zero_point_vec, partial_sums);
}

TORCHAO_ALWAYS_INLINE inline void dequantize_1x16_int32_t(
    const int32x4_t (&sums)[4],
    const float* lhs_scales,
    const float* rhs_scales,
    float32x4_t (&outputs)[4]) {
  float32x4_t scales_0123 = vmulq_n_f32(vld1q_f32(rhs_scales), lhs_scales[0]);
  float32x4_t scales_4567 =
      vmulq_n_f32(vld1q_f32(rhs_scales + 4), lhs_scales[0]);
  float32x4_t scales_89ab =
      vmulq_n_f32(vld1q_f32(rhs_scales + 8), lhs_scales[0]);
  float32x4_t scales_cdef =
      vmulq_n_f32(vld1q_f32(rhs_scales + 12), lhs_scales[0]);

  outputs[0] = vmulq_f32(vcvtq_f32_s32(sums[0]), scales_0123);
  outputs[1] = vmulq_f32(vcvtq_f32_s32(sums[1]), scales_4567);
  outputs[2] = vmulq_f32(vcvtq_f32_s32(sums[2]), scales_89ab);
  outputs[3] = vmulq_f32(vcvtq_f32_s32(sums[3]), scales_cdef);
}

template <
    bool a_has_zeros,
    bool b_has_zeros,
    bool a_transposed,
    bool b_transposed>
struct KernelImpl {
  static void run(
      int m,
      int n,
      int k,
      const void* lhs,
      int lhs_stride_m,
      const void* rhs,
      int rhs_stride_n,
      float32_t* output,
      int out_stride_m,
      const int8_t* lhs_zero_points,
      const int8_t* rhs_zero_points,
      const float* lhs_scales,
      const float* rhs_scales,
      const int lhs_qparams_stride,
      const int rhs_qparams_stride);
};

template <>
struct KernelImpl<true, true, false, false> {
  /**
   * @brief Implements quantized matrix multiplication for 8-bit channelwise
   * quantized matrices
   *
   * This specialized implementation handles the case where:
   * - Both LHS and RHS have zero points (true, true)
   * - Neither LHS nor RHS are transposed (false, false)
   *
   * The function performs a quantized matrix multiplication C = A * B where:
   * - A is an m×k matrix (LHS)
   * - B is a k×n matrix (RHS)
   * - C is an m×n matrix (output)
   *
   * The implementation uses NEON intrinsics for vectorized computation and
   * processes data in blocks of 16×16 for optimal performance on ARM
   * architecture.
   *
   * @param m Number of rows in LHS and output
   * @param n Number of columns in RHS and output
   * @param k Number of columns in LHS and rows in RHS
   * @param lhs Pointer to LHS matrix data (int8_t)
   * @param lhs_stride_m Stride between rows of LHS
   * @param rhs Pointer to RHS matrix data (int8_t)
   * @param rhs_stride_n Stride between rows of RHS
   * @param output Pointer to output matrix (float32_t)
   * @param out_stride_m Stride between rows of output
   * @param lhs_zero_points Zero points for LHS quantization (per-channel)
   * @param rhs_zero_points Zero points for RHS quantization (per-channel)
   * @param lhs_scales Scales for LHS quantization (per-channel)
   * @param rhs_scales Scales for RHS quantization (per-channel)
   * @param lhs_qparams_stride Stride for LHS quantization parameters
   * @param rhs_qparams_stride Stride for RHS quantization parameters
   */
  static void run(
      int m,
      int n,
      int k,
      const void* lhs,
      int lhs_stride_m,
      const void* rhs,
      int rhs_stride_n,
      float32_t* output,
      int out_stride_m,
      const int8_t* lhs_zero_points,
      const int8_t* rhs_zero_points,
      const float* lhs_scales,
      const float* rhs_scales,
      const int lhs_qparams_stride,
      const int rhs_qparams_stride) {
    // If lhs_zero_points and rhs_zero_points are not contiguous, transpose
    std::unique_ptr<int8_t[]> lhs_zero_points_transposed =
        std::make_unique<int8_t[]>(m);
    std::unique_ptr<float[]> lhs_scales_transposed =
        std::make_unique<float[]>(m);
    if (lhs_qparams_stride > 1) {
      utils::transpose_scales_and_zero_points(
          lhs_zero_points,
          lhs_scales,
          lhs_zero_points_transposed.get(),
          lhs_scales_transposed.get(),
          m,
          lhs_qparams_stride);
      lhs_zero_points = lhs_zero_points_transposed.get();
      lhs_scales = lhs_scales_transposed.get();
    }
    std::unique_ptr<int8_t[]> rhs_zero_points_transposed =
        std::make_unique<int8_t[]>(n);
    std::unique_ptr<float[]> rhs_scales_transposed =
        std::make_unique<float[]>(n);
    if (rhs_qparams_stride > 1) {
      utils::transpose_scales_and_zero_points(
          rhs_zero_points,
          rhs_scales,
          rhs_zero_points_transposed.get(),
          rhs_scales_transposed.get(),
          n,
          rhs_qparams_stride);
      rhs_zero_points = rhs_zero_points_transposed.get();
      rhs_scales = rhs_scales_transposed.get();
    }

    for (int m_idx = 0; m_idx < m; m_idx++) {
      // Loop over 16 cols at a time
      // Access to partial tiles must be protected:w
      constexpr int nr = 16;
      constexpr int kr = 16;
      assert(n >= nr);
      for (int n_idx = 0; n_idx < n; n_idx += nr) {
        // If remaining is < nr, that must mean that (nr - remaining) items
        // dont need to be computed.
        // In order to avoid out-of-bounds access, we need to rewind n_indx a
        // bit
        // |-------------------|-------------------|
        // 0-------------------8-------------------16
        // 0-------------------8-----10
        // If n = 10 and nr = 8 then at n_idx = 8, we need to rewind n_idx to
        // 8 - (8 - 10) = 2
        int remaining = std::min(n - n_idx, nr);
        n_idx = n_idx - (nr - remaining);
        // Set activation_ptr to start of activation qvals for row m_idx
        const int8_t* lhs_ptr = (const int8_t*)lhs + m_idx * lhs_stride_m;
        const int8_t* rhs_ptr = (const int8_t*)rhs + n_idx;
        int32x4_t int32_sums[nr / 4] = {vdupq_n_s32(0)};

        // Loop k_idx by group
        int k_idx = 0;
        for (; (k_idx + kr) <= k; k_idx += kr) {
          block_mul_1x16x16(
              lhs_ptr,
              rhs_ptr,
              rhs_stride_n,
              lhs_zero_points[m_idx],
              rhs_zero_points + n_idx,
              int32_sums);
          lhs_ptr += kr;
          rhs_ptr += kr * rhs_stride_n;
        }

        int8x16_t b_zero_point_vec = vld1q_s8(rhs_zero_points + n_idx);
        for (int ki = 0; ki < (k - k_idx); ++ki) {
          // For each of the remaining k values
          // Load 1 int8_t from lhs
          // Load 16 int8_t from rhs
          // And multiply + add into the 16 accumulators
          // arranged as int32x4_t[4]
          int16_t a_val = static_cast<int16_t>(lhs_ptr[ki]) -
              static_cast<int16_t>(lhs_zero_points[m_idx]);
          int8x16_t b_vec = vld1q_s8(rhs_ptr + ki * rhs_stride_n);
          int16x8_t b_vec_low =
              vsubl_s8(vget_low_s8(b_vec), vget_low_s8(b_zero_point_vec));
          int16x8_t b_vec_high =
              vsubl_s8(vget_high_s8(b_vec), vget_high_s8(b_zero_point_vec));
          int32_sums[0] =
              vmlal_n_s16(int32_sums[0], vget_low_s16(b_vec_low), a_val);
          int32_sums[1] =
              vmlal_n_s16(int32_sums[1], vget_high_s16(b_vec_low), a_val);
          int32_sums[2] =
              vmlal_n_s16(int32_sums[2], vget_low_s16(b_vec_high), a_val);
          int32_sums[3] =
              vmlal_n_s16(int32_sums[3], vget_high_s16(b_vec_high), a_val);
        }

        float32x4_t res[4];
        dequantize_1x16_int32_t(
            int32_sums, lhs_scales + m_idx, rhs_scales + n_idx, res);

        // Store result
        // Because we adjust n_idx, we may end up writing the same location
        // twice
        float* store_loc = output + m_idx * out_stride_m + n_idx;
        vst1q_f32(store_loc, res[0]);
        vst1q_f32(store_loc + 4, res[1]);
        vst1q_f32(store_loc + 8, res[2]);
        vst1q_f32(store_loc + 12, res[3]);
      } // n_idx
    } // m_idx
  }
};

} // namespace

} // namespace channelwise_8bit_a_channelwise_8bit_b_1x16x16_f32_smlal::internal

namespace channelwise_8bit_a_channelwise_8bit_b_1x16x16_f32_smlal {
template <
    bool a_has_zeros,
    bool b_has_zeros,
    bool a_transposed,
    bool b_transposed>
void kernel(
    int m,
    int n,
    int k,
    const void* lhs,
    int lhs_stride_m,
    const void* rhs,
    int rhs_stride_n,
    float32_t* output,
    int out_stride_m,
    const int8_t* lhs_zero_points,
    const int8_t* rhs_zero_points,
    const float* lhs_scales,
    const float* rhs_scales,
    const int lhs_qparams_stride,
    const int rhs_qparams_stride) {
  torchao::kernels::cpu::aarch64::quantized_matmul::
      channelwise_8bit_a_channelwise_8bit_b_1x16x16_f32_smlal::internal::
          KernelImpl<a_has_zeros, b_has_zeros, a_transposed, b_transposed>::run(
              m,
              n,
              k,
              lhs,
              lhs_stride_m,
              rhs,
              rhs_stride_n,
              output,
              out_stride_m,
              lhs_zero_points,
              rhs_zero_points,
              lhs_scales,
              rhs_scales,
              lhs_qparams_stride,
              rhs_qparams_stride);
}
} // namespace channelwise_8bit_a_channelwise_8bit_b_1x16x16_f32_smlal
} // namespace torchao::kernels::cpu::aarch64::quantized_matmul

#endif // defined(__aarch64__) && defined(__ARM_NEON)
