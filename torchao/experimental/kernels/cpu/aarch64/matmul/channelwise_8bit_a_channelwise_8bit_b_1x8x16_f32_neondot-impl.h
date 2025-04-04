// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <algorithm>
#include <cassert>
#include <cstring>

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <torchao/experimental/kernels/cpu/aarch64/matmul/matmul_utils.h>

namespace torchao::kernels::cpu::aarch64::quantized_matmul {
namespace channelwise_8bit_a_channelwise_8bit_b_1x8x16_f32_neondot::internal {

/*
This function loads int8x16_t value from a, and 8 int8x16_t values from b, and
computes 8 dot products, resulting in 8 int32x4_t values.
Furthermore the int8x16_t values from a are reduced via summing, resulting in
int32_t row_sum_a. Similar int8x16_t values from b are reduced via summing,
resulting in int32_t row_sum_b.
*/
TORCHAO_ALWAYS_INLINE static void block_mul_1x8x16(
    const int8_t* a,
    const int8_t* b,
    const size_t ldb,
    int32x4_t (&partial_sums)[8],
    int32_t& row_sum_a,
    int32_t (&row_sum_b)[8]) {
  int8x16_t a_vec = vld1q_s8(a);
  row_sum_a = row_sum_a + vaddlvq_s8(a_vec);

// godbolt (https://godbolt.org/z/9vbq1d1qY) shows this loops doesnt quantize
// get optimized by moving all the loads up in the unrolled loop. Just hoping
// OOO machine will take care of things Late replace this with macros so as to
// deconstruct the loop and do manual optimization. Or just write assembly.
#pragma unroll(8)
  for (int i = 0; i < 8; ++i) {
    int8x16_t b_vec = vld1q_s8(b + i * ldb);
    row_sum_b[i] = row_sum_b[i] + vaddlvq_s8(b_vec);
    partial_sums[i] = vdotq_s32(partial_sums[i], a_vec, b_vec);
  }
}

TORCHAO_ALWAYS_INLINE static void reduce_1x8_int32x4_t_sums(
    const int32x4_t (&partial_sums)[8],
    int32_t (&sums)[8]) {
#pragma unroll(8)
  for (int i = 0; i < 8; ++i) {
    sums[i] = vaddvq_s32(partial_sums[i]);
  }
}

TORCHAO_ALWAYS_INLINE static void dequantize_1x8_int32_t(
    const int32_t (&sums)[8],
    int32_t& row_sum_lhs,
    int32_t (&row_sum_rhs)[8],
    const int8_t* lhs_zero_points,
    const int8_t* rhs_zero_points,
    const float* lhs_scales,
    const float* rhs_scales,
    const int32_t k,
    float32x4x2_t& outputs) {
  int32x4_t vec_sum_0123 = vld1q_s32(sums);
  int32x4_t vec_sum_4567 = vld1q_s32(sums + 4);

  int32x4_t row_sum_rhs_x_lhs_zp_0123 =
      vmulq_n_s32(vld1q_s32(row_sum_rhs), (int32_t)lhs_zero_points[0]);
  int32x4_t row_sum_rhs_x_lhs_zp_4567 =
      vmulq_n_s32(vld1q_s32(row_sum_rhs + 4), (int32_t)lhs_zero_points[0]);

  // Extract rhs zero point in int8x8_t and convert to int32x4_t
  int16x8_t rhs_zero_points_vec_01234567 = vmovl_s8(vld1_s8(rhs_zero_points));
  int32x4_t rhs_zero_points_vec_0123 =
      vmovl_s16(vget_low_s16(rhs_zero_points_vec_01234567));
  int32x4_t rhs_zero_points_vec_4567 =
      vmovl_s16(vget_high_s16(rhs_zero_points_vec_01234567));
  int32x4_t row_sum_lhs_x_rhs_zp_0123 =
      vmulq_n_s32(rhs_zero_points_vec_0123, row_sum_lhs);
  int32x4_t row_sum_lhs_x_rhs_zp_4567 =
      vmulq_n_s32(rhs_zero_points_vec_4567, row_sum_lhs);

  int32x4_t zp_rhs_x_zp_lhs_0123 =
      vmulq_n_s32(rhs_zero_points_vec_0123, k * (int32_t)lhs_zero_points[0]);
  int32x4_t zp_rhs_x_zp_lhs_4567 =
      vmulq_n_s32(rhs_zero_points_vec_4567, k * (int32_t)lhs_zero_points[0]);

  vec_sum_0123 = vsubq_s32(vec_sum_0123, row_sum_rhs_x_lhs_zp_0123);
  vec_sum_0123 = vsubq_s32(vec_sum_0123, row_sum_lhs_x_rhs_zp_0123);
  vec_sum_0123 = vaddq_s32(vec_sum_0123, zp_rhs_x_zp_lhs_0123);

  vec_sum_4567 = vsubq_s32(vec_sum_4567, row_sum_rhs_x_lhs_zp_4567);
  vec_sum_4567 = vsubq_s32(vec_sum_4567, row_sum_lhs_x_rhs_zp_4567);
  vec_sum_4567 = vaddq_s32(vec_sum_4567, zp_rhs_x_zp_lhs_4567);

  float32x4_t scales_0123 = vmulq_n_f32(vld1q_f32(rhs_scales), lhs_scales[0]);
  float32x4_t scales_4567 =
      vmulq_n_f32(vld1q_f32(rhs_scales + 4), lhs_scales[0]);

  outputs.val[0] = vmulq_f32(vcvtq_f32_s32(vec_sum_0123), scales_0123);
  outputs.val[1] = vmulq_f32(vcvtq_f32_s32(vec_sum_4567), scales_4567);
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
struct KernelImpl<true, true, false, true> {
  /**
   * @brief Executes a quantized matrix multiplication with channelwise
   * quantization parameters
   *
   * This function performs matrix multiplication between two 8-bit quantized
   * matrices with per-channel quantization parameters. It handles the following
   * operations:
   * 1. Transposes quantization parameters if they're not contiguous
   * 2. Processes the matrices in blocks of 8 columns at a time
   * 3. Uses NEON dot product instructions for efficient computation
   * 4. Handles edge cases for remaining elements
   * 5. Dequantizes the results to floating point
   *
   * @param m Number of rows in the output matrix
   * @param n Number of columns in the output matrix
   * @param k Number of columns in lhs / rows in rhs
   * @param lhs Pointer to the left-hand side matrix (quantized int8)
   * @param lhs_stride_m Stride between rows of the lhs matrix
   * @param rhs Pointer to the right-hand side matrix (quantized int8)
   * @param rhs_stride_n Stride between rows of the rhs matrix. Expects matrix
   * to be transposed. Thus of size [n x k]
   * @param output Pointer to the output matrix (float32)
   * @param out_stride_m Stride between rows of the output matrix
   * @param lhs_zero_points Zero points for lhs quantization (per-channel)
   * @param rhs_zero_points Zero points for rhs quantization (per-channel)
   * @param lhs_scales Scales for lhs quantization (per-channel)
   * @param rhs_scales Scales for rhs quantization (per-channel)
   * @param lhs_qparams_stride Stride for lhs quantization parameters
   * @param rhs_qparams_stride Stride for rhs quantization parameters
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
      // Loop over 8 cols at a time
      // Access to partial tiles must be protected:w
      constexpr int nr = 8;
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
        const int8_t* rhs_ptr = (const int8_t*)rhs + n_idx * rhs_stride_n;
        int32x4_t int32_sums[nr] = {vdupq_n_s32(0)};
        int32_t row_sum_lhs = 0;
        int32_t row_sum_rhs[nr] = {0, 0, 0, 0, 0, 0, 0, 0};
        int32_t sums[nr];

        // Loop k_idx by group
        int k_idx = 0;
        for (; (k_idx + kr) <= k; k_idx += kr) {
          block_mul_1x8x16(
              lhs_ptr,
              rhs_ptr,
              rhs_stride_n,
              int32_sums,
              row_sum_lhs,
              row_sum_rhs);
          lhs_ptr += kr;
          rhs_ptr += kr;
        }

        reduce_1x8_int32x4_t_sums(int32_sums, sums);
        for (int ki = 0; ki < (k - k_idx); ++ki) {
          row_sum_lhs += (int32_t)lhs_ptr[ki];
        }
        for (int ni = 0; ni < nr; ++ni) {
          for (int ki = 0; ki < (k - k_idx); ++ki) {
            sums[ni] += (int32_t)lhs_ptr[ki] *
                (int32_t)(rhs_ptr + ni * rhs_stride_n)[ki];
            row_sum_rhs[ni] += (int32_t)(rhs_ptr + ni * rhs_stride_n)[ki];
          }
        }

        float32x4x2_t res;
        dequantize_1x8_int32_t(
            sums,
            row_sum_lhs,
            row_sum_rhs,
            lhs_zero_points + m_idx,
            rhs_zero_points + n_idx,
            lhs_scales + m_idx,
            rhs_scales + n_idx,
            k,
            res);

        // Store result
        // Because we adjust n_idx, we may end up writing the same location
        // twice
        float* store_loc = output + m_idx * out_stride_m + n_idx;
        vst1q_f32(store_loc, res.val[0]);
        vst1q_f32(store_loc + 4, res.val[1]);
      } // n_idx
    } // m_idx
  }
};

} // namespace
  // channelwise_8bit_a_channelwise_8bit_b_1x8x16_f32_neondot::internal

namespace channelwise_8bit_a_channelwise_8bit_b_1x8x16_f32_neondot {
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
      channelwise_8bit_a_channelwise_8bit_b_1x8x16_f32_neondot::internal::
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
} // namespace channelwise_8bit_a_channelwise_8bit_b_1x8x16_f32_neondot
} // namespace torchao::kernels::cpu::aarch64::quantized_matmul

#endif // defined(__aarch64__) || defined(__ARM_NEON)
