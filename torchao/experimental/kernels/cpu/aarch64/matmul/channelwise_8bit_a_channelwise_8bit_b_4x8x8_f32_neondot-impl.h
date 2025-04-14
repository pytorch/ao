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
namespace channelwise_8bit_a_channelwise_8bit_b_4x8x8_f32_neondot::internal {

TORCHAO_ALWAYS_INLINE static void block_mul_4x8x8(
    const int8_t* a,
    const size_t lda,
    const int8_t* b,
    int32x4_t (&partial_sums)[4][8 / 4],
    int32_t (&row_sum_a)[4],
    int32x4_t (&row_sum_b)[2]) {
  int8x8_t a_vec[4];
  a_vec[0] = vld1_s8(a + 0 * lda);
  a_vec[1] = vld1_s8(a + 1 * lda);
  a_vec[2] = vld1_s8(a + 2 * lda);
  a_vec[3] = vld1_s8(a + 3 * lda);
  int8x16_t ones = vdupq_n_s8(1);
  row_sum_a[0] = row_sum_a[0] + vaddlv_s8(a_vec[0]);
  row_sum_a[1] = row_sum_a[1] + vaddlv_s8(a_vec[1]);
  row_sum_a[2] = row_sum_a[2] + vaddlv_s8(a_vec[2]);
  row_sum_a[3] = row_sum_a[3] + vaddlv_s8(a_vec[3]);

  int8x16_t b_vec[2];
  b_vec[0] = vld1q_s8(b);
  b_vec[1] = vld1q_s8(b + 16);
  row_sum_b[0] = vdotq_s32(row_sum_b[0], b_vec[0], ones);
  row_sum_b[1] = vdotq_s32(row_sum_b[1], b_vec[1], ones);
  // First 4x4 of the 4x8 tile
  // Multiply with k = 0 thus (a_vec[0], 0) (a_vec[1], 0)...
  partial_sums[0][0] =
      vdotq_lane_s32(partial_sums[0][0], b_vec[0], a_vec[0], 0);
  partial_sums[1][0] =
      vdotq_lane_s32(partial_sums[1][0], b_vec[0], a_vec[1], 0);
  partial_sums[2][0] =
      vdotq_lane_s32(partial_sums[2][0], b_vec[0], a_vec[2], 0);
  partial_sums[3][0] =
      vdotq_lane_s32(partial_sums[3][0], b_vec[0], a_vec[3], 0);
  // Second 4x4 of the 4x8 til
  partial_sums[0][1] =
      vdotq_lane_s32(partial_sums[0][1], b_vec[1], a_vec[0], 0);
  partial_sums[1][1] =
      vdotq_lane_s32(partial_sums[1][1], b_vec[1], a_vec[1], 0);
  partial_sums[2][1] =
      vdotq_lane_s32(partial_sums[2][1], b_vec[1], a_vec[2], 0);
  partial_sums[3][1] =
      vdotq_lane_s32(partial_sums[3][1], b_vec[1], a_vec[3], 0);

  // Second set of 4 channels
  b = b + 32;
  b_vec[0] = vld1q_s8(b);
  b_vec[1] = vld1q_s8(b + 16);
  row_sum_b[0] = vdotq_s32(row_sum_b[0], b_vec[0], ones);
  row_sum_b[1] = vdotq_s32(row_sum_b[1], b_vec[1], ones);
  // First 4x4 of the 4x8 tile
  // Multiply with k = 0 thus (a_vec[0], 0) (a_vec[1], 0)...
  partial_sums[0][0] =
      vdotq_lane_s32(partial_sums[0][0], b_vec[0], a_vec[0], 1);
  partial_sums[1][0] =
      vdotq_lane_s32(partial_sums[1][0], b_vec[0], a_vec[1], 1);
  partial_sums[2][0] =
      vdotq_lane_s32(partial_sums[2][0], b_vec[0], a_vec[2], 1);
  partial_sums[3][0] =
      vdotq_lane_s32(partial_sums[3][0], b_vec[0], a_vec[3], 1);
  // Second 4x4 of the 4x8 til
  partial_sums[0][1] =
      vdotq_lane_s32(partial_sums[0][1], b_vec[1], a_vec[0], 1);
  partial_sums[1][1] =
      vdotq_lane_s32(partial_sums[1][1], b_vec[1], a_vec[1], 1);
  partial_sums[2][1] =
      vdotq_lane_s32(partial_sums[2][1], b_vec[1], a_vec[2], 1);
  partial_sums[3][1] =
      vdotq_lane_s32(partial_sums[3][1], b_vec[1], a_vec[3], 1);
}

TORCHAO_ALWAYS_INLINE static void dequantize_4x8_int32_t(
    int32x4_t (&sums)[4][8 / 4],
    int32_t (&row_sum_lhs)[4],
    int32x4_t (&row_sum_rhs)[2],
    const int8_t* lhs_zero_points,
    const int8_t* rhs_zero_points,
    const float* lhs_scales,
    const float* rhs_scales,
    const int32_t k,
    float32x4_t (&outputs)[4][8 / 4]) {
  int16x8_t rhs_zero_points_01234567 = vmovl_s8(vld1_s8(rhs_zero_points));
  int32x4_t rhs_zero_points_0123 =
      vmovl_s16(vget_low_s16(rhs_zero_points_01234567));
  int32x4_t rhs_zero_points_4567 =
      vmovl_s16(vget_high_s16(rhs_zero_points_01234567));
  int32x4_t row_sum_lhs_x_rhs_zp_0123 =
      vmulq_n_s32(rhs_zero_points_0123, row_sum_lhs[0]);
  int32x4_t row_sum_lhs_x_rhs_zp_4567 =
      vmulq_n_s32(rhs_zero_points_4567, row_sum_lhs[0]);
  // First 8 output channels adjustment
  sums[0][0] = vsubq_s32(sums[0][0], row_sum_lhs_x_rhs_zp_0123);
  sums[0][1] = vsubq_s32(sums[0][1], row_sum_lhs_x_rhs_zp_4567);

  // Add zp_rhs * zp_lhs * k
  int32x4_t zp_rhs_x_zp_lhs_0123 =
      vmulq_n_s32(rhs_zero_points_0123, k * (int32_t)lhs_zero_points[0]);
  int32x4_t zp_rhs_x_zp_lhs_4567 =
      vmulq_n_s32(rhs_zero_points_4567, k * (int32_t)lhs_zero_points[0]);
  sums[0][0] = vaddq_s32(sums[0][0], zp_rhs_x_zp_lhs_0123);
  sums[0][1] = vaddq_s32(sums[0][1], zp_rhs_x_zp_lhs_4567);

  row_sum_lhs_x_rhs_zp_0123 = vmulq_n_s32(rhs_zero_points_0123, row_sum_lhs[1]);
  row_sum_lhs_x_rhs_zp_4567 = vmulq_n_s32(rhs_zero_points_4567, row_sum_lhs[1]);
  // Second 8 output channels adjustment
  sums[1][0] = vsubq_s32(sums[1][0], row_sum_lhs_x_rhs_zp_0123);
  sums[1][1] = vsubq_s32(sums[1][1], row_sum_lhs_x_rhs_zp_4567);

  // Add zp_rhs * zp_lhs * k
  zp_rhs_x_zp_lhs_0123 =
      vmulq_n_s32(rhs_zero_points_0123, k * (int32_t)lhs_zero_points[1]);
  zp_rhs_x_zp_lhs_4567 =
      vmulq_n_s32(rhs_zero_points_4567, k * (int32_t)lhs_zero_points[1]);
  sums[1][0] = vaddq_s32(sums[1][0], zp_rhs_x_zp_lhs_0123);
  sums[1][1] = vaddq_s32(sums[1][1], zp_rhs_x_zp_lhs_4567);

  row_sum_lhs_x_rhs_zp_0123 = vmulq_n_s32(rhs_zero_points_0123, row_sum_lhs[2]);
  row_sum_lhs_x_rhs_zp_4567 = vmulq_n_s32(rhs_zero_points_4567, row_sum_lhs[2]);
  // Third 8 output channels adjustment
  sums[2][0] = vsubq_s32(sums[2][0], row_sum_lhs_x_rhs_zp_0123);
  sums[2][1] = vsubq_s32(sums[2][1], row_sum_lhs_x_rhs_zp_4567);

  // Add zp_rhs * zp_lhs * k
  zp_rhs_x_zp_lhs_0123 =
      vmulq_n_s32(rhs_zero_points_0123, k * (int32_t)lhs_zero_points[2]);
  zp_rhs_x_zp_lhs_4567 =
      vmulq_n_s32(rhs_zero_points_4567, k * (int32_t)lhs_zero_points[2]);
  sums[2][0] = vaddq_s32(sums[2][0], zp_rhs_x_zp_lhs_0123);
  sums[2][1] = vaddq_s32(sums[2][1], zp_rhs_x_zp_lhs_4567);

  row_sum_lhs_x_rhs_zp_0123 = vmulq_n_s32(rhs_zero_points_0123, row_sum_lhs[3]);
  row_sum_lhs_x_rhs_zp_4567 = vmulq_n_s32(rhs_zero_points_4567, row_sum_lhs[3]);
  // Fourth 8 output channels adjustment
  sums[3][0] = vsubq_s32(sums[3][0], row_sum_lhs_x_rhs_zp_0123);
  sums[3][1] = vsubq_s32(sums[3][1], row_sum_lhs_x_rhs_zp_4567);

  // Add zp_rhs * zp_lhs * k
  zp_rhs_x_zp_lhs_0123 =
      vmulq_n_s32(rhs_zero_points_0123, k * (int32_t)lhs_zero_points[3]);
  zp_rhs_x_zp_lhs_4567 =
      vmulq_n_s32(rhs_zero_points_4567, k * (int32_t)lhs_zero_points[3]);
  sums[3][0] = vaddq_s32(sums[3][0], zp_rhs_x_zp_lhs_0123);
  sums[3][1] = vaddq_s32(sums[3][1], zp_rhs_x_zp_lhs_4567);

  // Now adjust for rhs_zero_points * lhs_row_sum
  int32x4_t row_sum_rhs_0123_x_lhs_zp =
      vmulq_n_s32(row_sum_rhs[0], lhs_zero_points[0]);
  int32x4_t row_sum_rhs_4567_x_lhs_zp =
      vmulq_n_s32(row_sum_rhs[1], lhs_zero_points[0]);
  sums[0][0] = vsubq_s32(sums[0][0], row_sum_rhs_0123_x_lhs_zp);
  sums[0][1] = vsubq_s32(sums[0][1], row_sum_rhs_4567_x_lhs_zp);

  row_sum_rhs_0123_x_lhs_zp = vmulq_n_s32(row_sum_rhs[0], lhs_zero_points[1]);
  row_sum_rhs_4567_x_lhs_zp = vmulq_n_s32(row_sum_rhs[1], lhs_zero_points[1]);
  sums[1][0] = vsubq_s32(sums[1][0], row_sum_rhs_0123_x_lhs_zp);
  sums[1][1] = vsubq_s32(sums[1][1], row_sum_rhs_4567_x_lhs_zp);

  row_sum_rhs_0123_x_lhs_zp = vmulq_n_s32(row_sum_rhs[0], lhs_zero_points[2]);
  row_sum_rhs_4567_x_lhs_zp = vmulq_n_s32(row_sum_rhs[1], lhs_zero_points[2]);
  sums[2][0] = vsubq_s32(sums[2][0], row_sum_rhs_0123_x_lhs_zp);
  sums[2][1] = vsubq_s32(sums[2][1], row_sum_rhs_4567_x_lhs_zp);

  row_sum_rhs_0123_x_lhs_zp = vmulq_n_s32(row_sum_rhs[0], lhs_zero_points[3]);
  row_sum_rhs_4567_x_lhs_zp = vmulq_n_s32(row_sum_rhs[1], lhs_zero_points[3]);
  sums[3][0] = vsubq_s32(sums[3][0], row_sum_rhs_0123_x_lhs_zp);
  sums[3][1] = vsubq_s32(sums[3][1], row_sum_rhs_4567_x_lhs_zp);

  float32x4_t rhs_scales_0123 = vld1q_f32(rhs_scales);
  float32x4_t rhs_scales_4567 = vld1q_f32(rhs_scales + 4);

  float32x4_t scales_0123 = vmulq_n_f32(rhs_scales_0123, lhs_scales[0]);
  float32x4_t scales_4567 = vmulq_n_f32(rhs_scales_4567, lhs_scales[0]);

  outputs[0][0] = vmulq_f32(vcvtq_f32_s32(sums[0][0]), scales_0123);
  outputs[0][1] = vmulq_f32(vcvtq_f32_s32(sums[0][1]), scales_4567);

  scales_0123 = vmulq_n_f32(rhs_scales_0123, lhs_scales[1]);
  scales_4567 = vmulq_n_f32(rhs_scales_4567, lhs_scales[1]);
  outputs[1][0] = vmulq_f32(vcvtq_f32_s32(sums[1][0]), scales_0123);
  outputs[1][1] = vmulq_f32(vcvtq_f32_s32(sums[1][1]), scales_4567);

  scales_0123 = vmulq_n_f32(rhs_scales_0123, lhs_scales[2]);
  scales_4567 = vmulq_n_f32(rhs_scales_4567, lhs_scales[2]);
  outputs[2][0] = vmulq_f32(vcvtq_f32_s32(sums[2][0]), scales_0123);
  outputs[2][1] = vmulq_f32(vcvtq_f32_s32(sums[2][1]), scales_4567);

  scales_0123 = vmulq_n_f32(rhs_scales_0123, lhs_scales[3]);
  scales_4567 = vmulq_n_f32(rhs_scales_4567, lhs_scales[3]);
  outputs[3][0] = vmulq_f32(vcvtq_f32_s32(sums[3][0]), scales_0123);
  outputs[3][1] = vmulq_f32(vcvtq_f32_s32(sums[3][1]), scales_4567);
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
    std::vector<int8_t> lhs_zero_points_transposed;
    std::vector<float> lhs_scales_transposed;
    if (lhs_qparams_stride > 1) {
      lhs_zero_points_transposed.resize(m);
      lhs_scales_transposed.resize(m);
      utils::transpose_scales_and_zero_points(
          lhs_zero_points,
          lhs_scales,
          lhs_zero_points_transposed.data(),
          lhs_scales_transposed.data(),
          m,
          lhs_qparams_stride);
      lhs_zero_points = lhs_zero_points_transposed.data();
      lhs_scales = lhs_scales_transposed.data();
    }
    std::vector<int8_t> rhs_zero_points_transposed;
    std::vector<float> rhs_scales_transposed;
    if (rhs_qparams_stride > 1) {
      rhs_zero_points_transposed.resize(n);
      rhs_scales_transposed.resize(n);
      utils::transpose_scales_and_zero_points(
          rhs_zero_points,
          rhs_scales,
          rhs_zero_points_transposed.data(),
          rhs_scales_transposed.data(),
          n,
          rhs_qparams_stride);
      rhs_zero_points = rhs_zero_points_transposed.data();
      rhs_scales = rhs_scales_transposed.data();
    }

    constexpr int mr = 4;
    constexpr int nr = 8;
    constexpr int kr = 8;
    assert(m % mr == 0);
    assert(k % kr == 0);
    assert(n >= nr);
    std::vector<int8_t> rhs_packed(n * k);
    // Since we are casting int8_t to float32_t in order to tranpose matrix in a
    // way to keep 4 of the k values to gether, we must adjust stride as well as
    // k size
    const size_t k_adjusted = k / 4;
    const size_t rhs_stride_n_adjusted = rhs_stride_n / 4;
    utils::pack_kxn_b_matrix_for_mx8_dotprod_ukernel(
        static_cast<const float*>(rhs),
        rhs_stride_n_adjusted,
        reinterpret_cast<float*>(rhs_packed.data()),
        n,
        k_adjusted);
    size_t packed_block_stride = nr * k;
    constexpr size_t packed_k_stride = nr * kr;

    for (int m_idx = 0; m_idx < m; m_idx += mr) {
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
        const int8_t* rhs_ptr = (const int8_t*)rhs_packed.data() +
            (n_idx / nr) * packed_block_stride;
        int32x4_t int32_sums[mr][nr / 4] = {{vdupq_n_s32(0)}};
        int32x4_t row_sum_rhs_vec[nr / 4] = {vdupq_n_s32(0)};
        int32_t row_sum_lhs[mr] = {0};

        // Loop k_idx by group
        int k_idx = 0;
        for (; k_idx < k; k_idx += kr) {
          block_mul_4x8x8(
              lhs_ptr,
              lhs_stride_m,
              rhs_ptr,
              int32_sums,
              row_sum_lhs,
              row_sum_rhs_vec);
          lhs_ptr += kr;
          rhs_ptr += packed_k_stride;
        }

        float32x4_t res[mr][nr / 4];
        dequantize_4x8_int32_t(
            int32_sums,
            row_sum_lhs,
            row_sum_rhs_vec,
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
        vst1q_f32(store_loc, res[0][0]);
        vst1q_f32(store_loc + 4, res[0][1]);
        store_loc += out_stride_m;
        vst1q_f32(store_loc, res[1][0]);
        vst1q_f32(store_loc + 4, res[1][1]);
        store_loc += out_stride_m;
        vst1q_f32(store_loc, res[2][0]);
        vst1q_f32(store_loc + 4, res[2][1]);
        store_loc += out_stride_m;
        vst1q_f32(store_loc, res[3][0]);
        vst1q_f32(store_loc + 4, res[3][1]);
      } // n_idx
    } // m_idx
  }
};

} // namespace
  // channelwise_8bit_a_channelwise_8bit_b_4x8x8_f32_neondot::internal

namespace channelwise_8bit_a_channelwise_8bit_b_4x8x8_f32_neondot {
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
      channelwise_8bit_a_channelwise_8bit_b_4x8x8_f32_neondot::internal::
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
} // namespace channelwise_8bit_a_channelwise_8bit_b_4x8x8_f32_neondot
} // namespace torchao::kernels::cpu::aarch64::quantized_matmul

#endif // defined(__aarch64__) && defined(__ARM_NEON)
