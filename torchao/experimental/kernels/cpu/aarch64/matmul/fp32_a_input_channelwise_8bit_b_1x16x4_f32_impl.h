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
namespace fp32_a_input_channelwise_8bit_b_1x16x4_f32::internal {

namespace {

/*
This function loads float32x4_t value from a, and 16 int8x16_t values from b.
For each int8x16_t of b:
- 4 float32x4 accumulated values
- load 4 a in float32x4_t
- [The following repeats for each of the 4 lanes of a]
- for i in [0, 4]:
  - load b[i] in int8x16_t
  - subl to subtract b_zero_point from b, to get b_low, b_high
  - vmovl to get b_low_low, b_low_high, b_high_low, b_high_high
  - vcvtq to convert to float32x4_t, we will have 4 of these.
- for i in [0, 4]: for each of the 4 float32x4_t of b:
  - vfmaq_lane_fp32 to multiply a[lane] and b[i]
  - vfmaq_lane_fp32 to multiply a[lane] and b[i]
  - vfmaq_lane_fp32 to multiply a[lane] and b[i]
  - vfmaq_lane_fp32 to multiply a[lane] and b[i]
- By doing the above 4 times (lane=[0-3]), we used all values along k dim of a
  and accumulated 4 float32x4_t values
*/
TORCHAO_ALWAYS_INLINE void block_mul_1x16x1(
    const float32_t a,
    const int8x16_t& b_vec,
    const int8_t b_zero_point,
    const float b_scale,
    float32x4_t (&partial_sums)[4]) {
  int8x8_t b_zero_point_vec = vdup_n_s8(b_zero_point);
  int16x8_t b_vec_low = vsubl_s8(vget_low_s8(b_vec), b_zero_point_vec);
  int16x8_t b_vec_high = vsubl_s8(vget_high_s8(b_vec), b_zero_point_vec);
  float32x4_t b_vec_low_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_vec_low)));
  float32x4_t b_vec_low_high =
      vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_vec_low)));
  float32x4_t b_vec_high_low =
      vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_vec_high)));
  float32x4_t b_vec_high_high =
      vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_vec_high)));
  b_vec_low_low = vmulq_n_f32(b_vec_low_low, b_scale);
  b_vec_low_high = vmulq_n_f32(b_vec_low_high, b_scale);
  b_vec_high_low = vmulq_n_f32(b_vec_high_low, b_scale);
  b_vec_high_high = vmulq_n_f32(b_vec_high_high, b_scale);

  partial_sums[0] = vfmaq_n_f32(partial_sums[0], b_vec_low_low, a);
  partial_sums[1] = vfmaq_n_f32(partial_sums[1], b_vec_low_high, a);
  partial_sums[2] = vfmaq_n_f32(partial_sums[2], b_vec_high_low, a);
  partial_sums[3] = vfmaq_n_f32(partial_sums[3], b_vec_high_high, a);
}

void block_mul_1x16x4(
    const float32_t* a,
    const int8_t* b,
    const size_t ldb,
    const int8_t* b_zero_point,
    const float* b_scale,
    float32x4_t (&partial_sums)[4]) {
  #pragma unroll(8)
  for (int i = 0; i < 4; i++) {
    int8x16_t b_vec = vld1q_s8(b + i * ldb);
    block_mul_1x16x1(a[i], b_vec, b_zero_point[i], b_scale[i], partial_sums);
  }
}

} // namespace

template <bool b_has_zeros, bool a_transposed, bool b_transposed>
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
      const int8_t* rhs_zero_points,
      const float* rhs_scales,
      const float beta,
      const int rhs_qparams_stride);
};

/*
Document param meaning
rhs_stride_n: Since rhs transposed == false, the expected shape of rhs is k x n.
Thus rhs_stride_n is the stride of k dim, that how many bytes aparts elements
in k dim are.
*/
template <>
struct KernelImpl<true, false, false> {
  static void run(
      int m,
      int n,
      int k,
      const float* lhs,
      int lhs_stride_m,
      const int8_t* rhs,
      int rhs_stride_n,
      float32_t* output,
      int out_stride_m,
      const int8_t* rhs_zero_points,
      const float* rhs_scales,
      const float beta,
      const int rhs_qparams_stride) {
    std::unique_ptr<int8_t []> rhs_zero_points_transposed = std::make_unique<int8_t []>(k);
    std::unique_ptr<float []> rhs_scales_transposed = std::make_unique<float []>(k);
    if (rhs_qparams_stride > 1) {
      utils::transpose_scales_and_zero_points(
          rhs_zero_points,
          rhs_scales,
          rhs_zero_points_transposed.get(),
          rhs_scales_transposed.get(),
          k,
          rhs_qparams_stride);
      rhs_zero_points = rhs_zero_points_transposed.get();
      rhs_scales = rhs_scales_transposed.get();
    }

    constexpr int nr = 16;
    constexpr int kr = 4;
    for (int m_idx = 0; m_idx < m; m_idx++) {
      // Loop over 16 cols at a time
      // Access to partial tiles must be protected:w
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
        const float* lhs_ptr = lhs + m_idx * lhs_stride_m;
        const int8_t* rhs_ptr = rhs + n_idx;
        float32x4_t sums[nr / 4] = {vdupq_n_f32(0)};

        // Loop k_idx by group
        int k_idx = 0;
        for (; (k_idx + kr) <= k; k_idx += kr) {
          block_mul_1x16x4(
              lhs_ptr,
              rhs_ptr,
              rhs_stride_n,
              rhs_zero_points + k_idx,
              rhs_scales + k_idx,
              sums);
          lhs_ptr += kr;
          rhs_ptr += kr * rhs_stride_n;
        }

        for (int ki = 0; ki < (k - k_idx); ++ki) {
          // For each of the remaining k values
          // Load 1 int8_t from lhs
          // Load 16 int8_t from rhs
          // And multiply + add into the 16 accumulators
          // arranged as int32x4_t[4]
          int8x16_t rhs_vec = vld1q_s8(rhs_ptr + ki * rhs_stride_n);
          block_mul_1x16x1(
              lhs_ptr[ki],
              rhs_vec,
              rhs_zero_points[k_idx + ki],
              rhs_scales[k_idx + ki],
              sums);
        }

        // Store result
        // Because we adjust n_idx, we may end up writing the same location
        // twice
        // Note that the reason this case is being handled only for this kernel
        // and not others in this directory is because only for this kernel
        // we support accumulation.
        float* store_loc = output + m_idx * out_stride_m + n_idx;
        if (remaining < 16) {
          // If remaining is < 16, then not all of the 16 accumulators are
          // valid. That is not all of float32x4_t[4] are valid. We need to
          // find the first valid one, and then store the rest of the
          // accumulators in the same order.
          // First valid one is at 3 - ((remaining - 1) / 4) because:
          // If remaining is say 10 then first 6 are not valid.
          // Thus first group of 4 at sums[0] is not valid.
          // In the second group of 4, the first 2 are not valid.
          // Rest are valid.
          int start_sum_idx = 3 - ((remaining - 1) / 4);
          // If remaining is 11, then the sums[1] has 3 valid values
          // so 3 - (11 -1) % 4 = 3 - 10 % 4 = 3 - 2 = 1
          // Thus there is 1 invalid value in the first group of 4
          int invalid_values_in_32x4_reg = 3 - (remaining - 1) % 4;
          store_loc += start_sum_idx * 4;
          store_loc += invalid_values_in_32x4_reg;
          if (invalid_values_in_32x4_reg > 0) {
            for (int val_idx = invalid_values_in_32x4_reg; val_idx < 4;
                 ++val_idx) {
              *store_loc = sums[start_sum_idx][val_idx] + (*store_loc) * beta;
              store_loc += 1;
            }
            start_sum_idx++;
          }
          for (int out_idx = 0, sum_idx = start_sum_idx; sum_idx < nr / 4;
               out_idx += 4, ++sum_idx) {
            float32x4_t sum_val = vld1q_f32(store_loc + out_idx);
            sums[sum_idx] = vfmaq_n_f32(sums[sum_idx], sum_val, beta);
            vst1q_f32(store_loc + out_idx, sums[sum_idx]);
          }
        } else {
          for (int out_idx = 0, sum_idx = 0; out_idx < nr;
               out_idx += 4, ++sum_idx) {
            float32x4_t sum_val = vld1q_f32(store_loc + out_idx);
            sums[sum_idx] = vfmaq_n_f32(sums[sum_idx], sum_val, beta);
            vst1q_f32(store_loc + out_idx, sums[sum_idx]);
          }
        }
      } // n_idx
    } // m_idx
  }
};

} // namespace fp32_a_input_channelwise_8bit_b_1x16x4_f32::internal

namespace fp32_a_input_channelwise_8bit_b_1x16x4_f32 {
template <bool b_has_zeros, bool a_transposed, bool b_transposed>
void kernel(
    int m,
    int n,
    int k,
    const float* lhs,
    int lhs_stride_m,
    const int8_t* rhs,
    int rhs_stride_n,
    float32_t* output,
    int out_stride_m,
    const int8_t* rhs_zero_points,
    const float* rhs_scales,
    const float beta,
    const int rhs_qparams_stride) {
  torchao::kernels::cpu::aarch64::quantized_matmul::
      fp32_a_input_channelwise_8bit_b_1x16x4_f32::internal::
          KernelImpl<b_has_zeros, a_transposed, b_transposed>::run(
              m,
              n,
              k,
              lhs,
              lhs_stride_m,
              rhs,
              rhs_stride_n,
              output,
              out_stride_m,
              rhs_zero_points,
              rhs_scales,
              beta,
              rhs_qparams_stride);
}
} // namespace fp32_a_input_channelwise_8bit_b_1x16x4_f32
} // namespace torchao::kernels::cpu::aarch64::quantized_matmul

#endif // defined(__aarch64__) || defined(__ARM_NEON)
