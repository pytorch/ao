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
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/matmul/matmul_utils.h>

namespace torchao::kernels::cpu::aarch64::quantized_matmul {
namespace fp32_a_input_channelwise_8bit_b_4x16x4_f32::internal {

namespace {

/*
This function loads float32x4_t value from a, and 16 int8x16_t values from b.
For each int8x16_t of b:
- 4 float32x4 accumulated values
- load 4 a in float32x4_t
- [The following repeats for each of the 4 lanes of a]
- for i in [0, 4]:
  - load b[i] in int8x16_t
  - subl to subtarct b_zero_point from b, to get b_low, b_high
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
TORCHAO_ALWAYS_INLINE inline void block_mul_4x16x1(
    const float32x4_t& a,
    const int8x16_t& b_vec,
    const int8_t b_zero_point,
    const float b_scale,
    float32x4_t (&partial_sums)[4][4]) {
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

  partial_sums[0][0] = vfmaq_n_f32(partial_sums[0][0], b_vec_low_low, a[0]);
  partial_sums[0][1] = vfmaq_n_f32(partial_sums[0][1], b_vec_low_high, a[0]);
  partial_sums[0][2] = vfmaq_n_f32(partial_sums[0][2], b_vec_high_low, a[0]);
  partial_sums[0][3] = vfmaq_n_f32(partial_sums[0][3], b_vec_high_high, a[0]);

  partial_sums[1][0] = vfmaq_n_f32(partial_sums[1][0], b_vec_low_low, a[1]);
  partial_sums[1][1] = vfmaq_n_f32(partial_sums[1][1], b_vec_low_high, a[1]);
  partial_sums[1][2] = vfmaq_n_f32(partial_sums[1][2], b_vec_high_low, a[1]);
  partial_sums[1][3] = vfmaq_n_f32(partial_sums[1][3], b_vec_high_high, a[1]);

  partial_sums[2][0] = vfmaq_n_f32(partial_sums[2][0], b_vec_low_low, a[2]);
  partial_sums[2][1] = vfmaq_n_f32(partial_sums[2][1], b_vec_low_high, a[2]);
  partial_sums[2][2] = vfmaq_n_f32(partial_sums[2][2], b_vec_high_low, a[2]);
  partial_sums[2][3] = vfmaq_n_f32(partial_sums[2][3], b_vec_high_high, a[2]);

  partial_sums[3][0] = vfmaq_n_f32(partial_sums[3][0], b_vec_low_low, a[3]);
  partial_sums[3][1] = vfmaq_n_f32(partial_sums[3][1], b_vec_low_high, a[3]);
  partial_sums[3][2] = vfmaq_n_f32(partial_sums[3][2], b_vec_high_low, a[3]);
  partial_sums[3][3] = vfmaq_n_f32(partial_sums[3][3], b_vec_high_high, a[3]);
}

TORCHAO_ALWAYS_INLINE inline void block_mul_4x16x4(
    const float32_t* a,
    const size_t lda,
    const int8_t* b,
    const size_t ldb,
    const int8_t* b_zero_point,
    const float* b_scale,
    float32x4_t (&partial_sums)[4][4]) {
  float32x4_t a_vec[4];
  utils::transpose_4x4(a, lda, a_vec);

  int8x16_t b_vec = vld1q_s8(b + 0 * ldb);
  block_mul_4x16x1(a_vec[0], b_vec, b_zero_point[0], b_scale[0], partial_sums);
  b_vec = vld1q_s8(b + 1 * ldb);
  block_mul_4x16x1(a_vec[1], b_vec, b_zero_point[1], b_scale[1], partial_sums);
  b_vec = vld1q_s8(b + 2 * ldb);
  block_mul_4x16x1(a_vec[2], b_vec, b_zero_point[2], b_scale[2], partial_sums);
  b_vec = vld1q_s8(b + 3 * ldb);
  block_mul_4x16x1(a_vec[3], b_vec, b_zero_point[3], b_scale[3], partial_sums);
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
    std::vector<int8_t> rhs_zero_points_transposed;
    std::vector<float> rhs_scales_transposed;
    if (rhs_qparams_stride > 1) {
      rhs_zero_points_transposed.resize(k);
      rhs_scales_transposed.resize(k);
      utils::transpose_scales_and_zero_points(
          rhs_zero_points,
          rhs_scales,
          rhs_zero_points_transposed.data(),
          rhs_scales_transposed.data(),
          k,
          rhs_qparams_stride);
      rhs_zero_points = rhs_zero_points_transposed.data();
      rhs_scales = rhs_scales_transposed.data();
    }

    constexpr int mr = 4;
    constexpr int nr = 16;
    constexpr int kr = 4;
    assert(m % mr == 0);
    assert(kr == 4);
    assert(n >= nr);
    for (int m_idx = 0; m_idx < m; m_idx += mr) {
      const float* lhs_ptr = lhs + m_idx * lhs_stride_m;
      // Loop over 16 cols at a time
      // Access to partial tiles must be protected
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
        const int8_t* rhs_ptr = rhs + n_idx;
        float32x4_t sums[mr][(nr / 4)] = {{vdupq_n_f32(0)}};

        // Loop k_idx by group
        int k_idx = 0;
        const float* current_lhs_ptr = lhs_ptr;
        for (; (k_idx + kr) <= k; k_idx += kr) {
          block_mul_4x16x4(
              current_lhs_ptr,
              lhs_stride_m,
              rhs_ptr,
              rhs_stride_n,
              rhs_zero_points + k_idx,
              rhs_scales + k_idx,
              sums);
          current_lhs_ptr += kr;
          rhs_ptr += kr * rhs_stride_n;
        }

        for (int ki = 0; ki < (k - k_idx); ++ki) {
          // For each of the remaining k values
          // Load 1 int8_t from lhs
          // Load 16 int8_t from rhs
          // And multiply + add into the 16 accumulators
          // arranged as int32x4_t[4]
          int8x16_t rhs_vec = vld1q_s8(rhs_ptr + ki * rhs_stride_n);
          float32x4_t lhs_vec = {
              current_lhs_ptr[ki + 0 * lhs_stride_m],
              current_lhs_ptr[ki + 1 * lhs_stride_m],
              current_lhs_ptr[ki + 2 * lhs_stride_m],
              current_lhs_ptr[ki + 3 * lhs_stride_m]};
          block_mul_4x16x1(
              lhs_vec,
              rhs_vec,
              rhs_zero_points[k_idx + ki],
              rhs_scales[k_idx + ki],
              sums);
        }

        // Store result
        // Because we adjust n_idx, we may end up writing the same location
        // twice
        // Note that the reason this case is being handld only for this kernel
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
            for (int m_out_idx = 0; m_out_idx < mr; m_out_idx++) {
              float* store_loc_local = store_loc + m_out_idx * out_stride_m;
              for (int val_idx = invalid_values_in_32x4_reg; val_idx < 4;
                   ++val_idx) {
                *store_loc_local = sums[m_out_idx][start_sum_idx][val_idx] +
                    (*store_loc_local) * beta;
                store_loc_local += 1;
              }
            }
            start_sum_idx++;
            store_loc += (4 - invalid_values_in_32x4_reg);
          }
          for (int m_out_idx = 0; m_out_idx < mr; m_out_idx++) {
            float* store_loc_local = store_loc + m_out_idx * out_stride_m;
            for (int out_idx = 0, sum_idx = start_sum_idx; sum_idx < nr / 4;
                 out_idx += 4, ++sum_idx) {
              float32x4_t sum_val = vld1q_f32(store_loc_local + out_idx);
              sums[m_out_idx][sum_idx] =
                  vfmaq_n_f32(sums[m_out_idx][sum_idx], sum_val, beta);
              vst1q_f32(store_loc_local + out_idx, sums[m_out_idx][sum_idx]);
            }
          }
        } else {
          for (int m_out_idx = 0; m_out_idx < mr; m_out_idx++) {
            float* store_loc_local = store_loc + m_out_idx * out_stride_m;
            for (int out_idx = 0, sum_idx = 0; out_idx < nr;
                 out_idx += 4, ++sum_idx) {
              float32x4_t sum_val = vld1q_f32(store_loc_local + out_idx);
              sums[m_out_idx][sum_idx] =
                  vfmaq_n_f32(sums[m_out_idx][sum_idx], sum_val, beta);
              vst1q_f32(store_loc_local + out_idx, sums[m_out_idx][sum_idx]);
            }
          }
        }
      } // n_idx
    } // m_idx
  }
};

} // namespace fp32_a_input_channelwise_8bit_b_4x16x4_f32::internal

namespace fp32_a_input_channelwise_8bit_b_4x16x4_f32 {
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
      fp32_a_input_channelwise_8bit_b_4x16x4_f32::internal::
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
} // namespace fp32_a_input_channelwise_8bit_b_4x16x4_f32
} // namespace torchao::kernels::cpu::aarch64::quantized_matmul

#endif // defined(__aarch64__) && defined(__ARM_NEON)
