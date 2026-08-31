// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/kernel_1x8x16_f32_neondot-impl.h>
#include <cassert>
#include <cstddef>

namespace torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight::kernel {
namespace internal {

inline void store_8_f32_2rows(
    float* output,
    int remaining,
    float32x4_t values_0123,
    float32x4_t values_4567) {
  if (remaining >= 8) {
    vst1q_f32(output, values_0123);
    vst1q_f32(output + 4, values_4567);
  } else if (remaining >= 7) {
    vst1q_f32(output, values_0123);
    vst1_f32(output + 4, vget_low_f32(values_4567));
    output[6] = vgetq_lane_f32(values_4567, 2);
  } else if (remaining >= 6) {
    vst1q_f32(output, values_0123);
    vst1_f32(output + 4, vget_low_f32(values_4567));
  } else if (remaining >= 5) {
    vst1q_f32(output, values_0123);
    output[4] = vgetq_lane_f32(values_4567, 0);
  } else if (remaining >= 4) {
    vst1q_f32(output, values_0123);
  } else if (remaining >= 3) {
    vst1_f32(output, vget_low_f32(values_0123));
    output[2] = vgetq_lane_f32(values_0123, 2);
  } else if (remaining >= 2) {
    vst1_f32(output, vget_low_f32(values_0123));
  } else {
    output[0] = vgetq_lane_f32(values_0123, 0);
  }
}

} // namespace internal

// Computes two activation rows together so that unpacked low-bit weights are
// reused across the rows. The packed formats are identical to the 1x8x16
// kernel. Any final partial M tile is handled by that kernel.
template <int weight_nbit, bool has_weight_zeros>
void kernel_2x8x16_f32_neondot(
    float32_t* output,
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* weight_data,
    const void* activation_data,
    float clamp_min,
    float clamp_max,
    bool has_bias,
    bool has_clamp) {
  assert(k % group_size == 0);
  assert(group_size % 16 == 0);

  constexpr int mr = 2;
  constexpr int bytes_per_128_weight_values = 16 * weight_nbit;
  const std::size_t activation_row_size = sizeof(float) + sizeof(int8_t) + k +
      (has_weight_zeros ? (k / group_size) * sizeof(int32_t) : 0);
  const auto* activation_data_bytes = static_cast<const char*>(activation_data);

  int m_idx = 0;
  for (; m_idx + mr <= m; m_idx += mr) {
    float activation_scales[mr];
    int activation_zeros[mr];
    const char* activation_qvals[mr];

    for (int row = 0; row < mr; row++) {
      const char* row_data =
          activation_data_bytes + (m_idx + row) * activation_row_size;
      activation_scales[row] = *reinterpret_cast<const float*>(row_data);
      row_data += sizeof(float);
      activation_zeros[row] =
          static_cast<int>(*reinterpret_cast<const int8_t*>(row_data));
      row_data += sizeof(int8_t);
      activation_qvals[row] = row_data;
    }

    const char* weight_data_bytes = static_cast<const char*>(weight_data);
    for (int n_idx = 0; n_idx < n; n_idx += 8) {
      const char* activation_ptrs[mr];
      float32x4_t res_0123[mr];
      float32x4_t res_4567[mr];
      for (int row = 0; row < mr; row++) {
        activation_ptrs[row] = activation_qvals[row];
        res_0123[row] = vdupq_n_f32(0.0f);
        res_4567[row] = vdupq_n_f32(0.0f);
      }

      for (int k_idx = 0; k_idx < k; k_idx += group_size) {
        int32x4_t acc_cols0011[mr];
        int32x4_t acc_cols2233[mr];
        int32x4_t acc_cols4455[mr];
        int32x4_t acc_cols6677[mr];
        for (int row = 0; row < mr; row++) {
          acc_cols0011[row] = vdupq_n_s32(0);
          acc_cols2233[row] = vdupq_n_s32(0);
          acc_cols4455[row] = vdupq_n_s32(0);
          acc_cols6677[row] = vdupq_n_s32(0);
        }

        for (int i = 0; i < group_size; i += 16) {
          int8x16_t weight_q_cols01_0;
          int8x16_t weight_q_cols23_0;
          int8x16_t weight_q_cols45_0;
          int8x16_t weight_q_cols67_0;
          int8x16_t weight_q_cols01_1;
          int8x16_t weight_q_cols23_1;
          int8x16_t weight_q_cols45_1;
          int8x16_t weight_q_cols67_1;

          torchao::bitpacking::vec_unpack_128_lowbit_values<weight_nbit>(
              weight_q_cols01_0,
              weight_q_cols23_0,
              weight_q_cols45_0,
              weight_q_cols67_0,
              weight_q_cols01_1,
              weight_q_cols23_1,
              weight_q_cols45_1,
              weight_q_cols67_1,
              reinterpret_cast<const uint8_t*>(weight_data_bytes));
          weight_data_bytes += bytes_per_128_weight_values;

          for (int row = 0; row < mr; row++) {
            int8x16_t act_q =
                vld1q_s8(reinterpret_cast<const int8_t*>(activation_ptrs[row]));
            activation_ptrs[row] += 16;

            int8x16_t act_q_dup =
                vcombine_s8(vget_low_s8(act_q), vget_low_s8(act_q));
            acc_cols0011[row] =
                vdotq_s32(acc_cols0011[row], weight_q_cols01_0, act_q_dup);
            acc_cols2233[row] =
                vdotq_s32(acc_cols2233[row], weight_q_cols23_0, act_q_dup);
            acc_cols4455[row] =
                vdotq_s32(acc_cols4455[row], weight_q_cols45_0, act_q_dup);
            acc_cols6677[row] =
                vdotq_s32(acc_cols6677[row], weight_q_cols67_0, act_q_dup);

            act_q_dup = vcombine_s8(vget_high_s8(act_q), vget_high_s8(act_q));
            acc_cols0011[row] =
                vdotq_s32(acc_cols0011[row], weight_q_cols01_1, act_q_dup);
            acc_cols2233[row] =
                vdotq_s32(acc_cols2233[row], weight_q_cols23_1, act_q_dup);
            acc_cols4455[row] =
                vdotq_s32(acc_cols4455[row], weight_q_cols45_1, act_q_dup);
            acc_cols6677[row] =
                vdotq_s32(acc_cols6677[row], weight_q_cols67_1, act_q_dup);
          }
        }

        float32x4_t weight_scales_0123 =
            vld1q_f32(reinterpret_cast<const float*>(weight_data_bytes));
        weight_data_bytes += 16;
        float32x4_t weight_scales_4567 =
            vld1q_f32(reinterpret_cast<const float*>(weight_data_bytes));
        weight_data_bytes += 16;
        int32x4_t weight_qvals_sum_0123 =
            vld1q_s32(reinterpret_cast<const int32_t*>(weight_data_bytes));
        weight_data_bytes += 16;
        int32x4_t weight_qvals_sum_4567 =
            vld1q_s32(reinterpret_cast<const int32_t*>(weight_data_bytes));
        weight_data_bytes += 16;

        int32x4_t weight_zeros_0123;
        int32x4_t weight_zeros_4567;
        if constexpr (has_weight_zeros) {
          weight_zeros_0123 =
              vld1q_s32(reinterpret_cast<const int32_t*>(weight_data_bytes));
          weight_data_bytes += 16;
          weight_zeros_4567 =
              vld1q_s32(reinterpret_cast<const int32_t*>(weight_data_bytes));
          weight_data_bytes += 16;
        }

        for (int row = 0; row < mr; row++) {
          int32x4_t qval_dot_0123 =
              vpaddq_s32(acc_cols0011[row], acc_cols2233[row]);
          int32x4_t qval_dot_4567 =
              vpaddq_s32(acc_cols4455[row], acc_cols6677[row]);
          int32x4_t corrected_0123 = vsubq_s32(
              qval_dot_0123,
              vmulq_n_s32(weight_qvals_sum_0123, activation_zeros[row]));
          int32x4_t corrected_4567 = vsubq_s32(
              qval_dot_4567,
              vmulq_n_s32(weight_qvals_sum_4567, activation_zeros[row]));

          if constexpr (has_weight_zeros) {
            int32_t activation_qvals_sum =
                *reinterpret_cast<const int32_t*>(activation_ptrs[row]);
            activation_ptrs[row] += sizeof(int32_t);
            corrected_0123 = vsubq_s32(
                corrected_0123,
                vmulq_n_s32(weight_zeros_0123, activation_qvals_sum));
            corrected_0123 = vaddq_s32(
                corrected_0123,
                vmulq_n_s32(
                    weight_zeros_0123, group_size * activation_zeros[row]));
            corrected_4567 = vsubq_s32(
                corrected_4567,
                vmulq_n_s32(weight_zeros_4567, activation_qvals_sum));
            corrected_4567 = vaddq_s32(
                corrected_4567,
                vmulq_n_s32(
                    weight_zeros_4567, group_size * activation_zeros[row]));
          }

          float32x4_t activation_scale = vdupq_n_f32(activation_scales[row]);
          float32x4_t scale_factor_0123 =
              vmulq_f32(weight_scales_0123, activation_scale);
          float32x4_t scale_factor_4567 =
              vmulq_f32(weight_scales_4567, activation_scale);
          res_0123[row] = vmlaq_f32(
              res_0123[row], scale_factor_0123, vcvtq_f32_s32(corrected_0123));
          res_4567[row] = vmlaq_f32(
              res_4567[row], scale_factor_4567, vcvtq_f32_s32(corrected_4567));
        }
      }

      if (has_bias) {
        float32x4_t bias_0123 =
            vld1q_f32(reinterpret_cast<const float*>(weight_data_bytes));
        weight_data_bytes += 16;
        float32x4_t bias_4567 =
            vld1q_f32(reinterpret_cast<const float*>(weight_data_bytes));
        weight_data_bytes += 16;
        for (int row = 0; row < mr; row++) {
          res_0123[row] = vaddq_f32(res_0123[row], bias_0123);
          res_4567[row] = vaddq_f32(res_4567[row], bias_4567);
        }
      }

      if (has_clamp) {
        float32x4_t vec_min = vdupq_n_f32(clamp_min);
        float32x4_t vec_max = vdupq_n_f32(clamp_max);
        for (int row = 0; row < mr; row++) {
          res_0123[row] = internal::vec_clamp(res_0123[row], vec_min, vec_max);
          res_4567[row] = internal::vec_clamp(res_4567[row], vec_min, vec_max);
        }
      }

      const int remaining = n - n_idx;
      for (int row = 0; row < mr; row++) {
        internal::store_8_f32_2rows(
            output + (m_idx + row) * output_m_stride + n_idx,
            remaining,
            res_0123[row],
            res_4567[row]);
      }
    }
  }

  if (m_idx < m) {
    kernel_1x8x16_f32_neondot<weight_nbit, has_weight_zeros, false>(
        output + m_idx * output_m_stride,
        output_m_stride,
        m - m_idx,
        n,
        k,
        group_size,
        weight_data,
        activation_data_bytes + m_idx * activation_row_size,
        clamp_min,
        clamp_max,
        has_bias,
        has_clamp);
  }
}

} // namespace
  // torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::kernel

#endif // defined(__aarch64__) || defined(__ARM_NEON)
