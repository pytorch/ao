// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/kernel_1x8x16_f32_neondot-impl.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/kernel_2x8x16_f32_neondot-impl.h>
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight::kernel {
namespace internal {

inline void store_8_f32(
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

inline void transpose_4x16_s8(
    int8x16_t row0,
    int8x16_t row1,
    int8x16_t row2,
    int8x16_t row3,
    int8x16_t& cols_0_3,
    int8x16_t& cols_4_7,
    int8x16_t& cols_8_11,
    int8x16_t& cols_12_15) {
  int32x4_t row0_s32 = vreinterpretq_s32_s8(row0);
  int32x4_t row1_s32 = vreinterpretq_s32_s8(row1);
  int32x4_t row2_s32 = vreinterpretq_s32_s8(row2);
  int32x4_t row3_s32 = vreinterpretq_s32_s8(row3);

  int32x4_t rows01_lo = vzip1q_s32(row0_s32, row1_s32);
  int32x4_t rows01_hi = vzip2q_s32(row0_s32, row1_s32);
  int32x4_t rows23_lo = vzip1q_s32(row2_s32, row3_s32);
  int32x4_t rows23_hi = vzip2q_s32(row2_s32, row3_s32);

  cols_0_3 = vreinterpretq_s8_s64(vzip1q_s64(
      vreinterpretq_s64_s32(rows01_lo), vreinterpretq_s64_s32(rows23_lo)));
  cols_4_7 = vreinterpretq_s8_s64(vzip2q_s64(
      vreinterpretq_s64_s32(rows01_lo), vreinterpretq_s64_s32(rows23_lo)));
  cols_8_11 = vreinterpretq_s8_s64(vzip1q_s64(
      vreinterpretq_s64_s32(rows01_hi), vreinterpretq_s64_s32(rows23_hi)));
  cols_12_15 = vreinterpretq_s8_s64(vzip2q_s64(
      vreinterpretq_s64_s32(rows01_hi), vreinterpretq_s64_s32(rows23_hi)));
}

inline void transpose_4x4_f32(
    float32x4_t col0,
    float32x4_t col1,
    float32x4_t col2,
    float32x4_t col3,
    float32x4_t& row0,
    float32x4_t& row1,
    float32x4_t& row2,
    float32x4_t& row3) {
  float32x4_t cols01_lo = vzip1q_f32(col0, col1);
  float32x4_t cols01_hi = vzip2q_f32(col0, col1);
  float32x4_t cols23_lo = vzip1q_f32(col2, col3);
  float32x4_t cols23_hi = vzip2q_f32(col2, col3);

  row0 = vcombine_f32(vget_low_f32(cols01_lo), vget_low_f32(cols23_lo));
  row1 = vcombine_f32(vget_high_f32(cols01_lo), vget_high_f32(cols23_lo));
  row2 = vcombine_f32(vget_low_f32(cols01_hi), vget_low_f32(cols23_hi));
  row3 = vcombine_f32(vget_high_f32(cols01_hi), vget_high_f32(cols23_hi));
}

TORCHAO_ALWAYS_INLINE inline void dot_4_rows_8_cols(
    int32x4_t* accumulators,
    int8x16_t activations_0_3,
    int8x16_t activations_4_7,
    int8x16_t activations_8_11,
    int8x16_t activations_12_15,
    int8x16_t weights01_0,
    int8x16_t weights23_0,
    int8x16_t weights45_0,
    int8x16_t weights67_0,
    int8x16_t weights01_1,
    int8x16_t weights23_1,
    int8x16_t weights45_1,
    int8x16_t weights67_1) {
  accumulators[0] =
      vdotq_laneq_s32(accumulators[0], activations_0_3, weights01_0, 0);
  accumulators[0] =
      vdotq_laneq_s32(accumulators[0], activations_4_7, weights01_0, 1);
  accumulators[0] =
      vdotq_laneq_s32(accumulators[0], activations_8_11, weights01_1, 0);
  accumulators[0] =
      vdotq_laneq_s32(accumulators[0], activations_12_15, weights01_1, 1);
  accumulators[1] =
      vdotq_laneq_s32(accumulators[1], activations_0_3, weights01_0, 2);
  accumulators[1] =
      vdotq_laneq_s32(accumulators[1], activations_4_7, weights01_0, 3);
  accumulators[1] =
      vdotq_laneq_s32(accumulators[1], activations_8_11, weights01_1, 2);
  accumulators[1] =
      vdotq_laneq_s32(accumulators[1], activations_12_15, weights01_1, 3);

  accumulators[2] =
      vdotq_laneq_s32(accumulators[2], activations_0_3, weights23_0, 0);
  accumulators[2] =
      vdotq_laneq_s32(accumulators[2], activations_4_7, weights23_0, 1);
  accumulators[2] =
      vdotq_laneq_s32(accumulators[2], activations_8_11, weights23_1, 0);
  accumulators[2] =
      vdotq_laneq_s32(accumulators[2], activations_12_15, weights23_1, 1);
  accumulators[3] =
      vdotq_laneq_s32(accumulators[3], activations_0_3, weights23_0, 2);
  accumulators[3] =
      vdotq_laneq_s32(accumulators[3], activations_4_7, weights23_0, 3);
  accumulators[3] =
      vdotq_laneq_s32(accumulators[3], activations_8_11, weights23_1, 2);
  accumulators[3] =
      vdotq_laneq_s32(accumulators[3], activations_12_15, weights23_1, 3);

  accumulators[4] =
      vdotq_laneq_s32(accumulators[4], activations_0_3, weights45_0, 0);
  accumulators[4] =
      vdotq_laneq_s32(accumulators[4], activations_4_7, weights45_0, 1);
  accumulators[4] =
      vdotq_laneq_s32(accumulators[4], activations_8_11, weights45_1, 0);
  accumulators[4] =
      vdotq_laneq_s32(accumulators[4], activations_12_15, weights45_1, 1);
  accumulators[5] =
      vdotq_laneq_s32(accumulators[5], activations_0_3, weights45_0, 2);
  accumulators[5] =
      vdotq_laneq_s32(accumulators[5], activations_4_7, weights45_0, 3);
  accumulators[5] =
      vdotq_laneq_s32(accumulators[5], activations_8_11, weights45_1, 2);
  accumulators[5] =
      vdotq_laneq_s32(accumulators[5], activations_12_15, weights45_1, 3);

  accumulators[6] =
      vdotq_laneq_s32(accumulators[6], activations_0_3, weights67_0, 0);
  accumulators[6] =
      vdotq_laneq_s32(accumulators[6], activations_4_7, weights67_0, 1);
  accumulators[6] =
      vdotq_laneq_s32(accumulators[6], activations_8_11, weights67_1, 0);
  accumulators[6] =
      vdotq_laneq_s32(accumulators[6], activations_12_15, weights67_1, 1);
  accumulators[7] =
      vdotq_laneq_s32(accumulators[7], activations_0_3, weights67_0, 2);
  accumulators[7] =
      vdotq_laneq_s32(accumulators[7], activations_4_7, weights67_0, 3);
  accumulators[7] =
      vdotq_laneq_s32(accumulators[7], activations_8_11, weights67_1, 2);
  accumulators[7] =
      vdotq_laneq_s32(accumulators[7], activations_12_15, weights67_1, 3);
}

} // namespace internal

// Computes an output tile with one accumulator vector per output column. Each
// vector lane corresponds to one of four activation rows. This layout reuses
// unpacked weights across four rows without the register pressure of four
// independent copies of the 1x8 accumulator set.
template <int weight_nbit, bool has_weight_zeros>
void kernel_4x8x16_f32_neondot(
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

  constexpr int mr = 4;
  constexpr int nr = 8;
  constexpr int bytes_per_128_weight_values = 16 * weight_nbit;
  const std::size_t activation_row_size = sizeof(float) + sizeof(int8_t) + k +
      (has_weight_zeros ? (k / group_size) * sizeof(int32_t) : 0);
  const auto* activation_data_bytes = static_cast<const char*>(activation_data);

  int m_idx = 0;
  for (; m_idx + 3 <= m; m_idx += mr) {
    const int tile_m = std::min(mr, m - m_idx);
    const char* activation_block =
        activation_data_bytes + m_idx * activation_row_size;
    float32x4_t activation_scale_vec =
        vld1q_f32(reinterpret_cast<const float*>(activation_block));
    activation_block += mr * sizeof(float);
    int32_t activation_zeros[mr];
    for (int row = 0; row < mr; row++) {
      activation_zeros[row] =
          static_cast<int32_t>(*reinterpret_cast<const int8_t*>(
              activation_block + row * sizeof(int8_t)));
    }
    activation_block += mr * sizeof(int8_t);
    int32x4_t activation_zero_vec = vld1q_s32(activation_zeros);

    const char* weight_data_bytes = static_cast<const char*>(weight_data);
    for (int n_idx = 0; n_idx < n; n_idx += nr) {
      const char* activation_ptr = activation_block;
      float32x4_t results[nr];
      for (int col = 0; col < nr; col++) {
        results[col] = vdupq_n_f32(0.0f);
      }

      for (int k_idx = 0; k_idx < k; k_idx += group_size) {
        int32x4_t accumulators[nr];
        for (int col = 0; col < nr; col++) {
          accumulators[col] = vdupq_n_s32(0);
        }

        for (int i = 0; i < group_size; i += 16) {
          int8x16_t weights01_0;
          int8x16_t weights23_0;
          int8x16_t weights45_0;
          int8x16_t weights67_0;
          int8x16_t weights01_1;
          int8x16_t weights23_1;
          int8x16_t weights45_1;
          int8x16_t weights67_1;
          torchao::bitpacking::vec_unpack_128_lowbit_values<weight_nbit>(
              weights01_0,
              weights23_0,
              weights45_0,
              weights67_0,
              weights01_1,
              weights23_1,
              weights45_1,
              weights67_1,
              reinterpret_cast<const uint8_t*>(weight_data_bytes));
          weight_data_bytes += bytes_per_128_weight_values;

          int8x16_t activations_0_3 =
              vld1q_s8(reinterpret_cast<const int8_t*>(activation_ptr));
          int8x16_t activations_4_7 =
              vld1q_s8(reinterpret_cast<const int8_t*>(activation_ptr + 16));
          int8x16_t activations_8_11 =
              vld1q_s8(reinterpret_cast<const int8_t*>(activation_ptr + 32));
          int8x16_t activations_12_15 =
              vld1q_s8(reinterpret_cast<const int8_t*>(activation_ptr + 48));
          activation_ptr += 64;
          internal::dot_4_rows_8_cols(
              accumulators,
              activations_0_3,
              activations_4_7,
              activations_8_11,
              activations_12_15,
              weights01_0,
              weights23_0,
              weights45_0,
              weights67_0,
              weights01_1,
              weights23_1,
              weights45_1,
              weights67_1);
        }

        const float* weight_scales =
            reinterpret_cast<const float*>(weight_data_bytes);
        weight_data_bytes += nr * sizeof(float);
        const int32_t* weight_qvals_sums =
            reinterpret_cast<const int32_t*>(weight_data_bytes);
        weight_data_bytes += nr * sizeof(int32_t);
        const int32_t* weight_zeros = nullptr;
        if constexpr (has_weight_zeros) {
          weight_zeros = reinterpret_cast<const int32_t*>(weight_data_bytes);
          weight_data_bytes += nr * sizeof(int32_t);
        }

        int32x4_t activation_qvals_sum_vec;
        if constexpr (has_weight_zeros) {
          activation_qvals_sum_vec =
              vld1q_s32(reinterpret_cast<const int32_t*>(activation_ptr));
          activation_ptr += mr * sizeof(int32_t);
        }

        for (int col = 0; col < nr; col++) {
          int32x4_t corrected = vsubq_s32(
              accumulators[col],
              vmulq_n_s32(activation_zero_vec, weight_qvals_sums[col]));
          if constexpr (has_weight_zeros) {
            corrected = vsubq_s32(
                corrected,
                vmulq_n_s32(activation_qvals_sum_vec, weight_zeros[col]));
            corrected = vaddq_s32(
                corrected,
                vmulq_n_s32(
                    activation_zero_vec, group_size * weight_zeros[col]));
          }
          float32x4_t scale_factor =
              vmulq_n_f32(activation_scale_vec, weight_scales[col]);
          results[col] =
              vmlaq_f32(results[col], scale_factor, vcvtq_f32_s32(corrected));
        }
      }

      if (has_bias) {
        const float* bias = reinterpret_cast<const float*>(weight_data_bytes);
        weight_data_bytes += nr * sizeof(float);
        for (int col = 0; col < nr; col++) {
          results[col] = vaddq_f32(results[col], vdupq_n_f32(bias[col]));
        }
      }
      if (has_clamp) {
        float32x4_t vec_min = vdupq_n_f32(clamp_min);
        float32x4_t vec_max = vdupq_n_f32(clamp_max);
        for (int col = 0; col < nr; col++) {
          results[col] = internal::vec_clamp(results[col], vec_min, vec_max);
        }
      }

      float32x4_t output_0123[mr];
      float32x4_t output_4567[mr];
      internal::transpose_4x4_f32(
          results[0],
          results[1],
          results[2],
          results[3],
          output_0123[0],
          output_0123[1],
          output_0123[2],
          output_0123[3]);
      internal::transpose_4x4_f32(
          results[4],
          results[5],
          results[6],
          results[7],
          output_4567[0],
          output_4567[1],
          output_4567[2],
          output_4567[3]);
      const int remaining = n - n_idx;
      for (int row = 0; row < tile_m; row++) {
        internal::store_8_f32(
            output + (m_idx + row) * output_m_stride + n_idx,
            remaining,
            output_0123[row],
            output_4567[row]);
      }
    }
  }

  if (m_idx < m) {
    if (m - m_idx >= 2) {
      kernel_2x8x16_f32_neondot<weight_nbit, has_weight_zeros>(
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
    } else {
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
}

} // namespace
  // torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::kernel

#endif // defined(__aarch64__) || defined(__ARM_NEON)
