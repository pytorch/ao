// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/kernel_1x8x16_f32_neondot-impl.h>
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight::kernel {
namespace internal {

TORCHAO_ALWAYS_INLINE inline void dot_4_rows_2_cols(
    int32x4_t& accumulator0,
    int32x4_t& accumulator1,
    const int8x16_t (&activations)[4],
    int8x16_t weights_0_7,
    int8x16_t weights_8_15) {
  accumulator0 = vdotq_laneq_s32(accumulator0, activations[0], weights_0_7, 0);
  accumulator0 = vdotq_laneq_s32(accumulator0, activations[1], weights_0_7, 1);
  accumulator0 = vdotq_laneq_s32(accumulator0, activations[2], weights_8_15, 0);
  accumulator0 = vdotq_laneq_s32(accumulator0, activations[3], weights_8_15, 1);
  accumulator1 = vdotq_laneq_s32(accumulator1, activations[0], weights_0_7, 2);
  accumulator1 = vdotq_laneq_s32(accumulator1, activations[1], weights_0_7, 3);
  accumulator1 = vdotq_laneq_s32(accumulator1, activations[2], weights_8_15, 2);
  accumulator1 = vdotq_laneq_s32(accumulator1, activations[3], weights_8_15, 3);
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

inline void store_8_f32(
    float* output,
    int remaining,
    float32x4_t values_0123,
    float32x4_t values_4567) {
  if (remaining >= 8) {
    vst1q_f32(output, values_0123);
    vst1q_f32(output + 4, values_4567);
  } else if (remaining == 7) {
    vst1q_f32(output, values_0123);
    vst1_f32(output + 4, vget_low_f32(values_4567));
    output[6] = vgetq_lane_f32(values_4567, 2);
  } else if (remaining == 6) {
    vst1q_f32(output, values_0123);
    vst1_f32(output + 4, vget_low_f32(values_4567));
  } else if (remaining == 5) {
    vst1q_f32(output, values_0123);
    output[4] = vgetq_lane_f32(values_4567, 0);
  } else if (remaining == 4) {
    vst1q_f32(output, values_0123);
  } else if (remaining == 3) {
    vst1_f32(output, vget_low_f32(values_0123));
    output[2] = vgetq_lane_f32(values_0123, 2);
  } else if (remaining == 2) {
    vst1_f32(output, vget_low_f32(values_0123));
  } else {
    output[0] = vgetq_lane_f32(values_0123, 0);
  }
}

template <int column_half, bool shift_weights_to_signed>
TORCHAO_ALWAYS_INLINE inline void decode_4_cols_w3(
    int8x16_t& weights01_0,
    int8x16_t& weights01_1,
    int8x16_t& weights23_0,
    int8x16_t& weights23_1,
    const uint8_t* packed_weights) {
  static_assert(column_half == 0 || column_half == 1);
  const uint8x16_t packed0 = vld1q_u8(packed_weights);
  const uint8x16_t packed1 = vld1q_u8(packed_weights + 16);
  const uint8x16_t packed2 = vld1q_u8(packed_weights + 32);
  const uint8x16_t mask7 = vdupq_n_u8(7);

  if constexpr (column_half == 0) {
    weights01_0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed0, 3), mask7));
    weights01_1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed2, 3), mask7));
    weights23_0 = vreinterpretq_s8_u8(vandq_u8(packed0, mask7));
    weights23_1 = vreinterpretq_s8_u8(vandq_u8(packed2, mask7));
  } else {
    weights01_0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed1, 3), mask7));
    uint8x16_t weights45_1 = vandq_u8(vshrq_n_u8(packed2, 5), vdupq_n_u8(4));
    weights45_1 = vorrq_u8(weights45_1, vshrq_n_u8(packed0, 6));
    weights01_1 = vreinterpretq_s8_u8(weights45_1);
    weights23_0 = vreinterpretq_s8_u8(vandq_u8(packed1, mask7));
    uint8x16_t weights67_1 = vandq_u8(vshrq_n_u8(packed2, 4), vdupq_n_u8(4));
    weights67_1 = vorrq_u8(weights67_1, vshrq_n_u8(packed1, 6));
    weights23_1 = vreinterpretq_s8_u8(weights67_1);
  }

  if constexpr (shift_weights_to_signed) {
    const int8x16_t unshift = vdupq_n_s8(-4);
    weights01_0 = vaddq_s8(weights01_0, unshift);
    weights01_1 = vaddq_s8(weights01_1, unshift);
    weights23_0 = vaddq_s8(weights23_0, unshift);
    weights23_1 = vaddq_s8(weights23_1, unshift);
  }
}

template <int row_blocks, bool shift_weights_to_signed>
TORCHAO_ALWAYS_INLINE inline void dot_rows_8_cols_w3(
    int32x4_t (&accumulators)[row_blocks][8],
    const uint8_t* packed_weights,
    const int8_t* packed_activations) {
  static_assert(row_blocks == 1 || row_blocks == 2);
  int8x16_t weights01_0;
  int8x16_t weights01_1;
  int8x16_t weights23_0;
  int8x16_t weights23_1;
  int8x16_t activations[row_blocks][4];
  for (int block = 0; block < row_blocks; block++) {
    const int8_t* ptr = packed_activations + block * 64;
    activations[block][0] = vld1q_s8(ptr);
    activations[block][1] = vld1q_s8(ptr + 16);
    activations[block][2] = vld1q_s8(ptr + 32);
    activations[block][3] = vld1q_s8(ptr + 48);
  }

  decode_4_cols_w3<0, shift_weights_to_signed>(
      weights01_0,
      weights01_1,
      weights23_0,
      weights23_1,
      packed_weights);
  for (int block = 0; block < row_blocks; block++) {
    dot_4_rows_2_cols(
        accumulators[block][0],
        accumulators[block][1],
        activations[block],
        weights01_0,
        weights01_1);
    dot_4_rows_2_cols(
        accumulators[block][2],
        accumulators[block][3],
        activations[block],
        weights23_0,
        weights23_1);
  }

  decode_4_cols_w3<1, shift_weights_to_signed>(
      weights01_0,
      weights01_1,
      weights23_0,
      weights23_1,
      packed_weights);
  for (int block = 0; block < row_blocks; block++) {
    dot_4_rows_2_cols(
        accumulators[block][4],
        accumulators[block][5],
        activations[block],
        weights01_0,
        weights01_1);
    dot_4_rows_2_cols(
        accumulators[block][6],
        accumulators[block][7],
        activations[block],
        weights23_0,
        weights23_1);
  }
}

template <int row_blocks, int col, int lane, bool has_weight_zeros>
TORCHAO_ALWAYS_INLINE inline void accumulate_rows_column(
    volatile float32x4_t (&results)[8][row_blocks],
    int32x4_t (&accumulators)[row_blocks][8],
    const int32x4_t (&activation_zero_vec)[row_blocks],
    const float32x4_t (&activation_scale_vec)[row_blocks],
    const int32x4_t& weight_qvals_sum_vec,
    const float32x4_t& weight_scale_vec,
    const int32x4_t& weight_zero_vec,
    const int32x4_t (&activation_qvals_sum_vec)[row_blocks]) {
  for (int block = 0; block < row_blocks; block++) {
    int32x4_t corrected = vmlsq_laneq_s32(
        accumulators[block][col],
        activation_zero_vec[block],
        weight_qvals_sum_vec,
        lane);
    if constexpr (has_weight_zeros) {
      corrected = vmlsq_laneq_s32(
          corrected, activation_qvals_sum_vec[block], weight_zero_vec, lane);
    }
    const float32x4_t scale_factor =
        vmulq_laneq_f32(activation_scale_vec[block], weight_scale_vec, lane);
    results[col][block] = vmlaq_f32(
        results[col][block], scale_factor, vcvtq_f32_s32(corrected));
  }
}

template <int col, bool has_weight_zeros>
TORCHAO_ALWAYS_INLINE inline void accumulate_16_row_column(
    volatile float32x4_t (&results)[4][4],
    int32x4_t (&accumulators)[4][4],
    const int32x4_t (&activation_zero_vec)[4],
    const float32x4_t (&activation_scale_vec)[4],
    const int32x4_t& weight_qvals_sum_vec,
    const float32x4_t& weight_scale_vec,
    const int32x4_t& weight_zero_vec,
    const int32x4_t (&activation_qvals_sum_vec)[4]) {
  for (int row_block = 0; row_block < 4; row_block++) {
    int32x4_t corrected = vmlsq_laneq_s32(
        accumulators[row_block][col],
        activation_zero_vec[row_block],
        weight_qvals_sum_vec,
        col);
    if constexpr (has_weight_zeros) {
      corrected = vmlsq_laneq_s32(
          corrected, activation_qvals_sum_vec[row_block], weight_zero_vec, col);
    }
    const float32x4_t scale_factor =
        vmulq_laneq_f32(activation_scale_vec[row_block], weight_scale_vec, col);
    results[col][row_block] = vmlaq_f32(
        results[col][row_block], scale_factor, vcvtq_f32_s32(corrected));
  }
}

inline void store_4_f32(float* output, int remaining, float32x4_t values) {
  if (remaining >= 4) {
    vst1q_f32(output, values);
  } else if (remaining == 3) {
    vst1_f32(output, vget_low_f32(values));
    output[2] = vgetq_lane_f32(values, 2);
  } else if (remaining == 2) {
    vst1_f32(output, vget_low_f32(values));
  } else if (remaining == 1) {
    output[0] = vgetq_lane_f32(values, 0);
  }
}

template <int row_blocks, bool has_weight_zeros>
__attribute__((noinline)) void kernel_rows_8x16_w3(
    float* output,
    int output_m_stride,
    int remaining_n,
    int k,
    int group_size,
    const char* weight_panel,
    const char* activation_block,
    int row_block_offset,
    int valid_rows,
    float clamp_min,
    float clamp_max,
    bool has_bias,
    bool has_clamp) {
  static_assert(row_blocks == 1 || row_blocks == 2);
  assert(row_block_offset >= 0 && row_block_offset + row_blocks <= 2);
  assert(valid_rows >= 1 && valid_rows <= row_blocks * 4);
  constexpr int mr = 8;
  constexpr int nr = 8;
  constexpr int bytes_per_128_weight_values = 48;

  float32x4_t activation_scale_vec[row_blocks];
  for (int block = 0; block < row_blocks; block++) {
    activation_scale_vec[block] = vld1q_f32(
        reinterpret_cast<const float*>(
            activation_block + (row_block_offset + block) * 16));
  }
  const int16x8_t activation_zeros = vmovl_s8(vld1_s8(
      reinterpret_cast<const int8_t*>(
          activation_block + mr * sizeof(float))));
  int32x4_t activation_zero_vec[row_blocks];
  if (row_block_offset == 0) {
    activation_zero_vec[0] = vmovl_s16(vget_low_s16(activation_zeros));
    if constexpr (row_blocks == 2) {
      activation_zero_vec[1] = vmovl_s16(vget_high_s16(activation_zeros));
    }
  } else {
    activation_zero_vec[0] = vmovl_s16(vget_high_s16(activation_zeros));
  }

  const char* activation_ptr =
      activation_block + mr * (sizeof(float) + sizeof(int8_t));
  const char* weight_ptr = weight_panel;
  volatile float32x4_t results[nr][row_blocks];
  for (int col = 0; col < nr; col++) {
    for (int block = 0; block < row_blocks; block++) {
      results[col][block] = vdupq_n_f32(0.0f);
    }
  }

  for (int k_idx = 0; k_idx < k; k_idx += group_size) {
    int32x4_t accumulators[row_blocks][nr];
    for (int block = 0; block < row_blocks; block++) {
      for (int col = 0; col < nr; col++) {
        accumulators[block][col] = vdupq_n_s32(0);
      }
    }

    for (int i = 0; i < group_size; i += 16) {
      dot_rows_8_cols_w3<row_blocks, !has_weight_zeros>(
          accumulators,
          reinterpret_cast<const uint8_t*>(weight_ptr),
          reinterpret_cast<const int8_t*>(
              activation_ptr + row_block_offset * 64));
      weight_ptr += bytes_per_128_weight_values;
      activation_ptr += mr * 16;
    }

    const float* weight_scales = reinterpret_cast<const float*>(weight_ptr);
    const float32x4_t weight_scales_0123 = vld1q_f32(weight_scales);
    const float32x4_t weight_scales_4567 = vld1q_f32(weight_scales + 4);
    weight_ptr += nr * sizeof(float);
    const int32_t* weight_qvals_sums =
        reinterpret_cast<const int32_t*>(weight_ptr);
    int32x4_t weight_qvals_sums_0123 = vld1q_s32(weight_qvals_sums);
    int32x4_t weight_qvals_sums_4567 = vld1q_s32(weight_qvals_sums + 4);
    weight_ptr += nr * sizeof(int32_t);
    int32x4_t weight_zeros_0123 = vdupq_n_s32(0);
    int32x4_t weight_zeros_4567 = vdupq_n_s32(0);
    if constexpr (has_weight_zeros) {
      const int32_t* weight_zeros =
          reinterpret_cast<const int32_t*>(weight_ptr);
      weight_zeros_0123 = vld1q_s32(weight_zeros);
      weight_zeros_4567 = vld1q_s32(weight_zeros + 4);
      weight_ptr += nr * sizeof(int32_t);
      weight_qvals_sums_0123 = vmlsq_n_s32(
          weight_qvals_sums_0123, weight_zeros_0123, group_size);
      weight_qvals_sums_4567 = vmlsq_n_s32(
          weight_qvals_sums_4567, weight_zeros_4567, group_size);
      weight_zeros_0123 = vaddq_s32(weight_zeros_0123, vdupq_n_s32(4));
      weight_zeros_4567 = vaddq_s32(weight_zeros_4567, vdupq_n_s32(4));
    }

    int32x4_t activation_qvals_sum_vec[row_blocks];
    for (int block = 0; block < row_blocks; block++) {
      activation_qvals_sum_vec[block] = vdupq_n_s32(0);
    }
    if constexpr (has_weight_zeros) {
      for (int block = 0; block < row_blocks; block++) {
        activation_qvals_sum_vec[block] = vld1q_s32(
            reinterpret_cast<const int32_t*>(
                activation_ptr + (row_block_offset + block) * 16));
      }
      activation_ptr += mr * sizeof(int32_t);
    }

#define TORCHAO_ACCUMULATE_ROWS(COL, LANE, SUMS, SCALES, ZEROS)       \
  accumulate_rows_column<row_blocks, COL, LANE, has_weight_zeros>(   \
      results,                                                        \
      accumulators,                                                   \
      activation_zero_vec,                                           \
      activation_scale_vec,                                          \
      SUMS,                                                           \
      SCALES,                                                         \
      ZEROS,                                                          \
      activation_qvals_sum_vec)
    TORCHAO_ACCUMULATE_ROWS(
        0, 0, weight_qvals_sums_0123, weight_scales_0123, weight_zeros_0123);
    TORCHAO_ACCUMULATE_ROWS(
        1, 1, weight_qvals_sums_0123, weight_scales_0123, weight_zeros_0123);
    TORCHAO_ACCUMULATE_ROWS(
        2, 2, weight_qvals_sums_0123, weight_scales_0123, weight_zeros_0123);
    TORCHAO_ACCUMULATE_ROWS(
        3, 3, weight_qvals_sums_0123, weight_scales_0123, weight_zeros_0123);
    TORCHAO_ACCUMULATE_ROWS(
        4, 0, weight_qvals_sums_4567, weight_scales_4567, weight_zeros_4567);
    TORCHAO_ACCUMULATE_ROWS(
        5, 1, weight_qvals_sums_4567, weight_scales_4567, weight_zeros_4567);
    TORCHAO_ACCUMULATE_ROWS(
        6, 2, weight_qvals_sums_4567, weight_scales_4567, weight_zeros_4567);
    TORCHAO_ACCUMULATE_ROWS(
        7, 3, weight_qvals_sums_4567, weight_scales_4567, weight_zeros_4567);
#undef TORCHAO_ACCUMULATE_ROWS
  }

  if (has_bias) {
    const float* bias = reinterpret_cast<const float*>(weight_ptr);
    for (int col = 0; col < nr; col++) {
      for (int block = 0; block < row_blocks; block++) {
        results[col][block] =
            vaddq_f32(results[col][block], vdupq_n_f32(bias[col]));
      }
    }
  }
  if (has_clamp) {
    const float32x4_t vec_min = vdupq_n_f32(clamp_min);
    const float32x4_t vec_max = vdupq_n_f32(clamp_max);
    for (int col = 0; col < nr; col++) {
      for (int block = 0; block < row_blocks; block++) {
        results[col][block] =
            vec_clamp(results[col][block], vec_min, vec_max);
      }
    }
  }

  for (int block = 0; block < row_blocks; block++) {
    const int block_rows = std::min(4, valid_rows - block * 4);
    if (block_rows <= 0) {
      break;
    }
    float32x4_t output_0123[4];
    float32x4_t output_4567[4];
    transpose_4x4_f32(
        results[0][block],
        results[1][block],
        results[2][block],
        results[3][block],
        output_0123[0],
        output_0123[1],
        output_0123[2],
        output_0123[3]);
    transpose_4x4_f32(
        results[4][block],
        results[5][block],
        results[6][block],
        results[7][block],
        output_4567[0],
        output_4567[1],
        output_4567[2],
        output_4567[3]);
    for (int row = 0; row < block_rows; row++) {
      store_8_f32(
          output + (block * 4 + row) * output_m_stride,
          remaining_n,
          output_0123[row],
          output_4567[row]);
    }
  }
}

template <bool has_weight_zeros>
__attribute__((noinline)) void kernel_1x8x16_w3_interleaved(
    float* output,
    int remaining_n,
    int k,
    int group_size,
    const char* weight_panel,
    const char* activation_block,
    int row,
    float clamp_min,
    float clamp_max,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 8;
  constexpr int nr = 8;
  constexpr int bytes_per_128_weight_values = 48;
  const float activation_scale =
      reinterpret_cast<const float*>(activation_block)[row];
  const int32_t activation_zero = static_cast<int32_t>(
      reinterpret_cast<const int8_t*>(
          activation_block + mr * sizeof(float))[row]);
  const char* activation_ptr =
      activation_block + mr * (sizeof(float) + sizeof(int8_t));
  const char* weight_ptr = weight_panel;
  float32x4_t result_0123 = vdupq_n_f32(0.0f);
  float32x4_t result_4567 = vdupq_n_f32(0.0f);

  for (int k_idx = 0; k_idx < k; k_idx += group_size) {
    int32x4_t accumulators[4] = {
        vdupq_n_s32(0),
        vdupq_n_s32(0),
        vdupq_n_s32(0),
        vdupq_n_s32(0)};
    for (int i = 0; i < group_size; i += 16) {
      const int32x4x4_t rows = vld4q_s32(
          reinterpret_cast<const int32_t*>(
              activation_ptr + (row / 4) * 64));
      const int8x16_t activation =
          vreinterpretq_s8_s32(rows.val[row % 4]);
      const int8x16_t activation_lo =
          vcombine_s8(vget_low_s8(activation), vget_low_s8(activation));
      const int8x16_t activation_hi =
          vcombine_s8(vget_high_s8(activation), vget_high_s8(activation));
      int8x16_t weights01_0;
      int8x16_t weights01_1;
      int8x16_t weights23_0;
      int8x16_t weights23_1;
      decode_4_cols_w3<0, !has_weight_zeros>(
          weights01_0,
          weights01_1,
          weights23_0,
          weights23_1,
          reinterpret_cast<const uint8_t*>(weight_ptr));
      accumulators[0] =
          vdotq_s32(accumulators[0], weights01_0, activation_lo);
      accumulators[0] =
          vdotq_s32(accumulators[0], weights01_1, activation_hi);
      accumulators[1] =
          vdotq_s32(accumulators[1], weights23_0, activation_lo);
      accumulators[1] =
          vdotq_s32(accumulators[1], weights23_1, activation_hi);
      decode_4_cols_w3<1, !has_weight_zeros>(
          weights01_0,
          weights01_1,
          weights23_0,
          weights23_1,
          reinterpret_cast<const uint8_t*>(weight_ptr));
      accumulators[2] =
          vdotq_s32(accumulators[2], weights01_0, activation_lo);
      accumulators[2] =
          vdotq_s32(accumulators[2], weights01_1, activation_hi);
      accumulators[3] =
          vdotq_s32(accumulators[3], weights23_0, activation_lo);
      accumulators[3] =
          vdotq_s32(accumulators[3], weights23_1, activation_hi);
      weight_ptr += bytes_per_128_weight_values;
      activation_ptr += mr * 16;
    }

    const float32x4_t weight_scales_0123 =
        vld1q_f32(reinterpret_cast<const float*>(weight_ptr));
    const float32x4_t weight_scales_4567 =
        vld1q_f32(reinterpret_cast<const float*>(weight_ptr) + 4);
    weight_ptr += nr * sizeof(float);
    int32x4_t weight_qvals_sums_0123 =
        vld1q_s32(reinterpret_cast<const int32_t*>(weight_ptr));
    int32x4_t weight_qvals_sums_4567 =
        vld1q_s32(reinterpret_cast<const int32_t*>(weight_ptr) + 4);
    weight_ptr += nr * sizeof(int32_t);
    int32x4_t weight_zeros_0123 = vdupq_n_s32(0);
    int32x4_t weight_zeros_4567 = vdupq_n_s32(0);
    if constexpr (has_weight_zeros) {
      weight_zeros_0123 =
          vld1q_s32(reinterpret_cast<const int32_t*>(weight_ptr));
      weight_zeros_4567 =
          vld1q_s32(reinterpret_cast<const int32_t*>(weight_ptr) + 4);
      weight_ptr += nr * sizeof(int32_t);
      weight_qvals_sums_0123 = vmlsq_n_s32(
          weight_qvals_sums_0123, weight_zeros_0123, group_size);
      weight_qvals_sums_4567 = vmlsq_n_s32(
          weight_qvals_sums_4567, weight_zeros_4567, group_size);
      weight_zeros_0123 = vaddq_s32(weight_zeros_0123, vdupq_n_s32(4));
      weight_zeros_4567 = vaddq_s32(weight_zeros_4567, vdupq_n_s32(4));
    }

    int32x4_t corrected_0123 = vsubq_s32(
        vpaddq_s32(accumulators[0], accumulators[1]),
        vmulq_n_s32(weight_qvals_sums_0123, activation_zero));
    int32x4_t corrected_4567 = vsubq_s32(
        vpaddq_s32(accumulators[2], accumulators[3]),
        vmulq_n_s32(weight_qvals_sums_4567, activation_zero));
    if constexpr (has_weight_zeros) {
      const int32_t activation_qvals_sum =
          reinterpret_cast<const int32_t*>(activation_ptr)[row];
      corrected_0123 = vmlsq_n_s32(
          corrected_0123, weight_zeros_0123, activation_qvals_sum);
      corrected_4567 = vmlsq_n_s32(
          corrected_4567, weight_zeros_4567, activation_qvals_sum);
      activation_ptr += mr * sizeof(int32_t);
    }
    const float32x4_t activation_scale_vec =
        vdupq_n_f32(activation_scale);
    result_0123 = vmlaq_f32(
        result_0123,
        vmulq_f32(weight_scales_0123, activation_scale_vec),
        vcvtq_f32_s32(corrected_0123));
    result_4567 = vmlaq_f32(
        result_4567,
        vmulq_f32(weight_scales_4567, activation_scale_vec),
        vcvtq_f32_s32(corrected_4567));
  }

  if (has_bias) {
    result_0123 = vaddq_f32(
        result_0123, vld1q_f32(reinterpret_cast<const float*>(weight_ptr)));
    result_4567 = vaddq_f32(
        result_4567,
        vld1q_f32(reinterpret_cast<const float*>(weight_ptr) + 4));
  }
  if (has_clamp) {
    const float32x4_t vec_min = vdupq_n_f32(clamp_min);
    const float32x4_t vec_max = vdupq_n_f32(clamp_max);
    result_0123 = vec_clamp(result_0123, vec_min, vec_max);
    result_4567 = vec_clamp(result_4567, vec_min, vec_max);
  }
  store_8_f32(output, remaining_n, result_0123, result_4567);
}

template <int column_half, bool has_weight_zeros>
__attribute__((noinline)) void kernel_16x4x16_w3(
    float* output,
    int output_m_stride,
    int remaining_n,
    int k,
    int group_size,
    const char* weight_panel,
    const char* activation_block_0,
    const char* activation_block_1,
    int valid_rows,
    float clamp_min,
    float clamp_max,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 8;
  constexpr int nr = 8;
  constexpr int bytes_per_128_weight_values = 48;
  constexpr int column_offset = column_half * 4;
  assert(valid_rows >= 9 && valid_rows <= 16);

  float32x4_t activation_scale_vec[4] = {
      vld1q_f32(reinterpret_cast<const float*>(activation_block_0)),
      vld1q_f32(reinterpret_cast<const float*>(activation_block_0 + 16)),
      vld1q_f32(reinterpret_cast<const float*>(activation_block_1)),
      vld1q_f32(reinterpret_cast<const float*>(activation_block_1 + 16))};
  int16x8_t activation_zeros_0 = vmovl_s8(vld1_s8(
      reinterpret_cast<const int8_t*>(
          activation_block_0 + mr * sizeof(float))));
  int16x8_t activation_zeros_1 = vmovl_s8(vld1_s8(
      reinterpret_cast<const int8_t*>(
          activation_block_1 + mr * sizeof(float))));
  int32x4_t activation_zero_vec[4] = {
      vmovl_s16(vget_low_s16(activation_zeros_0)),
      vmovl_s16(vget_high_s16(activation_zeros_0)),
      vmovl_s16(vget_low_s16(activation_zeros_1)),
      vmovl_s16(vget_high_s16(activation_zeros_1))};

  const char* activation_ptrs[2] = {
      activation_block_0 + mr * (sizeof(float) + sizeof(int8_t)),
      activation_block_1 + mr * (sizeof(float) + sizeof(int8_t))};
  const char* weight_ptr = weight_panel;
  volatile float32x4_t results[4][4];
  for (int col = 0; col < 4; col++) {
    for (int row_block = 0; row_block < 4; row_block++) {
      results[col][row_block] = vdupq_n_f32(0.0f);
    }
  }

  for (int k_idx = 0; k_idx < k; k_idx += group_size) {
    int32x4_t accumulators[4][4];
    for (int row_block = 0; row_block < 4; row_block++) {
      for (int col = 0; col < 4; col++) {
        accumulators[row_block][col] = vdupq_n_s32(0);
      }
    }

    for (int i = 0; i < group_size; i += 16) {
      int8x16_t weights01_0;
      int8x16_t weights01_1;
      int8x16_t weights23_0;
      int8x16_t weights23_1;
      decode_4_cols_w3<column_half, !has_weight_zeros>(
          weights01_0,
          weights01_1,
          weights23_0,
          weights23_1,
          reinterpret_cast<const uint8_t*>(weight_ptr));

      for (int activation_block = 0; activation_block < 2; activation_block++) {
        for (int row_half = 0; row_half < 2; row_half++) {
          const int8_t* ptr = reinterpret_cast<const int8_t*>(
              activation_ptrs[activation_block] + row_half * 64);
          int8x16_t activations[4] = {
              vld1q_s8(ptr),
              vld1q_s8(ptr + 16),
              vld1q_s8(ptr + 32),
              vld1q_s8(ptr + 48)};
          const int row_block = activation_block * 2 + row_half;
          dot_4_rows_2_cols(
              accumulators[row_block][0],
              accumulators[row_block][1],
              activations,
              weights01_0,
              weights01_1);
          dot_4_rows_2_cols(
              accumulators[row_block][2],
              accumulators[row_block][3],
              activations,
              weights23_0,
              weights23_1);
        }
        activation_ptrs[activation_block] += mr * 16;
      }
      weight_ptr += bytes_per_128_weight_values;
    }

    const float* weight_scales = reinterpret_cast<const float*>(weight_ptr);
    const float32x4_t weight_scale_vec =
        vld1q_f32(weight_scales + column_offset);
    weight_ptr += nr * sizeof(float);
    const int32_t* weight_qvals_sums =
        reinterpret_cast<const int32_t*>(weight_ptr);
    int32x4_t weight_qvals_sum_vec =
        vld1q_s32(weight_qvals_sums + column_offset);
    weight_ptr += nr * sizeof(int32_t);
    int32x4_t weight_zero_vec = vdupq_n_s32(0);
    if constexpr (has_weight_zeros) {
      const int32_t* weight_zeros =
          reinterpret_cast<const int32_t*>(weight_ptr);
      weight_zero_vec = vld1q_s32(weight_zeros + column_offset);
      weight_ptr += nr * sizeof(int32_t);
      weight_qvals_sum_vec =
          vmlsq_n_s32(weight_qvals_sum_vec, weight_zero_vec, group_size);
      weight_zero_vec = vaddq_s32(weight_zero_vec, vdupq_n_s32(4));
    }

    int32x4_t activation_qvals_sum_vec[4] = {
        vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
    if constexpr (has_weight_zeros) {
      activation_qvals_sum_vec[0] =
          vld1q_s32(reinterpret_cast<const int32_t*>(activation_ptrs[0]));
      activation_qvals_sum_vec[1] =
          vld1q_s32(reinterpret_cast<const int32_t*>(activation_ptrs[0] + 16));
      activation_qvals_sum_vec[2] =
          vld1q_s32(reinterpret_cast<const int32_t*>(activation_ptrs[1]));
      activation_qvals_sum_vec[3] =
          vld1q_s32(reinterpret_cast<const int32_t*>(activation_ptrs[1] + 16));
      activation_ptrs[0] += mr * sizeof(int32_t);
      activation_ptrs[1] += mr * sizeof(int32_t);
    }

    accumulate_16_row_column<0, has_weight_zeros>(
        results,
        accumulators,
        activation_zero_vec,
        activation_scale_vec,
        weight_qvals_sum_vec,
        weight_scale_vec,
        weight_zero_vec,
        activation_qvals_sum_vec);
    accumulate_16_row_column<1, has_weight_zeros>(
        results,
        accumulators,
        activation_zero_vec,
        activation_scale_vec,
        weight_qvals_sum_vec,
        weight_scale_vec,
        weight_zero_vec,
        activation_qvals_sum_vec);
    accumulate_16_row_column<2, has_weight_zeros>(
        results,
        accumulators,
        activation_zero_vec,
        activation_scale_vec,
        weight_qvals_sum_vec,
        weight_scale_vec,
        weight_zero_vec,
        activation_qvals_sum_vec);
    accumulate_16_row_column<3, has_weight_zeros>(
        results,
        accumulators,
        activation_zero_vec,
        activation_scale_vec,
        weight_qvals_sum_vec,
        weight_scale_vec,
        weight_zero_vec,
        activation_qvals_sum_vec);
  }

  if (has_bias) {
    const float* bias =
        reinterpret_cast<const float*>(weight_ptr) + column_offset;
    for (int row_block = 0; row_block < 4; row_block++) {
      for (int col = 0; col < 4; col++) {
        results[col][row_block] =
            vaddq_f32(results[col][row_block], vdupq_n_f32(bias[col]));
      }
    }
  }
  if (has_clamp) {
    const float32x4_t vec_min = vdupq_n_f32(clamp_min);
    const float32x4_t vec_max = vdupq_n_f32(clamp_max);
    for (int row_block = 0; row_block < 4; row_block++) {
      for (int col = 0; col < 4; col++) {
        results[col][row_block] =
            vec_clamp(results[col][row_block], vec_min, vec_max);
      }
    }
  }

  for (int row_block = 0; row_block < 4; row_block++) {
    const int block_rows = std::min(4, valid_rows - row_block * 4);
    if (block_rows <= 0) {
      break;
    }
    float32x4_t rows[4];
    transpose_4x4_f32(
        results[0][row_block],
        results[1][row_block],
        results[2][row_block],
        results[3][row_block],
        rows[0],
        rows[1],
        rows[2],
        rows[3]);
    for (int row = 0; row < block_rows; row++) {
      store_4_f32(
          output + (row_block * 4 + row) * output_m_stride + column_offset,
          remaining_n - column_offset,
          rows[row]);
    }
  }
}

} // namespace internal

// Reuses each decoded W3 panel across 16 rows, then handles the M tail with
// dedicated 8-row, 4-row, and interleaved 1-row paths from the same source.
template <bool has_weight_zeros>
void kernel_16x8x16_f32_neondot(
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

  constexpr int mr = 8;
  constexpr int nr = 8;
  constexpr int weight_nbit = 3;
  const std::size_t activation_row_size = sizeof(float) + sizeof(int8_t) + k +
      (has_weight_zeros ? (k / group_size) * sizeof(int32_t) : 0);
  const std::size_t activation_block_size = mr * activation_row_size;
  const std::size_t weight_panel_size = k * weight_nbit +
      (k / group_size) * nr *
          (sizeof(float) + sizeof(int32_t) +
           (has_weight_zeros ? sizeof(int32_t) : 0)) +
      (has_bias ? nr * sizeof(float) : 0);
  const auto* activation_data_bytes = static_cast<const char*>(activation_data);
  const auto* weight_data_bytes = static_cast<const char*>(weight_data);

  int m_idx = 0;
  for (; m_idx + 16 <= m; m_idx += 16) {
    const char* activation_block_0 =
        activation_data_bytes + m_idx * activation_row_size;
    const char* activation_block_1 = activation_block_0 + activation_block_size;
    const char* weight_panel = weight_data_bytes;
    for (int n_idx = 0; n_idx < n; n_idx += nr) {
      const int remaining_n = n - n_idx;
      internal::kernel_16x4x16_w3<0, has_weight_zeros>(
          output + m_idx * output_m_stride + n_idx,
          output_m_stride,
          remaining_n,
          k,
          group_size,
          weight_panel,
          activation_block_0,
          activation_block_1,
          16,
          clamp_min,
          clamp_max,
          has_bias,
          has_clamp);
      if (remaining_n > 4) {
        internal::kernel_16x4x16_w3<1, has_weight_zeros>(
            output + m_idx * output_m_stride + n_idx,
            output_m_stride,
            remaining_n,
            k,
            group_size,
            weight_panel,
            activation_block_0,
            activation_block_1,
            16,
            clamp_min,
            clamp_max,
            has_bias,
            has_clamp);
      }
      weight_panel += weight_panel_size;
    }
  }

  if (m - m_idx > 8) {
    const int tail_rows = m - m_idx;
    const char* activation_block_0 =
        activation_data_bytes + m_idx * activation_row_size;
    const char* activation_block_1 = activation_block_0 + activation_block_size;
    const char* weight_panel = weight_data_bytes;
    for (int n_idx = 0; n_idx < n; n_idx += nr) {
      const int remaining_n = n - n_idx;
      internal::kernel_16x4x16_w3<0, has_weight_zeros>(
          output + m_idx * output_m_stride + n_idx,
          output_m_stride,
          remaining_n,
          k,
          group_size,
          weight_panel,
          activation_block_0,
          activation_block_1,
          tail_rows,
          clamp_min,
          clamp_max,
          has_bias,
          has_clamp);
      if (remaining_n > 4) {
        internal::kernel_16x4x16_w3<1, has_weight_zeros>(
            output + m_idx * output_m_stride + n_idx,
            output_m_stride,
            remaining_n,
            k,
            group_size,
            weight_panel,
            activation_block_0,
            activation_block_1,
            tail_rows,
            clamp_min,
            clamp_max,
            has_bias,
            has_clamp);
      }
      weight_panel += weight_panel_size;
    }
    m_idx += tail_rows;
  } else if (m - m_idx > 4) {
    const int tail_rows = m - m_idx;
    const char* weight_panel = weight_data_bytes;
    const char* activation_block =
        activation_data_bytes + m_idx * activation_row_size;
    for (int n_idx = 0; n_idx < n; n_idx += nr) {
      internal::kernel_rows_8x16_w3<2, has_weight_zeros>(
          output + m_idx * output_m_stride + n_idx,
          output_m_stride,
          n - n_idx,
          k,
          group_size,
          weight_panel,
          activation_block,
          0,
          tail_rows,
          clamp_min,
          clamp_max,
          has_bias,
          has_clamp);
      weight_panel += weight_panel_size;
    }
    m_idx += tail_rows;
  }

  if (m - m_idx > 1) {
    const int tail_rows = m - m_idx;
    const int activation_block_idx = m_idx / mr;
    const int row_block_offset = (m_idx % mr) / 4;
    const char* weight_panel = weight_data_bytes;
    const char* activation_block = activation_data_bytes +
        activation_block_idx * activation_block_size;
    for (int n_idx = 0; n_idx < n; n_idx += nr) {
      internal::kernel_rows_8x16_w3<1, has_weight_zeros>(
          output + m_idx * output_m_stride + n_idx,
          output_m_stride,
          n - n_idx,
          k,
          group_size,
          weight_panel,
          activation_block,
          row_block_offset,
          tail_rows,
          clamp_min,
          clamp_max,
          has_bias,
          has_clamp);
      weight_panel += weight_panel_size;
    }
    m_idx += tail_rows;
  }

  for (; m_idx < m; m_idx++) {
    const int activation_block_idx = m_idx / mr;
    const int row_in_block = m_idx % mr;
    const char* activation_block = activation_data_bytes +
        activation_block_idx * activation_block_size;
    const char* weight_panel = weight_data_bytes;
    for (int n_idx = 0; n_idx < n; n_idx += nr) {
      internal::kernel_1x8x16_w3_interleaved<has_weight_zeros>(
          output + m_idx * output_m_stride + n_idx,
          n - n_idx,
          k,
          group_size,
          weight_panel,
          activation_block,
          row_in_block,
          clamp_min,
          clamp_max,
          has_bias,
          has_clamp);
      weight_panel += weight_panel_size;
    }
  }
}

} // namespace
  // torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::kernel

#endif // defined(__aarch64__) || defined(__ARM_NEON)
