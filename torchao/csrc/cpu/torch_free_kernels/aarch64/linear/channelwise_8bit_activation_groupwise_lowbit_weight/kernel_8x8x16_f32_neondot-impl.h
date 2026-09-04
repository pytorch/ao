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

constexpr int kMr = 8;
constexpr int kNr = 8;
constexpr int kBytesPer128W3Values = 48;

// Accumulator lanes represent four rows. Select four weight bytes at a time so
// the same two output columns can be accumulated for all four rows.
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

// Convert four column vectors into four row vectors for contiguous stores.
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

// Store one row of an 8-column tile without writing past an N tail.
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
  // An 8-column by 16-K W3 panel occupies 48 bytes. Decode either four-column
  // half into the layout consumed by the lane dot-product instructions.
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
    // Packed W3 codes are 0..7; symmetric weights use signed values -4..3.
    const int8x16_t unshift = vdupq_n_s8(-4);
    weights01_0 = vaddq_s8(weights01_0, unshift);
    weights01_1 = vaddq_s8(weights01_1, unshift);
    weights23_0 = vaddq_s8(weights23_0, unshift);
    weights23_1 = vaddq_s8(weights23_1, unshift);
  }
}

template <
    int row_blocks,
    int column_blocks,
    int first_column_half,
    bool shift_weights_to_signed>
TORCHAO_ALWAYS_INLINE inline void dot_rows_cols_w3(
    int32x4_t (&accumulators)[row_blocks][column_blocks * 4],
    const uint8_t* packed_weights,
    const char* const (&activation_ptrs)[(row_blocks + 1) / 2]) {
  static_assert(row_blocks == 1 || row_blocks == 2 || row_blocks == 4);
  static_assert(column_blocks == 1 || column_blocks == 2);
  static_assert(first_column_half == 0 || first_column_half == 1);
  static_assert(first_column_half + column_blocks <= 2);

  int8x16_t weights01_0;
  int8x16_t weights01_1;
  int8x16_t weights23_0;
  int8x16_t weights23_1;
  if constexpr (column_blocks == 1) {
    // Keep only four activation vectors live in the 16-row specialization.
    // Holding all 16 at once would create unnecessary register pressure.
    decode_4_cols_w3<first_column_half, shift_weights_to_signed>(
        weights01_0,
        weights01_1,
        weights23_0,
        weights23_1,
        packed_weights);
    for (int block = 0; block < row_blocks; block++) {
      const int8_t* ptr = reinterpret_cast<const int8_t*>(
          activation_ptrs[block / 2] + (block % 2) * 64);
      int8x16_t activations[4] = {
          vld1q_s8(ptr),
          vld1q_s8(ptr + 16),
          vld1q_s8(ptr + 32),
          vld1q_s8(ptr + 48)};
      dot_4_rows_2_cols(
          accumulators[block][0],
          accumulators[block][1],
          activations,
          weights01_0,
          weights01_1);
      dot_4_rows_2_cols(
          accumulators[block][2],
          accumulators[block][3],
          activations,
          weights23_0,
          weights23_1);
    }
  } else {
    // For an eight-column tile, keep up to eight activation vectors live and
    // reuse them across both four-column weight halves.
    int8x16_t activations[row_blocks][4];
    for (int block = 0; block < row_blocks; block++) {
      const int8_t* ptr = reinterpret_cast<const int8_t*>(
          activation_ptrs[block / 2] + (block % 2) * 64);
      activations[block][0] = vld1q_s8(ptr);
      activations[block][1] = vld1q_s8(ptr + 16);
      activations[block][2] = vld1q_s8(ptr + 32);
      activations[block][3] = vld1q_s8(ptr + 48);
    }
#define TORCHAO_DOT_COLUMN_BLOCK(COLUMN_BLOCK, COLUMN_HALF)          \
  decode_4_cols_w3<COLUMN_HALF, shift_weights_to_signed>(           \
      weights01_0,                                                   \
      weights01_1,                                                   \
      weights23_0,                                                   \
      weights23_1,                                                   \
      packed_weights);                                               \
  for (int block = 0; block < row_blocks; block++) {                 \
    dot_4_rows_2_cols(                                               \
        accumulators[block][COLUMN_BLOCK * 4],                       \
        accumulators[block][COLUMN_BLOCK * 4 + 1],                   \
        activations[block],                                          \
        weights01_0,                                                 \
        weights01_1);                                                \
    dot_4_rows_2_cols(                                               \
        accumulators[block][COLUMN_BLOCK * 4 + 2],                   \
        accumulators[block][COLUMN_BLOCK * 4 + 3],                   \
        activations[block],                                          \
        weights23_0,                                                 \
        weights23_1);                                                \
  }
    TORCHAO_DOT_COLUMN_BLOCK(0, first_column_half);
    TORCHAO_DOT_COLUMN_BLOCK(1, first_column_half + 1);
#undef TORCHAO_DOT_COLUMN_BLOCK
  }
}

template <
    int row_blocks,
    int cols,
    int col,
    int lane,
    bool has_weight_zeros>
TORCHAO_ALWAYS_INLINE inline void accumulate_rows_column(
    volatile float32x4_t (&results)[cols][row_blocks],
    int32x4_t (&accumulators)[row_blocks][cols],
    const int32x4_t (&activation_zero_vec)[row_blocks],
    const float32x4_t (&activation_scale_vec)[row_blocks],
    const int32x4_t& weight_qvals_sum_vec,
    const float32x4_t& weight_scale_vec,
    const int32x4_t& weight_zero_vec,
    const int32x4_t (&activation_qvals_sum_vec)[row_blocks]) {
  // Correct the integer dot product, apply both quantization scales, and add
  // this K group's contribution to the FP32 result.
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

template <int row_blocks, int cols, int first_col, bool has_weight_zeros>
TORCHAO_ALWAYS_INLINE inline void accumulate_4_columns(
    volatile float32x4_t (&results)[cols][row_blocks],
    int32x4_t (&accumulators)[row_blocks][cols],
    const int32x4_t (&activation_zero_vec)[row_blocks],
    const float32x4_t (&activation_scale_vec)[row_blocks],
    const int32x4_t& weight_qvals_sum_vec,
    const float32x4_t& weight_scale_vec,
    const int32x4_t& weight_zero_vec,
    const int32x4_t (&activation_qvals_sum_vec)[row_blocks]) {
#define TORCHAO_ACCUMULATE_COLUMN(OFFSET)                \
  accumulate_rows_column<                                \
      row_blocks, cols, first_col + OFFSET, OFFSET,       \
      has_weight_zeros>(                                  \
      results,                                            \
      accumulators,                                       \
      activation_zero_vec,                                \
      activation_scale_vec,                               \
      weight_qvals_sum_vec,                               \
      weight_scale_vec,                                   \
      weight_zero_vec,                                    \
      activation_qvals_sum_vec)
  TORCHAO_ACCUMULATE_COLUMN(0);
  TORCHAO_ACCUMULATE_COLUMN(1);
  TORCHAO_ACCUMULATE_COLUMN(2);
  TORCHAO_ACCUMULATE_COLUMN(3);
#undef TORCHAO_ACCUMULATE_COLUMN
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

template <int column_blocks>
struct WeightGroupMetadata {
  float32x4_t scales[column_blocks];
  int32x4_t qvals_sums[column_blocks];
  int32x4_t zeros[column_blocks];
};

template <int column_blocks, int column_offset, bool has_weight_zeros>
TORCHAO_ALWAYS_INLINE inline WeightGroupMetadata<column_blocks>
load_weight_group_metadata(const char*& weight_ptr, int group_size) {
  static_assert(column_blocks == 1 || column_blocks == 2);
  static_assert(column_offset == 0 || column_offset == 4);
  static_assert(column_offset + column_blocks * 4 <= kNr);

  WeightGroupMetadata<column_blocks> metadata;
  const float* scales = reinterpret_cast<const float*>(weight_ptr);
  for (int block = 0; block < column_blocks; block++) {
    metadata.scales[block] =
        vld1q_f32(scales + column_offset + block * 4);
  }
  weight_ptr += kNr * sizeof(float);

  const int32_t* qvals_sums = reinterpret_cast<const int32_t*>(weight_ptr);
  for (int block = 0; block < column_blocks; block++) {
    metadata.qvals_sums[block] =
        vld1q_s32(qvals_sums + column_offset + block * 4);
    metadata.zeros[block] = vdupq_n_s32(0);
  }
  weight_ptr += kNr * sizeof(int32_t);

  if constexpr (has_weight_zeros) {
    const int32_t* zeros = reinterpret_cast<const int32_t*>(weight_ptr);
    for (int block = 0; block < column_blocks; block++) {
      metadata.zeros[block] =
          vld1q_s32(zeros + column_offset + block * 4);
      metadata.qvals_sums[block] = vmlsq_n_s32(
          metadata.qvals_sums[block], metadata.zeros[block], group_size);
      metadata.zeros[block] =
          vaddq_s32(metadata.zeros[block], vdupq_n_s32(4));
    }
    weight_ptr += kNr * sizeof(int32_t);
  }
  return metadata;
}

template <int cols, int row_blocks, int column_offset>
TORCHAO_ALWAYS_INLINE inline void apply_post_ops(
    volatile float32x4_t (&results)[cols][row_blocks],
    const char* weight_ptr,
    float clamp_min,
    float clamp_max,
    bool has_bias,
    bool has_clamp) {
  if (has_bias) {
    const float* bias =
        reinterpret_cast<const float*>(weight_ptr) + column_offset;
    for (int col = 0; col < cols; col++) {
      const float32x4_t bias_vec = vdupq_n_f32(bias[col]);
      for (int block = 0; block < row_blocks; block++) {
        results[col][block] = vaddq_f32(results[col][block], bias_vec);
      }
    }
  }
  if (has_clamp) {
    const float32x4_t vec_min = vdupq_n_f32(clamp_min);
    const float32x4_t vec_max = vdupq_n_f32(clamp_max);
    for (int col = 0; col < cols; col++) {
      for (int block = 0; block < row_blocks; block++) {
        results[col][block] =
            vec_clamp(results[col][block], vec_min, vec_max);
      }
    }
  }
}

template <int column_blocks, int row_blocks, int column_offset = 0>
TORCHAO_ALWAYS_INLINE inline void store_results(
    float* output,
    int output_m_stride,
    int remaining_n,
    int valid_rows,
    volatile float32x4_t (&results)[column_blocks * 4][row_blocks]) {
  static_assert(column_blocks == 1 || column_blocks == 2);
  float32x4_t rows[column_blocks][4];
  for (int row_block = 0; row_block < row_blocks; row_block++) {
    const int block_rows = std::min(4, valid_rows - row_block * 4);
    if (block_rows <= 0) {
      break;
    }
    for (int column_block = 0; column_block < column_blocks; column_block++) {
      const int col = column_block * 4;
      transpose_4x4_f32(
          results[col][row_block],
          results[col + 1][row_block],
          results[col + 2][row_block],
          results[col + 3][row_block],
          rows[column_block][0],
          rows[column_block][1],
          rows[column_block][2],
          rows[column_block][3]);
    }
    for (int row = 0; row < block_rows; row++) {
      float* row_output =
          output + (row_block * 4 + row) * output_m_stride + column_offset;
      if constexpr (column_blocks == 2) {
        store_8_f32(
            row_output, remaining_n, rows[0][row], rows[1][row]);
      } else {
        store_4_f32(
            row_output, remaining_n - column_offset, rows[0][row]);
      }
    }
  }
}

template <
    int row_blocks,
    int column_blocks,
    int column_offset,
    bool has_weight_zeros>
__attribute__((noinline)) void kernel_rows_w3(
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
  static_assert(row_blocks == 1 || row_blocks == 2 || row_blocks == 4);
  static_assert(column_blocks == 1 || column_blocks == 2);
  static_assert(
      (row_blocks == 4 && column_blocks == 1) ||
      (row_blocks <= 2 && column_blocks == 2));
  static_assert(column_offset == 0 || column_offset == 4);
  static_assert(column_offset + column_blocks * 4 <= kNr);
  constexpr int activation_blocks = (row_blocks + 1) / 2;
  assert(valid_rows >= 1 && valid_rows <= row_blocks * 4);

  const char* activation_block[activation_blocks];
  activation_block[0] = activation_block_0;
  if constexpr (activation_blocks == 2) {
    activation_block[1] = activation_block_1;
  }

  // Read the per-row quantization headers from the packed eight-row blocks.
  float32x4_t activation_scale_vec[row_blocks];
  int32x4_t activation_zero_vec[row_blocks];
  const char* activation_ptrs[activation_blocks];
  for (int block = 0; block < activation_blocks; block++) {
    const float* scales =
        reinterpret_cast<const float*>(activation_block[block]);
    const int16x8_t zeros = vmovl_s8(vld1_s8(
        reinterpret_cast<const int8_t*>(
            activation_block[block] + kMr * sizeof(float))));
    activation_scale_vec[block * 2] = vld1q_f32(scales);
    activation_zero_vec[block * 2] = vmovl_s16(vget_low_s16(zeros));
    if constexpr (row_blocks > 1) {
      if (block * 2 + 1 < row_blocks) {
        activation_scale_vec[block * 2 + 1] = vld1q_f32(scales + 4);
        activation_zero_vec[block * 2 + 1] =
            vmovl_s16(vget_high_s16(zeros));
      }
    }
    activation_ptrs[block] =
        activation_block[block] + kMr * (sizeof(float) + sizeof(int8_t));
  }

  const char* weight_ptr = weight_panel;
  volatile float32x4_t results[column_blocks * 4][row_blocks];
  for (int col = 0; col < column_blocks * 4; col++) {
    for (int block = 0; block < row_blocks; block++) {
      results[col][block] = vdupq_n_f32(0.0f);
    }
  }

  // Accumulate and dequantize one K group at a time.
  for (int k_idx = 0; k_idx < k; k_idx += group_size) {
    int32x4_t accumulators[row_blocks][column_blocks * 4];
    for (int block = 0; block < row_blocks; block++) {
      for (int col = 0; col < column_blocks * 4; col++) {
        accumulators[block][col] = vdupq_n_s32(0);
      }
    }

    for (int i = 0; i < group_size; i += 16) {
      dot_rows_cols_w3<
          row_blocks,
          column_blocks,
          column_offset / 4,
          !has_weight_zeros>(
          accumulators,
          reinterpret_cast<const uint8_t*>(weight_ptr),
          activation_ptrs);
      weight_ptr += kBytesPer128W3Values;
      for (int block = 0; block < activation_blocks; block++) {
        activation_ptrs[block] += kMr * 16;
      }
    }

    // Packed group metadata follows the W3 values for this group.
    const auto weight_metadata =
        load_weight_group_metadata<
            column_blocks,
            column_offset,
            has_weight_zeros>(
            weight_ptr, group_size);

    int32x4_t activation_qvals_sum_vec[row_blocks];
    for (int block = 0; block < row_blocks; block++) {
      activation_qvals_sum_vec[block] = vdupq_n_s32(0);
    }
    if constexpr (has_weight_zeros) {
      for (int block = 0; block < row_blocks; block++) {
        activation_qvals_sum_vec[block] = vld1q_s32(
            reinterpret_cast<const int32_t*>(
                activation_ptrs[block / 2] + (block % 2) * 16));
      }
      for (int block = 0; block < activation_blocks; block++) {
        activation_ptrs[block] += kMr * sizeof(int32_t);
      }
    }

    accumulate_4_columns<
        row_blocks,
        column_blocks * 4,
        0,
        has_weight_zeros>(
        results,
        accumulators,
        activation_zero_vec,
        activation_scale_vec,
        weight_metadata.qvals_sums[0],
        weight_metadata.scales[0],
        weight_metadata.zeros[0],
        activation_qvals_sum_vec);
    if constexpr (column_blocks == 2) {
      accumulate_4_columns<row_blocks, 8, 4, has_weight_zeros>(
          results,
          accumulators,
          activation_zero_vec,
          activation_scale_vec,
          weight_metadata.qvals_sums[1],
          weight_metadata.scales[1],
          weight_metadata.zeros[1],
          activation_qvals_sum_vec);
    }
  }

  // Apply optional post-ops after all K groups have accumulated.
  apply_post_ops<column_blocks * 4, row_blocks, column_offset>(
      results, weight_ptr, clamp_min, clamp_max, has_bias, has_clamp);

  // Accumulators are column-major; transpose them before row-major stores.
  store_results<column_blocks, row_blocks, column_offset>(
      output, output_m_stride, remaining_n, valid_rows, results);
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
  // Extract one requested row from the same eight-row interleaved layout.
  const float activation_scale =
      reinterpret_cast<const float*>(activation_block)[row];
  const int32_t activation_zero = static_cast<int32_t>(
      reinterpret_cast<const int8_t*>(
          activation_block + kMr * sizeof(float))[row]);
  const char* activation_ptr =
      activation_block + kMr * (sizeof(float) + sizeof(int8_t));
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
      // Deinterleave four rows and select the row needed by this tail path.
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
      weight_ptr += kBytesPer128W3Values;
      activation_ptr += kMr * 16;
    }

    const auto weight_metadata =
        load_weight_group_metadata<2, 0, has_weight_zeros>(
            weight_ptr, group_size);

    int32x4_t corrected_0123 = vsubq_s32(
        vpaddq_s32(accumulators[0], accumulators[1]),
        vmulq_n_s32(weight_metadata.qvals_sums[0], activation_zero));
    int32x4_t corrected_4567 = vsubq_s32(
        vpaddq_s32(accumulators[2], accumulators[3]),
        vmulq_n_s32(weight_metadata.qvals_sums[1], activation_zero));
    if constexpr (has_weight_zeros) {
      const int32_t activation_qvals_sum =
          reinterpret_cast<const int32_t*>(activation_ptr)[row];
      corrected_0123 = vmlsq_n_s32(
          corrected_0123,
          weight_metadata.zeros[0],
          activation_qvals_sum);
      corrected_4567 = vmlsq_n_s32(
          corrected_4567,
          weight_metadata.zeros[1],
          activation_qvals_sum);
      activation_ptr += kMr * sizeof(int32_t);
    }
    const float32x4_t activation_scale_vec =
        vdupq_n_f32(activation_scale);
    result_0123 = vmlaq_f32(
        result_0123,
        vmulq_f32(weight_metadata.scales[0], activation_scale_vec),
        vcvtq_f32_s32(corrected_0123));
    result_4567 = vmlaq_f32(
        result_4567,
        vmulq_f32(weight_metadata.scales[1], activation_scale_vec),
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

template <bool has_weight_zeros>
TORCHAO_ALWAYS_INLINE inline void run_16_row_tile(
    float* output,
    int output_m_stride,
    int n,
    int k,
    int group_size,
    const char* weight_data,
    std::size_t weight_panel_size,
    const char* activation_block_0,
    std::size_t activation_block_size,
    int valid_rows,
    float clamp_min,
    float clamp_max,
    bool has_bias,
    bool has_clamp) {
  const char* activation_block_1 =
      activation_block_0 + activation_block_size;
  const char* weight_panel = weight_data;
  for (int n_idx = 0; n_idx < n; n_idx += kNr) {
    const int remaining_n = n - n_idx;
    kernel_rows_w3<4, 1, 0, has_weight_zeros>(
        output + n_idx,
        output_m_stride,
        remaining_n,
        k,
        group_size,
        weight_panel,
        activation_block_0,
        activation_block_1,
        valid_rows,
        clamp_min,
        clamp_max,
        has_bias,
        has_clamp);
    if (remaining_n > 4) {
      kernel_rows_w3<4, 1, 4, has_weight_zeros>(
          output + n_idx,
          output_m_stride,
          remaining_n,
          k,
          group_size,
          weight_panel,
          activation_block_0,
          activation_block_1,
          valid_rows,
          clamp_min,
          clamp_max,
          has_bias,
          has_clamp);
    }
    weight_panel += weight_panel_size;
  }
}

template <int row_blocks, bool has_weight_zeros>
TORCHAO_ALWAYS_INLINE inline void run_up_to_8_row_tile(
    float* output,
    int output_m_stride,
    int n,
    int k,
    int group_size,
    const char* weight_data,
    std::size_t weight_panel_size,
    const char* activation_block,
    int valid_rows,
    float clamp_min,
    float clamp_max,
    bool has_bias,
    bool has_clamp) {
  const char* weight_panel = weight_data;
  for (int n_idx = 0; n_idx < n; n_idx += kNr) {
    kernel_rows_w3<row_blocks, 2, 0, has_weight_zeros>(
        output + n_idx,
        output_m_stride,
        n - n_idx,
        k,
        group_size,
        weight_panel,
        activation_block,
        nullptr,
        valid_rows,
        clamp_min,
        clamp_max,
        has_bias,
        has_clamp);
    weight_panel += weight_panel_size;
  }
}

} // namespace internal

// Public eight-row-granularity entry point. It pairs packed activation blocks
// for a 16-row fast path, then uses dedicated 8-, 4-, and 1-row tail paths.
template <int weight_nbit, bool has_weight_zeros>
void kernel_8x8x16_f32_neondot(
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
  static_assert(weight_nbit == 3);

  constexpr int mr = 8;
  constexpr int nr = 8;
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

  // Process complete pairs of eight-row activation blocks.
  int m_idx = 0;
  for (; m_idx + 16 <= m; m_idx += 16) {
    const char* activation_block_0 =
        activation_data_bytes + m_idx * activation_row_size;
    internal::run_16_row_tile<has_weight_zeros>(
        output + m_idx * output_m_stride,
        output_m_stride,
        n,
        k,
        group_size,
        weight_data_bytes,
        weight_panel_size,
        activation_block_0,
        activation_block_size,
        16,
        clamp_min,
        clamp_max,
        has_bias,
        has_clamp);
  }

  // A 9..15-row tail uses the 16-row path with padded activation rows.
  if (m - m_idx > 8) {
    const int tail_rows = m - m_idx;
    const char* activation_block_0 =
        activation_data_bytes + m_idx * activation_row_size;
    internal::run_16_row_tile<has_weight_zeros>(
        output + m_idx * output_m_stride,
        output_m_stride,
        n,
        k,
        group_size,
        weight_data_bytes,
        weight_panel_size,
        activation_block_0,
        activation_block_size,
        tail_rows,
        clamp_min,
        clamp_max,
        has_bias,
        has_clamp);
    m_idx += tail_rows;
  } else if (m - m_idx > 4) {
    // A 5..8-row tail uses the eight-row path.
    const int tail_rows = m - m_idx;
    const char* activation_block =
        activation_data_bytes + m_idx * activation_row_size;
    internal::run_up_to_8_row_tile<2, has_weight_zeros>(
        output + m_idx * output_m_stride,
        output_m_stride,
        n,
        k,
        group_size,
        weight_data_bytes,
        weight_panel_size,
        activation_block,
        tail_rows,
        clamp_min,
        clamp_max,
        has_bias,
        has_clamp);
    m_idx += tail_rows;
  }

  // A 2..4-row tail uses one four-row accumulator block.
  if (m - m_idx > 1) {
    const int tail_rows = m - m_idx;
    const int activation_block_idx = m_idx / mr;
    const char* activation_block = activation_data_bytes +
        activation_block_idx * activation_block_size;
    internal::run_up_to_8_row_tile<1, has_weight_zeros>(
        output + m_idx * output_m_stride,
        output_m_stride,
        n,
        k,
        group_size,
        weight_data_bytes,
        weight_panel_size,
        activation_block,
        tail_rows,
        clamp_min,
        clamp_max,
        has_bias,
        has_clamp);
    m_idx += tail_rows;
  }

  // Handle the possible final row without repacking its interleaved block.
  if (m_idx < m) {
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
