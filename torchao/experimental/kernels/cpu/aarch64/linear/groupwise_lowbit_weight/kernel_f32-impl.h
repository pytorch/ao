// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#if defined(aarch64) || defined(__ARM_NEON)
#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight/pack_weights.h>
#include <torchao/experimental/kernels/cpu/aarch64/lut/lut.h>
#include <array>
#include <cassert>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut::
    kernel {

namespace lut_utils = torchao::lut;
namespace weight_packing = torchao::kernels::cpu::aarch64::linear::
    groupwise_lowbit_weight_lut::weight_packing;

namespace internal {

/*
 * @brief Computes a single tile of the output matrix.
 * @tparam weight_nbit_ The bit-precision of the quantized weight indices.
 * @tparam has_scales A compile-time flag to enable the application of scales.
 *
 * @param accum A NEON vector of 4 floats used as an in-out accumulator.
 * @param activation_tile_ptr Pointer to the 32-float activation tile.
 * @param packed_indices_ptr Pointer to the bit-packed weight indices.
 * @param lut_neon The dequantization LUT, pre-formatted for NEON lookups.
 * @param scale_vec A NEON vector with the four dequantization scales.
 */
template <int weight_nbit_, bool has_scales>
TORCHAO_ALWAYS_INLINE static inline void compute_tile_1x4x32(
    float32x4_t& accum,
    const float* __restrict__ activation_tile_ptr,
    const uint8_t* __restrict__ packed_indices_ptr,
    const uint8x16x4_t& lut_neon,
    const float32x4_t scale_vec) {
  // 1. Unpack indices
  uint8x16_t idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;
  bitpacking::vec_unpack_128_uintx_values<weight_nbit_>(
      idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, packed_indices_ptr);

  const std::array<uint8x16_t, 8> unpacked_indices = {
      idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7};

  for (int sr_idx = 0; sr_idx < 8; ++sr_idx) {
    // Load the 4 activations corresponding to this chunk
    const float* activation_chunk_ptr = activation_tile_ptr + sr_idx * 4;
    float32x4_t a = vld1q_f32(activation_chunk_ptr);

    // Lookup the 4x4 weight sub-tile (as columns)
    float32x4_t w_col0, w_col1, w_col2, w_col3;
    lut_utils::lookup_from_fp32_lut(
        w_col0, w_col1, w_col2, w_col3, lut_neon, unpacked_indices[sr_idx]);

    float32x4x2_t tmp0 = vtrnq_f32(w_col0, w_col1);
    float32x4x2_t tmp1 = vtrnq_f32(w_col2, w_col3);
    float32x4_t w_row0 =
        vcombine_f32(vget_low_f32(tmp0.val[0]), vget_low_f32(tmp1.val[0]));
    float32x4_t w_row1 =
        vcombine_f32(vget_low_f32(tmp0.val[1]), vget_low_f32(tmp1.val[1]));
    float32x4_t w_row2 =
        vcombine_f32(vget_high_f32(tmp0.val[0]), vget_high_f32(tmp1.val[0]));
    float32x4_t w_row3 =
        vcombine_f32(vget_high_f32(tmp0.val[1]), vget_high_f32(tmp1.val[1]));

    // Conditionally apply scales at compile time
    if constexpr (has_scales) {
      w_row0 = vmulq_f32(w_row0, scale_vec);
      w_row1 = vmulq_f32(w_row1, scale_vec);
      w_row2 = vmulq_f32(w_row2, scale_vec);
      w_row3 = vmulq_f32(w_row3, scale_vec);
    }

    // Use vfmaq_n_f32 to multiply each row vector by the corresponding scalar
    // activation.
    accum = vfmaq_n_f32(
        accum, w_row0, vgetq_lane_f32(a, 0)); // accum += w_row0 * a[0]
    accum = vfmaq_n_f32(
        accum, w_row1, vgetq_lane_f32(a, 1)); // accum += w_row1 * a[1]
    accum = vfmaq_n_f32(
        accum, w_row2, vgetq_lane_f32(a, 2)); // accum += w_row2 * a[2]
    accum = vfmaq_n_f32(
        accum, w_row3, vgetq_lane_f32(a, 3)); // accum += w_row3 * a[3]
  }
}

/**
 * @brief Stores the accumulated values to the output matrix.
 * @tparam mr_ The row-tiling factor of the micro-kernel.
 * @tparam nr_ The column-tiling factor of the micro-kernel.
 *
 * @param output The output matrix.
 * @param ldc The leading dimension of the output matrix.
 * @param n_cols The number of columns in the output matrix.
 * @param n_tile_start The starting column index of the current tile.
 * @param accum The accumulated values.
 * @param bias_ptr The pointer to the bias vector.
 * @param has_clamp Whether to apply clamping.
 * @param clamp_min_vec The minimum value for clamping.
 * @param clamp_max_vec The maximum value for clamping.
 */
template <int mr_, int nr_>
TORCHAO_ALWAYS_INLINE static inline void post_process_and_store(
    float* __restrict__ output,
    int ldc,
    int n_cols,
    int n_tile_start,
    const float32x4_t accum[mr_][nr_ / 4],
    const float* __restrict__ bias_ptr,
    bool has_clamp,
    const float32x4_t& clamp_min_vec,
    const float32x4_t& clamp_max_vec) {
  constexpr int NR_VEC = nr_ / 4;
  for (int m = 0; m < mr_; ++m) {
    float* out_row = output + m * ldc;
    for (int nb = 0; nb < NR_VEC; ++nb) {
      float32x4_t res = accum[m][nb];
      if (bias_ptr != nullptr) {
        float32x4_t bias_vec = vld1q_f32(bias_ptr + nb * 4);
        res = vaddq_f32(res, bias_vec);
      }
      if (has_clamp) {
        res = vmaxq_f32(res, clamp_min_vec);
        res = vminq_f32(res, clamp_max_vec);
      }

      const int current_n_offset = n_tile_start + nb * 4;
      const int remaining_cols = n_cols - current_n_offset;
      if (remaining_cols < 4) {
        float temp_res[4];
        vst1q_f32(temp_res, res);
        for (int i = 0; i < remaining_cols; ++i) {
          *(out_row + current_n_offset + i) = temp_res[i];
        }
      } else {
        vst1q_f32(out_row + current_n_offset, res);
      }
    }
  }
}

} // namespace internal

/*
 * @brief The main kernel for groupwise low-bit weight LUT.
 */
template <int weight_nbit_, bool has_scales>
void groupwise_lowbit_weight_lut_kernel_1x4x32(
    float* output,
    int output_m_stride,
    int m,
    int n,
    int k,
    int scale_group_size,
    int lut_group_size,
    const void* packed_weights,
    const void* packed_activations,
    float clamp_min,
    float clamp_max,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr_ = 1;
  constexpr int nr_ = 4;
  constexpr int kr_ = 32;

  const auto* typed_activations_ptr =
      static_cast<const float*>(packed_activations);
  const float32x4_t clamp_min_vec = vdupq_n_f32(clamp_min);
  const float32x4_t clamp_max_vec = vdupq_n_f32(clamp_max);
  constexpr int bytes_per_weight_tile = ((nr_ * kr_ * weight_nbit_) + 7) / 8;

  for (int m_tile_start = 0; m_tile_start < m; m_tile_start += mr_) {
    const float* activation_row_ptr = typed_activations_ptr + m_tile_start * k;
    const uint8_t* packed_ptr = static_cast<const uint8_t*>(packed_weights);

    for (int n_tile_start = 0; n_tile_start < n; n_tile_start += nr_) {
      float32x4_t accumulators[mr_][nr_ / 4] = {{vdupq_n_f32(0.0f)}};

      uint8x16x4_t lut_neon;
      // Load the 16-float LUT for this tile.
      lut_utils::load_fp32_lut(
          lut_neon, reinterpret_cast<const float*>(packed_ptr));
      // Advance the pointer past the LUT.
      packed_ptr += 16 * sizeof(float);
      float32x4_t scale_vec = vdupq_n_f32(1.0f);
      for (int k_tile_start = 0; k_tile_start < k; k_tile_start += kr_) {
        if constexpr (has_scales) {
          const float* scale_for_tile = nullptr;

          if (k_tile_start % scale_group_size == 0) {
            scale_for_tile = reinterpret_cast<const float*>(packed_ptr);
            scale_vec = vld1q_f32(scale_for_tile);
            packed_ptr += nr_ * sizeof(float);
          }
        }

        // The current packed_ptr points to the weight indices.
        const uint8_t* indices_ptr = packed_ptr;

        internal::compute_tile_1x4x32<weight_nbit_, has_scales>(
            accumulators[0][0],
            activation_row_ptr + k_tile_start,
            indices_ptr,
            lut_neon,
            scale_vec);

        // Advance pointer past the weights that were just used.
        packed_ptr += bytes_per_weight_tile;
      }

      const float* bias_for_tile = nullptr;
      if (has_bias) {
        bias_for_tile = reinterpret_cast<const float*>(packed_ptr);
        packed_ptr += nr_ * sizeof(float);
      }

      float* output_row_ptr = output + m_tile_start * output_m_stride;
      internal::post_process_and_store<mr_, nr_>(
          output_row_ptr,
          output_m_stride,
          n,
          n_tile_start,
          accumulators,
          bias_for_tile,
          has_clamp,
          clamp_min_vec,
          clamp_max_vec);
    }
  }
}
} // namespace
  // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut::kernel
#endif // defined(aarch64) || defined(__ARM_NEON)
