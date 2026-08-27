// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/kernel_4x8x16_f32_neondot-impl.h>
#include <cassert>
#include <cstddef>

namespace torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight::kernel {

// Computes two four-row blocks together, reusing each unpacked weight block
// across all eight rows. FP32 partial results are kept separately from the
// integer dot-product accumulators to limit pressure in the inner loop.
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

  constexpr int mr = 8;
  constexpr int row_blocks = 2;
  constexpr int nr = 8;
  constexpr int bytes_per_128_weight_values = 16 * weight_nbit;
  const std::size_t activation_row_size = sizeof(float) + sizeof(int8_t) + k +
      (has_weight_zeros ? (k / group_size) * sizeof(int32_t) : 0);
  const auto* activation_data_bytes = static_cast<const char*>(activation_data);

  int m_idx = 0;
  for (; m_idx + mr <= m; m_idx += mr) {
    const char* activation_block =
        activation_data_bytes + m_idx * activation_row_size;
    float32x4_t activation_scale_vec[row_blocks] = {
        vld1q_f32(reinterpret_cast<const float*>(activation_block)),
        vld1q_f32(reinterpret_cast<const float*>(activation_block + 16))};
    activation_block += mr * sizeof(float);
    int8x8_t activation_zeros_s8 =
        vld1_s8(reinterpret_cast<const int8_t*>(activation_block));
    activation_block += mr * sizeof(int8_t);
    int16x8_t activation_zeros_s16 = vmovl_s8(activation_zeros_s8);
    int32x4_t activation_zero_vec[row_blocks] = {
        vmovl_s16(vget_low_s16(activation_zeros_s16)),
        vmovl_s16(vget_high_s16(activation_zeros_s16))};

    const char* weight_data_bytes = static_cast<const char*>(weight_data);
    for (int n_idx = 0; n_idx < n; n_idx += nr) {
      const char* activation_ptr = activation_block;
      float32x4_t results[row_blocks][nr];
      for (int block = 0; block < row_blocks; block++) {
        for (int col = 0; col < nr; col++) {
          results[block][col] = vdupq_n_f32(0.0f);
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

          for (int block = 0; block < row_blocks; block++) {
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
                accumulators[block],
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

        int32x4_t activation_qvals_sum_vec[row_blocks];
        if constexpr (has_weight_zeros) {
          activation_qvals_sum_vec[0] =
              vld1q_s32(reinterpret_cast<const int32_t*>(activation_ptr));
          activation_qvals_sum_vec[1] =
              vld1q_s32(reinterpret_cast<const int32_t*>(activation_ptr + 16));
          activation_ptr += mr * sizeof(int32_t);
        }

        for (int block = 0; block < row_blocks; block++) {
          for (int col = 0; col < nr; col++) {
            int32x4_t corrected = vsubq_s32(
                accumulators[block][col],
                vmulq_n_s32(
                    activation_zero_vec[block], weight_qvals_sums[col]));
            if constexpr (has_weight_zeros) {
              corrected = vsubq_s32(
                  corrected,
                  vmulq_n_s32(
                      activation_qvals_sum_vec[block], weight_zeros[col]));
              corrected = vaddq_s32(
                  corrected,
                  vmulq_n_s32(
                      activation_zero_vec[block],
                      group_size * weight_zeros[col]));
            }
            float32x4_t scale_factor =
                vmulq_n_f32(activation_scale_vec[block], weight_scales[col]);
            results[block][col] = vmlaq_f32(
                results[block][col], scale_factor, vcvtq_f32_s32(corrected));
          }
        }
      }

      if (has_bias) {
        const float* bias = reinterpret_cast<const float*>(weight_data_bytes);
        weight_data_bytes += nr * sizeof(float);
        for (int block = 0; block < row_blocks; block++) {
          for (int col = 0; col < nr; col++) {
            results[block][col] =
                vaddq_f32(results[block][col], vdupq_n_f32(bias[col]));
          }
        }
      }
      if (has_clamp) {
        float32x4_t vec_min = vdupq_n_f32(clamp_min);
        float32x4_t vec_max = vdupq_n_f32(clamp_max);
        for (int block = 0; block < row_blocks; block++) {
          for (int col = 0; col < nr; col++) {
            results[block][col] =
                internal::vec_clamp(results[block][col], vec_min, vec_max);
          }
        }
      }

      for (int block = 0; block < row_blocks; block++) {
        float32x4_t output_0123[4];
        float32x4_t output_4567[4];
        internal::transpose_4x4_f32(
            results[block][0],
            results[block][1],
            results[block][2],
            results[block][3],
            output_0123[0],
            output_0123[1],
            output_0123[2],
            output_0123[3]);
        internal::transpose_4x4_f32(
            results[block][4],
            results[block][5],
            results[block][6],
            results[block][7],
            output_4567[0],
            output_4567[1],
            output_4567[2],
            output_4567[3]);
        const int remaining = n - n_idx;
        for (int row = 0; row < 4; row++) {
          internal::store_8_f32(
              output + (m_idx + block * 4 + row) * output_m_stride + n_idx,
              remaining,
              output_0123[row],
              output_4567[row]);
        }
      }
    }
  }

  if (m_idx < m) {
    kernel_4x8x16_f32_neondot<weight_nbit, has_weight_zeros>(
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
