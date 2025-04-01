// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <cassert>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight::kernel {
namespace internal {

inline float32x4_t
vec_clamp(float32x4_t x, float32x4_t vec_min, float32x4_t vec_max) {
  float32x4_t tmp = vmaxq_f32(x, vec_min);
  return vminq_f32(tmp, vec_max);
}
} // namespace internal

// Implements variants of
// channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot
// to compute
//    output = F(activations * weights + bias)
// where
//
// * activations are mxk and transposed, stored in row-major order.
// * weights are kxn and transposed, stored in column-major order.
//   (can also be viewed as nxk non-transposed weights stored in row-major
//   order).
// * F is an element-wise activation function, either clamp (has_clamp = true)
//   or linear (has_clamp = false).
// * output is mxn.
//
// The suffix 1x8x16_f32_neondot indicates the tile sizes (1x8 = 1x16 @ 16x4),
// floating point type for output (f32), and main ISA instruction (neon_dot).
// There are 64 = 4*16 weight values unpacked in each inner loop iteration.
//
// Activations are channelwise 8-bit quantized, with a scale and zero per row
// Weights are groupwise lowbit (weight_nbit) quantized with a scale (and zero
// if has_weight_zeros = true) per group.
//
// Both activations and weights are dequantized with
//    scale * (qval - zero)
//
// The output is computed by dequantizing the activations and weights and
// computing F(activations * weights + bias).
//
// Activations and weights are stored in a prepared format specific to
// this kernel.  See prepare_weight_data_impl and prepare_activation_data_impl
// functions for details.
//
// Roughly inspired by
// https://gitlab.arm.com/kleidi/kleidiai/-/blob/main/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.c?ref_type=heads

template <int weight_nbit, bool has_lut>
void kernel_1x8x16_f32_neondot(
    // Outputs
    float32_t* output,
    // Inputs
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* weight_data,
    const void* activation_data,
    // Ignored if has_clamp is false
    float clamp_min,
    float clamp_max,
    bool has_weight_zeros,
    bool has_bias,
    bool has_clamp) {
  assert(k % group_size == 0);
  assert(group_size % 16 == 0);

  int8x16_t lut;
  if constexpr (!has_lut) {
    (void)lut; // unused
  }

  constexpr int bytes_per_128_weight_values = 16 * weight_nbit;

  auto activation_data_byte_ptr = (char*)(activation_data);
  char* activation_ptr;

  for (int m_idx = 0; m_idx < m; m_idx++) {
    // Read activation scale and zero
    float activation_scale = *((float*)activation_data_byte_ptr);
    activation_data_byte_ptr += sizeof(float);

    int activation_zero = (int)(*((int8_t*)activation_data_byte_ptr));
    activation_data_byte_ptr += sizeof(int8_t);

    // Set weight_data_byte_ptr to start of weight_data
    auto weight_data_byte_ptr = (char*)(weight_data);

    // Loop over 8 cols at a time
    // Weights and activations are padded when prepared, so the
    // reads are legal, even if on a partial tile
    for (int n_idx = 0; n_idx < n; n_idx += 8) {
      if constexpr (has_lut) {
        lut = vld1q_s8((int8_t*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;
      }

      // Set activation_ptr to start of activation qvals for row m_idx
      activation_ptr = activation_data_byte_ptr;
      float32x4_t res_0123 = vdupq_n_f32(0.0);
      float32x4_t res_4567 = vdupq_n_f32(0.0);

      // Loop k_idx by group
      for (int k_idx = 0; k_idx < k; k_idx += group_size) {
        // Iterating over k in chunks of 16, we compute the dot product
        // between 16 values of activation data with 16 values in each of 8 cols
        // of weight data. These dot products are stored in accumulators
        // acc_cols0011, acc_cols2233, acc_cols4455, acc_cols6677
        // as indicated in the below table:
        //
        // weight data          activation data     accumulator
        // -------------------------------------------------------
        // 1st 8 vals of col0   1st 8 vals          acc_cols0011[0]
        // 2nd 8 vals of col0   2nd 8 vals          acc_cols0011[1]
        // 1st 8 vals of col1   1st 8 vals          acc_cols0011[2]
        // 2nd 8 vals of col1   2nd 8 vals          acc_cols0011[3]
        // 1st 8 vals of col2   1st 8 vals          acc_cols2233[0]
        // 2nd 8 vals of col2   2nd 8 vals          acc_cols2233[1]
        // 1st 8 vals of col3   1st 8 vals          acc_cols2233[2]
        // 2nd 8 vals of col3   2nd 8 vals          acc_cols2233[3]
        // 1st 8 vals of col4   1st 8 vals          acc_cols4455[0]
        // 2nd 8 vals of col4   2nd 8 vals          acc_cols4455[1]
        // 1st 8 vals of col5   1st 8 vals          acc_cols4455[2]
        // 2nd 8 vals of col5   2nd 8 vals          acc_cols4455[3]
        // 1st 8 vals of col6   1st 8 vals          acc_cols6677[0]
        // 2nd 8 vals of col6   2nd 8 vals          acc_cols6677[1]
        // 1st 8 vals of col7   1st 8 vals          acc_cols6677[2]
        // 2nd 8 vals of col7   2nd 8 vals          acc_cols6677[3]
        //
        // The above computation scheme is what informs the weight valpacking
        int32x4_t acc_cols0011 = vdupq_n_s32(0);
        int32x4_t acc_cols2233 = vdupq_n_s32(0);
        int32x4_t acc_cols4455 = vdupq_n_s32(0);
        int32x4_t acc_cols6677 = vdupq_n_s32(0);

        // holds chunk of 16 activation_q values
        int8x16_t act_q;

        // holds chunk of 8 activation vals, duplicated twice
        int8x16_t act_q_dup;

        // holds chunk of 8 vals from weight_q col0, followed by 8 vals from
        // weight_q col1
        int8x16_t weight_q_cols01_0;
        int8x16_t weight_q_cols01_1;

        // holds chunk of 8 vals from weight_q col2, followed by 8 vals from
        // weight_q col3
        int8x16_t weight_q_cols23_0;
        int8x16_t weight_q_cols23_1;

        // holds chunk of 8 vals from weight_q col4, followed by 8 vals from
        // weight_q col5
        int8x16_t weight_q_cols45_0;
        int8x16_t weight_q_cols45_1;

        // holds chunk of 8 vals from weight_q col6, followed by 8 vals from
        // weight_q col7
        int8x16_t weight_q_cols67_0;
        int8x16_t weight_q_cols67_1;

        for (int i = 0; i < group_size; i += 16) {
          // Each chunk is 64 values of unpacked data (4 cols x 16 vals/col).
          // This comes out to (64 * weight_nbit / 8) bits = 8 * weight_nbit
          // bytes of bitpacked data

          if constexpr (has_lut) {
            torchao::bitpacking::vec_unpack_128_lowbit_values_with_lut<
                weight_nbit>(
                weight_q_cols01_0,
                weight_q_cols23_0,
                weight_q_cols45_0,
                weight_q_cols67_0,
                weight_q_cols01_1,
                weight_q_cols23_1,
                weight_q_cols45_1,
                weight_q_cols67_1,
                (uint8_t*)weight_data_byte_ptr,
                lut);
          } else {
            torchao::bitpacking::vec_unpack_128_lowbit_values<weight_nbit>(
                weight_q_cols01_0,
                weight_q_cols23_0,
                weight_q_cols45_0,
                weight_q_cols67_0,
                weight_q_cols01_1,
                weight_q_cols23_1,
                weight_q_cols45_1,
                weight_q_cols67_1,
                (uint8_t*)weight_data_byte_ptr);
          }

          weight_data_byte_ptr += bytes_per_128_weight_values;

          // Load 16 activation values
          act_q = vld1q_s8((int8_t*)activation_ptr);
          activation_ptr += 16;

          // Dot product of first 8 vals of activation data with first 8 vals of
          // weight data.  Note the sequence of operations here imply the
          // following order on weight_data stored in unpacked_buffer: (1st 8
          // vals col0), (1st 8 vals col1), (1st 8 vals col2), (1st 8 vals
          // col2).  This order is accomplished by valpacking
          act_q_dup = vcombine_s8(vget_low_s8(act_q), vget_low_s8(act_q));
          acc_cols0011 = vdotq_s32(acc_cols0011, weight_q_cols01_0, act_q_dup);
          acc_cols2233 = vdotq_s32(acc_cols2233, weight_q_cols23_0, act_q_dup);
          acc_cols4455 = vdotq_s32(acc_cols4455, weight_q_cols45_0, act_q_dup);
          acc_cols6677 = vdotq_s32(acc_cols6677, weight_q_cols67_0, act_q_dup);

          // Dot product of second 8 vals of activation data with second 8 vals
          // of weight data.
          act_q_dup = vcombine_s8(vget_high_s8(act_q), vget_high_s8(act_q));
          acc_cols0011 = vdotq_s32(acc_cols0011, weight_q_cols01_1, act_q_dup);
          acc_cols2233 = vdotq_s32(acc_cols2233, weight_q_cols23_1, act_q_dup);
          acc_cols4455 = vdotq_s32(acc_cols4455, weight_q_cols45_1, act_q_dup);
          acc_cols6677 = vdotq_s32(acc_cols6677, weight_q_cols67_1, act_q_dup);
        }
        // Reduce accumulators, so we have one dot product value per col
        int32x4_t qval_dot_0123 = vpaddq_s32(acc_cols0011, acc_cols2233);
        int32x4_t qval_dot_4567 = vpaddq_s32(acc_cols4455, acc_cols6677);

        // Result is updated with:
        // res += scale_factor * (qval_dot - term1 - term2 + term3), where
        // * scale_factor = (weight_scale * activation_scale)
        // * term1 = (activation_zero * weight_qvals_sum)
        // * term2 = (weight_zero * activation_qvals_sum)
        // * term3 = (group_size * weight_zero * activation_zero)
        // If has_weight_zeros is false, terms 2 and 3 disappaer.

        // Compute scale_factor
        float32x4_t activation_scales = vdupq_n_f32(activation_scale);

        float32x4_t weight_scales = vld1q_f32((float*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;

        float32x4_t scale_factor_0123 =
            vmulq_f32(weight_scales, activation_scales);

        weight_scales = vld1q_f32((float*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;

        float32x4_t scale_factor_4567 =
            vmulq_f32(weight_scales, activation_scales);

        // Compute term1
        int32x4_t weight_qvals_sum = vld1q_s32((int32_t*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;

        int32x4_t term1_0123 = vmulq_n_s32(weight_qvals_sum, activation_zero);

        weight_qvals_sum = vld1q_s32((int32_t*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;

        int32x4_t term1_4567 = vmulq_n_s32(weight_qvals_sum, activation_zero);

        if (has_weight_zeros) {
          // Compute term2 and term3

          int32_t activation_qvals_sum = *((int32_t*)activation_ptr);
          activation_ptr += sizeof(int32_t);

          int32x4_t weight_zeros = vld1q_s32((int32_t*)weight_data_byte_ptr);
          weight_data_byte_ptr += 16;

          int32x4_t term2_0123 =
              vmulq_n_s32(weight_zeros, activation_qvals_sum);

          int32x4_t term3_0123 =
              vmulq_n_s32(weight_zeros, group_size * activation_zero);

          weight_zeros = vld1q_s32((int32_t*)weight_data_byte_ptr);
          weight_data_byte_ptr += 16;

          int32x4_t term2_4567 =
              vmulq_n_s32(weight_zeros, activation_qvals_sum);

          int32x4_t term3_4567 =
              vmulq_n_s32(weight_zeros, group_size * activation_zero);

          // Do updates
          int32x4_t tmp = vsubq_s32(qval_dot_0123, term1_0123);
          tmp = vsubq_s32(tmp, term2_0123);
          tmp = vaddq_s32(tmp, term3_0123);
          res_0123 = vmlaq_f32(res_0123, scale_factor_0123, vcvtq_f32_s32(tmp));

          tmp = vsubq_s32(qval_dot_4567, term1_4567);
          tmp = vsubq_s32(tmp, term2_4567);
          tmp = vaddq_s32(tmp, term3_4567);
          res_4567 = vmlaq_f32(res_4567, scale_factor_4567, vcvtq_f32_s32(tmp));
        } else {
          // Do updates
          int32x4_t tmp = vsubq_s32(qval_dot_0123, term1_0123);
          res_0123 = vmlaq_f32(res_0123, scale_factor_0123, vcvtq_f32_s32(tmp));

          tmp = vsubq_s32(qval_dot_4567, term1_4567);
          res_4567 = vmlaq_f32(res_4567, scale_factor_4567, vcvtq_f32_s32(tmp));
        }

      } // k_idx
      if (has_bias) {
        float32x4_t bias;

        bias = vld1q_f32((float32_t*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;
        res_0123 = vaddq_f32(res_0123, bias);

        bias = vld1q_f32((float32_t*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;
        res_4567 = vaddq_f32(res_4567, bias);
      }
      if (has_clamp) {
        float32x4_t vec_min = vdupq_n_f32(clamp_min);
        float32x4_t vec_max = vdupq_n_f32(clamp_max);
        res_0123 = internal::vec_clamp(res_0123, vec_min, vec_max);
        res_4567 = internal::vec_clamp(res_4567, vec_min, vec_max);
      }

      // Store result
      int remaining = n - n_idx;
      float* store_loc = output + m_idx * output_m_stride + n_idx;
      if (remaining >= 8) {
        vst1q_f32(store_loc, res_0123);
        vst1q_f32(store_loc + 4, res_4567);
      } else if (remaining >= 7) {
        vst1q_f32(store_loc, res_0123);
        vst1_f32(store_loc + 4, vget_low_f32(res_4567));
        *(store_loc + 6) = res_4567[2];
      } else if (remaining >= 6) {
        vst1q_f32(store_loc, res_0123);
        vst1_f32(store_loc + 4, vget_low_f32(res_4567));
      } else if (remaining >= 5) {
        vst1q_f32(store_loc, res_0123);
        *(store_loc + 4) = res_4567[0];
      } else if (remaining >= 4) {
        vst1q_f32(store_loc, res_0123);
      } else if (remaining >= 3) {
        vst1_f32(store_loc, vget_low_f32(res_0123));
        *(store_loc + 2) = res_0123[2];
      } else if (remaining >= 2) {
        vst1_f32(store_loc, vget_low_f32(res_0123));
      } else {
        *store_loc = res_0123[0];
      }

    } // n_idx
    activation_data_byte_ptr += (activation_ptr - activation_data_byte_ptr);
  } // m_idx
}

} // namespace
  // torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::kernel

#endif // defined(__aarch64__) || defined(__ARM_NEON)
