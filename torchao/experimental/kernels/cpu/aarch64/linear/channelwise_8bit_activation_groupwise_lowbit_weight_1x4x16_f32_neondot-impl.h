// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_prepare_activation_data_1xk_f32-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/pack_weights.h>
#include <cassert>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear {
namespace channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
    internal {

inline float32x4_t clamp(float32x4_t x, float min, float max) {
  float32x4_t vec_min = vdupq_n_f32(min);
  float32x4_t vec_max = vdupq_n_f32(max);
  float32x4_t tmp = vmaxq_f32(x, vec_min);
  return vminq_f32(tmp, vec_max);
}

// Implements variants of
// channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot
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
// The suffix 1x4x16_f32_neondot indicates the tile sizes (1x4 = 1x16 @ 16x4),
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
// Kernel is roughly modeled on
// https://gitlab.arm.com/kleidi/kleidiai/-/blob/main/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c

template <int weight_nbit>
void kernel_impl(
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

  constexpr int bytes_per_64_weight_values = 8 * weight_nbit;

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

    // Loop over 4 cols at a time
    // Weights and activations are padded when prepared, so the
    // reads are legal, even if on a partial tile
    for (int n_idx = 0; n_idx < n; n_idx += 4) {
      // Set activation_ptr to start of activation qvals for row m_idx
      activation_ptr = activation_data_byte_ptr;
      float32x4_t res = vdupq_n_f32(0.0);

      // Loop k_idx by group
      for (int k_idx = 0; k_idx < k; k_idx += group_size) {
        // Iterating over k in chunks of 16, we compute the dot product
        // between 16 values of activation data with 16 values in each of 4 cols
        // of weight data. These dot products are stored in accumulators
        // acc_cols0011 and acc_cols2233 as indicated in the below table:
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
        //
        // The above computation scheme is what informs the weight valpacking
        int32x4_t acc_cols0011 = vdupq_n_s32(0);
        int32x4_t acc_cols2233 = vdupq_n_s32(0);

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

        for (int i = 0; i < group_size; i += 16) {
          // Each chunk is 64 values of unpacked data (4 cols x 16 vals/col).
          // This comes out to (64 * weight_nbit / 8) bits = 8 * weight_nbit
          // bytes of bitpacked data
          torchao::bitpacking::vec_unpack_64_lowbit_values<weight_nbit>(
              weight_q_cols01_0,
              weight_q_cols23_0,
              weight_q_cols01_1,
              weight_q_cols23_1,
              (uint8_t*)weight_data_byte_ptr);
          weight_data_byte_ptr += bytes_per_64_weight_values;

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

          // Dot product of second 8 vals of activation data with second 8 vals
          // of weight data.
          act_q_dup = vcombine_s8(vget_high_s8(act_q), vget_high_s8(act_q));
          acc_cols0011 = vdotq_s32(acc_cols0011, weight_q_cols01_1, act_q_dup);
          acc_cols2233 = vdotq_s32(acc_cols2233, weight_q_cols23_1, act_q_dup);
        }
        // Reduce accumulators, so we have one dot product value per col
        int32x4_t qval_dot = vpaddq_s32(acc_cols0011, acc_cols2233);

        // Dequantize and accumulate in result
        float32x4_t weight_scales = vld1q_f32((float*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;

        int32x4_t weight_qvals_sum = vld1q_s32((int32_t*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;

        float32x4_t scale_factor =
            vmulq_f32(weight_scales, vdupq_n_f32(activation_scale));
        int32x4_t term1 = vmulq_n_s32(weight_qvals_sum, activation_zero);

        if (has_weight_zeros) {
          // Compute
          // res += (weight_scale * activation_scale) *
          //     (qval_dot - (activation_zero * weight_qvals_sum) -
          //      (weight_zero * activation_qvals_sum) +
          //      (group_size * weight_zero * activation_zero));

          int32_t activation_qvals_sum = *((int32_t*)activation_ptr);
          activation_ptr += sizeof(int32_t);

          int32x4_t weight_zeros = vld1q_s32((int32_t*)weight_data_byte_ptr);
          weight_data_byte_ptr += 16;

          int32x4_t term2 = vmulq_n_s32(weight_zeros, activation_qvals_sum);
          int32x4_t term3 =
              vmulq_n_s32(weight_zeros, group_size * activation_zero);

          int32x4_t tmp = vsubq_s32(qval_dot, term1);
          tmp = vsubq_s32(tmp, term2);
          tmp = vaddq_s32(tmp, term3);
          res = vmlaq_f32(res, scale_factor, vcvtq_f32_s32(tmp));
        } else {
          // Compute
          //  res += (weight_scale * activation_scale) *
          //     (qval_dot - activation_zero * weight_qvals_sum);
          auto tmp = vsubq_s32(qval_dot, term1);
          res = vmlaq_f32(res, scale_factor, vcvtq_f32_s32(tmp));
        }

      } // k_idx
      if (has_bias) {
        float32x4_t bias = vld1q_f32((float32_t*)weight_data_byte_ptr);
        weight_data_byte_ptr += 16;
        res = vaddq_f32(res, bias);
      }
      if (has_clamp) {
        res = clamp(res, clamp_min, clamp_max);
      }

      // Store result
      int remaining = n - n_idx;
      float* store_loc = output + m_idx * output_m_stride + n_idx;
      if (remaining >= 4) {
        vst1q_f32(store_loc, res);
      } else if (remaining >= 3) {
        vst1_f32(store_loc, vget_low_f32(res));
        *(store_loc + 2) = res[2];
      } else if (remaining >= 2) {
        vst1_f32(store_loc, vget_low_f32(res));
      } else {
        *(store_loc) = res[0];
      }

    } // n_idx
    activation_data_byte_ptr += (activation_ptr - activation_data_byte_ptr);
  } // m_idx
}

// Prepares weight data for kernel_impl.

// Returns number of bytes required for weight_data
size_t inline weight_data_size_impl(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias) {
  return torchao::kernels::cpu::aarch64::linear::packing::packed_weights_size(
      n,
      k,
      group_size,
      weight_nbit,
      has_weight_zeros,
      has_bias,
      /*nr*/ 4);
}

template <int weight_nbit>
void prepare_weight_data_impl(
    // Output
    void* weight_data,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    // Ignored if has_weight_zeros = false
    const int8_t* weight_zeros,
    const float* bias) {
  torchao::kernels::cpu::aarch64::linear::packing::
      pack_weights<weight_nbit, /*nr*/ 4, /*kr*/ 16, /*sr*/ 2>(
          weight_data,
          n,
          k,
          group_size,
          weight_qvals,
          weight_scales,
          weight_zeros,
          bias);
}

} // namespace
  // channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::internal
} // namespace torchao::kernels::cpu::aarch64::linear

// Activation functions
size_t torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
        activation_data_size(
            int m,
            int k,
            int group_size,
            bool has_weight_zeros) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_prepare_activation_data_1xk_f32::internal::
          activation_data_size_impl(m, k, group_size, has_weight_zeros);
}

void torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
        prepare_activation_data(
            void* activation_data,
            // Inputs
            int m,
            int k,
            // Ignored if has_weight_zeros = false
            int group_size,
            const float* activations,
            bool has_weight_zeros) {
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_prepare_activation_data_1xk_f32::internal::
          prepare_activation_data_impl(
              activation_data, m, k, group_size, activations, has_weight_zeros);
}

// Weight functions
template <int weight_nbit>
size_t torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
        weight_data_size(
            int n,
            int k,
            int group_size,
            bool has_weight_zeros,
            bool has_bias) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
          internal::weight_data_size_impl(
              n, k, group_size, weight_nbit, has_weight_zeros, has_bias);
}

template <int weight_nbit>
void torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
        prepare_weight_data(
            void* weight_data,
            // Inputs
            int n,
            int k,
            int group_size,
            const int8_t* weight_qvals,
            const float* weight_scales,
            const int8_t* weight_zeros,
            const float* bias) {
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
          internal::prepare_weight_data_impl<weight_nbit>(
              weight_data,
              n,
              k,
              group_size,
              weight_qvals,
              weight_scales,
              weight_zeros,
              bias);
}

template <int weight_nbit>
void torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
        kernel(
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
            // Ignored if has_clamp = false
            float clamp_min,
            float clamp_max,
            bool has_weight_zeros,
            bool has_bias,
            bool has_clamp) {
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
          internal::kernel_impl<weight_nbit>(
              output,
              output_m_stride,
              m,
              n,
              k,
              group_size,
              weight_data,
              activation_data,
              clamp_min,
              clamp_max,
              has_weight_zeros,
              has_bias,
              has_clamp);
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
