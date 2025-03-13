// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_prepare_activation_data_1xk_f32-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/quantization/quantize.h>
#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>
#include <cassert>

namespace torchao::kernels::cpu::aarch64::linear {
namespace channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::
    internal {

inline float clamp(float x, float min, float max) {
  if (x < min)
    return min;
  if (x > max)
    return max;
  return x;
}

// Implements variants of
// channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot
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
// The suffix 1x1x32_f32_neondot indicates the tile size (1x1), the number of
// values unpacked in each inner loop (32), floating point type for output
// (f32), and main ISA instruction (neon_dot).
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
// this kernel:
//
// activation_data
//   Per m_idx (row), activations are stored as follows:
//     scale (float), zero (int8_t),
//     group0_qvals (int8_t[group_size]), [group0_qvals_sum (int32_t)]?
//     group1_qvals (int8_t[group_size]), [group1_qvals_sum (int32_t)]?
//     ...
//
//   The groupi_qvals_sum is only present if has_weight_zeros = true.
//
// weight_data
//  Per n_idx (column), weights are stored as follows:
//    group0_qvals (int8_t[group_size]), group0_scale (float), group0_qvals_sum
//    (int32_t), [group0_zero (int8_t)]?
//    ...
//  The groupi_zero is only present if has_weight_zeros = true.
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
  assert(group_size % 32 == 0);
  constexpr int bytes_per_32_weight_values = 4 * weight_nbit;

  auto activation_data_byte_ptr = (char*)(activation_data);
  char* activation_ptr;

  for (int m_idx = 0; m_idx < m; m_idx++) {
    // Read activation scale and zero
    float activation_scale = *((float*)activation_data_byte_ptr);
    activation_data_byte_ptr += sizeof(float);

    int8_t activation_zero = *((int8_t*)activation_data_byte_ptr);
    activation_data_byte_ptr += sizeof(int8_t);

    // Set weight_data_byte_ptr to start of weight_data
    auto weight_data_byte_ptr = (char*)(weight_data);
    for (int n_idx = 0; n_idx < n; n_idx++) {
      // Set activation_ptr to start of activation qvals for row m_idx
      activation_ptr = activation_data_byte_ptr;
      float res = 0.0;

      // Loop k_idx by group
      for (int k_idx = 0; k_idx < k; k_idx += group_size) {
        // Process group in chunks of 32, accumulating dot products in acc
        int32x4_t acc = vdupq_n_s32(0);
        int8x16_t wq0, wq1, aq;

        for (int i = 0; i < group_size; i += 32) {
          torchao::bitpacking::vec_unpack_32_lowbit_values<weight_nbit>(
              /*unpacked0=*/wq0,
              /*unpacked1=*/wq1,
              /*packed=*/(uint8_t*)weight_data_byte_ptr);

          weight_data_byte_ptr += bytes_per_32_weight_values;

          // Dot product of first 16 values in chunk
          aq = vld1q_s8((int8_t*)activation_ptr);
          activation_ptr += 16;
          acc = vdotq_s32(acc, wq0, aq);

          // Dot product of second 16 values in chunk
          aq = vld1q_s8((int8_t*)activation_ptr);
          activation_ptr += 16;
          acc = vdotq_s32(acc, wq1, aq);
        }
        int32_t qval_dot = vaddvq_s32(acc);

        // Dequantize and accumulate in result
        float weight_scale = *((float*)weight_data_byte_ptr);
        weight_data_byte_ptr += sizeof(float);

        int32_t weight_qvals_sum = *((int32_t*)weight_data_byte_ptr);
        weight_data_byte_ptr += sizeof(int32_t);

        if (has_weight_zeros) {
          int32_t activation_qvals_sum = *((int32_t*)activation_ptr);
          activation_ptr += sizeof(int32_t);

          int8_t weight_zero = *((int8_t*)weight_data_byte_ptr);
          weight_data_byte_ptr += sizeof(int8_t);

          res += (weight_scale * activation_scale) *
              (qval_dot - (activation_zero * weight_qvals_sum) -
               (weight_zero * activation_qvals_sum) +
               (group_size * weight_zero * activation_zero));
        } else {
          res += (weight_scale * activation_scale) *
              (qval_dot - activation_zero * weight_qvals_sum);
        }
      } // k_idx
      if (has_bias) {
        float bias = *((float*)weight_data_byte_ptr);
        weight_data_byte_ptr += sizeof(float);
        res += bias;
      }
      if (has_clamp) {
        res = clamp(res, clamp_min, clamp_max);
      }
      output[m_idx * output_m_stride + n_idx] = res;
    } // n_idx
    activation_data_byte_ptr += (activation_ptr - activation_data_byte_ptr);
  } // m_idx
}

// Prepares weight data for kernel_impl.
// Per n_idx (column), weights are stored as follows:
//    group0_qvals (int8_t[group_size]), group0_scale (float), group0_qvals_sum
//    (int32_t), [group0_zero (int8_t)]?
//    ...
//  The groupi_zero is only present if has_weight_zeros = true.

// Returns number of bytes required for weight_data
size_t inline weight_data_size_impl(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias) {
  assert(k % group_size == 0);
  assert(k % 32 == 0);
  int groups_per_col = k / group_size;
  int col_size = 0;

  // qvals
  // (k * weight_bit) bits -> ((k / 8) * weight_bit) bytes
  col_size += (k / 8) * weight_nbit;

  // scales
  col_size += sizeof(float) * groups_per_col;

  // qvals_sum
  col_size += sizeof(int32_t) * groups_per_col;

  // zeros
  if (has_weight_zeros) {
    col_size += sizeof(int8_t) * groups_per_col;
  }

  if (has_bias) {
    col_size += sizeof(float);
  }

  return col_size * n;
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
    const int8_t* weight_zeros,
    const float* bias) {
  assert(k % group_size == 0);
  assert(group_size % 32 == 0);

  bool has_weight_zeros = (weight_zeros != nullptr);
  bool has_bias = (bias != nullptr);

  auto weight_data_byte_ptr = (char*)weight_data;
  constexpr int bytes_per_32_weight_values = 4 * weight_nbit;

  int8x16_t wq0, wq1;

  const int8_t* qvals_ptr = weight_qvals;
  const float* scales_ptr = weight_scales;
  const int8_t* zeros_ptr = weight_zeros;
  const float* bias_ptr = bias;

  for (int n_idx = 0; n_idx < n; n_idx++) {
    for (int k_idx = 0; k_idx < k; k_idx += group_size) {
      int32_t group_qvals_sum = 0;
      for (int i = 0; i < group_size; i += 32) {
        wq0 = vld1q_s8(qvals_ptr);
        wq1 = vld1q_s8(qvals_ptr + 16);
        qvals_ptr += 32;

        group_qvals_sum += vaddlvq_s8(wq0) + vaddlvq_s8(wq1);

        torchao::bitpacking::vec_pack_32_lowbit_values<weight_nbit>(
            /*packed=*/(uint8_t*)weight_data_byte_ptr,
            /*unpacked0=*/wq0,
            /*unpacked1=*/wq1);
        weight_data_byte_ptr += bytes_per_32_weight_values;
      }
      *((float*)weight_data_byte_ptr) = *scales_ptr++;
      weight_data_byte_ptr += sizeof(float);

      *((int32_t*)weight_data_byte_ptr) = group_qvals_sum;
      weight_data_byte_ptr += sizeof(int32_t);

      if (has_weight_zeros) {
        *((int8_t*)weight_data_byte_ptr) = *zeros_ptr++;
        weight_data_byte_ptr += sizeof(int8_t);
      }
    }
    if (has_bias) {
      *((float*)weight_data_byte_ptr) = *bias_ptr++;
      weight_data_byte_ptr += sizeof(float);
    }
  }
}

} // namespace
  // channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::internal
} // namespace torchao::kernels::cpu::aarch64::linear

// Activation functions
size_t torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::
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
    channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::
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
    channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::
        weight_data_size(
            int n,
            int k,
            int group_size,
            bool has_weight_zeros,
            bool has_bias) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::
          internal::weight_data_size_impl(
              n, k, group_size, weight_nbit, has_weight_zeros, has_bias);
}

template <int weight_nbit>
void torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::
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
      channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::
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

// Kernel function
template <int weight_nbit>
void torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::
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
      channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot::
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
