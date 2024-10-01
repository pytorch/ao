// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_prepare_activation_data_1xk_f32-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>
#include <torchao/experimental/kernels/cpu/aarch64/valpacking/valpack.h>
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
// * bias are mx1.  Ignored if has_bias = false.
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

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
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
    // Ignored if has_bias is false
    const float* bias,
    // Ignored if has_clamp is false
    float clamp_min,
    float clamp_max) {
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

        if constexpr (has_weight_zeros) {
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
      if constexpr (has_bias) {
        res = vaddq_f32(res, vdupq_n_f32(bias[m_idx]));
      }
      if constexpr (has_clamp) {
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
int inline weight_data_size_impl(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros) {
  assert(k % group_size == 0);
  int groups_per_col = k / group_size;
  int col_size = 0;

  // qvals
  col_size += (k / 8) * weight_nbit;

  // scales
  col_size += sizeof(float) * groups_per_col;

  // qvals_sum
  col_size += sizeof(int32_t) * groups_per_col;

  // zeros
  if (has_weight_zeros) {
    col_size += sizeof(int32_t) * groups_per_col;
  }

  // Replace n with next multiple of 4 >= n
  n = ((n + 3) / 4) * 4;

  return col_size * n;
}

template <int weight_nbit, bool has_weight_zeros>
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
    const int8_t* weight_zeros) {
  assert(k % group_size == 0);
  assert(group_size % 16 == 0);
  int groups_per_k = k / group_size;
  constexpr int bytes_per_64_weight_values = 8 * weight_nbit;

  auto weight_data_byte_ptr = (char*)weight_data;
  const int8_t* qvals_ptr = weight_qvals;
  const float* scales_ptr = weight_scales;
  const int8_t* zeros_ptr = weight_zeros;

  int8_t interleaved_buffer[64];
  int8_t buffer[64];

  for (int n_idx = 0; n_idx < n; n_idx += 4) {
    for (int k_idx = 0; k_idx < k; k_idx += group_size) {
      // Loop over group in chunks of 16, processing 4 columns at at time
      int qvals_sum[4] = {0, 0, 0, 0};
      for (int i = 0; i < group_size; i += 16) {
        std::memset(buffer, 0, 64);
        // Loop over 4 cols
#pragma unroll(4)
        for (int j = 0; j < 4; j++) {
          if (n_idx + j < n) {
            // If qvals_ptr are pre-packed in a naive way, this is where
            // unpacking can occur
            std::memcpy(buffer + 16 * j, qvals_ptr + k * j, 16);
            qvals_sum[j] +=
                torchao::kernels::cpu::aarch64::reduction::compute_sum(
                    buffer + 16 * j, 16);
          }
        }
        torchao::kernels::cpu::valpacking::interleave_data(
            /*data_interleaved=*/interleaved_buffer,
            /*data=*/buffer,
            /*bytes_per_val=*/1,
            /*vals_per_channel=*/16,
            /*vals_per_group=*/16,
            /*vals_per_chunk=*/8,
            /*channels=*/4,
            /*channel_stride_in_vals=*/16);
        torchao::bitpacking::vec_pack_64_lowbit_values<weight_nbit>(
            (uint8_t*)weight_data_byte_ptr,
            vld1q_s8(interleaved_buffer),
            vld1q_s8(interleaved_buffer + 16),
            vld1q_s8(interleaved_buffer + 32),
            vld1q_s8(interleaved_buffer + 48));
        qvals_ptr += 16;
        weight_data_byte_ptr += bytes_per_64_weight_values;
      } // loop over group

      // Store weight scales
#pragma unroll(4)
      for (int j = 0; j < 4; j++) {
        float32_t scale = 0.0;
        if (n_idx + j < n) {
          scale = *(scales_ptr + j * groups_per_k);
        }
        *((float*)weight_data_byte_ptr) = scale;
        weight_data_byte_ptr += sizeof(float);
      }
      scales_ptr += 1;

      // Store weight qvals_sum
#pragma unroll(4)
      for (int j = 0; j < 4; j++) {
        *((int*)weight_data_byte_ptr) = qvals_sum[j];
        weight_data_byte_ptr += sizeof(int);
      }

      // Store weight zeros
      // I went back and forth on how to store weight_zero.
      // Kernel computation is done in int32, so I'm converting these to
      // int32 before storing (load 4 int32s in kernel).
      // In the 1x8 kernel, we may want to store as int16_t, which reduces
      // a load in the kernel (load 8 int16_ts in kernel, instead of 2
      // load 4 int32_ts), but adds 2 moves (int16 to int32).
      if constexpr (has_weight_zeros) {
#pragma unroll(4)
        for (int j = 0; j < 4; j++) {
          int32_t zero = 0;
          if (n_idx + j < n) {
            zero = (int)(*(zeros_ptr + j * groups_per_k));
          }
          *((int32_t*)weight_data_byte_ptr) = zero;
          weight_data_byte_ptr += sizeof(int32_t);
        }
        zeros_ptr += 1;
      }
    } // k_idx

    // In the previous loop over k, we processed 4 columns at a time,
    // but only advanced our pointers over the first column.
    // So we advance over the other 3 columns here.
    qvals_ptr += 3 * k;
    scales_ptr += 3 * groups_per_k;
    if constexpr (has_weight_zeros) {
      zeros_ptr += 3 * groups_per_k;
    }
  } // n_idx
}

} // namespace
  // channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::internal
} // namespace torchao::kernels::cpu::aarch64::linear

// Activation functions
template <bool has_weight_zeros>
int torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
        activation_data_size(int m, int k, int group_size) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_prepare_activation_data_1xk_f32::internal::
          activation_data_size_impl(m, k, group_size, has_weight_zeros);
}

template <bool has_weight_zeros>
void torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
        prepare_activation_data(
            void* activation_data,
            // Inputs
            int m,
            int k,
            // Ignored if has_weight_zeros = false
            int group_size,
            const float* activations) {
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_prepare_activation_data_1xk_f32::internal::
          prepare_activation_data_impl<has_weight_zeros>(
              activation_data, m, k, group_size, activations);
}

// Weight functions
template <int weight_nbit, bool has_weight_zeros>
int torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
        weight_data_size(int n, int k, int group_size) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
          internal::weight_data_size_impl(
              n, k, group_size, weight_nbit, has_weight_zeros);
}

template <int weight_nbit, bool has_weight_zeros>
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
            const int8_t* weight_zeros) {
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
          internal::prepare_weight_data_impl<weight_nbit, has_weight_zeros>(
              weight_data,
              n,
              k,
              group_size,
              weight_qvals,
              weight_scales,
              weight_zeros);
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
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
            // Not applied if nullptr
            const float* bias,
            // Ignored if has_clamp = false
            float clamp_min,
            float clamp_max) {
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot::
          internal::
              kernel_impl<weight_nbit, has_weight_zeros, has_bias, has_clamp>(
                  output,
                  output_m_stride,
                  m,
                  n,
                  k,
                  group_size,
                  weight_data,
                  activation_data,
                  bias,
                  clamp_min,
                  clamp_max);
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
