// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// TODO: this file will be deleted and replaced by
// torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/include.h
// It exists now to prevent breaking existing code in the interim.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <stddef.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/channelwise_8bit_activation_groupwise_lowbit_weight.h>

namespace torchao::kernels::cpu::aarch64::linear {
namespace channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot {

inline size_t
activation_data_size(int m, int k, int group_size, bool has_weight_zeros) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::
          packed_activations_size(
              m,
              k,
              group_size,
              has_weight_zeros,
              /*mr*/ 1,
              /*kr*/ 32,
              /*sr*/ 1);
}

inline void prepare_activation_data(
    void* activation_data,
    // Inputs
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    const float* activations,
    bool has_weight_zeros) {
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::
          pack_activations</*mr*/ 1, /*kr*/ 32, /*sr*/ 1>(
              activation_data, m, k, group_size, activations, has_weight_zeros);
}

template <int weight_nbit>
size_t weight_data_size(
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::packed_weights_size(
          n,
          k,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          /*nr*/ 1,
          /*kr*/ 32,
          /*sr*/ 1);
}

template <int weight_nbit>
void prepare_weight_data(
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
      channelwise_8bit_activation_groupwise_lowbit_weight::
          pack_weights<weight_nbit, /*nr*/ 1, /*kr*/ 32, /*sr*/ 1>(
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
void kernel(
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
      channelwise_8bit_activation_groupwise_lowbit_weight::
          kernel_1x1x32_f32_neondot<weight_nbit>(
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

} // namespace
  // channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot

namespace channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot {

inline size_t
activation_data_size(int m, int k, int group_size, bool has_weight_zeros) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::
          packed_activations_size(
              m,
              k,
              group_size,
              has_weight_zeros,
              /*mr*/ 1,
              /*kr*/ 16,
              /*sr*/ 2);
}

inline void prepare_activation_data(
    void* activation_data,
    // Inputs
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    const float* activations,
    bool has_weight_zeros) {
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::
          pack_activations</*mr*/ 1, /*kr*/ 16, /*sr*/ 2>(
              activation_data, m, k, group_size, activations, has_weight_zeros);
}

template <int weight_nbit>
inline size_t weight_data_size(
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::packed_weights_size(
          n,
          k,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          /*nr*/ 4,
          /*kr*/ 16,
          /*sr*/ 2);
}

template <int weight_nbit>
inline void prepare_weight_data(
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
      channelwise_8bit_activation_groupwise_lowbit_weight::
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

template <int weight_nbit>
void kernel(
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
      channelwise_8bit_activation_groupwise_lowbit_weight::
          kernel_1x4x16_f32_neondot<weight_nbit>(
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

} // namespace
  // channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot

namespace channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot {

inline size_t
activation_data_size(int m, int k, int group_size, bool has_weight_zeros) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::
          packed_activations_size(
              m,
              k,
              group_size,
              has_weight_zeros,
              /*mr*/ 1,
              /*kr*/ 16,
              /*sr*/ 2);
}

inline void prepare_activation_data(
    void* activation_data,
    // Inputs
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    const float* activations,
    bool has_weight_zeros) {
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::
          pack_activations</*mr*/ 1, /*kr*/ 16, /*sr*/ 2>(
              activation_data, m, k, group_size, activations, has_weight_zeros);
}

template <int weight_nbit>
size_t weight_data_size(
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias) {
  return torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::packed_weights_size(
          n,
          k,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          /*nr*/ 8,
          /*kr*/ 16,
          /*sr*/ 2);
}

template <int weight_nbit>
void prepare_weight_data(
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
      channelwise_8bit_activation_groupwise_lowbit_weight::
          pack_weights<weight_nbit, /*nr*/ 8, /*kr*/ 16, /*sr*/ 2>(
              weight_data,
              n,
              k,
              group_size,
              weight_qvals,
              weight_scales,
              weight_zeros,
              bias);
}

template <int weight_nbit, bool has_weight_zeros>
void kernel(
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
    bool has_weight_zeros_,
    bool has_bias,
    bool has_clamp) {
  (void)has_weight_zeros_; // unused
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::
          kernel_1x8x16_f32_neondot<weight_nbit, has_weight_zeros, /*has_lut*/ false>(
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

} // namespace
  // channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot

} // namespace torchao::kernels::cpu::aarch64::linear

#endif // defined(__aarch64__) || defined(__ARM_NEON)
