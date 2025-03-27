// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <stddef.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/pack_activations.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/pack_weights.h>

#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/kernel_1x1x32_f32_neondot-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/kernel_1x4x16_f32_neondot-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/kernel_1x8x16_f32_neondot-impl.h>

namespace torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight {

inline size_t packed_activations_size(
    int m,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  (void)mr; // unused
  (void)kr; // unused
  (void)sr; // unused
  return activation_packing::packed_activations_size(
      m, k, group_size, has_weight_zeros);
}

inline size_t packed_activations_offset(
    int m_idx,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  assert(m_idx % mr == 0);
  auto packed_activations_size_mr_rows =
      packed_activations_size(mr, k, group_size, has_weight_zeros, mr, kr, sr);
  return (m_idx / mr) * packed_activations_size_mr_rows;
}

template <int mr, int kr, int sr>
void pack_activations(
    void* packed_activations,
    int m,
    int k,
    int group_size,
    const float* activations,
    bool has_weight_zeros) {
  activation_packing::pack_activations<mr, kr, sr>(
      packed_activations, m, k, group_size, activations, has_weight_zeros);
}

inline size_t packed_weights_size(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  (void)kr; // unused
  (void)sr; // unused
  return weight_packing::packed_weights_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr);
}

inline size_t packed_weights_offset(
    int n_idx,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  assert(n_idx % nr == 0);
  auto packed_weights_size_nr_cols = packed_weights_size(
      nr, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr);
  return (n_idx / nr) * packed_weights_size_nr_cols;
}

template <int weight_nbit, int nr, int kr, int sr>
void pack_weights(
    void* packed_weights,
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros,
    const float* bias) {
  weight_packing::pack_weights<weight_nbit, nr, kr, sr>(
      packed_weights,
      n,
      k,
      group_size,
      weight_qvals,
      weight_scales,
      weight_zeros,
      bias);
}

template <int weight_nbit>
void kernel_1x1x32_f32_neondot(
    // Outputs
    float32_t* output,
    // Inputs
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* packed_weights,
    const void* packed_activations,
    // Ignored if has_clamp = false
    float clamp_min,
    float clamp_max,
    bool has_weight_zeros,
    bool has_bias,
    bool has_clamp) {
  kernel::kernel_1x1x32_f32_neondot<weight_nbit>(
      output,
      output_m_stride,
      m,
      n,
      k,
      group_size,
      packed_weights,
      packed_activations,
      clamp_min,
      clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);
}

template <int weight_nbit>
void kernel_1x4x16_f32_neondot(
    // Outputs
    float32_t* output,
    // Inputs
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* packed_weights,
    const void* packed_activations,
    // Ignored if has_clamp = false
    float clamp_min,
    float clamp_max,
    bool has_weight_zeros,
    bool has_bias,
    bool has_clamp) {
  kernel::kernel_1x4x16_f32_neondot<weight_nbit>(
      output,
      output_m_stride,
      m,
      n,
      k,
      group_size,
      packed_weights,
      packed_activations,
      clamp_min,
      clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);
}

template <int weight_nbit>
void kernel_1x8x16_f32_neondot(
    // Outputs
    float32_t* output,
    // Inputs
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* packed_weights,
    const void* packed_activations,
    // Ignored if has_clamp = false
    float clamp_min,
    float clamp_max,
    bool has_weight_zeros,
    bool has_bias,
    bool has_clamp) {
  kernel::kernel_1x8x16_f32_neondot<weight_nbit>(
      output,
      output_m_stride,
      m,
      n,
      k,
      group_size,
      packed_weights,
      packed_activations,
      clamp_min,
      clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);
}

} // namespace
  // torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight

#endif // defined(__aarch64__) || defined(__ARM_NEON)
