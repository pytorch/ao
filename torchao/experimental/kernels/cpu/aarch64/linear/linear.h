// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <stddef.h>

namespace torchao::kernels::cpu::aarch64::linear {

namespace channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot {

size_t
activation_data_size(int m, int k, int group_size, bool has_weight_zeros);

void prepare_activation_data(
    void* activation_data,
    // Inputs
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    const float* activations,
    bool has_weight_zeros);

template <int weight_nbit>
size_t weight_data_size(
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias);

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
    const float* bias);

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
    bool has_clamp);

} // namespace
  // channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot

namespace channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot {

size_t
activation_data_size(int m, int k, int group_size, bool has_weight_zeros);

void prepare_activation_data(
    void* activation_data,
    // Inputs
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    const float* activations,
    bool has_weight_zeros);

template <int weight_nbit>
size_t weight_data_size(
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias);

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
    const float* bias);

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
    bool has_clamp);

} // namespace
  // channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot

namespace channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot {

size_t
activation_data_size(int m, int k, int group_size, bool has_weight_zeros);

void prepare_activation_data(
    void* activation_data,
    // Inputs
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    const float* activations,
    bool has_weight_zeros);

template <int weight_nbit>
size_t weight_data_size(
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias);

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
    const float* bias);

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
    bool has_clamp);

} // namespace
  // channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot

} // namespace torchao::kernels::cpu::aarch64::linear

#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot-impl.h>

#endif // defined(__aarch64__) || defined(__ARM_NEON)
