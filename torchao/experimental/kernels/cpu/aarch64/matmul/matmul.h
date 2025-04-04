// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// TODO: this file will be deleted and replaced by
// torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/include.h
// It exists now to prevent breaking existing code in the interim.

#pragma once

#if defined(__aarch64__) && defined(__ARM_NEON)

#include <arm_neon.h>

namespace torchao::kernels::cpu::aarch64::quantized_matmul {
namespace channelwise_8bit_a_channelwise_8bit_b_1x8x16_f32_neondot {

template <
    bool a_has_zeros,
    bool b_has_zeros,
    bool a_transposed,
    bool b_tranposed>
void kernel(
    int m,
    int n,
    int k,
    const void* lhs,
    int lhs_stride_m,
    const void* rhs,
    int rhs_stride_n,
    float32_t* output,
    int out_stride_m,
    const int8_t* lhs_zero_points,
    const int8_t* rhs_zero_points,
    const float* lhs_scales,
    const float* rhs_scales,
    const int lhs_qparams_stride,
    const int rhs_qparams_stride);

} // namespace channelwise_8bit_a_channelwise_8bit_b_1x8x16_f32_neondot

namespace channelwise_8bit_a_channelwise_8bit_b_1x16x16_f32_smlal {

template <
    bool a_has_zeros,
    bool b_has_zeros,
    bool a_transposed,
    bool b_tranposed>
void kernel(
    int m,
    int n,
    int k,
    const void* lhs,
    int lhs_stride_m,
    const void* rhs,
    int rhs_stride_n,
    float32_t* output,
    int out_stride_m,
    const int8_t* lhs_zero_points,
    const int8_t* rhs_zero_points,
    const float* lhs_scales,
    const float* rhs_scales,
    const int lhs_qparams_stride,
    const int rhs_qparams_stride);

} // namespace channelwise_8bit_a_channelwise_8bit_b_1x16x16_f32_smlal

namespace fp32_a_input_channelwise_8bit_b_1x16x4_f32 {

template <bool b_has_zeros, bool a_transposed, bool b_tranposed>
void kernel(
    int m,
    int n,
    int k,
    const float* lhs,
    int lhs_stride_m,
    const int8_t* rhs,
    int rhs_stride_n,
    float32_t* output,
    int out_stride_m,
    const int8_t* rhs_zero_points,
    const float* rhs_scales,
    const float beta,
    const int rhs_qparams_stride);

} // namespace fp32_a_input_channelwise_8bit_b_1x16x4_f32
} // namespace torchao::kernels::cpu::aarch64::quantized_matmul

#include <torchao/experimental/kernels/cpu/aarch64/matmul/channelwise_8bit_a_channelwise_8bit_b_1x16x16_f32_smlal-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/matmul/channelwise_8bit_a_channelwise_8bit_b_1x8x16_f32_neondot-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/matmul/fp32_a_input_channelwise_8bit_b_1x16x4_f32_impl.h>

#endif // defined(__aarch64__) && defined(__ARM_NEON)
