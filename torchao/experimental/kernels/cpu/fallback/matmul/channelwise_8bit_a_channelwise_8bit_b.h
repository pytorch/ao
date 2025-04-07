// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>

namespace torchao::kernels::cpu::fallback::quantized_matmul {
namespace channelwise_8bit_a_channelwise_8bit_b::internal {

template <
    bool a_has_zeros,
    bool b_has_zeros,
    bool a_transposed,
    bool b_tranposed>
struct KernelImpl {
  static void run(
      int m,
      int n,
      int k,
      const void* lhs,
      int lhs_stride_m,
      const void* rhs,
      int rhs_stride_n,
      float* output,
      int out_stride_m,
      const int8_t* lhs_zero_points,
      const int8_t* rhs_zero_points,
      const float* lhs_scales,
      const float* rhs_scales,
      const int lhs_qparams_stride,
      const int rhs_qparams_stride);
};

template <bool b_transposed>
struct KernelImpl<true, true, false, b_transposed> {
  static void run(
      int m,
      int n,
      int k,
      const void* lhs,
      int lhs_stride_m,
      const void* rhs,
      int rhs_stride_n,
      float* output,
      int out_stride_m,
      const int8_t* lhs_zero_points,
      const int8_t* rhs_zero_points,
      const float* lhs_scales,
      const float* rhs_scales,
      const int lhs_qparams_stride,
      const int rhs_qparams_stride) {
    const int8_t* lhs_qvals = static_cast<const int8_t*>(lhs);
    const int8_t* rhs_qvals = static_cast<const int8_t*>(rhs);
    for (int m_idx = 0; m_idx < m; m_idx++) {
      for (int n_idx = 0; n_idx < n; n_idx++) {
        float res = 0.0;
        for (int k_idx = 0; k_idx < k; k_idx++) {
          int lhs_idx = m_idx * lhs_stride_m + k_idx;
          int rhs_idx = k_idx * rhs_stride_n + n_idx;
          if (b_transposed) {
            rhs_idx = n_idx * rhs_stride_n + k_idx;
          }

          float lhs_dequant = lhs_scales[m_idx * lhs_qparams_stride] *
              (static_cast<int16_t>(lhs_qvals[lhs_idx]) -
               static_cast<int16_t>(
                   lhs_zero_points[m_idx * lhs_qparams_stride]));

          float rhs_dequant = rhs_scales[n_idx * rhs_qparams_stride] *
              (static_cast<int16_t>(rhs_qvals[rhs_idx]) -
               static_cast<int16_t>(
                   rhs_zero_points[n_idx * rhs_qparams_stride]));

          res += lhs_dequant * rhs_dequant;
        }
        output[m_idx * n + n_idx] = res;
      }
    }
  }
};

} // namespace
  // channelwise_8bit_a_channelwise_8bit_b::internal
} // namespace torchao::kernels::cpu::fallback::quantized_matmul

// TODO: Remove all ::kernels. No need for extra namespace.
namespace torchao::kernels::cpu::fallback::quantized_matmul {
namespace channelwise_8bit_a_channelwise_8bit_b {
template <
    bool a_has_zeros,
    bool b_has_zeros,
    bool a_transposed,
    bool b_transposed>
void kernel(
    int m,
    int n,
    int k,
    const void* lhs,
    int lhs_stride_m,
    const void* rhs,
    int rhs_stride_n,
    float* output,
    int out_stride_m,
    const int8_t* lhs_zero_points,
    const int8_t* rhs_zero_points,
    const float* lhs_scales,
    const float* rhs_scales,
    const int lhs_qparams_stride,
    const int rhs_qparams_stride) {
  channelwise_8bit_a_channelwise_8bit_b::internal::
      KernelImpl<a_has_zeros, b_has_zeros, a_transposed, b_transposed>::run(
          m,
          n,
          k,
          lhs,
          lhs_stride_m,
          rhs,
          rhs_stride_n,
          output,
          out_stride_m,
          lhs_zero_points,
          rhs_zero_points,
          lhs_scales,
          rhs_scales,
          lhs_qparams_stride,
          rhs_qparams_stride);
}
} // namespace channelwise_8bit_a_channelwise_8bit_b
} // namespace torchao::kernels::cpu::fallback::quantized_matmul
