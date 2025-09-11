// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>

// TODO: Remove all ::kernels. No need for extra namespace.
namespace torchao::kernels::cpu::fallback::quantized_matmul {
namespace fp32_a_input_channelwise_8bit_b_fp32 {
template <bool b_has_zeros, bool a_transposed, bool b_transposed>
void kernel(
    int m,
    int n,
    int k,
    const float* lhs,
    int lhs_stride_m,
    const int8_t* rhs,
    int rhs_stride_n,
    float* output,
    int out_stride_m,
    const int8_t* rhs_zero_points,
    const float* rhs_scales,
    const float beta,
    const int rhs_qparams_stride) {
  assert(a_transposed == false);
  for (int m_idx = 0; m_idx < m; m_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      float res = 0.0;
      for (int k_idx = 0; k_idx < k; k_idx++) {
        int lhs_idx = m_idx * lhs_stride_m + k_idx;
        int rhs_idx = k_idx * rhs_stride_n + n_idx;
        if (b_transposed) {
          rhs_idx = n_idx * rhs_stride_n + k_idx;
        }
        float rhs_dequant = rhs_scales[k_idx * rhs_qparams_stride] *
            (static_cast<int16_t>(rhs[rhs_idx]) -
             static_cast<int16_t>(rhs_zero_points[k_idx * rhs_qparams_stride]));

        res += lhs[lhs_idx] * rhs_dequant;
      }
      output[m_idx * n + n_idx] = output[m_idx * n + n_idx] * beta + res;
    }
  }
}
} // namespace fp32_a_input_channelwise_8bit_b_fp32
} // namespace torchao::kernels::cpu::fallback::quantized_matmul
