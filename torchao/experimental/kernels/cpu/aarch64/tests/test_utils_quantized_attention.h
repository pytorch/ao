// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/quantization/quantize.h>
#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <cassert>
#include <functional>
#include <random>
#include <vector>

namespace torchao {
struct channelwise_8bit_a_channelwise_8bit_b_q_at_k_attention_test_case {
  int b;
  int s_q;
  int s_k;
  int h;
  int d;
  bool tranposed;

  size_t b_q_stride;
  size_t h_q_stride;
  size_t s_q_stride;

  size_t b_k_stride;
  size_t h_k_stride;
  size_t s_k_stride;

  size_t b_q_qparams_stride;
  size_t h_q_qparams_stride;
  size_t s_q_qparams_stride;

  size_t b_k_qparams_stride;
  size_t h_k_qparams_stride;
  size_t s_k_qparams_stride;

  std::vector<float> expected_output;

  std::vector<float> q;
  std::vector<int8_t> q_qvals;
  std::vector<float> q_scales;
  std::vector<int8_t> q_zeros;

  std::vector<float> k;
  std::vector<int8_t> k_qvals;
  std::vector<float> k_scales;
  std::vector<int8_t> k_zeros;

  channelwise_8bit_a_channelwise_8bit_b_q_at_k_attention_test_case(
      int b_,
      int s_q_,
      int s_k_,
      int h_,
      int d_,
      int transposed_,
      size_t b_q_stride_,
      size_t h_q_stride_,
      size_t s_q_stride_,
      size_t b_k_stride_,
      size_t h_k_stride_,
      size_t s_k_stride_,
      size_t b_q_qparams_stride_,
      size_t h_q_qparams_stride_,
      size_t s_q_qparams_stride_,
      size_t b_k_qparams_stride_,
      size_t h_k_qparams_stride_,
      size_t s_k_qparams_stride_,
      std::vector<float> expected_output_,
      std::vector<float> q_,
      std::vector<int8_t> q_qvals_,
      std::vector<float> q_scales_,
      std::vector<int8_t> q_zeros_,
      std::vector<float> k_,
      std::vector<int8_t> k_qvals_,
      std::vector<float> k_scales_,
      std::vector<int8_t> k_zeros_)
      : b(b_),
        s_q(s_q_),
        s_k(s_k_),
        h(h_),
        d(d_),
        tranposed(transposed_),
        b_q_stride(b_q_stride_),
        h_q_stride(h_q_stride_),
        s_q_stride(s_q_stride_),
        b_k_stride(b_k_stride_),
        h_k_stride(h_k_stride_),
        s_k_stride(s_k_stride_),
        b_q_qparams_stride(b_q_qparams_stride_),
        h_q_qparams_stride(h_q_qparams_stride_),
        s_q_qparams_stride(s_q_qparams_stride_),
        b_k_qparams_stride(b_k_qparams_stride_),
        h_k_qparams_stride(h_k_qparams_stride_),
        s_k_qparams_stride(s_k_qparams_stride_),
        expected_output(expected_output_),
        q(q_),
        q_qvals(q_qvals_),
        q_scales(q_scales_),
        q_zeros(q_zeros_),
        k(k_),
        k_qvals(k_qvals_),
        k_scales(k_scales_),
        k_zeros(k_zeros_) {
    assert(expected_output.size() == b * s_q * h * s_k);
    assert(q.size() == b * s_q * h * d);
    assert(q_qvals.size() == b * s_q * h * d);
    assert(q_scales.size() == b * s_q * h);
    assert(q_zeros.size() == b * s_q * h);
    assert(k.size() == b * s_k * h * d);
    assert(k_qvals.size() == b * s_k * h * d);
    assert(k_scales.size() == b * s_k * h);
    assert(k_zeros.size() == b * s_k * h);
  }

  static channelwise_8bit_a_channelwise_8bit_b_q_at_k_attention_test_case
  generate(int b, int s_q, int s_k, int h, int d, bool transposed = true) {
    // Generate activations
    auto [lhs, lhs_qvals, lhs_scales, lhs_zeros] =
        torchao::test_utils::generate_per_token_quantized_tensor(
            b * s_q * h, d);

    auto [rhs, rhs_qvals, rhs_scales, rhs_zeros] =
        torchao::test_utils::generate_per_token_quantized_tensor(
            b * s_k * h, d);
    // Above function produces nxk matrix and to produce kxn you need transposed
    // = true. we do !rhs_is_transposed becaues when rhs_is_transposed = true
    // the shape should be nxk instead of kxn.

    size_t b_q_stride = h * s_q * d;
    size_t h_q_stride = s_q * d;
    size_t s_q_stride = d;

    size_t b_k_stride = h * s_k * d;
    size_t h_k_stride = s_k * d;
    size_t s_k_stride = d;

    size_t b_q_qparams_stride = h * s_q;
    size_t h_q_qparams_stride = s_q;
    size_t s_q_qparams_stride = 1;

    size_t b_k_qparams_stride = h * s_k;
    size_t h_k_qparams_stride = s_k;
    size_t s_k_qparams_stride = 1;

    if (!transposed) {
      h_q_stride = d;
      s_q_stride = h * d;
      h_k_stride = d;
      s_k_stride = h * d;

      s_q_qparams_stride = h;
      h_q_qparams_stride = 1;

      s_k_qparams_stride = h;
      h_k_qparams_stride = 1;
    }

    // Compute expected output
    std::vector<float> expected_output(b * h * s_q * s_k);
    size_t b_out_stride = h * s_q * s_k;
    size_t h_out_stride = s_q * s_k;
    size_t s_q_out_stride = s_k;

    for (int b_idx = 0; b_idx < b; b_idx++) {
      for (int s_q_idx = 0; s_q_idx < s_q; s_q_idx++) {
        for (int h_idx = 0; h_idx < h; h_idx++) {
          for (int s_k_idx = 0; s_k_idx < s_k; s_k_idx++) {
            float res = 0.0;
            for (int d_idx = 0; d_idx < d; d_idx++) {
              int lhs_idx = b_idx * b_q_stride + s_q_idx * s_q_stride +
                  h_idx * h_q_stride + d_idx;
              int rhs_idx = b_idx * b_k_stride + s_k_idx * s_k_stride +
                  h_idx * h_k_stride + d_idx;
              int lhs_scales_zp_idx = b_idx * b_q_qparams_stride +
                  h_idx * h_q_qparams_stride + s_q_idx * s_q_qparams_stride;
              int rhs_scales_zp_idx = b_idx * b_k_qparams_stride * h +
                  h_idx * h_k_qparams_stride + s_k_idx * s_k_qparams_stride;
              float lhs_dequant = lhs_scales[lhs_scales_zp_idx] *
                  (lhs_qvals[lhs_idx] - lhs_zeros[lhs_scales_zp_idx]);

              float rhs_dequant = rhs_scales[rhs_scales_zp_idx] *
                  (rhs_qvals[rhs_idx] - rhs_zeros[rhs_scales_zp_idx]);

              res += lhs_dequant * rhs_dequant;
            }
            expected_output
                [b_idx * b_out_stride + s_q_idx * s_q_out_stride +
                 h_idx * h_out_stride + s_k_idx] = res;
          }
        }
      }
    }

    // Return test case
    return channelwise_8bit_a_channelwise_8bit_b_q_at_k_attention_test_case(
        b,
        s_q,
        s_k,
        h,
        d,
        transposed,
        b_q_stride,
        h_q_stride,
        s_q_stride,
        b_k_stride,
        h_k_stride,
        s_k_stride,
        b_q_qparams_stride,
        h_q_qparams_stride,
        s_q_qparams_stride,
        b_k_qparams_stride,
        h_k_qparams_stride,
        s_k_qparams_stride,
        expected_output,
        lhs,
        lhs_qvals,
        lhs_scales,
        lhs_zeros,
        rhs,
        rhs_qvals,
        rhs_scales,
        rhs_zeros);
  }
};

} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
