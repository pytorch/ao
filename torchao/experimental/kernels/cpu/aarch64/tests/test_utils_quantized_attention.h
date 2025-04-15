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

struct fp32_a_channelwise_8bit_b_attn_scores_at_v_test_case {
  int b;
  int s_attn;
  int s_v;
  int h;
  int d;
  size_t b_attn_stride;
  size_t h_attn_stride;
  size_t s_attn_stride;
  size_t b_v_stride;
  size_t h_v_stride;
  size_t s_v_stride;
  size_t b_v_qparams_stride;
  size_t h_v_qparams_stride;
  size_t s_v_qparams_stride;

  std::vector<float> expected_output;

  std::vector<float> attn_scores;

  std::vector<float> v;
  std::vector<int8_t> v_qvals;
  std::vector<float> v_scales;
  std::vector<int8_t> v_zeros;

  fp32_a_channelwise_8bit_b_attn_scores_at_v_test_case(
      int b_,
      int s_attn_,
      int s_v_,
      int h_,
      int d_,
      size_t b_attn_stride_,
      size_t h_attn_stride_,
      size_t s_attn_stride_,
      size_t b_v_stride_,
      size_t h_v_stride_,
      size_t s_v_stride_,
      size_t b_v_qparams_stride_,
      size_t h_v_qparams_stride_,
      size_t s_v_qparams_stride_,
      std::vector<float> expected_output_,
      std::vector<float> attn_scores_,
      std::vector<float> v_,
      std::vector<int8_t> v_qvals_,
      std::vector<float> v_scales_,
      std::vector<int8_t> v_zeros_)
      : b(b_),
        s_attn(s_attn_),
        s_v(s_v_),
        h(h_),
        d(d_),
        b_attn_stride(b_attn_stride_),
        h_attn_stride(h_attn_stride_),
        s_attn_stride(s_attn_stride_),
        b_v_stride(b_v_stride_),
        h_v_stride(h_v_stride_),
        s_v_stride(s_v_stride_),
        b_v_qparams_stride(b_v_qparams_stride_),
        h_v_qparams_stride(h_v_qparams_stride_),
        s_v_qparams_stride(s_v_qparams_stride_),
        expected_output(expected_output_),
        attn_scores(attn_scores_),
        v(v_),
        v_qvals(v_qvals_),
        v_scales(v_scales_),
        v_zeros(v_zeros_) {
    assert(expected_output.size() == b * s_attn * h * d);
    assert(attn_scores.size() == b * h * s_attn * s_v);
    assert(v.size() == b * h * s_v * d);
    assert(v_qvals.size() == b * h * s_v * d);
    assert(v_scales.size() == b * h * s_v);
    assert(v_zeros.size() == b * h * s_v);
  }

  static fp32_a_channelwise_8bit_b_attn_scores_at_v_test_case
  generate(int b, int s_attn, int s_v, int h, int d, bool transposed_v = true) {
    // Generate activations
    auto lhs = get_random_vector(b * h * s_attn * s_v, -1.0, 1.0);

    auto [rhs, rhs_qvals, rhs_scales, rhs_zeros] =
        torchao::test_utils::generate_per_token_quantized_tensor(
            b * h * s_v, d);
    // Above function produces nxk matrix and to produce kxn you need transposed
    // = true. we do !rhs_is_transposed becaues when rhs_is_transposed = true
    // the shape should be nxk instead of kxn.

    size_t b_attn_stride = h * s_attn * s_v;
    size_t h_attn_stride = s_attn * s_v;
    size_t s_attn_stride = s_v;

    size_t b_v_stride = h * s_v * d;
    size_t h_v_stride = s_v * d;
    size_t s_v_stride = d;

    size_t b_v_qparams_stride = h * s_v;
    size_t h_v_qparams_stride = s_v;
    size_t s_v_qparams_stride = 1;

    if (!transposed_v) {
      h_v_stride = d;
      s_v_stride = h * d;

      s_v_qparams_stride = h;
      h_v_qparams_stride = 1;
    }

    // Compute expected output
    // Note that while the inputs can be in shape b x h x s_attn x s_v,
    // and b x h x s_v x d the output is not in b x h x s_attn x s_v
    // but rather b x s_attn x h x d. This is because the output of
    // SDPA will normally be in b x h x s_attn x d, but we want to
    // avoid any tranposes. Thus just aim to output in b x s_attn x h x d
    // This is just for testing purposes. Kernel can actually write output
    // in [B, H, S, D] if needed.
    std::vector<float> expected_output(b * s_attn * h * d);
    size_t b_out_stride = s_attn * h * d;
    size_t s_attn_out_stride = h * d;
    size_t h_out_stride = d;

    for (int b_idx = 0; b_idx < b; b_idx++) {
      for (int s_attn_idx = 0; s_attn_idx < s_attn; s_attn_idx++) {
        for (int h_idx = 0; h_idx < h; h_idx++) {
          for (int d_idx = 0; d_idx < d; d_idx++) {
            float res = 0.0;
            for (int s_v_idx = 0; s_v_idx < s_v; s_v_idx++) {
              int lhs_idx = b_idx * b_attn_stride + s_attn_idx * s_attn_stride +
                  h_idx * h_attn_stride + s_v_idx;
              int rhs_idx = b_idx * b_v_stride + h_idx * h_v_stride + d_idx +
                  s_v_idx * s_v_stride;
              int rhs_scales_zp_idx = b_idx * b_v_qparams_stride +
                  h_idx * h_v_qparams_stride + s_v_idx * s_v_qparams_stride;
              float rhs_dequant = rhs_scales[rhs_scales_zp_idx] *
                  (rhs_qvals[rhs_idx] - rhs_zeros[rhs_scales_zp_idx]);

              res += lhs[lhs_idx] * rhs_dequant;
            }
            expected_output
                [b_idx * b_out_stride + s_attn_idx * s_attn_out_stride +
                 h_idx * h_out_stride + d_idx] = res;
          }
        }
      }
    }

    // Return test case
    return fp32_a_channelwise_8bit_b_attn_scores_at_v_test_case(
        b,
        s_attn,
        s_v,
        h,
        d,
        b_attn_stride,
        h_attn_stride,
        s_attn_stride,
        b_v_stride,
        h_v_stride,
        s_v_stride,
        b_v_qparams_stride,
        h_v_qparams_stride,
        s_v_qparams_stride,
        expected_output,
        lhs,
        rhs,
        rhs_qvals,
        rhs_scales,
        rhs_zeros);
  }
};
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
