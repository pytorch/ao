// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/quantization/quantize.h>
#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>
#include <cassert>
#include <functional>
#include <random>
#include <vector>

namespace torchao {
inline std::vector<float>
get_random_vector(int size, float min = -1.0, float max = 1.0) {
  assert(min < max);
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto dist = std::bind(std::uniform_real_distribution<float>(min, max), rng);
  std::vector<float> res(size);
  std::generate(res.begin(), res.end(), std::ref(dist));
  return res;
}

inline std::vector<uint8_t> get_random_lowbit_vector(int size, int nbit) {
  assert(nbit >= 1);
  assert(nbit <= 8);

  int min = 0;
  int max = (1 << nbit) - 1;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto dist = std::bind(std::uniform_int_distribution<>(min, max), rng);

  std::vector<uint8_t> res(size);
  std::generate(res.begin(), res.end(), std::ref(dist));
  return res;
}

inline std::vector<int8_t> get_random_signed_lowbit_vector(int size, int nbit) {
  assert(nbit >= 1);
  assert(nbit <= 8);

  int min = 0;
  int max = (1 << nbit) - 1;
  int offset = (1 << (nbit - 1));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto dist = std::bind(std::uniform_int_distribution<>(min, max), rng);

  std::vector<int8_t> res(size);
  std::vector<int16_t> tmp(size);
  std::generate(tmp.begin(), tmp.end(), std::ref(dist));
  for (int i = 0; i < size; i++) {
    res[i] = tmp[i] - offset;
  }
  return res;
}

// TODO move these to a common utils
inline uint16_t get_bf16_from_float(float f) {
  uint16_t bf16;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  memcpy(&bf16, &f, sizeof(uint16_t));
#else
  const void* fp = reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(&f) + sizeof(float) - sizeof(uint16_t));
  memcpy(&bf16, fp, sizeof(uint16_t));
#endif // __BYTE_ORDER__
  return bf16;
}

inline float get_float_from_bf16(uint16_t bf16) {
  float f;
  const uint32_t i32 = (bf16 << 16);
  memcpy(&f, &i32, sizeof(uint32_t));
  return f;
}

namespace {
auto generate_per_token_quantized_tensor(
    int m,
    int n,
    bool transposed = false) {
  auto activations = get_random_vector(m * n, -1.0, 1.0);
  auto activation_qvals = std::vector<int8_t>(m * n, 0);
  auto activation_scales = std::vector<float>(m, 0);
  auto activation_zeros = std::vector<int8_t>(m, 0);

  // Quantize activations with 8-bit asymmetric
  // TODO: replace with generic function that does not use aarch64
  // quantize method after we combine with torchao
  int qmin, qmax, zero;
  float vmin, vmax, scale;
  torchao::quantization::get_qvals_range(
      qmin, qmax, /*nbit=*/8, /*is_symmetric=*/false);
  for (int m_idx = 0; m_idx < m; m_idx++) {
    torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
        vmin, vmax, /*vals=*/activations.data() + m_idx * n, /*size=*/n);
    torchao::quantization::get_scale_and_zero(
        scale, zero, vmin, vmax, qmin, qmax);
    activation_scales[m_idx] = scale;
    activation_zeros[m_idx] = zero;
    torchao::kernels::cpu::aarch64::quantization::quantize(
        /*qvals=*/activation_qvals.data() + m_idx * n,
        /*vals=*/activations.data() + m_idx * n,
        /*size=*/n,
        scale,
        zero,
        qmin,
        qmax);
  }
  if (transposed) {
    auto activations_t = std::vector<float32_t>(m * n, 0);
    auto activation_qvals_t = std::vector<int8_t>(m * n, 0);
    for (int m_idx = 0; m_idx < m; m_idx++) {
      for (int n_idx = 0; n_idx < n; n_idx++) {
        int activation_idx = m_idx * n + n_idx;
        int tranposed_idx = n_idx * m + m_idx;
        activations_t[tranposed_idx] = activations[activation_idx];
        activation_qvals_t[tranposed_idx] = activation_qvals[activation_idx];
      }
    }
    activations = activations_t;
    activation_qvals = activation_qvals_t;
  }

  return std::make_tuple(
      activations, activation_qvals, activation_scales, activation_zeros);
}
} // namespace

struct channelwise_8bit_activation_groupwise_lowbit_weight_test_case {
  int m;
  int k;
  int n;
  int weight_group_size;
  int weight_nbit;
  bool has_weight_zeros;
  bool has_bias;
  bool has_clamp;
  float clamp_min;
  float clamp_max;

  std::vector<float> expected_output;

  std::vector<float> activations;
  std::vector<int8_t> activation_qvals;
  std::vector<float> activation_scales;
  std::vector<int8_t> activation_zeros;

  std::vector<float> weights;
  std::vector<int8_t> weight_qvals;
  std::vector<float> weight_scales;
  std::vector<int8_t> weight_zeros;

  std::vector<float> bias;

  channelwise_8bit_activation_groupwise_lowbit_weight_test_case(
      int m_,
      int k_,
      int n_,
      int weight_group_size_,
      int weight_nbit_,
      bool has_weight_zeros_,
      bool has_bias_,
      bool has_clamp_,
      float clamp_min_,
      float clamp_max_,
      std::vector<float> expected_output_,
      std::vector<float> activations_,
      std::vector<int8_t> activation_qvals_,
      std::vector<float> activation_scales_,
      std::vector<int8_t> activation_zeros_,
      std::vector<float> weights_,
      std::vector<int8_t> weight_qvals_,
      std::vector<float> weight_scales_,
      std::vector<int8_t> weight_zeros_,
      std::vector<float> bias_)
      : m(m_),
        k(k_),
        n(n_),
        weight_group_size(weight_group_size_),
        weight_nbit(weight_nbit_),
        has_weight_zeros(has_weight_zeros_),
        has_bias(has_bias_),
        has_clamp(has_clamp_),
        clamp_min(clamp_min_),
        clamp_max(clamp_max_),
        expected_output(expected_output_),
        activations(activations_),
        activation_qvals(activation_qvals_),
        activation_scales(activation_scales_),
        activation_zeros(activation_zeros_),
        weights(weights_),
        weight_qvals(weight_qvals_),
        weight_scales(weight_scales_),
        weight_zeros(weight_zeros_),
        bias(bias_) {
    assert(k % weight_group_size == 0);
    assert(expected_output.size() == m * n);
    assert(activations.size() == m * k);
    assert(activation_qvals.size() == m * k);
    assert(activation_scales.size() == m);
    assert(activation_zeros.size() == m);
    assert(weights.size() == n * k);
    assert(weight_qvals.size() == n * k);
    assert((weight_group_size * weight_scales.size()) == (n * k));
    assert((weight_group_size * weight_zeros.size()) == (n * k));
    assert(bias.size() == n);

    if (has_clamp) {
      assert(clamp_min < clamp_max);
    }
  }

  static channelwise_8bit_activation_groupwise_lowbit_weight_test_case generate(
      int m,
      int k,
      int n,
      int weight_group_size,
      int weight_nbit,
      bool has_weight_zeros,
      bool has_bias,
      bool has_clamp,
      bool round_weight_scales_to_bf16 = false) {
    // activations is m x k (stored in row-major)
    // weights is k x n (stored in column-major)

    // Generate activations
    auto [activations, activation_qvals, activation_scales, activation_zeros] =
        generate_per_token_quantized_tensor(m, k);

    //  Generate weights
    assert(k % weight_group_size == 0);
    int n_weight_groups = (n * k) / weight_group_size;
    auto weights = get_random_vector(n * k, -1.0, 1.0);
    auto weight_qvals = std::vector<int8_t>(n * k, 0);
    auto weight_scales = std::vector<float>(n_weight_groups, 0.0);
    auto weight_zeros = std::vector<int8_t>(n_weight_groups, 0);

    int qmin, qmax, zero;
    float vmin, vmax, scale;
    // Quantize weights with weight_nbit
    // TODO: replace with generic function that does not use aarch64
    // quantize method after we combine with torchao
    torchao::quantization::get_qvals_range(
        qmin, qmax, /*nbit=*/weight_nbit, /*is_symmetric=*/false);

    int n_groups = (n * k) / weight_group_size;
    for (int group_idx = 0; group_idx < n_groups; group_idx += 1) {
      torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
          vmin,
          vmax,
          /*vals=*/weights.data() + group_idx * weight_group_size,
          /*size=*/weight_group_size);

      if (has_weight_zeros) {
        torchao::quantization::get_scale_and_zero(
            scale, zero, vmin, vmax, qmin, qmax);
      } else {
        scale = torchao::quantization::get_scale(vmin, vmax, qmin, qmax);
        zero = 0;
      }
      if (round_weight_scales_to_bf16) {
        // weight scales are bf16 in the kernel
        // so we need to round trip them to bf16 and back to float to match it.
        scale = get_float_from_bf16(get_bf16_from_float(scale));
      }
      weight_scales[group_idx] = scale;
      weight_zeros[group_idx] = zero;

      torchao::kernels::cpu::aarch64::quantization::quantize(
          /*qvals=*/weight_qvals.data() + group_idx * weight_group_size,
          /*vals=*/weights.data() + group_idx * weight_group_size,
          /*size=*/weight_group_size,
          scale,
          zero,
          qmin,
          qmax);
    }

    std::vector<float> bias(n, 0.0);
    if (has_bias) {
      bias = get_random_vector(n, -1.0, 1.0);
    }

    float clamp_min = 0.0;
    float clamp_max = 0.0;
    if (has_clamp) {
      clamp_min = get_random_vector(1, -1.0, 0.2)[0];
      clamp_max = get_random_vector(1, 0.3, 1.0)[0];
    }

    // Compute expected output
    std::vector<float> expected_output(m * n);

    for (int m_idx = 0; m_idx < m; m_idx++) {
      for (int n_idx = 0; n_idx < n; n_idx++) {
        float res = 0.0;
        for (int k_idx = 0; k_idx < k; k_idx++) {
          int activation_idx = m_idx * k + k_idx;
          int weight_idx = n_idx * k + k_idx;
          int weight_group_idx = weight_idx / weight_group_size;

          float activation_dequant = activation_scales[m_idx] *
              (activation_qvals[activation_idx] - activation_zeros[m_idx]);

          float weight_dequant = weight_scales[weight_group_idx] *
              (weight_qvals[weight_idx] - weight_zeros[weight_group_idx]);

          res += activation_dequant * weight_dequant;
        }
        res += bias[n_idx];
        if (has_clamp) {
          res = std::min(std::max(res, clamp_min), clamp_max);
        }
        expected_output[m_idx * n + n_idx] = res;
      }
    }

    // Return test case
    return channelwise_8bit_activation_groupwise_lowbit_weight_test_case(
        m,
        k,
        n,
        weight_group_size,
        weight_nbit,
        has_weight_zeros,
        has_bias,
        has_clamp,
        clamp_min,
        clamp_max,
        expected_output,
        activations,
        activation_qvals,
        activation_scales,
        activation_zeros,
        weights,
        weight_qvals,
        weight_scales,
        weight_zeros,
        bias);
  }
};

struct channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case {
  int m;
  int k;
  int n;
  int stride;

  bool lhs_has_zeros;
  bool rhs_has_zeros;
  bool lhs_is_transposed;
  bool rhs_is_transposed;

  std::vector<float> expected_output;

  std::vector<float> lhs;
  std::vector<int8_t> lhs_qvals;
  std::vector<float> lhs_scales;
  std::vector<int8_t> lhs_zeros;

  std::vector<float> rhs;
  std::vector<int8_t> rhs_qvals;
  std::vector<float> rhs_scales;
  std::vector<int8_t> rhs_zeros;

  channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case(
      int m_,
      int k_,
      int n_,
      int stride_,
      bool lhs_has_zeros_,
      bool rhs_has_zeros_,
      bool lhs_is_transposed_,
      bool rhs_is_transposed_,
      std::vector<float> expected_output_,
      std::vector<float> lhs_,
      std::vector<int8_t> lhs_qvals_,
      std::vector<float> lhs_scales_,
      std::vector<int8_t> lhs_zeros_,
      std::vector<float> rhs_,
      std::vector<int8_t> rhs_qvals_,
      std::vector<float> rhs_scales_,
      std::vector<int8_t> rhs_zeros_)
      : m(m_),
        k(k_),
        n(n_),
        stride(stride_),
        lhs_has_zeros(lhs_has_zeros_),
        rhs_has_zeros(rhs_has_zeros_),
        lhs_is_transposed(lhs_is_transposed_),
        rhs_is_transposed(rhs_is_transposed_),
        expected_output(expected_output_),
        lhs(lhs_),
        lhs_qvals(lhs_qvals_),
        lhs_scales(lhs_scales_),
        lhs_zeros(lhs_zeros_),
        rhs(rhs_),
        rhs_qvals(rhs_qvals_),
        rhs_scales(rhs_scales_),
        rhs_zeros(rhs_zeros_) {
    assert(expected_output.size() == m * n);
    assert(lhs.size() == m * stride * k);
    assert(lhs_qvals.size() == m * stride * k);
    assert(lhs_scales.size() == m * stride);
    assert(lhs_zeros.size() == m * stride);
    assert(rhs.size() == n * stride * k);
    assert(rhs_qvals.size() == n * stride * k);
    assert(rhs_scales.size() == n * stride);
    assert(rhs_zeros.size() == n * stride);
  }

  static channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case generate(
      int m,
      int k,
      int n,
      bool lhs_has_zeros,
      bool rhs_has_zeros,
      bool lhs_is_transposed,
      // rhs_is_transposed means generated b matrix is mxk instead of kxm
      bool rhs_is_transposed,
      int stride = 1) {
    assert(!lhs_is_transposed);
    assert(lhs_has_zeros);
    assert(rhs_has_zeros);
    // !Rhs transposed was considered if we were doing quantized(softmax(q@k)) @
    // quantized(v) Since v would have been [B, H, S, D]. And [S, D] would be
    // rhs matrix which is not transposed when considered matmul terminology
    // because for matmul we would have A[S_q, S] x B[S, D].
    // It would have been transposed if A[S_q, S] x B[D, S].
    assert(rhs_is_transposed || stride == 1);
    // Generate activations
    auto [lhs, lhs_qvals, lhs_scales, lhs_zeros] =
        generate_per_token_quantized_tensor(m * stride, k);

    auto [rhs, rhs_qvals, rhs_scales, rhs_zeros] =
        generate_per_token_quantized_tensor(n * stride, k, !rhs_is_transposed);
    // Above function produces nxk matrix and to produce kxn you need transposed
    // = true. we do !rhs_is_transposed becaues when rhs_is_transposed = true
    // the shape should be nxk instead of kxn.

    // Compute expected output
    std::vector<float> expected_output(m * n);

    for (int m_idx = 0; m_idx < m; m_idx++) {
      for (int n_idx = 0; n_idx < n; n_idx++) {
        float res = 0.0;
        for (int k_idx = 0; k_idx < k; k_idx++) {
          int lhs_idx = m_idx * stride * k + k_idx;
          int rhs_idx = k_idx * stride * n + n_idx * stride;
          if (rhs_is_transposed) {
            rhs_idx = n_idx * stride * k + k_idx;
          }
          float lhs_dequant = lhs_scales[m_idx * stride] *
              (lhs_qvals[lhs_idx] - lhs_zeros[m_idx * stride]);

          float rhs_dequant = rhs_scales[n_idx * stride] *
              (rhs_qvals[rhs_idx] - rhs_zeros[n_idx * stride]);

          res += lhs_dequant * rhs_dequant;
        }
        expected_output[m_idx * n + n_idx] = res;
      }
    }

    // Return test case
    return channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case(
        m,
        k,
        n,
        stride,
        lhs_has_zeros,
        rhs_has_zeros,
        lhs_is_transposed,
        rhs_is_transposed,
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

template <int weight_nbit>
struct lowbit_embedding_test_case {
  int num_embeddings;
  int embedding_dim;
  int group_size;
  std::vector<int8_t> weight_qvals;
  std::vector<float> weight_scales;
  std::vector<int8_t> weight_zeros;
  std::vector<float> expected_outputs;

  lowbit_embedding_test_case(
      int num_embeddings,
      int embedding_dim,
      int group_size,
      std::vector<int8_t> weight_qvals,
      std::vector<float> weight_scales,
      std::vector<int8_t> weight_zeros,
      std::vector<float> expected_outputs)
      : num_embeddings{num_embeddings},
        embedding_dim{embedding_dim},
        group_size{group_size},
        weight_qvals{weight_qvals},
        weight_scales{weight_scales},
        weight_zeros{weight_zeros},
        expected_outputs{expected_outputs} {
    assert(embedding_dim % group_size == 0);
    assert(weight_qvals.size() == num_embeddings * embedding_dim);
    assert(
        weight_scales.size() == num_embeddings * (embedding_dim / group_size));
    assert(
        weight_zeros.size() == num_embeddings * (embedding_dim / group_size));
    assert(expected_outputs.size() == num_embeddings * embedding_dim);
  }

  static lowbit_embedding_test_case generate(
      int num_embeddings,
      int embedding_dim,
      int group_size,
      bool has_weight_zeros) {
    int groups_per_embedding = embedding_dim / group_size;

    auto weight_qvals = get_random_signed_lowbit_vector(
        num_embeddings * embedding_dim, weight_nbit);
    auto weight_scales =
        get_random_vector(num_embeddings * groups_per_embedding, 0.1, 1.0);

    std::vector<int8_t> weight_zeros;
    if (has_weight_zeros) {
      weight_zeros = get_random_signed_lowbit_vector(
          num_embeddings * groups_per_embedding, weight_nbit);
    } else {
      weight_zeros =
          std::vector<int8_t>(num_embeddings * groups_per_embedding, 0);
    }

    auto expected_outputs = std::vector<float>(num_embeddings * embedding_dim);
    for (int embedding_idx = 0; embedding_idx < num_embeddings;
         embedding_idx++) {
      for (int j = 0; j < embedding_dim; j++) {
        auto qval = weight_qvals[embedding_idx * embedding_dim + j];
        auto scale = weight_scales
            [embedding_idx * groups_per_embedding + j / group_size];
        auto zero =
            weight_zeros[embedding_idx * groups_per_embedding + j / group_size];
        expected_outputs[embedding_idx * embedding_dim + j] =
            scale * (qval - zero);
      }
    }

    return lowbit_embedding_test_case(
        num_embeddings,
        embedding_dim,
        group_size,
        weight_qvals,
        weight_scales,
        weight_zeros,
        expected_outputs);
  }
};

} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
