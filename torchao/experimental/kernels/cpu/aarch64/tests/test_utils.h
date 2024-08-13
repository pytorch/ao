// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
#include <torchao/experimental/kernels/cpu/aarch64/quantization/quantize.h>
#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>
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
  assert(nbit >= 2);
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
    assert(bias.size() == m);

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
      bool has_clamp) {
    // activations is m x k (stored in row-major)
    // weights is k x n (stored in column-major)

    // Generate activations
    auto activations = get_random_vector(m * k, -1.0, 1.0);
    auto activation_qvals = std::vector<int8_t>(m * k, 0);
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
          vmin, vmax, /*vals=*/activations.data() + m_idx * k, /*size=*/k);
      torchao::quantization::get_scale_and_zero(
          scale, zero, vmin, vmax, qmin, qmax);
      activation_scales[m_idx] = scale;
      activation_zeros[m_idx] = zero;
      torchao::kernels::cpu::aarch64::quantization::quantize(
          /*qvals=*/activation_qvals.data() + m_idx * k,
          /*vals=*/activations.data() + m_idx * k,
          /*size=*/k,
          scale,
          zero,
          qmin,
          qmax);
    }

    //  Generate weights
    assert(k % weight_group_size == 0);
    int n_weight_groups = (n * k) / weight_group_size;
    auto weights = get_random_vector(n * k, -1.0, 1.0);
    auto weight_qvals = std::vector<int8_t>(n * k, 0);
    auto weight_scales = std::vector<float>(n_weight_groups, 0.0);
    auto weight_zeros = std::vector<int8_t>(n_weight_groups, 0);

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

    std::vector<float> bias(m, 0.0);
    if (has_bias) {
      bias = get_random_vector(m, -1.0, 1.0);
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
        res += bias[m_idx];
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

} // namespace torchao
