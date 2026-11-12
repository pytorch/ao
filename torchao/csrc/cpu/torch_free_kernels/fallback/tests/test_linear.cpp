// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/linear/channelwise_8bit_activation_groupwise_lowbit_weight.h>
#include <torchao/csrc/cpu/torch_free_kernels/test_utils.h>

namespace {

constexpr float kTol = 0.0001;

using namespace torchao::kernels::cpu::fallback::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight;

// Portable counterpart of the aarch64 test case generator: random activations
// quantized per row to int8 as the kernel does, random weights quantized per
// group to weight_nbit, and the expected output computed from the dequantized
// values.
struct test_case {
  float clamp_min = 0.0;
  float clamp_max = 0.0;
  std::vector<float> expected_output;
  std::vector<float> activations;
  std::vector<int8_t> weight_qvals;
  std::vector<float> weight_scales;
  std::vector<int8_t> weight_zeros;
  std::vector<float> bias;
};

test_case generate(
    int m,
    int k,
    int n,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    bool has_clamp) {
  test_case tc;
  tc.activations = torchao::get_random_vector(m * k, -1.0, 1.0);
  std::vector<int8_t> activation_qvals(m * k);
  std::vector<float> activation_scales(m);
  std::vector<int> activation_zeros(m);
  for (int m_idx = 0; m_idx < m; m_idx++) {
    int zero;
    internal::row_scale_and_zero(
        activation_scales[m_idx], zero, tc.activations.data() + m_idx * k, k);
    activation_zeros[m_idx] = static_cast<int8_t>(zero);
    internal::quantize_row(
        activation_qvals.data() + m_idx * k,
        tc.activations.data() + m_idx * k,
        k,
        activation_scales[m_idx],
        static_cast<int8_t>(zero));
  }

  const int qmin = -(1 << (weight_nbit - 1));
  const int qmax = (1 << (weight_nbit - 1)) - 1;
  const int n_groups = (n * k) / group_size;
  auto weights = torchao::get_random_vector(n * k, -1.0, 1.0);
  tc.weight_qvals.resize(n * k);
  tc.weight_scales.resize(n_groups);
  tc.weight_zeros.resize(n_groups);
  for (int group_idx = 0; group_idx < n_groups; group_idx++) {
    const float* vals = weights.data() + group_idx * group_size;
    const float vmin = *std::min_element(vals, vals + group_size);
    const float vmax = *std::max_element(vals, vals + group_size);
    const float scale = (vmax - vmin) / (qmax - qmin);
    const int zero = has_weight_zeros ? qmin - std::round(vmin / scale) : 0;
    tc.weight_scales[group_idx] = scale;
    tc.weight_zeros[group_idx] = zero;
    for (int i = 0; i < group_size; i++) {
      const int q = static_cast<int>(std::round(vals[i] / scale)) + zero;
      tc.weight_qvals[group_idx * group_size + i] =
          static_cast<int8_t>(std::max(qmin, std::min(q, qmax)));
    }
  }

  tc.bias = has_bias ? torchao::get_random_vector(n, -1.0, 1.0)
                     : std::vector<float>(n, 0.0);
  if (has_clamp) {
    tc.clamp_min = torchao::get_random_vector(1, -1.0, 0.2)[0];
    tc.clamp_max = torchao::get_random_vector(1, 0.3, 1.0)[0];
  }

  tc.expected_output.resize(m * n);
  for (int m_idx = 0; m_idx < m; m_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      float res = 0.0;
      for (int k_idx = 0; k_idx < k; k_idx++) {
        const int weight_idx = n_idx * k + k_idx;
        const int weight_group_idx = weight_idx / group_size;
        const float activation_dequant = activation_scales[m_idx] *
            (activation_qvals[m_idx * k + k_idx] - activation_zeros[m_idx]);
        const float weight_dequant = tc.weight_scales[weight_group_idx] *
            (tc.weight_qvals[weight_idx] - tc.weight_zeros[weight_group_idx]);
        res += activation_dequant * weight_dequant;
      }
      res += tc.bias[n_idx];
      if (has_clamp) {
        res = std::min(std::max(res, tc.clamp_min), tc.clamp_max);
      }
      tc.expected_output[m_idx * n + n_idx] = res;
    }
  }
  return tc;
}

template <int weight_nbit, bool has_weight_zeros>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16(
    int m,
    int k,
    int n,
    int group_size,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 1;
  constexpr int nr = 8;
  constexpr int kr = 16;
  constexpr int sr = 2;

  auto tc = generate(
      m, k, n, group_size, weight_nbit, has_weight_zeros, has_bias, has_clamp);

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      packed_activations.data(),
      m,
      k,
      group_size,
      tc.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);

  std::vector<char> packed_weights(torchao::weight_packing::packed_weights_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  torchao::weight_packing::pack_weights<weight_nbit, nr, kr, sr>(
      packed_weights.data(),
      n,
      k,
      group_size,
      tc.weight_qvals.data(),
      tc.weight_scales.data(),
      has_weight_zeros ? tc.weight_zeros.data() : nullptr,
      has_bias ? tc.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x8x16_f32_fallback<weight_nbit, has_weight_zeros, /*has_lut*/ false>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      tc.clamp_min,
      tc.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], tc.expected_output[i], kTol);
  }
}

template <int weight_nbit, bool has_weight_zeros>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_lut(
    int m,
    int k,
    int n,
    int group_size,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 1;
  constexpr int nr = 8;
  constexpr int kr = 16;
  constexpr int sr = 2;

  auto tc = generate(
      m, k, n, group_size, weight_nbit, has_weight_zeros, has_bias, has_clamp);

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      packed_activations.data(),
      m,
      k,
      group_size,
      tc.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);

  // Define equivalent LUT for affine quantization
  constexpr int lut_size = (1 << weight_nbit);
  constexpr int offset = (1 << (weight_nbit - 1));
  std::vector<int8_t> weight_qval_idxs(tc.weight_qvals.size());
  std::vector<int8_t> lut(lut_size, 0);
  for (size_t i = 0; i < tc.weight_qvals.size(); i++) {
    weight_qval_idxs[i] = tc.weight_qvals[i] + offset;
  }
  for (int i = 0; i < lut_size; i++) {
    lut[i] = i - offset;
  }

  std::vector<char> packed_weights(packed_weights_with_lut_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  pack_weights_with_lut<weight_nbit, nr, kr, sr>(
      packed_weights.data(),
      n,
      k,
      group_size,
      weight_qval_idxs.data(),
      /*n_luts*/ 1,
      lut.data(),
      tc.weight_scales.data(),
      has_weight_zeros ? tc.weight_zeros.data() : nullptr,
      has_bias ? tc.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x8x16_f32_fallback<weight_nbit, has_weight_zeros, /*has_lut*/ true>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      tc.clamp_min,
      tc.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], tc.expected_output[i], kTol);
  }
}

} // namespace

TEST(Channelwise8bitActivationGroupwiseLowbitWeight, tile_1x8x16) {
  constexpr int weight_nbit = 4;

  // Standard
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With weight zeros
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      weight_nbit,
      /*has_weight_zeros=*/true>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With bias
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/true,
      /*has_clamp=*/false);

  // With clamp
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/true);

  // n less than 8
  for (int n = 1; n < 8; n++) {
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
        weight_nbit,
        /*has_weight_zeros=*/false>(
        /*m=*/7,
        /*k=*/64,
        /*n=*/n,
        /*group_size=*/16,
        /*has_bias=*/false,
        /*has_clamp=*/false);
  }

  // m of a 16-row prefill call
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      weight_nbit,
      /*has_weight_zeros=*/true>(
      /*m=*/16,
      /*k=*/128,
      /*n=*/24,
      /*group_size=*/32,
      /*has_bias=*/true,
      /*has_clamp=*/false);
}

TEST(
    Channelwise8bitActivationGroupwiseLowbitWeight,
    tile_1x8x16_other_bitwidths) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      /*weight_nbit*/ 1,
      /*has_weight_zeros=*/false>(7, 64, 13, 16, false, false);
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      /*weight_nbit*/ 2,
      /*has_weight_zeros=*/true>(7, 64, 13, 16, true, false);
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      /*weight_nbit*/ 3,
      /*has_weight_zeros=*/false>(7, 64, 13, 16, false, true);
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      /*weight_nbit*/ 5,
      /*has_weight_zeros=*/true>(7, 64, 13, 16, false, false);
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      /*weight_nbit*/ 6,
      /*has_weight_zeros=*/false>(7, 64, 13, 16, true, false);
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      /*weight_nbit*/ 7,
      /*has_weight_zeros=*/false>(7, 64, 13, 16, false, false);
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16<
      /*weight_nbit*/ 8,
      /*has_weight_zeros=*/false>(7, 64, 13, 16, false, false);
}

TEST(Channelwise8bitActivationGroupwiseLowbitWeight, LUT) {
  constexpr int weight_nbit = 4;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      weight_nbit,
      /*has_weight_zeros*/ false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // has_bias
  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/true,
      /*has_clamp=*/false);

  // has_clamp
  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      weight_nbit,
      /*has_weight_zeros*/ false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/true);

  // n less than 8 (nr)
  for (int n = 1; n < 8; n++) {
    test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
        weight_nbit,
        /*has_weight_zeros=*/false>(
        /*m=*/7,
        /*k=*/64,
        /*n=*/n,
        /*group_size=*/16,
        /*has_bias=*/false,
        /*has_clamp=*/false);
  }

  // Other bitwidths
  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      /*weight_nbit*/ 1,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      /*weight_nbit*/ 2,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      /*weight_nbit*/ 3,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // m of a 16-row prefill call
  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      /*weight_nbit*/ 2,
      /*has_weight_zeros=*/false>(
      /*m=*/16,
      /*k=*/128,
      /*n=*/24,
      /*group_size=*/128,
      /*has_bias=*/true,
      /*has_clamp=*/false);
}
