// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <vector>

#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/channelwise_8bit_activation_groupwise_lowbit_weight.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>

float kTol = 0.0001;

template <int weight_nbit, bool has_weight_zeros>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32(
    int m,
    int k,
    int n,
    int group_size,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 1;
  constexpr int nr = 1;
  constexpr int kr = 32;
  constexpr int sr = 1;

  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      (void*)packed_activations.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);

  std::vector<char> packed_weights(packed_weights_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  pack_weights<weight_nbit, nr, kr, sr>(
      (void*)packed_weights.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      has_weight_zeros ? test_case.weight_zeros.data() : nullptr,
      has_bias ? test_case.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x1x32_f32_neondot<weight_nbit>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

template <int weight_nbit, bool has_weight_zeros>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16(
    int m,
    int k,
    int n,
    int group_size,
    bool has_bias,
    bool has_clamp) {
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 16;
  constexpr int sr = 2;

  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      (void*)packed_activations.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);

  std::vector<char> packed_weights(packed_weights_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  pack_weights<weight_nbit, nr, kr, sr>(
      (void*)packed_weights.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      has_weight_zeros ? test_case.weight_zeros.data() : nullptr,
      has_bias ? test_case.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x4x16_f32_neondot<weight_nbit>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
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

  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      (void*)packed_activations.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);

  std::vector<char> packed_weights(packed_weights_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  pack_weights<weight_nbit, nr, kr, sr>(
      (void*)packed_weights.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      has_weight_zeros ? test_case.weight_zeros.data() : nullptr,
      has_bias ? test_case.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x8x16_f32_neondot<weight_nbit, has_weight_zeros, /*has_lut*/ false>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, tile_1x1x32) {
  constexpr int weight_nbit = 4;

  // Standard
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/32,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With weight zeros
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32<
      weight_nbit,
      /*has_weight_zeros=*/true>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/32,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With bias
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/32,
      /*has_bias=*/true,
      /*has_clamp=*/false);

  // With clamp
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/32,
      /*has_bias=*/false,
      /*has_clamp=*/true);
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, tile_1x4x16) {
  constexpr int weight_nbit = 4;

  // Standard
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With weight zeros
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
      weight_nbit,
      /*has_weight_zeros=*/true>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  // With bias
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/true,
      /*has_clamp=*/false);

  // With clamp
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
      weight_nbit,
      /*has_weight_zeros=*/false>(
      /*m=*/7,
      /*k=*/64,
      /*n=*/13,
      /*group_size=*/16,
      /*has_bias=*/false,
      /*has_clamp=*/true);

  // n less than 4
  for (int n = 1; n < 4; n++) {
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16<
        weight_nbit,
        /*has_weight_zeros=*/false>(
        /*m=*/7,
        /*k=*/64,
        /*n=*/n,
        /*group_size=*/16,
        /*has_bias=*/false,
        /*has_clamp=*/false);
  }
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, tile_1x8x16) {
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

  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  std::vector<char> packed_activations(
      packed_activations_size(m, k, group_size, has_weight_zeros, mr, kr, sr));
  pack_activations<mr, kr, sr>(
      (void*)packed_activations.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros,
      mr,
      kr,
      sr);

  // Define equivalent LUT for affine quantization
  constexpr int lut_size = (1 << weight_nbit);
  std::vector<int8_t> weight_qval_idxs(test_case.weight_qvals.size());
  std::vector<int8_t> lut(lut_size, 0);
  constexpr int offset = (1 << (weight_nbit - 1));
  for (int i = 0; i < test_case.weight_qvals.size(); i++) {
    weight_qval_idxs[i] = test_case.weight_qvals[i] + offset;
  }
  for (int i = 0; i < lut_size; i++) {
    lut[i] = i - offset;
  }

  std::vector<char> packed_weights(packed_weights_with_lut_size(
      n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr));
  pack_weights_with_lut<weight_nbit, nr, kr, sr>(
      (void*)packed_weights.data(),
      n,
      k,
      group_size,
      weight_qval_idxs.data(),
      /*n_luts*/ 1,
      lut.data(),
      test_case.weight_scales.data(),
      has_weight_zeros ? test_case.weight_zeros.data() : nullptr,
      has_bias ? test_case.bias.data() : nullptr,
      nr,
      kr,
      sr);

  std::vector<float> output(m * n);
  kernel_1x8x16_f32_neondot<weight_nbit, has_weight_zeros, /*has_lut*/ true>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      packed_weights.data(),
      packed_activations.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max,
      has_weight_zeros,
      has_bias,
      has_clamp);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, LUT) {
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

  // has_weight_zeros
  test_channelwise_8bit_activation_groupwise_lowbit_weight_lut<
      weight_nbit,
      /*has_weight_zeros*/ true>(
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
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
