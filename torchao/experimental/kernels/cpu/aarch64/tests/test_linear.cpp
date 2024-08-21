// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <arm_neon.h>
#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <vector>

float kTol = 0.0001;

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot() {
  int m = 7;
  int k = 128;
  int n = 13;
  int group_size = 32;

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
      channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot;

  std::vector<char> activation_data(
      activation_data_size<has_weight_zeros>(m, k, group_size));
  prepare_activation_data<has_weight_zeros>(
      (void*)activation_data.data(),
      m,
      k,
      group_size,
      test_case.activations.data());

  std::vector<char> weight_data(
      weight_data_size<weight_nbit, has_weight_zeros>(n, k, group_size));
  prepare_weight_data<weight_nbit, has_weight_zeros>(
      (void*)weight_data.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      /*weight_zeros=*/test_case.weight_zeros.data());

  std::vector<float> output(m * k);
  kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      weight_data.data(),
      activation_data.data(),
      /*bias=*/test_case.bias.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot,
    Standard) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot,
    HasWeightZeros) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = true;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot,
    HasBias) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = true;
  constexpr bool has_clamp = false;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot,
    HasClamp) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = true;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot() {
  int m = 7;
  int k = 64;
  int n = 13;
  int group_size = 16;

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
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot;

  std::vector<char> activation_data(
      activation_data_size<has_weight_zeros>(m, k, group_size));
  prepare_activation_data<has_weight_zeros>(
      (void*)activation_data.data(),
      m,
      k,
      group_size,
      test_case.activations.data());

  std::vector<char> weight_data(
      weight_data_size<weight_nbit, has_weight_zeros>(n, k, group_size));
  prepare_weight_data<weight_nbit, has_weight_zeros>(
      (void*)weight_data.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      /*weight_zeros=*/test_case.weight_zeros.data());

  std::vector<float> output(m * k);
  kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      weight_data.data(),
      activation_data.data(),
      /*bias=*/test_case.bias.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot,
    Standard) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot,
    HasWeightZeros) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = true;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot,
    HasBias) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = true;
  constexpr bool has_clamp = false;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot,
    HasClamp) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = true;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot() {
  int m = 7;
  int k = 64;
  int n = 13;
  int group_size = 16;

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
      channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot;

  std::vector<char> activation_data(
      activation_data_size<has_weight_zeros>(m, k, group_size));
  prepare_activation_data<has_weight_zeros>(
      (void*)activation_data.data(),
      m,
      k,
      group_size,
      test_case.activations.data());

  std::vector<char> weight_data(
      weight_data_size<weight_nbit, has_weight_zeros>(n, k, group_size));
  prepare_weight_data<weight_nbit, has_weight_zeros>(
      (void*)weight_data.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      /*weight_zeros=*/test_case.weight_zeros.data());

  std::vector<float> output(m * k);
  kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>(
      output.data(),
      /*output_m_stride=*/n,
      m,
      n,
      k,
      group_size,
      weight_data.data(),
      activation_data.data(),
      /*bias=*/test_case.bias.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot,
    Standard) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot,
    HasWeightZeros) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = true;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot,
    HasBias) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = true;
  constexpr bool has_clamp = false;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot,
    HasClamp) {
  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = true;

  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp>();
}
