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
#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>

#ifdef TORCHAO_ENABLE_KLEIDI
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#ifdef TORCHAO_ENABLE_ARM_I8MM
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h>
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h>
#endif // TORCHAO_ENABLE_ARM_I8MM
#endif // TORCHAO_ENABLE_KLEIDI

float kTol = 0.0001;

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot(
    int m, int k, int n, int group_size) {
  auto test_case =
      torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case::
          generate(m, k, n, group_size, weight_nbit, has_weight_zeros, has_bias,
                   has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot;

  std::vector<char> activation_data(
      activation_data_size<has_weight_zeros>(m, k, group_size));
  prepare_activation_data<has_weight_zeros>((void *)activation_data.data(), m,
                                            k, group_size,
                                            test_case.activations.data());

  std::vector<char> weight_data(
      weight_data_size<weight_nbit, has_weight_zeros, has_bias>(n, k,
                                                                group_size));
  prepare_weight_data<weight_nbit, has_weight_zeros, has_bias>(
      (void *)weight_data.data(), n, k, group_size,
      test_case.weight_qvals.data(), test_case.weight_scales.data(),
      /*weight_zeros=*/test_case.weight_zeros.data(), test_case.bias.data());

  std::vector<float> output(m * n);
  kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>(
      output.data(),
      /*output_m_stride=*/n, m, n, k, group_size, weight_data.data(),
      activation_data.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot,
    Standard) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/7, /*k=*/128, /*n=*/13, /*group_size=*/32);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot,
    HasWeightZeros) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot<
      4 /*weight_nbit*/, true /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/7, /*k=*/128, /*n=*/13, /*group_size=*/32);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot,
    HasBias) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/7, /*k=*/128, /*n=*/13, /*group_size=*/32);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot,
    HasClamp) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/>(
      /*m=*/7, /*k=*/128, /*n=*/13, /*group_size=*/32);
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot(
    int m, int k, int n, int group_size) {
  auto test_case =
      torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case::
          generate(m, k, n, group_size, weight_nbit, has_weight_zeros, has_bias,
                   has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot;

  std::vector<char> activation_data(
      activation_data_size<has_weight_zeros>(m, k, group_size));
  prepare_activation_data<has_weight_zeros>((void *)activation_data.data(), m,
                                            k, group_size,
                                            test_case.activations.data());

  std::vector<char> weight_data(
      weight_data_size<weight_nbit, has_weight_zeros, has_bias>(n, k,
                                                                group_size));
  prepare_weight_data<weight_nbit, has_weight_zeros, has_bias>(
      (void *)weight_data.data(), n, k, group_size,
      test_case.weight_qvals.data(), test_case.weight_scales.data(),
      /*weight_zeros=*/test_case.weight_zeros.data(), test_case.bias.data());

  std::vector<float> output(m * n);
  kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>(
      output.data(),
      /*output_m_stride=*/n, m, n, k, group_size, weight_data.data(),
      activation_data.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot,
    Standard) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/7, /*k=*/64, /*n=*/13, /*group_size=*/16);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot,
    HasWeightZeros) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<
      4 /*weight_nbit*/, true /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/7, /*k=*/64, /*n=*/13, /*group_size=*/16);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot,
    HasBias) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/7, /*k=*/64, /*n=*/13, /*group_size=*/16);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot,
    HasClamp) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/>(
      /*m=*/7, /*k=*/64, /*n=*/13, /*group_size=*/16);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot,
    NLessThan4) {
  for (int n = 1; n < 4; n++) {
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<
        4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
        true /*has_clamp*/>(
        /*m=*/7, /*k=*/64, /*n=*/n, /*group_size=*/16);
  }
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot(
    int m, int k, int n, int group_size) {
  auto test_case =
      torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case::
          generate(m, k, n, group_size, weight_nbit, has_weight_zeros, has_bias,
                   has_clamp);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot;

  std::vector<char> activation_data(
      activation_data_size<has_weight_zeros>(m, k, group_size));
  prepare_activation_data<has_weight_zeros>((void *)activation_data.data(), m,
                                            k, group_size,
                                            test_case.activations.data());

  std::vector<char> weight_data(
      weight_data_size<weight_nbit, has_weight_zeros, has_bias>(n, k,
                                                                group_size));
  prepare_weight_data<weight_nbit, has_weight_zeros, has_bias>(
      (void *)weight_data.data(), n, k, group_size,
      test_case.weight_qvals.data(), test_case.weight_scales.data(),
      /*weight_zeros=*/test_case.weight_zeros.data(), test_case.bias.data());

  std::vector<float> output(m * n);
  kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>(
      output.data(),
      /*output_m_stride=*/n, m, n, k, group_size, weight_data.data(),
      activation_data.data(),
      /*clamp_min=*/test_case.clamp_min,
      /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot,
    Standard) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/7, /*k=*/64, /*n=*/13, /*group_size=*/16);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot,
    HasWeightZeros) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<
      4 /*weight_nbit*/, true /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/7, /*k=*/64, /*n=*/13, /*group_size=*/16);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot,
    HasBias) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/7, /*k=*/64, /*n=*/13, /*group_size=*/16);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot,
    HasClamp) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/>(
      /*m=*/7, /*k=*/64, /*n=*/13, /*group_size=*/16);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot,
    NLessThan8) {
  for (int n = 1; n < 8; n++) {
    test_channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<
        4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
        true /*has_clamp*/>(
        /*m=*/7, /*k=*/64, /*n=*/n, /*group_size=*/16);
  }
}

#ifdef TORCHAO_ENABLE_KLEIDI
template <bool has_bias, bool has_clamp>
void test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
    int m, int k, int n, int group_size) {
  auto test_case =
      torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case::
          generate(m, k, n, group_size,
                   /*weight_nbit=*/4,
                   /*has_weight_zeros*/ false, has_bias, has_clamp,
                   /*weight_scale_bf16_round_trip=*/true);

  using namespace torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x4x32;

  std::vector<char> activation_data(activation_data_size(m, k, group_size));

  prepare_activation_data((void *)activation_data.data(), m, k, group_size,
                          test_case.activations.data());

  std::vector<char> weight_data(weight_data_size(n, k, group_size));

  prepare_weight_data((void *)weight_data.data(), n, k, group_size,
                      test_case.weight_qvals.data(),
                      test_case.weight_scales.data(),
                      /*weight_zeros=*/test_case.weight_zeros.data(),
                      /*bias=*/test_case.bias.data());

  std::vector<float> output(m * n);
  kernel(output.data(),
         /*output_m_stride=*/n, m, n, k, group_size, weight_data.data(),
         activation_data.data(),
         /*clamp_min=*/test_case.clamp_min,
         /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     k_eq_gs_32) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/32, /*n=*/4, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     large_k_n_gs32) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/1024, /*n=*/512, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     even_n_gs32) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/1024, /*n=*/182, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     clamp_k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod<
      false /*has_bias*/, true /*has_clamp*/>(
      /*m=*/1, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     m_clamp_k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod<
      false /*has_bias*/, true /*has_clamp*/>(
      /*m=*/11, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

template <bool has_bias, bool has_clamp>
void test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod(
    int m, int k, int n, int group_size) {
  auto test_case =
      torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case::
          generate(m, k, n, group_size,
                   /*weight_nbit=*/4,
                   /*has_weight_zeros=*/false, has_bias, has_clamp,
                   /*round_weight_scales_to_bf16=*/true);

  using namespace torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x8x32;

  std::vector<char> activation_data(activation_data_size(m, k, group_size));

  prepare_activation_data((void *)activation_data.data(), m, k, group_size,
                          test_case.activations.data());

  std::vector<char> weight_data(weight_data_size(n, k, group_size));

  prepare_weight_data((void *)weight_data.data(), n, k, group_size,
                      test_case.weight_qvals.data(),
                      test_case.weight_scales.data(),
                      /*weight_zeros=*/test_case.weight_zeros.data(),
                      /*bias=*/test_case.bias.data());

  std::vector<float> output(m * n);
  kernel(output.data(),
         /*output_m_stride=*/n, m, n, k, group_size, weight_data.data(),
         activation_data.data(),
         /*clamp_min=*/test_case.clamp_min,
         /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
     k_eq_gs_32) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/32, /*n=*/4, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
     large_k_n_gs32) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/1024, /*n=*/512, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
     even_n_gs32) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/1024, /*n=*/182, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
     k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
     clamp_k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod<
      false /*has_bias*/, true /*has_clamp*/>(
      /*m=*/1, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
     m_clamp_k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod<
      false /*has_bias*/, true /*has_clamp*/>(
      /*m=*/11, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

#ifdef TORCHAO_ENABLE_ARM_I8MM
template <bool has_bias, bool has_clamp>
void test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    int m, int k, int n, int group_size) {
  auto test_case =
      torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case::
          generate(m, k, n, group_size,
                   /*weight_nbit=*/4,
                   /*has_weight_zeros=*/false, has_bias, has_clamp,
                   /*round_weight_scales_to_bf16=*/true);

  using namespace torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_i8mm_8x4x32;

  std::vector<char> activation_data(activation_data_size(m, k, group_size));

  prepare_activation_data((void *)activation_data.data(), m, k, group_size,
                          test_case.activations.data());

  std::vector<char> weight_data(weight_data_size(n, k, group_size));

  prepare_weight_data((void *)weight_data.data(), n, k, group_size,
                      test_case.weight_qvals.data(),
                      test_case.weight_scales.data(),
                      /*weight_zeros=*/test_case.weight_zeros.data(),
                      /*bias=*/test_case.bias.data());

  std::vector<float> output(m * n);
  kernel(output.data(),
         /*output_m_stride=*/n, m, n, k, group_size, weight_data.data(),
         activation_data.data(),
         /*clamp_min=*/test_case.clamp_min,
         /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     k_eq_gs_32) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/32, /*n=*/4, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     large_k_n_gs32) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/1024, /*n=*/512, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     even_n_gs32) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/1024, /*n=*/182, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     clamp_k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm<
      false /*has_bias*/, true /*has_clamp*/>(
      /*m=*/1, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
     m_clamp_k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm<
      false /*has_bias*/, true /*has_clamp*/>(
      /*m=*/11, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

template <bool has_bias, bool has_clamp>
void test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm(
    int m, int k, int n, int group_size) {
  auto test_case =
      torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case::
          generate(m, k, n, group_size,
                   /*weight_nbit=*/4,
                   /*has_weight_zeros=*/false, has_bias, has_clamp,
                   /*round_weight_scales_to_bf16=*/true);

  using namespace torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_i8mm_4x8x32;

  std::vector<char> activation_data(activation_data_size(m, k, group_size));

  prepare_activation_data((void *)activation_data.data(), m, k, group_size,
                          test_case.activations.data());

  std::vector<char> weight_data(weight_data_size(n, k, group_size));

  prepare_weight_data((void *)weight_data.data(), n, k, group_size,
                      test_case.weight_qvals.data(),
                      test_case.weight_scales.data(),
                      /*weight_zeros=*/test_case.weight_zeros.data(),
                      /*bias=*/test_case.bias.data());

  std::vector<float> output(m * n);
  kernel(output.data(),
         /*output_m_stride=*/n, m, n, k, group_size, weight_data.data(),
         activation_data.data(),
         /*clamp_min=*/test_case.clamp_min,
         /*clamp_max=*/test_case.clamp_max);

  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
  }
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
     k_eq_gs_32) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/32, /*n=*/4, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
     large_k_n_gs32) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/1024, /*n=*/512, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
     even_n_gs32) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/1024, /*n=*/182, /*group_size=*/32);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
     k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm<
      false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/1, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
     clamp_k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm<
      false /*has_bias*/, true /*has_clamp*/>(
      /*m=*/1, /*k=*/128, /*n=*/182, /*group_size=*/128);
}

TEST(test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
     m_clamp_k_eq_gs128) {
  test_kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm<
      false /*has_bias*/, true /*has_clamp*/>(
      /*m=*/11, /*k=*/128, /*n=*/182, /*group_size=*/128);
}
#endif // TORCHAO_ENABLE_ARM_I8MM
#endif // TORCHAO_ENABLE_KLEIDI
#endif // defined(__aarch64__) || defined(__ARM_NEON)
