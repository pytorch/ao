// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <vector>

#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/aarch64/matmul/matmul.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>

float kTol = 0.0001;

template <
    bool a_has_zeros,
    bool b_has_zeros,
    bool a_transposed,
    bool b_transposed>
struct test_channelwise_8bit_channelwise_8bit_b {
  static void Run(int m, int k, int n);
};

template <bool a_has_zeros, bool b_has_zeros>
struct test_channelwise_8bit_channelwise_8bit_b<
    a_has_zeros,
    b_has_zeros,
    false,
    true> {
  static void Run(int m, int k, int n, int stride = 1) {
    auto test_case =
        torchao::channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case::
            generate(m, k, n, a_has_zeros, a_has_zeros, false, true, stride);

    using namespace torchao::kernels::cpu::aarch64::quantized_matmul::
        channelwise_8bit_a_channelwise_8bit_b_1x8x16_f32_neondot;

    std::vector<float> output(m * n);
    kernel<a_has_zeros, b_has_zeros, false, true>(
        m,
        n,
        k,
        test_case.lhs_qvals.data(),
        k * stride /*lsh_stride_m*/,
        test_case.rhs_qvals.data(),
        k * stride /*rsh_stride_n*/,
        output.data(),
        n /*out_stride_n*/,
        test_case.lhs_zeros.data(),
        test_case.rhs_zeros.data(),
        test_case.lhs_scales.data(),
        test_case.rhs_scales.data(),
        stride, /*lhs qparams stride*/
        stride /*rhs qparams stride*/);

    for (int i = 0; i < m * n; i++) {
      EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
    }
  }
};

template <bool a_has_zeros, bool b_has_zeros>
struct test_channelwise_8bit_channelwise_8bit_b<
    a_has_zeros,
    b_has_zeros,
    false,
    false> {
  static void Run(int m, int k, int n, int stride = 1) {
    auto test_case =
        torchao::channelwise_8bit_a_channelwise_8bit_b_qmatmul_test_case::
            generate(m, k, n, a_has_zeros, a_has_zeros, false, false);

    using namespace torchao::kernels::cpu::aarch64::quantized_matmul::
        channelwise_8bit_a_channelwise_8bit_b_1x16x16_f32_smlal;

    std::vector<float> output(m * n);
    kernel<a_has_zeros, b_has_zeros, false, false>(
        m,
        n,
        k,
        test_case.lhs_qvals.data(),
        k /*lsh_stride_m*/,
        test_case.rhs_qvals.data(),
        n /*rsh_stride_n*/,
        output.data(),
        n /*out_stride_n*/,
        test_case.lhs_zeros.data(),
        test_case.rhs_zeros.data(),
        test_case.lhs_scales.data(),
        test_case.rhs_scales.data(),
        stride, /*lhs qparams stride*/
        stride /*rhs qparams stride*/);

    for (int i = 0; i < m * n; i++) {
      EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
    }
  }
};

TEST(test_channelwise_8bit_channelwise_8bit_b, TransposedBWithZeroPoints) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/1, /*k=*/128, /*n=*/16);
}

TEST(test_channelwise_8bit_channelwise_8bit_b, TransposeBWithZeroPointsLargeM) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/128, /*n=*/16);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TransposedBWithZeroPointsOddSizes) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/24);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TransposedBWithZeroPointsOddSizes2) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/19);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TransposedBWithZeroPointsStrided) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/1, /*k=*/128, /*n=*/16, 5);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TransposedBWithZeroPointsOddSizes2Strided) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/19, 10);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    TransposedBWithZeroPointsOddSizes2Strided2) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      true /*b_transposed*/>::
      Run(
          /*m=*/3, /*k=*/64, /*n=*/24, 7);
}

TEST(test_channelwise_8bit_channelwise_8bit_b, NoTransposedWithZeroPoints) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      false /*b_transposed*/>::
      Run(
          /*m=*/1, /*k=*/128, /*n=*/16);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    NoTransposedWithZeroPointsLargeM) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      false /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/128, /*n=*/16);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    NoTransposedWithZeroPointsOddSizes) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      false /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/24);
}

TEST(
    test_channelwise_8bit_channelwise_8bit_b,
    NoTransposedWithZeroPointsOddSizes2) {
  test_channelwise_8bit_channelwise_8bit_b<
      true /*a_has_zeros*/,
      true /*b_has_zeros*/,
      false /*a_transposed*/,
      false /*b_transposed*/>::
      Run(
          /*m=*/4, /*k=*/37, /*n=*/19);
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
