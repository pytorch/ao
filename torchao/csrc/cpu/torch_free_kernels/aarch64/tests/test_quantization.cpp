// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <gtest/gtest.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/quantization/quantize.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/reduction.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <vector>

// Demonstrate some basic assertions.
TEST(test_quantize, ExpectedOutput) {
  std::array<float, 8> vals = {1.0, 2.5, -5.2, 10.2, 11.1, -3.15, -8.1, 7.3};
  std::array<std::pair<int, std::array<float, 8>>, 5> nBitToExpectedResult{
      {{2, {0.0, 0.0, -6.4, 12.8, 12.8, 0, -6.4, 6.4}},
       {3,
        {0.0,
         2.74286,
         -5.48571,
         10.9714,
         10.9714,
         -2.74286,
         -8.22857,
         8.22857}},
       {4, {1.28, 2.56, -5.12, 10.24, 11.52, -2.56, -7.68, 7.68}},
       {5,
        {1.23871,
         2.47742,
         -4.95484,
         9.90968,
         11.1484,
         -3.09677,
         -8.05161,
         7.43226}},
       {8,
        {0.978824,
         2.48471,
         -5.19529,
         10.1647,
         11.0682,
         -3.16235,
         -8.13177,
         7.30353}}}};

  int qmin, qmax, zero;
  float vmin, vmax, scale;

  torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
      vmin, vmax, vals.data(), vals.size());

  std::vector<int8_t> qvals(vals.size());

  for (auto [nbit, expectedResult] : nBitToExpectedResult) {
    torchao::quantization::get_qvals_range(
        qmin, qmax, nbit, /*is_symmetric=*/false);

    torchao::quantization::get_scale_and_zero(
        scale, zero, vmin, vmax, qmin, qmax);

    torchao::kernels::cpu::aarch64::quantization::quantize(
        qvals.data(), vals.data(), vals.size(), scale, zero, qmin, qmax);

    for (int i = 0; i < vals.size(); ++i) {
      float dq = scale * (qvals[i] - zero);
      EXPECT_NEAR(dq, expectedResult[i], 0.0001);
    }
  }
}

TEST(test_quantize, VectorAndScalarTailMatch) {
  constexpr int size = 17;
  constexpr float scale = 0.125f;
  constexpr int8_t zero = -3;
  constexpr int8_t qmin = -128;
  constexpr int8_t qmax = 127;
  std::array<float, size> vals = {
      -20.0f,
      -15.91f,
      -8.03f,
      -1.01f,
      -0.19f,
      0.01f,
      0.18f,
      1.01f,
      2.01f,
      3.91f,
      5.01f,
      7.01f,
      9.01f,
      12.91f,
      15.99f,
      19.99f,
      0.31f};
  std::array<int8_t, size> expected;
  std::array<int8_t, size> actual;

  const int original_rounding_mode = fegetround();
  fesetround(FE_TONEAREST);
  const float inv_scale = 1.0f / (scale + 1e-16f);
  for (int i = 0; i < size; i++) {
    const int qval = static_cast<int>(
        std::nearbyint(zero + vals[i] * inv_scale));
    expected[i] = static_cast<int8_t>(std::max(
        static_cast<int>(qmin), std::min(qval, static_cast<int>(qmax))));
  }

  fesetround(FE_DOWNWARD);
  torchao::kernels::cpu::aarch64::quantization::quantize(
      actual.data(), vals.data(), size, scale, zero, qmin, qmax);
  EXPECT_EQ(fegetround(), FE_DOWNWARD);
  fesetround(original_rounding_mode);

  EXPECT_EQ(actual, expected);
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
