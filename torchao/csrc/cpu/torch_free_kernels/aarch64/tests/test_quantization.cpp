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

#endif // defined(__aarch64__) || defined(__ARM_NEON)
