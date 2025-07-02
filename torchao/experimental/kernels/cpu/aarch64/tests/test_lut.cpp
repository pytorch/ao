// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/kernels/cpu/aarch64/lut/lut.h>
#include <vector>


TEST(test_fp32_lut, LutLookup) {
  auto lut = torchao::get_random_vector(16, -1.0, 1.0);
  auto idx = torchao::get_random_lowbit_vector(16, 4);

  uint8x16_t idx_vec = vld1q_u8(idx.data());
  uint8x16x4_t lut_vec;
  torchao::lut::load_fp32_lut(lut_vec, lut.data());

  float32x4_t out0, out1, out2, out3;
  torchao::lut::lookup_from_fp32_lut(out0, out1, out2, out3, lut_vec, idx_vec);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(out0[i], lut[idx[i]]);
    EXPECT_EQ(out1[i], lut[idx[i + 4]]);
    EXPECT_EQ(out2[i], lut[idx[i + 8]]);
    EXPECT_EQ(out3[i], lut[idx[i + 12]]);
  }
}


#endif // defined(__aarch64__) || defined(__ARM_NEON)
