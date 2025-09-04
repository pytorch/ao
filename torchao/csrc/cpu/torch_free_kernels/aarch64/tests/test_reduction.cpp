// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <gtest/gtest.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/reduction.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/tests/test_utils.h>
#include <algorithm>
#include <vector>

TEST(test_find_min_and_sum, SizeHasRemainderAfterDivideBy4) {
  auto vals = torchao::get_random_vector(19, -1.0, 1.0);
  float vmin, vmax;
  torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
      vmin, vmax, vals.data(), vals.size());

  auto expected_vmin = *std::min_element(vals.begin(), vals.end());
  auto expected_vmax = *std::max_element(vals.begin(), vals.end());
  EXPECT_EQ(vmin, expected_vmin);
  EXPECT_EQ(vmax, expected_vmax);
}

TEST(test_find_min_and_sum, SizeSmallerThan4) {
  auto vals = torchao::get_random_vector(3, -1.0, 1.0);
  float vmin, vmax;
  torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
      vmin, vmax, vals.data(), vals.size());

  auto expected_vmin = *std::min_element(vals.begin(), vals.end());
  auto expected_vmax = *std::max_element(vals.begin(), vals.end());
  EXPECT_EQ(vmin, expected_vmin);
  EXPECT_EQ(vmax, expected_vmax);
}

TEST(test_compute_sum, ExpectedOutput) {
  auto vals = torchao::get_random_lowbit_vector(/*size=*/19, /*int8*/ 3);
  int sum = torchao::kernels::cpu::aarch64::reduction::compute_sum(
      (int8_t*)vals.data(), vals.size());
  int expected_sum = std::accumulate(vals.begin(), vals.end(), 0);
  EXPECT_EQ(sum, expected_sum);
}

TEST(test_compute_sum, SizeHasRemainderAfterDivideBy16) {
  auto vals = torchao::get_random_lowbit_vector(/*size=*/17, /*int8*/ 3);
  int sum = torchao::kernels::cpu::aarch64::reduction::compute_sum(
      (int8_t*)vals.data(), vals.size());
  int expected_sum = std::accumulate(vals.begin(), vals.end(), 0);
  EXPECT_EQ(sum, expected_sum);
}

TEST(test_compute_sum, SizeSmallerThan16) {
  auto vals = torchao::get_random_lowbit_vector(/*size=*/3, /*int8*/ 3);
  int sum = torchao::kernels::cpu::aarch64::reduction::compute_sum(
      (int8_t*)vals.data(), vals.size());
  int expected_sum = std::accumulate(vals.begin(), vals.end(), 0);
  EXPECT_EQ(sum, expected_sum);
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
