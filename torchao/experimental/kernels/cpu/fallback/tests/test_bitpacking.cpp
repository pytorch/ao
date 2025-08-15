// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
// test pack with cpp unpack with arm_neon
#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/kernels/cpu/fallback/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/fallback/bitpacking/uint1.h>
#include <torchao/experimental/kernels/cpu/fallback/bitpacking/uint2.h>
#include <torchao/experimental/kernels/cpu/fallback/bitpacking/uint3.h>
#include <torchao/experimental/kernels/cpu/fallback/bitpacking/uint4.h>
#include <torchao/experimental/kernels/cpu/fallback/bitpacking/uint5.h>
#include <torchao/experimental/kernels/cpu/fallback/bitpacking/uint6.h>
#include <torchao/experimental/kernels/cpu/fallback/bitpacking/uint7.h>
#include <vector>

TEST(FallbackBitpackingTest, PackUnpack8_uint1) {
  int unpacked_bytes = 8;
  int packed_bytes = 1;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 1);
  std::vector<uint8_t> packed(packed_bytes);
  std::vector<uint8_t> unpacked(unpacked_bytes);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_8_uint1_values(
      packed.data(), input.data());
  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_8_uint1_values(
      unpacked.data(), packed.data());

  ASSERT_EQ(input, unpacked);
}

TEST(FallbackBitpackingTest, PackUnpack4_uint2) {
  int unpacked_bytes = 4;
  int packed_bytes = 1;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 2);
  std::vector<uint8_t> packed(packed_bytes);
  std::vector<uint8_t> unpacked(unpacked_bytes);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_4_uint2_values(
      packed.data(), input.data());
  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_4_uint2_values(
      unpacked.data(), packed.data());

  ASSERT_EQ(input, unpacked);
}

TEST(FallbackBitpackingTest, PackUnpack8_uint3) {
  int unpacked_bytes = 8;
  int packed_bytes = 3;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 3);
  std::vector<uint8_t> packed(packed_bytes);
  std::vector<uint8_t> unpacked(unpacked_bytes);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_8_uint3_values(
      packed.data(), input.data());
  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_8_uint3_values(
      unpacked.data(), packed.data());

  ASSERT_EQ(input, unpacked);
}

TEST(FallbackBitpackingTest, PackUnpack32_uint4) {
  int unpacked_bytes = 32;
  int packed_bytes = 16;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 4);
  std::vector<uint8_t> packed(packed_bytes);
  std::vector<uint8_t> unpacked(unpacked_bytes);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_32_uint4_values(
      packed.data(), input.data());
  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_32_uint4_values(
      unpacked.data(), packed.data());

  ASSERT_EQ(input, unpacked);
}

TEST(FallbackBitpackingTest, PackUnpack8_uint5) {
  int unpacked_bytes = 8;
  int packed_bytes = 5;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 5);
  std::vector<uint8_t> packed(packed_bytes);
  std::vector<uint8_t> unpacked(unpacked_bytes);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_8_uint5_values(
      packed.data(), input.data());
  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_8_uint5_values(
      unpacked.data(), packed.data());

  ASSERT_EQ(input, unpacked);
}

TEST(FallbackBitpackingTest, PackUnpack4_uint6) {
  int unpacked_bytes = 4;
  int packed_bytes = 3;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 6);
  std::vector<uint8_t> packed(packed_bytes);
  std::vector<uint8_t> unpacked(unpacked_bytes);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_4_uint6_values(
      packed.data(), input.data());
  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_4_uint6_values(
      unpacked.data(), packed.data());

  ASSERT_EQ(input, unpacked);
}

TEST(FallbackBitpackingTest, PackUnpack8_uint7) {
  int unpacked_bytes = 8;
  int packed_bytes = 7;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 7);
  std::vector<uint8_t> packed(packed_bytes);
  std::vector<uint8_t> unpacked(unpacked_bytes);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_8_uint7_values(
      packed.data(), input.data());
  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_8_uint7_values(
      unpacked.data(), packed.data());

  ASSERT_EQ(input, unpacked);
}

// --- Template test for the main dispatcher function ---
template <int nbit>
void test_bitpacking_128_lowbit_values() {
  const int unpacked_bytes = 128;
  const int packed_bytes = unpacked_bytes * nbit / 8;

  auto input = torchao::get_random_signed_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes);
  std::vector<int8_t> unpacked(unpacked_bytes);

  torchao::kernels::cpu::fallback::bitpacking::internal::
      pack_128_lowbit_int_values<nbit>(packed.data(), input.data());
  torchao::kernels::cpu::fallback::bitpacking::internal::
      unpack_128_lowbit_int_values<nbit>(unpacked.data(), packed.data());

  ASSERT_EQ(input, unpacked);
}

// --- Template test for the LUT dispatcher function ---
template <int nbit>
void test_bitpacking_128_lowbit_values_with_lut() {
  const int unpacked_bytes = 128;
  const int packed_bytes = unpacked_bytes * nbit / 8;
  const int num_lut_entries = 1 << nbit;

  // 1. Create a LUT and random indices
  auto lut = torchao::get_random_signed_lowbit_vector(num_lut_entries, 8);
  auto indices = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);

  // 2. Create the ground truth data by applying the LUT
  std::vector<int8_t> ground_truth(unpacked_bytes);
  for (int i = 0; i < unpacked_bytes; ++i) {
    ground_truth[i] = lut[indices[i]];
  }

  // 3. Pack the indices
  std::vector<uint8_t> packed(packed_bytes);
  if constexpr (nbit == 1)
    torchao::kernels::cpu::fallback::bitpacking::internal::
        pack_128_uint1_values(packed.data(), indices.data());
  if constexpr (nbit == 2) {
    torchao::kernels::cpu::fallback::bitpacking::internal::pack_64_uint2_values(
        packed.data(), indices.data());
    torchao::kernels::cpu::fallback::bitpacking::internal::pack_64_uint2_values(
        packed.data() + 16, indices.data() + 64);
  }
  if constexpr (nbit == 3)
    torchao::kernels::cpu::fallback::bitpacking::internal::
        pack_128_uint3_values(packed.data(), indices.data());
  if constexpr (nbit == 4) {
    torchao::kernels::cpu::fallback::bitpacking::internal::pack_32_uint4_values(
        packed.data(), indices.data());
    torchao::kernels::cpu::fallback::bitpacking::internal::pack_32_uint4_values(
        packed.data() + 16, indices.data() + 32);
    torchao::kernels::cpu::fallback::bitpacking::internal::pack_32_uint4_values(
        packed.data() + 32, indices.data() + 64);
    torchao::kernels::cpu::fallback::bitpacking::internal::pack_32_uint4_values(
        packed.data() + 48, indices.data() + 96);
  }

  // 4. Unpack using the LUT function
  std::vector<int8_t> unpacked(unpacked_bytes);
  torchao::kernels::cpu::fallback::bitpacking::internal::
      unpack_128_lowbit_values_with_lut<nbit>(
          unpacked.data(), packed.data(), lut.data());

  // 5. Verify the result matches the ground truth
  ASSERT_EQ(ground_truth, unpacked);
}

// --- Instantiate all test cases using macros ---
#define TEST_BITPACKING_128_LOWBIT_VALUES(nbit) \
  TEST(GenericBitpacking128, Lowbit_##nbit) {   \
    test_bitpacking_128_lowbit_values<nbit>();  \
  }

#define TEST_BITPACKING_128_LOWBIT_VALUES_WITH_LUT(nbit) \
  TEST(GenericBitpacking128, Lowbit_with_lut_##nbit) {   \
    test_bitpacking_128_lowbit_values_with_lut<nbit>();  \
  }

TEST_BITPACKING_128_LOWBIT_VALUES(1);
TEST_BITPACKING_128_LOWBIT_VALUES(2);
TEST_BITPACKING_128_LOWBIT_VALUES(3);
TEST_BITPACKING_128_LOWBIT_VALUES(4);
TEST_BITPACKING_128_LOWBIT_VALUES(5);
TEST_BITPACKING_128_LOWBIT_VALUES(6);
TEST_BITPACKING_128_LOWBIT_VALUES(7);
TEST_BITPACKING_128_LOWBIT_VALUES(8);

TEST_BITPACKING_128_LOWBIT_VALUES_WITH_LUT(1);
TEST_BITPACKING_128_LOWBIT_VALUES_WITH_LUT(2);
TEST_BITPACKING_128_LOWBIT_VALUES_WITH_LUT(3);
TEST_BITPACKING_128_LOWBIT_VALUES_WITH_LUT(4);
