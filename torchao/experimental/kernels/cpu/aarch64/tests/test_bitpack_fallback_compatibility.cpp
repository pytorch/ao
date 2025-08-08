// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
#if defined(__aarch64__) || defined(__ARM_NEON)

#include <gtest/gtest.h>
#include <vector>

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/kernels/cpu/fallback/bitpacking/bitpack.h>

// --- Compatibility Tests for uint1 ---

TEST(test_bitpacking_64_uint1_values, CppToNeon) {
  int unpacked_bytes = 64;
  int nbit = 1;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_64_uint1_values(
      packed.data(), input.data());

  uint8x16_t u0, u1, u2, u3;
  torchao::bitpacking::internal::vec_unpack_64_uint1_values(
      u0, u1, u2, u3, packed.data());
  vst1q_u8(unpacked.data(), u0);
  vst1q_u8(unpacked.data() + 16, u1);
  vst1q_u8(unpacked.data() + 32, u2);
  vst1q_u8(unpacked.data() + 48, u3);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint1_values, NeonToCpp) {
  int unpacked_bytes = 64;
  int nbit = 1;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t i0, i1, i2, i3;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i0, i1, i2, i3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint1_values(
      packed.data(), i0, i1, i2, i3);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_64_uint1_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_128_uint1_values, CppToNeon) {
  int unpacked_bytes = 128;
  int nbit = 1;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_128_uint1_values(
      packed.data(), input.data());

  uint8x16_t u0, u1, u2, u3, u4, u5, u6, u7;
  torchao::bitpacking::internal::vec_unpack_128_uint1_values(
      u0, u1, u2, u3, u4, u5, u6, u7, packed.data());
  vst1q_u8(unpacked.data(), u0);
  vst1q_u8(unpacked.data() + 16, u1);
  vst1q_u8(unpacked.data() + 32, u2);
  vst1q_u8(unpacked.data() + 48, u3);
  vst1q_u8(unpacked.data() + 64, u4);
  vst1q_u8(unpacked.data() + 80, u5);
  vst1q_u8(unpacked.data() + 96, u6);
  vst1q_u8(unpacked.data() + 112, u7);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_128_uint1_values, NeonToCpp) {
  int unpacked_bytes = 128;
  int nbit = 1;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t i0, i1, i2, i3, i4, i5, i6, i7;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i0, i1, i2, i3, input.data());
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i4, i5, i6, i7, input.data() + 64);
  torchao::bitpacking::internal::vec_pack_128_uint1_values(
      packed.data(), i0, i1, i2, i3, i4, i5, i6, i7);

  torchao::kernels::cpu::fallback::bitpacking::internal::
      unpack_128_uint1_values(unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

// --- Compatibility Tests for uint2 ---

TEST(test_bitpacking_32_uint2_values, CppToNeon) {
  int unpacked_bytes = 32;
  int nbit = 2;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_32_uint2_values(
      packed.data(), input.data());

  uint8x8_t u0, u1, u2, u3;
  torchao::bitpacking::internal::vec_unpack_32_uint2_values(
      u0, u1, u2, u3, packed.data());
  vst1_u8(unpacked.data(), u0);
  vst1_u8(unpacked.data() + 8, u1);
  vst1_u8(unpacked.data() + 16, u2);
  vst1_u8(unpacked.data() + 24, u3);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_32_uint2_values, NeonToCpp) {
  int unpacked_bytes = 32;
  int nbit = 2;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x8_t i0, i1, i2, i3;
  torchao::bitpacking::internal::vec_load_32_uint8_values(
      i0, i1, i2, i3, input.data());
  torchao::bitpacking::internal::vec_pack_32_uint2_values(
      packed.data(), i0, i1, i2, i3);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_32_uint2_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint2_values, CppToNeon) {
  int unpacked_bytes = 64;
  int nbit = 2;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_64_uint2_values(
      packed.data(), input.data());

  uint8x16_t u0, u1, u2, u3;
  torchao::bitpacking::internal::vec_unpack_64_uint2_values(
      u0, u1, u2, u3, packed.data());
  vst1q_u8(unpacked.data(), u0);
  vst1q_u8(unpacked.data() + 16, u1);
  vst1q_u8(unpacked.data() + 32, u2);
  vst1q_u8(unpacked.data() + 48, u3);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint2_values, NeonToCpp) {
  int unpacked_bytes = 64;
  int nbit = 2;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t i0, i1, i2, i3;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i0, i1, i2, i3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint2_values(
      packed.data(), i0, i1, i2, i3);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_64_uint2_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

// --- Compatibility Tests for uint3 ---

TEST(test_bitpacking_64_uint3_values, CppToNeon) {
  int unpacked_bytes = 64;
  int nbit = 3;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_64_uint3_values(
      packed.data(), input.data());

  uint8x16_t u0, u1, u2, u3;
  torchao::bitpacking::internal::vec_unpack_64_uint3_values(
      u0, u1, u2, u3, packed.data());
  vst1q_u8(unpacked.data(), u0);
  vst1q_u8(unpacked.data() + 16, u1);
  vst1q_u8(unpacked.data() + 32, u2);
  vst1q_u8(unpacked.data() + 48, u3);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint3_values, NeonToCpp) {
  int unpacked_bytes = 64;
  int nbit = 3;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t i0, i1, i2, i3;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i0, i1, i2, i3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint3_values(
      packed.data(), i0, i1, i2, i3);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_64_uint3_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_128_uint3_values, CppToNeon) {
  int unpacked_bytes = 128;
  int nbit = 3;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_128_uint3_values(
      packed.data(), input.data());

  uint8x16_t u0, u1, u2, u3, u4, u5, u6, u7;
  torchao::bitpacking::internal::vec_unpack_128_uint3_values(
      u0, u1, u2, u3, u4, u5, u6, u7, packed.data());
  vst1q_u8(unpacked.data(), u0);
  vst1q_u8(unpacked.data() + 16, u1);
  vst1q_u8(unpacked.data() + 32, u2);
  vst1q_u8(unpacked.data() + 48, u3);
  vst1q_u8(unpacked.data() + 64, u4);
  vst1q_u8(unpacked.data() + 80, u5);
  vst1q_u8(unpacked.data() + 96, u6);
  vst1q_u8(unpacked.data() + 112, u7);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_128_uint3_values, NeonToCpp) {
  int unpacked_bytes = 128;
  int nbit = 3;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t i0, i1, i2, i3, i4, i5, i6, i7;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i0, i1, i2, i3, input.data());
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i4, i5, i6, i7, input.data() + 64);
  torchao::bitpacking::internal::vec_pack_128_uint3_values(
      packed.data(), i0, i1, i2, i3, i4, i5, i6, i7);

  torchao::kernels::cpu::fallback::bitpacking::internal::
      unpack_128_uint3_values(unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

// --- Compatibility Tests for uint4 ---

TEST(test_bitpacking_16_uint4_values, CppToNeon) {
  int unpacked_bytes = 16;
  int nbit = 4;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_16_uint4_values(
      packed.data(), input.data());

  uint8x16_t unpacked0;
  torchao::bitpacking::internal::vec_unpack_16_uint4_values(
      unpacked0, packed.data());
  vst1q_u8(unpacked.data(), unpacked0);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_16_uint4_values, NeonToCpp) {
  int unpacked_bytes = 16;
  int nbit = 4;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t input0 = vld1q_u8(input.data());
  torchao::bitpacking::internal::vec_pack_16_uint4_values(
      packed.data(), input0);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_16_uint4_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_32_uint4_values, CppToNeon) {
  int unpacked_bytes = 32;
  int nbit = 4;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_32_uint4_values(
      packed.data(), input.data());

  uint8x16_t unpacked0, unpacked1;
  torchao::bitpacking::internal::vec_unpack_32_uint4_values(
      unpacked0, unpacked1, packed.data());
  vst1q_u8(unpacked.data(), unpacked0);
  vst1q_u8(unpacked.data() + 16, unpacked1);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_32_uint4_values, NeonToCpp) {
  int unpacked_bytes = 32;
  int nbit = 4;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t input0 = vld1q_u8(input.data());
  uint8x16_t input1 = vld1q_u8(input.data() + 16);
  torchao::bitpacking::internal::vec_pack_32_uint4_values(
      packed.data(), input0, input1);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_32_uint4_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

// --- Compatibility Tests for uint5 ---

TEST(test_bitpacking_64_uint5_values, CppToNeon) {
  int unpacked_bytes = 64;
  int nbit = 5;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_64_uint5_values(
      packed.data(), input.data());

  uint8x16_t unpacked0, unpacked1, unpacked2, unpacked3;
  torchao::bitpacking::internal::vec_unpack_64_uint5_values(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());
  vst1q_u8(unpacked.data(), unpacked0);
  vst1q_u8(unpacked.data() + 16, unpacked1);
  vst1q_u8(unpacked.data() + 32, unpacked2);
  vst1q_u8(unpacked.data() + 48, unpacked3);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint5_values, NeonToCpp) {
  int unpacked_bytes = 64;
  int nbit = 5;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t input0, input1, input2, input3;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint5_values(
      packed.data(), input0, input1, input2, input3);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_64_uint5_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_128_uint5_values, CppToNeon) {
  int unpacked_bytes = 128;
  int nbit = 5;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_128_uint5_values(
      packed.data(), input.data());

  uint8x16_t u0, u1, u2, u3, u4, u5, u6, u7;
  torchao::bitpacking::internal::vec_unpack_128_uint5_values(
      u0, u1, u2, u3, u4, u5, u6, u7, packed.data());
  vst1q_u8(unpacked.data(), u0);
  vst1q_u8(unpacked.data() + 16, u1);
  vst1q_u8(unpacked.data() + 32, u2);
  vst1q_u8(unpacked.data() + 48, u3);
  vst1q_u8(unpacked.data() + 64, u4);
  vst1q_u8(unpacked.data() + 80, u5);
  vst1q_u8(unpacked.data() + 96, u6);
  vst1q_u8(unpacked.data() + 112, u7);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_128_uint5_values, NeonToCpp) {
  int unpacked_bytes = 128;
  int nbit = 5;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t i0, i1, i2, i3, i4, i5, i6, i7;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i0, i1, i2, i3, input.data());
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i4, i5, i6, i7, input.data() + 64);
  torchao::bitpacking::internal::vec_pack_128_uint5_values(
      packed.data(), i0, i1, i2, i3, i4, i5, i6, i7);

  torchao::kernels::cpu::fallback::bitpacking::internal::
      unpack_128_uint5_values(unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

// --- Compatibility Tests for uint6 ---

TEST(test_bitpacking_32_uint6_values, CppToNeon) {
  int unpacked_bytes = 32;
  int nbit = 6;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_32_uint6_values(
      packed.data(), input.data());

  uint8x16_t u0, u1;
  torchao::bitpacking::internal::vec_unpack_32_uint6_values(
      u0, u1, packed.data());
  vst1q_u8(unpacked.data(), u0);
  vst1q_u8(unpacked.data() + 16, u1);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_32_uint6_values, NeonToCpp) {
  int unpacked_bytes = 32;
  int nbit = 6;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t i0 = vld1q_u8(input.data());
  uint8x16_t i1 = vld1q_u8(input.data() + 16);
  torchao::bitpacking::internal::vec_pack_32_uint6_values(
      packed.data(), i0, i1);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_32_uint6_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint6_values, CppToNeon) {
  int unpacked_bytes = 64;
  int nbit = 6;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_64_uint6_values(
      packed.data(), input.data());

  uint8x16_t u0, u1, u2, u3;
  torchao::bitpacking::internal::vec_unpack_64_uint6_values(
      u0, u1, u2, u3, packed.data());
  vst1q_u8(unpacked.data(), u0);
  vst1q_u8(unpacked.data() + 16, u1);
  vst1q_u8(unpacked.data() + 32, u2);
  vst1q_u8(unpacked.data() + 48, u3);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint6_values, NeonToCpp) {
  int unpacked_bytes = 64;
  int nbit = 6;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t i0, i1, i2, i3;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i0, i1, i2, i3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint6_values(
      packed.data(), i0, i1, i2, i3);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_64_uint6_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

// --- Compatibility Tests for uint7 ---

TEST(test_bitpacking_64_uint7_values, CppToNeon) {
  int unpacked_bytes = 64;
  int nbit = 7;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_64_uint7_values(
      packed.data(), input.data());

  uint8x16_t unpacked0, unpacked1, unpacked2, unpacked3;
  torchao::bitpacking::internal::vec_unpack_64_uint7_values(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());
  vst1q_u8(unpacked.data(), unpacked0);
  vst1q_u8(unpacked.data() + 16, unpacked1);
  vst1q_u8(unpacked.data() + 32, unpacked2);
  vst1q_u8(unpacked.data() + 48, unpacked3);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint7_values, NeonToCpp) {
  int unpacked_bytes = 64;
  int nbit = 7;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t input0, input1, input2, input3;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint7_values(
      packed.data(), input0, input1, input2, input3);

  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_64_uint7_values(
      unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_128_uint7_values, CppToNeon) {
  int unpacked_bytes = 128;
  int nbit = 7;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::kernels::cpu::fallback::bitpacking::internal::pack_128_uint7_values(
      packed.data(), input.data());

  uint8x16_t u0, u1, u2, u3, u4, u5, u6, u7;
  torchao::bitpacking::internal::vec_unpack_128_uint7_values(
      u0, u1, u2, u3, u4, u5, u6, u7, packed.data());
  vst1q_u8(unpacked.data(), u0);
  vst1q_u8(unpacked.data() + 16, u1);
  vst1q_u8(unpacked.data() + 32, u2);
  vst1q_u8(unpacked.data() + 48, u3);
  vst1q_u8(unpacked.data() + 64, u4);
  vst1q_u8(unpacked.data() + 80, u5);
  vst1q_u8(unpacked.data() + 96, u6);
  vst1q_u8(unpacked.data() + 112, u7);

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_128_uint7_values, NeonToCpp) {
  int unpacked_bytes = 128;
  int nbit = 7;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  uint8x16_t i0, i1, i2, i3, i4, i5, i6, i7;
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i0, i1, i2, i3, input.data());
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      i4, i5, i6, i7, input.data() + 64);
  torchao::bitpacking::internal::vec_pack_128_uint7_values(
      packed.data(), i0, i1, i2, i3, i4, i5, i6, i7);

  torchao::kernels::cpu::fallback::bitpacking::internal::
      unpack_128_uint7_values(unpacked.data(), packed.data());

  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
