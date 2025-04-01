// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint1.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint2.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint3.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint4.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint5.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint6.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint7.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <vector>

TEST(test_bitpacking_8_uint1_values, PackUnpackAreSame) {
  int unpacked_bytes = 8;
  int packed_bytes = 1;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 1);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::bitpacking::internal::pack_8_uint1_values(
      packed.data(), input.data());
  torchao::bitpacking::internal::unpack_8_uint1_values(
      unpacked.data(), packed.data());
  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint1_values, PackUnpackAreSame) {
  int unpacked_bytes = 64;
  int packed_bytes = 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 1);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());

  torchao::bitpacking::internal::vec_pack_64_uint1_values(
      packed.data(), input0, input1, input2, input3);

  torchao::bitpacking::internal::vec_unpack_64_uint1_values(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
  }
}

TEST(test_bitpacking_128_uint1_values, PackUnpackAreSame) {
  int unpacked_bytes = 128;
  int packed_bytes = 16;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 1);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;
  uint8x16_t input4;
  uint8x16_t input5;
  uint8x16_t input6;
  uint8x16_t input7;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;
  uint8x16_t unpacked4;
  uint8x16_t unpacked5;
  uint8x16_t unpacked6;
  uint8x16_t unpacked7;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input4, input5, input6, input7, input.data() + 64);
  torchao::bitpacking::internal::vec_pack_128_uint1_values(
      packed.data(),
      input0,
      input1,
      input2,
      input3,
      input4,
      input5,
      input6,
      input7);
  torchao::bitpacking::internal::vec_unpack_128_uint1_values(
      unpacked0,
      unpacked1,
      unpacked2,
      unpacked3,
      unpacked4,
      unpacked5,
      unpacked6,
      unpacked7,
      packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
    EXPECT_EQ(input4[i], unpacked4[i]);
    EXPECT_EQ(input5[i], unpacked5[i]);
    EXPECT_EQ(input6[i], unpacked6[i]);
    EXPECT_EQ(input7[i], unpacked7[i]);
  }
}

TEST(test_bitpacking_4_uint2_values, PackUnpackAreSame) {
  int unpacked_bytes = 4;
  int packed_bytes = 1;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 2);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::bitpacking::internal::pack_4_uint2_values(
      packed.data(), input.data());
  torchao::bitpacking::internal::unpack_4_uint2_values(
      unpacked.data(), packed.data());
  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_32_uint2_values, PackUnpackAreSame) {
  int unpacked_bytes = 32;
  int packed_bytes = 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 2);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x8_t input0;
  uint8x8_t input1;
  uint8x8_t input2;
  uint8x8_t input3;

  uint8x8_t unpacked0;
  uint8x8_t unpacked1;
  uint8x8_t unpacked2;
  uint8x8_t unpacked3;

  torchao::bitpacking::internal::vec_load_32_uint8_values(
      input0, input1, input2, input3, input.data());

  torchao::bitpacking::internal::vec_pack_32_uint2_values(
      packed.data(), input0, input1, input2, input3);
  torchao::bitpacking::internal::vec_unpack_32_uint2_values(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());

  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
  }
}

TEST(test_bitpacking_64_uint2_values, PackUnpackAreSame) {
  int unpacked_bytes = 64;
  int packed_bytes = 16;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 2);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());

  torchao::bitpacking::internal::vec_pack_64_uint2_values(
      packed.data(), input0, input1, input2, input3);
  torchao::bitpacking::internal::vec_unpack_64_uint2_values(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
  }
}

TEST(test_bitpacking_8_uint3_values, PackUnpackAreSame) {
  int unpacked_bytes = 8;
  int packed_bytes = 3;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 3);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::bitpacking::internal::pack_8_uint3_values(
      packed.data(), input.data());
  torchao::bitpacking::internal::unpack_8_uint3_values(
      unpacked.data(), packed.data());
  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint3_values, PackUnpackAreSame) {
  int unpacked_bytes = 64;
  int packed_bytes = 24;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 3);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint3_values(
      packed.data(), input0, input1, input2, input3);
  torchao::bitpacking::internal::vec_unpack_64_uint3_values(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
  }
}

TEST(test_bitpacking_128_uint3_values, PackUnpackAreSame) {
  int unpacked_bytes = 128;
  int packed_bytes = 48;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 3);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;
  uint8x16_t input4;
  uint8x16_t input5;
  uint8x16_t input6;
  uint8x16_t input7;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;
  uint8x16_t unpacked4;
  uint8x16_t unpacked5;
  uint8x16_t unpacked6;
  uint8x16_t unpacked7;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input4, input5, input6, input7, input.data() + 64);
  torchao::bitpacking::internal::vec_pack_128_uint3_values(
      packed.data(),
      input0,
      input1,
      input2,
      input3,
      input4,
      input5,
      input6,
      input7);
  torchao::bitpacking::internal::vec_unpack_128_uint3_values(
      unpacked0,
      unpacked1,
      unpacked2,
      unpacked3,
      unpacked4,
      unpacked5,
      unpacked6,
      unpacked7,
      packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
    EXPECT_EQ(input4[i], unpacked4[i]);
    EXPECT_EQ(input5[i], unpacked5[i]);
    EXPECT_EQ(input6[i], unpacked6[i]);
    EXPECT_EQ(input7[i], unpacked7[i]);
  }
}

TEST(test_bitpacking_2_uint4_values, PackUnpackAreSame) {
  int unpacked_bytes = 2;
  int nbit = 4;
  int packed_bytes = unpacked_bytes * nbit / 8;

  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::bitpacking::internal::pack_2_uint4_values(
      packed.data(), input.data());
  torchao::bitpacking::internal::unpack_2_uint4_values(
      unpacked.data(), packed.data());
  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_16_uint4_values, PackUnpackAreSame) {
  int unpacked_bytes = 16;
  int nbit = 4;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t unpacked0;

  input0 = vld1q_u8(input.data());
  torchao::bitpacking::internal::vec_pack_16_uint4_values(
      packed.data(), input0);
  torchao::bitpacking::internal::vec_unpack_16_uint4_values(
      unpacked0, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
  }
}

TEST(test_bitpacking_32_uint4_values, PackUnpackAreSame) {
  int unpacked_bytes = 32;
  int nbit = 2;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t unpacked0;
  uint8x16_t unpacked1;

  input0 = vld1q_u8(input.data());
  input1 = vld1q_u8(input.data() + 16);
  torchao::bitpacking::internal::vec_pack_32_uint4_values(
      packed.data(), input0, input1);
  torchao::bitpacking::internal::vec_unpack_32_uint4_values(
      unpacked0, unpacked1, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
  }
}

TEST(test_bitpacking_8_uint5_values, PackUnpackAreSame) {
  int unpacked_bytes = 8;
  int packed_bytes = 5;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 5);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::bitpacking::internal::pack_8_uint5_values(
      packed.data(), input.data());
  torchao::bitpacking::internal::unpack_8_uint5_values(
      unpacked.data(), packed.data());
  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint5_values, PackUnpackAreSame) {
  int unpacked_bytes = 64;
  int packed_bytes = 40;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 5);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint5_values(
      packed.data(), input0, input1, input2, input3);
  torchao::bitpacking::internal::vec_unpack_64_uint5_values(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
  }
}

TEST(test_bitpacking_128_uint5_values, PackUnpackAreSame) {
  int unpacked_bytes = 128;
  int packed_bytes = 80;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 5);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;
  uint8x16_t input4;
  uint8x16_t input5;
  uint8x16_t input6;
  uint8x16_t input7;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;
  uint8x16_t unpacked4;
  uint8x16_t unpacked5;
  uint8x16_t unpacked6;
  uint8x16_t unpacked7;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input4, input5, input6, input7, input.data() + 64);
  torchao::bitpacking::internal::vec_pack_128_uint5_values(
      packed.data(),
      input0,
      input1,
      input2,
      input3,
      input4,
      input5,
      input6,
      input7);
  torchao::bitpacking::internal::vec_unpack_128_uint5_values(
      unpacked0,
      unpacked1,
      unpacked2,
      unpacked3,
      unpacked4,
      unpacked5,
      unpacked6,
      unpacked7,
      packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
    EXPECT_EQ(input4[i], unpacked4[i]);
    EXPECT_EQ(input5[i], unpacked5[i]);
    EXPECT_EQ(input6[i], unpacked6[i]);
    EXPECT_EQ(input7[i], unpacked7[i]);
  }
}

TEST(test_bitpacking_4_uint6_values, PackUnpackAreSame) {
  int unpacked_bytes = 4;
  int packed_bytes = 3;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 6);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::bitpacking::internal::pack_4_uint6_values(
      packed.data(), input.data());
  torchao::bitpacking::internal::unpack_4_uint6_values(
      unpacked.data(), packed.data());
  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_32_uint6_values, PackUnpackAreSame) {
  int unpacked_bytes = 32;
  int packed_bytes = 24;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 6);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;

  input0 = vld1q_u8(input.data());
  input1 = vld1q_u8(input.data() + 16);
  torchao::bitpacking::internal::vec_pack_32_uint6_values(
      packed.data(), input0, input1);
  torchao::bitpacking::internal::vec_unpack_32_uint6_values(
      unpacked0, unpacked1, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
  }
}

TEST(test_bitpacking_64_uint6_values, PackUnpackAreSame) {
  int unpacked_bytes = 64;
  int packed_bytes = 48;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 6);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint6_values(
      packed.data(), input0, input1, input2, input3);
  torchao::bitpacking::internal::vec_unpack_64_uint6_values(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
  }
}

TEST(test_bitpacking_8_uint7_values, PackUnpackAreSame) {
  int unpacked_bytes = 8;
  int packed_bytes = 7;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 7);
  std::vector<uint8_t> packed(packed_bytes, 0);
  std::vector<uint8_t> unpacked(unpacked_bytes, 0);

  torchao::bitpacking::internal::pack_8_uint7_values(
      packed.data(), input.data());
  torchao::bitpacking::internal::unpack_8_uint7_values(
      unpacked.data(), packed.data());
  for (int i = 0; i < unpacked_bytes; ++i) {
    EXPECT_EQ(input[i], unpacked[i]);
  }
}

TEST(test_bitpacking_64_uint7_values, PackUnpackAreSame) {
  int unpacked_bytes = 64;
  int packed_bytes = 56;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 7);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_pack_64_uint7_values(
      packed.data(), input0, input1, input2, input3);
  torchao::bitpacking::internal::vec_unpack_64_uint7_values(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
  }
}

TEST(test_bitpacking_128_uint7_values, PackUnpackAreSame) {
  int unpacked_bytes = 128;
  int packed_bytes = 112;
  auto input = torchao::get_random_lowbit_vector(unpacked_bytes, 7);
  std::vector<uint8_t> packed(packed_bytes, 0);

  uint8x16_t input0;
  uint8x16_t input1;
  uint8x16_t input2;
  uint8x16_t input3;
  uint8x16_t input4;
  uint8x16_t input5;
  uint8x16_t input6;
  uint8x16_t input7;

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;
  uint8x16_t unpacked4;
  uint8x16_t unpacked5;
  uint8x16_t unpacked6;
  uint8x16_t unpacked7;

  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input0, input1, input2, input3, input.data());
  torchao::bitpacking::internal::vec_load_64_uint8_values(
      input4, input5, input6, input7, input.data() + 64);
  torchao::bitpacking::internal::vec_pack_128_uint7_values(
      packed.data(),
      input0,
      input1,
      input2,
      input3,
      input4,
      input5,
      input6,
      input7);
  torchao::bitpacking::internal::vec_unpack_128_uint7_values(
      unpacked0,
      unpacked1,
      unpacked2,
      unpacked3,
      unpacked4,
      unpacked5,
      unpacked6,
      unpacked7,
      packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
    EXPECT_EQ(input4[i], unpacked4[i]);
    EXPECT_EQ(input5[i], unpacked5[i]);
    EXPECT_EQ(input6[i], unpacked6[i]);
    EXPECT_EQ(input7[i], unpacked7[i]);
  }
}

// Universal bitpacking tests
template <int nbit>
void test_bitpacking_32_lowbit_values() {
  int unpacked_bytes = 32;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_signed_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);

  int8x16_t input0;
  int8x16_t input1;
  int8x16_t unpacked0;
  int8x16_t unpacked1;
  input0 = vld1q_s8(input.data());
  input1 = vld1q_s8(input.data() + 16);
  torchao::bitpacking::vec_pack_32_lowbit_values<nbit>(
      packed.data(), input0, input1);
  torchao::bitpacking::vec_unpack_32_lowbit_values<nbit>(
      unpacked0, unpacked1, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
  }
}

template <int nbit>
void test_bitpacking_64_lowbit_values() {
  int unpacked_bytes = 64;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_signed_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);

  int8x16_t input0;
  int8x16_t input1;
  int8x16_t input2;
  int8x16_t input3;
  int8x16_t unpacked0;
  int8x16_t unpacked1;
  int8x16_t unpacked2;
  int8x16_t unpacked3;
  input0 = vld1q_s8(input.data());
  input1 = vld1q_s8(input.data() + 16);
  input2 = vld1q_s8(input.data() + 32);
  input3 = vld1q_s8(input.data() + 48);
  torchao::bitpacking::vec_pack_64_lowbit_values<nbit>(
      packed.data(), input0, input1, input2, input3);
  torchao::bitpacking::vec_unpack_64_lowbit_values<nbit>(
      unpacked0, unpacked1, unpacked2, unpacked3, packed.data());

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
  }
}

template <int nbit>
void test_bitpacking_128_lowbit_values() {
  int unpacked_bytes = 128;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_signed_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);

  int8x16_t input0;
  int8x16_t input1;
  int8x16_t input2;
  int8x16_t input3;
  int8x16_t input4;
  int8x16_t input5;
  int8x16_t input6;
  int8x16_t input7;
  int8x16_t unpacked0;
  int8x16_t unpacked1;
  int8x16_t unpacked2;
  int8x16_t unpacked3;
  int8x16_t unpacked4;
  int8x16_t unpacked5;
  int8x16_t unpacked6;
  int8x16_t unpacked7;

  input0 = vld1q_s8(input.data());
  input1 = vld1q_s8(input.data() + 16);
  input2 = vld1q_s8(input.data() + 32);
  input3 = vld1q_s8(input.data() + 48);
  input4 = vld1q_s8(input.data() + 64);
  input5 = vld1q_s8(input.data() + 80);
  input6 = vld1q_s8(input.data() + 96);
  input7 = vld1q_s8(input.data() + 112);
  torchao::bitpacking::vec_pack_128_lowbit_values<nbit>(
      packed.data(),
      input0,
      input1,
      input2,
      input3,
      input4,
      input5,
      input6,
      input7);
  torchao::bitpacking::vec_unpack_128_lowbit_values<nbit>(
      unpacked0,
      unpacked1,
      unpacked2,
      unpacked3,
      unpacked4,
      unpacked5,
      unpacked6,
      unpacked7,
      packed.data());
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
    EXPECT_EQ(input4[i], unpacked4[i]);
    EXPECT_EQ(input5[i], unpacked5[i]);
    EXPECT_EQ(input6[i], unpacked6[i]);
    EXPECT_EQ(input7[i], unpacked7[i]);
  }
}

template <int nbit>
void test_bitpacking_128_lowbit_values_with_lut() {
  int unpacked_bytes = 128;
  int packed_bytes = unpacked_bytes * nbit / 8;
  auto input = torchao::get_random_signed_lowbit_vector(unpacked_bytes, nbit);
  std::vector<uint8_t> packed(packed_bytes, 0);

  int8x16_t input0;
  int8x16_t input1;
  int8x16_t input2;
  int8x16_t input3;
  int8x16_t input4;
  int8x16_t input5;
  int8x16_t input6;
  int8x16_t input7;
  int8x16_t unpacked0;
  int8x16_t unpacked1;
  int8x16_t unpacked2;
  int8x16_t unpacked3;
  int8x16_t unpacked4;
  int8x16_t unpacked5;
  int8x16_t unpacked6;
  int8x16_t unpacked7;

  input0 = vld1q_s8(input.data());
  input1 = vld1q_s8(input.data() + 16);
  input2 = vld1q_s8(input.data() + 32);
  input3 = vld1q_s8(input.data() + 48);
  input4 = vld1q_s8(input.data() + 64);
  input5 = vld1q_s8(input.data() + 80);
  input6 = vld1q_s8(input.data() + 96);
  input7 = vld1q_s8(input.data() + 112);

  int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
  uint8x16_t idx0 = vreinterpretq_u8_s8(vaddq_s8(input0, shift));
  uint8x16_t idx1 = vreinterpretq_u8_s8(vaddq_s8(input1, shift));
  uint8x16_t idx2 = vreinterpretq_u8_s8(vaddq_s8(input2, shift));
  uint8x16_t idx3 = vreinterpretq_u8_s8(vaddq_s8(input3, shift));
  uint8x16_t idx4 = vreinterpretq_u8_s8(vaddq_s8(input4, shift));
  uint8x16_t idx5 = vreinterpretq_u8_s8(vaddq_s8(input5, shift));
  uint8x16_t idx6 = vreinterpretq_u8_s8(vaddq_s8(input6, shift));
  uint8x16_t idx7 = vreinterpretq_u8_s8(vaddq_s8(input7, shift));

  int8x16_t lut;
  if constexpr (nbit == 1) {
    lut = {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  } else if constexpr (nbit == 2) {
    lut = {-2, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  } else if constexpr (nbit == 3) {
    lut = {-4, -3, -2, -1, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0};
  } else {
    lut = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
  }

  torchao::bitpacking::vec_pack_128_uintx_values<nbit>(
      packed.data(), idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7);
  torchao::bitpacking::vec_unpack_128_lowbit_values_with_lut<nbit>(
      unpacked0,
      unpacked1,
      unpacked2,
      unpacked3,
      unpacked4,
      unpacked5,
      unpacked6,
      unpacked7,
      packed.data(),
      lut);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(input0[i], unpacked0[i]);
    EXPECT_EQ(input1[i], unpacked1[i]);
    EXPECT_EQ(input2[i], unpacked2[i]);
    EXPECT_EQ(input3[i], unpacked3[i]);
    EXPECT_EQ(input4[i], unpacked4[i]);
    EXPECT_EQ(input5[i], unpacked5[i]);
    EXPECT_EQ(input6[i], unpacked6[i]);
    EXPECT_EQ(input7[i], unpacked7[i]);
  }
}

#define TEST_BITPACKING_32_LOWBIT_VALUES(nbit)                       \
  TEST(test_bitpacking_32_lowbit_values_##nbit, PackUnpackAreSame) { \
    test_bitpacking_32_lowbit_values<nbit>();                        \
  }

#define TEST_BITPACKING_64_LOWBIT_VALUES(nbit)                       \
  TEST(test_bitpacking_64_lowbit_values_##nbit, PackUnpackAreSame) { \
    test_bitpacking_64_lowbit_values<nbit>();                        \
  }

#define TEST_BITPACKING_128_LOWBIT_VALUES(nbit)                       \
  TEST(test_bitpacking_128_lowbit_values_##nbit, PackUnpackAreSame) { \
    test_bitpacking_128_lowbit_values<nbit>();                        \
  }

#define TEST_BITPACKING_128_LOWBIT_VALUES_WITH_LUT(nbit)                       \
  TEST(test_bitpacking_128_lowbit_values_with_lut_##nbit, PackUnpackAreSame) { \
    test_bitpacking_128_lowbit_values_with_lut<nbit>();                        \
  }

TEST_BITPACKING_32_LOWBIT_VALUES(1);
TEST_BITPACKING_32_LOWBIT_VALUES(2);
TEST_BITPACKING_32_LOWBIT_VALUES(3);
TEST_BITPACKING_32_LOWBIT_VALUES(4);
TEST_BITPACKING_32_LOWBIT_VALUES(5);
TEST_BITPACKING_32_LOWBIT_VALUES(6);
TEST_BITPACKING_32_LOWBIT_VALUES(7);
TEST_BITPACKING_32_LOWBIT_VALUES(8);

TEST_BITPACKING_64_LOWBIT_VALUES(1);
TEST_BITPACKING_64_LOWBIT_VALUES(2);
TEST_BITPACKING_64_LOWBIT_VALUES(3);
TEST_BITPACKING_64_LOWBIT_VALUES(4);
TEST_BITPACKING_64_LOWBIT_VALUES(5);
TEST_BITPACKING_64_LOWBIT_VALUES(6);
TEST_BITPACKING_64_LOWBIT_VALUES(7);
TEST_BITPACKING_64_LOWBIT_VALUES(8);

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

#endif // defined(__aarch64__) || defined(__ARM_NEON)
