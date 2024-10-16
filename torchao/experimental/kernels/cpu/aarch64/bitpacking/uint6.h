// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/macro.h>

// This file contains bitpacking and unpacking methods for uint5.
// These are not inteded to be used outside of bitpacking directory.
// See bitpack.h for the interface.

namespace torchao {
namespace bitpacking {
namespace internal {

TORCHAO_ALWAYS_INLINE inline void pack_4_uint6_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  //    Given 4 unpacked uint6 values: 01abcd, 23efgh, 45ijkl, 67mnop
  //    this function packs them as:
  //    b54: 67|45|23|01 (to hold upper 2 bits on all values)
  //    b3210_0: efgh|abcd (lower 4 bits for first 2 values)
  //    b3210_1: mnop|ijkl (lower 4 bits for last 2 values)

  // These are stored in packed as: b54, b3210_0, b3210_1
  //
  // Input is 4 bytes
  // Output is 6 * 4 bits/8 = 3 bytes

  // b54
  packed[0] = ((unpacked[0] & 48) >> 4) | ((unpacked[1] & 48) >> 2) |
      ((unpacked[2] & 48)) | ((unpacked[3] & 48) << 2);

  // b3210_0
  packed[1] = (unpacked[0] & 15) | ((unpacked[1] & 15) << 4);

  // b3210_1
  packed[2] = (unpacked[2] & 15) | ((unpacked[3] & 15) << 4);
}

TORCHAO_ALWAYS_INLINE inline void unpack_4_uint6_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // Unpacks data packed by pack_4_uint6_values
  //
  // Input is 24 bits = 3 bytes
  // Output is 4 bytes

  uint8_t b54 = packed[0];
  uint8_t b3210_0 = packed[1];
  uint8_t b3210_1 = packed[2];

  unpacked[0] = ((b54 & 3) << 4) | (b3210_0 & 15);
  unpacked[1] = ((b54 & 12) << 2) | (b3210_0 >> 4);

  unpacked[2] = (b54 & 48) | (b3210_1 & 15);
  unpacked[3] = ((b54 & 192) >> 2) | (b3210_1 >> 4);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_32_uint6_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1) {
  // This function is a vectorized version of pack_8_uint6_values
  // To understand it, please see pack_8_uint6_values first.
  // Before each code section, there is a comment indicating the
  // code in pack_8_uint6_values that is being vectorized
  //
  // Input is 32 bytes
  // Output is 6*32= 192 bits = 24 bytes

  uint8x8_t b54;
  uint8x8_t mask;

  // // b54
  // packed[0] = ((unpacked[0] & 48) >> 4) | ((unpacked[1] & 48) >> 2) |
  //     ((unpacked[2] & 48)) | ((unpacked[3] & 48) << 2);
  mask = vdup_n_u8(48);
  b54 = vshr_n_u8(vand_u8(vget_low_u8(unpacked0), mask), 4);
  b54 = vorr_u8(b54, vshr_n_u8(vand_u8(vget_high_u8(unpacked0), mask), 2));

  b54 = vorr_u8(b54, vand_u8(vget_low_u8(unpacked1), mask));
  b54 = vorr_u8(b54, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), mask), 2));

  vst1_u8(packed, b54);

  mask = vdup_n_u8(15);
  uint8x8_t b3210;

  // b3210_0
  // packed[1] = (unpacked[0] & 15) | ((unpacked[1] & 15) << 4);
  b3210 = vand_u8(vget_low_u8(unpacked0), mask);
  b3210 = vorr_u8(b3210, vshl_n_u8(vand_u8(vget_high_u8(unpacked0), mask), 4));
  vst1_u8(packed + 8, b3210);

  // b3210_1
  // packed[2] = (unpacked[2] & 15) | ((unpacked[3] & 15) << 4);
  b3210 = vand_u8(vget_low_u8(unpacked1), mask);
  b3210 = vorr_u8(b3210, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), mask), 4));
  vst1_u8(packed + 16, b3210);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_32_uint6_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    const uint8_t* packed) {
  // Unpacks data packed by pack_32_uint6_values
  //
  // This function vectorizes vec_unpack_4_uint6_values
  // To understand it, please see vec_unpack_4_uint6_values first.
  // Before each code section, there is a comment indicating the
  // code in vec_unpack_4_uint6_values that is being vectorized

  // Input is 24 bytes
  // Output is 32 bytes

  uint8x8_t b54 = vld1_u8(packed);
  uint8x8_t b3210;
  uint8x8_t unpacked_tmp0;
  uint8x8_t unpacked_tmp1;

  // unpacked[0] = ((b54 & 3) << 4) | (b3210_0 & 15);
  // unpacked[1] = ((b54 & 12) << 2) | (b3210_0 >> 4);
  b3210 = vld1_u8(packed + 8);

  unpacked_tmp0 = vshl_n_u8(vand_u8(b54, vdup_n_u8(3)), 4);
  unpacked_tmp0 = vorr_u8(unpacked_tmp0, vand_u8(b3210, vdup_n_u8(15)));

  unpacked_tmp1 = vshl_n_u8(vand_u8(b54, vdup_n_u8(12)), 2);
  unpacked_tmp1 = vorr_u8(unpacked_tmp1, vshr_n_u8(b3210, 4));

  unpacked0 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  // unpacked[2] = (b54 & 48) | (b3210_1 & 15);
  // unpacked[3] = ((b54 & 192) >> 2) | (b3210_1 >> 4);
  b3210 = vld1_u8(packed + 16);

  unpacked_tmp0 = vand_u8(b54, vdup_n_u8(48));
  unpacked_tmp0 = vorr_u8(unpacked_tmp0, vand_u8(b3210, vdup_n_u8(15)));

  unpacked_tmp1 = vshr_n_u8(vand_u8(b54, vdup_n_u8(192)), 2);
  unpacked_tmp1 = vorr_u8(unpacked_tmp1, vshr_n_u8(b3210, 4));

  unpacked1 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint6_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3) {
  // This function is a vectorized version of pack_4_uint6_values
  // To understand it, please see pack_4_uint6_values first.
  // Before each code section, there is a comment indicating the
  // code in pack_4_uint6_values that is being vectorized
  //
  // Input is 64 bytes
  // Output is 6*64= 384 bits = 48 bytes

  uint8x16_t b54;
  uint8x16_t mask;

  // b54
  // packed[0] = ((unpacked[0] & 48) >> 4) | ((unpacked[1] & 48) >> 2) |
  //     ((unpacked[2] & 48)) | ((unpacked[3] & 48) << 2);
  mask = vdupq_n_u8(48);
  b54 = vshrq_n_u8(vandq_u8(unpacked0, mask), 4);
  b54 = vorrq_u8(b54, vshrq_n_u8(vandq_u8(unpacked1, mask), 2));
  b54 = vorrq_u8(b54, vandq_u8(unpacked2, mask));
  b54 = vorrq_u8(b54, vshlq_n_u8(vandq_u8(unpacked3, mask), 2));

  vst1q_u8(packed, b54);

  mask = vdupq_n_u8(15);
  uint8x16_t b3210;

  // b3210_0
  // packed[1] = (unpacked[0] & 15) | ((unpacked[1] & 15) << 4);
  b3210 = vandq_u8(unpacked0, mask);
  b3210 = vorrq_u8(b3210, vshlq_n_u8(vandq_u8(unpacked1, mask), 4));
  vst1q_u8(packed + 16, b3210);

  // b3210_1
  // packed[2] = (unpacked[2] & 15) | ((unpacked[3] & 15) << 4);
  b3210 = vandq_u8(unpacked2, mask);
  b3210 = vorrq_u8(b3210, vshlq_n_u8(vandq_u8(unpacked3, mask), 4));
  vst1q_u8(packed + 32, b3210);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint6_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    const uint8_t* packed) {
  // Unpacks data packed by pack_64_uint6_values
  //
  // This function vectorizes vec_unpack_4_uint6_values
  // To understand it, please see vec_unpack_4_uint6_values first.
  // Before each code section, there is a comment indicating the
  // code in vec_unpack_4_uint6_values that is being vectorized

  // Input is 48 bytes
  // Output is 64 bytes

  uint8x16_t b54 = vld1q_u8(packed);
  uint8x16_t b3210;

  // unpacked[0] = ((b54 & 3) << 4) | (b3210_0 & 15);
  // unpacked[1] = ((b54 & 12) << 2) | (b3210_0 >> 4);
  b3210 = vld1q_u8(packed + 16);

  unpacked0 = vshlq_n_u8(vandq_u8(b54, vdupq_n_u8(3)), 4);
  unpacked0 = vorrq_u8(unpacked0, vandq_u8(b3210, vdupq_n_u8(15)));

  unpacked1 = vshlq_n_u8(vandq_u8(b54, vdupq_n_u8(12)), 2);
  unpacked1 = vorrq_u8(unpacked1, vshrq_n_u8(b3210, 4));

  // unpacked[2] = (b54 & 48) | (b3210_1 & 15);
  // unpacked[3] = ((b54 & 192) >> 2) | (b3210_1 >> 4);
  b3210 = vld1q_u8(packed + 32);

  unpacked2 = vandq_u8(b54, vdupq_n_u8(48));
  unpacked2 = vorrq_u8(unpacked2, vandq_u8(b3210, vdupq_n_u8(15)));

  unpacked3 = vshrq_n_u8(vandq_u8(b54, vdupq_n_u8(192)), 2);
  unpacked3 = vorrq_u8(unpacked3, vshrq_n_u8(b3210, 4));
}

TORCHAO_ALWAYS_INLINE inline void pack_4_uint6_values_v2(
    uint8_t* packed,
    const uint8_t* unpacked) {
  //    Given 4 unpacked uint6 values: abcdef, ghijkl, mnopqr, 123456
  //    this function packs them as:
  //    packed[0]: 56 | abcdef
  //    packed[1]: 34 | ghijkl
  //    packed[2]: 12 | mnopqr
  //
  // Input is 4 bytes
  // Output is 6 * 4 bits/8 = 3 bytes
  packed[0] = unpacked[0];
  packed[1] = unpacked[1];
  packed[2] = unpacked[2];
  // Last value is packed in the upper 2 bits of the three bytes
  packed[0] |= ((unpacked[3] & 0b00'0011u) << 6);
  packed[1] |= ((unpacked[3] & 0b00'1100u) << 4);
  packed[2] |= ((unpacked[3] & 0b11'0000u) << 2);
}

TORCHAO_ALWAYS_INLINE inline void unpack_4_uint6_values_v2(
  uint8_t* unpacked,
  const uint8_t* packed) {
  // Unpacks data packed by pack_4_uint6_values_v2
  //
  // Input is 24 bits = 3 bytes
  // Output is 4 bytes
  unpacked[0] = packed[0] & 0b111111u;
  unpacked[1] = packed[1] & 0b111111u;
  unpacked[2] = packed[2] & 0b111111u;
  // Last value is packed in the upper 2 bits of the three bytes
  unpacked[3] = ((packed[0] & 0b1100'0000u) >> 6) |
                ((packed[1] & 0b1100'0000u) >> 4) |
                ((packed[2] & 0b1100'0000u) >> 2);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_32_uint6_values_v2(
  uint8_t* packed,
  const uint8x16_t& unpacked0,
  const uint8x16_t& unpacked1) {
  // This function is a vectorized version of pack_4_uint6_values_v2.
  // To understand the following code, please see pack_4_uint6_values_v2 first and
  // consider the following mapping for the unpacked parameter of that function:
  //
  // unpacked[0] -> vget_low_u8(unpacked0)
  // unpacked[1] -> vget_high_u8(unpacked0)
  // unpacked[2] -> vget_low_u8(unpacked1)
  // unpacked[3] -> vget_high_u8(unpacked1)
  //
  // Before each code section, there is a comment indicating the
  // code in pack_4_uint6_values_v2 that is being vectorized.
  //
  // Input is 32 bytes.
  // Output is 6*32= 192 bits = 24 bytes.
  uint8x8_t r;

  // packed[0] = unpacked[0]
  // packed[0] |= ((unpacked[3] & 0b00'0011u) << 6)
  r = vget_low_u8(unpacked0);
  r = vorr_u8(r, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), vdup_n_u8(0b00'0011u)), 6));
  vst1_u8(packed, r);

  // packed[1] = unpacked[1]
  // packed[1] |= ((unpacked[3] & 0b00'1100u) << 4)
  r = vget_high_u8(unpacked0);
  r = vorr_u8(r, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), vdup_n_u8(0b00'1100u)), 4));
  vst1_u8(packed + 8, r);

  // packed[2] = unpacked[2]
  // packed[2] |= ((unpacked[3] & 0b11'0000u) << 2)
  r = vget_low_u8(unpacked1);
  r = vorr_u8(r, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), vdup_n_u8(0b11'0000u)), 2));
  vst1_u8(packed + 16, r);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_32_uint6_values_v2(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    const uint8_t* packed) {
  // Unpacks data packed by vec_pack_32_uint6_values_v2.
  //
  // This function vectorizes unpack_4_uint6_values_v2.
  // To understand it, please see unpack_4_uint6_values_v2 first.
  // Before each code section, there is a comment indicating the
  // code in unpack_4_uint6_values_v2 that is being vectorized.
  //
  // Input is 24 bytes.
  // Output is 32 bytes.
  uint8x8_t packed0 = vld1_u8(packed);
  uint8x8_t packed1 = vld1_u8(packed + 8);
  uint8x8_t packed2 = vld1_u8(packed + 16);

  // unpacked[3] = ((packed[0] & 0b1100'0000u) >> 6) |
  //               ((packed[1] & 0b1100'0000u) >> 4) |
  //               ((packed[2] & 0b1100'0000u) >> 2);
  const uint8x8_t high = vdup_n_u8(0b1100'0000u);
  uint8x8_t unpacked3;
  unpacked3 = vorr_u8(vshr_n_u8(vand_u8(packed0, high), 6),
                      vshr_n_u8(vand_u8(packed1, high), 4));
  unpacked3 = vorr_u8(unpacked3,
                      vshr_n_u8(vand_u8(packed2, high), 2));

  // unpacked[i] = packed[i] & 0b11'1111u;
  const uint8x8_t mask = vdup_n_u8(0b11'1111u);
  unpacked0 = vcombine_u8(vand_u8(packed0, mask), vand_u8(packed1, mask));
  unpacked1 = vcombine_u8(vand_u8(packed2, mask), unpacked3);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint6_values_v2(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3) {
  // This function is a vectorized version of pack_4_uint6_values_v2.
  // To understand the following code, please see pack_4_uint6_values_v2 first.
  // Before each code section, there is a comment indicating the
  // code in pack_4_uint6_values_v2 that is being vectorized.
  //
  // Input is 48 bytes.
  // Output is 64 bytes.
  uint8x16_t r;

  // packed[0] = unpacked[0]
  // packed[0] |= ((unpacked[3] & 0b00'0011u) << 6)
  r = unpacked0;
  r = vorrq_u8(r, vshlq_n_u8(vandq_u8(unpacked3, vdupq_n_u8(0b00'0011u)), 6));
  vst1q_u8(packed, r);

  // packed[1] = unpacked[1]
  // packed[1] |= ((unpacked[3] & 0b00'1100u) << 4)
  r = unpacked1;
  r = vorrq_u8(r, vshlq_n_u8(vandq_u8(unpacked3, vdupq_n_u8(0b00'1100u)), 4));
  vst1q_u8(packed + 16, r);

  // packed[2] = unpacked[2]
  // packed[2] |= ((unpacked[3] & 0b11'0000u) << 2)
  r = unpacked2;
  r = vorrq_u8(r, vshlq_n_u8(vandq_u8(unpacked3, vdupq_n_u8(0b11'0000u)), 2));
  vst1q_u8(packed + 32, r);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint6_values_v2(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    const uint8_t* packed) {
  // Unpacks data packed by vec_pack_64_uint6_values_v2.
  //
  // This function vectorizes unpack_4_uint6_values_v2.
  // To understand it, please see unpack_4_uint6_values_v2 first.
  // Before each code section, there is a comment indicating the
  // code in unpack_4_uint6_values that is being vectorized

  // Input is 48 bytes.
  // Output is 64 bytes.
  unpacked0 = vld1q_u8(packed);
  unpacked1 = vld1q_u8(packed + 16);
  unpacked2 = vld1q_u8(packed + 32);

  // unpacked[3] = ((packed[0] & 0b1100'0000u) >> 6) |
  //               ((packed[1] & 0b1100'0000u) >> 4) |
  //               ((packed[2] & 0b1100'0000u) >> 2);
  const uint8x16_t high = vdupq_n_u8(0b1100'0000u);
  unpacked3 = vorrq_u8(vshrq_n_u8(vandq_u8(unpacked0, high), 6),
                       vshrq_n_u8(vandq_u8(unpacked1, high), 4));
  unpacked3 = vorrq_u8(unpacked3,
                       vshrq_n_u8(vandq_u8(unpacked2, high), 2));

  // unpacked[i] = packed[i] & 0b11'1111u;
  const uint8x16_t mask = vdupq_n_u8(0b11'1111u);
  unpacked0 = vandq_u8(unpacked0, mask);
  unpacked1 = vandq_u8(unpacked1, mask);
  unpacked2 = vandq_u8(unpacked2, mask);
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
