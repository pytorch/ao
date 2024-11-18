// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>

// This file contains bitpacking and unpacking methods for uint5.
// These are not inteded to be used outside of bitpacking directory.
// See bitpack.h for the interface.

namespace torchao {
namespace bitpacking {
namespace internal {

TORCHAO_ALWAYS_INLINE inline void pack_8_uint5_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // Given 8 unpacked uint5 values u0, u1, u2, u3, u4, u5, u6, u7,
  // pack them into 5 uint8 values.
  // All 5 bits of u0, u2, ... are stored into the lower parts of p0-p3.
  // The lower 3 bits of u1, u3, ... are stored in the upper parts of p0-p3.
  // The upper 2 bits of u1, u3, ... are stored in p4.

  packed[0] = unpacked[0] | (unpacked[1] << 5);
  packed[1] = unpacked[2] | (unpacked[3] << 5);
  packed[2] = unpacked[4] | (unpacked[5] << 5);
  packed[3] = unpacked[6] | (unpacked[7] << 5);
  packed[4] = (unpacked[1] >> 3) | ((unpacked[3] >> 3) << 2) |
      ((unpacked[5] >> 3) << 4) | ((unpacked[7] >> 3) << 6);
}

TORCHAO_ALWAYS_INLINE inline void unpack_8_uint5_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // Given 5 packed uint8 values p0, p1, p2, p3, p4, packed by
  // pack_8_uint5_values, unpack them into 8 unpacked uint5 values u0, u1,
  // u2, u3, u4, u5, u6, u7.
  // All 5 bits of u0, u2, ... can be restored by extracting the lower 5 bits of
  // p0-p3. The lower 3 bits of u1, u3, ... can be restored by extracting the
  // upper 3 bits of p0-p3. The upper 2 bits of u1, u3, ... can be extracted
  // from p4.

  uint8_t p0 = packed[0];
  uint8_t p1 = packed[1];
  uint8_t p2 = packed[2];
  uint8_t p3 = packed[3];
  uint8_t p4 = packed[4];

  unpacked[0] = p0 & 0b11111;
  unpacked[1] = p0 >> 5 | ((p4 & 0b00000011) << 3);
  unpacked[2] = p1 & 0b11111;
  unpacked[3] = p1 >> 5 | ((p4 & 0b00001100) << 1);
  unpacked[4] = p2 & 0b11111;
  unpacked[5] = p2 >> 5 | ((p4 & 0b00110000) >> 1);
  unpacked[6] = p3 & 0b11111;
  unpacked[7] = p3 >> 5 | ((p4 & 0b11000000) >> 3);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint5_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3) {
  // This function is an optimized version of pack_8_uint5_values,
  // for 64 uint5 values.

  // The first 2 * 128 bits are packed in the following way:
  // p0 = unpacked0 | (unpacked1 << 5)
  // p1 = unpacked2 | (unpacked3 << 5)
  // The remaining 64 bits are packed in the following way:
  // p2 = (unpacked1_0 >> 3) | ((unpacked1_1 >> 3) << 2) |
  //      ((unpacke3_0 >> 3) << 4) | ((unpacked3_1 >> 3) << 6)
  // where _0, _1 suffixes denote the low and high 64 bits, respectively.

  uint8x16_t p0 = vorrq_u8(unpacked0, vshlq_n_u8(unpacked1, 5));
  uint8x16_t p1 = vorrq_u8(unpacked2, vshlq_n_u8(unpacked3, 5));

  uint8x8_t p2 = vshr_n_u8(vget_low_u8(unpacked1), 3);
  p2 = vorr_u8(p2, vshl_n_u8(vshr_n_u8(vget_high_u8(unpacked1), 3), 2));
  p2 = vorr_u8(p2, vshl_n_u8(vshr_n_u8(vget_low_u8(unpacked3), 3), 4));
  p2 = vorr_u8(p2, vshl_n_u8(vshr_n_u8(vget_high_u8(unpacked3), 3), 6));

  vst1q_u8(packed, p0);
  vst1q_u8(packed + 16, p1);
  vst1_u8(packed + 32, p2);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint5_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    const uint8_t* packed) {
  // Unpacks data packed by vec_pack_64_uint5_values

  // unpacked0 = p0 & 0b11111
  // unpacked1_0 = p0 >> 5 | ((p2 & 0b00000011) << 3)
  // unpacked1_1 = p0 >> 5 | ((p2 & 0b00001100) << 1)
  // unpacked2 = p1 & 0b11111
  // unpacked3_0 = p1 >> 5 | ((p2 & 0b00110000) >> 1)
  // unpacked3_1 = p1 >> 5 | ((p2 & 0b11000000) >> 3)
  // where _0, _1 suffixes denote the low and high 64 bits, respectively.

  uint8x16_t p0 = vld1q_u8(packed);
  uint8x16_t p1 = vld1q_u8(packed + 16);
  uint8x8_t p2 = vld1_u8(packed + 32);

  unpacked0 = vandq_u8(p0, vdupq_n_u8(0b11111));
  unpacked1 = vcombine_u8(
      vshl_n_u8(vand_u8(p2, vdup_n_u8(0b00000011)), 3),
      vshl_n_u8(vand_u8(p2, vdup_n_u8(0b00001100)), 1));
  unpacked1 = vorrq_u8(unpacked1, vshrq_n_u8(p0, 5));
  unpacked2 = vandq_u8(p1, vdupq_n_u8(0b11111));
  unpacked3 = vcombine_u8(
      vshr_n_u8(vand_u8(p2, vdup_n_u8(0b00110000)), 1),
      vshr_n_u8(vand_u8(p2, vdup_n_u8(0b11000000)), 3));
  unpacked3 = vorrq_u8(unpacked3, vshrq_n_u8(p1, 5));
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_128_uint5_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3,
    const uint8x16_t& unpacked4,
    const uint8x16_t& unpacked5,
    const uint8x16_t& unpacked6,
    const uint8x16_t& unpacked7) {
  // This function is a vectorized version of pack_8_uint5_values,
  // for 128 uint5 values.

  // The first 4 * 128 bits are packed in the following way:
  // p0 = unpacked0 | (unpacked1 << 5)
  // p1 = unpacked2 | (unpacked3 << 5)
  // p2 = unpacked4 | (unpacked5 << 5)
  // p3 = unpacked6 | (unpacked7 << 5)
  // The remaining 128 bits are packed in the following way:
  // p4 = (unpacked1 >> 3) | ((unpacked3 >> 3) << 2) |
  //      ((unpacked5 >> 3) << 4) | ((unpacked7 >> 3) << 6)

  uint8x16_t p0 = vorrq_u8(unpacked0, vshlq_n_u8(unpacked1, 5));
  uint8x16_t p1 = vorrq_u8(unpacked2, vshlq_n_u8(unpacked3, 5));
  uint8x16_t p2 = vorrq_u8(unpacked4, vshlq_n_u8(unpacked5, 5));
  uint8x16_t p3 = vorrq_u8(unpacked6, vshlq_n_u8(unpacked7, 5));

  uint8x16_t p4 = vshrq_n_u8(unpacked1, 3);
  p4 = vorrq_u8(p4, vshlq_n_u8(vshrq_n_u8(unpacked3, 3), 2));
  p4 = vorrq_u8(p4, vshlq_n_u8(vshrq_n_u8(unpacked5, 3), 4));
  p4 = vorrq_u8(p4, vshlq_n_u8(vshrq_n_u8(unpacked7, 3), 6));

  vst1q_u8(packed, p0);
  vst1q_u8(packed + 16, p1);
  vst1q_u8(packed + 32, p2);
  vst1q_u8(packed + 48, p3);
  vst1q_u8(packed + 64, p4);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_128_uint5_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    uint8x16_t& unpacked4,
    uint8x16_t& unpacked5,
    uint8x16_t& unpacked6,
    uint8x16_t& unpacked7,
    const uint8_t* packed) {
  // Unpacks data packed by vec_pack_128_uint5_values

  // unpacked0 = p0 & 0b11111
  // unpacked1 = p0 >> 5 | ((p4 & 0b00000011) << 3)
  // unpacked2 = p1 & 0b11111
  // unpacked3 = p1 >> 5 | ((p4 & 0b00001100) << 1)
  // unpacked4 = p2 & 0b11111
  // unpacked5 = p2 >> 5 | ((p4 & 0b00110000) >> 1)
  // unpacked6 = p3 & 0b11111
  // unpacked7 = p3 >> 5 | ((p4 & 0b11000000) >> 3)

  uint8x16_t p0 = vld1q_u8(packed);
  uint8x16_t p1 = vld1q_u8(packed + 16);
  uint8x16_t p2 = vld1q_u8(packed + 32);
  uint8x16_t p3 = vld1q_u8(packed + 48);
  uint8x16_t p4 = vld1q_u8(packed + 64);

  unpacked0 = vandq_u8(p0, vdupq_n_u8(0b11111));
  unpacked1 = vorrq_u8(
      vshrq_n_u8(p0, 5), vshlq_n_u8(vandq_u8(p4, vdupq_n_u8(0b00000011)), 3));
  unpacked2 = vandq_u8(p1, vdupq_n_u8(0b11111));
  unpacked3 = vorrq_u8(
      vshrq_n_u8(p1, 5), vshlq_n_u8(vandq_u8(p4, vdupq_n_u8(0b00001100)), 1));
  unpacked4 = vandq_u8(p2, vdupq_n_u8(0b11111));
  unpacked5 = vorrq_u8(
      vshrq_n_u8(p2, 5), vshrq_n_u8(vandq_u8(p4, vdupq_n_u8(0b00110000)), 1));
  unpacked6 = vandq_u8(p3, vdupq_n_u8(0b11111));
  unpacked7 = vorrq_u8(
      vshrq_n_u8(p3, 5), vshrq_n_u8(vandq_u8(p4, vdupq_n_u8(0b11000000)), 3));
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
