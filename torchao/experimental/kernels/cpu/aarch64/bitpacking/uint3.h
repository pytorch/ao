// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/macro.h>

// This file contains bitpacking and unpacking methods for uint3.
// These are not inteded to be used outside of bitpacking directory.
// See bitpack.h for the interface.

namespace torchao {
namespace bitpacking {
namespace internal {

TORCHAO_ALWAYS_INLINE inline void pack_8_uint3_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // Given 8 unpacked uint3 values: abc, def, ghi, jkl, mno, pqr, 012, 345
  // this function packs them as:
  //    b0: 12|abc|def (bottom bits from 7th value, full bits from 1st/2nd
  //    value)
  //    b1: 45|ghi|jkl (bottom bits from 8th value, full bits from
  //    3rd/4th value)
  //    b2: 03|mno|pqr (top bit from 7th/8th value, full bits
  //    from 5th/6th value)
  // These are stored in packed as: b0, b1, b2
  //
  // Input is 8 bytes
  // Output is 24 bits = 3 bytes

  // b0
  packed[0] = ((unpacked[6] & 3) << 6) | ((unpacked[0] & 7) << 3) | unpacked[1];

  // b1
  packed[1] = ((unpacked[7] & 3) << 6) | ((unpacked[2] & 7) << 3) | unpacked[3];

  // b2
  packed[2] = ((unpacked[6] & 4) << 5) | ((unpacked[7] & 4) << 4) |
      ((unpacked[4] & 7) << 3) | unpacked[5];
}

TORCHAO_ALWAYS_INLINE inline void unpack_8_uint3_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // Unpacks data packed by pack_8_uint3_values
  //
  // Input is 24 bits = 3 bytes
  // Output is 8 bytes

  uint8_t b0 = packed[0];
  uint8_t b1 = packed[1];
  uint8_t b2 = packed[2];

  unpacked[0] = ((b0 >> 3) & 7);
  unpacked[1] = b0 & 7;

  unpacked[2] = ((b1 >> 3) & 7);
  unpacked[3] = b1 & 7;

  unpacked[4] = ((b2 >> 3) & 7);
  unpacked[5] = b2 & 7;

  unpacked[6] = (b0 >> 6) | ((b2 >> 5) & 4);
  unpacked[7] = (b1 >> 6) | ((b2 >> 4) & 4);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint3_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0, // 0, 1
    const uint8x16_t& unpacked1, // 2, 3
    const uint8x16_t& unpacked2, // 4, 5
    const uint8x16_t& unpacked3) { // 6, 7
  // This function is a vectorized version of pack_8_uint3_values
  // To understand it, please see pack_8_uint3_values first.
  // Before each code section, there is a comment indicating the
  // code in pack_8_uint3_values that is being vectorized
  //
  // Input is 64 bytes
  // Output is 3*64= 192 bits = 24 bytes

  uint8x8_t b;
  // b0
  //   packed[0] = ((unpacked[6] & 3) << 6) | ((unpacked[0] & 7) << 3) |
  //   unpacked[1];
  b = vshl_n_u8(vand_u8(vget_low_u8(unpacked3), vdup_n_u8(3)), 6);
  b = vorr_u8(b, vshl_n_u8(vand_u8(vget_low_u8(unpacked0), vdup_n_u8(7)), 3));
  b = vorr_u8(b, vget_high_u8(unpacked0));
  vst1_u8(packed, b);

  // b1
  //    packed[1] = ((unpacked[7] & 3) << 6) | ((unpacked[2] & 7) << 3) |
  //    unpacked[3];
  b = vshl_n_u8(vand_u8(vget_high_u8(unpacked3), vdup_n_u8(3)), 6);
  b = vorr_u8(b, vshl_n_u8(vand_u8(vget_low_u8(unpacked1), vdup_n_u8(7)), 3));
  b = vorr_u8(b, vget_high_u8(unpacked1));
  vst1_u8(packed + 8, b);

  // b2
  //   packed[2] = ((unpacked[6] & 4) << 5) | ((unpacked[7] & 4) << 4) |
  //       ((unpacked[4] & 7) << 3) | unpacked[5];
  b = vshl_n_u8(vand_u8(vget_low_u8(unpacked3), vdup_n_u8(4)), 5);
  b = vorr_u8(b, vshl_n_u8(vand_u8(vget_high_u8(unpacked3), vdup_n_u8(4)), 4));
  b = vorr_u8(b, vshl_n_u8(vand_u8(vget_low_u8(unpacked2), vdup_n_u8(7)), 3));
  b = vorr_u8(b, vget_high_u8(unpacked2));
  vst1_u8(packed + 16, b);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint3_values(
    uint8x16_t& unpacked0, // 0, 1
    uint8x16_t& unpacked1, // 2, 3
    uint8x16_t& unpacked2, // 4, 5
    uint8x16_t& unpacked3, // 6, 7
    const uint8_t* packed) {
  // Unpacks data packed by pack_64_uint3_values
  //
  // This function vectorizes vec_unpack_8_uint3_values
  // To understand it, please see vec_unpack_8_uint3_values first.
  // Before each code section, there is a comment indicating the
  // code in vec_unpack_8_uint3_values that is being vectorized

  // Input is 3*64= 192 bits = 24 bytes
  // Output is 64 bytes

  uint8x8_t b0 = vld1_u8(packed);
  uint8x8_t b1 = vld1_u8(packed + 8);
  uint8x8_t b2 = vld1_u8(packed + 16);
  uint8x8_t unpacked_tmp0;
  uint8x8_t unpacked_tmp1;

  //   unpacked[0] = ((b0 >> 3) & 7);
  uint8x8_t mask = vdup_n_u8(7);
  unpacked_tmp0 = vand_u8(vshr_n_u8(b0, 3), mask);

  //   unpacked[1] = b0 & 7;
  unpacked_tmp1 = vand_u8(b0, mask);
  unpacked0 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  //   unpacked[2] = ((b1 >> 3) & 7);
  unpacked_tmp0 = vand_u8(vshr_n_u8(b1, 3), mask);

  //   unpacked[3] = b1 & 7;
  unpacked_tmp1 = vand_u8(b1, mask);
  unpacked1 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  //   unpacked[4] = ((b2 >> 3) & 7);
  unpacked_tmp0 = vand_u8(vshr_n_u8(b2, 3), mask);

  //   unpacked[5] = b2 & 7;
  unpacked_tmp1 = vand_u8(b2, mask);
  unpacked2 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  //   unpacked[6] = (b0 >> 6) | ((b2 >> 5) & 4);
  mask = vdup_n_u8(4);
  unpacked_tmp0 = vshr_n_u8(b0, 6);
  unpacked_tmp0 = vorr_u8(unpacked_tmp0, vand_u8(vshr_n_u8(b2, 5), mask));

  //   unpacked[7] = (b1 >> 6) | ((b2 >> 4) & 4);
  unpacked_tmp1 = vshr_n_u8(b1, 6);
  unpacked_tmp1 = vorr_u8(unpacked_tmp1, vand_u8(vshr_n_u8(b2, 4), mask));

  unpacked3 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_128_uint3_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3,
    const uint8x16_t& unpacked4,
    const uint8x16_t& unpacked5,
    const uint8x16_t& unpacked6,
    const uint8x16_t& unpacked7) {
  // This function is a vectorized version of pack_8_uint3_values
  // To understand it, please see pack_8_uint3_values first.
  // Before each code section, there is a comment indicating the
  // code in pack_8_uint3_values that is being vectorized
  //
  // Input is 128 bytes
  // Output is 3*128= 384 bits = 48 bytes

  uint8x16_t b;
  // b0
  //   packed[0] = ((unpacked[6] & 3) << 6) | ((unpacked[0] & 7) << 3) |
  //   unpacked[1];
  b = vshlq_n_u8(vandq_u8(unpacked6, vdupq_n_u8(3)), 6);
  b = vorrq_u8(b, vshlq_n_u8(vandq_u8(unpacked0, vdupq_n_u8(7)), 3));
  b = vorrq_u8(b, unpacked1);
  vst1q_u8(packed, b);

  // b1
  //    packed[1] = ((unpacked[7] & 3) << 6) | ((unpacked[2] & 7) << 3) |
  //    unpacked[3];
  b = vshlq_n_u8(vandq_u8(unpacked7, vdupq_n_u8(3)), 6);
  b = vorrq_u8(b, vshlq_n_u8(vandq_u8(unpacked2, vdupq_n_u8(7)), 3));
  b = vorrq_u8(b, unpacked3);
  vst1q_u8(packed + 16, b);

  // b2
  //   packed[2] = ((unpacked[6] & 4) << 5) | ((unpacked[7] & 4) << 4) |
  //       ((unpacked[4] & 7) << 3) | unpacked[5];
  b = vshlq_n_u8(vandq_u8(unpacked6, vdupq_n_u8(4)), 5);
  b = vorrq_u8(b, vshlq_n_u8(vandq_u8(unpacked7, vdupq_n_u8(4)), 4));
  b = vorrq_u8(b, vshlq_n_u8(vandq_u8(unpacked4, vdupq_n_u8(7)), 3));
  b = vorrq_u8(b, unpacked5);
  vst1q_u8(packed + 32, b);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_128_uint3_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    uint8x16_t& unpacked4,
    uint8x16_t& unpacked5,
    uint8x16_t& unpacked6,
    uint8x16_t& unpacked7,
    const uint8_t* packed) {
  // Unpacks data packed by pack_128_uint3_values
  //
  // This function vectorizes vec_unpack_8_uint3_values
  // To understand it, please see vec_unpack_8_uint3_values first.
  // Before each code section, there is a comment indicating the
  // code in vec_unpack_8_uint3_values that is being vectorized

  // Input is 3*128 = 384 bits = 48 bytes
  // Output is 128 bytes

  uint8x16_t b0 = vld1q_u8(packed);
  uint8x16_t b1 = vld1q_u8(packed + 16);
  uint8x16_t b2 = vld1q_u8(packed + 32);

  //   unpacked[0] = ((b0 >> 3) & 7);
  uint8x16_t mask = vdupq_n_u8(7);
  unpacked0 = vandq_u8(vshrq_n_u8(b0, 3), mask);

  //   unpacked[1] = b0 & 7;
  unpacked1 = vandq_u8(b0, mask);

  //   unpacked[2] = ((b1 >> 3) & 7);
  unpacked2 = vandq_u8(vshrq_n_u8(b1, 3), mask);

  //   unpacked[3] = b1 & 7;
  unpacked3 = vandq_u8(b1, mask);

  //   unpacked[4] = ((b2 >> 3) & 7);
  unpacked4 = vandq_u8(vshrq_n_u8(b2, 3), mask);

  //   unpacked[5] = b2 & 7;
  unpacked5 = vandq_u8(b2, mask);

  //   unpacked[6] = (b0 >> 6) | ((b2 >> 5) & 4);
  mask = vdupq_n_u8(4);
  unpacked6 = vshrq_n_u8(b0, 6);
  unpacked6 = vorrq_u8(unpacked6, vandq_u8(vshrq_n_u8(b2, 5), mask));

  //   unpacked[7] = (b1 >> 6) | ((b2 >> 4) & 4);
  unpacked7 = vshrq_n_u8(b1, 6);
  unpacked7 = vorrq_u8(unpacked7, vandq_u8(vshrq_n_u8(b2, 4), mask));
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
