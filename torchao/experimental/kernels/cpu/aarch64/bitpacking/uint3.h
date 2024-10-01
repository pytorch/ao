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
  // Given 8 unpacked uint3 values: 0ab, 1cd, 2ef, 3gh, 4ij, 5kl, 6mn, 7op,
  // this function packs them as:
  //    b2: 7|6|5|4|3|2|1|0 (upper bits for all values)
  //    b10_0: gh|ef|cd|ab (lower 2 bits for first 4 values)
  //    b10_1: op|mn|kl|ij (lower 2 bits for last 4 values)
  // These are stored in packed as: b2, b10_0, b10_1
  //
  // Input is 8 bytes
  // Output is 24 bits = 3 bytes

  // b2
  packed[0] = ((unpacked[0] & 4) >> 2) | ((unpacked[1] & 4) >> 1) |
      ((unpacked[2] & 4)) | ((unpacked[3] & 4) << 1) |
      ((unpacked[4] & 4) << 2) | ((unpacked[5] & 4) << 3) |
      ((unpacked[6] & 4) << 4) | ((unpacked[7] & 4) << 5);

  // b10_0
  packed[1] = (unpacked[0] & 3) | ((unpacked[1] & 3) << 2) |
      ((unpacked[2] & 3) << 4) | ((unpacked[3] & 3) << 6);

  // b10_1
  packed[2] = (unpacked[4] & 3) | ((unpacked[5] & 3) << 2) |
      ((unpacked[6] & 3) << 4) | ((unpacked[7] & 3) << 6);
}

TORCHAO_ALWAYS_INLINE inline void unpack_8_uint3_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // Unpacks data packed by pack_8_uint3_values
  //
  // Input is 24 bits = 3 bytes
  // Output is 8 bytes

  uint8_t b2 = packed[0];
  uint8_t b10_0 = packed[1];
  uint8_t b10_1 = packed[2];

  unpacked[0] = ((b2 & 1) << 2) | (b10_0 & 3);
  unpacked[1] = ((b2 & 2) << 1) | ((b10_0 & 12) >> 2);
  unpacked[2] = (b2 & 4) | ((b10_0 & 48) >> 4);
  unpacked[3] = ((b2 & 8) >> 1) | ((b10_0 & 192) >> 6);

  unpacked[4] = ((b2 & 16) >> 2) | (b10_1 & 3);
  unpacked[5] = ((b2 & 32) >> 3) | ((b10_1 & 12) >> 2);
  unpacked[6] = ((b2 & 64) >> 4) | ((b10_1 & 48) >> 4);
  unpacked[7] = ((b2 & 128) >> 5) | ((b10_1 & 192) >> 6);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint3_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3) {
  // This function is a vectorized version of pack_8_uint3_values
  // To understand it, please see pack_8_uint3_values first.
  // Before each code section, there is a comment indicating the
  // code in pack_8_uint3_values that is being vectorized
  //
  // Input is 64 bytes
  // Output is 3*64= 192 bits = 24 bytes

  uint8x8_t b2;
  uint8x8_t mask;

  // b2
  // packed[0] = ((unpacked[0] & 4) >> 2) | ((unpacked[1] & 4) >> 1) |
  //     ((unpacked[2] & 4)) | ((unpacked[3] & 4) << 1) |
  //     ((unpacked[4] & 4) << 2) | ((unpacked[5] & 4) << 3) |
  //     ((unpacked[6] & 4) << 4) | ((unpacked[7] & 4) << 5);
  mask = vdup_n_u8(4);
  b2 = vshr_n_u8(vand_u8(vget_low_u8(unpacked0), mask), 2);
  b2 = vorr_u8(b2, vshr_n_u8(vand_u8(vget_high_u8(unpacked0), mask), 1));

  b2 = vorr_u8(b2, vand_u8(vget_low_u8(unpacked1), mask));
  b2 = vorr_u8(b2, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), mask), 1));

  b2 = vorr_u8(b2, vshl_n_u8(vand_u8(vget_low_u8(unpacked2), mask), 2));
  b2 = vorr_u8(b2, vshl_n_u8(vand_u8(vget_high_u8(unpacked2), mask), 3));

  b2 = vorr_u8(b2, vshl_n_u8(vand_u8(vget_low_u8(unpacked3), mask), 4));
  b2 = vorr_u8(b2, vshl_n_u8(vand_u8(vget_high_u8(unpacked3), mask), 5));

  vst1_u8(packed, b2);

  // b10_0
  // packed[1] = (unpacked[0] & 3) | ((unpacked[1] & 3) << 2) |
  //     ((unpacked[2] & 3) << 4) | ((unpacked[3] & 3) << 6);
  mask = vdup_n_u8(3);
  uint8x8_t b10_0;

  b10_0 = vand_u8(vget_low_u8(unpacked0), mask);
  b10_0 = vorr_u8(b10_0, vshl_n_u8(vand_u8(vget_high_u8(unpacked0), mask), 2));

  b10_0 = vorr_u8(b10_0, vshl_n_u8(vand_u8(vget_low_u8(unpacked1), mask), 4));
  b10_0 = vorr_u8(b10_0, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), mask), 6));

  vst1_u8(packed + 8, b10_0);

  // b10_1
  // packed[2] = (unpacked[4] & 3) | ((unpacked[5] & 3) << 2) |
  //     ((unpacked[6] & 3) << 4) | ((unpacked[7] & 3) << 6);
  uint8x8_t b10_1;

  b10_1 = vand_u8(vget_low_u8(unpacked2), mask);
  b10_1 = vorr_u8(b10_1, vshl_n_u8(vand_u8(vget_high_u8(unpacked2), mask), 2));

  b10_1 = vorr_u8(b10_1, vshl_n_u8(vand_u8(vget_low_u8(unpacked3), mask), 4));
  b10_1 = vorr_u8(b10_1, vshl_n_u8(vand_u8(vget_high_u8(unpacked3), mask), 6));

  vst1_u8(packed + 16, b10_1);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint3_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    const uint8_t* packed) {
  // Unpacks data packed by pack_64_uint3_values
  //
  // This function vectorizes vec_unpack_8_uint3_values
  // To understand it, please see vec_unpack_8_uint3_values first.
  // Before each code section, there is a comment indicating the
  // code in vec_unpack_8_uint3_values that is being vectorized

  // Input is 3*64= 192 bits = 24 bytes
  // Output is 64 bytes

  uint8x8_t b2 = vld1_u8(packed);
  uint8x8_t b10_0 = vld1_u8(packed + 8);
  uint8x8_t unpacked_tmp0;
  uint8x8_t unpacked_tmp1;

  // unpacked[0] = ((b2 & 1) << 2) | (b10_0 & 3);
  unpacked_tmp0 = vshl_n_u8(vand_u8(b2, vdup_n_u8(1)), 2);
  unpacked_tmp0 = vorr_u8(unpacked_tmp0, vand_u8(b10_0, vdup_n_u8(3)));

  // unpacked[1] = ((b2 & 2) << 1) | ((b10_0 & 12) >> 2);
  unpacked_tmp1 = vshl_n_u8(vand_u8(b2, vdup_n_u8(2)), 1);
  unpacked_tmp1 =
      vorr_u8(unpacked_tmp1, vshr_n_u8(vand_u8(b10_0, vdup_n_u8(12)), 2));

  unpacked0 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  // unpacked[2] = (b2 & 4) | ((b10_0 & 48) >> 4);
  unpacked_tmp0 = vand_u8(b2, vdup_n_u8(4));
  unpacked_tmp0 =
      vorr_u8(unpacked_tmp0, vshr_n_u8(vand_u8(b10_0, vdup_n_u8(48)), 4));

  // unpacked[3] = ((b2 & 8) >> 1) | ((b10_0 & 192) >> 6);
  unpacked_tmp1 = vshr_n_u8(vand_u8(b2, vdup_n_u8(8)), 1);
  unpacked_tmp1 =
      vorr_u8(unpacked_tmp1, vshr_n_u8(vand_u8(b10_0, vdup_n_u8(192)), 6));

  unpacked1 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  // unpacked[4] = ((b2 & 16) >> 2) | (b10_1 & 3);
  uint8x8_t b10_1 = vld1_u8(packed + 16);
  unpacked_tmp0 = vshr_n_u8(vand_u8(b2, vdup_n_u8(16)), 2);
  unpacked_tmp0 = vorr_u8(unpacked_tmp0, vand_u8(b10_1, vdup_n_u8(3)));

  // unpacked[5] = ((b2 & 32) >> 3) | ((b10_1 & 12) >> 2);
  unpacked_tmp1 = vshr_n_u8(vand_u8(b2, vdup_n_u8(32)), 3);
  unpacked_tmp1 =
      vorr_u8(unpacked_tmp1, vshr_n_u8(vand_u8(b10_1, vdup_n_u8(12)), 2));

  unpacked2 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  // unpacked[6] = ((b2 & 64) >> 4) | ((b10_1 & 48) >> 4);
  unpacked_tmp0 = vshr_n_u8(vand_u8(b2, vdup_n_u8(64)), 4);
  unpacked_tmp0 =
      vorr_u8(unpacked_tmp0, vshr_n_u8(vand_u8(b10_1, vdup_n_u8(48)), 4));

  // unpacked[7] = ((b2 & 128) >> 5) | ((b10_1 & 192) >> 6);
  unpacked_tmp1 = vshr_n_u8(vand_u8(b2, vdup_n_u8(128)), 5);
  unpacked_tmp1 =
      vorr_u8(unpacked_tmp1, vshr_n_u8(vand_u8(b10_1, vdup_n_u8(192)), 6));
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

  uint8x16_t b2;
  uint8x16_t mask;

  // b2
  // packed[0] = ((unpacked[0] & 4) >> 2) | ((unpacked[1] & 4) >> 1) |
  //     ((unpacked[2] & 4)) | ((unpacked[3] & 4) << 1) |
  //     ((unpacked[4] & 4) << 2) | ((unpacked[5] & 4) << 3) |
  //     ((unpacked[6] & 4) << 4) | ((unpacked[7] & 4) << 5);
  mask = vdupq_n_u8(4);
  b2 = vshrq_n_u8(vandq_u8(unpacked0, mask), 2);
  b2 = vorrq_u8(b2, vshrq_n_u8(vandq_u8(unpacked1, mask), 1));
  b2 = vorrq_u8(b2, vandq_u8(unpacked2, mask));
  b2 = vorrq_u8(b2, vshlq_n_u8(vandq_u8(unpacked3, mask), 1));
  b2 = vorrq_u8(b2, vshlq_n_u8(vandq_u8(unpacked4, mask), 2));
  b2 = vorrq_u8(b2, vshlq_n_u8(vandq_u8(unpacked5, mask), 3));
  b2 = vorrq_u8(b2, vshlq_n_u8(vandq_u8(unpacked6, mask), 4));
  b2 = vorrq_u8(b2, vshlq_n_u8(vandq_u8(unpacked7, mask), 5));

  vst1q_u8(packed, b2);

  // b10_0
  // packed[1] = (unpacked[0] & 3) | ((unpacked[1] & 3) << 2) |
  //     ((unpacked[2] & 3) << 4) | ((unpacked[3] & 3) << 6);
  mask = vdupq_n_u8(3);
  uint8x16_t b10_0;

  b10_0 = vandq_u8(unpacked0, mask);
  b10_0 = vorrq_u8(b10_0, vshlq_n_u8(vandq_u8(unpacked1, mask), 2));
  b10_0 = vorrq_u8(b10_0, vshlq_n_u8(vandq_u8(unpacked2, mask), 4));
  b10_0 = vorrq_u8(b10_0, vshlq_n_u8(vandq_u8(unpacked3, mask), 6));

  vst1q_u8(packed + 16, b10_0);

  // b10_1
  // packed[2] = (unpacked[4] & 3) | ((unpacked[5] & 3) << 2) |
  //     ((unpacked[6] & 3) << 4) | ((unpacked[7] & 3) << 6);
  uint8x16_t b10_1;
  b10_1 = vandq_u8(unpacked4, mask);
  b10_1 = vorrq_u8(b10_1, vshlq_n_u8(vandq_u8(unpacked5, mask), 2));
  b10_1 = vorrq_u8(b10_1, vshlq_n_u8(vandq_u8(unpacked6, mask), 4));
  b10_1 = vorrq_u8(b10_1, vshlq_n_u8(vandq_u8(unpacked7, mask), 6));

  vst1q_u8(packed + 32, b10_1);
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

  uint8x16_t b2 = vld1q_u8(packed);
  uint8x16_t b10_0 = vld1q_u8(packed + 16);

  // unpacked[0] = ((b2 & 1) << 2) | (b10_0 & 3);
  unpacked0 = vshlq_n_u8(vandq_u8(b2, vdupq_n_u8(1)), 2);
  unpacked0 = vorrq_u8(unpacked0, vandq_u8(b10_0, vdupq_n_u8(3)));

  // unpacked[1] = ((b2 & 2) << 1) | ((b10_0 & 12) >> 2);
  unpacked1 = vshlq_n_u8(vandq_u8(b2, vdupq_n_u8(2)), 1);
  unpacked1 =
      vorrq_u8(unpacked1, vshrq_n_u8(vandq_u8(b10_0, vdupq_n_u8(12)), 2));

  // unpacked[2] = (b2 & 4) | ((b10_0 & 48) >> 4);
  unpacked2 = vandq_u8(b2, vdupq_n_u8(4));
  unpacked2 =
      vorrq_u8(unpacked2, vshrq_n_u8(vandq_u8(b10_0, vdupq_n_u8(48)), 4));

  // unpacked[3] = ((b2 & 8) >> 1) | ((b10_0 & 192) >> 6);
  unpacked3 = vshrq_n_u8(vandq_u8(b2, vdupq_n_u8(8)), 1);
  unpacked3 =
      vorrq_u8(unpacked3, vshrq_n_u8(vandq_u8(b10_0, vdupq_n_u8(192)), 6));

  // unpacked[4] = ((b2 & 16) >> 2) | (b10_1 & 3);
  uint8x16_t b10_1 = vld1q_u8(packed + 32);
  unpacked4 = vshrq_n_u8(vandq_u8(b2, vdupq_n_u8(16)), 2);
  unpacked4 = vorrq_u8(unpacked4, vandq_u8(b10_1, vdupq_n_u8(3)));

  // unpacked[5] = ((b2 & 32) >> 3) | ((b10_1 & 12) >> 2);
  unpacked5 = vshrq_n_u8(vandq_u8(b2, vdupq_n_u8(32)), 3);
  unpacked5 =
      vorrq_u8(unpacked5, vshrq_n_u8(vandq_u8(b10_1, vdupq_n_u8(12)), 2));

  // unpacked[6] = ((b2 & 64) >> 4) | ((b10_1 & 48) >> 4);
  unpacked6 = vshrq_n_u8(vandq_u8(b2, vdupq_n_u8(64)), 4);
  unpacked6 =
      vorrq_u8(unpacked6, vshrq_n_u8(vandq_u8(b10_1, vdupq_n_u8(48)), 4));

  // unpacked[7] = ((b2 & 128) >> 5) | ((b10_1 & 192) >> 6);
  unpacked7 = vshrq_n_u8(vandq_u8(b2, vdupq_n_u8(128)), 5);
  unpacked7 =
      vorrq_u8(unpacked7, vshrq_n_u8(vandq_u8(b10_1, vdupq_n_u8(192)), 6));
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
