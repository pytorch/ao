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

TORCHAO_ALWAYS_INLINE inline void pack_8_uint5_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // Given 8 unpacked uint5 values: 0abcd, 1efgh, 2ijkl, 3mnop, 4qrst, 5uvwx,
  // 6yzAB, 7CDEF, this function packs them as:
  //    b4: 7|6|5|4|3|2|1|0 (upper bits for all values)
  //    b3210_0: efgh|abcd (lower 4 bits for first 2 values)
  //    b3210_1: mnop|ijkl (lower 4 bits for second 2 values)
  //    b3210_2: qrst|uvwx (lower 4 bits for third 2 values)
  //    b3210_3: yzAB|CDEF (lower 4 bits for fourth 2 values)

  // These are stored in packed as: b2, b3210_0, b3210_1, b3210_2, b3210_3
  //
  // Input is 8 bytes
  // Output is 5 * 8 bits = 5 bytes

  // b4
  packed[0] = ((unpacked[0] & 16) >> 4) | ((unpacked[1] & 16) >> 3) |
      ((unpacked[2] & 16) >> 2) | ((unpacked[3] & 16) >> 1) |
      ((unpacked[4] & 16)) | ((unpacked[5] & 16) << 1) |
      ((unpacked[6] & 16) << 2) | ((unpacked[7] & 16) << 3);

  // b3210_0
  packed[1] = (unpacked[0] & 15) | ((unpacked[1] & 15) << 4);

  // b3210_1
  packed[2] = (unpacked[2] & 15) | ((unpacked[3] & 15) << 4);

  // b3210_2
  packed[3] = (unpacked[4] & 15) | ((unpacked[5] & 15) << 4);

  // b3210_3
  packed[4] = (unpacked[6] & 15) | ((unpacked[7] & 15) << 4);
}

TORCHAO_ALWAYS_INLINE inline void unpack_8_uint5_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // Unpacks data packed by pack_8_uint5_values
  //
  // Input is 40 bits = 5 bytes
  // Output is 8 bytes

  uint8_t b4 = packed[0];
  uint8_t b3210_0 = packed[1];
  uint8_t b3210_1 = packed[2];
  uint8_t b3210_2 = packed[3];
  uint8_t b3210_3 = packed[4];

  unpacked[0] = ((b4 & 1) << 4) | (b3210_0 & 15);
  unpacked[1] = ((b4 & 2) << 3) | (b3210_0 >> 4);

  unpacked[2] = ((b4 & 4) << 2) | (b3210_1 & 15);
  unpacked[3] = ((b4 & 8) << 1) | (b3210_1 >> 4);

  unpacked[4] = (b4 & 16) | (b3210_2 & 15);
  unpacked[5] = ((b4 & 32) >> 1) | (b3210_2 >> 4);

  unpacked[6] = ((b4 & 64) >> 2) | (b3210_3 & 15);
  unpacked[7] = ((b4 & 128) >> 3) | (b3210_3 >> 4);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint5_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3) {
  // This function is a vectorized version of pack_8_uint5_values
  // To understand it, please see pack_8_uint5_values first.
  // Before each code section, there is a comment indicating the
  // code in pack_8_uint5_values that is being vectorized
  //
  // Input is 64 bytes
  // Output is 5*64= 320 bits = 40 bytes

  uint8x8_t b4;
  uint8x8_t mask;

  // b4
  //  packed[0] = ((unpacked[0] & 16) >> 4) | ((unpacked[1] & 16) >> 3) |
  //       ((unpacked[2] & 16) >> 2) | ((unpacked[3] & 16) >> 1) |
  //       ((unpacked[4] & 16)) | ((unpacked[5] & 16) << 1) |
  //       ((unpacked[6] & 16) << 2) | ((unpacked[7] & 16) << 3);
  mask = vdup_n_u8(16);
  b4 = vshr_n_u8(vand_u8(vget_low_u8(unpacked0), mask), 4);
  b4 = vorr_u8(b4, vshr_n_u8(vand_u8(vget_high_u8(unpacked0), mask), 3));

  b4 = vorr_u8(b4, vshr_n_u8(vand_u8(vget_low_u8(unpacked1), mask), 2));
  b4 = vorr_u8(b4, vshr_n_u8(vand_u8(vget_high_u8(unpacked1), mask), 1));

  b4 = vorr_u8(b4, vand_u8(vget_low_u8(unpacked2), mask));
  b4 = vorr_u8(b4, vshl_n_u8(vand_u8(vget_high_u8(unpacked2), mask), 1));

  b4 = vorr_u8(b4, vshl_n_u8(vand_u8(vget_low_u8(unpacked3), mask), 2));
  b4 = vorr_u8(b4, vshl_n_u8(vand_u8(vget_high_u8(unpacked3), mask), 3));

  vst1_u8(packed, b4);

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

  // b3210_2
  // packed[3] = (unpacked[4] & 15) | ((unpacked[5] & 15) << 4);
  b3210 = vand_u8(vget_low_u8(unpacked2), mask);
  b3210 = vorr_u8(b3210, vshl_n_u8(vand_u8(vget_high_u8(unpacked2), mask), 4));
  vst1_u8(packed + 24, b3210);

  // b3210_3
  //  packed[4] = (unpacked[6] & 15) | ((unpacked[7] & 15) << 4);
  b3210 = vand_u8(vget_low_u8(unpacked3), mask);
  b3210 = vorr_u8(b3210, vshl_n_u8(vand_u8(vget_high_u8(unpacked3), mask), 4));
  vst1_u8(packed + 32, b3210);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint5_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    const uint8_t* packed) {
  // Unpacks data packed by pack_64_uint5_values
  //
  // This function vectorizes vec_unpack_8_uint5_values
  // To understand it, please see vec_unpack_8_uint5_values first.
  // Before each code section, there is a comment indicating the
  // code in vec_unpack_8_uint5_values that is being vectorized

  // Input is 5*64 = 320 bits = 40 bytes
  // Output is 64 bytes

  uint8x8_t b4 = vld1_u8(packed);
  uint8x8_t b3210;
  uint8x8_t unpacked_tmp0;
  uint8x8_t unpacked_tmp1;

  // unpacked[0] = ((b4 & 1) << 4) | (b3210_0 & 15);
  // unpacked[1] = ((b4 & 2) << 3) | (b3210_0 >> 4);
  b3210 = vld1_u8(packed + 8);

  unpacked_tmp0 = vshl_n_u8(vand_u8(b4, vdup_n_u8(1)), 4);
  unpacked_tmp0 = vorr_u8(unpacked_tmp0, vand_u8(b3210, vdup_n_u8(15)));

  unpacked_tmp1 = vshl_n_u8(vand_u8(b4, vdup_n_u8(2)), 3);
  unpacked_tmp1 = vorr_u8(unpacked_tmp1, vshr_n_u8(b3210, 4));

  unpacked0 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  // unpacked[2] = ((b4 & 4) << 2) | (b3210_1 & 15);
  // unpacked[3] = ((b4 & 8) << 1) | (b3210_1 >> 4);
  b3210 = vld1_u8(packed + 16);

  unpacked_tmp0 = vshl_n_u8(vand_u8(b4, vdup_n_u8(4)), 2);
  unpacked_tmp0 = vorr_u8(unpacked_tmp0, vand_u8(b3210, vdup_n_u8(15)));

  unpacked_tmp1 = vshl_n_u8(vand_u8(b4, vdup_n_u8(8)), 1);
  unpacked_tmp1 = vorr_u8(unpacked_tmp1, vshr_n_u8(b3210, 4));

  unpacked1 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  // unpacked[4] = (b4 & 16) | (b3210_2 & 15);
  // unpacked[5] = ((b4 & 32) >> 1) | (b3210_2 >> 4);

  b3210 = vld1_u8(packed + 24);

  unpacked_tmp0 = vand_u8(b4, vdup_n_u8(16));
  unpacked_tmp0 = vorr_u8(unpacked_tmp0, vand_u8(b3210, vdup_n_u8(15)));

  unpacked_tmp1 = vshr_n_u8(vand_u8(b4, vdup_n_u8(32)), 1);
  unpacked_tmp1 = vorr_u8(unpacked_tmp1, vshr_n_u8(b3210, 4));

  unpacked2 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);

  // unpacked[6] = ((b4 & 64) >> 2) | (b3210_3 & 15);
  // unpacked[7] = ((b4 & 128) >> 3) | (b3210_3 >> 4);

  b3210 = vld1_u8(packed + 32);

  unpacked_tmp0 = vshr_n_u8(vand_u8(b4, vdup_n_u8(64)), 2);
  unpacked_tmp0 = vorr_u8(unpacked_tmp0, vand_u8(b3210, vdup_n_u8(15)));

  unpacked_tmp1 = vshr_n_u8(vand_u8(b4, vdup_n_u8(128)), 3);
  unpacked_tmp1 = vorr_u8(unpacked_tmp1, vshr_n_u8(b3210, 4));

  unpacked3 = vcombine_u8(unpacked_tmp0, unpacked_tmp1);
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
  // This function is a vectorized version of pack_8_uint5_values
  // To understand it, please see pack_8_uint5_values first.
  // Before each code section, there is a comment indicating the
  // code in pack_8_uint5_values that is being vectorized
  //
  // Input is 128 bytes
  // Output is 5*128= 640 bits = 80 bytes

  uint8x16_t b4;
  uint8x16_t mask;

  // b4
  //  packed[0] = ((unpacked[0] & 16) >> 4) | ((unpacked[1] & 16) >> 3) |
  //       ((unpacked[2] & 16) >> 2) | ((unpacked[3] & 16) >> 1) |
  //       ((unpacked[4] & 16)) | ((unpacked[5] & 16) << 1) |
  //       ((unpacked[6] & 16) << 2) | ((unpacked[7] & 16) << 3);
  mask = vdupq_n_u8(16);
  b4 = vshrq_n_u8(vandq_u8(unpacked0, mask), 4);
  b4 = vorrq_u8(b4, vshrq_n_u8(vandq_u8(unpacked1, mask), 3));

  b4 = vorrq_u8(b4, vshrq_n_u8(vandq_u8(unpacked2, mask), 2));
  b4 = vorrq_u8(b4, vshrq_n_u8(vandq_u8(unpacked3, mask), 1));

  b4 = vorrq_u8(b4, vandq_u8(unpacked4, mask));
  b4 = vorrq_u8(b4, vshlq_n_u8(vandq_u8(unpacked5, mask), 1));

  b4 = vorrq_u8(b4, vshlq_n_u8(vandq_u8(unpacked6, mask), 2));
  b4 = vorrq_u8(b4, vshlq_n_u8(vandq_u8(unpacked7, mask), 3));

  vst1q_u8(packed, b4);

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

  // b3210_2
  // packed[3] = (unpacked[4] & 15) | ((unpacked[5] & 15) << 4);
  b3210 = vandq_u8(unpacked4, mask);
  b3210 = vorrq_u8(b3210, vshlq_n_u8(vandq_u8(unpacked5, mask), 4));
  vst1q_u8(packed + 48, b3210);

  // b3210_3
  //  packed[4] = (unpacked[6] & 15) | ((unpacked[7] & 15) << 4);
  b3210 = vandq_u8(unpacked6, mask);
  b3210 = vorrq_u8(b3210, vshlq_n_u8(vandq_u8(unpacked7, mask), 4));
  vst1q_u8(packed + 64, b3210);
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
  // Unpacks data packed by pack_128_uint5_values
  //
  // This function vectorizes vec_unpack_8_uint5_values
  // To understand it, please see vec_unpack_8_uint5_values first.
  // Before each code section, there is a comment indicating the
  // code in vec_unpack_8_uint5_values that is being vectorized

  // Input is 5*128 = 640 bits = 80 bytes
  // Output is 128 bytes

  uint8x16_t b4 = vld1q_u8(packed);
  uint8x16_t b3210;

  // unpacked[0] = ((b4 & 1) << 4) | (b3210_0 & 15);
  // unpacked[1] = ((b4 & 2) << 3) | (b3210_0 >> 4);
  b3210 = vld1q_u8(packed + 16);

  unpacked0 = vshlq_n_u8(vandq_u8(b4, vdupq_n_u8(1)), 4);
  unpacked0 = vorrq_u8(unpacked0, vandq_u8(b3210, vdupq_n_u8(15)));

  unpacked1 = vshlq_n_u8(vandq_u8(b4, vdupq_n_u8(2)), 3);
  unpacked1 = vorrq_u8(unpacked1, vshrq_n_u8(b3210, 4));

  // unpacked[2] = ((b4 & 4) << 2) | (b3210_1 & 15);
  // unpacked[3] = ((b4 & 8) << 1) | (b3210_1 >> 4);
  b3210 = vld1q_u8(packed + 32);

  unpacked2 = vshlq_n_u8(vandq_u8(b4, vdupq_n_u8(4)), 2);
  unpacked2 = vorrq_u8(unpacked2, vandq_u8(b3210, vdupq_n_u8(15)));

  unpacked3 = vshlq_n_u8(vandq_u8(b4, vdupq_n_u8(8)), 1);
  unpacked3 = vorrq_u8(unpacked3, vshrq_n_u8(b3210, 4));

  // unpacked[4] = (b4 & 16) | (b3210_2 & 15);
  // unpacked[5] = ((b4 & 32) >> 1) | (b3210_2 >> 4);

  b3210 = vld1q_u8(packed + 48);

  unpacked4 = vandq_u8(b4, vdupq_n_u8(16));
  unpacked4 = vorrq_u8(unpacked4, vandq_u8(b3210, vdupq_n_u8(15)));

  unpacked5 = vshrq_n_u8(vandq_u8(b4, vdupq_n_u8(32)), 1);
  unpacked5 = vorrq_u8(unpacked5, vshrq_n_u8(b3210, 4));

  // unpacked[6] = ((b4 & 64) >> 2) | (b3210_3 & 15);
  // unpacked[7] = ((b4 & 128) >> 3) | (b3210_3 >> 4);

  b3210 = vld1q_u8(packed + 64);

  unpacked6 = vshrq_n_u8(vandq_u8(b4, vdupq_n_u8(64)), 2);
  unpacked6 = vorrq_u8(unpacked6, vandq_u8(b3210, vdupq_n_u8(15)));

  unpacked7 = vshrq_n_u8(vandq_u8(b4, vdupq_n_u8(128)), 3);
  unpacked7 = vorrq_u8(unpacked7, vshrq_n_u8(b3210, 4));
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
