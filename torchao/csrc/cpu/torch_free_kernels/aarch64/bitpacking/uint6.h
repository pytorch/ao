// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>

// This file contains bitpacking and unpacking methods for uint5.
// These are not inteded to be used outside of bitpacking directory.
// See bitpack.h for the interface.

namespace torchao {
namespace bitpacking {
namespace internal {

TORCHAO_ALWAYS_INLINE inline void pack_4_uint6_values(
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

TORCHAO_ALWAYS_INLINE inline void unpack_4_uint6_values(
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
      ((packed[1] & 0b1100'0000u) >> 4) | ((packed[2] & 0b1100'0000u) >> 2);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_32_uint6_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1) {
  // This function is a vectorized version of pack_4_uint6_values_v2.
  // To understand the following code, please see pack_4_uint6_values_v2 first
  // and consider the following mapping for the unpacked parameter of that
  // function:
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
  r = vorr_u8(
      r, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), vdup_n_u8(0b00'0011u)), 6));
  vst1_u8(packed, r);

  // packed[1] = unpacked[1]
  // packed[1] |= ((unpacked[3] & 0b00'1100u) << 4)
  r = vget_high_u8(unpacked0);
  r = vorr_u8(
      r, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), vdup_n_u8(0b00'1100u)), 4));
  vst1_u8(packed + 8, r);

  // packed[2] = unpacked[2]
  // packed[2] |= ((unpacked[3] & 0b11'0000u) << 2)
  r = vget_low_u8(unpacked1);
  r = vorr_u8(
      r, vshl_n_u8(vand_u8(vget_high_u8(unpacked1), vdup_n_u8(0b11'0000u)), 2));
  vst1_u8(packed + 16, r);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_32_uint6_values(
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

  uint8x8_t unpacked3;
  // We want to extract bits 123456 and place them in unpacked3.
  // Packed structure is:
  //
  // packed0: 56 | abcdef
  // packed1: 34 | ghijkl
  // packed2: 12 | mnopqr
  //
  // unpacked3 = 1234 ghij
  unpacked3 = vsri_n_u8(packed2, packed1, 2);
  // unpacked3 = 1234 56ab
  unpacked3 = vsri_n_u8(unpacked3, packed0, 4);
  // unpacked3 = 0012 3456
  unpacked3 = vshr_n_u8(unpacked3, 2);

  // unpacked[i] = packed[i] & 0b11'1111u;
  const uint8x8_t mask = vdup_n_u8(0b11'1111u);
  unpacked0 = vcombine_u8(vand_u8(packed0, mask), vand_u8(packed1, mask));
  unpacked1 = vcombine_u8(vand_u8(packed2, mask), unpacked3);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint6_values(
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

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint6_values(
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

  // We want to extract bits 123456 and place them in unpacked3.
  // Packed structure is:
  //
  // packed0: 56 | abcdef
  // packed1: 34 | ghijkl
  // packed2: 12 | mnopqr
  //
  // unpacked3 = 1234 ghij
  unpacked3 = vsriq_n_u8(unpacked2, unpacked1, 2);
  // unpacked3 = 1234 56ab
  unpacked3 = vsriq_n_u8(unpacked3, unpacked0, 4);
  // unpacked3 = 0012 3456
  unpacked3 = vshrq_n_u8(unpacked3, 2);

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
