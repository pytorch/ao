// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>

// This file contains bitpacking and unpacking methods for uint1.
// These are not inteded to be used outside of bitpacking directory.
// See bitpack.h for the interface.

namespace torchao {
namespace bitpacking {
namespace internal {

TORCHAO_ALWAYS_INLINE inline void pack_8_uint1_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // Input is 8 bytes
  // Output is 1 bytes
  packed[0] = 0;
  for (int i = 0; i < 8; i++) {
    packed[0] |= (unpacked[i] << (7 - i));
  }
}

TORCHAO_ALWAYS_INLINE inline void unpack_8_uint1_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // Unpacks data packed by pack_8_uint1_values
  //
  // Input is 8 bits = 1 byte
  // Output is 8 bytes
  for (int i = 0; i < 8; i++) {
    unpacked[i] = (packed[0] >> (7 - i)) & 1;
  }
}

// This function is a vectorized version of pack_8_uint1_values
// To understand it, please see pack_8_uint1_values first.
//
// Input is 64 bytes
// Output is 64 bits = 8 bytes
TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint1_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3) {
  uint8x16_t vec_packed;
  uint8x8_t vec_packed_low;
  uint8x8_t vec_packed_high;
  vec_packed = vshlq_n_u8(unpacked0, 3);
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked1, 2));
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked2, 1));
  vec_packed = vorrq_u8(vec_packed, unpacked3);

  vec_packed_low = vget_low_u8(vec_packed);
  vec_packed_high = vget_high_u8(vec_packed);

  vst1_u8(packed, vsli_n_u8(vec_packed_low, vec_packed_high, 4));
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint1_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    const uint8_t* packed) {
  uint8x8_t vec_packed;
  vec_packed = vld1_u8(packed);

  uint8x8_t vec_packed_low;
  uint8x8_t vec_packed_high;
  vec_packed_low = vand_u8(vec_packed, vdup_n_u8(0xF));
  vec_packed_high = vshr_n_u8(vec_packed, 4);

  uint8x16_t combined = vcombine_u8(vec_packed_low, vec_packed_high);
  unpacked0 = vshrq_n_u8(vandq_u8(combined, vdupq_n_u8(8)), 3);
  unpacked1 = vshrq_n_u8(vandq_u8(combined, vdupq_n_u8(4)), 2);
  unpacked2 = vshrq_n_u8(vandq_u8(combined, vdupq_n_u8(2)), 1);
  unpacked3 = vandq_u8(combined, vdupq_n_u8(1));
}

// This function is a vectorized version of pack_8_uint1_values
// To understand it, please see `pack_8_uint1_values` first.
//
// Input is 128 bytes
// Output is 128 bytes * 1 bit/8bits = 16 bytes
TORCHAO_ALWAYS_INLINE inline void vec_pack_128_uint1_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3,
    const uint8x16_t& unpacked4,
    const uint8x16_t& unpacked5,
    const uint8x16_t& unpacked6,
    const uint8x16_t& unpacked7) {
  uint8x16_t vec_packed;

  vec_packed = vshlq_n_u8(unpacked0, 7);
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked1, 6));
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked2, 5));
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked3, 4));
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked4, 3));
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked5, 2));
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked6, 1));
  vec_packed = vorrq_u8(vec_packed, unpacked7);

  vst1q_u8(packed, vec_packed);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_128_uint1_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    uint8x16_t& unpacked4,
    uint8x16_t& unpacked5,
    uint8x16_t& unpacked6,
    uint8x16_t& unpacked7,
    const uint8_t* packed) {
  uint8x16_t vec_packed;
  vec_packed = vld1q_u8(packed);

  unpacked0 = vandq_u8(vshrq_n_u8(vec_packed, 7), vdupq_n_u8(1));
  unpacked1 = vandq_u8(vshrq_n_u8(vec_packed, 6), vdupq_n_u8(1));
  unpacked2 = vandq_u8(vshrq_n_u8(vec_packed, 5), vdupq_n_u8(1));
  unpacked3 = vandq_u8(vshrq_n_u8(vec_packed, 4), vdupq_n_u8(1));
  unpacked4 = vandq_u8(vshrq_n_u8(vec_packed, 3), vdupq_n_u8(1));
  unpacked5 = vandq_u8(vshrq_n_u8(vec_packed, 2), vdupq_n_u8(1));
  unpacked6 = vandq_u8(vshrq_n_u8(vec_packed, 1), vdupq_n_u8(1));
  unpacked7 = vandq_u8(vec_packed, vdupq_n_u8(1));
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
