// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>

// This file contains bitpacking and unpacking methods for uint4.
// These are not inteded to be used outside of bitpacking directory.
// See bitpack.h for the interface.

namespace torchao {
namespace bitpacking {
namespace internal {

TORCHAO_ALWAYS_INLINE inline void pack_4_uint2_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // Input is 4 bytes
  // Output is 1 bytes

  packed[0] = (unpacked[0] << 6) | (unpacked[1] << 4) | (unpacked[2] << 2) |
      (unpacked[3]);
}

TORCHAO_ALWAYS_INLINE inline void unpack_4_uint2_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // Input is 1 bytes
  // Output is 4 bytes
  unpacked[0] = (packed[0] & 192) >> 6;
  unpacked[1] = (packed[0] & 48) >> 4;
  unpacked[2] = (packed[0] & 12) >> 2;
  unpacked[3] = (packed[0] & 3);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_32_uint2_values(
    uint8_t* packed,
    const uint8x8_t& unpacked0,
    const uint8x8_t& unpacked1,
    const uint8x8_t& unpacked2,
    const uint8x8_t& unpacked3) {
  // Input is 32 bytes
  // Output is 8 bytes

  // Vectorize the following:
  // packed[0] = (unpacked[0] << 6) | (unpacked[1] << 4) | (unpacked[2] << 2) |
  // (unpacked[3]);

  uint8x8_t vec_packed;
  vec_packed = vshl_n_u8(unpacked0, 6);
  vec_packed = vorr_u8(vec_packed, vshl_n_u8(unpacked1, 4));
  vec_packed = vorr_u8(vec_packed, vshl_n_u8(unpacked2, 2));
  vec_packed = vorr_u8(vec_packed, unpacked3);
  vst1_u8(packed, vec_packed);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_32_uint2_values(
    uint8x8_t& unpacked0,
    uint8x8_t& unpacked1,
    uint8x8_t& unpacked2,
    uint8x8_t& unpacked3,
    const uint8_t* packed) {
  // Input is 8 bytes
  // Output is 32 bytes

  // Vectorize the following:
  // unpacked[0] = (packed[0] & 192) >> 6;
  // unpacked[1] = (packed[0] & 48) >> 4;
  // unpacked[2] = (packed[0] & 12) >> 2;
  // unpacked[3] = (packed[0] & 3);

  uint8x8_t vec_packed;

  vec_packed = vld1_u8(packed);
  unpacked0 = vshr_n_u8(vand_u8(vec_packed, vdup_n_u8(192)), 6);
  unpacked1 = vshr_n_u8(vand_u8(vec_packed, vdup_n_u8(48)), 4);
  unpacked2 = vshr_n_u8(vand_u8(vec_packed, vdup_n_u8(12)), 2);
  unpacked3 = vand_u8(vec_packed, vdup_n_u8(3));
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint2_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3) {
  // Input is 64 bytes
  // Output is 16 bytes

  // Vectorize the following:
  // packed[0] = (unpacked[0] << 6) | (unpacked[1] << 4) | (unpacked[2] << 2) |
  // (unpacked[3]);

  uint8x16_t vec_packed;
  vec_packed = vshlq_n_u8(unpacked0, 6);
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked1, 4));
  vec_packed = vorrq_u8(vec_packed, vshlq_n_u8(unpacked2, 2));
  vec_packed = vorrq_u8(vec_packed, unpacked3);
  vst1q_u8(packed, vec_packed);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint2_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    const uint8_t* packed) {
  // Input is 16 bytes
  // Output is 64 bytes

  // Vectorize the following:
  // unpacked[0] = (packed[0] & 192) >> 6;
  // unpacked[1] = (packed[0] & 48) >> 4;
  // unpacked[2] = (packed[0] & 12) >> 2;
  // unpacked[3] = (packed[0] & 3);

  uint8x16_t vec_packed;

  vec_packed = vld1q_u8(packed);
  unpacked0 = vshrq_n_u8(vandq_u8(vec_packed, vdupq_n_u8(192)), 6);
  unpacked1 = vshrq_n_u8(vandq_u8(vec_packed, vdupq_n_u8(48)), 4);
  unpacked2 = vshrq_n_u8(vandq_u8(vec_packed, vdupq_n_u8(12)), 2);
  unpacked3 = vandq_u8(vec_packed, vdupq_n_u8(3));
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
