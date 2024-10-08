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

TORCHAO_ALWAYS_INLINE inline void pack_2_uint4_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  packed[0] = (unpacked[0] << 4) | (unpacked[1] & 0xF);
}

TORCHAO_ALWAYS_INLINE inline void unpack_2_uint4_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  unpacked[0] = packed[0] >> 4;
  unpacked[1] = packed[0] & 0xF;
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_16_uint4_values(
    uint8_t* packed,
    const uint8x16_t& unpacked) {
  uint8x8_t unpacked_low = vget_low_u8(unpacked);
  uint8x8_t unpacked_high = vshl_n_u8(vget_high_u8(unpacked), 4);
  uint8x8_t packed_to_st = vorr_u8(unpacked_low, unpacked_high);
  vst1_u8(packed, packed_to_st);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_16_uint4_values(
    uint8x16_t& unpacked,
    const uint8_t* packed) {
  uint8x8_t packed_ld = vld1_u8(packed);
  uint8x8_t high = vshr_n_u8(packed_ld, 4);
  uint8x8_t low = vand_u8(packed_ld, vdup_n_u8(0xF));
  unpacked = vcombine_u8(low, high);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_32_uint4_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1) {
  uint8x16_t high = vshlq_n_u8(unpacked1, 4);
  uint8x16_t packed_to_st = vorrq_u8(unpacked0, high);
  vst1q_u8(packed, packed_to_st);
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_32_uint4_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    const uint8_t* packed) {
  uint8x16_t packed_ld = vld1q_u8(packed);
  unpacked1 = vshrq_n_u8(packed_ld, 4);
  unpacked0 = vandq_u8(packed_ld, vdupq_n_u8(0xF));
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
