// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <cstdint>

namespace torchao::kernels::cpu::fallback::bitpacking {
namespace internal {

/**
 * @brief Packs 8 bytes, each holding a 5-bit value (0-31), into 5 bytes.
 *
 * @param packed Pointer to the destination memory (5 bytes).
 * @param unpacked Pointer to the source memory (8 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void pack_8_uint5_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // pack 8 uint5 values (u0..u7) into 5 bytes (p0..p4)
  // p0 = u0_all | u1_low_3_bits
  // p1 = u2_all | u3_low_3_bits
  // p2 = u4_all | u5_low_3_bits
  // p3 = u6_all | u7_low_3_bits
  // p4 = u1_high_2_bits | u3_high_2_bits | u5_high_2_bits | u7_high_2_bits
  packed[0] = (unpacked[0] & 0x1F) | ((unpacked[1] & 0x1F) << 5);
  packed[1] = (unpacked[2] & 0x1F) | ((unpacked[3] & 0x1F) << 5);
  packed[2] = (unpacked[4] & 0x1F) | ((unpacked[5] & 0x1F) << 5);
  packed[3] = (unpacked[6] & 0x1F) | ((unpacked[7] & 0x1F) << 5);
  packed[4] = ((unpacked[1] & 0x1F) >> 3) | (((unpacked[3] & 0x1F) >> 3) << 2) |
      (((unpacked[5] & 0x1F) >> 3) << 4) | (((unpacked[7] & 0x1F) >> 3) << 6);
}

/**
 * @brief Unpacks 5 bytes into 8 bytes, each containing a 5-bit value.
 *
 * @param unpacked Pointer to the destination memory (8 bytes).
 * @param packed Pointer to the source memory (5 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void unpack_8_uint5_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  const uint8_t p0 = packed[0];
  const uint8_t p1 = packed[1];
  const uint8_t p2 = packed[2];
  const uint8_t p3 = packed[3];
  const uint8_t p4 = packed[4];

  // This is compatible with the scalar NEON version.
  unpacked[0] = p0 & 0x1F;
  unpacked[1] = (p0 >> 5) | ((p4 & 0x03) << 3);
  unpacked[2] = p1 & 0x1F;
  unpacked[3] = (p1 >> 5) | ((p4 & 0x0C) << 1);
  unpacked[4] = p2 & 0x1F;
  unpacked[5] = (p2 >> 5) | ((p4 & 0x30) >> 1);
  unpacked[6] = p3 & 0x1F;
  unpacked[7] = (p3 >> 5) | ((p4 & 0xC0) >> 3);
}

/**
 * @brief Packs 64 bytes (each a 5-bit value) into 40 bytes.
 * @param packed Pointer to the destination memory (40 bytes).
 * @param unpacked Pointer to the source memory (64 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_64_uint5_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_64_uint5_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // Pack the first 32 bytes (p0, p1)
  for (int i = 0; i < 16; ++i) {
    packed[i] = (unpacked[i] & 0x1F) | ((unpacked[i + 16] & 0x1F) << 5);
    packed[i + 16] = (unpacked[i + 32] & 0x1F) | ((unpacked[i + 48] & 0x1F) << 5);
  }

  // Pack the final 8 bytes (p2)
  for (int i = 0; i < 8; ++i) {
    uint8_t val1 = (unpacked[16 + i] >> 3) & 0x03;
    uint8_t val2 = (unpacked[24 + i] >> 3) & 0x03;
    uint8_t val3 = (unpacked[48 + i] >> 3) & 0x03;
    uint8_t val4 = (unpacked[56 + i] >> 3) & 0x03;
    packed[32 + i] = val1 | (val2 << 2) | (val3 << 4) | (val4 << 6);
  }
}

/**
 * @brief Unpacks 40 bytes into 64 bytes (each a 5-bit value).
 * @param unpacked Pointer to the destination memory (64 bytes).
 * @param packed Pointer to the source memory (40 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_64_uint5_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_64_uint5_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t p0 = packed[i];
    const uint8_t p1 = packed[i + 16];
    // p2 is only 8 bytes wide, so we use modulo to access it correctly.
    const uint8_t p2 = packed[32 + (i % 8)];

    unpacked[i] = p0 & 0x1F;
    unpacked[i + 32] = p1 & 0x1F;

    if (i < 8) {
      unpacked[i + 16] = (p0 >> 5) | ((p2 & 0x03) << 3);
      unpacked[i + 48] = (p1 >> 5) | ((p2 & 0x30) >> 1);
    } else {
      unpacked[i + 16] = (p0 >> 5) | ((p2 & 0x0C) << 1);
      unpacked[i + 48] = (p1 >> 5) | ((p2 & 0xC0) >> 3);
    }
  }
}

/**
 * @brief Packs 128 bytes (each a 5-bit value) into 80 bytes.
 * @param packed Pointer to the destination memory (80 bytes).
 * @param unpacked Pointer to the source memory (128 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_128_uint5_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_128_uint5_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // Pack the first 64 bytes (p0, p1, p2, p3)
  for (int i = 0; i < 16; ++i) {
    packed[i] = (unpacked[i] & 0x1F) | ((unpacked[i + 16] & 0x1F) << 5);
    packed[i + 16] = (unpacked[i + 32] & 0x1F) | ((unpacked[i + 48] & 0x1F) << 5);
    packed[i + 32] = (unpacked[i + 64] & 0x1F) | ((unpacked[i + 80] & 0x1F) << 5);
    packed[i + 48] = (unpacked[i + 96] & 0x1F) | ((unpacked[i + 112] & 0x1F) << 5);
  }

  // Pack the final 16 bytes (p4)
  for (int i = 0; i < 16; ++i) {
    uint8_t val1 = (unpacked[16 + i] >> 3) & 0x03;
    uint8_t val2 = (unpacked[48 + i] >> 3) & 0x03;
    uint8_t val3 = (unpacked[80 + i] >> 3) & 0x03;
    uint8_t val4 = (unpacked[112 + i] >> 3) & 0x03;
    packed[64 + i] = val1 | (val2 << 2) | (val3 << 4) | (val4 << 6);
  }
}

/**
 * @brief Unpacks 80 bytes into 128 bytes (each a 5-bit value).
 * @param unpacked Pointer to the destination memory (128 bytes).
 * @param packed Pointer to the source memory (80 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_128_uint5_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_128_uint5_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t p0 = packed[i];
    const uint8_t p1 = packed[i + 16];
    const uint8_t p2 = packed[i + 32];
    const uint8_t p3 = packed[i + 48];
    const uint8_t p4 = packed[i + 64];

    unpacked[i + 16 * 0] = p0 & 0x1F;
    unpacked[i + 16 * 1] = (p0 >> 5) | ((p4 & 0x03) << 3);
    unpacked[i + 16 * 2] = p1 & 0x1F;
    unpacked[i + 16 * 3] = (p1 >> 5) | ((p4 & 0x0C) << 1);
    unpacked[i + 16 * 4] = p2 & 0x1F;
    unpacked[i + 16 * 5] = (p2 >> 5) | ((p4 & 0x30) >> 1);
    unpacked[i + 16 * 6] = p3 & 0x1F;
    unpacked[i + 16 * 7] = (p3 >> 5) | ((p4 & 0xC0) >> 3);
  }
}

}}
