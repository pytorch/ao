// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <cstdint>

namespace torchao::kernels::cpu::fallback::bitpacking {
namespace internal {

/**
 * @brief Packs 4 bytes, each holding a 6-bit value (0-63), into 3 bytes.
 *
 * @param packed Pointer to the destination memory (3 bytes).
 * @param unpacked Pointer to the source memory (4 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void pack_4_uint6_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // pack 4 uint6 values (u0..u3) into 3 bytes (p0..p2)
  // p0's low 6 bits = u0; p0's high 2 bits = u3's low 2 bits
  // p1's low 6 bits = u1; p1's high 2 bits = u3's mid 2 bits
  // p2's low 6 bits = u2; p2's high 2 bits = u3's high 2 bits
  const uint8_t u3 = unpacked[3] & 0x3F;
  packed[0] = (unpacked[0] & 0x3F) | ((u3 & 0x03) << 6);
  packed[1] = (unpacked[1] & 0x3F) | ((u3 & 0x0C) << 4);
  packed[2] = (unpacked[2] & 0x3F) | ((u3 & 0x30) << 2);
}

/**
 * @brief Unpacks 3 bytes into 4 bytes, each containing a 6-bit value.
 *
 * @param unpacked Pointer to the destination memory (4 bytes).
 * @param packed Pointer to the source memory (3 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void unpack_4_uint6_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // This is compatible with the scalar NEON version.
  unpacked[0] = packed[0] & 0x3F;
  unpacked[1] = packed[1] & 0x3F;
  unpacked[2] = packed[2] & 0x3F;
  unpacked[3] = ((packed[0] & 0xC0) >> 6) | ((packed[1] & 0xC0) >> 4) |
      ((packed[2] & 0xC0) >> 2);
}

/**
 * @brief Packs 32 bytes (each a 6-bit value) into 24 bytes.
 * @param packed Pointer to the destination memory (24 bytes).
 * @param unpacked Pointer to the source memory (32 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_32_uint6_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_32_uint6_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 8; ++i) {
    const uint8_t u0 = unpacked[i];
    const uint8_t u1 = unpacked[i + 8];
    const uint8_t u2 = unpacked[i + 16];
    const uint8_t u3 = unpacked[i + 24];

    packed[i] = (u0 & 0x3F) | ((u3 & 0x03) << 6);
    packed[i + 8] = (u1 & 0x3F) | ((u3 & 0x0C) << 4);
    packed[i + 16] = (u2 & 0x3F) | ((u3 & 0x30) << 2);
  }
}

/**
 * @brief Unpacks 24 bytes into 32 bytes (each a 6-bit value).
 * @param unpacked Pointer to the destination memory (32 bytes).
 * @param packed Pointer to the source memory (24 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_32_uint6_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_32_uint6_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 8; ++i) {
    const uint8_t p0 = packed[i];
    const uint8_t p1 = packed[i + 8];
    const uint8_t p2 = packed[i + 16];

    unpacked[i] = p0 & 0x3F;
    unpacked[i + 8] = p1 & 0x3F;
    unpacked[i + 16] = p2 & 0x3F;
    unpacked[i + 24] =
        ((p0 & 0xC0) >> 6) | ((p1 & 0xC0) >> 4) | ((p2 & 0xC0) >> 2);
  }
}

/**
 * @brief Packs 64 bytes (each a 6-bit value) into 48 bytes.
 * @param packed Pointer to the destination memory (48 bytes).
 * @param unpacked Pointer to the source memory (64 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_64_uint6_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void pack_64_uint6_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t u0 = unpacked[i];
    const uint8_t u1 = unpacked[i + 16];
    const uint8_t u2 = unpacked[i + 32];
    const uint8_t u3 = unpacked[i + 48];

    packed[i] = (u0 & 0x3F) | ((u3 & 0x03) << 6);
    packed[i + 16] = (u1 & 0x3F) | ((u3 & 0x0C) << 4);
    packed[i + 32] = (u2 & 0x3F) | ((u3 & 0x30) << 2);
  }
}

/**
 * @brief Unpacks 48 bytes into 64 bytes (each a 6-bit value).
 * @param unpacked Pointer to the destination memory (64 bytes).
 * @param packed Pointer to the source memory (48 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_64_uint6_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_64_uint6_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t p0 = packed[i];
    const uint8_t p1 = packed[i + 16];
    const uint8_t p2 = packed[i + 32];

    unpacked[i] = p0 & 0x3F;
    unpacked[i + 16] = p1 & 0x3F;
    unpacked[i + 32] = p2 & 0x3F;
    unpacked[i + 48] =
        ((p0 & 0xC0) >> 6) | ((p1 & 0xC0) >> 4) | ((p2 & 0xC0) >> 2);
  }
}

} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
