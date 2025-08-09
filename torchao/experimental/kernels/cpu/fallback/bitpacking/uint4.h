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
 * @brief Packs 2 bytes, each holding a 4-bit value (0-15), into a single
 * byte. The first value goes into the high nibble, the second into the low
 * nibble.
 * @param packed Pointer to the destination memory (1 byte).
 * @param unpacked Pointer to the source memory (2 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void pack_2_uint4_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // This is compatible with the scalar NEON version.
  packed[0] = (unpacked[0] << 4) | (unpacked[1] & 0x0F);
}

/**
 * @brief Unpacks a single byte into 2 bytes, each containing a 4-bit value.
 * @param unpacked Pointer to the destination memory (2 bytes).
 * @param packed Pointer to the source memory (1 byte).
 */
TORCHAO_ALWAYS_INLINE inline void unpack_2_uint4_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // This is compatible with the scalar NEON version.
  unpacked[0] = packed[0] >> 4;
  unpacked[1] = packed[0] & 0x0F;
}

/**
 * @brief Packs 16 bytes (each a 4-bit value) into 8 bytes.
 * @param packed Pointer to the destination memory (8 bytes).
 * @param unpacked Pointer to the source memory (16 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_16_uint4_values` function (a transpose-and-pack operation) to
 * ensure compatibility. It packs unpacked[i] and unpacked[i+8] into
 * packed[i].
 */
TORCHAO_ALWAYS_INLINE inline void pack_16_uint4_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 8; ++i) {
    packed[i] = ((unpacked[i + 8] & 0x0F) << 4) | (unpacked[i] & 0x0F);
  }
}

/**
 * @brief Unpacks 8 bytes into 16 bytes (each a 4-bit value).
 * @param unpacked Pointer to the destination memory (16 bytes).
 * @param packed Pointer to the source memory (8 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_16_uint4_values` function (an unpack-and-transpose operation)
 * to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_16_uint4_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 8; ++i) {
    unpacked[i] = packed[i] & 0x0F;
    unpacked[i + 8] = packed[i] >> 4;
  }
}

/**
 * @brief Packs 32 bytes (each a 4-bit value) into 16 bytes.
 * @param packed Pointer to the destination memory (16 bytes).
 * @param unpacked Pointer to the source memory (32 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_32_uint4_values` function (a transpose-and-pack operation) to
 * ensure compatibility. It packs unpacked[i] and unpacked[i+16] into
 * packed[i].
 */
TORCHAO_ALWAYS_INLINE inline void pack_32_uint4_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 16; ++i) {
    packed[i] = ((unpacked[i + 16] & 0x0F) << 4) | (unpacked[i] & 0x0F);
  }
}

/**
 * @brief Unpacks 16 bytes into 32 bytes (each a 4-bit value).
 * @param unpacked Pointer to the destination memory (32 bytes).
 * @param packed Pointer to the source memory (16 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_32_uint4_values` function (an unpack-and-transpose operation)
 * to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_32_uint4_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 16; ++i) {
    unpacked[i] = packed[i] & 0x0F;
    unpacked[i + 16] = packed[i] >> 4;
  }
}
} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
