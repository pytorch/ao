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
 * @brief Packs 4 bytes, each containing a 2-bit value (0-3), into a single
 * byte.
 * @param packed Pointer to the destination memory (1 byte).
 * @param unpacked Pointer to the source memory (4 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void pack_4_uint2_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // unpacked = {v0, v1, v2, v3} -> packed[0] = | v0 | v1 | v2 | v3 |
  packed[0] = (unpacked[0] << 6) | (unpacked[1] << 4) | (unpacked[2] << 2) |
      (unpacked[3]);
}

/**
 * @brief Unpacks a single byte into 4 bytes, each containing a 2-bit value.
 * @param unpacked Pointer to the destination memory (4 bytes).
 * @param packed Pointer to the source memory (1 byte).
 */
TORCHAO_ALWAYS_INLINE inline void unpack_4_uint2_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  unpacked[0] = (packed[0] >> 6) & 0x03; // Mask 0b11000000
  unpacked[1] = (packed[0] >> 4) & 0x03; // Mask 0b00110000
  unpacked[2] = (packed[0] >> 2) & 0x03; // Mask 0b00001100
  unpacked[3] = packed[0] & 0x03; // Mask 0b00000011
}

/**
 * @brief Packs 32 bytes (each a 2-bit value) into 8 bytes.
 * @param packed Pointer to the destination memory (8 bytes).
 * @param unpacked Pointer to the source memory (32 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_32_uint2_values` function (a transpose-and-pack operation) to
 * ensure compatibility. The unpacked data is assumed to be organized as four
 * 8-byte blocks.
 */
TORCHAO_ALWAYS_INLINE inline void pack_32_uint2_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 8; ++i) {
    packed[i] = (unpacked[i + 8 * 0] << 6) | (unpacked[i + 8 * 1] << 4) |
        (unpacked[i + 8 * 2] << 2) | (unpacked[i + 8 * 3] << 0);
  }
}

/**
 * @brief Unpacks 8 bytes into 32 bytes (each a 2-bit value).
 * @param unpacked Pointer to the destination memory (32 bytes).
 * @param packed Pointer to the source memory (8 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_32_uint2_values` function (an unpack-and-transpose operation)
 * to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_32_uint2_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 8; ++i) {
    const uint8_t packed_byte = packed[i];
    unpacked[i + 8 * 0] = (packed_byte >> 6) & 0x03;
    unpacked[i + 8 * 1] = (packed_byte >> 4) & 0x03;
    unpacked[i + 8 * 2] = (packed_byte >> 2) & 0x03;
    unpacked[i + 8 * 3] = (packed_byte >> 0) & 0x03;
  }
}

/**
 * @brief Packs 64 bytes (each a 2-bit value) into 16 bytes.
 * @param packed Pointer to the destination memory (16 bytes).
 * @param unpacked Pointer to the source memory (64 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_64_uint2_values` function (a transpose-and-pack operation) to
 * ensure compatibility. The unpacked data is assumed to be organized as four
 * 16-byte blocks.
 */
TORCHAO_ALWAYS_INLINE inline void pack_64_uint2_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 16; ++i) {
    packed[i] = (unpacked[i + 16 * 0] << 6) | (unpacked[i + 16 * 1] << 4) |
        (unpacked[i + 16 * 2] << 2) | (unpacked[i + 16 * 3] << 0);
  }
}

/**
 * @brief Unpacks 16 bytes into 64 bytes (each a 2-bit value).
 * @param unpacked Pointer to the destination memory (64 bytes).
 * @param packed Pointer to the source memory (16 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_64_uint2_values` function (an unpack-and-transpose operation)
 * to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_64_uint2_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t packed_byte = packed[i];
    unpacked[i + 16 * 0] = (packed_byte >> 6) & 0x03;
    unpacked[i + 16 * 1] = (packed_byte >> 4) & 0x03;
    unpacked[i + 16 * 2] = (packed_byte >> 2) & 0x03;
    unpacked[i + 16 * 3] = (packed_byte >> 0) & 0x03;
  }
}

} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
