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
 * @brief Packs 8 bytes, each containing a 1-bit value (0 or 1), into a single
 * byte.
 * @param packed Pointer to the destination memory (1 byte).
 * @param unpacked Pointer to the source memory (8 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void pack_8_uint1_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  packed[0] = (unpacked[0] << 7) | (unpacked[1] << 6) | (unpacked[2] << 5) |
      (unpacked[3] << 4) | (unpacked[4] << 3) | (unpacked[5] << 2) |
      (unpacked[6] << 1) | (unpacked[7] << 0);
}

/**
 * @brief Unpacks a single byte into 8 bytes, each containing a 1-bit value.
 * @param unpacked Pointer to the destination memory (8 bytes).
 * @param packed Pointer to the source memory (1 byte).
 */
TORCHAO_ALWAYS_INLINE inline void unpack_8_uint1_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  const uint8_t packed_byte = packed[0];
  unpacked[0] = (packed_byte >> 7) & 1;
  unpacked[1] = (packed_byte >> 6) & 1;
  unpacked[2] = (packed_byte >> 5) & 1;
  unpacked[3] = (packed_byte >> 4) & 1;
  unpacked[4] = (packed_byte >> 3) & 1;
  unpacked[5] = (packed_byte >> 2) & 1;
  unpacked[6] = (packed_byte >> 1) & 1;
  unpacked[7] = (packed_byte >> 0) & 1;
}

/**
 * @brief Packs 64 bytes (each a 1-bit value) into 8 bytes.
 * @param packed Pointer to the destination memory (8 bytes).
 * @param unpacked Pointer to the source memory (64 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_64_uint1_values` function to ensure compatibility. The unpacked
 * data is assumed to be organized as four 16-byte blocks.
 */
TORCHAO_ALWAYS_INLINE inline void pack_64_uint1_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  const uint8_t* unpacked0 = unpacked;
  const uint8_t* unpacked1 = unpacked + 16;
  const uint8_t* unpacked2 = unpacked + 32;
  const uint8_t* unpacked3 = unpacked + 48;

  for (int i = 0; i < 8; ++i) {
    // Combine 4 bits for the low nibble of the output byte
    uint8_t low_nibble = (unpacked0[i] << 3) | (unpacked1[i] << 2) |
        (unpacked2[i] << 1) | (unpacked3[i] << 0);

    // Combine 4 bits for the high nibble of the output byte
    uint8_t high_nibble_src = (unpacked0[i + 8] << 3) |
        (unpacked1[i + 8] << 2) | (unpacked2[i + 8] << 1) |
        (unpacked3[i + 8] << 0);

    // Assemble the final byte
    packed[i] = low_nibble | (high_nibble_src << 4);
  }
}

/**
 * @brief Unpacks 8 bytes into 64 bytes (each a 1-bit value).
 * @param unpacked Pointer to the destination memory (64 bytes).
 * @param packed Pointer to the source memory (8 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_64_uint1_values` function to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_64_uint1_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  uint8_t* unpacked0 = unpacked;
  uint8_t* unpacked1 = unpacked + 16;
  uint8_t* unpacked2 = unpacked + 32;
  uint8_t* unpacked3 = unpacked + 48;

  uint8_t combined[16];
  for (int i = 0; i < 8; ++i) {
    combined[i] = packed[i] & 0x0F; // Low nibbles
    combined[i + 8] = packed[i] >> 4; // High nibbles
  }

  // Unpack from the combined buffer into the four destination blocks
  for (int i = 0; i < 16; ++i) {
    const uint8_t temp = combined[i];
    unpacked0[i] = (temp >> 3) & 1;
    unpacked1[i] = (temp >> 2) & 1;
    unpacked2[i] = (temp >> 1) & 1;
    unpacked3[i] = (temp >> 0) & 1;
  }
}

/**
 * @brief Packs 128 bytes (each a 1-bit value) into 16 bytes.
 * @param packed Pointer to the destination memory (16 bytes).
 * @param unpacked Pointer to the source memory (128 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_128_uint1_values` function (a transpose-and-pack operation) to
 * ensure compatibility. The unpacked data is assumed to be organized as eight
 * 16-byte blocks.
 */
TORCHAO_ALWAYS_INLINE inline void pack_128_uint1_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 16; ++i) {
    packed[i] = (unpacked[i + 16 * 0] << 7) | (unpacked[i + 16 * 1] << 6) |
        (unpacked[i + 16 * 2] << 5) | (unpacked[i + 16 * 3] << 4) |
        (unpacked[i + 16 * 4] << 3) | (unpacked[i + 16 * 5] << 2) |
        (unpacked[i + 16 * 6] << 1) | (unpacked[i + 16 * 7] << 0);
  }
}

/**
 * @brief Unpacks 16 bytes into 128 bytes (each a 1-bit value).
 * @param unpacked Pointer to the destination memory (128 bytes).
 * @param packed Pointer to the source memory (16 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_128_uint1_values` function (an unpack-and-transpose operation)
 * to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_128_uint1_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t packed_byte = packed[i];
    unpacked[i + 16 * 0] = (packed_byte >> 7) & 1;
    unpacked[i + 16 * 1] = (packed_byte >> 6) & 1;
    unpacked[i + 16 * 2] = (packed_byte >> 5) & 1;
    unpacked[i + 16 * 3] = (packed_byte >> 4) & 1;
    unpacked[i + 16 * 4] = (packed_byte >> 3) & 1;
    unpacked[i + 16 * 5] = (packed_byte >> 2) & 1;
    unpacked[i + 16 * 6] = (packed_byte >> 1) & 1;
    unpacked[i + 16 * 7] = (packed_byte >> 0) & 1;
  }
}
} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
