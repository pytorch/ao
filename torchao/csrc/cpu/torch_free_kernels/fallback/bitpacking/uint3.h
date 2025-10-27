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
 * @brief Packs 8 bytes, each holding a 3-bit value (0-7), into 3 bytes.
 *
 * The packing scheme is non-trivial. Given 8 input values v0..v7, they are
 * arranged into 3 bytes (b0, b1, b2) as follows:
 * - b0: [v6(low 2 bits), v0(all 3 bits), v1(all 3 bits)]
 * - b1: [v7(low 2 bits), v2(all 3 bits), v3(all 3 bits)]
 * - b2: [v6(high 1 bit), v7(high 1 bit), v4(all 3 bits), v5(all 3 bits)]
 *
 * @param packed Pointer to the destination memory (3 bytes).
 * @param unpacked Pointer to the source memory (8 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void pack_8_uint3_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // byte 0
  packed[0] = ((unpacked[6] & 0x03) << 6) | ((unpacked[0] & 0x07) << 3) |
      (unpacked[1] & 0x07);

  // byte 1
  packed[1] = ((unpacked[7] & 0x03) << 6) | ((unpacked[2] & 0x07) << 3) |
      (unpacked[3] & 0x07);

  // byte 2
  packed[2] = ((unpacked[6] & 0x04) << 5) | ((unpacked[7] & 0x04) << 4) |
      ((unpacked[4] & 0x07) << 3) | (unpacked[5] & 0x07);
}

/**
 * @brief Unpacks 3 bytes into 8 bytes, each containing a 3-bit value.
 * @param unpacked Pointer to the destination memory (8 bytes).
 * @param packed Pointer to the source memory (3 bytes).
 */
TORCHAO_ALWAYS_INLINE inline void unpack_8_uint3_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  const uint8_t b0 = packed[0];
  const uint8_t b1 = packed[1];
  const uint8_t b2 = packed[2];

  unpacked[0] = (b0 >> 3) & 0x07;
  unpacked[1] = b0 & 0x07;

  unpacked[2] = (b1 >> 3) & 0x07;
  unpacked[3] = b1 & 0x07;

  unpacked[4] = (b2 >> 3) & 0x07;
  unpacked[5] = b2 & 0x07;

  unpacked[6] = (b0 >> 6) | ((b2 >> 5) & 0x04);
  unpacked[7] = (b1 >> 6) | ((b2 >> 4) & 0x04);
}

/**
 * @brief Packs 64 bytes (each a 3-bit value) into 24 bytes.
 * @param packed Pointer to the destination memory (24 bytes).
 * @param unpacked Pointer to the source memory (64 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_64_uint3_values` function (a transpose-and-pack operation) to
 * ensure compatibility. The unpacked data is assumed to be organized as eight
 * 8-byte blocks.
 */
TORCHAO_ALWAYS_INLINE inline void pack_64_uint3_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 8; ++i) {
    const uint8_t unpacked0 = unpacked[i + 8 * 0];
    const uint8_t unpacked1 = unpacked[i + 8 * 1];
    const uint8_t unpacked2 = unpacked[i + 8 * 2];
    const uint8_t unpacked3 = unpacked[i + 8 * 3];
    const uint8_t unpacked4 = unpacked[i + 8 * 4];
    const uint8_t unpacked5 = unpacked[i + 8 * 5];
    const uint8_t unpacked6 = unpacked[i + 8 * 6];
    const uint8_t unpacked7 = unpacked[i + 8 * 7];

    // byte 0
    packed[i] = ((unpacked6 & 0x03) << 6) | ((unpacked0 & 0x07) << 3) |
        (unpacked1 & 0x07);

    // byte 1
    packed[i + 8] = ((unpacked7 & 0x03) << 6) | ((unpacked2 & 0x07) << 3) |
        (unpacked3 & 0x07);

    // byte 2
    packed[i + 16] = ((unpacked6 & 0x04) << 5) | ((unpacked7 & 0x04) << 4) |
        ((unpacked4 & 0x07) << 3) | (unpacked5 & 0x07);
  }
}

/**
 * @brief Unpacks 24 bytes into 64 bytes (each a 3-bit value).
 * @param unpacked Pointer to the destination memory (64 bytes).
 * @param packed Pointer to the source memory (24 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_64_uint3_values` function (an unpack-and-transpose operation)
 * to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_64_uint3_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 8; ++i) {
    const uint8_t b0 = packed[i];
    const uint8_t b1 = packed[i + 8];
    const uint8_t b2 = packed[i + 16];

    unpacked[i + 8 * 0] = (b0 >> 3) & 0x07;
    unpacked[i + 8 * 1] = b0 & 0x07;
    unpacked[i + 8 * 2] = (b1 >> 3) & 0x07;
    unpacked[i + 8 * 3] = b1 & 0x07;
    unpacked[i + 8 * 4] = (b2 >> 3) & 0x07;
    unpacked[i + 8 * 5] = b2 & 0x07;
    unpacked[i + 8 * 6] = (b0 >> 6) | ((b2 >> 5) & 0x04);
    unpacked[i + 8 * 7] = (b1 >> 6) | ((b2 >> 4) & 0x04);
  }
}

/**
 * @brief Packs 128 bytes (each a 3-bit value) into 48 bytes.
 * @param packed Pointer to the destination memory (48 bytes).
 * @param unpacked Pointer to the source memory (128 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_pack_128_uint3_values` function (a transpose-and-pack operation) to
 * ensure compatibility. The unpacked data is assumed to be organized as eight
 * 16-byte blocks.
 */
TORCHAO_ALWAYS_INLINE inline void pack_128_uint3_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t unpacked0 = unpacked[i + 16 * 0];
    const uint8_t unpacked1 = unpacked[i + 16 * 1];
    const uint8_t unpacked2 = unpacked[i + 16 * 2];
    const uint8_t unpacked3 = unpacked[i + 16 * 3];
    const uint8_t unpacked4 = unpacked[i + 16 * 4];
    const uint8_t unpacked5 = unpacked[i + 16 * 5];
    const uint8_t unpacked6 = unpacked[i + 16 * 6];
    const uint8_t unpacked7 = unpacked[i + 16 * 7];

    // byte 0
    packed[i] = ((unpacked6 & 0x03) << 6) | ((unpacked0 & 0x07) << 3) |
        (unpacked1 & 0x07);

    // byte 1
    packed[i + 16] = ((unpacked7 & 0x03) << 6) | ((unpacked2 & 0x07) << 3) |
        (unpacked3 & 0x07);

    // byte 2
    packed[i + 32] = ((unpacked6 & 0x04) << 5) | ((unpacked7 & 0x04) << 4) |
        ((unpacked4 & 0x07) << 3) | (unpacked5 & 0x07);
  }
}

/**
 * @brief Unpacks 48 bytes into 128 bytes (each a 3-bit value).
 * @param unpacked Pointer to the destination memory (128 bytes).
 * @param packed Pointer to the source memory (48 bytes).
 * @note This implementation mirrors the logic of the ARM NEON
 * `vec_unpack_128_uint3_values` function (an unpack-and-transpose operation)
 * to ensure compatibility.
 */
TORCHAO_ALWAYS_INLINE inline void unpack_128_uint3_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  for (int i = 0; i < 16; ++i) {
    const uint8_t b0 = packed[i];
    const uint8_t b1 = packed[i + 16];
    const uint8_t b2 = packed[i + 32];

    unpacked[i + 16 * 0] = (b0 >> 3) & 0x07;
    unpacked[i + 16 * 1] = b0 & 0x07;
    unpacked[i + 16 * 2] = (b1 >> 3) & 0x07;
    unpacked[i + 16 * 3] = b1 & 0x07;
    unpacked[i + 16 * 4] = (b2 >> 3) & 0x07;
    unpacked[i + 16 * 5] = b2 & 0x07;
    unpacked[i + 16 * 6] = (b0 >> 6) | ((b2 >> 5) & 0x04);
    unpacked[i + 16 * 7] = (b1 >> 6) | ((b2 >> 4) & 0x04);
  }
}

} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
