// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/macro.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint3.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint4.h>
#include <cassert>

namespace torchao {
namespace bitpacking {

namespace internal {
TORCHAO_ALWAYS_INLINE inline void vec_store_64_uint8_values(
    uint8_t* dest,
    const uint8x16_t& vec0,
    const uint8x16_t& vec1,
    const uint8x16_t& vec2,
    const uint8x16_t& vec3) {
  vst1q_u8(dest, vec0);
  vst1q_u8(dest + 16, vec1);
  vst1q_u8(dest + 32, vec2);
  vst1q_u8(dest + 48, vec3);
}

TORCHAO_ALWAYS_INLINE inline void vec_load_64_uint8_values(
    uint8x16_t& vec0,
    uint8x16_t& vec1,
    uint8x16_t& vec2,
    uint8x16_t& vec3,
    const uint8_t* src) {
  vec0 = vld1q_u8(src);
  vec1 = vld1q_u8(src + 16);
  vec2 = vld1q_u8(src + 32);
  vec3 = vld1q_u8(src + 48);
}
} // namespace internal

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_pack_32_lowbit_values(
    uint8_t* packed,
    const int8x16_t& unpacked0,
    const int8x16_t& unpacked1) {
  static_assert(nbit < 8);
  static_assert(nbit >= 2);

  // Currently supported values
  static_assert(nbit >= 3);
  static_assert(nbit <= 4);

  // Shift unpacked values to nonnegative range
  int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
  uint8x16_t shifted0 = vreinterpretq_u8_s8(vaddq_s8(unpacked0, shift));
  uint8x16_t shifted1 = vreinterpretq_u8_s8(vaddq_s8(unpacked1, shift));

  switch (nbit) {
    case 3:
      uint8_t buffer[32];
      vst1q_u8(buffer, shifted0);
      vst1q_u8(buffer + 16, shifted1);

      torchao::bitpacking::internal::pack_8_uint3_values(packed, buffer);
      torchao::bitpacking::internal::pack_8_uint3_values(
          packed + 3, buffer + 8);
      torchao::bitpacking::internal::pack_8_uint3_values(
          packed + 6, buffer + 16);
      torchao::bitpacking::internal::pack_8_uint3_values(
          packed + 9, buffer + 24);
      break;
    case 4:
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed, shifted0, shifted1);
      break;
    default:
      assert(false);
  }
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_unpack_32_lowbit_values(
    int8x16_t& unpacked0,
    int8x16_t& unpacked1,
    uint8_t* packed) {
  static_assert(nbit < 8);
  static_assert(nbit >= 2);

  // Currently supported values
  static_assert(nbit >= 3);
  static_assert(nbit <= 4);

  uint8x16_t shifted0;
  uint8x16_t shifted1;

  switch (nbit) {
    case 3:
      uint8_t buffer[32];
      torchao::bitpacking::internal::unpack_8_uint3_values(buffer, packed);
      torchao::bitpacking::internal::unpack_8_uint3_values(
          buffer + 8, packed + 3);
      torchao::bitpacking::internal::unpack_8_uint3_values(
          buffer + 16, packed + 6);
      torchao::bitpacking::internal::unpack_8_uint3_values(
          buffer + 24, packed + 9);
      shifted0 = vld1q_u8(buffer);
      shifted1 = vld1q_u8(buffer + 16);
      break;
    case 4:
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          shifted0, shifted1, packed);
      break;
    default:
      assert(false);
  }

  // unshift to move unpacked values to full range
  int8x16_t unshift = vdupq_n_s8(-(1 << (nbit - 1)));
  unpacked0 = vaddq_s8(vreinterpretq_s8_u8(shifted0), unshift);
  unpacked1 = vaddq_s8(vreinterpretq_s8_u8(shifted1), unshift);
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_pack_64_lowbit_values(
    uint8_t* packed,
    const int8x16_t& unpacked0,
    const int8x16_t& unpacked1,
    const int8x16_t& unpacked2,
    const int8x16_t& unpacked3) {
  static_assert(nbit < 8);
  static_assert(nbit >= 2);

  // Currently supported values
  static_assert(nbit >= 3);
  static_assert(nbit <= 4);

  // Shift unpacked values to nonnegative range
  int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
  uint8x16_t shifted0 = vreinterpretq_u8_s8(vaddq_s8(unpacked0, shift));
  uint8x16_t shifted1 = vreinterpretq_u8_s8(vaddq_s8(unpacked1, shift));
  uint8x16_t shifted2 = vreinterpretq_u8_s8(vaddq_s8(unpacked2, shift));
  uint8x16_t shifted3 = vreinterpretq_u8_s8(vaddq_s8(unpacked3, shift));

  switch (nbit) {
    case 3:
      torchao::bitpacking::internal::vec_pack_64_uint3_values(
          packed, shifted0, shifted1, shifted2, shifted3);
      break;
    case 4:
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed, shifted0, shifted1);
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed + 16, shifted2, shifted3);
      break;
    default:
      assert(false);
  }
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_lowbit_values(
    int8x16_t& unpacked0,
    int8x16_t& unpacked1,
    int8x16_t& unpacked2,
    int8x16_t& unpacked3,
    uint8_t* packed) {
  static_assert(nbit < 8);
  static_assert(nbit >= 2);

  // Currently supported values
  static_assert(nbit >= 3);
  static_assert(nbit <= 4);

  uint8x16_t shifted0;
  uint8x16_t shifted1;
  uint8x16_t shifted2;
  uint8x16_t shifted3;

  switch (nbit) {
    case 3:
      torchao::bitpacking::internal::vec_unpack_64_uint3_values(
          shifted0, shifted1, shifted2, shifted3, packed);
      break;
    case 4:
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          shifted0, shifted1, packed);
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          shifted2, shifted3, packed + 16);
      break;
    default:
      assert(false);
  }

  // unshift to move unpacked values to full range
  int8x16_t unshift = vdupq_n_s8(-(1 << (nbit - 1)));
  unpacked0 = vaddq_s8(vreinterpretq_s8_u8(shifted0), unshift);
  unpacked1 = vaddq_s8(vreinterpretq_s8_u8(shifted1), unshift);
  unpacked2 = vaddq_s8(vreinterpretq_s8_u8(shifted2), unshift);
  unpacked3 = vaddq_s8(vreinterpretq_s8_u8(shifted3), unshift);
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_pack_128_lowbit_values(
    uint8_t* packed,
    const int8x16_t& unpacked0,
    const int8x16_t& unpacked1,
    const int8x16_t& unpacked2,
    const int8x16_t& unpacked3,
    const int8x16_t& unpacked4,
    const int8x16_t& unpacked5,
    const int8x16_t& unpacked6,
    const int8x16_t& unpacked7) {
  static_assert(nbit < 8);
  static_assert(nbit >= 2);

  // Currently supported values
  static_assert(nbit >= 3);
  static_assert(nbit <= 4);

  // Shift unpacked values to nonnegative range
  int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
  uint8x16_t shifted0 = vreinterpretq_u8_s8(vaddq_s8(unpacked0, shift));
  uint8x16_t shifted1 = vreinterpretq_u8_s8(vaddq_s8(unpacked1, shift));
  uint8x16_t shifted2 = vreinterpretq_u8_s8(vaddq_s8(unpacked2, shift));
  uint8x16_t shifted3 = vreinterpretq_u8_s8(vaddq_s8(unpacked3, shift));
  uint8x16_t shifted4 = vreinterpretq_u8_s8(vaddq_s8(unpacked4, shift));
  uint8x16_t shifted5 = vreinterpretq_u8_s8(vaddq_s8(unpacked5, shift));
  uint8x16_t shifted6 = vreinterpretq_u8_s8(vaddq_s8(unpacked6, shift));
  uint8x16_t shifted7 = vreinterpretq_u8_s8(vaddq_s8(unpacked7, shift));

  switch (nbit) {
    case 3:
      torchao::bitpacking::internal::vec_pack_128_uint3_values(
          packed,
          shifted0,
          shifted1,
          shifted2,
          shifted3,
          shifted4,
          shifted5,
          shifted6,
          shifted7);
      break;
    case 4:
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed, shifted0, shifted1);
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed + 16, shifted2, shifted3);
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed + 32, shifted4, shifted5);
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed + 48, shifted6, shifted7);
      break;
    default:
      assert(false);
  }
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_unpack_128_lowbit_values(
    int8x16_t& unpacked0,
    int8x16_t& unpacked1,
    int8x16_t& unpacked2,
    int8x16_t& unpacked3,
    int8x16_t& unpacked4,
    int8x16_t& unpacked5,
    int8x16_t& unpacked6,
    int8x16_t& unpacked7,
    uint8_t* packed) {
  static_assert(nbit < 8);
  static_assert(nbit >= 2);

  // Currently supported values
  static_assert(nbit >= 3);
  static_assert(nbit <= 4);

  uint8x16_t shifted0;
  uint8x16_t shifted1;
  uint8x16_t shifted2;
  uint8x16_t shifted3;
  uint8x16_t shifted4;
  uint8x16_t shifted5;
  uint8x16_t shifted6;
  uint8x16_t shifted7;

  switch (nbit) {
    case 3:
      torchao::bitpacking::internal::vec_unpack_128_uint3_values(
          shifted0,
          shifted1,
          shifted2,
          shifted3,
          shifted4,
          shifted5,
          shifted6,
          shifted7,
          packed);
      break;
    case 4:
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          shifted0, shifted1, packed);
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          shifted2, shifted3, packed + 16);
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          shifted4, shifted5, packed + 32);
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          shifted6, shifted7, packed + 48);
      break;
    default:
      assert(false);
  }

  // unshift to move unpacked values to full range
  int8x16_t unshift = vdupq_n_s8(-(1 << (nbit - 1)));
  unpacked0 = vaddq_s8(vreinterpretq_s8_u8(shifted0), unshift);
  unpacked1 = vaddq_s8(vreinterpretq_s8_u8(shifted1), unshift);
  unpacked2 = vaddq_s8(vreinterpretq_s8_u8(shifted2), unshift);
  unpacked3 = vaddq_s8(vreinterpretq_s8_u8(shifted3), unshift);
  unpacked4 = vaddq_s8(vreinterpretq_s8_u8(shifted4), unshift);
  unpacked5 = vaddq_s8(vreinterpretq_s8_u8(shifted5), unshift);
  unpacked6 = vaddq_s8(vreinterpretq_s8_u8(shifted6), unshift);
  unpacked7 = vaddq_s8(vreinterpretq_s8_u8(shifted7), unshift);
}

} // namespace bitpacking
} // namespace torchao
