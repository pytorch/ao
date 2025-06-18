// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint1.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint2.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint3.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint4.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint5.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint6.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint7.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <cassert>

namespace torchao {
namespace bitpacking {

namespace internal {
TORCHAO_ALWAYS_INLINE inline void vec_store_32_uint8_values(
    uint8_t* dest,
    const uint8x8_t& vec0,
    const uint8x8_t& vec1,
    const uint8x8_t& vec2,
    const uint8x8_t& vec3) {
  vst1_u8(dest, vec0);
  vst1_u8(dest + 8, vec1);
  vst1_u8(dest + 16, vec2);
  vst1_u8(dest + 24, vec3);
}

TORCHAO_ALWAYS_INLINE inline void vec_load_32_uint8_values(
    uint8x8_t& vec0,
    uint8x8_t& vec1,
    uint8x8_t& vec2,
    uint8x8_t& vec3,
    const uint8_t* src) {
  vec0 = vld1_u8(src);
  vec1 = vld1_u8(src + 8);
  vec2 = vld1_u8(src + 16);
  vec3 = vld1_u8(src + 24);
}

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
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  // Shift unpacked values to nonnegative range for quantization of 1-7 bits
  // No shifting is needed for 8-bit packing
  uint8x16_t shifted0;
  uint8x16_t shifted1;
  if constexpr (nbit < 8) {
    int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
    shifted0 = vreinterpretq_u8_s8(vaddq_s8(unpacked0, shift));
    shifted1 = vreinterpretq_u8_s8(vaddq_s8(unpacked1, shift));
  }

  switch (nbit) {
    case 1:
      uint8_t buffer1[32];
      vst1q_u8(buffer1, shifted0);
      vst1q_u8(buffer1 + 16, shifted1);

      torchao::bitpacking::internal::pack_8_uint1_values(packed, buffer1);
      torchao::bitpacking::internal::pack_8_uint1_values(
          packed + 1, buffer1 + 8);
      torchao::bitpacking::internal::pack_8_uint1_values(
          packed + 2, buffer1 + 16);
      torchao::bitpacking::internal::pack_8_uint1_values(
          packed + 3, buffer1 + 24);
      break;
    case 2:
      torchao::bitpacking::internal::vec_pack_32_uint2_values(
          packed,
          vget_low_u8(shifted0),
          vget_high_u8(shifted0),
          vget_low_u8(shifted1),
          vget_high_u8(shifted1));
      break;
    case 3:
      uint8_t buffer3[32];
      vst1q_u8(buffer3, shifted0);
      vst1q_u8(buffer3 + 16, shifted1);

      torchao::bitpacking::internal::pack_8_uint3_values(packed, buffer3);
      torchao::bitpacking::internal::pack_8_uint3_values(
          packed + 3, buffer3 + 8);
      torchao::bitpacking::internal::pack_8_uint3_values(
          packed + 6, buffer3 + 16);
      torchao::bitpacking::internal::pack_8_uint3_values(
          packed + 9, buffer3 + 24);
      break;
    case 4:
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed, shifted0, shifted1);
      break;
    case 5:
      uint8_t buffer5[32];
      vst1q_u8(buffer5, shifted0);
      vst1q_u8(buffer5 + 16, shifted1);

      torchao::bitpacking::internal::pack_8_uint5_values(packed, buffer5);
      torchao::bitpacking::internal::pack_8_uint5_values(
          packed + 5, buffer5 + 8);
      torchao::bitpacking::internal::pack_8_uint5_values(
          packed + 10, buffer5 + 16);
      torchao::bitpacking::internal::pack_8_uint5_values(
          packed + 15, buffer5 + 24);
      break;
    case 6:
      torchao::bitpacking::internal::vec_pack_32_uint6_values(
          packed, shifted0, shifted1);
      break;
    case 7:
      uint8_t buffer7[32];
      vst1q_u8(buffer7, shifted0);
      vst1q_u8(buffer7 + 16, shifted1);

      torchao::bitpacking::internal::pack_8_uint7_values(packed, buffer7);
      torchao::bitpacking::internal::pack_8_uint7_values(
          packed + 7, buffer7 + 8);
      torchao::bitpacking::internal::pack_8_uint7_values(
          packed + 14, buffer7 + 16);
      torchao::bitpacking::internal::pack_8_uint7_values(
          packed + 21, buffer7 + 24);
      break;
    case 8:
      vst1q_u8(packed, vreinterpretq_u8_s8(unpacked0));
      vst1q_u8(packed + 16, vreinterpretq_u8_s8(unpacked1));
      break;
    default:
      assert(false);
  }
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_unpack_32_lowbit_values(
    int8x16_t& unpacked0,
    int8x16_t& unpacked1,
    const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  uint8x16_t shifted0;
  uint8x16_t shifted1;

  switch (nbit) {
    case 1:
      uint8_t buffer1[32];
      torchao::bitpacking::internal::unpack_8_uint1_values(buffer1, packed);
      torchao::bitpacking::internal::unpack_8_uint1_values(
          buffer1 + 8, packed + 1);
      torchao::bitpacking::internal::unpack_8_uint1_values(
          buffer1 + 16, packed + 2);
      torchao::bitpacking::internal::unpack_8_uint1_values(
          buffer1 + 24, packed + 3);
      shifted0 = vld1q_u8(buffer1);
      shifted1 = vld1q_u8(buffer1 + 16);
      break;
    case 2:
      uint8x8_t shifted0_low;
      uint8x8_t shifted0_high;
      uint8x8_t shifted1_low;
      uint8x8_t shifted1_high;
      torchao::bitpacking::internal::vec_unpack_32_uint2_values(
          shifted0_low, shifted0_high, shifted1_low, shifted1_high, packed);
      shifted0 = vcombine_u8(shifted0_low, shifted0_high);
      shifted1 = vcombine_u8(shifted1_low, shifted1_high);
      break;
    case 3:
      uint8_t buffer3[32];
      torchao::bitpacking::internal::unpack_8_uint3_values(buffer3, packed);
      torchao::bitpacking::internal::unpack_8_uint3_values(
          buffer3 + 8, packed + 3);
      torchao::bitpacking::internal::unpack_8_uint3_values(
          buffer3 + 16, packed + 6);
      torchao::bitpacking::internal::unpack_8_uint3_values(
          buffer3 + 24, packed + 9);
      shifted0 = vld1q_u8(buffer3);
      shifted1 = vld1q_u8(buffer3 + 16);
      break;
    case 4:
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          shifted0, shifted1, packed);
      break;
    case 5:
      uint8_t buffer5[32];
      torchao::bitpacking::internal::unpack_8_uint5_values(buffer5, packed);
      torchao::bitpacking::internal::unpack_8_uint5_values(
          buffer5 + 8, packed + 5);
      torchao::bitpacking::internal::unpack_8_uint5_values(
          buffer5 + 16, packed + 10);
      torchao::bitpacking::internal::unpack_8_uint5_values(
          buffer5 + 24, packed + 15);
      shifted0 = vld1q_u8(buffer5);
      shifted1 = vld1q_u8(buffer5 + 16);
      break;
    case 6:
      torchao::bitpacking::internal::vec_unpack_32_uint6_values(
          shifted0, shifted1, packed);
      break;
    case 7:
      uint8_t buffer7[32];
      torchao::bitpacking::internal::unpack_8_uint7_values(buffer7, packed);
      torchao::bitpacking::internal::unpack_8_uint7_values(
          buffer7 + 8, packed + 7);
      torchao::bitpacking::internal::unpack_8_uint7_values(
          buffer7 + 16, packed + 14);
      torchao::bitpacking::internal::unpack_8_uint7_values(
          buffer7 + 24, packed + 21);
      shifted0 = vld1q_u8(buffer7);
      shifted1 = vld1q_u8(buffer7 + 16);
      break;
    case 8:
      unpacked0 = vreinterpretq_s8_u8(vld1q_u8(packed));
      unpacked1 = vreinterpretq_s8_u8(vld1q_u8(packed + 16));
      break;
    default:
      assert(false);
  }

  // unshift to move unpacked values to full range
  // no shifting is needed for 8-bit packing
  if constexpr (nbit < 8) {
    int8x16_t unshift = vdupq_n_s8(-(1 << (nbit - 1)));
    unpacked0 = vaddq_s8(vreinterpretq_s8_u8(shifted0), unshift);
    unpacked1 = vaddq_s8(vreinterpretq_s8_u8(shifted1), unshift);
  }
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_pack_64_lowbit_values(
    uint8_t* packed,
    const int8x16_t& unpacked0,
    const int8x16_t& unpacked1,
    const int8x16_t& unpacked2,
    const int8x16_t& unpacked3) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  // Shift unpacked values to nonnegative range for quantization of 1-7 bits
  // No shifting is needed for 8-bit packing
  uint8x16_t shifted0;
  uint8x16_t shifted1;
  uint8x16_t shifted2;
  uint8x16_t shifted3;
  if constexpr (nbit < 8) {
    int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
    shifted0 = vreinterpretq_u8_s8(vaddq_s8(unpacked0, shift));
    shifted1 = vreinterpretq_u8_s8(vaddq_s8(unpacked1, shift));
    shifted2 = vreinterpretq_u8_s8(vaddq_s8(unpacked2, shift));
    shifted3 = vreinterpretq_u8_s8(vaddq_s8(unpacked3, shift));
  }

  switch (nbit) {
    case 1:
      torchao::bitpacking::internal::vec_pack_64_uint1_values(
          packed, shifted0, shifted1, shifted2, shifted3);
      break;
    case 2:
      torchao::bitpacking::internal::vec_pack_64_uint2_values(
          packed, shifted0, shifted1, shifted2, shifted3);
      break;
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
    case 5:
      torchao::bitpacking::internal::vec_pack_64_uint5_values(
          packed, shifted0, shifted1, shifted2, shifted3);
      break;
    case 6:
      torchao::bitpacking::internal::vec_pack_64_uint6_values(
          packed, shifted0, shifted1, shifted2, shifted3);
      break;
    case 7:
      torchao::bitpacking::internal::vec_pack_64_uint7_values(
          packed, shifted0, shifted1, shifted2, shifted3);
      break;
    case 8:
      vst1q_u8(packed, vreinterpretq_u8_s8(unpacked0));
      vst1q_u8(packed + 16, vreinterpretq_u8_s8(unpacked1));
      vst1q_u8(packed + 32, vreinterpretq_u8_s8(unpacked2));
      vst1q_u8(packed + 48, vreinterpretq_u8_s8(unpacked3));
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
    const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  uint8x16_t shifted0;
  uint8x16_t shifted1;
  uint8x16_t shifted2;
  uint8x16_t shifted3;

  switch (nbit) {
    case 1:
      torchao::bitpacking::internal::vec_unpack_64_uint1_values(
          shifted0, shifted1, shifted2, shifted3, packed);
      break;
    case 2:
      torchao::bitpacking::internal::vec_unpack_64_uint2_values(
          shifted0, shifted1, shifted2, shifted3, packed);
      break;
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
    case 5:
      torchao::bitpacking::internal::vec_unpack_64_uint5_values(
          shifted0, shifted1, shifted2, shifted3, packed);
      break;
    case 6:
      torchao::bitpacking::internal::vec_unpack_64_uint6_values(
          shifted0, shifted1, shifted2, shifted3, packed);
      break;
    case 7:
      torchao::bitpacking::internal::vec_unpack_64_uint7_values(
          shifted0, shifted1, shifted2, shifted3, packed);
      break;
    case 8:
      unpacked0 = vreinterpretq_s8_u8(vld1q_u8(packed));
      unpacked1 = vreinterpretq_s8_u8(vld1q_u8(packed + 16));
      unpacked2 = vreinterpretq_s8_u8(vld1q_u8(packed + 32));
      unpacked3 = vreinterpretq_s8_u8(vld1q_u8(packed + 48));
      break;
    default:
      assert(false);
  }

  // unshift to move unpacked values to full range
  // no shifting is needed for 8-bit packing
  if constexpr (nbit < 8) {
    int8x16_t unshift = vdupq_n_s8(-(1 << (nbit - 1)));
    unpacked0 = vaddq_s8(vreinterpretq_s8_u8(shifted0), unshift);
    unpacked1 = vaddq_s8(vreinterpretq_s8_u8(shifted1), unshift);
    unpacked2 = vaddq_s8(vreinterpretq_s8_u8(shifted2), unshift);
    unpacked3 = vaddq_s8(vreinterpretq_s8_u8(shifted3), unshift);
  }
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_pack_128_uintx_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3,
    const uint8x16_t& unpacked4,
    const uint8x16_t& unpacked5,
    const uint8x16_t& unpacked6,
    const uint8x16_t& unpacked7) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);
  switch (nbit) {
    case 1:
      torchao::bitpacking::internal::vec_pack_128_uint1_values(
          packed,
          unpacked0,
          unpacked1,
          unpacked2,
          unpacked3,
          unpacked4,
          unpacked5,
          unpacked6,
          unpacked7);
      break;
    case 2:
      torchao::bitpacking::internal::vec_pack_64_uint2_values(
          packed, unpacked0, unpacked1, unpacked2, unpacked3);
      torchao::bitpacking::internal::vec_pack_64_uint2_values(
          packed + 16, unpacked4, unpacked5, unpacked6, unpacked7);
      break;
    case 3:
      torchao::bitpacking::internal::vec_pack_128_uint3_values(
          packed,
          unpacked0,
          unpacked1,
          unpacked2,
          unpacked3,
          unpacked4,
          unpacked5,
          unpacked6,
          unpacked7);
      break;
    case 4:
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed, unpacked0, unpacked1);
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed + 16, unpacked2, unpacked3);
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed + 32, unpacked4, unpacked5);
      torchao::bitpacking::internal::vec_pack_32_uint4_values(
          packed + 48, unpacked6, unpacked7);
      break;
    case 5:
      torchao::bitpacking::internal::vec_pack_128_uint5_values(
          packed,
          unpacked0,
          unpacked1,
          unpacked2,
          unpacked3,
          unpacked4,
          unpacked5,
          unpacked6,
          unpacked7);
      break;
    case 6:
      torchao::bitpacking::internal::vec_pack_64_uint6_values(
          packed, unpacked0, unpacked1, unpacked2, unpacked3);
      torchao::bitpacking::internal::vec_pack_64_uint6_values(
          packed + 48, unpacked4, unpacked5, unpacked6, unpacked7);
      break;
    case 7:
      torchao::bitpacking::internal::vec_pack_128_uint7_values(
          packed,
          unpacked0,
          unpacked1,
          unpacked2,
          unpacked3,
          unpacked4,
          unpacked5,
          unpacked6,
          unpacked7);
      break;
    case 8:
      vst1q_u8(packed, unpacked0);
      vst1q_u8(packed + 16, unpacked1);
      vst1q_u8(packed + 32, unpacked2);
      vst1q_u8(packed + 48, unpacked3);
      vst1q_u8(packed + 64, unpacked4);
      vst1q_u8(packed + 80, unpacked5);
      vst1q_u8(packed + 96, unpacked6);
      vst1q_u8(packed + 112, unpacked7);
      break;
    default:
      assert(false);
  }
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_unpack_128_uintx_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    uint8x16_t& unpacked4,
    uint8x16_t& unpacked5,
    uint8x16_t& unpacked6,
    uint8x16_t& unpacked7,
    const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);
  switch (nbit) {
    case 1:
      torchao::bitpacking::internal::vec_unpack_128_uint1_values(
          unpacked0,
          unpacked1,
          unpacked2,
          unpacked3,
          unpacked4,
          unpacked5,
          unpacked6,
          unpacked7,
          packed);
      break;
    case 2:
      torchao::bitpacking::internal::vec_unpack_64_uint2_values(
          unpacked0, unpacked1, unpacked2, unpacked3, packed);
      torchao::bitpacking::internal::vec_unpack_64_uint2_values(
          unpacked4, unpacked5, unpacked6, unpacked7, packed + 16);
      break;
    case 3:
      torchao::bitpacking::internal::vec_unpack_128_uint3_values(
          unpacked0,
          unpacked1,
          unpacked2,
          unpacked3,
          unpacked4,
          unpacked5,
          unpacked6,
          unpacked7,
          packed);
      break;
    case 4:
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          unpacked0, unpacked1, packed);
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          unpacked2, unpacked3, packed + 16);
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          unpacked4, unpacked5, packed + 32);
      torchao::bitpacking::internal::vec_unpack_32_uint4_values(
          unpacked6, unpacked7, packed + 48);
      break;
    case 5:
      torchao::bitpacking::internal::vec_unpack_128_uint5_values(
          unpacked0,
          unpacked1,
          unpacked2,
          unpacked3,
          unpacked4,
          unpacked5,
          unpacked6,
          unpacked7,
          packed);
      break;
    case 6:
      torchao::bitpacking::internal::vec_unpack_64_uint6_values(
          unpacked0, unpacked1, unpacked2, unpacked3, packed);
      torchao::bitpacking::internal::vec_unpack_64_uint6_values(
          unpacked4, unpacked5, unpacked6, unpacked7, packed + 48);
      break;
    case 7:
      torchao::bitpacking::internal::vec_unpack_128_uint7_values(
          unpacked0,
          unpacked1,
          unpacked2,
          unpacked3,
          unpacked4,
          unpacked5,
          unpacked6,
          unpacked7,
          packed);
      break;
    case 8:
      unpacked0 = vld1q_u8(packed);
      unpacked1 = vld1q_u8(packed + 16);
      unpacked2 = vld1q_u8(packed + 32);
      unpacked3 = vld1q_u8(packed + 48);
      unpacked4 = vld1q_u8(packed + 64);
      unpacked5 = vld1q_u8(packed + 80);
      unpacked6 = vld1q_u8(packed + 96);
      unpacked7 = vld1q_u8(packed + 112);
      break;
    default:
      assert(false);
  }
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
  static_assert(nbit < 9);
  static_assert(nbit >= 1);

  // Shift unpacked values to nonnegative range for quantization of 1-7 bits
  // No shifting is needed for 8-bit packing
  uint8x16_t uintx0;
  uint8x16_t uintx1;
  uint8x16_t uintx2;
  uint8x16_t uintx3;
  uint8x16_t uintx4;
  uint8x16_t uintx5;
  uint8x16_t uintx6;
  uint8x16_t uintx7;
  if constexpr (nbit < 8) {
    int8x16_t shift = vdupq_n_s8(1 << (nbit - 1));
    uintx0 = vreinterpretq_u8_s8(vaddq_s8(unpacked0, shift));
    uintx1 = vreinterpretq_u8_s8(vaddq_s8(unpacked1, shift));
    uintx2 = vreinterpretq_u8_s8(vaddq_s8(unpacked2, shift));
    uintx3 = vreinterpretq_u8_s8(vaddq_s8(unpacked3, shift));
    uintx4 = vreinterpretq_u8_s8(vaddq_s8(unpacked4, shift));
    uintx5 = vreinterpretq_u8_s8(vaddq_s8(unpacked5, shift));
    uintx6 = vreinterpretq_u8_s8(vaddq_s8(unpacked6, shift));
    uintx7 = vreinterpretq_u8_s8(vaddq_s8(unpacked7, shift));
  } else {
    static_assert(nbit == 8);
    uintx0 = vreinterpretq_u8_s8(unpacked0);
    uintx1 = vreinterpretq_u8_s8(unpacked1);
    uintx2 = vreinterpretq_u8_s8(unpacked2);
    uintx3 = vreinterpretq_u8_s8(unpacked3);
    uintx4 = vreinterpretq_u8_s8(unpacked4);
    uintx5 = vreinterpretq_u8_s8(unpacked5);
    uintx6 = vreinterpretq_u8_s8(unpacked6);
    uintx7 = vreinterpretq_u8_s8(unpacked7);
  }
  vec_pack_128_uintx_values<nbit>(
      packed, uintx0, uintx1, uintx2, uintx3, uintx4, uintx5, uintx6, uintx7);
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
    const uint8_t* packed) {
  static_assert(nbit < 9);
  static_assert(nbit >= 1);
  uint8x16_t uintx0;
  uint8x16_t uintx1;
  uint8x16_t uintx2;
  uint8x16_t uintx3;
  uint8x16_t uintx4;
  uint8x16_t uintx5;
  uint8x16_t uintx6;
  uint8x16_t uintx7;
  vec_unpack_128_uintx_values<nbit>(
      uintx0, uintx1, uintx2, uintx3, uintx4, uintx5, uintx6, uintx7, packed);

  // unshift to move unpacked values to full range
  // no shifting is needed for 8-bit packing
  if constexpr (nbit < 8) {
    int8x16_t unshift = vdupq_n_s8(-(1 << (nbit - 1)));
    unpacked0 = vaddq_s8(vreinterpretq_s8_u8(uintx0), unshift);
    unpacked1 = vaddq_s8(vreinterpretq_s8_u8(uintx1), unshift);
    unpacked2 = vaddq_s8(vreinterpretq_s8_u8(uintx2), unshift);
    unpacked3 = vaddq_s8(vreinterpretq_s8_u8(uintx3), unshift);
    unpacked4 = vaddq_s8(vreinterpretq_s8_u8(uintx4), unshift);
    unpacked5 = vaddq_s8(vreinterpretq_s8_u8(uintx5), unshift);
    unpacked6 = vaddq_s8(vreinterpretq_s8_u8(uintx6), unshift);
    unpacked7 = vaddq_s8(vreinterpretq_s8_u8(uintx7), unshift);
  } else {
    static_assert(nbit == 8);
    unpacked0 = vreinterpretq_s8_u8(uintx0);
    unpacked1 = vreinterpretq_s8_u8(uintx1);
    unpacked2 = vreinterpretq_s8_u8(uintx2);
    unpacked3 = vreinterpretq_s8_u8(uintx3);
    unpacked4 = vreinterpretq_s8_u8(uintx4);
    unpacked5 = vreinterpretq_s8_u8(uintx5);
    unpacked6 = vreinterpretq_s8_u8(uintx6);
    unpacked7 = vreinterpretq_s8_u8(uintx7);
  }
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void vec_unpack_128_lowbit_values_with_lut(
    int8x16_t& unpacked0,
    int8x16_t& unpacked1,
    int8x16_t& unpacked2,
    int8x16_t& unpacked3,
    int8x16_t& unpacked4,
    int8x16_t& unpacked5,
    int8x16_t& unpacked6,
    int8x16_t& unpacked7,
    const uint8_t* packed,
    const int8x16_t& lut) {
  static_assert(nbit <= 4);
  static_assert(nbit >= 1);
  uint8x16_t idx0;
  uint8x16_t idx1;
  uint8x16_t idx2;
  uint8x16_t idx3;
  uint8x16_t idx4;
  uint8x16_t idx5;
  uint8x16_t idx6;
  uint8x16_t idx7;
  vec_unpack_128_uintx_values<nbit>(
      idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, packed);
  unpacked0 = vqtbl1q_s8(lut, idx0);
  unpacked1 = vqtbl1q_s8(lut, idx1);
  unpacked2 = vqtbl1q_s8(lut, idx2);
  unpacked3 = vqtbl1q_s8(lut, idx3);
  unpacked4 = vqtbl1q_s8(lut, idx4);
  unpacked5 = vqtbl1q_s8(lut, idx5);
  unpacked6 = vqtbl1q_s8(lut, idx6);
  unpacked7 = vqtbl1q_s8(lut, idx7);
}


TORCHAO_ALWAYS_INLINE inline void lookup_and_store_16_fp32_values(
  float* out,
  const uint8x16_t& idx,
  const int8x16x4_t& lut) {

const int8x16_t s_idx = vreinterpretq_s8_u8(idx);
int8x16_t b0 = vqtbl1q_s8(lut.val[0], s_idx);
int8x16_t b1 = vqtbl1q_s8(lut.val[1], s_idx);
int8x16_t b2 = vqtbl1q_s8(lut.val[2], s_idx);
int8x16_t b3 = vqtbl1q_s8(lut.val[3], s_idx);

int8x16x4_t result_bytes = {b0, b1, b2, b3};
vst4q_s8(reinterpret_cast<int8_t*>(out), result_bytes);
}

template <int nbit>
TORCHAO_ALWAYS_INLINE inline void unpack_128_lowbit_values_with_fp32_lut(
  float* unpacked,
  const uint8_t* packed,
  const int8x16x4_t& lut
) {
  uint8x16_t idx0;
  uint8x16_t idx1;
  uint8x16_t idx2;
  uint8x16_t idx3;
  uint8x16_t idx4;
  uint8x16_t idx5;
  uint8x16_t idx6;
  uint8x16_t idx7;
  vec_unpack_128_uintx_values<nbit>(
    idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, packed);
  lookup_and_store_16_fp32_values(unpacked + 0, idx0, lut);
  lookup_and_store_16_fp32_values(unpacked + 16, idx1, lut);
  lookup_and_store_16_fp32_values(unpacked + 32, idx2, lut);
  lookup_and_store_16_fp32_values(unpacked + 48, idx3, lut);
  lookup_and_store_16_fp32_values(unpacked + 64, idx4, lut);
  lookup_and_store_16_fp32_values(unpacked + 80, idx5, lut);
  lookup_and_store_16_fp32_values(unpacked + 96, idx6, lut);
  lookup_and_store_16_fp32_values(unpacked + 112, idx7, lut);
}

} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
