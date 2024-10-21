/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * This implementation is copied from
 * executorch/runtime/core/portable_type/bfloat16.h
 */

inline float f32_from_bits(uint16_t src) {
  float res = 0;
  uint32_t tmp = src;
  tmp <<= 16;
  std::memcpy(&res, &tmp, sizeof(tmp));
  return res;
}

inline uint16_t bits_from_f32(float src) {
  uint32_t res = 0;
  std::memcpy(&res, &src, sizeof(res));
  return res >> 16;
}

inline uint16_t round_to_nearest_even(float src) {
  if (std::isnan(src)) {
    return UINT16_C(0x7FC0);
  }
  uint32_t U32 = 0;
  std::memcpy(&U32, &src, sizeof(U32));
  uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
  return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
}

/**
 * The "brain floating-point" type, compatible with c10/util/BFloat16.h from
 * pytorch core.
 *
 * This representation uses 1 bit for the sign, 8 bits for the exponent and 7
 * bits for the mantissa.
 */
struct alignas(2) BFloat16 {
  uint16_t x;

  BFloat16() = default;
  struct from_bits_t {};
  static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr BFloat16(unsigned short bits, from_bits_t) : x(bits) {}
  /* implicit */ BFloat16(float value) : x(round_to_nearest_even(value)) {}
  operator float() const {
    return f32_from_bits(x);
  }
};
