// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint1.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint2.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint3.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint4.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint5.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint6.h>
#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint7.h>
#include <cassert>

namespace torchao::kernels::cpu::fallback::bitpacking {
namespace internal {
/**
 * @brief Packs 128 unsigned 8-bit integers into a packed format of 'nbit' bits.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @param packed Pointer to the destination memory for the packed data.
 * @param unpacked_values Pointer to the source memory with 128 uint8_t values.
 */
template <int nbit>
inline void pack_128_uint_values(
    uint8_t* packed,
    const uint8_t* unpacked_values) {
  static_assert(nbit >= 1 && nbit <= 8, "nbit must be between 1 and 8");

  // Dispatch to the correct packing function
  if constexpr (nbit == 1) {
    pack_128_uint1_values(packed, unpacked_values);
  } else if constexpr (nbit == 2) {
    pack_64_uint2_values(packed, unpacked_values);
    pack_64_uint2_values(packed + 16, unpacked_values + 64);
  } else if constexpr (nbit == 3) {
    pack_128_uint3_values(packed, unpacked_values);
  } else if constexpr (nbit == 4) {
    pack_32_uint4_values(packed, unpacked_values);
    pack_32_uint4_values(packed + 16, unpacked_values + 32);
    pack_32_uint4_values(packed + 32, unpacked_values + 64);
    pack_32_uint4_values(packed + 48, unpacked_values + 96);
  } else if constexpr (nbit == 5) {
    pack_128_uint5_values(packed, unpacked_values);
  } else if constexpr (nbit == 6) {
    pack_64_uint6_values(packed, unpacked_values);
    pack_64_uint6_values(packed + 48, unpacked_values + 64);
  } else if constexpr (nbit == 7) {
    pack_128_uint7_values(packed, unpacked_values);
  } else if constexpr (nbit == 8) {
    // For 8-bit, it's a direct memory copy
    for (int i = 0; i < 128; ++i) {
      packed[i] = unpacked_values[i];
    }
  }
}
/**
 * @brief Unpacks 'nbit' data into 128 unsigned 8-bit integers.
 *
 * @tparam nbit The number of bits per value in the packed format (1-8).
 * @param unpacked_values Pointer to the destination memory (128 uint8_t
 * values).
 * @param packed Pointer to the source packed data.
 */
template <int nbit>
inline void unpack_128_uint_values(
    uint8_t* unpacked_values,
    const uint8_t* packed) {
  static_assert(nbit >= 1 && nbit <= 8, "nbit must be between 1 and 8");

  // Dispatch to the correct unpacking function, writing directly to the output.
  if constexpr (nbit == 1) {
    unpack_128_uint1_values(unpacked_values, packed);
  } else if constexpr (nbit == 2) {
    unpack_64_uint2_values(unpacked_values, packed);
    unpack_64_uint2_values(unpacked_values + 64, packed + 16);
  } else if constexpr (nbit == 3) {
    unpack_128_uint3_values(unpacked_values, packed);
  } else if constexpr (nbit == 4) {
    unpack_32_uint4_values(unpacked_values, packed);
    unpack_32_uint4_values(unpacked_values + 32, packed + 16);
    unpack_32_uint4_values(unpacked_values + 64, packed + 32);
    unpack_32_uint4_values(unpacked_values + 96, packed + 48);
  } else if constexpr (nbit == 5) {
    unpack_128_uint5_values(unpacked_values, packed);
  } else if constexpr (nbit == 6) {
    unpack_64_uint6_values(unpacked_values, packed);
    unpack_64_uint6_values(unpacked_values + 64, packed + 48);
  } else if constexpr (nbit == 7) {
    unpack_128_uint7_values(unpacked_values, packed);
  } else if constexpr (nbit == 8) {
    // For 8-bit, it's a direct memory copy
    for (int i = 0; i < 128; ++i) {
      unpacked_values[i] = packed[i];
    }
  }
}

/**
 * @brief Packs 128 signed 8-bit integers into a packed format of 'nbit' bits.
 *
 * @tparam nbit The number of bits to pack each value into (1-8).
 * @param packed Pointer to the destination memory.
 * @param unpacked Pointer to the source memory containing 128 int8_t values.
 */
template <int nbit>
inline void pack_128_lowbit_int_values(
    uint8_t* packed,
    const int8_t* unpacked) {
  // 1. Convert signed input to a temporary buffer of unsigned values.
  uint8_t temp_unpacked[128];
  if constexpr (nbit < 8) {
    const int8_t shift = 1 << (nbit - 1);
    for (int i = 0; i < 128; ++i) {
      temp_unpacked[i] = static_cast<uint8_t>(unpacked[i] + shift);
    }
  } else { // nbit == 8
    for (int i = 0; i < 128; ++i) {
      temp_unpacked[i] = static_cast<uint8_t>(unpacked[i]);
    }
  }

  // 2. Call the generalized uint packing function.
  pack_128_uint_values<nbit>(packed, temp_unpacked);
}

template <int nbit>
inline void unpack_128_lowbit_int_values(
    int8_t* unpacked,
    const uint8_t* packed) {
  // 1. Get the raw unsigned values by calling the base function.
  uint8_t temp_unpacked[128];
  unpack_128_uint_values<nbit>(temp_unpacked, packed);

  // 2. Perform the signed conversion.
  if constexpr (nbit < 8) {
    const int8_t unshift = -(1 << (nbit - 1));
    for (int i = 0; i < 128; ++i) {
      unpacked[i] = static_cast<int8_t>(temp_unpacked[i]) + unshift;
    }
  } else { // nbit == 8
    for (int i = 0; i < 128; ++i) {
      unpacked[i] = static_cast<int8_t>(temp_unpacked[i]);
    }
  }
}

/**
 * @brief Unpacks 'nbit' data and de-quantizes it using a lookup table (LUT).
 *
 * @tparam nbit The number of bits per value in the packed format (1-4).
 * @param unpacked Pointer to the destination memory (128 int8_t values).
 * @param packed Pointer to the source packed data.
 * @param lut Pointer to the lookup table (must have 2^nbit entries).
 */
template <int nbit>
inline void unpack_128_lowbit_values_with_lut(
    int8_t* unpacked,
    const uint8_t* packed,
    const int8_t* lut) {
  static_assert(nbit >= 1 && nbit <= 4, "LUT version only supports nbit <= 4");

  // Create a temporary buffer on the stack for the indices.
  uint8_t indices[128];

  // 1. Call the utility function to handle all the unpacking logic.
  unpack_128_uint_values<nbit>(indices, packed);

  // 2. Apply the lookup table.
  for (int i = 0; i < 128; ++i) {
    unpacked[i] = lut[indices[i]];
  }
}
} // namespace internal
} // namespace torchao::kernels::cpu::fallback::bitpacking
