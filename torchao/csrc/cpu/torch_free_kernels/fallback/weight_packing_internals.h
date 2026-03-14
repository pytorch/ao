// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Scalar fallback implementations of pack_buffer, unpack_buffer, and compute_sum
// used by the shared weight_packing algorithm on non-ARM architectures (x86, etc.).

#pragma once

#if !defined(__aarch64__) && !defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>

#include <cassert>
#include <cstdint>

namespace torchao::weight_packing::fallback {

template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void pack_buffer(
    void* packed_weights,
    const int8_t* buffer) {
  auto packed_ptr = reinterpret_cast<uint8_t*>(packed_weights);

  if constexpr (kr * nr == 128) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        pack_128_lowbit_int_values<weight_nbit>(packed_ptr, buffer);
    return;
  }
  if constexpr (kr * nr == 64) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        pack_64_lowbit_int_values<weight_nbit>(packed_ptr, buffer);
    return;
  }
  if constexpr (kr * nr == 32) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        pack_32_lowbit_int_values<weight_nbit>(packed_ptr, buffer);
    return;
  }
  assert(false);
}

template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void unpack_buffer(
    int8_t* buffer,
    const void* packed_weights) {
  auto packed_ptr = reinterpret_cast<const uint8_t*>(packed_weights);

  if constexpr (kr * nr == 128) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        unpack_128_lowbit_int_values<weight_nbit>(buffer, packed_ptr);
    return;
  }
  if constexpr (kr * nr == 64) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        unpack_64_lowbit_int_values<weight_nbit>(buffer, packed_ptr);
    return;
  }
  if constexpr (kr * nr == 32) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        unpack_32_lowbit_int_values<weight_nbit>(buffer, packed_ptr);
    return;
  }
  assert(false);
}

inline int32_t compute_sum(const int8_t* vals, int count) {
  int32_t sum = 0;
  for (int i = 0; i < count; ++i) {
    sum += vals[i];
  }
  return sum;
}

} // namespace torchao::weight_packing::fallback

#endif // !defined(__aarch64__) && !defined(__ARM_NEON)
