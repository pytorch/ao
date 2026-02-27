// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// NEON implementations of pack_buffer, unpack_buffer, and compute_sum
// used by the shared weight_packing algorithm.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/reduction.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>

#include <cassert>

namespace torchao::weight_packing::aarch64 {

template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void pack_buffer(
    void* packed_weights,
    const int8_t* buffer) {
  if constexpr (kr * nr == 128) {
    torchao::bitpacking::vec_pack_128_lowbit_values<weight_nbit>(
        reinterpret_cast<uint8_t*>(packed_weights),
        vld1q_s8(buffer),
        vld1q_s8(buffer + 16),
        vld1q_s8(buffer + 32),
        vld1q_s8(buffer + 48),
        vld1q_s8(buffer + 64),
        vld1q_s8(buffer + 80),
        vld1q_s8(buffer + 96),
        vld1q_s8(buffer + 112));
    return;
  }
  if constexpr (kr * nr == 64) {
    torchao::bitpacking::vec_pack_64_lowbit_values<weight_nbit>(
        reinterpret_cast<uint8_t*>(packed_weights),
        vld1q_s8(buffer),
        vld1q_s8(buffer + 16),
        vld1q_s8(buffer + 32),
        vld1q_s8(buffer + 48));
    return;
  }
  if constexpr (kr * nr == 32) {
    torchao::bitpacking::vec_pack_32_lowbit_values<weight_nbit>(
        reinterpret_cast<uint8_t*>(packed_weights),
        vld1q_s8(buffer),
        vld1q_s8(buffer + 16));
    return;
  }
  assert(false);
}

template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void unpack_buffer(
    int8_t* buffer,
    const void* packed_weights) {
  int8x16_t vals0;
  int8x16_t vals1;
  int8x16_t vals2;
  int8x16_t vals3;
  int8x16_t vals4;
  int8x16_t vals5;
  int8x16_t vals6;
  int8x16_t vals7;

  if constexpr (kr * nr == 128) {
    torchao::bitpacking::vec_unpack_128_lowbit_values<weight_nbit>(
        vals0,
        vals1,
        vals2,
        vals3,
        vals4,
        vals5,
        vals6,
        vals7,
        reinterpret_cast<const uint8_t*>(packed_weights));
    vst1q_s8(buffer, vals0);
    vst1q_s8(buffer + 16, vals1);
    vst1q_s8(buffer + 32, vals2);
    vst1q_s8(buffer + 48, vals3);
    vst1q_s8(buffer + 64, vals4);
    vst1q_s8(buffer + 80, vals5);
    vst1q_s8(buffer + 96, vals6);
    vst1q_s8(buffer + 112, vals7);
    return;
  }
  if constexpr (kr * nr == 64) {
    torchao::bitpacking::vec_unpack_64_lowbit_values<weight_nbit>(
        vals0,
        vals1,
        vals2,
        vals3,
        reinterpret_cast<const uint8_t*>(packed_weights));
    vst1q_s8(buffer, vals0);
    vst1q_s8(buffer + 16, vals1);
    vst1q_s8(buffer + 32, vals2);
    vst1q_s8(buffer + 48, vals3);
    return;
  }
  if constexpr (kr * nr == 32) {
    torchao::bitpacking::vec_unpack_32_lowbit_values<weight_nbit>(
        vals0, vals1, reinterpret_cast<const uint8_t*>(packed_weights));
    vst1q_s8(buffer, vals0);
    vst1q_s8(buffer + 16, vals1);
    return;
  }
  assert(false);
}

inline int32_t compute_sum(const int8_t* vals, int count) {
  return torchao::kernels::cpu::aarch64::reduction::compute_sum(vals, count);
}

} // namespace torchao::weight_packing::aarch64

#endif // defined(__aarch64__) || defined(__ARM_NEON)
