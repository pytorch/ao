// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace torchao::kernels::cpu::fallback::embedding {

template <int weight_nbit>
inline void pack_embedding_weight_qvals_(
    // Output
    void* packed_qvals,
    // Inputs
    int embedding_dim,
    const int8_t* qvals) {
  assert(embedding_dim % 8 == 0);

  constexpr int bytes_per_packed_128_values = (128 * weight_nbit) / 8;
  constexpr int bytes_per_packed_64_values = (64 * weight_nbit) / 8;
  constexpr int bytes_per_packed_32_values = (32 * weight_nbit) / 8;

  auto packed_qvals_byte_ptr = reinterpret_cast<uint8_t*>(packed_qvals);

  int packed_offset = 0;
  int i = 0;

  // Pack 128 values at a time
  for (; i + 128 - 1 < embedding_dim; i += 128) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        pack_128_lowbit_int_values<weight_nbit>(
            packed_qvals_byte_ptr + packed_offset, qvals + i);
    packed_offset += bytes_per_packed_128_values;
  }

  // Pack 64 values if remaining
  if (i + 64 - 1 < embedding_dim) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        pack_64_lowbit_int_values<weight_nbit>(
            packed_qvals_byte_ptr + packed_offset, qvals + i);
    packed_offset += bytes_per_packed_64_values;
    i += 64;
  }

  // Pack 32 values if remaining
  if (i + 32 - 1 < embedding_dim) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        pack_32_lowbit_int_values<weight_nbit>(
            packed_qvals_byte_ptr + packed_offset, qvals + i);
    packed_offset += bytes_per_packed_32_values;
    i += 32;
  }

  assert(i == embedding_dim);
}

template <int weight_nbit>
inline void pack_embedding_weight_qvals(
    // Output
    void* packed_qvals,
    // Inputs
    int embedding_dim,
    const int8_t* qvals,
    int index) {
  assert(embedding_dim % 8 == 0);
  int packed_bytes_per_embedding = embedding_dim * weight_nbit / 8;
  auto packed_qvals_byte_ptr = reinterpret_cast<uint8_t*>(packed_qvals);

  pack_embedding_weight_qvals_<weight_nbit>(
      packed_qvals_byte_ptr + index * packed_bytes_per_embedding,
      embedding_dim,
      qvals + index * embedding_dim);
}

} // namespace torchao::kernels::cpu::fallback::embedding
