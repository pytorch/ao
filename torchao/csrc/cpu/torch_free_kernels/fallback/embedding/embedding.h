// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace torchao::kernels::cpu::fallback::embedding {

namespace internal {

TORCHAO_ALWAYS_INLINE inline void dequantize_and_store_values(
    float* out,
    const int8_t* qvals,
    int count,
    float scale,
    float zero) {
  for (int i = 0; i < count; ++i) {
    out[i] = (static_cast<float>(qvals[i]) - zero) * scale;
  }
}

} // namespace internal

template <int weight_nbit>
inline void pack_embedding_weight_qvals_(
    // Output
    void* packed_qvals,
    // Inputs
    int embedding_dim,
    const int8_t* qvals) {
  assert(embedding_dim % 32 == 0);

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
inline void embedding_(
    // Output
    float* out,
    // Inputs
    int embedding_dim,
    int group_size,
    const void* packed_weight_qvals,
    const float* weight_scales,
    // If weight_zeros is nullptr, they are assumed zeros
    const int8_t* weight_zeros) {
  assert(embedding_dim % 32 == 0);

  constexpr int bytes_per_packed_128_values = (128 * weight_nbit) / 8;
  constexpr int bytes_per_packed_64_values = (64 * weight_nbit) / 8;
  constexpr int bytes_per_packed_32_values = (32 * weight_nbit) / 8;

  auto packed_weight_qvals_byte_ptr =
      reinterpret_cast<const uint8_t*>(packed_weight_qvals);

  int8_t qvals_buffer[128];
  int packed_offset = 0;
  int i = 0;

  // Unpack and dequantize 128 values at a time
  for (; i + 128 - 1 < embedding_dim; i += 128) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        unpack_128_lowbit_int_values<weight_nbit>(
            qvals_buffer, packed_weight_qvals_byte_ptr + packed_offset);
    packed_offset += bytes_per_packed_128_values;

    // Dequantize in chunks of group_size (or 32 for simplicity)
    for (int j = 0; j < 128; j += 32) {
      int group_idx = (i + j) / group_size;
      float scale = weight_scales[group_idx];
      float zero = (weight_zeros != nullptr)
          ? static_cast<float>(weight_zeros[group_idx])
          : 0.0f;

      int chunk_size = std::min(32, 128 - j);
      internal::dequantize_and_store_values(
          out + i + j, qvals_buffer + j, chunk_size, scale, zero);
    }
  }

  // Unpack and dequantize 64 values if remaining
  if (i + 64 - 1 < embedding_dim) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        unpack_64_lowbit_int_values<weight_nbit>(
            qvals_buffer, packed_weight_qvals_byte_ptr + packed_offset);
    packed_offset += bytes_per_packed_64_values;

    for (int j = 0; j < 64; j += 32) {
      int group_idx = (i + j) / group_size;
      float scale = weight_scales[group_idx];
      float zero = (weight_zeros != nullptr)
          ? static_cast<float>(weight_zeros[group_idx])
          : 0.0f;

      int chunk_size = std::min(32, 64 - j);
      internal::dequantize_and_store_values(
          out + i + j, qvals_buffer + j, chunk_size, scale, zero);
    }
    i += 64;
  }

  // Unpack and dequantize 32 values if remaining
  if (i + 32 - 1 < embedding_dim) {
    torchao::kernels::cpu::fallback::bitpacking::internal::
        unpack_32_lowbit_int_values<weight_nbit>(
            qvals_buffer, packed_weight_qvals_byte_ptr + packed_offset);
    packed_offset += bytes_per_packed_32_values;

    int group_idx = i / group_size;
    float scale = weight_scales[group_idx];
    float zero = (weight_zeros != nullptr)
        ? static_cast<float>(weight_zeros[group_idx])
        : 0.0f;

    internal::dequantize_and_store_values(out + i, qvals_buffer, 32, scale, zero);
    i += 32;
  }

  assert(i == embedding_dim);
}

template <int weight_nbit>
inline void embedding(
    // Output
    float* out,
    // Inputs
    int embedding_dim,
    int group_size,
    const void* packed_weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros,
    int index) {
  assert(group_size % 32 == 0);
  assert(embedding_dim % group_size == 0);

  auto packed_weight_qvals_byte_ptr =
      reinterpret_cast<const uint8_t*>(packed_weight_qvals);

  int groups_per_embedding = embedding_dim / group_size;
  int packed_bytes_per_embedding = embedding_dim * weight_nbit / 8;

  packed_weight_qvals_byte_ptr += (index * packed_bytes_per_embedding);
  weight_scales += index * groups_per_embedding;
  if (weight_zeros != nullptr) {
    weight_zeros += index * groups_per_embedding;
  }

  embedding_<weight_nbit>(
      out,
      embedding_dim,
      group_size,
      packed_weight_qvals_byte_ptr,
      weight_scales,
      weight_zeros);
}

template <int weight_nbit>
inline void pack_embedding_weight_qvals(
    // Output
    void* packed_qvals,
    // Inputs
    int embedding_dim,
    const int8_t* qvals,
    int index) {
  assert(embedding_dim % 32 == 0);
  int packed_bytes_per_embedding = embedding_dim * weight_nbit / 8;
  auto packed_qvals_byte_ptr = reinterpret_cast<uint8_t*>(packed_qvals);

  pack_embedding_weight_qvals_<weight_nbit>(
      packed_qvals_byte_ptr + index * packed_bytes_per_embedding,
      embedding_dim,
      qvals + index * embedding_dim);
}

} // namespace torchao::kernels::cpu::fallback::embedding
