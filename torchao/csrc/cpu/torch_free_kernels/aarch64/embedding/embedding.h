// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/pack_weights.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <cassert>
#include <vector>

namespace torchao::kernels::cpu::aarch64::embedding {

namespace internal {

TORCHAO_ALWAYS_INLINE inline void vec_dequantize_i32_fp32(
    float32x4_t& out,
    const int32x4_t& qvals,
    const float32x4_t& scales,
    const float32x4_t& zeros) {
  out = vcvtq_f32_s32(qvals);
  out = vsubq_f32(out, zeros);
  out = vmulq_f32(out, scales);
}

TORCHAO_ALWAYS_INLINE inline void vec_dequantize_and_store_16_values(
    float* out,
    const int8x16_t& input,
    float scale,
    float zero) {
  float32x4_t dequant;
  float32x4_t scales = vdupq_n_f32(scale);
  float32x4_t zeros = vdupq_n_f32(zero);

  int16x8_t low8 = vmovl_s8(vget_low_s8(input));
  int32x4_t qvals = vmovl_s16(vget_low_s16(low8));
  vec_dequantize_i32_fp32(dequant, qvals, scales, zeros);
  vst1q_f32(out, dequant);

  qvals = vmovl_s16(vget_high_s16(low8));
  vec_dequantize_i32_fp32(dequant, qvals, scales, zeros);
  vst1q_f32(out + 4, dequant);

  int16x8_t high8 = vmovl_s8(vget_high_s8(input));
  qvals = vmovl_s16(vget_low_s16(high8));
  vec_dequantize_i32_fp32(dequant, qvals, scales, zeros);
  vst1q_f32(out + 8, dequant);

  qvals = vmovl_s16(vget_high_s16(high8));
  vec_dequantize_i32_fp32(dequant, qvals, scales, zeros);
  vst1q_f32(out + 12, dequant);
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

  int8x16_t qvals0;
  int8x16_t qvals1;
  int8x16_t qvals2;
  int8x16_t qvals3;
  int8x16_t qvals4;
  int8x16_t qvals5;
  int8x16_t qvals6;
  int8x16_t qvals7;

  int packed_offset = 0;
  int i = 0;
  for (; i + 128 - 1 < embedding_dim; i += 128) {
    qvals0 = vld1q_s8(qvals + i);
    qvals1 = vld1q_s8(qvals + i + 16);
    qvals2 = vld1q_s8(qvals + i + 32);
    qvals3 = vld1q_s8(qvals + i + 48);
    qvals4 = vld1q_s8(qvals + i + 64);
    qvals5 = vld1q_s8(qvals + i + 80);
    qvals6 = vld1q_s8(qvals + i + 96);
    qvals7 = vld1q_s8(qvals + i + 112);
    torchao::bitpacking::vec_pack_128_lowbit_values<weight_nbit>(
        packed_qvals_byte_ptr + packed_offset,
        qvals0,
        qvals1,
        qvals2,
        qvals3,
        qvals4,
        qvals5,
        qvals6,
        qvals7);
    packed_offset += bytes_per_packed_128_values;
  }

  if (i + 64 - 1 < embedding_dim) {
    qvals0 = vld1q_s8(qvals + i);
    qvals1 = vld1q_s8(qvals + i + 16);
    qvals2 = vld1q_s8(qvals + i + 32);
    qvals3 = vld1q_s8(qvals + i + 48);
    torchao::bitpacking::vec_pack_64_lowbit_values<weight_nbit>(
        packed_qvals_byte_ptr + packed_offset, qvals0, qvals1, qvals2, qvals3);
    packed_offset += bytes_per_packed_64_values;
    i += 64;
  }

  if (i + 32 - 1 < embedding_dim) {
    qvals0 = vld1q_s8(qvals + i);
    qvals1 = vld1q_s8(qvals + i + 16);
    torchao::bitpacking::vec_pack_32_lowbit_values<weight_nbit>(
        packed_qvals_byte_ptr + packed_offset, qvals0, qvals1);
    packed_offset += bytes_per_packed_32_values;
    i += 32;
  }

  assert(i == embedding_dim); // because 32 | embedding_dim
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

  int8x16_t qvals0;
  int8x16_t qvals1;
  int8x16_t qvals2;
  int8x16_t qvals3;
  int8x16_t qvals4;
  int8x16_t qvals5;
  int8x16_t qvals6;
  int8x16_t qvals7;

  int packed_offset = 0;
  int i = 0;
  for (; i + 128 - 1 < embedding_dim; i += 128) {
    torchao::bitpacking::vec_unpack_128_lowbit_values<weight_nbit>(
        qvals0,
        qvals1,
        qvals2,
        qvals3,
        qvals4,
        qvals5,
        qvals6,
        qvals7,
        packed_weight_qvals_byte_ptr + packed_offset);
    packed_offset += bytes_per_packed_128_values;

    // Dequantize and store first 32 values
    int group_idx = i / group_size;
    float scale = weight_scales[group_idx];
    float zero = 0.0;
    if (weight_zeros != nullptr) {
      zero = weight_zeros[group_idx];
    }

    internal::vec_dequantize_and_store_16_values(out + i, qvals0, scale, zero);
    internal::vec_dequantize_and_store_16_values(
        out + i + 16, qvals1, scale, zero);

    // Dequantize and store second 32 values
    group_idx = (i + 32) / group_size;
    scale = weight_scales[group_idx];
    if (weight_zeros != nullptr) {
      zero = weight_zeros[group_idx];
    }
    internal::vec_dequantize_and_store_16_values(
        out + i + 32, qvals2, scale, zero);
    internal::vec_dequantize_and_store_16_values(
        out + i + 48, qvals3, scale, zero);

    // Dequantize and store third 32 values
    group_idx = (i + 64) / group_size;
    scale = weight_scales[group_idx];
    if (weight_zeros != nullptr) {
      zero = weight_zeros[group_idx];
    }
    internal::vec_dequantize_and_store_16_values(
        out + i + 64, qvals4, scale, zero);
    internal::vec_dequantize_and_store_16_values(
        out + i + 80, qvals5, scale, zero);

    // Dequantize and store fourth 32 values
    group_idx = (i + 96) / group_size;
    scale = weight_scales[group_idx];
    if (weight_zeros != nullptr) {
      zero = weight_zeros[group_idx];
    }
    internal::vec_dequantize_and_store_16_values(
        out + i + 96, qvals6, scale, zero);
    internal::vec_dequantize_and_store_16_values(
        out + i + 112, qvals7, scale, zero);
  }

  if (i + 64 - 1 < embedding_dim) {
    torchao::bitpacking::vec_unpack_64_lowbit_values<weight_nbit>(
        qvals0,
        qvals1,
        qvals2,
        qvals3,
        packed_weight_qvals_byte_ptr + packed_offset);
    packed_offset += bytes_per_packed_64_values;

    // Dequantize and store first 32 values
    int group_idx = i / group_size;
    float scale = weight_scales[group_idx];
    float zero = 0.0;
    if (weight_zeros != nullptr) {
      zero = weight_zeros[group_idx];
    }
    internal::vec_dequantize_and_store_16_values(out + i, qvals0, scale, zero);
    internal::vec_dequantize_and_store_16_values(
        out + i + 16, qvals1, scale, zero);

    // Dequantize and store second 32 values
    group_idx = (i + 32) / group_size;
    scale = weight_scales[group_idx];
    if (weight_zeros != nullptr) {
      zero = weight_zeros[group_idx];
    }
    internal::vec_dequantize_and_store_16_values(
        out + i + 32, qvals2, scale, zero);
    internal::vec_dequantize_and_store_16_values(
        out + i + 48, qvals3, scale, zero);

    i += 64;
  }

  if (i + 32 - 1 < embedding_dim) {
    torchao::bitpacking::vec_unpack_32_lowbit_values<weight_nbit>(
        qvals0, qvals1, packed_weight_qvals_byte_ptr + packed_offset);
    packed_offset += bytes_per_packed_32_values;

    int group_idx = i / group_size;
    float scale = weight_scales[group_idx];
    float zero = 0.0;
    if (weight_zeros != nullptr) {
      zero = weight_zeros[group_idx];
    }
    internal::vec_dequantize_and_store_16_values(out + i, qvals0, scale, zero);
    internal::vec_dequantize_and_store_16_values(
        out + i + 16, qvals1, scale, zero);

    i += 32;
  }

  assert(i == embedding_dim); // because 32 | embedding_dim
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
  assert(embedding_dim % 8 == 0);
  int packed_bytes_per_embedding = embedding_dim * weight_nbit / 8;
  auto packed_qvals_byte_ptr = reinterpret_cast<uint8_t*>(packed_qvals);

  pack_embedding_weight_qvals_<weight_nbit>(
      packed_qvals_byte_ptr + index * packed_bytes_per_embedding,
      embedding_dim,
      qvals + index * embedding_dim);
}

// Embedding op that shares weights with unembedding linear op
template <int weight_nbit, int nr, int kr, int sr>
inline void shared_embedding(
    // Output
    float* out,
    // Inputs
    const void* packed_weights,
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias,
    int index) {
  assert(k % group_size == 0);
  assert(group_size % 16 == 0);

  int groups_per_k = k / group_size;
  std::vector<int8_t> weight_qvals(k * nr);
  std::vector<float> weight_scales(groups_per_k * nr);
  std::vector<int8_t> weight_zeros(groups_per_k * nr);
  std::vector<float> bias(nr);

  // Set n_idx to multiple of nr that is at most index
  // j is index of "index" in nr group
  int n_idx = index / nr;
  n_idx = n_idx * nr;
  int j = index - n_idx;

  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::weight_packing::
          unpack_weights_at_n_idx<weight_nbit, nr, kr, sr>(
              weight_qvals.data(),
              weight_scales.data(),
              has_weight_zeros ? weight_zeros.data() : nullptr,
              has_bias ? bias.data() : nullptr,
              n_idx,
              n,
              k,
              group_size,
              has_weight_zeros,
              has_bias,
              packed_weights);

  // Dequantize and store to output (size k)
  int8x16_t qvals;
  for (int i = 0; i < k; i += 16) {
    qvals = vld1q_s8(weight_qvals.data() + j * k + i);
    float scale = weight_scales[j * groups_per_k + i / group_size];
    float zero = 0.0;
    if (has_weight_zeros) {
      zero = weight_zeros[j * groups_per_k + i / group_size];
    }
    internal::vec_dequantize_and_store_16_values(out + i, qvals, scale, zero);
  }
}

} // namespace torchao::kernels::cpu::aarch64::embedding

#endif // defined(__aarch64__) || defined(__ARM_NEON)
