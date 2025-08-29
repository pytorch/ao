// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/lut/lut.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <cassert>
#include <vector>

namespace torchao::kernels::cpu::aarch64::embedding {

/**
 * @brief Calculates the size in bytes for a single row of packed embeddings.
 *
 * This function computes the memory stride for one row, accounting for three
 * components:
 * 1. Bit-packed weight indices.
 * 2. Optional, group-quantized scales.
 * 3. Padded look-up tables (LUTs).
 *
 * @param weight_nbit The number of bits for each weight index (e.g., 2, 4).
 * @param embedding_dim The dimension of the embedding vector (i.e., number of
 * weights per row).
 * @param scale_group_size The number of weights that share a single
 * quantization scale.
 * @param lut_group_size The number of weights that share a single look-up
 * table.
 * @param has_scales A flag indicating whether quantization scales are stored.
 * @return The total size in bytes (stride) for one packed row.
 */
inline size_t packed_embedding_size_per_row(
    int weight_nbit,
    int embedding_dim,
    int scale_group_size,
    int lut_group_size,
    bool has_scales) {
  // We need to account for the padding of the LUTs. The LUTs are padded to 16
  // floats (64 bytes) for alignment.
  constexpr int kLutPaddedSize = 16;
  // Number of LUTs per row, it could be 1 or more LUTs per row.
  const int lut_per_row = (embedding_dim + lut_group_size - 1) / lut_group_size;
  // LUT size in bytes
  const int lut_bytes = lut_per_row * kLutPaddedSize * sizeof(float);

  // Scales are packed if has_scales is true.
  const int scales_per_row =
      (embedding_dim + scale_group_size - 1) / scale_group_size;
  const int scale_bytes = has_scales ? (scales_per_row * sizeof(float)) : 0;

  // The indices are bit-packed.
  const int index_bytes = (embedding_dim * weight_nbit + 7) / 8;

  const size_t packed_row_stride = lut_bytes + scale_bytes + index_bytes;
  return packed_row_stride;
}

/**
 * @brief Calculates the total size in bytes for an entire table of packed
 * embeddings.
 *
 * This is a convenience function that multiplies the size of a single packed
 * row by the total number of embeddings (rows) to find the total memory
 * required.
 *
 * @param weight_nbit The number of bits for each weight index.
 * @param num_embeddings The total number of rows (embeddings) in the weight
 * table.
 * @param embedding_dim The dimension of the embedding vector.
 * @param scale_group_size The number of weights sharing a single scale.
 * @param lut_group_size The number of weights sharing a single LUT.
 * @param has_scales A flag indicating if scales are present.
 * @return The total size in bytes required for the entire packed weight table.
 */
inline size_t packed_embedding_size(
    int weight_nbit,
    int num_embeddings,
    int embedding_dim,
    int scale_group_size,
    int lut_group_size,
    bool has_scales) {
  // Pass the correct arguments to the helper function.
  return num_embeddings *
      packed_embedding_size_per_row(
             weight_nbit,
             embedding_dim,
             scale_group_size,
             lut_group_size,
             has_scales);
}

template <int weight_nbit>
inline void pack_embedding_row_at_index_lut(
    // Destination
    void* packed_table,
    int index,
    // Source Tables
    const uint8_t* source_indices_table,
    const float* source_scales_table,
    const float* source_luts_table,
    // Dimensions
    int num_embeddings,
    int embedding_dim,
    int scale_group_size,
    int lut_group_size,
    bool has_scales) {
  assert(index >= 0 && index < num_embeddings);
  assert(embedding_dim > 0 && embedding_dim % 32 == 0);

  // 1. Calculate the stride of one packed row (for the destination table)
  constexpr int kLutPaddedSize = 16;
  const int lut_size = 1 << weight_nbit;
  const int lut_per_row = (embedding_dim + lut_group_size - 1) / lut_group_size;
  const int scales_per_row =
      (embedding_dim + scale_group_size - 1) / scale_group_size;

  const size_t packed_row_stride = packed_embedding_size_per_row(
      weight_nbit, embedding_dim, scale_group_size, lut_group_size, has_scales);

  constexpr int bytes_per_packed_128_values = (128 * weight_nbit) / 8;
  constexpr int bytes_per_packed_64_values = (64 * weight_nbit) / 8;
  constexpr int bytes_per_packed_32_values = (32 * weight_nbit) / 8;
  // 2. Calculate the starting pointer for the destination row
  uint8_t* output_ptr = reinterpret_cast<uint8_t*>(packed_table) +
      (static_cast<size_t>(index) * packed_row_stride);

  // --- 3. Calculate the starting pointers for the SOURCE data row ---
  // This is the key change to support 1D indexing.
  const size_t linear_idx_start_of_row =
      static_cast<size_t>(index) * embedding_dim;

  // Find the global group index for the start of our row.
  const size_t start_lut_group_idx = linear_idx_start_of_row / lut_group_size;
  const size_t start_scale_group_idx =
      linear_idx_start_of_row / scale_group_size;

  const uint8_t* source_indices_for_row =
      source_indices_table + linear_idx_start_of_row;
  const float* source_scales_for_row =
      source_scales_table + start_scale_group_idx;
  const float* source_luts_for_row =
      source_luts_table + start_lut_group_idx * lut_size;

  // 4. Pack LUTs
  std::vector<float> lut_buffer(kLutPaddedSize, 0.0f);
  for (int i = 0; i < lut_per_row; i++) {
    std::memcpy(
        lut_buffer.data(),
        source_luts_for_row + i * lut_size,
        lut_size * sizeof(float));
    std::memcpy(output_ptr, lut_buffer.data(), kLutPaddedSize * sizeof(float));
    output_ptr += kLutPaddedSize * sizeof(float);
  }

  // 5. Pack Scales
  if (has_scales) {
    std::memcpy(
        output_ptr, source_scales_for_row, scales_per_row * sizeof(float));
    output_ptr += scales_per_row * sizeof(float);
  }

  // 6. Pack Weight Indices (Quantized Values)
  int i = 0;
  // Process in chunks of 128
  for (; i + 128 <= embedding_dim; i += 128) {
    uint8x16_t qvals0 = vld1q_u8(source_indices_for_row + i);
    uint8x16_t qvals1 = vld1q_u8(source_indices_for_row + i + 16);
    uint8x16_t qvals2 = vld1q_u8(source_indices_for_row + i + 32);
    uint8x16_t qvals3 = vld1q_u8(source_indices_for_row + i + 48);
    uint8x16_t qvals4 = vld1q_u8(source_indices_for_row + i + 64);
    uint8x16_t qvals5 = vld1q_u8(source_indices_for_row + i + 80);
    uint8x16_t qvals6 = vld1q_u8(source_indices_for_row + i + 96);
    uint8x16_t qvals7 = vld1q_u8(source_indices_for_row + i + 112);

    torchao::bitpacking::vec_pack_128_uintx_values<weight_nbit>(
        output_ptr,
        qvals0,
        qvals1,
        qvals2,
        qvals3,
        qvals4,
        qvals5,
        qvals6,
        qvals7);
    output_ptr += bytes_per_packed_128_values;
  }

  // Process in chunks of 64
  if (i + 64 <= embedding_dim) {
    uint8x16_t qvals0 = vld1q_u8(source_indices_for_row + i);
    uint8x16_t qvals1 = vld1q_u8(source_indices_for_row + i + 16);
    uint8x16_t qvals2 = vld1q_u8(source_indices_for_row + i + 32);
    uint8x16_t qvals3 = vld1q_u8(source_indices_for_row + i + 48);

    torchao::bitpacking::vec_pack_64_uintx_values<weight_nbit>(
        output_ptr, qvals0, qvals1, qvals2, qvals3);
    output_ptr += bytes_per_packed_64_values;
    i += 64;
  }

  // Process in chunks of 32
  if (i + 32 <= embedding_dim) {
    uint8x16_t qvals0 = vld1q_u8(source_indices_for_row + i);
    uint8x16_t qvals1 = vld1q_u8(source_indices_for_row + i + 16);
    torchao::bitpacking::vec_pack_32_uintx_values<weight_nbit>(
        output_ptr, qvals0, qvals1);
    output_ptr += bytes_per_packed_32_values;
    i += 32;
  }

  assert(i == embedding_dim); // Final check: Ensure all elements were processed
}

/**
 * @brief Reads a single embedding vector from the packed format and dequantizes
 * it.
 *
 * @tparam weight_nbit The number of bits used for the quantized weights (e.g.,
 * 2, 4).
 * @param out Pointer to the output buffer for the dequantized float vector.
 * Must have space for `embedding_dim` floats.
 * @param packed_data Pointer to the beginning of the entire packed embedding
 * table.
 * @param index The row index of the embedding vector to retrieve.
 * @param num_embeddings The total number of embeddings in the table (for
 * boundary checks).
 * @param embedding_dim The dimension of a single embedding vector.
 * @param scale_group_size The number of values sharing a single scale.
 * @param lut_group_size The number of values sharing a single LUT.
 * @param has_scales A flag indicating if scales were packed.
 */
template <int weight_nbit>
inline void dequantize_embedding_row_at_idx_lut(
    // Output
    float* out,
    // Inputs
    const void* packed_data,
    int index,
    int num_embeddings,
    int embedding_dim,
    int scale_group_size,
    int lut_group_size,
    bool has_scales) {
  assert(index >= 0 && index < num_embeddings);
  assert(embedding_dim > 0 && embedding_dim % 32 == 0);

  // 1. Calculate the total size (stride) of one packed embedding row

  // LUTs are padded to 16 floats (64 bytes) for alignment.
  constexpr int kLutPaddedSize = 16;
  const int lut_per_row = (embedding_dim + lut_group_size - 1) / lut_group_size;
  const int lut_bytes = lut_per_row * kLutPaddedSize * sizeof(float);

  // Scales are packed if has_scales is true.
  const int scales_per_row =
      (embedding_dim + scale_group_size - 1) / scale_group_size;
  const int scale_bytes = has_scales ? (scales_per_row * sizeof(float)) : 0;

  // The indices are bit-packed.
  const int index_bytes = (embedding_dim * weight_nbit) / 8;

  const size_t total_row_stride = lut_bytes + scale_bytes + index_bytes;

  // 2. Calculate the memory offset to the start of the desired row
  const uint8_t* row_start_ptr = reinterpret_cast<const uint8_t*>(packed_data) +
      (static_cast<size_t>(index) * total_row_stride);

  // 3. Get pointers to the LUTs, scales, and packed indices for this row
  const float* luts_ptr = reinterpret_cast<const float*>(row_start_ptr);
  const float* scales_ptr = has_scales
      ? reinterpret_cast<const float*>(row_start_ptr + lut_bytes)
      : nullptr;
  const uint8_t* packed_indices_ptr = row_start_ptr + lut_bytes + scale_bytes;

  // 4. Unpack the n-bit indices into a temporary 8-bit buffer
  std::vector<uint8_t> unpacked_indices(embedding_dim);
  const uint8_t* read_ptr = packed_indices_ptr;
  uint8_t* write_ptr = unpacked_indices.data();
  int i = 0;

  constexpr int bytes_per_packed_128_values = (128 * weight_nbit) / 8;
  constexpr int bytes_per_packed_64_values = (64 * weight_nbit) / 8;
  constexpr int bytes_per_packed_32_values = (32 * weight_nbit) / 8;

  // Process in chunks of 128
  for (; i + 128 <= embedding_dim; i += 128) {
    // 1. Declare NEON registers for the output
    uint8x16_t u0, u1, u2, u3, u4, u5, u6, u7;
    // 2. Unpack directly into the registers
    torchao::bitpacking::vec_unpack_128_lut_indices<weight_nbit>(
        u0, u1, u2, u3, u4, u5, u6, u7, read_ptr);
    // 3. Store the results from registers to memory
    vst1q_u8(write_ptr + 0, u0);
    vst1q_u8(write_ptr + 16, u1);
    vst1q_u8(write_ptr + 32, u2);
    vst1q_u8(write_ptr + 48, u3);
    vst1q_u8(write_ptr + 64, u4);
    vst1q_u8(write_ptr + 80, u5);
    vst1q_u8(write_ptr + 96, u6);
    vst1q_u8(write_ptr + 112, u7);

    write_ptr += 128;
    read_ptr += bytes_per_packed_128_values;
  }

  // Process in chunks of 64
  if (i + 64 <= embedding_dim) {
    uint8x16_t u0, u1, u2, u3;
    torchao::bitpacking::vec_unpack_64_lut_indices<weight_nbit>(
        u0, u1, u2, u3, read_ptr);
    vst1q_u8(write_ptr + 0, u0);
    vst1q_u8(write_ptr + 16, u1);
    vst1q_u8(write_ptr + 32, u2);
    vst1q_u8(write_ptr + 48, u3);

    write_ptr += 64;
    read_ptr += bytes_per_packed_64_values;
    i += 64;
  }

  // Process in chunks of 32
  if (i + 32 <= embedding_dim) {
    uint8x16_t u0, u1;
    torchao::bitpacking::vec_unpack_32_lut_indices<weight_nbit>(
        u0, u1, read_ptr);
    vst1q_u8(write_ptr + 0, u0);
    vst1q_u8(write_ptr + 16, u1);

    write_ptr += 32;
    read_ptr += bytes_per_packed_32_values;
    i += 32;
  }

  assert(i == embedding_dim);
  // Dequantize using vectorized LUT lookup
  for (int j = 0; j < embedding_dim; j += 16) {
    // Identify and load the LUT for this 16-element chunk.
    // Since lut_group_size % 16 == 0, all 16 elements use the same LUT.
    const int lut_group_idx = j / lut_group_size;
    const float* current_lut_ptr = luts_ptr + lut_group_idx * kLutPaddedSize;
    uint8x16x4_t lut_neon;
    torchao::lut::load_fp32_lut(lut_neon, current_lut_ptr);

    // Load the 16 indices to be looked up.
    uint8x16_t indices_neon = vld1q_u8(unpacked_indices.data() + j);

    // Perform the vectorized lookup. The results are in out0..3.
    float32x4_t out0, out1, out2, out3;
    torchao::lut::lookup_from_fp32_lut(
        out0, out1, out2, out3, lut_neon, indices_neon);
    float scale_val = 1.0f;
    // Apply scales vectorially.
    if (has_scales) {
      // Since scale_group_size % 16 == 0, all 16 elements use the same scale.
      const int scale_group_idx = j / scale_group_size;
      scale_val = scales_ptr[scale_group_idx];
      // Load the single scale value into all 4 lanes of a vector register.
      float32x4_t scale_vec = vdupq_n_f32(scale_val);

      // Multiply the looked-up values by the scale.
      out0 = vmulq_f32(out0, scale_vec);
      out1 = vmulq_f32(out1, scale_vec);
      out2 = vmulq_f32(out2, scale_vec);
      out3 = vmulq_f32(out3, scale_vec);
    }

    // Store the final 16 float results back to the output buffer.
    vst1q_f32(out + j + 0, out0);
    vst1q_f32(out + j + 4, out1);
    vst1q_f32(out + j + 8, out2);
    vst1q_f32(out + j + 12, out3);
  }
}
} // namespace torchao::kernels::cpu::aarch64::embedding

#endif // defined(__aarch64__) || defined(__ARM_NEON)
