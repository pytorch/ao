#pragma once

#if defined(aarch64) || defined(__ARM_NEON)
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/lut/lut.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <torchao/experimental/kernels/cpu/aarch64/packing/utils.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut::
    weight_packing {
namespace lut_utils = torchao::lut;
namespace packing_utils = torchao::packing;

/**
 * @brief Calculates the exact buffer size in bytes for packed weights.
 *
 * This function computes the total memory required for a weight buffer based on
 * a specific packing layout. The calculation accounts for tiled weights, a
 * Look-Up Table (LUT), and optional interleaved scales and biases. It assumes
 * the 'n' dimension is padded to be a multiple of the tile height 'nr'.
 *
 * @param n The number of output channels (columns) in the weight matrix.
 * @param k The number of input channels (rows) in the weight matrix.
 * @param weight_nbit The bit precision for each weight (e.g., 4, 8).
 * @param scale_group_size The number of weights that share a single scale
 * factor.
 * @param has_scales Set to true to include space for scaling factors.
 * @param has_bias Set to true to include space for a bias vector.
 * @param nr The tile height used for packing along the 'n' dimension.
 * @param kr The tile width used for packing along the 'k' dimension.
 * @return The total required size in bytes for the complete packed buffer.
 */
inline size_t packed_weights_size(
    int n,
    int k,
    int weight_nbit,
    int scale_group_size,
    bool has_scales,
    bool has_bias,
    int nr,
    int kr) {
  size_t size_per_n_strip = 0;

  // 1. Size of the LUT, written once per strip.
  size_per_n_strip += 16 * sizeof(float);

  // 2. Size of the interleaved scales.
  if (has_scales) {
    assert(
        k % scale_group_size == 0 &&
        "k must be a multiple of scale_group_size");
    size_t num_scale_blocks = k / scale_group_size;
    size_per_n_strip += num_scale_blocks * nr * sizeof(float);
  }

  // 3. Size of the packed weight tiles.
  assert(k % kr == 0 && "k must be a multiple of kr");
  size_t num_k_tiles = k / kr;
  size_t bytes_per_weight_tile = ((nr * kr * weight_nbit) + 7) / 8;
  size_per_n_strip += num_k_tiles * bytes_per_weight_tile;

  // 4. Size of the bias, written once per strip.
  if (has_bias) {
    size_per_n_strip += nr * sizeof(float);
  }

  // Calculate the total number of n-strips, padding n to a multiple of nr.
  int num_n_strips = (n + nr - 1) / nr;

  return size_per_n_strip * num_n_strips;
}

/**
 * @brief Packs weights, LUTs, scales and bias into a kernel-optimized format.
 * @details The function organizes the output buffer into "n-strips," where
each strip corresponds to a tile of `nr_` columns from the weight matrix.
 * The memory layout for each strip is as follows:
 * 1.  **Look-Up Table (LUT):** A 16-element float LUT is written once at
 * the beginning of the strip.
 * 2.  **Interleaved Scales:** If `has_scales` is true, dequantization
 * scales are interleaved. For each group of `scale_group_size`
 * elements along the k-dimension, `nr_` scale values (one for each
 * column in the strip) are written.
 * 3.  **Packed Weight Tiles:** The core weight data is tiled into
 * (`nr_` x `kr_`) blocks. These blocks are then bit-packed and
 * interleaved according to the `sr_` ratio before being written.
 * 4.  **Bias:** If `has_bias` is true, `nr_` bias values are appended
 * at the end of the strip.
 *
 * @tparam weight_nbit_ The true bit-width of the weights.
 * @tparam nr_ The column-tiling factor for the kernel (e.g., 4).
 * @tparam kr_ The column-tiling factor of the micro-kernel (e.g., 32).
 * @tparam sr_ Split ratio determine how the k dimension of a weight tile is
chunked and interleaved during the packing process.
 * @param packed_weights_ptr Pointer to the destination buffer.
 * @param weight_qval_indices Pointer to the quantized weight matrix (uint8,
row-major).
 * @param weight_scales Pointer to the scale factors (float32, row-major).
 * @param weight_luts Pointer to the LUTs (float32, row-major).
 * @param n The number of columns in the weight matrix.
 * @param k The number of rows in the weight matrix.
 * @param scale_group_size The number of weights that share a scale factor.
 * @param lut_group_size The number of weights that share a LUT.
 * @param has_scales If true, the packed buffer will contain scale factors.
 * @param has_bias If true, the packed buffer will contain bias terms.
 * @param bias Pointer to the bias vector (float32, row-major).
 */
template <int weight_nbit_, int nr_, int kr_, int sr_>
TORCHAO_ALWAYS_INLINE inline void pack_weights(
    // Output
    void* packed_weights_ptr,
    // Inputs
    const uint8_t* weight_qval_indices,
    const float* weight_scales,
    const float* weight_luts,
    int n,
    int k,
    int scale_group_size,
    int lut_group_size,
    bool has_scales,
    bool has_bias,
    const float* bias) {
  static_assert(nr_ == 4);
  static_assert(kr_ == 32);
  static_assert(sr_ == 8);
  static_assert(kr_ % sr_ == 0, "kr must be divisible by sr");
  assert(k % kr_ == 0 && "K must be a multiple of tile dimension kr");
  assert(scale_group_size > 0 && "Scale group size must be positive");
  assert(lut_group_size > 0 && "LUT group size must be positive");

  // Grouping hierarchy constraint
  assert(
      lut_group_size % scale_group_size == 0 &&
      "LUT group size must be a multiple of scale group size");

  // Group compatibility constraints with tile dimensions
  assert(
      lut_group_size % (k * nr_) == 0 &&
      "LUT group size must be compatible with tile dimensions");
  assert(scale_group_size % kr_ == 0 && "Scale group size % kr must be 0");

  auto* out_ptr = reinterpret_cast<uint8_t*>(packed_weights_ptr);
  constexpr int kLutBufferSize = 16;
  std::vector<float> lut_buffer(kLutBufferSize);

  std::vector<uint8_t> padded_tile(nr_ * kr_);

  std::vector<uint8_t> tmp_buffer(128);
  constexpr int bytes_per_128_packed_values =
      ((nr_ * kr_ * weight_nbit_) + 7) / 8;

  const int lut_size = 1 << weight_nbit_;
  const int scales_per_col = k / scale_group_size;

  for (int n_idx = 0; n_idx < n; n_idx += nr_) {
    int current_lut_idx = (n_idx * k) / lut_group_size;

    std::memset(lut_buffer.data(), 0, 16 * sizeof(float));
    std::memcpy(out_ptr, lut_buffer.data(), 16 * sizeof(float));

    std::memcpy(
        lut_buffer.data(),
        weight_luts + current_lut_idx * lut_size,
        lut_size * sizeof(float));
    std::memcpy(out_ptr, lut_buffer.data(), 16 * sizeof(float));
    out_ptr += 16 * sizeof(float);

    for (int k_idx = 0; k_idx < k; k_idx += kr_) {
      int w_idx = n_idx * k + k_idx;
      // Write scales if k_idx is a multiple of scale_group_size
      if (has_scales && (k_idx % scale_group_size == 0)) {
        int scale_idx = w_idx / scale_group_size;
        // Write scales for next nr columns
        for (int j = 0; j < nr_; j++) {
          float scale = 0.0;
          if (n_idx + j < n) {
            scale = weight_scales[scale_idx + j * scales_per_col];
          }
          std::memcpy(out_ptr, &scale, sizeof(float));
          out_ptr += sizeof(float);
        }
      }
      // Write 128 packed tile (kr x nr)
      std::memset(padded_tile.data(), 0, 128);
      for (int j = 0; j < nr_; j++) {
        if (n_idx + j < n) {
          std::memcpy(
              padded_tile.data() + j * kr_,
              weight_qval_indices + w_idx + j * k,
              kr_);
        }
      }
      packing_utils::pack_values(
          tmp_buffer.data(), padded_tile.data(), nr_, kr_, sr_);
      const uint8_t* buffer = tmp_buffer.data();
      torchao::bitpacking::vec_pack_128_uintx_values<weight_nbit_>(
          reinterpret_cast<uint8_t*>(out_ptr),
          vld1q_u8(buffer),
          vld1q_u8(buffer + 16),
          vld1q_u8(buffer + 32),
          vld1q_u8(buffer + 48),
          vld1q_u8(buffer + 64),
          vld1q_u8(buffer + 80),
          vld1q_u8(buffer + 96),
          vld1q_u8(buffer + 112));
      out_ptr += bytes_per_128_packed_values;
    } // k_idx

    if (has_bias) {
      for (int i = 0; i < nr_; i++) {
        float current_bias = 0.0;
        if (n_idx + i < n) {
          current_bias = bias[n_idx + i];
        }
        std::memcpy(out_ptr, &current_bias, sizeof(float));
        out_ptr += sizeof(float);
      }
    }
  }
}
} // namespace
  // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut::weight_packing
#endif // defined(aarch64) || defined(__ARM_NEON)
