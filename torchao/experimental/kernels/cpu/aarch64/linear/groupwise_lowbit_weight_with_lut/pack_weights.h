#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace torchao::kernels::cpu::aarch64::linear::
    groupwise_lowbit_weight::weight_packing {

namespace internal {

constexpr int gcd(int a, int b) {
    while (b) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

/**
 * @brief Promotes a low-bit LUT to a 16-byte format and copies it to the destination.
 *
 * This function takes a source LUT of any size (e.g., 4 entries for 2-bit, 8 for 3-bit),
 * pads it with zeros to fill a 16-byte buffer, and writes that buffer to the
 * destination. This allows a single, highly-optimized 4-bit NEON lookup
 * instruction to be used for all bit-widths.
 *
 * @tparam weight_nbit The original number of bits for the weight indices (1, 2, 3, or 4).
 * @param dst Pointer to the destination in the final packed buffer (must be 16-byte aligned).
 * @param src Pointer to the source LUT values (e.g., from the input std::vector<float>).
 */
template <int weight_nbit>
TORCHAO_ALWAYS_INLINE inline void promote_and_pack_lut(
    int8_t* dst,
    const float* src) {

  // The actual number of entries in the source LUT.
  constexpr int lut_size_per_entry = (1 << weight_nbit);

  // The size of the destination buffer is always 16 bytes for the NEON instruction.
  constexpr int promoted_lut_buffer_size = 16;

  // Create a temporary buffer on the stack. Modern compilers are very
  // efficient at handling this. Initializing with {} ensures it's zero-filled.
  int8_t promoted_lut_buffer[promoted_lut_buffer_size] = {};

  // Copy the actual LUT values from the source, casting from float to int8_t.
  for(int i = 0; i < lut_size_per_entry; ++i) {
      promoted_lut_buffer[i] = static_cast<int8_t>(src[i]);
  }
  // The rest of the 'promoted_lut_buffer' remains zero, providing the necessary padding.

  // Copy the entire 16-byte promoted LUT to its final destination in the packed buffer.
  std::memcpy(dst, promoted_lut_buffer, promoted_lut_buffer_size);
}

// Packs a buffer of (kr * nr) lowbit values (stored as int8_t) down to bits
template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void pack_buffer_for_lut(
    void* packed_weights,
    const int8_t* buffer) {
  static_assert(weight_nbit >= 1);
  static_assert(weight_nbit <= 4);
  const uint8_t* buffer_u8 = reinterpret_cast<const uint8_t*>(buffer);
  if constexpr (kr * nr == 128) {
    torchao::bitpacking::vec_pack_128_uintx_values<weight_nbit>(
        (uint8_t*)packed_weights,
        vld1q_u8(buffer_u8),
        vld1q_u8(buffer_u8 + 16),
        vld1q_u8(buffer_u8 + 32),
        vld1q_u8(buffer_u8 + 48),
        vld1q_u8(buffer_u8 + 64),
        vld1q_u8(buffer_u8 + 80),
        vld1q_u8(buffer_u8 + 96),
        vld1q_u8(buffer_u8 + 112));
    return;
  }
  if constexpr (kr * nr == 64) {
    torchao::bitpacking::vec_pack_64_uintx_values<weight_nbit>(
        (uint8_t*)packed_weights,
        vld1q_u8(buffer_u8),
        vld1q_u8(buffer_u8 + 16),
        vld1q_u8(buffer_u8 + 32),
        vld1q_u8(buffer_u8 + 48));
    return;
  }
  if constexpr (kr * nr == 32) {
    torchao::bitpacking::vec_pack_32_uintx_values<weight_nbit>(
        (uint8_t*)packed_weights, vld1q_u8(buffer_u8), vld1q_u8(buffer_u8 + 16));
    return;
  }
  assert(false && "Unsupported kr*nr value for pack_buffer_for_lut");
}


// Packs nr * kr values for GEMM with packing params (nr, kr, sr)
template <typename T>
void pack_values(
    T* packed_values, const T* values, int nr, int kr, int sr) {
  assert(kr % sr == 0);
  int kr_per_sr = kr / sr;
  int dst_idx = 0;
  for (int sr_idx = 0; sr_idx < sr; sr_idx++) {
    for (int n_idx = 0; n_idx < nr; n_idx++) {
      std::memcpy(
          packed_values + dst_idx,
          values + n_idx * kr + sr_idx * kr_per_sr,
          sizeof(T) * kr_per_sr);
      dst_idx += kr_per_sr;
    }
  }
}

// Maps source values to destination using a LUT.
TORCHAO_ALWAYS_INLINE inline void
map_values(int8_t* dst, const uint8_t* src, int8x16_t lut, int size) {
  assert(size % 16 == 0);
  for (int i = 0; i < size; i += 16) {
    uint8x16_t idx = vld1q_u8(src + i);
    vst1q_s8(dst + i, vqtbl1q_s8(lut, idx));
  }
}

} // namespace internal


/**
 * @brief Calculates the total size in bytes required for the packed weight buffer.
 *
 * The caller should use this function to allocate a sufficiently large buffer
 * before calling pack_weights_granular.
 */
template<int weight_nbit, int NR>
size_t packed_weights_size(
    int N,
    int K,
    bool has_bias,
    int scale_group_size,
    int lut_group_size,
    int packing_group_size) {

  constexpr int lut_size_per_entry = (1 << weight_nbit);
  constexpr int lut_buffer_size = 16;

  int N_padded = ((N + NR - 1) / NR) * NR;
  int num_scale_groups_per_col = (K + scale_group_size - 1) / scale_group_size;
  int num_lut_groups_per_col = (K + lut_group_size - 1) / lut_group_size;
  int num_packing_groups_per_col = K / packing_group_size;

  size_t packed_qvals_bytes = (size_t)N_padded * K * weight_nbit / 8;
  size_t luts_bytes = (size_t)N_padded * num_lut_groups_per_col * lut_buffer_size;
  size_t scales_bytes = (size_t)N_padded * num_scale_groups_per_col * sizeof(float);
  size_t qvals_sum_bytes = (size_t)N_padded * num_packing_groups_per_col * sizeof(int32_t);
  size_t biases_bytes = has_bias ? (size_t)N_padded * sizeof(float) : 0;

  return packed_qvals_bytes + luts_bytes + scales_bytes + qvals_sum_bytes + biases_bytes;
}


template <int weight_nbit, int NR, int kr, int sr>
void pack_weights_flexible(
    // Output
    void* packed_weights_ptr,
    // Inputs
    const uint8_t* B_qvals, // Shape (K, N)
    const std::vector<float>& weight_scales,
    const std::vector<float>& weight_luts,
    const std::vector<std::array<float, NR>>& biases,
    bool has_bias,
    int N,
    int K,
    int scale_group_size,
    int lut_group_size) {

  // --- 1. Define Fundamental Granularity and Validate Parameters ---
  const int fundamental_group_size = internal::gcd(scale_group_size, lut_group_size);

  static_assert(weight_nbit >= 1 && weight_nbit <= 4, "weight_nbit must be between 1 and 4.");
  assert(K > 0 && N > 0 && packed_weights_ptr != nullptr);
  assert(K % fundamental_group_size == 0 && "K must be a multiple of the fundamental group size (GCD).");
  assert(fundamental_group_size % kr == 0 && "Fundamental group size must be a multiple of kr.");
  assert(scale_group_size % fundamental_group_size == 0 && "Scale group size must be a multiple of the fundamental size.");
  assert(lut_group_size % fundamental_group_size == 0 && "LUT group size must be a multiple of the fundamental size.");

  constexpr int lut_size_per_entry = (1 << weight_nbit);
  constexpr int lut_buffer_size = 16;

  // --- 2. Calculate Memory Layout ---
  int N_padded = ((N + NR - 1) / NR) * NR;
  int num_scale_groups_per_col = K / scale_group_size;
  int num_lut_groups_per_col = K / lut_group_size;
  int num_packing_groups_per_col = K / fundamental_group_size;

  size_t packed_qvals_bytes = (size_t)N_padded * K * weight_nbit / 8;
  size_t luts_bytes = (size_t)N_padded * num_lut_groups_per_col * lut_buffer_size;
  size_t scales_bytes = (size_t)N_padded * num_scale_groups_per_col * sizeof(float);
  size_t qvals_sum_bytes = (size_t)N_padded * num_packing_groups_per_col * sizeof(int32_t);

  // --- 3. Set Up Output Pointers ---
  auto base_ptr = static_cast<char*>(packed_weights_ptr);
  auto packed_qvals_out_ptr = reinterpret_cast<uint8_t*>(base_ptr);
  auto luts_out_ptr = reinterpret_cast<int8_t*>(base_ptr + packed_qvals_bytes);
  auto scales_out_ptr = reinterpret_cast<float*>(base_ptr + packed_qvals_bytes + luts_bytes);
  auto qvals_sum_out_ptr = reinterpret_cast<int32_t*>(base_ptr + packed_qvals_bytes + luts_bytes + scales_bytes);
  float* biases_out_ptr = has_bias ? reinterpret_cast<float*>(base_ptr + packed_qvals_bytes + luts_bytes + scales_bytes + qvals_sum_bytes) : nullptr;

  // --- 4. Allocate Temporary Buffers ---
  std::vector<uint8_t> B_block_col_major(K * NR);
  std::array<int8_t, kr * NR> buffer;
  int8_t packed_values_interleaved[buffer.size()];
  std::array<int8_t, kr> mapped_val_buffer;

  // --- 5. Main Packing Loop ---
  for (int n_start = 0; n_start < N_padded; n_start += NR) {
    // 5.1. Transpose a KxNR block for cache efficiency
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        for (int j = 0; j < NR; ++j) {
            B_block_col_major[j * K + k_idx] = (n_start + j < N) ? B_qvals[k_idx * N + (n_start + j)] : 0;
        }
    }

    // 5.2. Pre-pack all metadata for the KxNR block
    for (int k_meta = 0; k_meta < K; k_meta += kr) { // Iterate at finest grain to be safe
        if (k_meta % scale_group_size == 0) {
            for (int j = 0; j < NR; ++j) {
                if (n_start + j < N) {
                    int scale_group_idx = k_meta / scale_group_size;
                    scales_out_ptr[(n_start + j) * num_scale_groups_per_col + scale_group_idx] = weight_scales[(n_start + j) * num_scale_groups_per_col + scale_group_idx];
                }
            }
        }
        if (k_meta % lut_group_size == 0) {
            for (int j = 0; j < NR; ++j) {
                if (n_start + j < N) {
                    int lut_group_idx = k_meta / lut_group_size;
                    int8_t* lut_dst = luts_out_ptr + ((n_start + j) * num_lut_groups_per_col + lut_group_idx) * lut_buffer_size;
                    const float* lut_src = &weight_luts[((n_start + j) * num_lut_groups_per_col + lut_group_idx) * lut_size_per_entry];
                    internal::promote_and_pack_lut<weight_nbit>(lut_dst, lut_src);
                }
            }
        }
    }

    // 5.3. Process weights and calculate sums group by group
    for (int k_outer = 0; k_outer < K; k_outer += fundamental_group_size) {

      std::array<int32_t, NR> current_qvals_sum;
      current_qvals_sum.fill(0);

      // This inner loop processes one fundamental group
      for (int k_inner = 0; k_inner < fundamental_group_size; k_inner += kr) {
        int k_current = k_outer + k_inner;

        // Select the correct LUT for this kr-sized micro-block. This is robust.
        int lut_group_idx = k_current / lut_group_size;
        int8x16_t lut_for_mapping = vld1q_s8(luts_out_ptr + ((n_start * num_lut_groups_per_col) + lut_group_idx) * lut_buffer_size);

        for (int j = 0; j < NR; j++) {
            std::memcpy(buffer.data() + kr * j, B_block_col_major.data() + j * K + k_current, kr);
            internal::map_values(mapped_val_buffer.data(), (const uint8_t*)(buffer.data() + kr * j), lut_for_mapping, kr);
            current_qvals_sum[j] += reduction::compute_sum(mapped_val_buffer.data(), kr);
        }

        internal::pack_values(packed_values_interleaved, buffer.data(), NR, kr, sr);
        size_t qvals_write_offset = ((size_t)n_start * K + (size_t)k_current * NR) * weight_nbit / 8;
        internal::pack_buffer_for_lut<weight_nbit, kr, NR>(packed_qvals_out_ptr + qvals_write_offset, packed_values_interleaved);
      }

      // Store the final accumulated sum for the fundamental group.
      int packing_group_idx = k_outer / fundamental_group_size;
      for (int j = 0; j < NR; j++) {
        qvals_sum_out_ptr[((n_start + j) * num_packing_groups_per_col) + packing_group_idx] = current_qvals_sum[j];
      }
    }
  }

  if (has_bias) {
    for (int n_block = 0; n_block < N_padded / NR; ++n_block) {
      if (n_block < biases.size()) {
        std::memcpy(biases_out_ptr + n_block * NR, biases[n_block].data(), NR * sizeof(float));
      } else {
        std::memset(biases_out_ptr + n_block * NR, 0, NR * sizeof(float));
      }
    }
  }
}

} // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight::weight_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
