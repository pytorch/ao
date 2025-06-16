#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight_with_lut/utils.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace torchao::kernels::cpu::aarch64::linear::
    groupwise_lowbit_weight_with_lut::weight_packing {

namespace internal {
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



void pack_region(uint8_t* dst, const uint8_t* src, size_t count, int nbit) {
  using namespace torchao::bitpacking;
  if (nbit == 4) {
      assert(count % 32 == 0 && "For NEON 4-bit packing, count should be a multiple of 32");
      for (size_t i = 0; i < count; i += 32) {
          // Load and de-interleave 32 source bytes into two 16-byte vectors.
          // in.val[0] will get src[i], src[i+2], ... (low nibbles)
          // in.val[1] will get src[i+1], src[i+3], ... (high nibbles)
          uint8x16x2_t in = vld2q_u8(src + i);

          // Call the highly-optimized packing routine.
          vec_pack_32_lowbit_values<nbit>(dst, in.val[0], in.val[1]);

          // Advance the destination pointer by the number of bytes written.
          dst += 16;
      }
      return;
  }
}

// Use the shared layout definition from the common header
using torchao::kernels::cpu::aarch64::common::layouts::FusedLutPackedWeightGroup;

/**
 * @brief A temporary, packer-side structure for holding a group's metadata
 *        before it is transposed into its final, hardware-friendly format.
 *
 * This struct is designed to be easy to populate with the fused LUT and bias.
 */
template <int NR>
struct PlainMetadataGroup {
    static_assert(NR > 0 && NR % 4 == 0, "NR must be a positive multiple of 4");
    // The LUT is always 16 floats (promoted and fused with scale)
    alignas(16 * sizeof(float)) float fused_lut[16];
    // Bias is a simple array of NR floats.
    float bias[NR];
};

/**
 * @brief (UPDATED) Transposes the easy-to-populate PlainMetadataGroup into the
 *        final FusedLutPackedWeightGroup format for the packed buffer.
 *
 * @param out Pointer to the destination in the packed buffer (the header).
 * @param in Pointer to the temporary, populated metadata group.
 * @param has_bias Flag to indicate if the bias data should be packed.
 */
template <int NR>
inline void transpose_metadata_to_packed_format(
    FusedLutPackedWeightGroup<NR>* out,
    const PlainMetadataGroup<NR>* in,
    bool has_bias)
{
    // 1. Transpose the 16-entry fused LUT.
    // This uses a 4x4 transpose operation on the 16 floats.
    float32x4x4_t loaded_lut = vld4q_f32(in->fused_lut);
    out->transposed_lut[0] = loaded_lut.val[0];
    out->transposed_lut[1] = loaded_lut.val[1];
    out->transposed_lut[2] = loaded_lut.val[2];
    out->transposed_lut[3] = loaded_lut.val[3];

    // 2. Block the bias into NEON vectors.
    if (has_bias) {
        for (int i = 0; i < (NR / 4); ++i) {
            out->bias[i] = vld1q_f32(&in->bias[i * 4]);
        }
    }
}

} // namespace internal

/**
 * @brief Calculates the total size by delegating to the shared layout factory.
 */
 template<int NR>
 size_t packed_weights_size_for_fused_lut_kernel(
     int N, int K, bool has_bias, int scale_group_size, int lut_group_size,
     int weight_nbit,
     bool promote_to_4bit_layout) {

     // The sizer's only job is to create the layout and return the total size.
     FusedLutPackedLayout layout = internal::create_fused_lut_layout<NR>(
         N, K, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout
     );

     return layout.total_buffer_size;
}



/**
* @brief Packs weights by pre-fusing the scale into the LUT,
*        using a shared layout definition for all size calculations.
*/
/**
* @brief (Final Corrected) Packs weights by pre-fusing the scale into the LUT.
*/
template <int weight_nbit, int NR>
void pack_weights_with_fused_lut(
    void* packed_weights_ptr,
    const uint8_t* B_qvals,
    const std::vector<float>& weight_scales,
    const std::vector<float>& weight_luts,
    const std::vector<float>& biases,
    bool has_bias,
    int N, int K, int scale_group_size, int lut_group_size,
    bool promote_to_4bit_layout) {

    namespace packing_internal = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight::weight_packing::internal;

    // --- 1. Get the single source of truth for the layout ---
    FusedLutPackedLayout layout = create_fused_lut_layout<NR>(
        N, K, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout);

    // --- 2. Define Packing Granularity and Constants ---
    const int packing_group_size = std::gcd(scale_group_size, lut_group_size);
    assert(K % packing_group_size == 0);

    // This is the number of entries in the SOURCE LUT. It's always based on the original weight_nbit.
    constexpr int src_lut_size_per_entry = (1 << weight_nbit);

    // --- 3. Allocate Temporary Buffers ---
    packing_internal::PlainMetadataGroup<NR> temp_group = {}; // Reusable temporary group
    std::vector<uint8_t> B_block_col_major(K * NR);
    std::vector<uint8_t> indices_to_pack(packing_group_size * NR);

    // --- 4. Main Packing Loop ---
    char* out_ptr = static_cast<char*>(packed_weights_ptr);
    const int N_padded = ((N + NR - 1) / NR) * NR;

    for (int n_start = 0; n_start < N_padded; n_start += NR) {
        // 4.1. Transpose a KxNR block for cache efficiency
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            for (int j = 0; j < NR; ++j) {
                B_block_col_major[j * K + k_idx] = (n_start + j < N) ? B_qvals[k_idx * N + (n_start + j)] : 0;
            }
        }

        // 4.2. Process the block in packing_group_size chunks
        for (int k_group_start = 0; k_group_start < K; k_group_start += packing_group_size) {
            // --- Fusing Logic ---
            int weight_1d_idx = n_start * K + k_group_start;
            int scale_idx = weight_1d_idx / scale_group_size;
            int lut_idx = weight_1d_idx / lut_group_size;
            float scale = weight_scales.empty() ? 1.0f : weight_scales[scale_idx];
            // The source LUT pointer calculation correctly uses the original bit width.
            const float* lut_src = weight_luts.data() + lut_idx * src_lut_size_per_entry;

            // Populate the temporary metadata group
            std::fill(std::begin(temp_group.fused_lut), std::end(temp_group.fused_lut), 0.0f);
            for (int i = 0; i < src_lut_size_per_entry; ++i) {
                temp_group.fused_lut[i] = scale * lut_src[i];
            }
            if (has_bias) {
                std::copy(&biases[n_start], &biases[n_start + NR], temp_group.bias);
            }

            // A. Pack the header
            auto* header_dst = reinterpret_cast<FusedLutPackedWeightGroup<NR>*>(out_ptr);
            packing_internal::transpose_metadata_to_packed_format<NR>(header_dst, &temp_group, has_bias);

            // B. Gather indices for this group
            auto* indices_dst = out_ptr + layout.header_bytes_per_group;
            for (int k_offset = 0; k_offset < packing_group_size; ++k_offset) {
                for (int nr_idx = 0; nr_idx < NR; ++nr_idx) {
                    indices_to_pack[k_offset * NR + nr_idx] =
                        B_block_col_major[(nr_idx * K) + (k_group_start + k_offset)];
                }
            }

            // C. Call the bitpacking routine with the correct effective bit width
            int effective_bit_width = promote_to_4bit_layout ? 4 : weight_nbit;
            pack_region_scalar(indices_dst, indices_to_pack.data(), packing_group_size * NR, effective_bit_width)
            // D. Advance the main output pointer
            out_ptr += layout.size_of_one_physical_group;
        }
    }
}

} // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight::weight_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
