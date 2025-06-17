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

void pack_region(uint8_t* dst, const uint8_t* src, size_t count, int nbit) {
  using namespace torchao::bitpacking;
  if (nbit == 4) {
    assert(count % 32 == 0 && "For NEON 4-bit packing, count should be a multiple of 32");
    for (size_t i = 0; i < count; i += 32) {
        uint8x16x2_t in = vld2q_u8(src + i);

        // FIX: Use the compile-time constant '4'
        torchao::bitpacking::vec_pack_32_lowbit_values<4>(dst, in.val[0], in.val[1]);

        dst += 16;
    }
    return;
}
}

namespace utils = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::utils;

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
 * @brief Transposes the easy-to-populate PlainMetadataGroup into the
 *        final FusedLutPackedWeightGroup format for the packed buffer.
 *
 * @param out Pointer to the destination in the packed buffer (the header).
 * @param in Pointer to the temporary, populated metadata group.
 * @param has_bias Flag to indicate if the bias data should be packed.
 */
template <int NR>
inline void transpose_metadata_to_packed_format(
    utils::FusedLutPackedWeightGroup<NR>* out,
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
 template<int weight_nbit, int NR>
 size_t packed_weights_size_for_fused_lut_kernel(
     int N, int K, bool has_bias, int scale_group_size, int lut_group_size,
     bool promote_to_4bit_layout) {

     // The sizer's only job is to create the layout and return the total size.
     utils::FusedLutPackedLayout layout = utils::create_fused_lut_layout<NR>(
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

    namespace packing_internal = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::weight_packing::internal;

    // --- 1. Get the single source of truth for the layout ---
    utils::FusedLutPackedLayout layout = utils::create_fused_lut_layout<NR>(
        N, K, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout);

    // --- 2. Define Packing Granularity and Constants ---
    const int packing_group_size = std::gcd(scale_group_size, lut_group_size);
    assert(K % packing_group_size == 0);

    // This is the number of entries in the SOURCE LUT. It's always based on the original weight_nbit.
    constexpr int src_lut_size_per_entry = (1 << weight_nbit);

    // --- 3. Allocate Temporary Buffers ---
    packing_internal::PlainMetadataGroup<NR> temp_group = {};
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
            auto* header_dst = reinterpret_cast<utils::FusedLutPackedWeightGroup<NR>*>(out_ptr);
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
            packing_internal::pack_region(    reinterpret_cast<uint8_t*>(indices_dst), indices_to_pack.data(), packing_group_size * NR, effective_bit_width);
            // D. Advance the main output pointer
            out_ptr += layout.group_stride_bytes;
        }
    }
}

} // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight::weight_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
