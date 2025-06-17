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
    if (nbit != 4) {
        // This routine is specialized for 4-bit packing.
        return;
    }

    assert(count % 16 == 0 && "Count must be a multiple of 16 for this 4-bit pack routine.");

    for (size_t i = 0; i < count; i += 16) {
        const uint8_t* current_src = src + i;
        // Destination pointer advances by 8 for every 16 source bytes.
        uint8_t* current_dst = dst + (i / 2);

        // 1. Load one 16-byte vector from the source.
        // v_src = [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15]
        uint8x16_t v_src = vld1q_u8(current_src);

        // 2. De-interleave the vector to separate even-indexed and odd-indexed bytes.
        // p.val[0] will get all the even-indexed bytes: [i0, i2, i4, ..., i14]
        // p.val[1] will get all the odd-indexed bytes:  [i1, i3, i5, ..., i15]
        uint8x16x2_t p = vuzpq_u8(v_src, v_src);

        // 3. Shift the upper nibbles (the odd indices) into position.
        // vshlq_n_u8(p.val[1], 4) results in: [(i1<<4), (i3<<4), (i5<<4), ...]
        uint8x16_t upper_nibbles = vshlq_n_u8(p.val[1], 4);

        // 4. Combine the lower and upper nibbles with a bitwise OR.
        // We only care about the first 8 bytes of the result, as that's where
        // the 8 pairs are formed.
        // result_16_bytes = [(i1<<4)|i0, (i3<<4)|i2, ..., (i15<<4)|i14]
        uint8x16_t result_16_bytes = vorrq_u8(p.val[0], upper_nibbles);

        // 5. Store the 8 packed bytes.
        // We only need the first half of our 16-byte result vector.
        // vget_low_u8 extracts the lower 8 bytes.
        vst1_u8(current_dst, vget_low_u8(result_16_bytes));
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

template <int NR>
inline void transpose_metadata_to_packed_format(
    utils::FusedLutPackedWeightGroup<NR>* out,
    const PlainMetadataGroup<NR>* in,
    bool has_bias)
{
    {
        // 1a. Load the 16 floats (64 bytes total) as four 16-byte raw byte vectors.
        const uint8_t* lut_ptr = reinterpret_cast<const uint8_t*>(in->fused_lut);
        uint8x16_t b0 = vld1q_u8(lut_ptr + 0);
        uint8x16_t b1 = vld1q_u8(lut_ptr + 16);
        uint8x16_t b2 = vld1q_u8(lut_ptr + 32);
        uint8x16_t b3 = vld1q_u8(lut_ptr + 48);

        // 1b. Perform a 4x4 matrix transpose on the bytes using a two-stage de-interleave.
        uint8x16x2_t t01 = vuzpq_u8(b0, b1);
        uint8x16x2_t t23 = vuzpq_u8(b2, b3);
        uint8x16x2_t p02 = vuzpq_u8(t01.val[0], t23.val[0]);
        uint8x16x2_t p13 = vuzpq_u8(t01.val[1], t23.val[1]);

        // 1c. Store the resulting four byte-planes directly into the output struct.
        out->lut_soa_planes[0] = p02.val[0];
        out->lut_soa_planes[1] = p13.val[0];
        out->lut_soa_planes[2] = p02.val[1];
        out->lut_soa_planes[3] = p13.val[1];
    }
    if (has_bias) {
        memcpy(out->bias, in->bias, NR * sizeof(float));
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
    namespace utils = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::utils; // Make sure you have this

    // --- 1. Get the single source of truth for the layout ---
    utils::FusedLutPackedLayout layout = utils::create_fused_lut_layout<NR>(
        N, K, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout);

    // --- 2. Define Packing Granularity and Constants ---
    const int packing_group_size = std::gcd(scale_group_size, lut_group_size);
    assert(K % packing_group_size == 0);
    constexpr int src_lut_size_per_entry = (1 << weight_nbit);

    // --- 3. Allocate Temporary Buffers ---
    packing_internal::PlainMetadataGroup<NR> temp_group = {};
    std::vector<uint8_t> B_block_col_major(K * NR);
    std::vector<uint8_t> indices_to_pack(packing_group_size * NR);

    // --- 4. Main Packing Loop ---
    char* out_ptr = static_cast<char*>(packed_weights_ptr);
    const int N_padded = ((N + NR - 1) / NR) * NR;

    for (int n_start = 0; n_start < N_padded; n_start += NR) {
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            for (int j = 0; j < NR; ++j) {
                B_block_col_major[j * K + k_idx] = (n_start + j < N) ? B_qvals[k_idx * N + (n_start + j)] : 0;
            }
        }

        for (int k_group_start = 0; k_group_start < K; k_group_start += packing_group_size) {
            int weight_1d_idx = n_start * K + k_group_start;
            int scale_idx = weight_1d_idx / scale_group_size;
            int lut_idx = weight_1d_idx / lut_group_size;
            float scale = weight_scales.empty() ? 1.0f : weight_scales[scale_idx];
            const float* lut_src = weight_luts.data() + lut_idx * src_lut_size_per_entry;

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

            // C. Call the bitpacking routine
            int effective_bit_width = promote_to_4bit_layout ? 4 : weight_nbit;
            packing_internal::pack_region(reinterpret_cast<uint8_t*>(indices_dst), indices_to_pack.data(), packing_group_size * NR, effective_bit_width);

            // D. Advance the main output pointer
            out_ptr += layout.group_stride_bytes;
        }
    }
}
} // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight::weight_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
