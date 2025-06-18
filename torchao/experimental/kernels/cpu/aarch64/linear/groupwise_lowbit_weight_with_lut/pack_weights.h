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
 * @brief Pack 2 4-bit indices into 1 byte.
*/
void pack_region(uint8_t* dst, const uint8_t* src, size_t count, int nbit) {
    if (nbit != 4) {
        return;
    }

    assert(count % 16 == 0);

    for (size_t i = 0; i < count; i += 16) {
        const uint8_t* current_src = src + i;
        uint8_t* current_dst = dst + (i / 2);

        // 1. Load: [i0, i1, i2, i3, ...]
        uint8x16_t v_src = vld1q_u8(current_src);

        // 2. De-interleave:
        //    p.val[0] = [i0, i2, i4, ...] (even indices)
        //    p.val[1] = [i1, i3, i5, ...] (odd indices)
        uint8x16x2_t p = vuzpq_u8(v_src, v_src);

        // 3. vshlq_n_u8(p.val[0], 4) results in: [(i0<<4), (i2<<4), (i4<<4), ...]
        uint8x16_t high_nibbles = vshlq_n_u8(p.val[0], 4);

        // 4. result_16_bytes = [(i0<<4)|i1, (i2<<4)|i3, ..., (i14<<4)|i15]
        uint8x16_t result_16_bytes = vorrq_u8(p.val[1], high_nibbles);

        // 5. Store the 8 packed bytes.
        vst1_u8(current_dst, vget_low_u8(result_16_bytes));
    }
}

namespace utils = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::utils;

/**
 * @brief A packer-side structure for holding a group's metadata
 *        before it is transposed into its final, hardware-friendly format.
*/
template <int NR>
struct PlainMetadataGroup {
    alignas(16 * sizeof(float)) float fused_lut[16];
};

template <int NR>
inline void transpose_metadata_to_packed_format(
    utils::FusedLutPackedWeightGroup<NR>* out,
    const PlainMetadataGroup<NR>* in)
{
    const uint8_t* lut_ptr = reinterpret_cast<const uint8_t*>(in->fused_lut);
    uint8x16x4_t soa_lut = vld4q_u8(lut_ptr);

    out->lut_soa_planes[0] = soa_lut.val[0];
    out->lut_soa_planes[1] = soa_lut.val[1];
    out->lut_soa_planes[2] = soa_lut.val[2];
    out->lut_soa_planes[3] = soa_lut.val[3];
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
    const uint8_t* B_qvals, // Expected in (K, N) layout
    const std::vector<float>& weight_scales,
    const std::vector<float>& weight_luts,
    int N, int K, int scale_group_size, int lut_group_size,
    bool promote_to_4bit_layout) {

    namespace packing_internal = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::weight_packing::internal;
    namespace utils = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::utils;

    // --- 1. Get layout and constant ---
    utils::FusedLutPackedLayout layout = utils::create_fused_lut_layout<NR>(
        N, K, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout);
    const int packing_group_size = std::gcd(scale_group_size, lut_group_size);
    constexpr int src_lut_size_per_entry = (1 << weight_nbit);

    // --- 2. Allocate temporary buffers ---
    packing_internal::PlainMetadataGroup<NR> temp_group = {};
    std::vector<uint8_t> indices_to_pack(packing_group_size * NR);

    // --- 3. Main packing logic ---
    char* out_ptr = static_cast<char*>(packed_weights_ptr);
    const int N_padded = ((N + NR - 1) / NR) * NR;

    for (int n_start = 0; n_start < N_padded; n_start += NR) {
        for (int k_group_start = 0; k_group_start < K; k_group_start += packing_group_size) {

            int32_t scale_idx = k_group_start / scale_group_size;
            int32_t lut_idx   = k_group_start / lut_group_size;

            float scale = weight_scales.empty() ? 1.0f : weight_scales[scale_idx];
            const float* lut_src = weight_luts.data() + lut_idx * src_lut_size_per_entry;

            for (int i = 0; i < src_lut_size_per_entry; ++i) {
                temp_group.fused_lut[i] = scale * lut_src[i];
            }

            for (int k_offset = 0; k_offset < packing_group_size; ++k_offset) {
                for (int nr_idx = 0; nr_idx < NR; ++nr_idx) {
                    const int current_k = k_group_start + k_offset;
                    const int current_n = n_start + nr_idx;
                    if (current_n < N) {
                        indices_to_pack[k_offset * NR + nr_idx] = B_qvals[current_k * N + current_n];
                    } else {
                        indices_to_pack[k_offset * NR + nr_idx] = 0; // Padding
                    }
                }
            }

            auto* header_dst = reinterpret_cast<utils::FusedLutPackedWeightGroup<NR>*>(out_ptr);
            packing_internal::transpose_metadata_to_packed_format<NR>(header_dst, &temp_group);

            auto* indices_dst = out_ptr + layout.header_bytes_per_group;
            int effective_bit_width = promote_to_4bit_layout ? 4 : weight_nbit;
            packing_internal::pack_region(
                reinterpret_cast<uint8_t*>(indices_dst),
                indices_to_pack.data(),
                packing_group_size * NR,
                effective_bit_width);

            out_ptr += layout.group_stride_bytes;
        }
    }
}

} // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight::weight_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
