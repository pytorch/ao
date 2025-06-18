// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight_with_lut/utils.h>
#include <cassert>
#include <arm_neon.h>
#include <array>

namespace torchao::kernels::cpu::aarch64::linear::
  groupwise_lowbit_weight_with_lut::kernel {


namespace internal {

inline void compute_4x16_4bit_promoted(
    float32x4_t accum[4][4],
    const float* __restrict__ activation,
    const uint8_t* __restrict__ weight_indices,
    int K,
    const uint8x16x4_t& tbl)
{
    constexpr int MR = 4;
    constexpr int NR = 16;
    assert(K > 0 && "K must be positive");
    namespace utils = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::utils;

    const uint8x16_t SIXTEEN = vdupq_n_u8(16);
    const uint8x16_t THIRTY_TWO = vdupq_n_u8(32);
    const uint8x16_t FORTY_EIGHT = vdupq_n_u8(48);
    const uint8_t* idx_ptr = weight_indices;
    const float* a_ptr = activation;

    for (int k_idx = 0; k_idx < K; ++k_idx) {

        uint8x8_t packed_neon = vld1_u8(idx_ptr);
        // Unpack the 8-bit indices into 16 4-bit indices.
        uint8x8x2_t interleaved = vzip_u8(vshr_n_u8(packed_neon, 4), vand_u8(packed_neon, vdup_n_u8(0x0F)));
        uint8x16_t unpacked_indices_neon = vcombine_u8(interleaved.val[0], interleaved.val[1]);

        uint8x16_t idx_plane0 = unpacked_indices_neon;
        uint8x16_t idx_plane1 = vaddq_u8(unpacked_indices_neon, SIXTEEN);
        uint8x16_t idx_plane2 = vaddq_u8(unpacked_indices_neon, THIRTY_TWO);
        uint8x16_t idx_plane3 = vaddq_u8(unpacked_indices_neon, FORTY_EIGHT);

        uint8x16_t b0 = vqtbl4q_u8(tbl, idx_plane0);
        uint8x16_t b1 = vqtbl4q_u8(tbl, idx_plane1);
        uint8x16_t b2 = vqtbl4q_u8(tbl, idx_plane2);
        uint8x16_t b3 = vqtbl4q_u8(tbl, idx_plane3);

        uint8x16x2_t zip_b01 = vzipq_u8(b0, b1);
        uint8x16x2_t zip_b23 = vzipq_u8(b2, b3);
        uint16x8x2_t trn_16_0 = vtrnq_u16(vreinterpretq_u16_u8(zip_b01.val[0]), vreinterpretq_u16_u8(zip_b23.val[0]));
        uint16x8x2_t trn_16_1 = vtrnq_u16(vreinterpretq_u16_u8(zip_b01.val[1]), vreinterpretq_u16_u8(zip_b23.val[1]));
        float32x4x2_t final_zip_0 = vzipq_f32(vreinterpretq_f32_u16(trn_16_0.val[0]), vreinterpretq_f32_u16(trn_16_0.val[1]));
        float32x4x2_t final_zip_1 = vzipq_f32(vreinterpretq_f32_u16(trn_16_1.val[0]), vreinterpretq_f32_u16(trn_16_1.val[1]));

        float32x4_t w0_3   = final_zip_0.val[0];
        float32x4_t w4_7   = final_zip_0.val[1];
        float32x4_t w8_11  = final_zip_1.val[0];
        float32x4_t w12_15 = final_zip_1.val[1];

        float32x4_t a_col = vld1q_f32(a_ptr);

        float32x4_t a0 = vdupq_laneq_f32(a_col, 0);
        float32x4_t a1 = vdupq_laneq_f32(a_col, 1);
        float32x4_t a2 = vdupq_laneq_f32(a_col, 2);
        float32x4_t a3 = vdupq_laneq_f32(a_col, 3);

        accum[0][0] = vfmaq_f32(accum[0][0], w0_3,   a0);
        accum[0][1] = vfmaq_f32(accum[0][1], w4_7,   a0);
        accum[0][2] = vfmaq_f32(accum[0][2], w8_11,  a0);
        accum[0][3] = vfmaq_f32(accum[0][3], w12_15, a0);

        accum[1][0] = vfmaq_f32(accum[1][0], w0_3,   a1);
        accum[1][1] = vfmaq_f32(accum[1][1], w4_7,   a1);
        accum[1][2] = vfmaq_f32(accum[1][2], w8_11,  a1);
        accum[1][3] = vfmaq_f32(accum[1][3], w12_15, a1);

        accum[2][0] = vfmaq_f32(accum[2][0], w0_3,   a2);
        accum[2][1] = vfmaq_f32(accum[2][1], w4_7,   a2);
        accum[2][2] = vfmaq_f32(accum[2][2], w8_11,  a2);
        accum[2][3] = vfmaq_f32(accum[2][3], w12_15, a2);

        accum[3][0] = vfmaq_f32(accum[3][0], w0_3,   a3);
        accum[3][1] = vfmaq_f32(accum[3][1], w4_7,   a3);
        accum[3][2] = vfmaq_f32(accum[3][2], w8_11,  a3);
        accum[3][3] = vfmaq_f32(accum[3][3], w12_15, a3);

        a_ptr += MR;
        idx_ptr += (NR / 2);
    }
}

template <int WEIGHT_NBIT>
inline void micro_kernel_lut_4x16(
    float32x4_t accum[4][4],
    const float* __restrict__ A,
    const uint8_t* __restrict__ W,
    int K_group_size)
{
    static_assert(WEIGHT_NBIT >= 1 && WEIGHT_NBIT <= 4, "WEIGHT_NBIT must be 1-4");
    namespace utils = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::utils;

    // 1. Get pointers to the packed data components
    const auto* grp = reinterpret_cast<const utils::FusedLutPackedWeightGroup<16>*>(W);
    const uint8_t* indices_ptr = reinterpret_cast<const uint8_t*>(grp + 1);

    // 2. Perform LUT expansion
    uint8x16x4_t tbl;
    if constexpr (WEIGHT_NBIT < 4) {
        const int src_lut_size = 1 << WEIGHT_NBIT;
        alignas(16) uint8_t expanded_lut_soa[64];
        for (int plane = 0; plane < 4; ++plane) {
            const uint8_t* src_plane = reinterpret_cast<const uint8_t*>(&grp->lut_soa_planes[plane]);
            uint8_t* dst_plane = &expanded_lut_soa[plane * 16];
            for (int i = 0; i < 16; ++i) {
                dst_plane[i] = src_plane[i % src_lut_size];
            }
        }
        memcpy(&tbl, expanded_lut_soa, 64);
    } else { // WEIGHT_NBIT == 4
        memcpy(&tbl, grp->lut_soa_planes, 64);
    }

    // 3. Call the pure compute kernel with the prepared LUT
    compute_4x16_4bit_promoted(
        accum,
        A,
        indices_ptr,
        K_group_size,
        tbl);
}

inline void post_process_and_store_4x16(
    float* __restrict__ output,
    int ldc,
    float32x4_t accum[4][4],
    const float* __restrict__ bias_ptr,
    bool has_clamp,
    float32x4_t clamp_min_vec,
    float32x4_t clamp_max_vec)
{
    constexpr int MR = 4;
    constexpr int NR = 16;
    constexpr int NR_VEC = NR / 4;

    for (int m = 0; m < MR; ++m) {
        float* out_row = output + m * ldc;
        for (int nb = 0; nb < NR_VEC; ++nb) {
            float32x4_t res = accum[m][nb];

            if (bias_ptr != nullptr) {
                float32x4_t bias_vec = vld1q_f32(bias_ptr + nb * 4);
                res = vaddq_f32(res, bias_vec);
            }

            if (has_clamp) {
                res = vmaxq_f32(res, clamp_min_vec);
                res = vminq_f32(res, clamp_max_vec);
            }
            vst1q_f32(out_row + nb * 4, res);
        }
    }
}
} // namespace internal


/**
 * @brief Computes a group-wise low-bit GEMM using the 4x16 fused LUT kernel.
 *
 * It assumes activations have been pre-packed by `pack_activations(..., MR=4)`
 * and weights by `pack_weights(..., NR=16)`.
 */
void groupwise_lowbit_lut_kernel_4x16(
    float* output,
    int output_m_stride,
    int m, int n, int k,
    int scale_group_size,
    int lut_group_size,
    const void* packed_weights,
    const void* packed_activations, const float* biases,
    float clamp_min, float clamp_max,
    bool has_bias, bool has_clamp, int weight_nbit) {

    // --- 1. Define kernel parameters ---
    constexpr int MR = 4;
    constexpr int NR = 16;
    constexpr bool promote_to_4bit_layout = true;

    const int packing_group_size = std::gcd(scale_group_size, lut_group_size);

    assert(n % NR == 0 && "N must be divisible by tile width NR");
    assert(m % MR == 0 && "M must be divisible by tile height MR");
    assert(k % packing_group_size == 0 && "K must be a multiple of the packing group size");

    // --- 2. Get the memory layout ---
    auto layout = utils::create_fused_lut_layout<NR>(
        n, k, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout
    );

    // --- 3. Main loop ---
    const float32x4_t clamp_min_vec = vdupq_n_f32(clamp_min);
    const float32x4_t clamp_max_vec = vdupq_n_f32(clamp_max);

    const int num_groups_per_k_tile = k / packing_group_size;

    for (int m_tile_start = 0; m_tile_start < m; m_tile_start += MR) {

        const size_t activation_tile_size = (size_t)MR * k;

        // Calculate pointer by advancing by the number of tiles to skip.
        const auto* current_packed_activations = static_cast<const float*>(packed_activations) + (m_tile_start / MR) * activation_tile_size;

        for (int n_tile_start = 0; n_tile_start < n; n_tile_start += NR) {

            float32x4_t accumulators[MR][NR / 4] = {{0}};


            const auto* weights_for_tile_n = static_cast<const uint8_t*>(packed_weights) + (n_tile_start / NR) * layout.n_tile_stride_bytes;
            for (int k_group_idx = 0; k_group_idx < num_groups_per_k_tile; ++k_group_idx) {
                // Calculate the starting column index for this group
                const int k_group_start = k_group_idx * packing_group_size;

                // A_group_ptr holds the pointer to the relevant slice of the activation data
                const float* A_group_ptr = current_packed_activations + k_group_start * MR;

                // W_group_ptr holds the pointer to the relevant packed weight group
                const uint8_t* W_group_ptr = weights_for_tile_n + k_group_idx * layout.group_stride_bytes;

                switch (weight_nbit) {
                    case 4:
                        internal::micro_kernel_lut_4x16<4>(
                            accumulators, A_group_ptr, W_group_ptr, packing_group_size);
                        break;
                    case 3:
                        internal::micro_kernel_lut_4x16<3>(
                            accumulators, A_group_ptr, W_group_ptr, packing_group_size);
                        break;
                    case 2:
                        internal::micro_kernel_lut_4x16<2>(
                            accumulators, A_group_ptr, W_group_ptr, packing_group_size);
                        break;
                    case 1:
                        internal::micro_kernel_lut_4x16<1>(
                            accumulators, A_group_ptr, W_group_ptr, packing_group_size);
                        break;
                    default:
                        throw std::invalid_argument("Unsupported weight_nbit in kernel.");
                }
            }
            internal::post_process_and_store_4x16(
                output + m_tile_start * output_m_stride + n_tile_start,
                output_m_stride,
                accumulators,
                has_bias ? biases + n_tile_start : nullptr,
                has_clamp,
                clamp_min_vec,
                clamp_max_vec
            );
        }
    }
}

} // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::kernel

#endif // defined(__aarch64__) || defined(__ARM_NEON)
