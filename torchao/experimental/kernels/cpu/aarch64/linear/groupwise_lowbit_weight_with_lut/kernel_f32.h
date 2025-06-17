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

template <int MR, int NR>
inline void micro_kernel_lut(
    float32x4_t accum[MR][NR / 4],
    const float* __restrict__ A,
    const uint8_t* __restrict__ W,
    int K)
{
    assert(K > 0 && "K must be positive");
    namespace utils = torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::utils;

    const auto* grp = reinterpret_cast<const utils::FusedLutPackedWeightGroup<NR>*>(W);
    const uint8_t* idx_ptr = reinterpret_cast<const uint8_t*>(grp + 1);

    uint8x16x4_t tbl;
    memcpy(tbl.val, grp->lut_soa_planes, 64);

    const float* a_ptr = A; // A pointer to the start of the (K x MR) tile for this group

    for (int k_idx = 0; k_idx < K; ++k_idx) {


        uint8x8_t packed_neon = vld1_u8(idx_ptr);
        uint8x8_t low_nibbles  = vand_u8(packed_neon, vdup_n_u8(0x0F));
        uint8x8_t high_nibbles = vshr_n_u8(packed_neon, 4);
        uint8x8x2_t interleaved = vzip_u8(low_nibbles, high_nibbles);
        uint8x16_t unpacked_indices_neon = vcombine_u8(interleaved.val[0], interleaved.val[1]);

        const uint8x16_t SIXTEEN = vdupq_n_u8(16);
        const uint8x16_t THIRTY_TWO = vdupq_n_u8(32);
        const uint8x16_t FORTY_EIGHT = vdupq_n_u8(48);
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


template <int MR, int NR>
inline void post_process_and_store(
    float* __restrict__ output,
    int ldc,
    float32x4_t accum[MR][NR / 4],
    const utils::FusedLutPackedWeightGroup<NR>* __restrict__ grp,
    bool has_bias,
    bool has_clamp,
    float32x4_t clamp_min_vec,
    float32x4_t clamp_max_vec)
{
    constexpr int NR_VEC = NR / 4;
    for (int m = 0; m < MR; ++m) {
        float* out_row = output + m * ldc;
        for (int nb = 0; nb < NR_VEC; ++nb) {
            float32x4_t res = accum[m][nb];
            if (has_bias) {
                // *** CORRECTED BIAS LOADING ***
                res = vaddq_f32(res, grp->bias[nb]);
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


template <int MR, int NR>
void groupwise_lowbit_lut_kernel(
    float* output,
    int output_m_stride,
    int m, int n, int k,
    // Use clear, logical names for group sizes
    int scale_group_size,
    int lut_group_size,
    const void* packed_weights,
    const void* packed_activations,
    float clamp_min, float clamp_max,
    bool has_bias, bool has_clamp) {

    // --- 1. Define Kernel Parameters and Validate ---
    // This kernel uses the "promote to 4-bit" strategy for simplicity and speed.
    constexpr bool promote_to_4bit_layout = true;
    constexpr int weight_nbit = 4;

    const int packing_group_size = std::gcd(scale_group_size, lut_group_size);

    assert(n % NR == 0 && "N must be divisible by tile width NR");
    assert(m % MR == 0 && "M must be divisible by tile height MR");
    assert(k % packing_group_size == 0 && "K must be a multiple of the packing group size");

    // --- 2. Get the Memory Layout from the Shared Utility ---
    // This is the single source of truth for all strides.
    auto layout = utils::create_fused_lut_layout<NR>(
        n, k, packing_group_size,packing_group_size, weight_nbit, promote_to_4bit_layout
    );

    // --- 3. Pre-computation and Main Loops ---
    const float32x4_t clamp_min_vec = vdupq_n_f32(clamp_min);
    const float32x4_t clamp_max_vec = vdupq_n_f32(clamp_max);

    const int num_groups_per_k_tile = k / packing_group_size;

    for (int m_tile_start = 0; m_tile_start < m; m_tile_start += MR) {
        for (int n_tile_start = 0; n_tile_start < n; n_tile_start += NR) {

            float32x4_t accumulators[MR][NR / 4] = {{0}};

            // Calculate base pointers for the current tile using the layout strides
            const auto* current_packed_activations = static_cast<const float*>(packed_activations) + m_tile_start * k;
            const auto* weights_for_tile_n = static_cast<const uint8_t*>(packed_weights) + (n_tile_start / NR) * layout.n_tile_stride_bytes;

            // K-Loop: Accumulate over the K dimension
            for (int k_group_idx = 0; k_group_idx < num_groups_per_k_tile; ++k_group_idx) {
                // The micro-kernel is called with a pointer to the start of the current group.
                // The stride is obtained from the layout object.
                internal::micro_kernel_lut<MR, NR>(
                    accumulators,
                    current_packed_activations + k_group_idx * packing_group_size,
                    weights_for_tile_n + k_group_idx * layout.group_stride_bytes,
                    packing_group_size
                );
            }

            internal::post_process_and_store<MR, NR>(
                output + m_tile_start * output_m_stride + n_tile_start,
                output_m_stride,
                accumulators,
                reinterpret_cast<const utils::FusedLutPackedWeightGroup<NR>*>(weights_for_tile_n),
                has_bias,
                has_clamp,
                clamp_min_vec,
                clamp_max_vec
            );
        }
    }
}


} // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::kernel

#endif // defined(__aarch64__) || defined(__ARM_NEON)
