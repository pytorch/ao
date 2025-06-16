// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <cassert>
#include <arm_neon.h>
#include <array>

namespace torchao::kernels::cpu::aarch64::linear::
  groupwise_lowbit_weight_with_lut::kernel {

namespace internal {

template <int MR, int NR>
inline void micro_kernel_lut(
    float32x4_t accum[MR][NR / 4],        // accumulators (C‑tile)
    const float* __restrict__ A,          // activations, size MR×K
    const uint8_t* __restrict__ W,        // packed weights & indices
    int K                                  // depth to iterate over
)
{
    assert(K > 0 && "K must be positive");

    const auto* grp = reinterpret_cast<const TransposedWeightGroup_4bit<NR>*>(W);
    uint8x16x4_t tbl = {
        vreinterpretq_u8_f32(grp->transposed_lut[0]),
        vreinterpretq_u8_f32(grp->transposed_lut[1]),
        vreinterpretq_u8_f32(grp->transposed_lut[2]),
        vreinterpretq_u8_f32(grp->transposed_lut[3])
    };
    const uint8_t* idx_ptr = reinterpret_cast<const uint8_t*>(grp + 1);

    const uint8x16_t ONE   = vdupq_n_u8(1);
    const uint8x16_t TWO   = vdupq_n_u8(2);
    const uint8x16_t THREE = vdupq_n_u8(3);

    for (int k = 0; k < K; ++k)
    {
        // (1) Load activations and broadcast per row.
        float32x4_t a_vec = vld1q_f32(A + k * MR);
        float32x4_t a0 = vdupq_laneq_f32(a_vec, 0);
        float32x4_t a1 = vdupq_laneq_f32(a_vec, 1);
        float32x4_t a2 = vdupq_laneq_f32(a_vec, 2);
        float32x4_t a3 = vdupq_laneq_f32(a_vec, 3);

        // (2) Load and unpack indices, CORRECTING THE ORDER.
        uint8x8_t packed = vld1_u8(idx_ptr);
        idx_ptr += NR / 2;

        uint8x8_t even_indices = vshr_n_u8(packed, 4);
        uint8x8_t odd_indices  = vand_u8(packed, vdup_n_u8(0x0F));

        // **FIX 1: De-interleave indices to restore sequential order.**
        // This is the key step to fix the logical error.
        uint8x8x2_t deinterleaved = vuzp_u8(even_indices, odd_indices);

        // Now combine them into a single 16-byte vector of sequential indices.
        uint8x16_t idx_seq = vcombine_u8(deinterleaved.val[0], deinterleaved.val[1]);

        // Scale to byte addresses inside the LUT.
        uint8x16_t idx_bytes = vshlq_n_u8(idx_seq, 2);

        // (3) Gather 4 byte-planes using the corrected sequential indices.
        uint8x16_t b0 = vqtbl4q_u8(tbl, idx_bytes);
        uint8x16_t b1 = vqtbl4q_u8(tbl, vaddq_u8(idx_bytes, ONE));
        uint8x16_t b2 = vqtbl4q_u8(tbl, vaddq_u8(idx_bytes, TWO));
        uint8x16_t b3 = vqtbl4q_u8(tbl, vaddq_u8(idx_bytes, THREE));

        // **FIX 2: In-register transpose to avoid stack spill.**
        // This is the key step to fix the performance bottleneck.
        // Stage 1: Interleave 8-bit elements within 16-byte registers.
        uint8x16x2_t uzp_b01 = vuzpq_u8(b0, b1); // uzp_b01.val[0] has even bytes, .val[1] has odd bytes
        uint8x16x2_t uzp_b23 = vuzpq_u8(b2, b3);

        // Stage 2: Interleave 16-bit elements.
        uint16x8x2_t trn_16_0 = vtrnq_u16(vreinterpretq_u16_u8(uzp_b01.val[0]), vreinterpretq_u16_u8(uzp_b23.val[0]));
        uint16x8x2_t trn_16_1 = vtrnq_u16(vreinterpretq_u16_u8(uzp_b01.val[1]), vreinterpretq_u16_u8(uzp_b23.val[1]));

        // Stage 3: Interleave 32-bit elements to get the final float vectors.
        float32x4_t w0_3  = vreinterpretq_f32_u32(vtrnq_u32(vreinterpretq_u32_u16(trn_16_0.val[0]), vreinterpretq_u32_u16(trn_16_1.val[0])).val[0]);
        float32x4_t w4_7  = vreinterpretq_f32_u32(vtrnq_u32(vreinterpretq_u32_u16(trn_16_0.val[1]), vreinterpretq_u32_u16(trn_16_1.val[1])).val[0]);
        float32x4_t w8_11 = vreinterpretq_f32_u32(vtrnq_u32(vreinterpretq_u32_u16(trn_16_0.val[0]), vreinterpretq_u32_u16(trn_16_1.val[0])).val[1]);
        float32x4_t w12_15= vreinterpretq_f32_u32(vtrnq_u32(vreinterpretq_u32_u16(trn_16_0.val[1]), vreinterpretq_u32_u16(trn_16_1.val[1])).val[1]);

        // (4) Fused multiply-add. This part remains the same and is now correct.
        accum[0][0] = vfmaq_f32(accum[0][0], w0_3 , a0);
        accum[0][1] = vfmaq_f32(accum[0][1], w4_7 , a0);
        accum[0][2] = vfmaq_f32(accum[0][2], w8_11, a0);
        accum[0][3] = vfmaq_f32(accum[0][3], w12_15, a0);

        accum[1][0] = vfmaq_f32(accum[1][0], w0_3 , a1);
        accum[1][1] = vfmaq_f32(accum[1][1], w4_7 , a1);
        accum[1][2] = vfmaq_f32(accum[1][2], w8_11, a1);
        accum[1][3] = vfmaq_f32(accum[1][3], w12_15, a1);

        accum[2][0] = vfmaq_f32(accum[2][0], w0_3 , a2);
        accum[2][1] = vfmaq_f32(accum[2][1], w4_7 , a2);
        accum[2][2] = vfmaq_f32(accum[2][2], w8_11, a2);
        accum[2][3] = vfmaq_f32(accum[2][3], w12_15, a2);

        accum[3][0] = vfmaq_f32(accum[3][0], w0_3 , a3);
        accum[3][1] = vfmaq_f32(accum[3][1], w4_7 , a3);
        accum[3][2] = vfmaq_f32(accum[3][2], w8_11, a3);
        accum[3][3] = vfmaq_f32(accum[3][3], w12_15, a3);
    }
}

template <int MR, int NR>
inline void post_process_and_store(
    float* __restrict__ output,
    int ldc,
    float32x4_t accum[MR][NR / 4],
    const TransposedWeightGroup_4bit<NR>* __restrict__ grp,
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
void groupwise_lowbit_lut_kernel_T(
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
    constexpr int weight_nbit = 4; // The effective bit-width the kernel sees

    // Determine the fundamental packing group size from the logical group sizes.
    const int packing_group_size = torchao::kernels::cpu::aarch64::common::math::gcd(scale_group_size, lut_group_size);

    assert(n % NR == 0 && "N must be divisible by tile width NR");
    assert(m % MR == 0 && "M must be divisible by tile height MR");
    assert(k % packing_group_size == 0 && "K must be a multiple of the packing group size");

    // --- 2. Get the Memory Layout from the Shared Utility ---
    // This is the single source of truth for all strides.
    auto layout = torchao::kernels::cpu::aarch64::common::layouts::create_fused_lut_layout<NR>(
        k, n, packing_group_size, weight_nbit, promote_to_4bit_layout
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
                micro_kernel_lut<MR, NR>(
                    accumulators,
                    current_packed_activations + k_group_idx * packing_group_size * MR,
                    weights_for_tile_n + k_group_idx * layout.group_stride_bytes,
                    packing_group_size
                );
            }

            // Post-Process and Store the tile
            post_process_and_store<MR, NR>(
                output + m_tile_start * output_m_stride + n_tile_start,
                output_m_stride,
                accumulators,
                // The header for bias is always at the start of the N-tile's weight data.
                reinterpret_cast<const torchao::kernels::cpu::aarch64::common::layouts::FusedLutPackedWeightGroup<NR>*>(weights_for_tile_n),
                has_bias,
                has_clamp,
                clamp_min_vec,
                clamp_max_vec
            );
        }
    }
}


} // torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::kernel

#endif // defined(__aarch64__) || defined(__ARM_NEON)
