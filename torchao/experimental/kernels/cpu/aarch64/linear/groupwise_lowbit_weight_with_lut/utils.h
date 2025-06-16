#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace torchao::kernels::cpu::aarch64::linear::
    groupwise_lowbit_weight_with_lut::utils {


/**
 * @brief Defines the memory layout for a block of group-wise quantized
 *        weights with a pre-fused, transposed Look-Up Table (LUT).
 *
 * This structure is designed for consumption by high-performance NEON kernels.
 * The LUT is transposed to enable efficient loading with `vld4q_f32`, and the
 * bias is blocked to align with the kernel's `4xNR` processing tiles.
 *
 * @tparam NR The column-tiling factor of the kernel (e.g., 16).
 */
 template <int NR>
 struct FusedLutPackedWeightGroup {
     static_assert(NR > 0 && NR % 4 == 0, "NR must be a positive multiple of 4");
     constexpr static int NR_VEC = NR / 4;

     // Transposed LUT for 4-bit indices.
     // L0 = [lut[0], lut[4], lut[8],  lut[12]]
     // L1 = [lut[1], lut[5], lut[9],  lut[13]]
     // etc.
     float32x4_t transposed_lut[4];

     // Bias blocked into 4-element vectors.
     float32x4_t bias[NR_VEC];
 };



struct FusedLutPackedLayout {
    // --- Per-group sizes (Strides within a physical group) ---
    size_t header_bytes_per_group;
    size_t packed_indices_bytes_per_group;

    // --- Strides between physical groups ---
    size_t group_stride_bytes; // Stride between k-groups for the same n-tile
    size_t n_tile_stride_bytes; // Stride between n-tiles

    // --- Total Size ---
    size_t total_buffer_size;
};

// The factory function now also calculates strides
template<int NR>
inline FusedLutPackedLayout create_fused_lut_layout(
    int K, int N, int packing_group_size,
    int weight_nbit, bool promote_to_4bit_layout) {

    FusedLutPackedLayout layout;

    // ... (logic for header_bytes_per_group and packed_indices_bytes_per_group) ...
    if (promote_to_4bit_layout) {
        layout.packed_indices_bytes_per_group = (size_t)packing_group_size * NR / 2;
    } else {
        layout.packed_indices_bytes_per_group = (size_t)packing_group_size * NR * weight_nbit / 8;
    }
    layout.header_bytes_per_group = sizeof(FusedLutPackedWeightGroup<NR>);

    // --- Calculate Strides ---
    layout.group_stride_bytes = layout.header_bytes_per_group + layout.packed_indices_bytes_per_group;

    const int num_groups_per_k_tile = K / packing_group_size;
    layout.n_tile_stride_bytes = num_groups_per_k_tile * layout.group_stride_bytes;

    // --- Calculate Total Size ---
    const int N_padded = ((N + NR - 1) / NR) * NR;
    const int num_n_tiles = N_padded / NR;
    layout.total_buffer_size = num_n_tiles * layout.n_tile_stride_bytes;

    return layout;
}



}
