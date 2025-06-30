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


template <int NR>
struct FusedLutPackedWeightGroup {
    uint8x16_t lut_soa_planes[4];
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

template<int NR>
inline FusedLutPackedLayout create_fused_lut_layout(
    int N, int K, int scale_group_size, int lut_group_size,
    int weight_nbit, bool promote_to_4bit_layout) {

    FusedLutPackedLayout layout;

    int packing_group_size = std::gcd(scale_group_size, lut_group_size);

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
#endif // defined(__aarch64__) || defined(__ARM_NEON)
