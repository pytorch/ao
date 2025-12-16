#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

#define BLOCK_ROWS 128
#define BLOCK_COLS 4
#define BLOCK_ROWS_LARGE 512
#define BYTES_PER_THREAD 16
#define SCALE_FACTOR_ROWS 128

__device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ void find_group_and_local_offset(
    int col_block_pid,
    const int32_t* __restrict__ input_group_end_offsets,
    int num_groups,
    int cols_per_block,
    int* __restrict__ smem_cumsum,
    int& group_id,
    int& local_col_block
) {
    if (threadIdx.x == 0) {
        int cumsum = 0;
        for (int g = 0; g < num_groups; g++) {
            int input_group_start = (g > 0) ? input_group_end_offsets[g - 1] : 0;
            int input_group_end = input_group_end_offsets[g];
            int group_size = input_group_end - input_group_start;
            int num_col_blocks = ceil_div(group_size, cols_per_block);
            cumsum += num_col_blocks;
            smem_cumsum[g] = cumsum;
        }
    }
    __syncthreads();

    group_id = 0;
    int cumsum_before = 0;
    for (int g = 0; g < num_groups; g++) {
        int cumsum_at_g = smem_cumsum[g];
        if (col_block_pid < cumsum_at_g) {
            group_id = g;
            local_col_block = col_block_pid - cumsum_before;
            break;
        }
        cumsum_before = cumsum_at_g;
    }
}

// =============================================================================
// PIPELINED SUPERBLOCK MAPPING FUNCTION
// =============================================================================
// Maps super_block_pid to (group_id, first_chunk_in_group, chunks_until_group_end).
// Each super-block processes up to CHUNKS_PER_TB chunks within a single group.
template <int CHUNKS_PER_TB>
__device__ void find_group_and_local_offset_for_superblock(
    int super_col_block_pid,
    const int32_t* __restrict__ input_group_end_offsets,
    int num_groups,
    int cols_per_block,
    int* __restrict__ smem_data,  // smem_data[0..num_groups-1] = chunks_in_group, smem_data[num_groups..2*num_groups-1] = super_blocks_in_group cumsum
    int& group_id,
    int& first_chunk_in_group,
    int& chunks_until_group_end
) {
    if (threadIdx.x == 0) {
        int superblock_cumsum = 0;
        for (int g = 0; g < num_groups; g++) {
            int input_group_start = (g > 0) ? input_group_end_offsets[g - 1] : 0;
            int input_group_end = input_group_end_offsets[g];
            int group_size = input_group_end - input_group_start;
            int chunks_in_group = ceil_div(group_size, cols_per_block);
            int superblocks_in_group = ceil_div(chunks_in_group, CHUNKS_PER_TB);
            smem_data[g] = chunks_in_group;
            superblock_cumsum += superblocks_in_group;
            smem_data[num_groups + g] = superblock_cumsum;
        }
    }
    __syncthreads();

    group_id = 0;
    int superblock_cumsum_before = 0;
    for (int g = 0; g < num_groups; g++) {
        int cumsum_at_g = smem_data[num_groups + g];
        if (super_col_block_pid < cumsum_at_g) {
            group_id = g;
            int local_superblock = super_col_block_pid - superblock_cumsum_before;
            first_chunk_in_group = local_superblock * CHUNKS_PER_TB;
            int chunks_in_group = smem_data[g];
            chunks_until_group_end = chunks_in_group - first_chunk_in_group;
            return;
        }
        superblock_cumsum_before = cumsum_at_g;
    }

    first_chunk_in_group = 0;
    chunks_until_group_end = 0;
}

__device__ __forceinline__ int compute_output_group_start_col(
    int group_id,
    const int32_t* input_group_end_offsets,
    int num_groups,
    int padding_size
) {
    int start_idx = 0;
    for (int i = 0; i < group_id; i++) {
        int prev_offset = (i > 0) ? input_group_end_offsets[i - 1] : 0;
        int curr_offset = input_group_end_offsets[i];
        int group_size = curr_offset - prev_offset;
        int padded_size = ceil_div(group_size, padding_size) * padding_size;
        start_idx += padded_size;
    }
    return start_idx;
}

__device__ __forceinline__ int compute_swizzled_index(int row, int col) {
    int r_div_32 = row / 32;
    int r_mod_32 = row % 32;
    return r_mod_32 * 16 + r_div_32 * 4 + col;
}

// =============================================================================
// ROW-MAJOR KERNEL
// =============================================================================
__global__ void mx_block_rearrange_2d_K_groups_rowmajor_kernel(
    const uint8_t* __restrict__ scales_ptr,
    int scales_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* __restrict__ input_group_end_offsets,
    uint8_t* __restrict__ output_scales_ptr,
    int output_stride_per_block,
    int num_groups
) {
    const int col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ __align__(16) uint8_t smem_block[BLOCK_ROWS * BLOCK_COLS];
    __shared__ int smem_cumsum[32];
    __shared__ int output_group_start_col;

    int group_id, local_col_block;
    find_group_and_local_offset(
        col_block_pid,
        input_group_end_offsets,
        num_groups,
        BLOCK_COLS,
        smem_cumsum,
        group_id,
        local_col_block
    );

    int input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
    int input_group_end_col = input_group_end_offsets[group_id];
    int curr_input_start_col = input_group_start_col + local_col_block * BLOCK_COLS;

    if (curr_input_start_col >= input_group_end_col) {
        return;
    }

    if (tid == 0) {
        output_group_start_col = compute_output_group_start_col(
            group_id, input_group_end_offsets, num_groups, 4
        );
    }

    int input_row = row_block_pid * BLOCK_ROWS + tid;
    int cols_remaining = input_group_end_col - curr_input_start_col;
    int cols_to_load = min(BLOCK_COLS, cols_remaining);

    uint32_t row_data = 0;
    if (input_row < scale_rows && curr_input_start_col < input_group_end_col) {
        int input_offset = input_row * scales_stride_dim0 + curr_input_start_col;
        const uint8_t* input_ptr = scales_ptr + input_offset;

        uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(input_ptr);
        if (cols_to_load >= 4 && ptr_addr % 4 == 0 && curr_input_start_col + 4 <= input_group_end_col) {
            row_data = __ldg(reinterpret_cast<const uint32_t*>(input_ptr));
        } else {
            uint8_t* row_bytes = reinterpret_cast<uint8_t*>(&row_data);
            for (int i = 0; i < cols_to_load && (curr_input_start_col + i) < input_group_end_col; i++) {
                row_bytes[i] = __ldg(input_ptr + i);
            }
        }
    }

    uint8_t* row_bytes = reinterpret_cast<uint8_t*>(&row_data);
    #pragma unroll
    for (int col = 0; col < BLOCK_COLS; col++) {
        int swizzled_idx = compute_swizzled_index(tid, col);
        smem_block[swizzled_idx] = row_bytes[col];
    }

    __syncthreads();

    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);
    int stride_per_row_of_blocks_in_group = num_col_blocks_in_group * output_stride_per_block;

    int offset_in_group = row_block_pid * stride_per_row_of_blocks_in_group +
                          local_col_block * output_stride_per_block;
    int final_offset = out_group_base_offset + offset_in_group;

    uint8_t* output_ptr = output_scales_ptr + final_offset + tid * BLOCK_COLS;
    uintptr_t out_ptr_addr = reinterpret_cast<uintptr_t>(output_ptr);

    if (out_ptr_addr % 4 == 0 && cols_to_load >= 4) {
        *reinterpret_cast<uint32_t*>(output_ptr) =
            *reinterpret_cast<const uint32_t*>(&smem_block[tid * BLOCK_COLS]);
    } else {
        const uint8_t* smem_ptr = &smem_block[tid * BLOCK_COLS];
        #pragma unroll
        for (int i = 0; i < cols_to_load; i++) {
            output_ptr[i] = smem_ptr[i];
        }
    }
}

// =============================================================================
// COLUMN-MAJOR KERNEL
// =============================================================================
__global__ void mx_block_rearrange_2d_K_groups_colmajor_kernel(
    const uint8_t* __restrict__ scales_ptr,
    int scales_stride_dim0,
    int scales_stride_dim1,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* __restrict__ input_group_end_offsets,
    uint8_t* __restrict__ output_scales_ptr,
    int output_stride_per_block,
    int num_groups
) {
    const int col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ __align__(16) uint8_t smem_swizzled[BLOCK_ROWS * BLOCK_COLS];
    __shared__ int smem_cumsum[32];
    __shared__ int output_group_start_col;

    int group_id, local_col_block;
    find_group_and_local_offset(
        col_block_pid,
        input_group_end_offsets,
        num_groups,
        BLOCK_COLS,
        smem_cumsum,
        group_id,
        local_col_block
    );

    int input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
    int input_group_end_col = input_group_end_offsets[group_id];
    int curr_input_start_col = input_group_start_col + local_col_block * BLOCK_COLS;

    if (curr_input_start_col >= input_group_end_col) {
        return;
    }

    if (tid == 0) {
        output_group_start_col = compute_output_group_start_col(
            group_id, input_group_end_offsets, num_groups, 4
        );
    }

    int cols_remaining = input_group_end_col - curr_input_start_col;
    int row_in_block = tid;
    int global_row = row_block_pid * BLOCK_ROWS + row_in_block;

    uint32_t packed_scales = 0;
    uint8_t* local_vals = reinterpret_cast<uint8_t*>(&packed_scales);

    if (global_row < scale_rows) {
        #pragma unroll
        for (int c = 0; c < BLOCK_COLS; ++c) {
            if (c < cols_remaining) {
                int global_col = curr_input_start_col + c;
                size_t offset = static_cast<size_t>(global_col) * scales_stride_dim1 + global_row;
                local_vals[c] = __ldg(scales_ptr + offset);
            }
        }
    }

    int r_div_32 = row_in_block >> 5;
    int r_mod_32 = row_in_block & 31;
    int smem_offset = (r_mod_32 << 4) + (r_div_32 << 2);

    *reinterpret_cast<uint32_t*>(&smem_swizzled[smem_offset]) = packed_scales;

    __syncthreads();

    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);
    int stride_per_row_of_blocks_in_group = num_col_blocks_in_group * output_stride_per_block;

    int offset_in_group = row_block_pid * stride_per_row_of_blocks_in_group +
                          local_col_block * output_stride_per_block;
    int final_offset = out_group_base_offset + offset_in_group;

    uint8_t* output_ptr = output_scales_ptr + final_offset + tid * BLOCK_COLS;

    *reinterpret_cast<uint32_t*>(output_ptr) =
        *reinterpret_cast<const uint32_t*>(&smem_swizzled[tid * BLOCK_COLS]);
}

// =============================================================================
// COLUMN-MAJOR VECTORIZED KERNEL
// =============================================================================
__global__ void mx_block_rearrange_2d_K_groups_colmajor_vectorized_kernel(
    const uint8_t* __restrict__ scales_ptr,
    int scales_stride_dim1,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* __restrict__ input_group_end_offsets,
    uint8_t* __restrict__ output_scales_ptr,
    int output_stride_per_block,
    int num_groups
) {
    const int col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __shared__ __align__(16) uint8_t smem_block[BLOCK_ROWS * BLOCK_COLS];
    __shared__ int smem_cumsum[32];
    __shared__ int output_group_start_col;

    int group_id, local_col_block;
    find_group_and_local_offset(
        col_block_pid,
        input_group_end_offsets,
        num_groups,
        BLOCK_COLS,
        smem_cumsum,
        group_id,
        local_col_block
    );

    int input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
    int input_group_end_col = input_group_end_offsets[group_id];
    int curr_input_start_col = input_group_start_col + local_col_block * BLOCK_COLS;

    if (curr_input_start_col >= input_group_end_col) {
        return;
    }

    if (tid == 0) {
        output_group_start_col = compute_output_group_start_col(
            group_id, input_group_end_offsets, num_groups, 4
        );
    }

    int cols_remaining = input_group_end_col - curr_input_start_col;
    int global_row_base = row_block_pid * BLOCK_ROWS;

    uint32_t loaded_data = 0;
    int col_idx = warp_id;

    if (col_idx < cols_remaining) {
        int global_col = curr_input_start_col + col_idx;
        int row_start = global_row_base + lane_id * 4;

        const uint8_t* col_ptr = scales_ptr +
            static_cast<size_t>(global_col) * scales_stride_dim1;

        if (row_start + 3 < scale_rows) {
            loaded_data = __ldg(reinterpret_cast<const uint32_t*>(col_ptr + row_start));
        } else if (row_start < scale_rows) {
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&loaded_data);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (row_start + i < scale_rows) {
                    bytes[i] = __ldg(col_ptr + row_start + i);
                }
            }
        }
    }

    uint8_t* bytes = reinterpret_cast<uint8_t*>(&loaded_data);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row_in_block = lane_id * 4 + i;
        int r_div_32 = row_in_block >> 5;
        int r_mod_32 = row_in_block & 31;
        int swizzle_idx = (r_mod_32 << 4) + (r_div_32 << 2) + col_idx;
        smem_block[swizzle_idx] = bytes[i];
    }

    __syncthreads();

    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);
    int stride_per_row_of_blocks_in_group = num_col_blocks_in_group * output_stride_per_block;

    int offset_in_group = row_block_pid * stride_per_row_of_blocks_in_group +
                          local_col_block * output_stride_per_block;
    int final_offset = out_group_base_offset + offset_in_group;

    uint8_t* output_ptr = output_scales_ptr + final_offset + tid * BLOCK_COLS;

    *reinterpret_cast<uint32_t*>(output_ptr) =
        *reinterpret_cast<const uint32_t*>(&smem_block[tid * BLOCK_COLS]);
}

// =============================================================================
// COLUMN-MAJOR 16B VECTORIZED KERNEL (512-ROW BLOCKS)
// =============================================================================
__global__ void mx_block_rearrange_2d_K_groups_colmajor_vectorized_16B_kernel(
    const uint8_t* __restrict__ scales_ptr,
    int scales_stride_dim1,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* __restrict__ input_group_end_offsets,
    uint8_t* __restrict__ output_scales_ptr,
    int output_stride_per_block,
    int num_groups
) {
    const int col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __shared__ __align__(16) uint8_t smem_block[BLOCK_ROWS_LARGE * BLOCK_COLS];
    __shared__ int smem_cumsum[32];
    __shared__ int output_group_start_col;

    int group_id, local_col_block;
    find_group_and_local_offset(
        col_block_pid,
        input_group_end_offsets,
        num_groups,
        BLOCK_COLS,
        smem_cumsum,
        group_id,
        local_col_block
    );

    int input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
    int input_group_end_col = input_group_end_offsets[group_id];
    int curr_input_start_col = input_group_start_col + local_col_block * BLOCK_COLS;

    if (curr_input_start_col >= input_group_end_col) {
        return;
    }

    if (tid == 0) {
        output_group_start_col = compute_output_group_start_col(
            group_id, input_group_end_offsets, num_groups, 4
        );
    }

    __syncthreads();

    int cols_remaining = input_group_end_col - curr_input_start_col;
    int global_row_base = row_block_pid * BLOCK_ROWS_LARGE;

    uint4 loaded_data = make_uint4(0, 0, 0, 0);
    int col_idx = warp_id;

    if (col_idx < cols_remaining) {
        int global_col = curr_input_start_col + col_idx;
        int row_start = global_row_base + lane_id * BYTES_PER_THREAD;

        const uint8_t* col_ptr = scales_ptr +
            static_cast<size_t>(global_col) * scales_stride_dim1;

        if (row_start + BYTES_PER_THREAD - 1 < scale_rows) {
            loaded_data = __ldg(reinterpret_cast<const uint4*>(col_ptr + row_start));
        } else if (row_start < scale_rows) {
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&loaded_data);
            #pragma unroll
            for (int i = 0; i < BYTES_PER_THREAD; i++) {
                if (row_start + i < scale_rows) {
                    bytes[i] = __ldg(col_ptr + row_start + i);
                }
            }
        }
    }

    uint8_t* bytes = reinterpret_cast<uint8_t*>(&loaded_data);

    #pragma unroll
    for (int i = 0; i < BYTES_PER_THREAD; i++) {
        int row_in_block = lane_id * BYTES_PER_THREAD + i;
        int tile_idx = row_in_block / SCALE_FACTOR_ROWS;
        int row_in_tile = row_in_block % SCALE_FACTOR_ROWS;
        int tile_base_offset = tile_idx * SCALE_FACTOR_ROWS * BLOCK_COLS;

        int r_div_32 = row_in_tile >> 5;
        int r_mod_32 = row_in_tile & 31;
        int swizzle_idx = (r_mod_32 << 4) + (r_div_32 << 2) + col_idx;

        smem_block[tile_base_offset + swizzle_idx] = bytes[i];
    }

    __syncthreads();

    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);

    constexpr int TILE_SIZE = SCALE_FACTOR_ROWS * BLOCK_COLS;
    int stride_per_row_of_blocks_in_group = num_col_blocks_in_group * TILE_SIZE;

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int row_idx = tid + r * 128;
        int tile_idx = row_idx >> 7;
        int row_in_tile = row_idx & 127;
        int actual_row_block = row_block_pid * 4 + tile_idx;

        int offset_in_group = actual_row_block * stride_per_row_of_blocks_in_group +
                              local_col_block * TILE_SIZE;
        int final_offset = out_group_base_offset + offset_in_group;

        uint8_t* output_ptr = output_scales_ptr + final_offset + row_in_tile * BLOCK_COLS;

        *reinterpret_cast<uint32_t*>(output_ptr) =
            *reinterpret_cast<const uint32_t*>(&smem_block[row_idx * BLOCK_COLS]);
    }
}

// =============================================================================
// ROW-MAJOR VECTORIZED KERNEL (512-ROW BLOCKS)
// =============================================================================
__global__ void mx_block_rearrange_2d_K_groups_rowmajor_vectorized_kernel(
    const uint8_t* __restrict__ scales_ptr,
    int scales_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* __restrict__ input_group_end_offsets,
    uint8_t* __restrict__ output_scales_ptr,
    int output_stride_per_block,
    int num_groups
) {
    const int col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ __align__(16) uint8_t smem_block[BLOCK_ROWS_LARGE * BLOCK_COLS];
    __shared__ int smem_cumsum[32];
    __shared__ int output_group_start_col;

    int group_id, local_col_block;
    find_group_and_local_offset(
        col_block_pid,
        input_group_end_offsets,
        num_groups,
        BLOCK_COLS,
        smem_cumsum,
        group_id,
        local_col_block
    );

    int input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
    int input_group_end_col = input_group_end_offsets[group_id];
    int curr_input_start_col = input_group_start_col + local_col_block * BLOCK_COLS;

    if (curr_input_start_col >= input_group_end_col) {
        return;
    }

    if (tid == 0) {
        output_group_start_col = compute_output_group_start_col(
            group_id, input_group_end_offsets, num_groups, 4
        );
    }

    __syncthreads();

    int cols_remaining = input_group_end_col - curr_input_start_col;
    int cols_to_load = min(BLOCK_COLS, cols_remaining);
    int global_row_base = row_block_pid * BLOCK_ROWS_LARGE;

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int row_idx = tid + r * 128;
        int global_row = global_row_base + row_idx;

        uint32_t row_data = 0;

        if (global_row < scale_rows) {
            const uint8_t* row_ptr = scales_ptr +
                static_cast<size_t>(global_row) * scales_stride_dim0 + curr_input_start_col;

            uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(row_ptr);
            bool aligned = (ptr_addr % 4 == 0);

            if (cols_to_load == 4 && aligned) {
                row_data = __ldg(reinterpret_cast<const uint32_t*>(row_ptr));
            } else {
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&row_data);
                #pragma unroll
                for (int c = 0; c < BLOCK_COLS; c++) {
                    if (c < cols_to_load) {
                        bytes[c] = __ldg(row_ptr + c);
                    }
                }
            }
        }

        uint8_t* bytes = reinterpret_cast<uint8_t*>(&row_data);

        int tile_idx = row_idx >> 7;
        int row_in_tile = row_idx & 127;
        int tile_base_offset = tile_idx * SCALE_FACTOR_ROWS * BLOCK_COLS;

        int r_div_32 = row_in_tile >> 5;
        int r_mod_32 = row_in_tile & 31;
        int swizzle_base = (r_mod_32 << 4) + (r_div_32 << 2);

        #pragma unroll
        for (int c = 0; c < BLOCK_COLS; c++) {
            smem_block[tile_base_offset + swizzle_base + c] = bytes[c];
        }
    }

    __syncthreads();

    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);

    constexpr int TILE_SIZE = SCALE_FACTOR_ROWS * BLOCK_COLS;
    int stride_per_row_of_blocks_in_group = num_col_blocks_in_group * TILE_SIZE;

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int row_idx = tid + r * 128;
        int tile_idx = row_idx >> 7;
        int row_in_tile = row_idx & 127;
        int actual_row_block = row_block_pid * 4 + tile_idx;

        int offset_in_group = actual_row_block * stride_per_row_of_blocks_in_group +
                              local_col_block * TILE_SIZE;
        int final_offset = out_group_base_offset + offset_in_group;

        uint8_t* output_ptr = output_scales_ptr + final_offset + row_in_tile * BLOCK_COLS;

        *reinterpret_cast<uint32_t*>(output_ptr) =
            *reinterpret_cast<const uint32_t*>(&smem_block[row_idx * BLOCK_COLS]);
    }
}

// =============================================================================
// TEMPLATED ROW-MAJOR VECTORIZED KERNEL
// =============================================================================
// Template parameter MAX_COLS controls data tile width (64 or 128 columns).
// Each thread loads 16 bytes from GMEM and scatter-writes to swizzled SMEM.

template <int MAX_COLS>
__global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_kernel(
    const uint8_t* __restrict__ scales_ptr,
    int scales_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* __restrict__ input_group_end_offsets,
    uint8_t* __restrict__ output_scales_ptr,
    int output_stride_per_block,
    int num_groups
) {
    constexpr int THREADS_PER_ROW = MAX_COLS / 16;
    constexpr int NUM_TILES_PER_THREAD = 4;
    constexpr int TILE_SIZE = SCALE_FACTOR_ROWS * BLOCK_COLS;
    constexpr int NUM_TILES = MAX_COLS / BLOCK_COLS;
    constexpr int SMEM_SIZE = NUM_TILES * TILE_SIZE;

    const int col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ __align__(16) uint8_t smem[SMEM_SIZE];
    __shared__ int smem_cumsum[32];
    __shared__ int output_group_start_col;

    int group_id, local_col_block;
    find_group_and_local_offset(
        col_block_pid,
        input_group_end_offsets,
        num_groups,
        MAX_COLS,
        smem_cumsum,
        group_id,
        local_col_block
    );

    int input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
    int input_group_end_col = input_group_end_offsets[group_id];
    int curr_input_start_col = input_group_start_col + local_col_block * MAX_COLS;

    if (curr_input_start_col >= input_group_end_col) {
        return;
    }

    if (tid == 0) {
        output_group_start_col = compute_output_group_start_col(
            group_id, input_group_end_offsets, num_groups, 4
        );
    }

    __syncthreads();

    int cols_remaining = input_group_end_col - curr_input_start_col;
    int cols_to_load = min(MAX_COLS, cols_remaining);
    int global_row_base = row_block_pid * BLOCK_ROWS;

    int row_idx = tid / THREADS_PER_ROW;
    int col_idx = tid % THREADS_PER_ROW;
    int global_row = global_row_base + row_idx;

    int r_div_32 = row_idx >> 5;
    int r_mod_32 = row_idx & 31;
    int swizzle_base = (r_mod_32 << 4) + (r_div_32 << 2);
    int thread_col_start = col_idx * 16;

    // PHASE 1: Load from GMEM and scatter-write to swizzled SMEM
    uint4 data = make_uint4(0, 0, 0, 0);

    if (global_row < scale_rows && thread_col_start < cols_to_load)
    {
        const uint8_t* row_ptr = scales_ptr +
            static_cast<size_t>(global_row) * scales_stride_dim0 + curr_input_start_col;

        uintptr_t gmem_addr = reinterpret_cast<uintptr_t>(row_ptr + thread_col_start);
        bool aligned = (gmem_addr % 16 == 0);

        if (thread_col_start + 16 <= cols_to_load && aligned)
        {
            data = __ldg(reinterpret_cast<const uint4*>(row_ptr + thread_col_start));
        }
        else
        {
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&data);
            int bytes_to_load = min(16, cols_to_load - thread_col_start);
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                if (i < bytes_to_load) {
                    bytes[i] = __ldg(row_ptr + thread_col_start + i);
                }
            }
        }
    }
    // XOR swizzle for bank conflict avoidance
    uint32_t* data32 = reinterpret_cast<uint32_t*>(&data);
    int first_tile_idx = thread_col_start >> 2;  // thread_col_start / 4

    int tile_xor = (col_idx & 3) << 2;
    int superrow_xor = ((swizzle_base >> 7) & 3) << 2;
    int combined_xor = tile_xor ^ superrow_xor;

    #pragma unroll
    for (int t = 0; t < NUM_TILES_PER_THREAD; t++) {
        int tile_idx = first_tile_idx + t;
        int tile_base = tile_idx * TILE_SIZE;
        int swizzled_idx = tile_base + (swizzle_base ^ combined_xor);
        *reinterpret_cast<uint32_t*>(&smem[swizzled_idx]) = data32[t];
    }

    __syncthreads();

    // PHASE 2: Copy from swizzled SMEM to GMEM
    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_4col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);

    int stride_per_row_of_4col_blocks = num_4col_blocks_in_group * TILE_SIZE;

    int tiles_before_this_block = local_col_block * (MAX_COLS / BLOCK_COLS);

    uint8_t* out_base = output_scales_ptr + out_group_base_offset +
                        row_block_pid * stride_per_row_of_4col_blocks +
                        tiles_before_this_block * TILE_SIZE;

    int num_tiles_this_block = ceil_div(cols_to_load, BLOCK_COLS);
    int bytes_to_copy = num_tiles_this_block * TILE_SIZE;

    // Apply reverse XOR to read from swizzled SMEM
    int byte_offset = tid * 16;
    if (byte_offset < bytes_to_copy)
    {
        uint32_t out_data[4];

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int out_byte = byte_offset + i * 4;
            int tile_idx = out_byte / TILE_SIZE;
            int within_tile_offset = out_byte % TILE_SIZE;

            int writer_col_idx = (tile_idx / NUM_TILES_PER_THREAD) % THREADS_PER_ROW;
            int read_tile_xor = (writer_col_idx & 3) << 2;

            int read_superrow_xor = ((within_tile_offset >> 7) & 3) << 2;

            int read_combined_xor = read_tile_xor ^ read_superrow_xor;
            int smem_addr = tile_idx * TILE_SIZE + (within_tile_offset ^ read_combined_xor);
            out_data[i] = *reinterpret_cast<uint32_t*>(&smem[smem_addr]);
        }

        *reinterpret_cast<uint4*>(out_base + byte_offset) =
            *reinterpret_cast<uint4*>(out_data);
    }
}

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_kernel<64>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_kernel<128>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

// =============================================================================
// PIPELINED ROW-MAJOR VECTORIZED KERNEL
// =============================================================================
// Uses 2-stage software pipelining with double buffering to overlap memory
// transfers with compute. Each threadblock processes CHUNKS_PER_TB consecutive
// chunks within the same group.

template <int MAX_COLS, int CHUNKS_PER_TB = 4>
__global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel(
    const uint8_t* __restrict__ scales_ptr,
    int scales_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* __restrict__ input_group_end_offsets,
    uint8_t* __restrict__ output_scales_ptr,
    int output_stride_per_block,
    int num_groups
) {
    constexpr int THREADS_PER_ROW = MAX_COLS / 16;
    constexpr int NUM_TILES_PER_THREAD = 4;  // 16 bytes / 4 bytes per tile
    constexpr int TILE_SIZE = SCALE_FACTOR_ROWS * BLOCK_COLS;  // 128 * 4 = 512
    constexpr int SMEM_SIZE = BLOCK_ROWS * MAX_COLS;
    constexpr int NUM_BUFFERS = 2;

    const int super_col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ __align__(16) uint8_t smem[NUM_BUFFERS][SMEM_SIZE];
    __shared__ int smem_cumsum[64]; // 32 groups max, 1 group size per group, 1 cumsum per group
    __shared__ int s_output_group_start_col;
    __shared__ int s_input_group_start_col;
    __shared__ int s_input_group_end_col;
    __shared__ int s_first_chunk_in_group;
    __shared__ int s_num_chunks_to_process;

    // PHASE 0: Map super-block to (group, first_chunk) pair

    int group_id, first_chunk_in_group, chunks_until_group_end;
    find_group_and_local_offset_for_superblock<CHUNKS_PER_TB>(
        super_col_block_pid,
        input_group_end_offsets,
        num_groups,
        MAX_COLS,
        smem_cumsum,  // Re-use smem_cumsum (needs 2*num_groups ints)
        group_id,
        first_chunk_in_group,
        chunks_until_group_end
    );

    // 1 thread compute group boundaries and output group start, then broadcasts via SMEM.
    // This avoids unnecessary extra GMEM accesses (input_group_end_offsets).
    if (tid == 0) {
        s_input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
        s_input_group_end_col = input_group_end_offsets[group_id];
        s_first_chunk_in_group = first_chunk_in_group;
        s_output_group_start_col = compute_output_group_start_col(
            group_id, input_group_end_offsets, num_groups, 4
        );
        s_num_chunks_to_process = (chunks_until_group_end > 0) ? min(CHUNKS_PER_TB, chunks_until_group_end) : 0;
    }

    __syncthreads();

    int input_group_start_col = s_input_group_start_col;
    int input_group_end_col = s_input_group_end_col;
    first_chunk_in_group = s_first_chunk_in_group;
    int output_group_start_col = s_output_group_start_col;
    int num_chunks_to_process = s_num_chunks_to_process;

    if (num_chunks_to_process <= 0) {
        return;
    }

    // PHASE 1: Precompute thread-constant values
    int global_row_base = row_block_pid * BLOCK_ROWS;
    int row_idx = tid / THREADS_PER_ROW;
    int col_idx = tid % THREADS_PER_ROW;
    int global_row = global_row_base + row_idx;
    bool row_valid = (global_row < scale_rows);

    int r_div_32 = row_idx >> 5;
    int r_mod_32 = row_idx & 31;
    int swizzle_base = (r_mod_32 << 4) + (r_div_32 << 2);
    int thread_col_start = col_idx * 16;
    int first_tile_idx = thread_col_start >> 2;

    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_4col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);
    int stride_per_row_of_4col_blocks = num_4col_blocks_in_group * TILE_SIZE;

    const uint8_t* row_base_ptr = scales_ptr +
        static_cast<size_t>(global_row) * scales_stride_dim0;

    // PHASE 2: Pipelined execution with double buffering
    auto load_chunk_async = [&](int chunk_idx, int buf_idx) {
        int curr_chunk_in_group = first_chunk_in_group + chunk_idx;
        int curr_input_start_col = input_group_start_col + curr_chunk_in_group * MAX_COLS;
        int cols_remaining = input_group_end_col - curr_input_start_col;
        int cols_to_load = min(MAX_COLS, cols_remaining);
        bool can_load = row_valid && (thread_col_start < cols_to_load);

        if (can_load) {
            const uint8_t* src_ptr = row_base_ptr + curr_input_start_col + thread_col_start;
            uintptr_t gmem_addr = reinterpret_cast<uintptr_t>(src_ptr);
            bool aligned = (gmem_addr % 16 == 0);
            bool full_vec = (thread_col_start + 16 <= cols_to_load) && aligned;

            if (full_vec) {
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :
                    : "l"(&smem[buf_idx][row_idx * MAX_COLS + thread_col_start]),
                      "l"(src_ptr)
                );
            } else {
                uint4 data = make_uint4(0, 0, 0, 0);
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&data);
                int bytes_to_load = min(16, cols_to_load - thread_col_start);
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    if (i < bytes_to_load) bytes[i] = __ldg(src_ptr + i);
                }
                *reinterpret_cast<uint4*>(&smem[buf_idx][row_idx * MAX_COLS + thread_col_start]) = data;
            }
        } else {
            *reinterpret_cast<uint4*>(&smem[buf_idx][row_idx * MAX_COLS + thread_col_start]) = make_uint4(0, 0, 0, 0);
        }
    };

    // Process chunk: read from linear SMEM, swizzle, store to GMEM
    auto process_chunk = [&](int chunk_idx, int buf_idx) {
        int curr_chunk_in_group = first_chunk_in_group + chunk_idx;
        int curr_input_start_col = input_group_start_col + curr_chunk_in_group * MAX_COLS;
        int cols_remaining = input_group_end_col - curr_input_start_col;
        int cols_to_load = min(MAX_COLS, cols_remaining);

        uint4 data = *reinterpret_cast<uint4*>(&smem[buf_idx][row_idx * MAX_COLS + thread_col_start]);

        __syncthreads();

        int tile_xor = (col_idx & 3) << 2;
        int superrow_xor = ((swizzle_base >> 7) & 3) << 2;
        int combined_xor = tile_xor ^ superrow_xor;

        uint32_t* data32 = reinterpret_cast<uint32_t*>(&data);
        #pragma unroll
        for (int t = 0; t < NUM_TILES_PER_THREAD; t++) {
            int tile_idx = first_tile_idx + t;
            int tile_base = tile_idx * SCALE_FACTOR_ROWS * BLOCK_COLS;
            int swizzled_idx = tile_base + (swizzle_base ^ combined_xor);
            *reinterpret_cast<uint32_t*>(&smem[buf_idx][swizzled_idx]) = data32[t];
        }

        __syncthreads();
        int tiles_before_this_chunk = curr_chunk_in_group * (MAX_COLS / BLOCK_COLS);
        uint8_t* out_base = output_scales_ptr + out_group_base_offset +
                            row_block_pid * stride_per_row_of_4col_blocks +
                            tiles_before_this_chunk * TILE_SIZE;

        int num_tiles_this_chunk = ceil_div(cols_to_load, BLOCK_COLS);
        int bytes_to_copy = num_tiles_this_chunk * TILE_SIZE;

        int byte_offset = tid * 16;
        if (byte_offset < bytes_to_copy) {
            uint32_t out_data[4];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int out_byte = byte_offset + i * 4;
                int read_tile_idx = out_byte / TILE_SIZE;
                int within_tile_offset = out_byte % TILE_SIZE;

                int writer_col_idx = (read_tile_idx / NUM_TILES_PER_THREAD) % THREADS_PER_ROW;
                int read_tile_xor = (writer_col_idx & 3) << 2;
                int read_superrow_xor = ((within_tile_offset >> 7) & 3) << 2;
                int read_combined_xor = read_tile_xor ^ read_superrow_xor;
                int smem_addr = read_tile_idx * TILE_SIZE + (within_tile_offset ^ read_combined_xor);
                out_data[i] = *reinterpret_cast<uint32_t*>(&smem[buf_idx][smem_addr]);
            }

            *reinterpret_cast<uint4*>(out_base + byte_offset) =
                *reinterpret_cast<uint4*>(out_data);
        }
    };

    if (num_chunks_to_process == 1) {
        load_chunk_async(0, 0);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        process_chunk(0, 0);
    } else {
        // PROLOGUE: Load first chunk
        load_chunk_async(0, 0);
        asm volatile("cp.async.commit_group;\n");

        // STEADY STATE: Overlap load N+1 with processing N
        for (int chunk = 0; chunk < num_chunks_to_process - 1; chunk++) {
            int curr_buf = chunk & 1;        // chunk % 2
            int next_buf = (chunk + 1) & 1;  // (chunk + 1) % 2

            // Kick off async load of chunk N+1
            load_chunk_async(chunk + 1, next_buf);
            asm volatile("cp.async.commit_group;\n");

            // Wait for async load of chunk N, without waiting for chunk N+1.
            // async loads completed in commit order (FIFO) so this should work.
            asm volatile("cp.async.wait_group 1;\n");
            __syncthreads();

            // Process chunk N.
            process_chunk(chunk, curr_buf);
        }

        // EPILOGUE: Process final chunk
        int last_chunk = num_chunks_to_process - 1;
        int last_buf = last_chunk & 1;
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        process_chunk(last_chunk, last_buf);
    }
}

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<64, 4>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<64, 8>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<64, 16>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<128, 4>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<128, 8>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<128, 16>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

namespace mxfp8 {

void launch_mx_block_rearrange_2d_K_groups_rowmajor(
    const uint8_t* scales_ptr,
    int scales_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    cudaStream_t stream
) {
    int num_row_blocks = (scale_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int output_stride_per_block = BLOCK_ROWS * BLOCK_COLS;
    int total_col_blocks = (scale_cols + BLOCK_COLS - 1) / BLOCK_COLS + num_groups;

    dim3 grid(total_col_blocks, num_row_blocks);
    dim3 block(128);

    mx_block_rearrange_2d_K_groups_rowmajor_kernel<<<grid, block, 0, stream>>>(
        scales_ptr,
        scales_stride_dim0,
        scale_rows,
        scale_cols,
        padded_rows,
        input_group_end_offsets,
        output_scales_ptr,
        output_stride_per_block,
        num_groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_mx_block_rearrange_2d_K_groups_colmajor(
    const uint8_t* scales_ptr,
    int scales_stride_dim0,
    int scales_stride_dim1,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    cudaStream_t stream
) {
    int num_row_blocks = (scale_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int output_stride_per_block = BLOCK_ROWS * BLOCK_COLS;
    int total_col_blocks = (scale_cols + BLOCK_COLS - 1) / BLOCK_COLS + num_groups;

    dim3 grid(total_col_blocks, num_row_blocks);
    dim3 block(128);

    mx_block_rearrange_2d_K_groups_colmajor_kernel<<<grid, block, 0, stream>>>(
        scales_ptr,
        scales_stride_dim0,
        scales_stride_dim1,
        scale_rows,
        scale_cols,
        padded_rows,
        input_group_end_offsets,
        output_scales_ptr,
        output_stride_per_block,
        num_groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_mx_block_rearrange_2d_K_groups_colmajor_vectorized(
    const uint8_t* scales_ptr,
    int scales_stride_dim1,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    cudaStream_t stream
) {
    int num_row_blocks = (scale_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int output_stride_per_block = BLOCK_ROWS * BLOCK_COLS;
    int total_col_blocks = (scale_cols + BLOCK_COLS - 1) / BLOCK_COLS + num_groups;

    dim3 grid(total_col_blocks, num_row_blocks);
    dim3 block(128);

    mx_block_rearrange_2d_K_groups_colmajor_vectorized_kernel<<<grid, block, 0, stream>>>(
        scales_ptr,
        scales_stride_dim1,
        scale_rows,
        scale_cols,
        padded_rows,
        input_group_end_offsets,
        output_scales_ptr,
        output_stride_per_block,
        num_groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_mx_block_rearrange_2d_K_groups_colmajor_vectorized_16B(
    const uint8_t* scales_ptr,
    int scales_stride_dim1,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    cudaStream_t stream
) {
    int num_row_blocks = (scale_rows + BLOCK_ROWS_LARGE - 1) / BLOCK_ROWS_LARGE;
    int output_stride_per_block = BLOCK_ROWS_LARGE * BLOCK_COLS;
    int total_col_blocks = (scale_cols + BLOCK_COLS - 1) / BLOCK_COLS + num_groups;

    dim3 grid(total_col_blocks, num_row_blocks);
    dim3 block(128);

    mx_block_rearrange_2d_K_groups_colmajor_vectorized_16B_kernel<<<grid, block, 0, stream>>>(
        scales_ptr,
        scales_stride_dim1,
        scale_rows,
        scale_cols,
        padded_rows,
        input_group_end_offsets,
        output_scales_ptr,
        output_stride_per_block,
        num_groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_mx_block_rearrange_2d_K_groups_rowmajor_vectorized(
    const uint8_t* scales_ptr,
    int scales_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    cudaStream_t stream
) {
    int num_row_blocks = (scale_rows + BLOCK_ROWS_LARGE - 1) / BLOCK_ROWS_LARGE;
    int output_stride_per_block = BLOCK_ROWS_LARGE * BLOCK_COLS;
    int total_col_blocks = (scale_cols + BLOCK_COLS - 1) / BLOCK_COLS + num_groups;

    dim3 grid(total_col_blocks, num_row_blocks);
    dim3 block(128);

    mx_block_rearrange_2d_K_groups_rowmajor_vectorized_kernel<<<grid, block, 0, stream>>>(
        scales_ptr,
        scales_stride_dim0,
        scale_rows,
        scale_cols,
        padded_rows,
        input_group_end_offsets,
        output_scales_ptr,
        output_stride_per_block,
        num_groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec(
    const uint8_t* scales_ptr,
    int scales_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    int max_cols,  // Template selector: 64 or 128
    cudaStream_t stream
) {
    int num_row_blocks = (scale_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int output_stride_per_block = BLOCK_ROWS * BLOCK_COLS;
    int total_col_blocks = (scale_cols + max_cols - 1) / max_cols + num_groups;

    dim3 grid(total_col_blocks, num_row_blocks);

    switch (max_cols) {
        case 64: {
            dim3 block(512);
            mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_kernel<64><<<grid, block, 0, stream>>>(
                scales_ptr, scales_stride_dim0, scale_rows, scale_cols, padded_rows,
                input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups
            );
            break;
        }
        case 128: {
            dim3 block(1024);
            mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_kernel<128><<<grid, block, 0, stream>>>(
                scales_ptr, scales_stride_dim0, scale_rows, scale_cols, padded_rows,
                input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups
            );
            break;
        }
        default:
            printf("CUDA Error: Unsupported max_cols value %d. Must be 64 or 128.\n", max_cols);
            return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (128x4 vec max_cols=%d): %s\n", max_cols, cudaGetErrorString(err));
    }
}

void launch_mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined(
    const uint8_t* scales_ptr,
    int scales_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    int max_cols,  // Template selector: 64 or 128
    int chunks_per_tb,  // Chunks per super-block: 4, 8, or 16
    cudaStream_t stream
) {
    int num_row_blocks = (scale_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int output_stride_per_block = BLOCK_ROWS * BLOCK_COLS;

    int total_chunks = (scale_cols + max_cols - 1) / max_cols + num_groups;
    int total_super_col_blocks = (total_chunks + chunks_per_tb - 1) / chunks_per_tb + num_groups;

    dim3 grid(total_super_col_blocks, num_row_blocks);

    if (max_cols == 64) {
        dim3 block(512);
        switch (chunks_per_tb) {
            case 4:
                mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<64, 4><<<grid, block, 0, stream>>>(
                    scales_ptr, scales_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            case 8:
                mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<64, 8><<<grid, block, 0, stream>>>(
                    scales_ptr, scales_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            case 16:
                mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<64, 16><<<grid, block, 0, stream>>>(
                    scales_ptr, scales_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            default:
                printf("CUDA Error: chunks_per_tb must be 4, 8, or 16, got %d\n", chunks_per_tb);
                return;
        }
    } else if (max_cols == 128) {
        dim3 block(1024);
        switch (chunks_per_tb) {
            case 4:
                mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<128, 4><<<grid, block, 0, stream>>>(
                    scales_ptr, scales_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            case 8:
                mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<128, 8><<<grid, block, 0, stream>>>(
                    scales_ptr, scales_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            case 16:
                mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<128, 16><<<grid, block, 0, stream>>>(
                    scales_ptr, scales_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            default:
                printf("CUDA Error: chunks_per_tb must be 4, 8, or 16, got %d\n", chunks_per_tb);
                return;
        }
    } else {
        printf("CUDA Error: max_cols must be 64 or 128, got %d\n", max_cols);
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (pipelined max_cols=%d, chunks_per_tb=%d): %s\n",
               max_cols, chunks_per_tb, cudaGetErrorString(err));
    }
}

} // namespace mxfp8
