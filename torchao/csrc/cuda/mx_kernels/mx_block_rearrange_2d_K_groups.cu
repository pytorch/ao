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
// For pipelined execution, each super-block processes up to CHUNKS_PER_TB chunks.
// This function maps a super_block_pid to:
//   - group_id: which group this super-block starts in
//   - first_chunk_in_group: the local chunk index within that group
//   - chunks_until_group_end: how many chunks can be processed before hitting group boundary
//
// KEY INSIGHT: We cannot simply multiply super_col_block_pid * CHUNKS_PER_TB because
// chunks within groups don't form a contiguous global numbering. Instead, we must
// iterate through the groups and assign super-blocks based on the chunks each group contains.
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
    // Thread 0 computes chunks per group and cumulative super-blocks
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

    // Find which group this super-block belongs to
    group_id = 0;
    int superblock_cumsum_before = 0;
    for (int g = 0; g < num_groups; g++) {
        int cumsum_at_g = smem_data[num_groups + g];
        if (super_col_block_pid < cumsum_at_g) {
            group_id = g;
            // local_superblock_in_group = super_col_block_pid - superblock_cumsum_before
            int local_superblock = super_col_block_pid - superblock_cumsum_before;
            // first_chunk_in_group = local_superblock * CHUNKS_PER_TB
            first_chunk_in_group = local_superblock * CHUNKS_PER_TB;
            // How many chunks remain until group boundary
            int chunks_in_group = smem_data[g];
            chunks_until_group_end = chunks_in_group - first_chunk_in_group;
            return;
        }
        superblock_cumsum_before = cumsum_at_g;
    }

    // Fallback (should not reach here if super_col_block_pid is valid)
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

// Row-major kernel: Input tensor has cols contiguous
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

// Column-major kernel: Input tensor has rows contiguous
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

// Column-major vectorized kernel: 4 warps, each processing one column with uint32_t loads
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

// Column-major 16B vectorized kernel: 512-row blocks, uint4 loads (16 bytes per thread)
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

// Row-major vectorized kernel: 512-row blocks, uint32_t loads per row
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
// Template parameter MAX_COLS controls the width of data processed per threadblock:
//   MAX_COLS = 64:  512 threads (128 rows × 4 cols/16B),  8KB SMEM, 128×64 data tile
//   MAX_COLS = 128: 1024 threads (128 rows × 8 cols/16B), 16KB SMEM, 128×128 data tile
//
// Thread layout: (BLOCK_ROWS * MAX_COLS / 16) threads
// Each thread loads 16 bytes from GMEM, scatter-writes to SMEM in swizzled format,
// then copies 16 bytes from SMEM to GMEM.

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
    // Compile-time constants derived from MAX_COLS
    constexpr int THREADS_PER_ROW = MAX_COLS / 16;  // Each thread handles 16 bytes
    constexpr int NUM_TILES_PER_THREAD = 4;  // Each thread writes 4 tiles (16 bytes = 4 × 4-byte writes)
    constexpr int TILE_SIZE = SCALE_FACTOR_ROWS * BLOCK_COLS;  // 128 * 4 = 512
    constexpr int NUM_TILES = MAX_COLS / BLOCK_COLS;  // Number of 4-col tiles per chunk

    // BANK CONFLICT AVOIDANCE via XOR swizzle:
    // Problem: Tiles are 512 bytes apart. 512 % 128 = 0, so all tiles start at bank 0.
    // Within a warp, threads write to different tiles at the same within-tile offset,
    // causing all writes to hit the same bank → multi-way conflict.
    //
    // Solution: XOR the within-tile offset by (tile_idx % 4) * 4 bytes.
    // This rotates the bank assignment per tile:
    //   Tile 0: XOR 0  → original banks
    //   Tile 1: XOR 4  → banks rotated by 1
    //   Tile 2: XOR 8  → banks rotated by 2
    //   Tile 3: XOR 12 → banks rotated by 3
    // Now threads writing to tiles 0,1,2,3 at the same row offset hit different banks.
    //
    // Trade-off: XOR scrambles data within tiles. Read phase must apply same XOR
    // to recover correct data. This requires scalar (4-byte) reads instead of
    // vectorized (16-byte) reads, since XOR breaks 16-byte contiguity.
    constexpr int SMEM_SIZE = NUM_TILES * TILE_SIZE;  // No padding needed with XOR

    const int col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;

    // Shared memory buffer for swizzled output (XOR swizzle for bank conflict avoidance)
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

    // Thread layout: (128 × THREADS_PER_ROW) threads = 128 rows × THREADS_PER_ROW threads per row
    int row_idx = tid / THREADS_PER_ROW;   // 0-127
    int col_idx = tid % THREADS_PER_ROW;   // 0 to (THREADS_PER_ROW-1)
    int global_row = global_row_base + row_idx;

    // Compute swizzle base offset for this thread's row
    int r_div_32 = row_idx >> 5;   // row / 32
    int r_mod_32 = row_idx & 31;   // row % 32
    int swizzle_base = (r_mod_32 << 4) + (r_div_32 << 2);  // r_mod_32 * 16 + r_div_32 * 4
    int thread_col_start = col_idx * 16;  // 0, 16, 32, 48, ... up to (MAX_COLS - 16)

    // ============================================================
    // PHASE 1: Load from GMEM directly to SMEM in swizzled format
    // Each thread loads 16 bytes from GMEM using vectorized load,
    // then scatter-writes to swizzled positions in SMEM.
    uint4 data = make_uint4(0, 0, 0, 0);

    if (global_row < scale_rows && thread_col_start < cols_to_load)
    {
        const uint8_t* row_ptr = scales_ptr +
            static_cast<size_t>(global_row) * scales_stride_dim0 + curr_input_start_col;

        uintptr_t gmem_addr = reinterpret_cast<uintptr_t>(row_ptr + thread_col_start);
        bool aligned = (gmem_addr % 16 == 0);

        // Load 16 bytes from GMEM (vectorized if aligned and full)
        if (thread_col_start + 16 <= cols_to_load && aligned)
        {
            data = __ldg(reinterpret_cast<const uint4*>(row_ptr + thread_col_start));
        }
        else
        {
            // Partial/unaligned load
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
    // else: data remains zero (padding rows or columns beyond cols_to_load)

    // Scatter-write to swizzled SMEM positions using vectorized uint32 writes
    // Apply XOR swizzle based on col_idx to spread bank accesses across tiles
    // col_idx determines which set of tiles a thread writes to:
    //   col_idx=0 writes tiles 0,1,2,3
    //   col_idx=1 writes tiles 4,5,6,7
    //   col_idx=2 writes tiles 8,9,10,11
    //   col_idx=3 writes tiles 12,13,14,15
    // By XORing by col_idx, threads writing to different tile sets hit different banks.
    //
    // ADDITIONAL XOR for read phase: We also XOR by (swizzle_base / 128) % 4 to spread
    // reads within each tile. This is needed because the read phase reads contiguous
    // superrows, and superrows 0, 8, 16, 24 would otherwise hit the same bank.
    uint32_t* data32 = reinterpret_cast<uint32_t*>(&data);
    int first_tile_idx = thread_col_start >> 2;  // thread_col_start / 4

    // XOR 1: Spread across tiles (col_idx * 4)
    int tile_xor = (col_idx & 3) << 2;  // (col_idx % 4) * 4

    // XOR 2: Spread within tiles based on superrow group (swizzle_base / 128) % 4 * 4
    // swizzle_base ranges 0-508, divide by 128 gives 0-3 for the 4 groups of 32 rows
    int superrow_xor = ((swizzle_base >> 7) & 3) << 2;  // (swizzle_base / 128) % 4 * 4

    // Combined XOR
    int combined_xor = tile_xor ^ superrow_xor;

    #pragma unroll
    for (int t = 0; t < NUM_TILES_PER_THREAD; t++) {
        int tile_idx = first_tile_idx + t;
        int tile_base = tile_idx * TILE_SIZE;
        int swizzled_idx = tile_base + (swizzle_base ^ combined_xor);
        *reinterpret_cast<uint32_t*>(&smem[swizzled_idx]) = data32[t];
    }

    __syncthreads();

    // PHASE 2: Store from SMEM to GMEM
    // Read from padded SMEM layout, write contiguous tiles to GMEM
    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_4col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);

    int stride_per_row_of_4col_blocks = num_4col_blocks_in_group * TILE_SIZE;

    // tiles_before_this_block: 4-col tiles that came before this MAX_COLS-col block
    int tiles_before_this_block = local_col_block * (MAX_COLS / BLOCK_COLS);

    // Base output pointer for this threadblock
    uint8_t* out_base = output_scales_ptr + out_group_base_offset +
                        row_block_pid * stride_per_row_of_4col_blocks +
                        tiles_before_this_block * TILE_SIZE;

    // Number of 4-column tiles in this threadblock
    int num_tiles_this_block = ceil_div(cols_to_load, BLOCK_COLS);
    int bytes_to_copy = num_tiles_this_block * TILE_SIZE;

    // Each thread copies 16 bytes to GMEM
    // With XOR swizzle, we must apply the same combined XOR to read back correctly.
    // Since XOR breaks 16-byte contiguity, we use 4-byte scalar reads.
    //
    // During write, combined_xor = tile_xor ^ superrow_xor where:
    //   tile_xor = (col_idx & 3) << 2
    //   superrow_xor = ((swizzle_base >> 7) & 3) << 2
    //
    // To reverse the XOR during read:
    //   - Determine which col_idx wrote this tile: (tile_idx / NUM_TILES_PER_THREAD) % THREADS_PER_ROW
    //   - Determine the superrow group from the within-tile offset: (within_tile_offset / 128) % 4
    //   - Apply the same combined XOR to get the swizzled address
    int byte_offset = tid * 16;
    if (byte_offset < bytes_to_copy)
    {
        // Read 4 × 4-byte chunks, applying combined XOR to each address
        uint32_t out_data[4];

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int out_byte = byte_offset + i * 4;
            int tile_idx = out_byte / TILE_SIZE;
            int within_tile_offset = out_byte % TILE_SIZE;

            // Reverse tile_xor: determine which col_idx wrote this tile
            int writer_col_idx = (tile_idx / NUM_TILES_PER_THREAD) % THREADS_PER_ROW;
            int read_tile_xor = (writer_col_idx & 3) << 2;

            // Reverse superrow_xor: determine which superrow group this offset belongs to
            // within_tile_offset / 128 gives the row group (0-3) within the tile
            int read_superrow_xor = ((within_tile_offset >> 7) & 3) << 2;

            // Combined XOR to get swizzled address
            int read_combined_xor = read_tile_xor ^ read_superrow_xor;
            int smem_addr = tile_idx * TILE_SIZE + (within_tile_offset ^ read_combined_xor);
            out_data[i] = *reinterpret_cast<uint32_t*>(&smem[smem_addr]);
        }

        // Write 16 bytes to GMEM (output is contiguous, so this is aligned)
        *reinterpret_cast<uint4*>(out_base + byte_offset) =
            *reinterpret_cast<uint4*>(out_data);
    }
}

// Explicit template instantiations
template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_kernel<64>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_kernel<128>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

// =============================================================================
// PIPELINED ROW-MAJOR VECTORIZED KERNEL (TRUE 2-STAGE SOFTWARE PIPELINE)
// =============================================================================
// This kernel uses true 2-stage software pipelining with double buffering to
// overlap memory transfers with compute:
//   - While processing (swizzle + store) chunk N from buffer A
//   - Simultaneously loading chunk N+1 into buffer B via cp.async
//
// KEY DESIGN: Each threadblock processes MULTIPLE consecutive chunks within
// the SAME group. This respects group boundaries while enabling true overlap.
//
// Pipeline stages:
//   Prologue: Async load chunk 0 into buffer 0, wait for completion
//   Steady state loop (for each chunk 0..N-2):
//     - Kick off async load of chunk K+1 into buffer[(K+1)%2]
//     - Process chunk K from buffer[K%2] (swizzle + store to GMEM)
//     - Wait for chunk K+1 load to complete
//   Epilogue: Process final chunk
//
// Template parameter CHUNKS_PER_TB controls how many MAX_COLS-wide chunks
// each threadblock processes within a single group.

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
    // Compile-time constants
    constexpr int THREADS_PER_ROW = MAX_COLS / 16;
    constexpr int NUM_TILES_PER_THREAD = 4;  // 16 bytes / 4 bytes per tile
    constexpr int TILE_SIZE = SCALE_FACTOR_ROWS * BLOCK_COLS;  // 128 * 4 = 512
    constexpr int SMEM_SIZE = BLOCK_ROWS * MAX_COLS;
    constexpr int NUM_BUFFERS = 2;

    const int super_col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;

    // Double-buffered shared memory for true pipelining
    __shared__ __align__(16) uint8_t smem[NUM_BUFFERS][SMEM_SIZE];
    __shared__ int smem_cumsum[32];
    __shared__ int s_output_group_start_col;
    __shared__ int s_input_group_start_col;
    __shared__ int s_input_group_end_col;
    __shared__ int s_first_chunk_in_group;
    __shared__ int s_num_chunks_to_process;

    // =========================================================================
    // PHASE 0: Map this super-block to a (group, first_chunk) pair
    // =========================================================================
    // Use the superblock-aware mapping function that correctly handles
    // the relationship between super-blocks and group boundaries.

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

    // Compute group boundaries and output group start
    if (tid == 0) {
        s_input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
        s_input_group_end_col = input_group_end_offsets[group_id];
        s_first_chunk_in_group = first_chunk_in_group;
        s_output_group_start_col = compute_output_group_start_col(
            group_id, input_group_end_offsets, num_groups, 4
        );
        // Early stop: only process chunks until group boundary
        // chunks_until_group_end could be <= 0 for over-allocated super-blocks
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

    // =========================================================================
    // PHASE 1: Precompute thread-constant values
    // =========================================================================
    int global_row_base = row_block_pid * BLOCK_ROWS;
    int row_idx = tid / THREADS_PER_ROW;
    int col_idx = tid % THREADS_PER_ROW;
    int global_row = global_row_base + row_idx;
    bool row_valid = (global_row < scale_rows);

    // Swizzle computation (constant for this thread)
    int r_div_32 = row_idx >> 5;
    int r_mod_32 = row_idx & 31;
    int swizzle_base = (r_mod_32 << 4) + (r_div_32 << 2);
    int thread_col_start = col_idx * 16;
    int first_tile_idx = thread_col_start >> 2;

    // Output address constants
    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_4col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);
    int stride_per_row_of_4col_blocks = num_4col_blocks_in_group * TILE_SIZE;

    // Base input pointer for this row
    const uint8_t* row_base_ptr = scales_ptr +
        static_cast<size_t>(global_row) * scales_stride_dim0;

    // =========================================================================
    // PIPELINED EXECUTION WITH DOUBLE BUFFERING
    // =========================================================================

    // Lambda to load a chunk asynchronously into specified buffer
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
                // Fallback for partial/unaligned - load to registers then store
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

    // Lambda to process a chunk: read from linear SMEM, swizzle, write to SMEM, store to GMEM
    // Uses dual XOR swizzle to avoid SMEM bank conflicts:
    //   tile_xor: spreads writes across tiles (based on col_idx)
    //   superrow_xor: spreads writes within tiles (based on swizzle_base / 128)
    auto process_chunk = [&](int chunk_idx, int buf_idx) {
        int curr_chunk_in_group = first_chunk_in_group + chunk_idx;
        int curr_input_start_col = input_group_start_col + curr_chunk_in_group * MAX_COLS;
        int cols_remaining = input_group_end_col - curr_input_start_col;
        int cols_to_load = min(MAX_COLS, cols_remaining);

        // Read from linear layout in SMEM
        uint4 data = *reinterpret_cast<uint4*>(&smem[buf_idx][row_idx * MAX_COLS + thread_col_start]);

        __syncthreads();  // Ensure all reads complete before overwriting

        // Compute XOR offsets for bank conflict avoidance
        // XOR 1: Spread across tiles (col_idx * 4)
        int tile_xor = (col_idx & 3) << 2;  // (col_idx % 4) * 4

        // XOR 2: Spread within tiles based on superrow group (swizzle_base / 128) % 4 * 4
        int superrow_xor = ((swizzle_base >> 7) & 3) << 2;

        // Combined XOR
        int combined_xor = tile_xor ^ superrow_xor;

        // Write to swizzled positions in same buffer with XOR swizzle
        uint32_t* data32 = reinterpret_cast<uint32_t*>(&data);
        #pragma unroll
        for (int t = 0; t < NUM_TILES_PER_THREAD; t++) {
            int tile_idx = first_tile_idx + t;
            int tile_base = tile_idx * SCALE_FACTOR_ROWS * BLOCK_COLS;
            int swizzled_idx = tile_base + (swizzle_base ^ combined_xor);
            *reinterpret_cast<uint32_t*>(&smem[buf_idx][swizzled_idx]) = data32[t];
        }

        __syncthreads();  // Ensure swizzle writes complete

        // Copy swizzled data to GMEM
        // Must reverse the XOR to read correct data
        int tiles_before_this_chunk = curr_chunk_in_group * (MAX_COLS / BLOCK_COLS);
        uint8_t* out_base = output_scales_ptr + out_group_base_offset +
                            row_block_pid * stride_per_row_of_4col_blocks +
                            tiles_before_this_chunk * TILE_SIZE;

        int num_tiles_this_chunk = ceil_div(cols_to_load, BLOCK_COLS);
        int bytes_to_copy = num_tiles_this_chunk * TILE_SIZE;

        int byte_offset = tid * 16;
        if (byte_offset < bytes_to_copy) {
            // Read 4 × 4-byte chunks, applying combined XOR to each address
            uint32_t out_data[4];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int out_byte = byte_offset + i * 4;
                int read_tile_idx = out_byte / TILE_SIZE;
                int within_tile_offset = out_byte % TILE_SIZE;

                // Reverse tile_xor: determine which col_idx wrote this tile
                int writer_col_idx = (read_tile_idx / NUM_TILES_PER_THREAD) % THREADS_PER_ROW;
                int read_tile_xor = (writer_col_idx & 3) << 2;

                // Reverse superrow_xor: determine which superrow group this offset belongs to
                int read_superrow_xor = ((within_tile_offset >> 7) & 3) << 2;

                // Combined XOR to get swizzled address
                int read_combined_xor = read_tile_xor ^ read_superrow_xor;
                int smem_addr = read_tile_idx * TILE_SIZE + (within_tile_offset ^ read_combined_xor);
                out_data[i] = *reinterpret_cast<uint32_t*>(&smem[buf_idx][smem_addr]);
            }

            // Write 16 bytes to GMEM (output is contiguous, so this is aligned)
            *reinterpret_cast<uint4*>(out_base + byte_offset) =
                *reinterpret_cast<uint4*>(out_data);
        }
    };

    if (num_chunks_to_process == 1) {
        // Single chunk: no pipelining benefit, just load-process-store
        load_chunk_async(0, 0);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        process_chunk(0, 0);
    } else {
        // =====================================================================
        // TRUE PIPELINING: Overlap load of N+1 with compute of N
        // =====================================================================

        // PROLOGUE: Load first chunk into buffer 0
        load_chunk_async(0, 0);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        // STEADY STATE: For chunks 0 to N-2
        for (int chunk = 0; chunk < num_chunks_to_process - 1; chunk++) {
            int curr_buf = chunk & 1;        // chunk % 2
            int next_buf = (chunk + 1) & 1;  // (chunk + 1) % 2

            // Kick off async load of NEXT chunk into alternate buffer
            // This runs in parallel with the processing below!
            load_chunk_async(chunk + 1, next_buf);
            asm volatile("cp.async.commit_group;\n");

            // Process CURRENT chunk (swizzle + store to GMEM)
            // This overlaps with the async load above
            process_chunk(chunk, curr_buf);

            // Wait for next chunk load to complete before next iteration
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
        }

        // EPILOGUE: Process the final chunk (already loaded)
        int last_chunk = num_chunks_to_process - 1;
        int last_buf = last_chunk & 1;
        process_chunk(last_chunk, last_buf);
    }
}

// Explicit template instantiations for pipelined kernel
// MAX_COLS = 64, various CHUNKS_PER_TB values
template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<64, 4>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<64, 8>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

template __global__ void mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_pipelined_kernel<64, 16>(
    const uint8_t* __restrict__, int, int, int, int,
    const int32_t* __restrict__, uint8_t* __restrict__, int, int);

// MAX_COLS = 128, various CHUNKS_PER_TB values
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

    // Dispatch to appropriate template instantiation based on max_cols
    switch (max_cols) {
        case 64: {
            dim3 block(512);  // 128 rows × 4 threads/row = 512 threads
            mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec_kernel<64><<<grid, block, 0, stream>>>(
                scales_ptr, scales_stride_dim0, scale_rows, scale_cols, padded_rows,
                input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups
            );
            break;
        }
        case 128: {
            dim3 block(1024);  // 128 rows × 8 threads/row = 1024 threads
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

    // Compute upper bound for super-blocks:
    // Each group needs ceil(chunks_in_group / chunks_per_tb) super-blocks.
    // Add num_groups to account for group boundary fragmentation.
    int total_chunks = (scale_cols + max_cols - 1) / max_cols + num_groups;
    int total_super_col_blocks = (total_chunks + chunks_per_tb - 1) / chunks_per_tb + num_groups;

    dim3 grid(total_super_col_blocks, num_row_blocks);

    // Dispatch based on max_cols and chunks_per_tb
    if (max_cols == 64) {
        dim3 block(512);  // 128 rows × 4 threads/row
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
        dim3 block(1024);  // 128 rows × 8 threads/row
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
