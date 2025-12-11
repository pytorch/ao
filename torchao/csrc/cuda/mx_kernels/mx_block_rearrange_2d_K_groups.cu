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
            int num_col_blocks = ceil_div(group_size, BLOCK_COLS);
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

#define MAX_COLS 128 // 4 threads * 4 uint4 which are each each 4 uint32s

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
    const int col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ __align__(16) uint8_t smem_block[BLOCK_ROWS * MAX_COLS];
    __shared__ int smem_cumsum[32];
    __shared__ int output_group_start_col;

    int group_id, local_col_block;
    find_group_and_local_offset(
        col_block_pid,
        input_group_end_offsets,
        num_groups,
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

    int row_idx = tid / MAX_COLS;
    int col_idx = tid % MAX_COLS;
    int global_row = global_row_base + row_idx;

    if (global_row < scale_rows) {
        const uint8_t* row_ptr = scales_ptr +
            static_cast<size_t>(global_row) * scales_stride_dim0 + curr_input_start_col * BLOCK_COLS;

        uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(row_ptr);
        bool aligned = (ptr_addr % 4 == 0);

        // 4 threads wide * 4 int32 each = 4*4*4bytes per int32 = 128 bytes
        uint4 row_data = make_uint4(0, 0, 0, 0);
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&row_data);
        int bytes_to_load = cols_to_load;

        while (bytes_to_load > 0) {
            // Load full 128 bytes per row at once if we can
            if (bytes_to_load == MAX_COLS && aligned)
            {
                row_data = __ldg(reinterpret_cast<const uint4*>(row_ptr));
                break;
            }
            // 4 threads wide * 2 uint32 each = 4*4*2 bytes per int32 = 64 bytes
            else if (bytes_to_load >= 64)
            {
                *reinterpret_cast<uint2*>(&row_data) = __ldg(reinterpret_cast<const uint2*>(row_ptr));
                bytes_to_load -= 64;

                // Increment row_ptr by 2 uint4s (64 bytes)
                row_ptr += 64;
            }
            // 4 threads wide * 1 uint32 each = 4*4*1 byte per int32 = 32 bytes
            else if (bytes_to_load >= 32)
            {
                *reinterpret_cast<uint32_t*>(&row_data) = __ldg(reinterpret_cast<const uint32_t*>(row_ptr));
                bytes_to_load -= 32;

                // Increment row_ptr by 1 uint32 (32 bytes)
                row_ptr += 32;
            }
            // Fall back to single byte loads if final chunk < 32 bytes
            else {
                #pragma unroll
                for (int c = 0; c < BLOCK_COLS; c++) {
                    // 4 threads still, so mask if we have fewer than 4 cols to load
                    if (c * BLOCK_COLS < bytes_to_load)
                    {
                        bytes[c] = __ldg(row_ptr + c);
                    }
                }
                break;
            }
        }

        int r_div_32 = row_idx >> 5; // row / 32
        int r_mod_32 = row_idx & 31; // row % 32
        int swizzle_base = (r_mod_32 << 4) + (r_div_32 << 2); // r_mod_32 * 16 + r_div_32 * 4

        #pragma unroll
        for (int c = 0; c < cols_to_load; c++) {
            // We loaded potentially multiple 128x4 tiles, which we need to swizzle individually.
            // Determine which tile we are in, and swizzle accordingly.
            int tile_idx = c >> 2; // col / 4
            int col_in_tile = c & 3; // col % 4
            int tile_base_offset = tile_idx * SCALE_FACTOR_ROWS * BLOCK_COLS;
            smem_block[tile_base_offset + swizzle_base + col_in_tile] = bytes[col_in_tile];
        }
    }

    __syncthreads();

    // Store phase: Direct copy from SMEM to GMEM
    // SMEM layout matches GMEM layout exactly - just copy with coalesced vectorized stores
    // Same pattern as load phase: try 128 bytes, then 64, then 32, then byte-by-byte

    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_tiles = ceil_div(cols_to_load, BLOCK_COLS);
    constexpr int TILE_SIZE = SCALE_FACTOR_ROWS * BLOCK_COLS;  // 128 * 4 = 512
    int total_bytes = num_tiles * TILE_SIZE;

    // Calculate output base for this threadblock's contiguous output region
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_128col_blocks_in_group = ceil_div(num_cols_in_group, MAX_COLS);
    int tiles_per_128col_block = MAX_COLS / BLOCK_COLS;  // 32
    int stride_per_row_block = num_128col_blocks_in_group * tiles_per_128col_block * TILE_SIZE;

    int output_offset = out_group_base_offset +
                        row_block_pid * stride_per_row_block +
                        local_col_block * tiles_per_128col_block * TILE_SIZE;

    uint8_t* output_base = output_scales_ptr + output_offset;

    // Distribute bytes across all 512 threads
    // Each thread handles (total_bytes / 512) bytes, with vectorized stores
    constexpr int NUM_THREADS = 512;
    int bytes_per_thread = total_bytes / NUM_THREADS;
    int thread_start_byte = tid * bytes_per_thread;

    // Handle any remainder bytes with the last few threads
    int extra_bytes = total_bytes % NUM_THREADS;
    if (tid < extra_bytes) {
        thread_start_byte += tid;
        bytes_per_thread += 1;
    } else {
        thread_start_byte += extra_bytes;
    }

    const uint8_t* smem_ptr = smem_block + thread_start_byte;
    uint8_t* gmem_ptr = output_base + thread_start_byte;
    int bytes_remaining = bytes_per_thread;

    // Try 128-byte stores (uint4 = 16 bytes, need 8 consecutive threads for 128 bytes)
    while (bytes_remaining >= 16 && (reinterpret_cast<uintptr_t>(gmem_ptr) % 16 == 0)) {
        *reinterpret_cast<uint4*>(gmem_ptr) = *reinterpret_cast<const uint4*>(smem_ptr);
        gmem_ptr += 16;
        smem_ptr += 16;
        bytes_remaining -= 16;
    }

    // Try 8-byte stores (uint2)
    while (bytes_remaining >= 8 && (reinterpret_cast<uintptr_t>(gmem_ptr) % 8 == 0)) {
        *reinterpret_cast<uint2*>(gmem_ptr) = *reinterpret_cast<const uint2*>(smem_ptr);
        gmem_ptr += 8;
        smem_ptr += 8;
        bytes_remaining -= 8;
    }

    // Try 4-byte stores (uint32_t)
    while (bytes_remaining >= 4 && (reinterpret_cast<uintptr_t>(gmem_ptr) % 4 == 0)) {
        *reinterpret_cast<uint32_t*>(gmem_ptr) = *reinterpret_cast<const uint32_t*>(smem_ptr);
        gmem_ptr += 4;
        smem_ptr += 4;
        bytes_remaining -= 4;
    }

    // Fall back to single byte stores for remainder
    while (bytes_remaining > 0) {
        *gmem_ptr++ = *smem_ptr++;
        bytes_remaining--;
    }
}

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

} // namespace mxfp8
