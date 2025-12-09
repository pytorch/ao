#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

#define BLOCK_ROWS 128
#define BLOCK_COLS 4

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

// =============================================================================
// ROW-MAJOR KERNEL: Input tensor is in row-major layout (cols contiguous)
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
// COLUMN-MAJOR KERNEL: Input tensor is in column-major layout (rows contiguous)
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
// OPTIMIZED COLUMN-MAJOR KERNEL: 4 warps, each processing one column with
// vectorized uint32_t loads (4 bytes per thread = 128 bytes per warp)
// =============================================================================
__global__ void mx_block_rearrange_2d_K_groups_colmajor_vectorized_kernel(
    const uint8_t* __restrict__ scales_ptr,
    int scales_stride_dim1,  // stride between columns (= num_rows for col-major)
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

    // Warp and lane decomposition: 128 threads = 4 warps × 32 lanes
    const int warp_id = tid >> 5;      // tid / 32
    const int lane_id = tid & 31;      // tid % 32

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

    // =========================================================================
    // VECTORIZED LOAD PHASE
    // Each warp (32 threads) handles one column
    // Each lane loads 4 consecutive rows using a uint32_t vectorized load
    // Total per warp: 32 lanes × 4 bytes = 128 bytes (one coalesced transaction)
    // Total per block: 4 warps × 128 bytes = 512 bytes
    // =========================================================================

    uint32_t loaded_data = 0;

    // Each warp processes one column (warp 0 -> col 0, warp 1 -> col 1, etc.)
    int col_idx = warp_id;  // Which column this warp handles (0-3)

    if (col_idx < cols_remaining) {
        int global_col = curr_input_start_col + col_idx;

        // Each lane loads 4 consecutive rows: rows [lane_id*4, lane_id*4+3]
        int row_start = global_row_base + lane_id * 4;

        // Pointer to start of this column
        const uint8_t* col_ptr = scales_ptr +
            static_cast<size_t>(global_col) * scales_stride_dim1;

        if (row_start + 3 < scale_rows) {
            // Fast path: vectorized 4-byte load (LDG.32)
            loaded_data = __ldg(reinterpret_cast<const uint32_t*>(col_ptr + row_start));
        } else if (row_start < scale_rows) {
            // Boundary handling: load available bytes
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&loaded_data);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (row_start + i < scale_rows) {
                    bytes[i] = __ldg(col_ptr + row_start + i);
                }
            }
        }
    }

    // =========================================================================
    // STORE TO SHARED MEMORY WITH SWIZZLE PATTERN
    // Each lane writes 4 bytes to 4 swizzled locations
    // =========================================================================

    uint8_t* bytes = reinterpret_cast<uint8_t*>(&loaded_data);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Row index within the block for this byte
        int row_in_block = lane_id * 4 + i;

        // Apply swizzle: output layout is [r_mod_32 * 16 + r_div_32 * 4 + col]
        int r_div_32 = row_in_block >> 5;
        int r_mod_32 = row_in_block & 31;
        int swizzle_idx = (r_mod_32 << 4) + (r_div_32 << 2) + col_idx;

        smem_block[swizzle_idx] = bytes[i];
    }

    __syncthreads();

    // =========================================================================
    // OUTPUT WRITE PHASE (same as original)
    // Each thread writes 4 bytes (its row's data across all 4 columns)
    // =========================================================================

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
// HIGH-BANDWIDTH COLUMN-MAJOR KERNEL: 4 warps, each processing one column with
// 16-byte (uint4) vectorized loads. Each thread loads 16 consecutive rows.
// Block size: 512 rows × 4 cols = 2048 elements per block (4× more work)
// =============================================================================
#define BLOCK_ROWS_LARGE 512
#define BYTES_PER_THREAD 16

__global__ void mx_block_rearrange_2d_K_groups_colmajor_vectorized_16B_kernel(
    const uint8_t* __restrict__ scales_ptr,
    int scales_stride_dim1,  // stride between columns (= num_rows for col-major)
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

    // Warp and lane decomposition: 128 threads = 4 warps × 32 lanes
    const int warp_id = tid >> 5;      // tid / 32
    const int lane_id = tid & 31;      // tid % 32

    // Shared memory for 512 rows × 4 cols = 2048 bytes
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

    int cols_remaining = input_group_end_col - curr_input_start_col;
    int global_row_base = row_block_pid * BLOCK_ROWS_LARGE;

    // =========================================================================
    // VECTORIZED LOAD PHASE - 16 bytes per thread
    // Each warp (32 threads) handles one column
    // Each lane loads 16 consecutive rows using a uint4 (128-bit) vectorized load
    // Total per warp: 32 lanes × 16 bytes = 512 bytes (one coalesced transaction)
    // Total per block: 4 warps × 512 bytes = 2048 bytes
    // =========================================================================

    uint4 loaded_data = make_uint4(0, 0, 0, 0);

    // Each warp processes one column (warp 0 -> col 0, warp 1 -> col 1, etc.)
    int col_idx = warp_id;  // Which column this warp handles (0-3)

    if (col_idx < cols_remaining) {
        int global_col = curr_input_start_col + col_idx;

        // Each lane loads 16 consecutive rows: rows [lane_id*16, lane_id*16+15]
        int row_start = global_row_base + lane_id * BYTES_PER_THREAD;

        // Pointer to start of this column
        const uint8_t* col_ptr = scales_ptr +
            static_cast<size_t>(global_col) * scales_stride_dim1;

        if (row_start + BYTES_PER_THREAD - 1 < scale_rows) {
            // Fast path: vectorized 16-byte load (LDG.128)
            loaded_data = __ldg(reinterpret_cast<const uint4*>(col_ptr + row_start));
        } else if (row_start < scale_rows) {
            // Boundary handling: load available bytes
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&loaded_data);
            #pragma unroll
            for (int i = 0; i < BYTES_PER_THREAD; i++) {
                if (row_start + i < scale_rows) {
                    bytes[i] = __ldg(col_ptr + row_start + i);
                }
            }
        }
    }

    // =========================================================================
    // STORE TO SHARED MEMORY WITH SWIZZLE PATTERN
    // Each lane writes 16 bytes to 16 swizzled locations
    // Swizzle pattern: r_mod_32 * 16 + r_div_32 * 4 + col
    // For 512 rows: r_div_32 ranges from 0 to 15
    // =========================================================================

    uint8_t* bytes = reinterpret_cast<uint8_t*>(&loaded_data);

    #pragma unroll
    for (int i = 0; i < BYTES_PER_THREAD; i++) {
        // Row index within the block for this byte
        int row_in_block = lane_id * BYTES_PER_THREAD + i;

        // Apply swizzle: output layout is [r_mod_32 * 16 + r_div_32 * 4 + col]
        int r_div_32 = row_in_block >> 5;
        int r_mod_32 = row_in_block & 31;
        int swizzle_idx = (r_mod_32 << 4) + (r_div_32 << 2) + col_idx;

        smem_block[swizzle_idx] = bytes[i];
    }

    __syncthreads();

    // =========================================================================
    // OUTPUT WRITE PHASE
    // 128 threads need to write 512 rows × 4 cols = 2048 bytes
    // Each thread writes 16 bytes (4 rows × 4 cols each iteration, 4 iterations)
    // Or: each thread writes 4 bytes per row for 4 rows
    // =========================================================================

    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);
    int stride_per_row_of_blocks_in_group = num_col_blocks_in_group * output_stride_per_block;

    int offset_in_group = row_block_pid * stride_per_row_of_blocks_in_group +
                          local_col_block * output_stride_per_block;
    int final_offset = out_group_base_offset + offset_in_group;

    // Each thread handles 4 rows (512 rows / 128 threads = 4 rows per thread)
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int row_idx = tid + r * 128;  // Rows: tid, tid+128, tid+256, tid+384
        uint8_t* output_ptr = output_scales_ptr + final_offset + row_idx * BLOCK_COLS;

        *reinterpret_cast<uint32_t*>(output_ptr) =
            *reinterpret_cast<const uint32_t*>(&smem_block[row_idx * BLOCK_COLS]);
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
    dim3 block(128);  // 4 warps × 32 threads

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
    // Use larger block size: 512 rows instead of 128
    int num_row_blocks = (scale_rows + BLOCK_ROWS_LARGE - 1) / BLOCK_ROWS_LARGE;
    int output_stride_per_block = BLOCK_ROWS_LARGE * BLOCK_COLS;
    int total_col_blocks = (scale_cols + BLOCK_COLS - 1) / BLOCK_COLS + num_groups;

    dim3 grid(total_col_blocks, num_row_blocks);
    dim3 block(128);  // 4 warps × 32 threads

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

} // namespace mxfp8
