#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

#define BLOCK_ROWS 128
#define BLOCK_COLS 16

// Helper function to compute ceil division
__device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Helper function to compute the start index of a group after padding
__device__ int compute_group_start_col(
    int group_id,
    const int32_t* input_group_end_offsets,
    int num_groups,
    int padding_size
) {
    int start_idx = 0;

    // Compute prefix sum of padded group sizes
    for (int i = 0; i < group_id; i++) {
        int prev_offset = (i > 0) ? input_group_end_offsets[i - 1] : 0;
        int curr_offset = input_group_end_offsets[i];
        int group_size = curr_offset - prev_offset;
        int padded_size = ceil_div(group_size, padding_size) * padding_size;
        start_idx += padded_size;
    }

    return start_idx;
}

// Compute destination index for swizzled block layout
__device__ __forceinline__ int compute_swizzled_index(int row, int col) {
    int r_div_32 = row / 32;
    int r_mod_32 = row % 32;
    return r_mod_32 * 16 + r_div_32 * 4 + col;
}

// Vectorized search to find which group a global column block belongs to
// Computes prefix sum on-the-fly from input_group_end_offsets
__device__ void find_group_for_col_block_from_offsets(
    int tb_col_idx,
    const int32_t* input_group_end_offsets,
    int num_groups,
    int block_cols,
    int& group_id,
    int& local_col_block_idx
) {
    // Compute prefix sums on-the-fly (num_groups is small: 4-16)
    // Only need exclusive cumsum; upper bound computed on-the-fly
    int cumsum_exclusive[32];  // Max 32 groups

    int running_sum = 0;
    for (int i = 0; i < num_groups; i++) {
        cumsum_exclusive[i] = running_sum;

        // Compute group size and number of column blocks
        int group_start = (i > 0) ? input_group_end_offsets[i - 1] : 0;
        int group_end = input_group_end_offsets[i];
        int group_size = group_end - group_start;
        int num_col_blocks = (group_size + block_cols - 1) / block_cols;  // Ceiling division

        running_sum += num_col_blocks;
    }

    // Find which group this block belongs to
    group_id = 0;
    local_col_block_idx = 0;

    for (int i = 0; i < num_groups; i++) {
        // Upper bound is the start of next group (or INT_MAX for last group)
        int next_cumsum = (i < num_groups - 1) ? cumsum_exclusive[i + 1] : INT_MAX;
        if (tb_col_idx >= cumsum_exclusive[i] &&
            tb_col_idx < next_cumsum) {
            group_id = i;
            local_col_block_idx = tb_col_idx - cumsum_exclusive[i];
            break;
        }
    }
}

__global__ void mx_block_rearrange_2d_K_groups_kernel_parallel(
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
    const int tb_col_idx = blockIdx.x;
    const int tb_row_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // Each thread handles 4 consecutive bytes. Each thread block handles 128x16 input/output block
    const int tid_row = tid / 4;
    const int tid_col = tid % 4;

    // Shared memory for:
    // 1. Prefix sums (computed once per thread block)
    // 2. 4 128x4 blocks in swizzled layout. 16 byte alignment for vectorized stores.
    extern __shared__ int cumsum_exclusive[];
    __shared__ __align__(16) uint8_t smem_block[BLOCK_ROWS * BLOCK_COLS];

    // Thread 0 computes prefix sums for all threads to use
    if (tid == 0) {
        int running_sum = 0;
        for (int i = 0; i < num_groups; i++) {
            cumsum_exclusive[i] = running_sum;

            // Compute group size and number of column blocks
            int group_start = (i > 0) ? input_group_end_offsets[i - 1] : 0;
            int group_end = input_group_end_offsets[i];
            int group_size = group_end - group_start;
            int num_col_blocks = (group_size + BLOCK_COLS - 1) / BLOCK_COLS;

            running_sum += num_col_blocks;
        }
    }
    __syncthreads();  // Ensure prefix sums are available to all threads

    // All threads now find which group this column block belongs to
    int group_id = 0;
    int local_tb_col_idx = 0;

    for (int i = 0; i < num_groups; i++) {
        // Upper bound is the start of next group (or INT_MAX for last group)
        int next_cumsum = (i < num_groups - 1) ? cumsum_exclusive[i + 1] : INT_MAX;
        if (tb_col_idx >= cumsum_exclusive[i] &&
            tb_col_idx < next_cumsum) {
            group_id = i;
            local_tb_col_idx = tb_col_idx - cumsum_exclusive[i];
            break;
        }
    }

    // Get input group bounds
    int input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
    int input_group_end_col = input_group_end_offsets[group_id];

    // Compute input column offset for this specific column block
    int curr_input_start_col = input_group_start_col + local_tb_col_idx * BLOCK_COLS;

    // Early exit if beyond group boundary
    if (curr_input_start_col >= input_group_end_col) {
        return;
    }

    // Get output group start column
    int output_group_start_col = compute_group_start_col(
        group_id,
        input_group_end_offsets,
        num_groups,
        4); // scaling factor column padding size

    // Compute base offset for this group in output
    int out_group_base_offset = output_group_start_col * padded_rows;

    // Compute number of column blocks in this group
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_col_blocks_in_group = ceil_div(num_cols_in_group, 4);
    int stride_per_row_of_blocks_in_group = num_col_blocks_in_group * output_stride_per_block;

    // Each thread handles one row
    int input_row = tb_row_idx * BLOCK_ROWS + tid_row;

    // tid_col=0 loads the first 4 bytes, tid_col=1 loads the next 4 bytes, etc along a 16 byte row
    uint32_t row_data = 0;
    if (input_row < scale_rows) {
        int input_offset = input_row * scales_stride_dim0 + curr_input_start_col * scales_stride_dim1;
        const uint8_t* input_ptr = scales_ptr + input_offset + tid_col * 4;

        // Check if the address is 4-byte aligned for vectorized uint32_t load
        uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(input_ptr);
        if (ptr_addr % 4 == 0) {
            // Aligned access - use vectorized load
            row_data = *reinterpret_cast<const uint32_t*>(input_ptr);
        } else {
            // Unaligned access - load byte by byte and pack into uint32_t
            row_data = static_cast<uint32_t>(input_ptr[0]) |
                       (static_cast<uint32_t>(input_ptr[1]) << 8) |
                       (static_cast<uint32_t>(input_ptr[2]) << 16) |
                       (static_cast<uint32_t>(input_ptr[3]) << 24);
        }
    }

    // Write to swizzled positions in shared memory
    uint8_t bytes[4];
    bytes[0] = row_data & 0xFF;
    bytes[1] = (row_data >> 8) & 0xFF;
    bytes[2] = (row_data >> 16) & 0xFF;
    bytes[3] = (row_data >> 24) & 0xFF;

    #pragma unroll
    for (int col = 0; col < 4; col++) {
        int global_col = tid_col * 4 + col;  // Actual column: 0-15
        // todo: this swizzle is designed for a single 128x4 block of ((32,4),4) layout
        // need to update this for this 128x16 blocks of ((32,4), 16) layout
        int swizzled_idx = compute_swizzled_index(tid_row, global_col);
        smem_block[swizzled_idx] = bytes[col];
    }

    __syncthreads();

    // Vectorized stores from shared memory to global memory
    int offset_in_group = tb_row_idx * stride_per_row_of_blocks_in_group +
                          local_tb_col_idx * output_stride_per_block;
    int final_offset = out_group_base_offset + offset_in_group;

    // Cast to uint32_t pointer for vectorized write
    uint32_t* output_uint32 = reinterpret_cast<uint32_t*>(output_scales_ptr + final_offset);
    const uint32_t* smem_uint32 = reinterpret_cast<const uint32_t*>(smem_block);

    // Each thread writes one uint32_t
    output_uint32[tid] = smem_uint32[tid];
}

// Host function to launch the kernel
namespace mxfp8 {

void launch_mx_block_rearrange_2d_K_groups(
    const uint8_t* scales_ptr,
    int scales_stride_dim0,
    int scales_stride_dim1,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int output_stride_per_block,
    int num_groups,
    cudaStream_t stream
) {
    int num_row_blocks = (scale_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int num_col_blocks = (scale_cols + BLOCK_COLS - 1) / BLOCK_COLS;

    dim3 grid(num_col_blocks, num_row_blocks);
    dim3 block(128 * 4); // 128x4 threadblock, each handling 128x16 input/output block

    int dyn_smem_bytes = num_groups * sizeof(int32_t); // 4 bytes per int
    mx_block_rearrange_2d_K_groups_kernel_parallel<<<grid, block, dyn_smem_bytes, stream>>>(
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
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // }
}

} // namespace mxfp8
