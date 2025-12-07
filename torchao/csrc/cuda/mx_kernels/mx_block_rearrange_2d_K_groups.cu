#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#define BLOCK_ROWS 128
#define BLOCK_COLS 4
// Helper function to compute ceil division
__device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
// Helper function to compute the start index of a group after padding
__device__ __forceinline__ int compute_output_group_start_col(
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
// For a 128x4 block: r_div_32 = row / 32, r_mod_32 = row % 32
// Swizzle: dest = r_mod_32 * 16 + r_div_32 * 4 + col
__device__ __forceinline__ int compute_swizzled_index(int row, int col) {
    int r_div_32 = row / 32;
    int r_mod_32 = row % 32;
    return r_mod_32 * 16 + r_div_32 * 4 + col;
}
__global__ void mx_block_rearrange_2d_K_groups_naive_kernel(
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
    const int group_id = blockIdx.x;
    const int block_row_id = blockIdx.y;
    const int tid = threadIdx.x;  // 128 threads, each handles one row
    // Shared memory for one 128x4 block
    __shared__ __align__(16) uint8_t smem_block[BLOCK_ROWS * BLOCK_COLS];
    // Get start/end cols of this input group
    int input_group_start_col = (group_id > 0) ? input_group_end_offsets[group_id - 1] : 0;
    int input_group_end_col = input_group_end_offsets[group_id];
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    // Get output group start column
    int output_group_start_col = compute_output_group_start_col(
        group_id,
        input_group_end_offsets,
        num_groups,
        4); // scaling factor column padding size
    // Compute base offset for this group in output
    int out_group_base_offset = output_group_start_col * padded_rows;
    // Compute stride per row of blocks in this group
    int num_col_blocks_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);
    int stride_per_row_of_blocks_in_group = num_col_blocks_in_group * output_stride_per_block;
    // Each thread handles one row
    int input_row = block_row_id * BLOCK_ROWS + tid;
    // Loop through column blocks in this group
    int curr_input_start_col = input_group_start_col;
    int curr_out_col_block = 0;
    while (curr_input_start_col < input_group_end_col) {
        // Calculate how many columns to load for this block
        int cols_remaining = input_group_end_col - curr_input_start_col;
        int cols_to_load = min(BLOCK_COLS, cols_remaining);
        // Load data for this row using vectorized loads when possible
        uint32_t row_data = 0;
        if (input_row < scale_rows && curr_input_start_col < input_group_end_col) {
            int input_offset = input_row * scales_stride_dim0 + curr_input_start_col;
            const uint8_t* input_ptr = scales_ptr + input_offset;
            // Check alignment and available columns within this group
            uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(input_ptr);
            if (cols_to_load >= 4 && ptr_addr % 4 == 0 && curr_input_start_col + 4 <= input_group_end_col) {
                // 4-byte aligned and have 4 columns within group: use uint32_t load
                row_data = __ldg(reinterpret_cast<const uint32_t*>(input_ptr));
            } else {
                // Byte-by-byte loads for unaligned or partial blocks
                uint8_t* row_bytes = reinterpret_cast<uint8_t*>(&row_data);
                for (int i = 0; i < cols_to_load && (curr_input_start_col + i) < input_group_end_col; i++) {
                    row_bytes[i] = __ldg(input_ptr + i);
                }
            }
        }
        // Write to swizzled positions in shared memory
        uint8_t* row_bytes = reinterpret_cast<uint8_t*>(&row_data);
        #pragma unroll
        for (int col = 0; col < BLOCK_COLS; col++) {
            int swizzled_idx = compute_swizzled_index(tid, col);
            smem_block[swizzled_idx] = row_bytes[col];
        }
        __syncthreads();
        // Write from shared memory to global memory
        // Calculate the output offset for this specific block
        int offset_in_group = block_row_id * stride_per_row_of_blocks_in_group +
                              curr_out_col_block * output_stride_per_block;
        int final_offset = out_group_base_offset + offset_in_group;
        // Each thread writes 4 bytes (one row of the 128x4 block)
        uint8_t* output_ptr = output_scales_ptr + final_offset + tid * BLOCK_COLS;
        // Check output alignment for vectorized write
        uintptr_t out_ptr_addr = reinterpret_cast<uintptr_t>(output_ptr);
        if (out_ptr_addr % 4 == 0) {
            // Aligned: use uint32_t store
            *reinterpret_cast<uint32_t*>(output_ptr) =
                *reinterpret_cast<const uint32_t*>(&smem_block[tid * BLOCK_COLS]);
        } else {
            // Unaligned: byte by byte
            const uint8_t* smem_ptr = &smem_block[tid * BLOCK_COLS];
            #pragma unroll
            for (int i = 0; i < BLOCK_COLS; i++) {
                output_ptr[i] = smem_ptr[i];
            }
        }
        // Advance to next column block
        curr_input_start_col += BLOCK_COLS;
        curr_out_col_block += 1;
        // Only sync if there's another iteration
        if (curr_input_start_col < input_group_end_col) {
            __syncthreads();
        }
    }
}
// Host function to launch the kernel
namespace mxfp8 {
void launch_mx_block_rearrange_2d_K_groups(
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
    // Grid parallelizes over (num_groups, num_row_blocks)
    // Each thread block loops through column blocks within its group
    dim3 grid(num_groups, num_row_blocks);
    dim3 block(128); // 128 threads, each handling one row
    int output_stride_per_block = BLOCK_ROWS * BLOCK_COLS;
    mx_block_rearrange_2d_K_groups_naive_kernel<<<grid, block, 0, stream>>>(
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
