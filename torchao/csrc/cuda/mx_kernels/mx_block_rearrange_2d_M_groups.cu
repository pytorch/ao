#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

#define BLOCK_ROWS 128
#define BLOCK_COLS 4
#define BYTES_PER_THREAD 16
#define NUM_BUFFERS 2

__device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Here the chunks are 128x64/128x128 and superblock contains multiple horizontally (e.g. 4 of 128x64 -> 128x256).
// Group boundaries are along rows so this makes it simpler to compute blocks per group (rowwise) and local row block idx in the group.
__device__ void find_group_and_local_offset_for_superblock(
    int row_block_pid,
    const int32_t* __restrict__ input_group_end_offsets,
    int num_groups,
    int* __restrict__ smem_data,  // smem_data[0..num_groups-1] = chunks_in_group, smem_data[num_groups..2*num_groups-1] = super_blocks_in_group cumsum
    int& group_id,
    int& first_chunk_in_group,
    int& chunks_until_group_end
) {
    // Cumsum tiles rowwise so we can figure out how tiles span this group, what local tile row idx we are within the group.
    // We use one thread to compute this.
    if (threadIdx.x == 0) {
        int block_cumsum = 0;
        for (int g = 0; g < num_groups; g++) {
            int input_group_start = (g > 0) ? input_group_end_offsets[g - 1] : 0;
            int input_group_end = input_group_end_offsets[g];
            int group_size = input_group_end - input_group_start;
            int num_blocks_rowwise = ceil_div(group_size, BLOCK_ROWS);
            smem_data[g] = num_blocks_rowwise;
            block_cumsum += num_blocks_rowwise;
            smem_data[num_groups + g] = block_cumsum;
        }
    }
    __syncthreads();

    // Now every thread uses the SMEM cumsum to find (1) which group their row block pid belongs to,
    // and (2) the number of row blocks remaining in the group, and store those values in its local registers.
    group_id = 0;
    int block_cumsum_before = 0;
    for (int g = 0; g < num_groups; g++) {
        int cumsum_at_g = smem_data[num_groups + g];
        if (row_block_pid < cumsum_at_g) {
            group_id = g;
            int local_row_block_idx = row_block_pid - block_cumsum_before;
            first_chunk_in_group = local_row_block_idx;
            int chunks_in_group = smem_data[g];
            chunks_until_group_end = chunks_in_group - first_chunk_in_group;
            return;
        }
        block_cumsum_before = cumsum_at_g;
    }

    first_chunk_in_group = 0;
    chunks_until_group_end = 0;
}

__device__ __forceinline__ int compute_output_group_start_row(
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

template <int MAX_COLS, int CHUNKS_PER_TB>
__global__ void mx_block_rearrange_2d_M_groups_128x4_vec_pipelined(
    const uint8_t* __restrict__ input_scales,
    const int input_scales_stride_dim0,
    const int32_t* __restrict__ input_group_end_offsets,
    const uint8_t* __restrict__ output_scales,
    const int num_groups,
) {
    const int row_block_pid = blockIdx.y;
    const int col_block_pid = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_idx = tid / BLOCK_COLS;
    const int col_idx = tid % BLOCK_COLS;
    const int SMEM_SIZE = BLOCK_ROWS * MAX_COLS;

    // Shared memory
    __shared__ int smem_group_data[64]; // max 32 groups; 32 ints for group sizes, 32 for cumsums at each group
    __shared__ __align__(16) uint8_t smem_buffers[NUM_BUFFERS][SMEM_SIZE];

    // Find:
    // 1. The group idx this block belongs to.
    // 2. The local row block idx within the group.
    // 3. The number of row blocks remaining until the end of the group.
    int group_idx;
    int local_row_block;
    int row_blocks_until_end;
    find_group_and_local_offset_for_superblock(
        // inputs
        row_block_pid,
        input_group_end_offsets,
        num_groups,
        smem_group_data,
        // outputs
        &group_idx,
        &local_row_block,
        &row_blocks_until_end
    );


    // Use one thread in the threadblock to compute group boundaries and output group start,
    // then broadcast the values via SMEM.
    // This avoids (1) unnecessary extra global accesses, and (2) extra register pressure, and
    // (3) extra ALU usage by every thread computing redundant values, in a kernel which is already ALU heavy.
    // It comes at the cost of these few SMEM accesses and thread block sync, but benchmarks show this is slightly
    // better than having all threads do this redundant work.
    __shared__ int s_input_group_start_row;
    __shared__ int s_input_group_end_row;
    __shared__ int s_output_group_start_row;
    if (tid == 0)
    {
        s_input_group_start_row = (group_idx > 0) ? input_group_end_offsets[group_idx - 1] : 0;
        s_input_group_end_row = input_group_end_offsets[group_idx];
        s_output_group_start_row = compute_output_group_start_row(group_idx, input_group_end_offsets, num_groups, BLOCK_ROWS);
    }
    int input_group_start_row = s_input_group_start_row;
    int input_group_end_row = s_input_group_end_row;
    int output_group_start_row = s_output_group_start_row;

    // Prepare for pipeline phase, compute thread constant vals
    int thread_col_start = col_idx * BYTES_PER_THREAD;
    int global_row_base = input_group_start_row + (local_row_block * BLOCK_ROWS);
    int global_col_base = (col_block_pid * MAX_COLS) + thread_col_start;
    const uint8_t* base_ptr = input_scales + static_cast<size_t>(global_row_base) * input_scales_stride_dim0 + static_cast<size_t>(global_col_base);

    // Blocked layout swizzle base
    int r_div_32 = row_idx << 5;
    int r_mod_32 = row_idx & 31;
    int blocked_layout_base = (r_mod_32 << 4) + (r_div_32 << 2);

    // Compute XOR swizzle to avoid bank conflicts on both reads and writes
    int tile_xor = (col_idx >> 2) & 3;                  // (col_idx / 4) % 4
    int superrow_xor = (blocked_layout_base >> 7) & 3;  // (blocked_layout_base / 128) % 4
    int combined_xor = tile_xor ^ superrow_xor;

    // Chunk async load func
    auto load_chunk = [&](int chunk_idx, int buf_idx) {
        int rows_to_load = min(BLOCK_ROWS, input_group_end_row - global_row_base);
        bool thread_can_load = row_idx < rows_to_load;
        if (thread_can_load)
        {
            // Shift to next chunk colwise
            const uint8_t* src_ptr = base_ptr + static_cast<size_t>(chunk_idx) * MAX_COLS;
            bool aligned = reinterpret_cast<uintptr_t>(src_ptr) % 16 == 0;

            // Use uint4 vectorized loads if possible
            if (aligned)
            {
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :
                    : "l"(&smem_buffers[buf_idx][row_idx * MAX_COLS + thread_col_start]),
                      "l"(src_ptr)
                );
            }

            // Fall back to single byte loads if alignment is invalid
            else
            {
                uint4 row_data = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // Chunk process func
    auto process_chunk = [&](int chunk_idx, int buffer_idx) {

    };

    // Pipeline to overlap loads of chunk N+1 with process + store of chunk N
}

namespace mxfp8 {

} // namespace mxfp8
