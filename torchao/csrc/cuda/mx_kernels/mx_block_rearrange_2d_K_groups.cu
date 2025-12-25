#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

#define BLOCK_ROWS 128
#define BLOCK_COLS 4
#define BYTES_PER_THREAD 16
#define SCALE_FACTOR_ROWS 128

__device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Terminology:
// tile = 128x4 scaling factor tile
// chunk = chunk of data consisting of multiple tiles (e.g., 128x64 or 128x128)
// superblock = consists of CHUNKS_PER_TB chunks along the column dimension
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
    constexpr int TILES_PER_THREAD = 4;  // Each thread processes 16 bytes = 4 tiles (4 bytes per tile)
    constexpr int TILE_SIZE = SCALE_FACTOR_ROWS * BLOCK_COLS;  // 128 rows * 4 cols = 512 bytes per tile
    constexpr int SMEM_SIZE = BLOCK_ROWS * MAX_COLS;
    constexpr int NUM_BUFFERS = 2;

    const int super_col_block_pid = blockIdx.x;
    const int row_block_pid = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ __align__(16) uint8_t smem[NUM_BUFFERS][SMEM_SIZE];
    __shared__ int smem_group_data[64]; // max 32 groups; 1 int per group size, 1 int per group prefix sum
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
        smem_group_data,
        group_id,
        first_chunk_in_group,
        chunks_until_group_end
    );

    // Use one thread in the threadblock to compute group boundaries and output group start,
    // then broadcast the values via SMEM.
    // This avoids (1) unnecessary extra global accesses, and (2) extra register pressure, and
    // (3) extra ALU usage by every thread computing redundant values, in a kernel which is already ALU heavy.
    // It comes at the cost of these few SMEM accesses and thread block sync, but benchmarks show this is slightly
    // better than having all threads do this redundant work.
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

    int r_div_32 = row_idx >> 5; // row / 32
    int r_mod_32 = row_idx & 31; // row % 32
    int swizzle_base = (r_mod_32 << 4) + (r_div_32 << 2); // (row % 32) * 16 + (row / 32) * 4
    int thread_col_start = col_idx * 16;
    int first_tile_idx = thread_col_start >> 2; // thread_col_start / 4

    int out_group_base_offset = output_group_start_col * padded_rows;
    int num_cols_in_group = input_group_end_col - input_group_start_col;
    int num_tiles_in_group = ceil_div(num_cols_in_group, BLOCK_COLS);
    int tiles_stride_per_row_block = num_tiles_in_group * TILE_SIZE;

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

        int tile_xor = (col_idx & 3) << 2; // col_idx % 4
        int superrow_xor = ((swizzle_base >> 7) & 3) << 2; // ((swizzle_base / 128) % 4) * 4
        int combined_xor = tile_xor ^ superrow_xor;

        uint32_t* data32 = reinterpret_cast<uint32_t*>(&data);
        #pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; t++) {
            int tile_idx = first_tile_idx + t;
            int tile_base = tile_idx * SCALE_FACTOR_ROWS * BLOCK_COLS;
            int swizzled_idx = tile_base + (swizzle_base ^ combined_xor);
            *reinterpret_cast<uint32_t*>(&smem[buf_idx][swizzled_idx]) = data32[t];
        }

        __syncthreads();

        // Compute output pointer: skip past tiles from previous chunks in this group
        int tiles_before_this_chunk = curr_chunk_in_group * (MAX_COLS / BLOCK_COLS);
        uint8_t* out_base = output_scales_ptr + out_group_base_offset +
                            row_block_pid * tiles_stride_per_row_block +
                            tiles_before_this_chunk * TILE_SIZE;

        int num_tiles_this_chunk = ceil_div(cols_to_load, BLOCK_COLS);
        int bytes_to_copy = num_tiles_this_chunk * TILE_SIZE;

        // Each thread writes 16 bytes (4 tiles worth of data for its row position)
        int byte_offset = tid * 16;
        if (byte_offset < bytes_to_copy) {
            uint32_t out_data[4];

            // Read 4 uint32s from swizzled SMEM layout, accounting for writer's XOR pattern
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int out_byte = byte_offset + i * 4;
                int tile_idx = out_byte / TILE_SIZE;
                int within_tile_offset = out_byte % TILE_SIZE;

                int writer_col_idx = (tile_idx / TILES_PER_THREAD) % THREADS_PER_ROW;
                int writer_tile_xor = (writer_col_idx & 3) << 2; // (writer_col % 4) * 4
                int writer_superrow_xor = ((within_tile_offset >> 7) & 3) << 2; // ((within_tile_offset / 128) % 4) * 4
                int writer_combined_xor = writer_tile_xor ^ writer_superrow_xor;
                int smem_addr = tile_idx * TILE_SIZE + (within_tile_offset ^ writer_combined_xor);
                out_data[i] = *reinterpret_cast<uint32_t*>(&smem[buf_idx][smem_addr]);
            }

            *reinterpret_cast<uint4*>(out_base + byte_offset) =
                *reinterpret_cast<uint4*>(out_data);
        }
    };

    if (num_chunks_to_process == 1)
    {
        load_chunk_async(0, 0);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        process_chunk(0, 0);
    }
    else
    {
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

            // Wait for async load of chunk N, without waiting for chunk N+1, to achieve overlap.
            // async loads completed in commit order (FIFO) so this should work.
            asm volatile("cp.async.wait_group 1;\n");
            __syncthreads();

            // Process chunk N, overlapping with async load of chunk N+1.
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
