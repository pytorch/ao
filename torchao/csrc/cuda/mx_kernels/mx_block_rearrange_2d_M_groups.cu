/**
 * MX Block Rearrange 2D Kernel for M-Groups (Groups along row dimension)
 *
 * TERMINOLOGY:
 * - SF Tile: A 128x4 scale factor tile (512 bytes in blocked layout)
 *   - SF_ROWS = 128 rows
 *   - SF_COLS = 4 columns
 *   - Each SF tile is stored contiguously in the output blocked layout
 *
 * - Chunk: A 128xCHUNK_WIDTH region (128 rows × 64 or 128 columns)
 *   - One TMA load brings one chunk from GMEM to SMEM
 *   - Contains multiple SF tiles horizontally (CHUNK_WIDTH/SF_COLS = 16 or 32)
 *   - Each chunk is processed independently in the pipeline
 *
 * - Superblock: CHUNKS_PER_TB chunks stacked vertically
 *   - Processed by one thread block
 *   - Height = SF_ROWS * CHUNKS_PER_TB (512, 1024, or 2048 rows)
 *   - Width = CHUNK_WIDTH (64 or 128 columns)
 *   - Grid is organized as (num_col_chunks, num_row_superblocks)
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include "ptx.cuh"

// Overloaded error checking for both CUDA driver API (CUresult) and runtime API (cudaError_t)
inline void cuda_check_impl(CUresult result, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA Driver Error at %s:%d - Error code: %d\n", file, line, (int)result);
        exit(EXIT_FAILURE);
    }
}

inline void cuda_check_impl(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d - %s\n", file, line,
                cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(result) cuda_check_impl((result), __FILE__, __LINE__)

// Get the driver entry point for cuTensorMapEncodeTiled (used for TMA tensor maps)
inline void* get_driver_ptr() {
    static void *driver_ptr = nullptr;
    if (!driver_ptr) {
        cudaDriverEntryPointQueryResult result;
        CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &driver_ptr,
                                cudaEnableDefault, &result));
    }
    return driver_ptr;
}

// Device helper: ceiling division
__device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

#define SF_ROWS 128
#define SF_COLS 4

// Each thread reads 16 bytes (vectorized uint4 load)
#define BYTES_PER_THREAD 16

// Double buffering for pipelined TMA loads/stores
#define NUM_BUFFERS 2



// Here the chunks are 128x64/128x128 and superblock contains multiple vertical chunks (e.g., 4 chunks of 128x64 would be 512x64).
template <int CHUNKS_PER_TB>
__device__ void find_group_and_local_offset_for_superblock(
    int row_superblock_pid,
    const int32_t* __restrict__ input_group_end_offsets,
    int num_groups,
    int* __restrict__ smem_data,
    int& group_id,
    int& superblock_idx_in_group,
    int& superblocks_until_end
) {
    // Thread 0 computes how many superblocks span each group
    if (threadIdx.x == 0) {
        constexpr int SUPERBLOCK_ROWS = SF_ROWS * CHUNKS_PER_TB;
        int superblock_cumsum = 0;
        for (int g = 0; g < num_groups; g++) {
            int input_group_start = (g > 0) ? input_group_end_offsets[g - 1] : 0;
            int input_group_end = input_group_end_offsets[g];
            int group_size = input_group_end - input_group_start;
            int num_superblocks_in_group = ceil_div(group_size, SUPERBLOCK_ROWS);
            smem_data[g] = num_superblocks_in_group;
            superblock_cumsum += num_superblocks_in_group;
            smem_data[num_groups + g] = superblock_cumsum;
        }
    }
    __syncthreads();

    // All threads use the SMEM cumsum to find their group
    group_id = 0;
    int superblock_cumsum_before = 0;
    for (int g = 0; g < num_groups; g++) {
        int cumsum_at_g = smem_data[num_groups + g];
        if (row_superblock_pid < cumsum_at_g) {
            group_id = g;
            superblock_idx_in_group = row_superblock_pid - superblock_cumsum_before;
            int superblocks_in_group = smem_data[g];
            superblocks_until_end = superblocks_in_group - superblock_idx_in_group;
            return;
        }
        superblock_cumsum_before = cumsum_at_g;
    }

    superblock_idx_in_group = 0;
    superblocks_until_end = 0;
}

/**
 * Compute the starting row in the output buffer for a given group.
 * Each group is padded to a multiple of SF_ROWS (128) in the output.
 */
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

/**
 * Main kernel for rearranging row-major scale data to blocked layout.
 *
 * Template parameters:
 * - CHUNK_WIDTH: Width of each chunk (64 or 128 columns)
 * - CHUNKS_PER_TB: Number of chunks per superblock (4, 8, or 16)
 *
 * Grid dimensions:
 * - blockIdx.x = column chunk index
 * - blockIdx.y = row superblock index
 *
 * Thread block dimensions:
 * - 512 threads for CHUNK_WIDTH=64 (128 rows × 4 threads/row)
 * - 1024 threads for CHUNK_WIDTH=128 (128 rows × 8 threads/row)
 */
template <int CHUNK_WIDTH, int CHUNKS_PER_TB>
__global__ void mx_blocked_layout_2d_M_groups_kernel(
    const uint8_t* __restrict__ input_scales,
    const int input_scale_stride_dim0,
    const int scale_rows,
    const int scale_cols,
    const int padded_rows,
    const int32_t* __restrict__ input_group_end_offsets,
    uint8_t* __restrict__ output_scales,
    const int output_stride_per_row_of_sf_tiles,
    const int num_groups,
    const __grid_constant__ CUtensorMap input_tensor_map
) {
    constexpr int output_stride_per_sf_tile = SF_ROWS * SF_COLS;  // 512 bytes per SF tile
    const int row_superblock_pid = blockIdx.y;
    const int col_chunk_pid = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_master_thread = tid == 0;
    constexpr int THREADS_PER_ROW = CHUNK_WIDTH / BYTES_PER_THREAD;
    const int row_idx = tid / THREADS_PER_ROW;
    const int col_idx = tid % THREADS_PER_ROW;

    constexpr int CHUNK_SIZE_BYTES = SF_ROWS * CHUNK_WIDTH;

    __shared__ int smem_group_data[64]; // Max 32 groups: [0..31] = superblocks per group, [32..63] = cumsum
    __shared__ __align__(128) uint8_t smem_buffers[NUM_BUFFERS][CHUNK_SIZE_BYTES];

    // Find which group this superblock belongs to and compute offsets
    int group_idx;
    int superblock_idx_in_group;
    int superblocks_until_end;
    find_group_and_local_offset_for_superblock<CHUNKS_PER_TB>(
        row_superblock_pid,
        input_group_end_offsets,
        num_groups,
        smem_group_data,
        group_idx,
        superblock_idx_in_group,
        superblocks_until_end
    );

    // Early exit for padding superblocks beyond all groups
    if (superblocks_until_end <= 0) {
        return;
    }

    // Thread 0 computes group boundaries and broadcasts via SMEM
    __shared__ int s_input_group_start_row;
    __shared__ int s_input_group_end_row;
    __shared__ int s_output_group_start_row;
    if (tid == 0) {
        s_input_group_start_row = (group_idx > 0) ? input_group_end_offsets[group_idx - 1] : 0;
        s_input_group_end_row = input_group_end_offsets[group_idx];
        s_output_group_start_row = compute_output_group_start_row(
            group_idx, input_group_end_offsets, num_groups, SF_ROWS);
    }
    __syncthreads();
    uint input_group_start_row = s_input_group_start_row;
    uint input_group_end_row = s_input_group_end_row;
    uint output_group_start_row = s_output_group_start_row;

    // Compute base addresses for this superblock
    constexpr int SUPERBLOCK_ROWS = SF_ROWS * CHUNKS_PER_TB;
    const uint global_row_base = input_group_start_row + (superblock_idx_in_group * SUPERBLOCK_ROWS);
    const uint global_col_base = col_chunk_pid * CHUNK_WIDTH;

    // Thread's column offset and first SF tile index
    const uint thread_col_start = col_idx * BYTES_PER_THREAD;
    const uint first_sf_tile_idx = thread_col_start / SF_COLS;

    // Initialize mbarriers for TMA load coordination
    constexpr int NUM_THREADS = SF_ROWS * THREADS_PER_ROW;
    __shared__ __align__(8) uint64_t mbar[NUM_BUFFERS];
    initialize_barriers<NUM_BUFFERS, NUM_THREADS>(mbar, is_master_thread);

    // Track barrier parity for double-buffered pipeline
    int buf_parity[NUM_BUFFERS] = {0};

    // Compute columns in this chunk (may be less than CHUNK_WIDTH for partial chunks)
    // This is used for SMEM read stride and output bounds checking
    // NOTE: TMA always loads the full tensor map box size regardless of this value
    auto compute_cols_in_chunk = [&]() -> int {
        const int remaining_cols = scale_cols - global_col_base;
        return min(CHUNK_WIDTH, remaining_cols);
    };
    const int cols_in_chunk = compute_cols_in_chunk();

    // The tensor map box width is min(CHUNK_WIDTH, scale_cols)
    // TMA always loads this full box size (with OOB elements zero-filled)
    const int tma_box_width = min(CHUNK_WIDTH, scale_cols);

    // Func for async TMA load of a chunk of data to the given SMEM buffer
    auto load_chunk = [&](int chunk_idx, int buf_idx) {
        if (chunk_idx >= NUM_BUFFERS) {
            // Wait for pending TMA stores to finish reading the SMEM buffer we're about to re-use
            ptx::cp_async_bulk_wait_group_read<0>();
            __syncthreads();
        }

        uint64_t* mbar_ptr = &mbar[buf_idx];
        const uint32_t tma_x = global_col_base;
        const uint32_t tma_y = global_row_base + (chunk_idx * SF_ROWS);

        // IMPORTANT: TMA with OOB_FILL_NONE zero-fills OOB elements but reports
        // the FULL box size to mbarrier, not the clamped size. The box size is
        // fixed in the tensor map descriptor as (tma_box_width x SF_ROWS).
        const int expected_bytes = SF_ROWS * tma_box_width;

        copy_2d_to_shared(
            (void*)&smem_buffers[buf_idx],
            reinterpret_cast<const void*>(&input_tensor_map),
            tma_x,
            tma_y,
            expected_bytes,
            mbar_ptr,
            is_master_thread
        );
    };

    // Precompute blocked layout offset for this thread's row
    int r_div_32 = row_idx >> 5;    // row_idx / 32
    int r_mod_32 = row_idx & 31;    // row_idx % 32
    int blocked_layout_row_offset = (r_mod_32 << 4) + (r_div_32 << 2); // r_mod_32 * 16 + r_div_32 * 4

    // Func to process blocked layout transform in SMEM then store result to GMEM
    auto process_chunk = [&](int chunk_idx, int buf_idx) {
        // Wait for TMA load to complete
        uint64_t* mbar_ptr = &mbar[buf_idx];
        ptx::mbarrier_wait_parity(mbar_ptr, buf_parity[buf_idx]);
        buf_parity[buf_idx] ^= 1;

        // Fence to ensure TMA-written data is visible to all threads
        ptx::fence_proxy_async_shared_cta();

        // Read 16 bytes from row-major layout in SMEM. Coalesced, vectorized reads.
        // TMA writes rows with stride = tma_box_width (the tensor map box size)
        uint4 row_data = make_uint4(0, 0, 0, 0);
        const uint32_t chunk_start_row = global_row_base + (chunk_idx * SF_ROWS);
        const bool row_valid = chunk_start_row + row_idx < input_group_end_row;
        const bool col_valid = thread_col_start < static_cast<uint>(cols_in_chunk);

        if (row_valid && col_valid) {
            row_data = *reinterpret_cast<uint4*>(
                &smem_buffers[buf_idx][row_idx * tma_box_width + thread_col_start]);
        }
        __syncthreads();  // All threads must finish reading before overwriting

        // Scatter 16 bytes to blocked layout (4 bytes per SF tile)
        // Only write to SMEM if this thread's columns are within the valid range
        constexpr int SF_TILES_PER_THREAD = BYTES_PER_THREAD / SF_COLS;
        if (col_valid) {
            #pragma unroll
            for (int i = 0; i < SF_TILES_PER_THREAD; i++) {
                const int sf_tile_idx = first_sf_tile_idx + i;
                const uint32_t data = reinterpret_cast<const uint32_t*>(&row_data)[i];
                *reinterpret_cast<uint32_t*>(
                    &smem_buffers[buf_idx][blocked_layout_row_offset + sf_tile_idx * output_stride_per_sf_tile]
                ) = data;
            }
        }
        __syncthreads();

        // Fence to ensure SMEM writes are visible to TMA async proxy
        ptx::fence_proxy_async_shared_cta();

        // Compute output tile coordinates
        const int chunk_sf_row_tile = (superblock_idx_in_group * CHUNKS_PER_TB) + chunk_idx;
        const int global_sf_row_tile = (output_group_start_row / SF_ROWS) + chunk_sf_row_tile;

        // Number of valid SF tiles horizontally in this chunk (may be fewer for partial chunks)
        const int valid_sf_tiles = ceil_div(cols_in_chunk, SF_COLS);
        constexpr int SF_TILES_PER_CHUNK = CHUNK_WIDTH / SF_COLS;  // Max tiles for full chunk
        const int col_sf_tile_start = col_chunk_pid * SF_TILES_PER_CHUNK;

        // Check if this chunk has valid data
        const uint32_t rows_remaining_in_group = (input_group_end_row > chunk_start_row)
            ? (input_group_end_row - chunk_start_row) : 0;

        // Issue TMA 1D stores for each valid SF tile
        // Each store moves one 512-byte SF tile from SMEM to GMEM
        if (rows_remaining_in_group > 0 && is_master_thread) {
            constexpr int SF_TILE_SIZE_BYTES = SF_ROWS * SF_COLS;  // 512 bytes
            const int sf_tiles_per_row = output_stride_per_row_of_sf_tiles / output_stride_per_sf_tile;

            // Only store valid SF tiles (not the full CHUNK_WIDTH for partial chunks)
            for (int tile = 0; tile < valid_sf_tiles; tile++) {
                const int col_sf_tile = col_sf_tile_start + tile;
                const int linear_sf_tile_idx = global_sf_row_tile * sf_tiles_per_row + col_sf_tile;

                uint64_t* dst_gmem = reinterpret_cast<uint64_t*>(
                    output_scales + linear_sf_tile_idx * SF_TILE_SIZE_BYTES);
                const uint32_t smem_sf_tile_offset = tile * SF_TILE_SIZE_BYTES;

                ptx::cp_async_bulk_tensor_1d_shared_to_global(
                    dst_gmem,
                    reinterpret_cast<const uint64_t*>(&smem_buffers[buf_idx][smem_sf_tile_offset]),
                    SF_TILE_SIZE_BYTES
                );
            }
            ptx::cp_async_bulk_commit_group();
        }
    };

    // --- Main processing loop for superblock ---
    // Compute how many chunks to process in this superblock
    constexpr int BUFFER_MOD = NUM_BUFFERS - 1;
    const int first_row_in_superblock = input_group_start_row + (superblock_idx_in_group * SUPERBLOCK_ROWS);
    const int remaining_rows_in_group = input_group_end_row - first_row_in_superblock;
    const int remaining_chunks = ceil_div(remaining_rows_in_group, SF_ROWS);
    const int chunks_to_process = min(CHUNKS_PER_TB, remaining_chunks);

    if (chunks_to_process == 0) {
        destroy_barriers<NUM_BUFFERS>(mbar, is_master_thread);
        return;
    }

    // Pipeline to overlap loads of chunk N+1 with process + store of chunk N
    for (int chunk = 0; chunk < chunks_to_process; chunk++) {
        int buf_idx = chunk & BUFFER_MOD;                   // chunk % num_buffers
        load_chunk(chunk, buf_idx);
        if (chunk > 0) {
            int prev_buf_idx = (chunk - 1) & BUFFER_MOD;    // (chunk - 1) % num_buffers
            process_chunk(chunk - 1, prev_buf_idx);
        }
    }

    // Process last chunk
    int last_buf_idx = (chunks_to_process - 1) & BUFFER_MOD;
    process_chunk(chunks_to_process - 1, last_buf_idx);

    // Wait for all pending TMA stores to complete before exiting
    ptx::cp_async_bulk_wait_group();

    // Clean up barriers
    destroy_barriers<NUM_BUFFERS>(mbar, is_master_thread);
}

// Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays
void create_tensor_map(
    void* tensor_ptr,
    CUtensorMap& tensor_map,
    const uint64_t gmem_width,
    const uint64_t gmem_height,
    const uint32_t smem_width,
    const uint32_t smem_height
) {
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {gmem_width, gmem_height};
    uint64_t stride[rank - 1] = {gmem_width}; // Row major, 1 byte per e8m0
    uint32_t box_size[rank] = {smem_width, smem_height};
    uint32_t elem_stride[rank] = {1, 1};

    void *driver_ptr = get_driver_ptr();
    auto cuTensorMapEncodeTiled = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(driver_ptr);

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,
        tensor_ptr,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    CUDA_CHECK(res);
}


namespace mxfp8 {

void launch_mx_block_rearrange_2d_M_groups_cuda(
    const uint8_t* scales_ptr,
    int scale_stride_dim0,
    int scale_rows,
    int scale_cols,
    int padded_rows,
    const int32_t* input_group_end_offsets,
    uint8_t* output_scales_ptr,
    int num_groups,
    int chunk_width,    // Chunk width: 64 or 128
    int chunks_per_tb,  // Chunks per superblock: 4, 8, or 16
    cudaStream_t stream
) {
    // Calculate grid dimensions
    int rows_per_superblock = SF_ROWS * chunks_per_tb;

    // Upper bound on superblocks: each group may add 1 due to ceiling division
    int num_row_superblocks = (scale_rows + rows_per_superblock - 1) / rows_per_superblock + num_groups;
    int num_col_chunks = (scale_cols + chunk_width - 1) / chunk_width;

    // Output strides for SF tile addressing
    int sf_tiles_per_row = (scale_cols + SF_COLS - 1) / SF_COLS;
    int output_stride_per_sf_tile = SF_ROWS * SF_COLS;  // 512 bytes
    int output_stride_per_row_of_sf_tiles = output_stride_per_sf_tile * sf_tiles_per_row;

    // Create TMA tensor map for input (chunk size = SF_ROWS × effective_chunk_width)
    // When scale_cols < chunk_width, use scale_cols as box width to avoid OOB issues
    const int effective_chunk_width = min(chunk_width, scale_cols);
    alignas(64) CUtensorMap input_tensor_map = {};
    create_tensor_map(
        (void*)scales_ptr,
        input_tensor_map,
        scale_cols,
        scale_rows,
        effective_chunk_width,
        SF_ROWS
    );

    dim3 grid(num_col_chunks, num_row_superblocks);

    if (chunk_width == 64) {
        dim3 block(512);  // 128 rows × 4 threads/row
        switch (chunks_per_tb) {
            case 1:
                mx_blocked_layout_2d_M_groups_kernel<64, 1><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr,
                    output_stride_per_row_of_sf_tiles,
                    num_groups, input_tensor_map);
                break;
            case 4:
                mx_blocked_layout_2d_M_groups_kernel<64, 4><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr,
                    output_stride_per_row_of_sf_tiles,
                    num_groups, input_tensor_map);
                break;
            case 8:
                mx_blocked_layout_2d_M_groups_kernel<64, 8><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr,
                    output_stride_per_row_of_sf_tiles,
                    num_groups, input_tensor_map);
                break;
            case 16:
                mx_blocked_layout_2d_M_groups_kernel<64, 16><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr,
                    output_stride_per_row_of_sf_tiles,
                    num_groups, input_tensor_map);
                break;
            default:
                printf("CUDA Error: chunks_per_tb must be 1, 4, 8, or 16, got %d\n", chunks_per_tb);
                return;
        }
    } else if (chunk_width == 128) {
        dim3 block(1024);  // 128 rows × 8 threads/row
        switch (chunks_per_tb) {
            case 1:
                mx_blocked_layout_2d_M_groups_kernel<128, 1><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr,
                    output_stride_per_row_of_sf_tiles,
                    num_groups, input_tensor_map);
                break;
            case 4:
                mx_blocked_layout_2d_M_groups_kernel<128, 4><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr,
                    output_stride_per_row_of_sf_tiles,
                    num_groups, input_tensor_map);
                break;
            case 8:
                mx_blocked_layout_2d_M_groups_kernel<128, 8><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr,
                    output_stride_per_row_of_sf_tiles,
                    num_groups, input_tensor_map);
                break;
            case 16:
                mx_blocked_layout_2d_M_groups_kernel<128, 16><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr,
                    output_stride_per_row_of_sf_tiles,
                    num_groups, input_tensor_map);
                break;
            default:
                printf("CUDA Error: chunks_per_tb must be 1, 4, 8, or 16, got %d\n", chunks_per_tb);
                return;
        }
    } else {
        printf("CUDA Error: chunk_width must be 64 or 128, got %d\n", chunk_width);
        return;
    }
}

} // namespace mxfp8
