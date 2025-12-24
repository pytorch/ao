#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include "ptx.cuh"

// Overloaded error checking for both CUDA driver API (CUresult) and runtime API (cudaError_t)
inline void cuda_check_impl(CUresult result, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        fprintf(stderr, "CUDA Driver Error at %s:%d - %s\n", file, line, error_str);
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
#define BYTES_PER_THREAD 16
#define NUM_BUFFERS 2


// Here the chunks are 128x64/128x128 and superblock contains multiple vertical chunks (e.g., 4 chunks of 128x64 would be 512x64).
template <int CHUNKS_PER_TB>
__device__ void find_group_and_local_offset_for_superblock(
    int row_block_pid,
    const int32_t* __restrict__ input_group_end_offsets,
    int num_groups,
    int* __restrict__ smem_data,  // smem_data[0..num_groups-1] = chunks_in_group, smem_data[num_groups..2*num_groups-1] = super_blocks_in_group cumsum
    int& group_id,
    int& first_superblock_in_group,
    int& superblocks_until_group_end
) {
    // 1 thread computes how many superblocks span this group (along the row dim)
    if (threadIdx.x == 0) {
        constexpr int SUPERBLOCK_ROWS = SF_ROWS * CHUNKS_PER_TB;
        int block_cumsum = 0;
        for (int g = 0; g < num_groups; g++) {
            int input_group_start = (g > 0) ? input_group_end_offsets[g - 1] : 0;
            int input_group_end = input_group_end_offsets[g];
            int group_size = input_group_end - input_group_start;
            int num_superblocks_rowwise = ceil_div(group_size, SUPERBLOCK_ROWS);
            smem_data[g] = num_superblocks_rowwise;
            block_cumsum += num_superblocks_rowwise;
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
            int group_local_row_superblock_idx = row_block_pid - block_cumsum_before;
            first_superblock_in_group = group_local_row_superblock_idx;
            int chunks_in_group = smem_data[g];
            superblocks_until_group_end = chunks_in_group - first_superblock_in_group;
            return;
        }
        block_cumsum_before = cumsum_at_g;
    }

    first_superblock_in_group = 0;
    superblocks_until_group_end = 0;
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
__global__ void mx_blocked_layout_2d_M_groups_kernel(
    const uint8_t* __restrict__ input_scales,
    const int input_scale_stride_dim0,
    const int scale_rows,
    const int scale_cols,
    const int padded_rows,
    const int32_t* __restrict__ input_group_end_offsets,
    uint8_t* __restrict__ output_scales,
    const int output_stride_per_block,
    const int output_stride_per_row_of_blocks,
    const int num_groups,
    const __grid_constant__ CUtensorMap input_tensor_map
) {
    const int row_block_pid = blockIdx.y;
    const int col_block_pid = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_master_thread = tid == 0;
    const int row_idx = tid / SF_COLS;
    const int col_idx = tid % SF_COLS;
    constexpr int SMEM_SIZE = SF_ROWS * MAX_COLS;

    __shared__ int smem_group_data[64]; // max 32 groups; 32 ints for group sizes, 32 for cumsums at each group
    __shared__ __align__(128) uint8_t smem_buffers[NUM_BUFFERS][SMEM_SIZE]; // 128 byte alignment for TMA

    // Find:
    // 1. The group idx this block belongs to.
    // 2. The local row block idx within the group.
    // 3. The number of row blocks remaining until the end of the group.
    int group_idx;
    int group_local_row_superblock;
    int row_blocks_until_end;
    find_group_and_local_offset_for_superblock<CHUNKS_PER_TB>(
        // inputs
        row_block_pid,
        input_group_end_offsets,
        num_groups,
        smem_group_data,
        // outputs
        group_idx,
        group_local_row_superblock,
        row_blocks_until_end
    );

    // Early exit for padding blocks that are beyond all groups
    if (row_blocks_until_end <= 0) {
        return;
    }


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
        s_output_group_start_row = compute_output_group_start_row(group_idx, input_group_end_offsets, num_groups, SF_ROWS);
    }
    __syncthreads();
    uint input_group_start_row = s_input_group_start_row;
    uint input_group_end_row = s_input_group_end_row;
    uint output_group_start_row = s_output_group_start_row;

    // Prepare for pipeline phase, compute thread constant vals
    constexpr int SUPERBLOCK_ROWS = SF_ROWS * CHUNKS_PER_TB;
    const uint global_row_base = input_group_start_row + (group_local_row_superblock * SUPERBLOCK_ROWS);
    const uint global_col_base = col_block_pid * MAX_COLS;
    const uint thread_col_start = col_idx * BYTES_PER_THREAD;
    const uint first_tile_idx = thread_col_start / SF_COLS;

    // Initialize barriers for TMA load coordination
    constexpr int NUM_THREADS = SF_ROWS * (MAX_COLS / BYTES_PER_THREAD);
    __shared__ __align__(8) uint64_t mbar[NUM_BUFFERS];
    initialize_barriers<NUM_BUFFERS, NUM_THREADS>(mbar, is_master_thread);

    // Track barrier parity per buffer for double-buffered pipeline
    int buf_parity[NUM_BUFFERS] = {0};
    auto load_chunk = [&](int chunk_idx, int buf_idx) {
        if (chunk_idx >= NUM_BUFFERS) {
            // Wait for pending TMA loads to complete (at most 1 pending)
            ptx::cp_async_bulk_wait_group_read<1>();
            // Wait for pending TMA stores to complete before reusing buffer
            // (the TMA store from process_chunk reads from SMEM asynchronously)
            ptx::cp_async_bulk_wait_group();
            __syncthreads();
        }

        uint64_t* mbar_ptr = &mbar[buf_idx];
        constexpr uint32_t chunk_bytes = SF_ROWS * MAX_COLS;
        const uint32_t tma_x = global_col_base;
        const uint32_t tma_y = global_row_base + (chunk_idx * SF_ROWS);

        copy_2d_to_shared(
            (void*)&smem_buffers[buf_idx],
            reinterpret_cast<const void*>(&input_tensor_map),
            tma_x,
            tma_y,
            chunk_bytes,
            mbar_ptr,
            is_master_thread
        );
    };

    // Blocked layout base offset for the row
    int r_div_32 = row_idx >> 5;
    int r_mod_32 = row_idx & 31;
    int blocked_layout_base = (r_mod_32 << 4) + (r_div_32 << 2);

    auto process_chunk = [&](int chunk_idx, int buf_idx) {
        uint64_t* mbar_ptr = &mbar[buf_idx];
        ptx::mbarrier_wait_parity(mbar_ptr, buf_parity[buf_idx]);
        buf_parity[buf_idx] ^= 1;
        // Fence to ensure TMA-written data is visible to all threads after barrier wait
        ptx::fence_proxy_async_shared_cta();

        // Read 16 bytes via coalesced vectorized loads from SMEM (row-major input)
        uint4 row_data = make_uint4(0, 0, 0, 0);
        const uint32_t chunk_start_row = global_row_base + (chunk_idx * SF_ROWS);
        if (chunk_start_row + row_idx < input_group_end_row) {
            row_data = *reinterpret_cast<uint4*>(&smem_buffers[buf_idx][row_idx * MAX_COLS + thread_col_start]);
        }
        __syncthreads();

        // Scatter to blocked layout in SMEM (4-byte writes per tile)
        constexpr int tiles_per_thread = BYTES_PER_THREAD / SF_COLS;
        #pragma unroll
        for (int i = 0; i < tiles_per_thread; i++) {
            const int tile_idx = first_tile_idx + i;
            const uint32_t data = reinterpret_cast<const uint32_t*>(&row_data)[i];
            *reinterpret_cast<uint32_t*>(&smem_buffers[buf_idx][blocked_layout_base + tile_idx * output_stride_per_block]) = data;
        }
        __syncthreads();  // Ensure all threads finished writing blocked layout before TMA stores

        // Fence to ensure SMEM writes are visible to TMA async proxy before issuing stores
        ptx::fence_proxy_async_shared_cta();

        // Store to global memory using async TMA 1D stores with 128x4 tile granularity
        // Each tile is 512 contiguous bytes in blocked layout
        // Output memory layout: [tile_row][tile_col][128][4]
        // Linear tile offset = (row_tile * tiles_per_row + col_tile) * 512

        const int sf_row_tile_within_group = (group_local_row_superblock * CHUNKS_PER_TB) + chunk_idx;
        const int sf_row_tile_global = (output_group_start_row / SF_ROWS) + sf_row_tile_within_group;

        // col_block_pid is the column SUPERBLOCK index (MAX_COLS columns each)
        // Need to convert to tile offset: superblock * tiles_per_superblock
        constexpr int TILES_PER_SUPERBLOCK = MAX_COLS / SF_COLS;  // 16 for MAX_COLS=64, 32 for MAX_COLS=128
        const int col_sf_tile_start = col_block_pid * TILES_PER_SUPERBLOCK;

        // Compute tiles_per_row from output_stride_per_row_of_blocks
        // output_stride_per_row_of_blocks = tiles_per_row * SF_ROWS * SF_COLS
        const int tiles_per_row = output_stride_per_row_of_blocks / output_stride_per_block;

        // Check if this chunk has any valid rows
        const uint32_t rows_remaining_in_group = (input_group_end_row > chunk_start_row) ? (input_group_end_row - chunk_start_row) : 0;

        // Issue multiple async TMA 1D stores (one per 128x4 SF tile in the buffer)
        // Each TMA store moves one 512-byte tile from SMEM to GMEM
        // Master thread issues all TMA stores, other threads wait
        if (rows_remaining_in_group > 0 && is_master_thread) {
            constexpr int TILE_SIZE = SF_ROWS * SF_COLS;  // 512 bytes per tile
            #pragma unroll
            for (int tile = 0; tile < TILES_PER_SUPERBLOCK; tile++) {
                // Linear tile index = row_tile * tiles_per_row + col_tile
                const int col_tile = col_sf_tile_start + tile;
                const int linear_tile_index = sf_row_tile_global * tiles_per_row + col_tile;

                // Compute destination address in GMEM
                uint64_t* dst_gmem = reinterpret_cast<uint64_t*>(output_scales + linear_tile_index * TILE_SIZE);
                const uint32_t smem_tile_offset = tile * TILE_SIZE;

                ptx::cp_async_bulk_tensor_1d_shared_to_global(
                    dst_gmem,
                    reinterpret_cast<const uint64_t*>(&smem_buffers[buf_idx][smem_tile_offset]),
                    TILE_SIZE
                );
            }
            ptx::cp_async_bulk_commit_group();
        }
    };

    constexpr int BUFFER_MOD = NUM_BUFFERS - 1;
    const int first_row_in_superblock = input_group_start_row + (group_local_row_superblock * SUPERBLOCK_ROWS);
    const int remaining_rows_in_group = input_group_end_row - first_row_in_superblock;
    const int remaining_chunks = ceil_div(remaining_rows_in_group, SF_ROWS);
    const int chunks_to_process = min(CHUNKS_PER_TB, remaining_chunks);
    if (chunks_to_process == 0)
    {
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
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {gmem_width, gmem_height};
  uint64_t stride[rank - 1] = {gmem_width}; // row major, 1 byte per e8m0
  uint32_t box_size[rank] = {smem_width, smem_height};

  // The distance between elements in units of sizeof(element)
  uint32_t elem_stride[rank] = {1, 1};

  // Get function pointer to cuTensorMapEncodeTiled
  void *driver_ptr = get_driver_ptr();
  auto cuTensorMapEncodeTiled = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(driver_ptr);


  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    rank,                       // cuuint32_t tensorRank,
    tensor_ptr,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  CUDA_CHECK(res);
}


namespace mxfp8 {

void launch_mx_block_rearrange_2d_M_groups_rowmajor_128x4_vec_pipelined(
    const uint8_t* scales_ptr,
    int scale_stride_dim0,
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
    // Calculate total superblocks by summing per-group superblocks (matches kernel logic)
    int rows_per_superblock = SF_ROWS * chunks_per_tb;

    // For now, use a conservative upper bound that matches the kernel's calculation
    // The kernel calculates ceil_div(group_size, rows_per_superblock) for each group
    // In the worst case, each group adds 1 extra superblock due to ceiling division
    // So the upper bound is ceil_div(scale_rows, rows_per_superblock) + num_groups
    int num_row_superblocks = (scale_rows + rows_per_superblock - 1) / rows_per_superblock + num_groups;
    int num_col_superblocks = (scale_cols + max_cols - 1) / max_cols;

    int sf_tiles_per_row = (scale_cols + SF_COLS - 1) / SF_COLS;
    int output_stride_per_sf_tile = SF_ROWS * SF_COLS;
    int output_stride_per_row_of_sf_tiles = output_stride_per_sf_tile * sf_tiles_per_row;

    // Create input tensor map for TMA loads of 128x64 / 128x128 chunks
    alignas(64) CUtensorMap input_tensor_map = {};
    create_tensor_map(
        (void*)scales_ptr,
        input_tensor_map,
        scale_cols,
        scale_rows,
        max_cols,
        SF_ROWS
    );

    dim3 grid(num_col_superblocks, num_row_superblocks);

    if (max_cols == 64) {
        dim3 block(512);
        switch (chunks_per_tb) {
            case 4:
                mx_blocked_layout_2d_M_groups_kernel<64, 4><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows, input_group_end_offsets,
                    output_scales_ptr, output_stride_per_sf_tile,
                    output_stride_per_row_of_sf_tiles, num_groups,
                    input_tensor_map);
                break;
            case 8:
                mx_blocked_layout_2d_M_groups_kernel<64, 8><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows, input_group_end_offsets,
                    output_scales_ptr, output_stride_per_sf_tile,
                    output_stride_per_row_of_sf_tiles, num_groups,
                    input_tensor_map);
                break;
            case 16:
                mx_blocked_layout_2d_M_groups_kernel<64, 16><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows, input_group_end_offsets,
                    output_scales_ptr, output_stride_per_sf_tile,
                    output_stride_per_row_of_sf_tiles, num_groups,
                    input_tensor_map);
                break;
            default:
                printf("CUDA Error: chunks_per_tb must be 4, 8, or 16, got %d\n", chunks_per_tb);
                return;
        }
    } else if (max_cols == 128) {
        dim3 block(1024);
        switch (chunks_per_tb) {
            case 4:
                mx_blocked_layout_2d_M_groups_kernel<128, 4><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows, input_group_end_offsets,
                    output_scales_ptr, output_stride_per_sf_tile,
                    output_stride_per_row_of_sf_tiles, num_groups,
                    input_tensor_map);
                break;
            case 8:
                mx_blocked_layout_2d_M_groups_kernel<128, 8><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows, input_group_end_offsets,
                    output_scales_ptr, output_stride_per_sf_tile,
                    output_stride_per_row_of_sf_tiles, num_groups,
                    input_tensor_map);
                break;
            case 16:
                mx_blocked_layout_2d_M_groups_kernel<128, 16><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows, input_group_end_offsets,
                    output_scales_ptr, output_stride_per_sf_tile,
                    output_stride_per_row_of_sf_tiles, num_groups,
                    input_tensor_map);
                break;
            default:
                printf("CUDA Error: chunks_per_tb must be 4, 8, or 16, got %d\n", chunks_per_tb);
                return;
        }
    } else {
        printf("CUDA Error: max_cols must be 64 or 128, got %d\n", max_cols);
        return;
    }

}
} // namespace mxfp8
