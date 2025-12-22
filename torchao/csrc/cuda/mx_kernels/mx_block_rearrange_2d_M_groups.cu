#include <cstdint>
#include <cstdio>

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include "ptx.cuh"

#define CUDA_CHECK(result)                                                     \
  do {                                                                         \
    CUresult status = result;                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char* error_str;                                                   \
      cuGetErrorString(status, &error_str);                                    \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              error_str);                                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

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
__global__ void mx_blocked_layout_2d_M_groups_kernel(
    const uint8_t* __restrict__ input_scales,
    const int input_scale_stride_dim0,
    const int32_t* __restrict__ input_group_end_offsets,
    const uint8_t* __restrict__ output_scales,
    const int num_groups,
    const __grid_constant__ CUtensorMap* __restrict__ tensor_map
) {
    const int row_block_pid = blockIdx.y;
    const int col_block_pid = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_master_thread = tid == 0;
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
    int global_col_base = col_block_pid * MAX_COLS;
    const uint8_t* base_ptr = input_scales + static_cast<size_t>(global_row_base) * input_scale_stride_dim0 + static_cast<size_t>(global_col_base);

    // Blocked layout base offset
    int r_div_32 = row_idx << 5;
    int r_mod_32 = row_idx & 31;
    int blocked_layout_base = (r_mod_32 << 4) + (r_div_32 << 2);


    __shared__ __align__(8) uint64_t mbar[NUM_BUFFERS];
    initialize_barriers<NUM_BUFFERS, BLOCK_ROWS>(mbar, is_master_thread);
    int parity = 0;

    auto load_chunk = [&](int chunk_idx, int buf_idx) {
        if (is_master_thread) {
            uint64_t* mbar_ptr = &mbar[buf_idx];
            constexpr uint32_t tile_bytes = BLOCK_ROWS * MAX_COLS; // 1 byte per uint8

            // TMA async load chunk
            int col_offset = (col_block_pid * MAX_COLS) + (chunk_idx * MAX_COLS);
            int row_offset = global_row_base;
            void* chunk_ptr = (void*)(base_ptr + chunk_idx * MAX_COLS);
            copy_2d_to_shared(
                &smem_buffers[buf_idx],
                chunk_ptr,
                MAX_COLS,
                BLOCK_ROWS,
                tile_bytes,
                mbar_ptr,
                is_master_thread
            );
        }
    };

    // Chunk process func
    auto process_chunk = [&](int chunk_idx, int buffer_idx) {

    };

    // Pipeline to overlap loads of chunk N+1 with process + store of chunk N
}

// Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays
CUtensorMap create_tensor_map(
    const void* tensor_ptr,
    const uint64_t gmem_width,
    const uint64_t gmem_height,
    const uint32_t smem_width,
    const uint32_t smem_height,
) {
  CUtensorMap tensor_map{};

  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {gmem_width, gmem_height};
  uint64_t stride[rank - 1] = {gmem_width}; // 1 byte per e8m0
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
  return tensor_map;
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
    int num_row_blocks = (scale_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int output_stride_per_block = BLOCK_ROWS * BLOCK_COLS;

    int total_chunks = (scale_cols + max_cols - 1) / max_cols + num_groups;
    int total_super_col_blocks = (total_chunks + chunks_per_tb - 1) / chunks_per_tb + num_groups;

    alignas(64) CUtensorMap tensor_map = CUtensorMap{};

    dim3 grid(total_super_col_blocks, num_row_blocks);

    if (max_cols == 64) {
        dim3 block(512);
        switch (chunks_per_tb) {
            case 4:
                mx_blocked_layout_2d_M_groups_kernel<64, 4><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            case 8:
                mx_blocked_layout_2d_M_groups_kernel<64, 8><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            case 16:
                mx_blocked_layout_2d_M_groups_kernel<64, 16><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
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
                mx_blocked_layout_2d_M_groups_kernel<128, 4><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            case 8:
                mx_blocked_layout_2d_M_groups_kernel<128, 8><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
                    input_group_end_offsets, output_scales_ptr, output_stride_per_block, num_groups);
                break;
            case 16:
                mx_blocked_layout_2d_M_groups_kernel<128, 16><<<grid, block, 0, stream>>>(
                    scales_ptr, scale_stride_dim0, scale_rows, scale_cols, padded_rows,
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
