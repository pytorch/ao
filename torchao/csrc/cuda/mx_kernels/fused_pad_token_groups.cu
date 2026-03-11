#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ int find_group_id(
    int row_idx,
    const int32_t* __restrict__ group_end_offsets,
    int num_groups
) {
    // linear prefix sum lookup (same or faster than binary search for small num groups, which ours are <= 32)
    int group_id = 0;
    while (group_id < num_groups - 1 && row_idx >= group_end_offsets[group_id]) {
        ++group_id;
    }
    return group_id;
}

__global__ void fused_pad_token_groups_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int32_t* __restrict__ group_end_offsets,
    const int32_t* __restrict__ padded_group_start_offsets,
    __nv_bfloat16* __restrict__ output,
    int num_tokens,
    int dim,
    int num_groups
) {
    constexpr int WARP_SIZE = 32;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE; // each warp handles 1 row
    const int global_row_idx = blockIdx.x * warps_per_block + warp_id_in_block;

    if (global_row_idx >= num_tokens) return;

    // one thread finds which group this row belongs to, then broadcasts to other lanes via warp shuffle
    int group_id;
    if (lane_id == 0) {
        group_id = find_group_id(global_row_idx, group_end_offsets, num_groups);
    }
    group_id = __shfl_sync(0xffffffff, group_id, 0);

    int group_start = (group_id > 0) ? group_end_offsets[group_id - 1] : 0;
    int padded_start = padded_group_start_offsets[group_id];
    int offset_in_group = global_row_idx - group_start;
    int output_row = padded_start + offset_in_group;

    const __nv_bfloat16* row_in = input + global_row_idx * dim;
    __nv_bfloat16* row_out = output + output_row * dim;

    // vectorized copy for all complete float4 chunks (8 bf16 elements each)
    int num_vecs = dim / 8;
    if (num_vecs > 0) {
        const float4* in_vec = reinterpret_cast<const float4*>(row_in);
        float4* out_vec = reinterpret_cast<float4*>(row_out);

        for (int i = lane_id; i < num_vecs; i += 32) {
            out_vec[i] = in_vec[i];
        }
    }

    // scalar copy for remaining elems
    int elems_covered = num_vecs * 8;
    for (int i = elems_covered + lane_id; i < dim; i += 32) {
        row_out[i] = row_in[i];
    }
}

__global__ void fused_unpad_token_groups_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int32_t* __restrict__ group_end_offsets,
    const int32_t* __restrict__ padded_group_start_offsets,
    __nv_bfloat16* __restrict__ output,
    int num_tokens,
    int dim,
    int num_groups
) {
    constexpr int WARP_SIZE = 32;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE; // each warp handles 1 row
    const int global_row_idx = blockIdx.x * warps_per_block + warp_id_in_block;

    if (global_row_idx >= num_tokens) return;

    // one thread finds which group this row belongs to, then broadcasts to other lanes via warp shuffle
    int group_id;
    if (lane_id == 0) {
        group_id = find_group_id(global_row_idx, group_end_offsets, num_groups);
    }
    group_id = __shfl_sync(0xffffffff, group_id, 0);

    int group_start = (group_id > 0) ? group_end_offsets[group_id - 1] : 0;
    int padded_start = padded_group_start_offsets[group_id];
    int offset_in_group = global_row_idx - group_start;
    int input_row = padded_start + offset_in_group;

    const __nv_bfloat16* row_in = input + input_row * dim;
    __nv_bfloat16* row_out = output + global_row_idx * dim;

    // vectorized copy for all complete float4 chunks (8 bf16 elements each)
    int num_vecs = dim / 8;
    if (num_vecs > 0) {
        const float4* in_vec = reinterpret_cast<const float4*>(row_in);
        float4* out_vec = reinterpret_cast<float4*>(row_out);

        for (int i = lane_id; i < num_vecs; i += 32) {
            out_vec[i] = in_vec[i];
        }
    }

    // scalar copy for remaining elems
    int elems_covered = num_vecs * 8;
    for (int i = elems_covered + lane_id; i < dim; i += 32) {
        row_out[i] = row_in[i];
    }
}

__global__ void compute_padded_offsets_kernel(
    const int32_t* __restrict__ group_end_offsets,
    int32_t* __restrict__ padded_group_start_offsets,
    int32_t* __restrict__ padded_group_end_offsets,
    int num_groups,
    int alignment_size
) {
    if (threadIdx.x == 0) {
        int cumulative = 0;

        for (int g = 0; g < num_groups; g++) {
            int group_start = (g > 0) ? group_end_offsets[g - 1] : 0;
            int group_end = group_end_offsets[g];
            int group_size = group_end - group_start;

            int padded_size = ((group_size + alignment_size - 1) / alignment_size) * alignment_size;

            padded_group_start_offsets[g] = cumulative;
            cumulative += padded_size;
            padded_group_end_offsets[g] = cumulative;
        }
    }
}

namespace mxfp8 {

void launch_compute_padded_offsets_cuda(
    const int32_t* group_end_offsets_ptr,
    int32_t* padded_group_start_offsets_ptr,
    int32_t* padded_group_end_offsets_ptr,
    int num_groups,
    int alignment_size,
    cudaStream_t stream
) {
    // Single warp is enough for num_groups <= 32
    compute_padded_offsets_kernel<<<1, 32, 0, stream>>>(
        group_end_offsets_ptr,
        padded_group_start_offsets_ptr,
        padded_group_end_offsets_ptr,
        num_groups,
        alignment_size
    );

    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_pad_token_groups_cuda(
    const void* input_ptr,
    const int32_t* group_end_offsets_ptr,
    const int32_t* padded_group_start_offsets_ptr,
    void* output_ptr,
    int num_tokens,
    int dim,
    int num_groups,
    int dtype_size,
    int dtype_enum,
    cudaStream_t stream
) {
    const int WARPS_PER_BLOCK = 8;
    const int THREADS_PER_WARP = 32;
    const int THREADS_PER_BLOCK = WARPS_PER_BLOCK * THREADS_PER_WARP;  // 256 threads

    const int num_blocks = ceil_div(num_tokens, WARPS_PER_BLOCK);

    fused_pad_token_groups_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input_ptr),
        group_end_offsets_ptr,
        padded_group_start_offsets_ptr,
        reinterpret_cast<__nv_bfloat16*>(output_ptr),
        num_tokens,
        dim,
        num_groups
    );

    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_unpad_token_groups_cuda(
    const void* input_ptr,
    const int32_t* group_end_offsets_ptr,
    const int32_t* padded_group_start_offsets_ptr,
    void* output_ptr,
    int num_tokens,
    int dim,
    int num_groups,
    int dtype_size,
    int dtype_enum,
    cudaStream_t stream
) {
    const int WARPS_PER_BLOCK = 8;
    const int THREADS_PER_WARP = 32;
    const int THREADS_PER_BLOCK = WARPS_PER_BLOCK * THREADS_PER_WARP;  // 256 threads

    const int num_blocks = ceil_div(num_tokens, WARPS_PER_BLOCK);

    fused_unpad_token_groups_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input_ptr),
        group_end_offsets_ptr,
        padded_group_start_offsets_ptr,
        reinterpret_cast<__nv_bfloat16*>(output_ptr),
        num_tokens,
        dim,
        num_groups
    );

    CUDA_CHECK(cudaGetLastError());
}

} // namespace mxfp8