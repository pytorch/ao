// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

/*
   This code takes heavy inspiration from MLX:
   https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h
   Specifically:
     - Multiplying activation by inverse scaling factor to reduce compute
   boundedness
     - Handling zero point by accumulating act in separate sum term. Needed with
   optimization done above. MLX MIT License:
   https://github.com/ml-explore/mlx/blob/main/LICENSE
*/

/*
   A matrix is [M x K] (right now this kernel does not support M > 1 but this is
   a very easy fix that will follow right after) B matrix is [N x K]. For 4 bit
   2 of the k values are packed in one byte so you can think of B as [N x K/2]
   matrix from layout perspective.

   Since this kernel is optimizing for gemv case, we split work, along reduction
   dim k, among the threads of same simdgroup. Ex: if K = 4096 and simdgroup
   size is 32 (current algorithm should work as long as simdgroup size is > 32).
   Then each thread will accumulate 4096/32 = 128 k values. However these 128
   values, handled by each thread are not laid out contiguously. Each thread
   handles 4 contiguous k values and then jumps 128 elements, k_jump =
   thread_per_channel (32) * ks_per_thread (4). Take a simpler example where
   simdgroup is of size 4. In this case threads_per_channel = 4. Assume K = 32
      k                thread
   [0, 1, 2, 3,          0
    4, 5, 6, 7,          1
    8, 9, 10, 11,        2
    12, 13, 14, 15,      3
    16, 17, 18, 19,      0
    20, 21, 22, 23,      1
    24, 25, 26, 27,      2
    28, 29, 30, 31]      3
   thread id in simd group that handle corresponding
   ks
   Thread 0 here is handling (0, 1, 2, 3) and then (16, 17, 18, 19). They are
   apart by k_jump = 4 * 4 = 16 This is done to improve memory access locality
   amonng threads that are working co-operatively. Once each thread has their
   partial sums accumulated, we use tree reduction (Metal offers simd_sum but
   not used so that we support simdgroup size = 64). In the
   example above we will have 4 partial sums.

   Each thread also handles 4 different output rows. Thus each simdgroup will be
   responsible for (1x4) tile of the output. We haven't evaluated whether a
   different tile size is better or not. We probably will do some auto-tuning
   once initial work is done.
*/

/*
   @brief This shader implements 4-bit matrix-vector multiplication where A
   matrix is fp16, bfloat or float and B matrix is a 4-bit groupwise-quantized weight
   matrix.
   @param [in] A is activation matrix of size M x K.
   @param [in] B is weight matrix of size M x K. Each byte contains 2 4-bit
   values, along K dim, packed together.
   @param [in] scales_ptr is scales ptr corresponding each
   output channel x groups. These are packed as [num_groups = ceil(K / group_size), N]. N = output
   channels.
   @param [in] zeros_ptr is zero points corresponding each
   output channel x groups. These are packed as [num_groups = ceil(K / group_size), N]. N = output
   channels.
   output channel x groups. These are packed as [num_groups = ceil(K / group_size), N, 2]. N = output
   @param [out] output_data is output matrix of size M x N.
   @param [in] sizes array contains values of M, K and N.
   @param [in] thread_index is global thread id.
   @param [in] tid_in_simdgruop is thread id in simdgroup. e.g. in simdgroup of size 32 it can be in [0-31].
*/
template <typename T, unsigned group_size>
kernel void int4pack_mm(constant T *A [[buffer(0)]],
                        constant uchar *B [[buffer(1)]],
                        constant T *scales_ptr [[buffer(2)]],
                        constant T *zeros_ptr [[buffer(3)]],
                        device T *output_data [[buffer(4)]],
                        constant uint3 &sizes [[buffer(5)]], // M, K, N
                        uint3 thread_index [[thread_position_in_grid]],
                        uint tid_in_simdgroup [[thread_index_in_simdgroup]]) {
  constexpr uint threads_per_channel = 32;
  constexpr uint ks_per_thread = 4;
  constexpr uint k_pack_factor = 2;
  const uint K = sizes.y;
  const uint N = sizes.z;
  uint n = thread_index.x; // 0..N/4-1
  uint m = thread_index.z; // 0..M
  n = n / threads_per_channel;
  n = n * 4;
  // This is starting k for each thread. In the example above, for thread 1 this
  // value will be 4.
  uint k = (tid_in_simdgroup % threads_per_channel) * ks_per_thread;
  constexpr int k_jump = threads_per_channel * ks_per_thread;

  using vecT = typename Vec4Type<T>::type;
  constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A + m * K);
  constant uchar *B_ptr = B + ((n * K) / k_pack_factor);

  thread float4 result = float4(0.0);
  // We multipy group of 4 channels with these scales.
  // Because corresponding values from weight matrix are effectively left
  // shifted. This is to avoid doing right shift on those values which ends up
  // affecting performance. This is the trick applied in MLX kernels.
  float4 act_div_scales = {1.f, 1 / 16.f, 1 / 256.f, 1 / 4096.f};

  for (; k < K; k += k_jump) {
    // Find specific group to which channels handled by this thread
    // belong.
    uint k_block_index = k / group_size;
    uint scales_group_offset = (k_block_index * N + n);

    vecT scales =
        (reinterpret_cast<constant vecT *>(scales_ptr + scales_group_offset))[0];
    // Adding zero point results in 10% perf penalty.
    vecT zeros =
        (reinterpret_cast<constant vecT *>(zeros_ptr + scales_group_offset))[0];
    float4 zeros_float = float4(zeros);

    float4 a_val = float4(A_ptr[k / 4]);
    // We are gonna skip right-shifts of the weights and hence divide by corresponding factor.
    float4 a_vec = a_val * act_div_scales;
    float a_val_sum = a_val[0] + a_val[1] + a_val[2] + a_val[3];

    float4x4 b_mat;
    ushort b_val0 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 0 * K) / k_pack_factor))[0];
    ushort b_val1 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 1 * K) / k_pack_factor))[0];
    ushort b_val2 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 2 * K) / k_pack_factor))[0];
    ushort b_val3 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 3 * K) / k_pack_factor))[0];
    b_mat[0] = scales[0] * float4(float(b_val0 & 0x000f), float(b_val0 & 0x00f0),
                               float(b_val0 & 0x0f00), float(b_val0 & 0xf000));
    b_mat[1] = scales[1] * float4(float(b_val1 & 0x000f), float(b_val1 & 0x00f0),
                               float(b_val1 & 0x0f00), float(b_val1 & 0xf000));
    b_mat[2] = scales[2] * float4(float(b_val2 & 0x000f), float(b_val2 & 0x00f0),
                               float(b_val2 & 0x0f00), float(b_val2 & 0xf000));
    b_mat[3] = scales[3] * float4(float(b_val3 & 0x000f), float(b_val3 & 0x00f0),
                               float(b_val3 & 0x0f00), float(b_val3 & 0xf000));

    result += a_vec * b_mat;
    result += a_val_sum * zeros_float;
  }
  result += simd_shuffle_down(result, 1);
  result += simd_shuffle_down(result, 2);
  result += simd_shuffle_down(result, 4);
  result += simd_shuffle_down(result, 8);
  result += simd_shuffle_down(result, 16);
  if (tid_in_simdgroup % threads_per_channel == 0) {
    reinterpret_cast<device vecT *>(output_data + m * N)[n / 4] = vecT(result);
  }
}

#define INSTANTIATE_INT4MM(DTYPE, GSIZE)                                       \
  template [[host_name("int4pack_mm_" #GSIZE "_" #DTYPE)]] kernel void         \
  int4pack_mm<DTYPE, GSIZE>(                                                   \
      constant DTYPE * A [[buffer(0)]], constant uchar * B [[buffer(1)]],      \
      constant DTYPE * scales_ptr [[buffer(2)]],                               \
      constant DTYPE * zeros_ptr [[buffer(3)]],                                \
      device DTYPE * output_data [[buffer(4)]],                                \
      constant uint3 & sizes [[buffer(5)]],                                    \
      uint3 thread_index [[thread_position_in_grid]],                          \
      uint tid_in_simdgroup [[thread_index_in_simdgroup]])

INSTANTIATE_INT4MM(float, 32);
INSTANTIATE_INT4MM(half, 32);
INSTANTIATE_INT4MM(float, 64);
INSTANTIATE_INT4MM(half, 64);
INSTANTIATE_INT4MM(float, 128);
INSTANTIATE_INT4MM(half, 128);
INSTANTIATE_INT4MM(float, 256);
INSTANTIATE_INT4MM(half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT4MM(bfloat, 32);
INSTANTIATE_INT4MM(bfloat, 64);
INSTANTIATE_INT4MM(bfloat, 128);
INSTANTIATE_INT4MM(bfloat, 256);
#endif
