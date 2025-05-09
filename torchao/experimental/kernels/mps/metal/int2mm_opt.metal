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
   @brief This shader implements 2-bit matrix-vector multiplication where A
   matrix is fp16, bfloat or float and B matrix is a 2-bit groupwise-quantized weight
   matrix.
   @param [in] A is activation matrix of size M x K.
   @param [in] B is weight matrix of size M x K. Each byte contains 4 2-bit
   values, along K dim, packed together.
   @param [in] scales_ptr is scales ptr corresponding each
   output channel x groups. These are packed as [N, num_groups = ceil(K / group_size)]. N = output
   channels.
   @param [in] zeros_ptr is zero points corresponding each
   output channel x groups. These are packed as [N, num_groups = ceil(K / group_size)]. N = output
   channels.
   @param [out] output_data is output matrix of size M x N.
   @param [in] sizes array contains values of M, K and N.
   @param [in] thread_index is global thread id.
   @param [in] tid_in_simdgruop is thread id in simdgroup. e.g. in simdgroup of size 32 it can be in [0-31].
*/
template <typename T, unsigned group_size>
kernel void int2pack_mm(constant T *A [[buffer(0)]],
                        constant uchar *B [[buffer(1)]],
                        constant T *scales_ptr [[buffer(2)]],
                        constant T *zeros_ptr [[buffer(3)]],
                        device T *output_data [[buffer(4)]],
                        constant uint3 &sizes [[buffer(5)]], // M, K, N
                        uint3 thread_index [[thread_position_in_grid]],
                        uint tid_in_simdgroup [[thread_index_in_simdgroup]]) {
  constexpr uint threads_per_channel = 32;
  constexpr uint ks_per_thread = 4;
  constexpr uint k_pack_factor = 4;
  const uint K = sizes.y;
  const uint N = sizes.z;
  const uint num_groups = (K + group_size - 1) / group_size;
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
  float4 act_div_scales = {1.f, 1 / 4.f, 1 / 16.f, 1 / 64.f};

  for (; k < K; k += k_jump) {
    // Find specific group to which channels handled by this thread
    // belong.
    uint k_block_index = k / group_size;
    uint scales_group_offset = (n * num_groups + k_block_index);

    vecT scales =
        vecT(scales_ptr[scales_group_offset],
             scales_ptr[scales_group_offset + num_groups],
             scales_ptr[scales_group_offset + 2 * num_groups],
             scales_ptr[scales_group_offset + 3 * num_groups]);
    vecT zeros =
        vecT(zeros_ptr[scales_group_offset],
             zeros_ptr[scales_group_offset + num_groups],
             zeros_ptr[scales_group_offset + 2 * num_groups],
             zeros_ptr[scales_group_offset + 3 * num_groups]);
    float4 zeros_float = float4(zeros);

    float4 a_val = float4(A_ptr[k / 4]);
    // We are gonna skip right-shifts of the weights and hence divide by corresponding factor.
    float4 a_vec = a_val * act_div_scales;
    float a_val_sum = a_val[0] + a_val[1] + a_val[2] + a_val[3];

    float4x4 b_mat;
    ushort b_val0 = (B_ptr + (k + 0 * K) / k_pack_factor)[0];
    ushort b_val1 = (B_ptr + (k + 1 * K) / k_pack_factor)[0];
    ushort b_val2 = (B_ptr + (k + 2 * K) / k_pack_factor)[0];
    ushort b_val3 = (B_ptr + (k + 3 * K) / k_pack_factor)[0];
    b_mat[0] = scales[0] * float4(float(b_val0 & 0x03), float(b_val0 & 0x0c),
                               float(b_val0 & 0x30), float(b_val0 & 0xc0));
    b_mat[1] = scales[1] * float4(float(b_val1 & 0x03), float(b_val1 & 0x0c),
                               float(b_val1 & 0x30), float(b_val1 & 0xc0));
    b_mat[2] = scales[2] * float4(float(b_val2 & 0x03), float(b_val2 & 0x0c),
                               float(b_val2 & 0x30), float(b_val2 & 0xc0));
    b_mat[3] = scales[3] * float4(float(b_val3 & 0x03), float(b_val3 & 0x0c),
                               float(b_val3 & 0x30), float(b_val3 & 0xc0));

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

#define INSTANTIATE_INT2MM(DTYPE, GSIZE)                                       \
  template [[host_name("int2pack_mm_" #GSIZE "_" #DTYPE)]] kernel void         \
  int2pack_mm<DTYPE, GSIZE>(                                                   \
      constant DTYPE * A [[buffer(0)]], constant uchar * B [[buffer(1)]],      \
      constant DTYPE * scales_ptr [[buffer(2)]],                               \
      constant DTYPE * zeros_ptr [[buffer(3)]],                                \
      device DTYPE * output_data [[buffer(4)]],                                \
      constant uint3 & sizes [[buffer(5)]],                                    \
      uint3 thread_index [[thread_position_in_grid]],                          \
      uint tid_in_simdgroup [[thread_index_in_simdgroup]])

INSTANTIATE_INT2MM(float, 32);
INSTANTIATE_INT2MM(half, 32);
INSTANTIATE_INT2MM(float, 64);
INSTANTIATE_INT2MM(half, 64);
INSTANTIATE_INT2MM(float, 128);
INSTANTIATE_INT2MM(half, 128);
INSTANTIATE_INT2MM(float, 256);
INSTANTIATE_INT2MM(half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT2MM(bfloat, 32);
INSTANTIATE_INT2MM(bfloat, 64);
INSTANTIATE_INT2MM(bfloat, 128);
INSTANTIATE_INT2MM(bfloat, 256);
#endif
