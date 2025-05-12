// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

inline void unpack_3bit(const uchar3 b, thread float* w) {
  w[0] = float(b[0] & 0x07);
  w[1] = float((b[0] & 0x38) >> 3);
  w[2] = float(((b[0] & 0xc0) >> 6) | ((b[1] & 0x01) << 2));
  w[3] = float((b[1] & 0x0e) >> 1);
  w[4] = float((b[1] & 0x70) >> 4);
  w[5] = float(((b[1] & 0x80) >> 7) | ((b[2] & 0x03) << 1));
  w[6] = float((b[2] & 0x1c) >> 2);
  w[7] = float((b[2] & 0xe0) >> 5);
}

/**
 * 3-Bit Quantized Linear.
 *
 * @param[A] M x K input tensor of floating point dtype (Float, Half, BFloat16)
 * @param[B] Packed & quantized weight tensor of uint8 dtype. Expected shape is N x (3 * K / 8)
 * @param[scales] 2D tensor containg the scales for each group. Expected shape is N x #groups
 * @param[zeros] 2D tensor containg the zero points for each group. Expected shape is N x #groups
 * @param[outputData] M x N output tensor of floating point dtype (same as input)
 * @param[sizes] The sizes involved in the order: M, K, N
 *
 */
template <typename T, unsigned group_size>
kernel void int3pack_mm(constant T *A [[buffer(0)]],
                        constant uchar *B [[buffer(1)]],
                        constant T *scales_ptr [[buffer(2)]],
                        constant T *zeros_ptr [[buffer(3)]],
                        device T *output_data [[buffer(4)]],
                        constant uint3 &sizes [[buffer(5)]], // M, K, N
                        uint3 thread_index [[thread_position_in_grid]],
                        uint tid_in_simdgroup [[thread_index_in_simdgroup]]) {
  constexpr uint threads_per_channel = 32;
  constexpr uint ks_per_thread = 8;
  constexpr uint bytes_per_pack = 3;
  constexpr uint k_pack_factor = 8;
  const uint K = sizes.y;
  const uint N = sizes.z;
  const uint num_groups = (K + group_size - 1) / group_size;
  uint n = thread_index.x; // 0..N/4-1
  uint m = thread_index.z; // 0..M
  n = n / threads_per_channel;
  n = n * 4;

  // This is starting k for each thread.
  uint k = (tid_in_simdgroup % threads_per_channel) * ks_per_thread;
  constexpr int k_jump = threads_per_channel * ks_per_thread;

  using vecT = typename Vec4Type<T>::type;
  constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A + m * K);
  constant uchar *B_ptr = B + n * bytes_per_pack * K / k_pack_factor;

  thread float4 result = float4(0.0);

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

    float4 a_val[2];
    a_val[0] = float4(A_ptr[k / 4]);
    a_val[1] = float4(A_ptr[k / 4 + 1]);

    float a_val_sum = a_val[0][0] + a_val[0][1] + a_val[0][2] + a_val[0][3];
    a_val_sum += a_val[1][0] + a_val[1][1] + a_val[1][2] + a_val[1][3];

    uchar3 b_val0 = (reinterpret_cast<constant uchar3 *>(
        B_ptr + bytes_per_pack * (k + 0 * K) / k_pack_factor))[0];
    uchar3 b_val1 = (reinterpret_cast<constant uchar3 *>(
        B_ptr + bytes_per_pack * (k + 1 * K) / k_pack_factor))[0];
    uchar3 b_val2 = (reinterpret_cast<constant uchar3 *>(
        B_ptr + bytes_per_pack * (k + 2 * K) / k_pack_factor))[0];
    uchar3 b_val3 = (reinterpret_cast<constant uchar3 *>(
        B_ptr + bytes_per_pack * (k + 3 * K) / k_pack_factor))[0];

    float4x4 b_mat[2];

    thread float w0[8];
    unpack_3bit(b_val0, w0);

    thread float w1[8];
    unpack_3bit(b_val1, w1);

    thread float w2[8];
    unpack_3bit(b_val2, w2);

    thread float w3[8];
    unpack_3bit(b_val3, w3);

    b_mat[0][0] = scales[0] * float4(w0[0], w0[1], w0[2], w0[3]),
    b_mat[1][0] = scales[0] * float4(w0[4], w0[5], w0[6], w0[7]),
    b_mat[0][1] = scales[1] * float4(w1[0], w1[1], w1[2], w1[3]),
    b_mat[1][1] = scales[1] * float4(w1[4], w1[5], w1[6], w1[7]),
    b_mat[0][2] = scales[2] * float4(w2[0], w2[1], w2[2], w2[3]),
    b_mat[1][2] = scales[2] * float4(w2[4], w2[5], w2[6], w2[7]),
    b_mat[0][3] = scales[3] * float4(w3[0], w3[1], w3[2], w3[3]),
    b_mat[1][3] = scales[3] * float4(w3[4], w3[5], w3[6], w3[7]),

    result += a_val[0] * b_mat[0];
    result += a_val[1] * b_mat[1];
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

#define INSTANTIATE_INT3MM(DTYPE, GSIZE)                                       \
  template [[host_name("int3pack_mm_" #GSIZE "_" #DTYPE)]] kernel void         \
  int3pack_mm<DTYPE, GSIZE>(                                                   \
      constant DTYPE * A [[buffer(0)]], constant uchar * B [[buffer(1)]],      \
      constant DTYPE * scales_ptr [[buffer(2)]],                               \
      constant DTYPE * zeros_ptr [[buffer(3)]],                                \
      device DTYPE * output_data [[buffer(4)]],                                \
      constant uint3 & sizes [[buffer(5)]],                                    \
      uint3 thread_index [[thread_position_in_grid]],                          \
      uint tid_in_simdgroup [[thread_index_in_simdgroup]])

INSTANTIATE_INT3MM(float, 32);
INSTANTIATE_INT3MM(half, 32);
INSTANTIATE_INT3MM(float, 64);
INSTANTIATE_INT3MM(half, 64);
INSTANTIATE_INT3MM(float, 128);
INSTANTIATE_INT3MM(half, 128);
INSTANTIATE_INT3MM(float, 256);
INSTANTIATE_INT3MM(half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT3MM(bfloat, 32);
INSTANTIATE_INT3MM(bfloat, 64);
INSTANTIATE_INT3MM(bfloat, 128);
INSTANTIATE_INT3MM(bfloat, 256);
#endif
