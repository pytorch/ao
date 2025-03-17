// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#include <metal_stdlib>
using namespace metal;

/**
 * 1-Bit Quantized Linear.
 *
 * @param[A] M x K input tensor of floating point dtype (Float, Half, BFloat16)
 * @param[B] Packed & quantized weight tensor of uint8 dtype. Expected shape is N x (K / 8)
 * @param[scales] 2D tensor containg the scales for each group. Expected shape is #groups x N
 * @param[zeros] 2D tensor containg the zero points for each group. Expected shape is #groups x N
 * @param[outputData] M x N output tensor of floating point dtype (same as input)
 * @param[sizes] The sizes involved in the order: M, K, N
 *
 * Dispatched threads: N x M x 1
 */
template<typename T, unsigned groupSize>
kernel void int1pack_mm(
    constant T                 * A              [[buffer(0)]],
    constant uchar             * B              [[buffer(1)]],
    constant T                 * scales         [[buffer(2)]],
    constant T                 * zeros          [[buffer(3)]],
    device   T                 * outputData     [[buffer(4)]],
    constant uint3             & sizes          [[buffer(5)]], // M, K, N
    uint2                        thread_index   [[thread_position_in_grid]]) {
    const uint K = sizes.y;
    const uint N = sizes.z;
    const uint m = thread_index.y; // 0..M-1
    const uint n = thread_index.x; // 0..N-1
    const uint32_t k_block = (K + groupSize - 1) / groupSize;
    constant T *A_ptr = A + m * K;
    constant uchar *B_ptr = B + n * K / 8;

    float rc = 0.0;
    uint k = 0;
    for (uint32_t kb = 0; kb < k_block ; kb ++) {
      const float scale = float(scales[kb * N + n]);
      const float zero = float(zeros[kb * N + n]);
      for(uint idx = 0; idx < groupSize && k < K; idx+=8, k+=8) {
        const auto a_val0 = float(A_ptr[k + 0]);
        const auto a_val1 = float(A_ptr[k + 1]);
        const auto a_val2 = float(A_ptr[k + 2]);
        const auto a_val3 = float(A_ptr[k + 3]);
        const auto a_val4 = float(A_ptr[k + 4]);
        const auto a_val5 = float(A_ptr[k + 5]);
        const auto a_val6 = float(A_ptr[k + 6]);
        const auto a_val7 = float(A_ptr[k + 7]);

        uchar b0 = B_ptr[(k / 8)];

        uchar w_val0 = b0 & 0x01;
        uchar w_val1 = (b0 & 0x02) >> 1;
        uchar w_val2 = (b0 & 0x04) >> 2;
        uchar w_val3 = (b0 & 0x08) >> 3;
        uchar w_val4 = (b0 & 0x10) >> 4;
        uchar w_val5 = (b0 & 0x20) >> 5;
        uchar w_val6 = (b0 & 0x40) >> 6;
        uchar w_val7 = (b0 & 0x80) >> 7;

        rc += a_val0 * (scale * float(w_val0) + zero);
        rc += a_val1 * (scale * float(w_val1) + zero);
        rc += a_val2 * (scale * float(w_val2) + zero);
        rc += a_val3 * (scale * float(w_val3) + zero);
        rc += a_val4 * (scale * float(w_val4) + zero);
        rc += a_val5 * (scale * float(w_val5) + zero);
        rc += a_val6 * (scale * float(w_val6) + zero);
        rc += a_val7 * (scale * float(w_val7) + zero);
      }
    }
    outputData[m * N + n] = T(rc);
}

#define INSTANTIATE_INT1MM(DTYPE, GSIZE)                                 \
template                                                                 \
[[host_name("int1pack_mm_" #GSIZE "_" #DTYPE)]]                          \
kernel void int1pack_mm<DTYPE, GSIZE>(                                   \
    constant DTYPE             * A              [[buffer(0)]],           \
    constant uchar             * B              [[buffer(1)]],           \
    constant DTYPE             * scales         [[buffer(2)]],           \
    constant DTYPE             * zeros          [[buffer(3)]],           \
    device   DTYPE             * outputData     [[buffer(4)]],           \
    constant uint3             & sizes          [[buffer(5)]],           \
    uint2                        thread_index [[thread_position_in_grid]])

INSTANTIATE_INT1MM(float, 32);
INSTANTIATE_INT1MM(half, 32);
INSTANTIATE_INT1MM(float, 64);
INSTANTIATE_INT1MM(half, 64);
INSTANTIATE_INT1MM(float, 128);
INSTANTIATE_INT1MM(half, 128);
INSTANTIATE_INT1MM(float, 256);
INSTANTIATE_INT1MM(half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT1MM(bfloat, 32);
INSTANTIATE_INT1MM(bfloat, 64);
INSTANTIATE_INT1MM(bfloat, 128);
INSTANTIATE_INT1MM(bfloat, 256);
#endif
