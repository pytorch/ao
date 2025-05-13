// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

/*
   This code was taken from MLX, and modified to add support for 1, 5 & 7 bit packing.
   The original code is Copyright © 2023-2024 Apple Inc.
   https://github.com/ml-explore/mlx/blob/481349495b8c3d094eb699e678077bbe1406392d/mlx/backend/metal/kernels/quantized.h#L1
   MLX MIT License: https://github.com/ml-explore/mlx/blob/main/LICENSE
*/

#include <metal_simdgroup>
#include <metal_stdlib>

static constant constexpr const int SIMD_SIZE = 32;

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(constant T* x, thread U* x_thread) {
  static_assert(
      1 <= bits && bits <= 7,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 7}");

  U sum = 0;

  if (bits == 1) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 2.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 8.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 32.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 128.0f;
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 7) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 128.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 32.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 8.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 2.0f;
    }
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    constant uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      1 <= bits && bits <= 7,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 7}");

  U accum = 0;

  if (bits == 1) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;

      accum +=
          (x_thread[0] * (w[i] & 0x01) +
           x_thread[1] * (w[i] & 0x02) +
           x_thread[2] * (w[i] & 0x04) +
           x_thread[3] * (w[i] & 0x08) +
           x_thread[4] * (w[i] & 0x10) +
           x_thread[5] * (w[i] & 0x20) +
           x_thread[6] * (w[i] & 0x40) +
           x_thread[7] * (w[i] & 0x80));
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    constant uint16_t* ws = (constant uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];

      accum += (w[1] & 0x03) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];

      accum += (w[2] & 0x0f) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];

      accum += (w[3] & 0x01) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];

      accum += (w[4] & 0x07) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 7) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 7 * i;

      accum += (w[0] & 0x7f) * x_thread[0];
      accum += (w[0] & 0x80) * x_thread[1];

      accum += (w[1] & 0x3f) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0xc0) * x_thread[2];

      accum += (w[2] & 0x1f) * (x_thread[2] * 256.0f);
      accum += (w[2] & 0xe0) * x_thread[3];

      accum += (w[3] & 0x0f) * (x_thread[3] * 256.0f);
      accum += (w[3] & 0xf0) * x_thread[4];

      accum += (w[4] & 0x07) * (x_thread[4] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[5];

      accum += (w[5] & 0x03) * (x_thread[5] * 256.0f);
      accum += (w[5] & 0xfc) * x_thread[6];

      accum += (w[6] & 0x01) * (x_thread[6] * 256.0f);
      accum += (w[6] & 0xfe) * x_thread[7];
    }
  }

  return scale * accum + sum * bias;
}

template <typename T, int group_size, int bits>
[[kernel]] void qmv_fast(
    constant T* x [[buffer(0)]],
    constant uchar* w [[buffer(1)]],
    constant T* scales [[buffer(2)]],
    constant T* biases [[buffer(3)]],
    device T* y [[buffer(4)]],
    constant uint3 &sizes [[buffer(5)]], // M, K, N
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int in_vec_size = static_cast<int>(sizes.y); // K
  const int out_vec_size = static_cast<int>(sizes.z); // N

  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int packs_per_thread = (bits == 1 || bits == 2) ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = bits == 1 ? 16 : power_of_2_bits ? 32 / bits : bits == 6 ? 4 : 8;
  constexpr int bytes_per_pack = bits == 1 ? 2 : power_of_2_bits ? 4 : bits == 6 ? 3 : bits;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  constant uint8_t* ws = (constant uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
      constant T* sl = scales + row * in_vec_size_g;
      constant T* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

#define INSTANTIATE_QMV_FAST(DTYPE, GSIZE, NBIT)                                 \
  template [[host_name("qmv_fast_" #NBIT "bit_" #GSIZE "_" #DTYPE)]] kernel void \
  qmv_fast<DTYPE, GSIZE, NBIT>(                                                  \
      constant DTYPE * A [[buffer(0)]],                                          \
      constant uchar * B [[buffer(1)]],                                          \
      constant DTYPE * scales_ptr [[buffer(2)]],                                 \
      constant DTYPE * zeros_ptr [[buffer(3)]],                                  \
      device DTYPE * output_data [[buffer(4)]],                                  \
      constant uint3 & sizes [[buffer(5)]],                                      \
      uint3 thread_index [[thread_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                          \
      uint tid_in_simdgroup [[thread_index_in_simdgroup]])

#define INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, GSIZE) \
  INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 1);               \
  INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 2);               \
  INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 3);               \
  INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 4);               \
  INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 5);               \
  INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 6);               \
  INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 7);

#define INSTANTIATE_QMV_FAST_DTYPE(DTYPE)       \
  INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, 32);  \
  INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, 64);  \
  INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, 128); \
  INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, 256);

INSTANTIATE_QMV_FAST_DTYPE(float);
INSTANTIATE_QMV_FAST_DTYPE(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_QMV_FAST_DTYPE(bfloat);
#endif
