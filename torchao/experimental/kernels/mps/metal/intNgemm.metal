// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
//
// Generalized low-bit GEMM kernel using simdgroup matrix multiply accumulation.
// Supports int1 through int8 weight-only quantization for the prefill case (M > 1).
//
// The tiling structure (32x64 output tiles, 4 simdgroups, outer-product
// accumulation, threadgroup staging) is adapted from PyTorch's kernel_mul_mm
// in aten/src/ATen/native/mps/kernels/Quantized.metal, which is itself
// credited as "heavily inspired by llama.cpp" (MIT).
//
// The per-bitwidth dequantization logic (Dequant8<nbit>) is the bitwise
// inverse of the packing functions in packing.h (same BSD-3-Clause license,
// Meta). The packing format matches MLX's format (MLX MIT license, via
// qmv.metal in this directory), but the dequantization code here is derived
// from packing.h, not from MLX.
//
// Performance context:
//   The primary benefits of low-bit weight-only quantization on Apple Silicon
//   are (1) weight memory compression and (2) faster decode (M=1), which is
//   memory-bound and handled by the qmv kernel.  Prefill (M>1) is
//   compute-bound, so dequant + matmul is generally slower than a native
//   bf16 matmul -- Apple Silicon has no low-bit MMA, and dequantization is
//   pure overhead.  This kernel does NOT make low-bit prefill faster than
//   bf16 prefill.  What it does is make low-bit prefill significantly faster
//   than the previous per-bitwidth pack_mm paths (which were GEMV-shaped with
//   no simdgroup MMA), so that choosing low-bit for memory/decode reasons is
//   less penalizing at prefill.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 2
#define THREAD_MAT_N 4
#define THREAD_PER_ROW_A 4
#define THREAD_PER_ROW_B 2
#define SG_MAT_SIZE 64
#define SG_MAT_ROW 8

template <typename T> struct BlockType {};

template <> struct BlockType<float> {
  using simdgroup_type8x8 = simdgroup_float8x8;
  using type4 = float4;
};

template <> struct BlockType<half> {
  using simdgroup_type8x8 = simdgroup_half8x8;
  using type4 = half4;
};

#if __METAL_VERSION__ >= 310
template <> struct BlockType<bfloat> {
  using simdgroup_type8x8 = simdgroup_bfloat8x8;
  using type4 = bfloat4;
};
#endif

// ---------------------------------------------------------------------------
// Dequantization: extract 8 weight values from packed bytes.
// Each specialization matches the packing format in packing.h.
// All bit widths use the same final formula: weight = scale * val + zero
// Uses struct specialization (proven pattern in Metal, same as BlockType).
// ---------------------------------------------------------------------------
template <int nbit>
struct Dequant8 {
  // Bytes consumed: 8 values * nbit bits / 8 bits = nbit bytes
  static inline void apply(constant uchar* b, thread float* w);
};

// 1-bit: 8 values in 1 byte
template <>
struct Dequant8<1> {

  static inline void apply(constant uchar* b, thread float* w) {
    w[0] = float(b[0] & 0x01);
    w[1] = float((b[0] & 0x02) >> 1);
    w[2] = float((b[0] & 0x04) >> 2);
    w[3] = float((b[0] & 0x08) >> 3);
    w[4] = float((b[0] & 0x10) >> 4);
    w[5] = float((b[0] & 0x20) >> 5);
    w[6] = float((b[0] & 0x40) >> 6);
    w[7] = float((b[0] & 0x80) >> 7);
  }
};

// 2-bit: 4 values in 1 byte, 8 values in 2 bytes
template <>
struct Dequant8<2> {

  static inline void apply(constant uchar* b, thread float* w) {
    w[0] = float(b[0] & 0x03);
    w[1] = float((b[0] & 0x0c) >> 2);
    w[2] = float((b[0] & 0x30) >> 4);
    w[3] = float((b[0] & 0xc0) >> 6);
    w[4] = float(b[1] & 0x03);
    w[5] = float((b[1] & 0x0c) >> 2);
    w[6] = float((b[1] & 0x30) >> 4);
    w[7] = float((b[1] & 0xc0) >> 6);
  }
};

// 3-bit: 8 values in 3 bytes
template <>
struct Dequant8<3> {

  static inline void apply(constant uchar* b, thread float* w) {
    w[0] = float(b[0] & 0x07);
    w[1] = float((b[0] & 0x38) >> 3);
    w[2] = float(((b[0] & 0xc0) >> 6) | ((b[1] & 0x01) << 2));
    w[3] = float((b[1] & 0x0e) >> 1);
    w[4] = float((b[1] & 0x70) >> 4);
    w[5] = float(((b[1] & 0x80) >> 7) | ((b[2] & 0x03) << 1));
    w[6] = float((b[2] & 0x1c) >> 2);
    w[7] = float((b[2] & 0xe0) >> 5);
  }
};

// 4-bit: 2 values in 1 byte, 8 values in 4 bytes
template <>
struct Dequant8<4> {

  static inline void apply(constant uchar* b, thread float* w) {
    w[0] = float(b[0] & 0x0f);
    w[1] = float((b[0] & 0xf0) >> 4);
    w[2] = float(b[1] & 0x0f);
    w[3] = float((b[1] & 0xf0) >> 4);
    w[4] = float(b[2] & 0x0f);
    w[5] = float((b[2] & 0xf0) >> 4);
    w[6] = float(b[3] & 0x0f);
    w[7] = float((b[3] & 0xf0) >> 4);
  }
};

// 5-bit: 8 values in 5 bytes
template <>
struct Dequant8<5> {

  static inline void apply(constant uchar* b, thread float* w) {
    w[0] = float(b[0] & 0x1f);
    w[1] = float(((b[0] & 0xe0) >> 5) | ((b[1] & 0x03) << 3));
    w[2] = float((b[1] & 0x7c) >> 2);
    w[3] = float(((b[1] & 0x80) >> 7) | ((b[2] & 0x0f) << 1));
    w[4] = float(((b[2] & 0xf0) >> 4) | ((b[3] & 0x01) << 4));
    w[5] = float((b[3] & 0x3e) >> 1);
    w[6] = float(((b[3] & 0xc0) >> 6) | ((b[4] & 0x07) << 2));
    w[7] = float((b[4] & 0xf8) >> 3);
  }
};

// 6-bit: 4 values in 3 bytes, 8 values in 6 bytes
template <>
struct Dequant8<6> {

  static inline void apply(constant uchar* b, thread float* w) {
    w[0] = float(b[0] & 0x3f);
    w[1] = float(((b[0] & 0xc0) >> 6) | ((b[1] & 0x0f) << 2));
    w[2] = float(((b[1] & 0xf0) >> 4) | ((b[2] & 0x03) << 4));
    w[3] = float((b[2] & 0xfc) >> 2);
    w[4] = float(b[3] & 0x3f);
    w[5] = float(((b[3] & 0xc0) >> 6) | ((b[4] & 0x0f) << 2));
    w[6] = float(((b[4] & 0xf0) >> 4) | ((b[5] & 0x03) << 4));
    w[7] = float((b[5] & 0xfc) >> 2);
  }
};

// 7-bit: 8 values in 7 bytes
template <>
struct Dequant8<7> {

  static inline void apply(constant uchar* b, thread float* w) {
    w[0] = float(b[0] & 0x7f);
    w[1] = float((b[0] >> 7) | ((b[1] & 0x3f) << 1));
    w[2] = float((b[1] >> 6) | ((b[2] & 0x1f) << 2));
    w[3] = float((b[2] >> 5) | ((b[3] & 0x0f) << 3));
    w[4] = float((b[3] >> 4) | ((b[4] & 0x07) << 4));
    w[5] = float((b[4] >> 3) | ((b[5] & 0x03) << 5));
    w[6] = float((b[5] >> 2) | ((b[6] & 0x01) << 6));
    w[7] = float(b[6] >> 1);
  }
};

// 8-bit: 1 value per byte, no extraction needed
template <>
struct Dequant8<8> {

  static inline void apply(constant uchar* b, thread float* w) {
    w[0] = float(b[0]);
    w[1] = float(b[1]);
    w[2] = float(b[2]);
    w[3] = float(b[3]);
    w[4] = float(b[4]);
    w[5] = float(b[5]);
    w[6] = float(b[6]);
    w[7] = float(b[7]);
  }
};

/**

Generalized low-bit GEMM kernel
A: [M, K] activation matrix
B: [N, nbit*K/8] packed low-bit weight matrix
S: [N, num_groups] scales
Z: [N, num_groups] zeros, stored as the actual zero (bias) term, i.e.
   Z = -scale * zero_point, so that weight = scale * nibble + Z.
output_data: [M, N] output matrix

Algorithm (adapted from PyTorch's kernel_mul_mm, which is heavily inspired
by llama.cpp (MIT); see aten/src/ATen/native/mps/kernels/Quantized.metal):
  1. Load A block (32x32) and B block (64x32 packed low-bit) into shared
     memory, dequantizing B inline (scale * val + zero -> half) during the
     load. (PyTorch loads raw int8 and dequantizes post-accumulation; we
     dequantize inline because we support group-wise quantization with
     non-zero zero points across 1-8 bits.)
  2. In 4 simdgroups, calculate the outer product of the loaded blocks.
     Each simdgroup produces a 2x4 arrangement of 8x8 result tiles.
     For how to use outer product to perform matrix multiplication, refer to
       https://web.archive.org/web/20230521063455/http://mlwiki.org/index.php/Matrix-Matrix_Multiplication#Sum_of_Outer_Products
  3. Repeat steps 1 & 2 along the K axis (block size 32), accumulating
     into the 2x4 8x8 result tiles.
  4. After a threadgroup_barrier, reuse the A region to store the
     accumulated result, then write to the output matrix. (No post-
     accumulation dequant needed -- scale/zero were applied inline in
     step 1; PyTorch's kernel_mul_mm applies scale post-accumulation via a
     diagonal scale matrix.)

 */
template <typename T, unsigned group_size_template, int nbit>
kernel void intNgemm_mm(
    constant T *A [[buffer(0)]],
    constant uchar *B [[buffer(1)]],
    constant T *scales_ptr [[buffer(2)]],
    constant T *zeros_ptr [[buffer(3)]],
    device T *output_data [[buffer(4)]],
    constant uint3 &sizes [[buffer(5)]], // M, K, N
    threadgroup char *shared_memory [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {

  using T4 = typename BlockType<T>::type4;
  using Tsimd8x8 = typename BlockType<T>::simdgroup_type8x8;

  // group_size is a compile-time constant (template parameter).
  const unsigned group_size = group_size_template;

  const uint M = sizes.x;
  const uint K = sizes.y;
  const uint N = sizes.z;
  const uint num_groups = (K + group_size - 1) / group_size;
  // 8 values * nbit bits / 8 bits = nbit bytes per group of 8
  constexpr int bytes_per_8_vals = nbit;
  constexpr int bytes_per_16_vals = 2 * bytes_per_8_vals;

  // Shared memory: dynamic allocation (size set by dispatch_gemm to 12288).
  //   A: 8192 bytes (BLOCK_SIZE_M * BLOCK_SIZE_K * sizeof(T), worst case float)
  //   B: 4096 bytes (BLOCK_SIZE_N * BLOCK_SIZE_K * sizeof(half))
  // After the K-loop, A is dead and reused for the 32x64 result matrix
  // (8192 bytes of float). This matches PyTorch's kernel_mul_mm layout.
  // A threadgroup_barrier is required before the reuse (see result-store
  // section below).
  threadgroup T *shared_A = (threadgroup T *)(shared_memory);
  threadgroup half *shared_B = (threadgroup half *)(shared_memory + 8192);

  const uint threadgroup_M = tgpig.x;
  const uint threadgroup_N = tgpig.y;

  // Bound the number of rows for this block (edge case handling)
  short n_rows_A = (short)min((uint)BLOCK_SIZE_M, M - threadgroup_M * BLOCK_SIZE_M);
  short n_rows_B = (short)min((uint)BLOCK_SIZE_N, N - threadgroup_N * BLOCK_SIZE_N);

  // A thread shouldn't load data outside the matrix
  short thread_row_A = (short)min((int)(tiitg / THREAD_PER_ROW_A), (int)(n_rows_A - 1));
  short thread_row_B = (short)min((int)(tiitg / THREAD_PER_ROW_B), (int)(n_rows_B - 1));

  Tsimd8x8 simdgroup_A[2];
  simdgroup_half8x8 simdgroup_B[4];
  simdgroup_float8x8 simdgroup_C[8];

  for (short i = 0; i < 8; i++) {
    simdgroup_C[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
  }

  // Pointer to A for this threadgroup's block (base, before K offset)
  constant T *a_ptr_base = A
    + (threadgroup_M * BLOCK_SIZE_M + thread_row_A) * K
    + (BLOCK_SIZE_K / THREAD_PER_ROW_A) * (tiitg % THREAD_PER_ROW_A);

  // Pointer to B for this threadgroup's block (base, before K offset)
  // B is [N, nbit*K/8] packed bytes. Row stride = nbit*K/8 bytes.
  // Each thread loads 16 k-values = bytes_per_16_vals bytes.
  constant uchar *b_ptr_base = B
    + (threadgroup_N * BLOCK_SIZE_N + thread_row_B) * (nbit * K / 8)
    + bytes_per_16_vals * (tiitg % THREAD_PER_ROW_B);

  /**

  Tiling structure adapted from PyTorch's kernel_mul_mm
  (aten/src/ATen/native/mps/kernels/Quantized.metal), which is heavily
  inspired by llama.cpp (MIT).

  Load weight and input into shared memory (2-region layout):
  8192: BLOCK_SIZE_M x BLOCK_SIZE_K x 4 (max bytes per value) <----- numbers don't checkout, should be 4096. Changing it to 4096 gives wrong value.
                                                                     (Same quirk as PyTorch's kernel_mul_mm.)
  4096: BLOCK_SIZE_N x BLOCK_SIZE_K x 2 (storing dequantized weight in half)

                            K
                 +------------------------+              8192(A)             4096(B)
                 |                        |   +------------------------+------------+
                 |                        |   |++++++++++++++++++++++++|++++++++++++|
                 |                        |   +------------------------+------------+
                 |                        |
                 |32(BLOCK_SIZE_K)        |
                 +--+--+------------------+
                 |++|  |                  |
               64|++|  |...               |
   (BLOCK_SIZE_N)|++|  |                  |
                 +--+--+------------------+
                 |                        |
                 |      -------->         |                           K
                 |       for loop         |               +------------------------+
                 |                        |               |                        |
                 |                        |               |                        |
                 |                        |               |32(BLOCK_SIZE_K)        |
                 |                        |               +--+--+------------------+
                 |                        |             32|++|  | ...              |
                 |                        | (BLOCK_SIZE_M)+--+--+------------------+
                 |                        |               |         ----------->   |
                 |                        |               |            for loop    |
                 +------------------------+               +------------------------+
                             B                                        A

  During the K-loop: A and B are both live (loaded each iteration).
  After the K-loop:  A is dead; its 8192 bytes are reused to stage the
                     32x64 float result matrix before writing to device
                     memory. A threadgroup_barrier is required before the
                     reuse (see result-store section below).

   */
  for (uint loop_k = 0; loop_k < K; loop_k += BLOCK_SIZE_K) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constant T *a_ptr = a_ptr_base + loop_k;
    constant uchar *b_ptr = b_ptr_base + (nbit * loop_k / 8);

    // --- Load B (low-bit weights) and dequantize to half in shared memory ---
    // Each thread loads 16 weight values (= 2 groups of 8).
    // All 16 values fall within the same group (BLOCK_SIZE_K=32 <= group_size).
    //
    // Deviation from PyTorch's kernel_mul_mm: we dequantize inline during the
    // load (applying scale * val + zero here) rather than loading raw int8 and
    // dequantizing post-accumulation. This is because we support group-wise
    // quantization with non-zero zero points across 1-8 bits, which makes the
    // post-accumulation diagonal-scale-matrix approach (used by PyTorch for
    // per-channel symmetric int8) inapplicable.
    uint n_abs = threadgroup_N * BLOCK_SIZE_N + thread_row_B;
    float scale;
    float zero;
    {
      uint k_block_index = loop_k / group_size;
      uint sz_idx = n_abs * num_groups + k_block_index;
      scale = float(scales_ptr[sz_idx]);
      // Deviation from PyTorch's kernel_mul_mm: scales and zeros are passed as
      // separate buffers (not interleaved into a single scales_and_zeros
      // buffer). The stored zero is the actual zero (bias) term:
      // Z = -scale * zero_point, so dequantization is weight = scale * nibble
      // + zero.  This matches the format produced by
      // IntxMPSExperimentalTensor.from_hp and is NOT the tinygemm convention
      // (zero = scale * (8 - zero_point)).
      zero = float(zeros_ptr[sz_idx]);
    }

    float w_vals[16];
    Dequant8<nbit>::apply(b_ptr, w_vals);
    Dequant8<nbit>::apply(b_ptr + bytes_per_8_vals, w_vals + 8);

    #pragma unroll(16)
    for (short i = 0; i < 16; i++) {
      float weight = scale * w_vals[i] + zero;

      // Store to shared_B in simdgroup matrix layout (same as kernel_mul_mm)
      short sg_mat_grid_row_index = (tiitg % THREAD_PER_ROW_B) * THREAD_PER_ROW_B + i / 8;
      short sg_mat_grid_col_index = tiitg / THREAD_PER_ROW_B / 8;
      short row_offset = i % 8;
      short col_offset = (tiitg / THREAD_PER_ROW_B) % 8;
      short sb_offset = (sg_mat_grid_row_index * 8 + sg_mat_grid_col_index) * 64
        + (row_offset * 8 + col_offset);
      *(shared_B + sb_offset) = half(weight);
    }

    // --- Load A (activations) into shared memory ---
    // Each thread loads 2 x T4 = 8 values
    #pragma unroll(2)
    for (short i = 0; i < 2; i++) {
      *((threadgroup T4 *)(shared_A
        + (tiitg % THREAD_PER_ROW_A) * 8 * 32
        + 8 * (tiitg / THREAD_PER_ROW_A)) + i) = *((constant T4 *)a_ptr + i);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /**

    Outer product accumulation.
    Adapted from PyTorch's kernel_mul_mm (aten/src/ATen/native/mps/kernels/
    Quantized.metal), which is heavily inspired by llama.cpp (MIT).

    Outer product:
                  K
           ----------->
         8    for loop              8   8
       +---+---+---+---+          +---+---+---+---+---+---+---+---+
     8 |+++|   |   |   |      |  8|+++|+++|+++|+++|###|###|###|###|
       +---+---+---+---+      |   +---+---+---+---+---+---+---+---+
       |+++|   |   |   |      |   |   |   |   |   |   |   |   |   |
       +---+---+---+---+      | K +---+---+---+---+---+---+---+---+
       |###|   |   |   |      |   |   |   |   |   |   |   |   |   |
       +---+---+---+---+      |   +---+---+---+---+---+---+---+---+
       |###|   |   |   |      |   |   |   |   |   |   |   |   |   |
       +---+---+---+---+      v   +---+---+---+---+---+---+---+---+
                           for loop
        + simdgroup 0,1                + simdgroup 0,2
        # simdgroup 2,3                # simdgroup 1,3

     */
    threadgroup T *simdgroup_A_ptr = shared_A + THREAD_MAT_M * SG_MAT_SIZE * (sgitg / 2);
    threadgroup half *simdgroup_B_ptr = shared_B + THREAD_MAT_N * SG_MAT_SIZE * (sgitg % 2);

    #pragma unroll(4)
    for (short ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
      #pragma unroll(4)
      for (short i = 0; i < 4; i++) {
        simdgroup_load(simdgroup_B[i], simdgroup_B_ptr + SG_MAT_SIZE * i);
      }
      simdgroup_barrier(mem_flags::mem_none);
      #pragma unroll(2)
      for (short i = 0; i < 2; i++) {
        simdgroup_load(simdgroup_A[i], simdgroup_A_ptr + SG_MAT_SIZE * i);
      }

      simdgroup_A_ptr += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
      simdgroup_B_ptr += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

      #pragma unroll(8)
      for (short i = 0; i < 8; i++) {
        simdgroup_multiply_accumulate(
            simdgroup_C[i], simdgroup_A[i / 4], simdgroup_B[i % 4], simdgroup_C[i]);
      }
    }
  }

  /**

  Store results.
  Adapted from PyTorch's kernel_mul_mm (aten/src/ATen/native/mps/kernels/
  Quantized.metal). Each sgitg 0,1,2,3 handles a 2x4 arrangement of 8x8
  result tiles:

     8   8
   +---+---+---+---+---+---+---+---+
 8 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
   +---+---+---+---+---+---+---+---+
   | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
   +---+---+---+---+---+---+---+---+
   | 2 | 2 | 2 | 2 | 3 | 3 | 3 | 3 |
   +---+---+---+---+---+---+---+---+
   | 2 | 2 | 2 | 2 | 3 | 3 | 3 | 3 |
   +---+---+---+---+---+---+---+---+

  Deviation from PyTorch's kernel_mul_mm: PyTorch's diagram includes a
  scale-diagonal-matrix part (it applies scale post-accumulation via a
  diagonal scale matrix stored in shared_memory_B). We omit that because
  we apply scale/zero inline during the B load (see step 1 of the
  algorithm above), so no post-accumulation dequant is needed.

  threadgroup_barrier required before reusing shared_A for results.
  The K-loop's outer-product accumulation only uses simdgroup_barrier
  (within-simdgroup sync), not threadgroup_barrier (across-simdgroup sync).
  Without this barrier, one simdgroup could start writing results to
  shared_A while another simdgroup is still reading from it in its last
  simdgroup_load. This pattern follows PyTorch's kernel_mul_mm, which has
  an equivalent threadgroup_barrier inside its post-K-loop dequant loop
  (aten/src/ATen/native/mps/kernels/Quantized.metal, ~line 456).

  Reuse the A region (offset 0) for the result matrix. A is dead after the
  K-loop completes; the barrier above ensures all simdgroups have finished
  reading from shared_A before any simdgroup writes to it.

   */
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float *temp_str = (threadgroup float *)(shared_memory)
    + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * BLOCK_SIZE_N;

  for (int i = 0; i < 8; i++) {
    simdgroup_store(
        simdgroup_C[i],
        temp_str + 8 * (i % 4) + 8 * BLOCK_SIZE_N * (i / 4),
        BLOCK_SIZE_N);
  }

  // Ensure all simdgroup stores to threadgroup memory are visible.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write to output_data [M, N]
  device T *C = output_data + (BLOCK_SIZE_N * threadgroup_N)
    + (BLOCK_SIZE_M * threadgroup_M) * N;
  if (sgitg == 0) {
    for (int i = 0; i < n_rows_B; i++) {
      // Deviation note: PyTorch's kernel_mul_mm uses `j += BLOCK_SIZE_M` here.
      // We use the same stride (BLOCK_SIZE_M = 32), which is equivalent to the
      // previous `j += 128` since n_rows_A <= BLOCK_SIZE_M and there are 128
      // threads per threadgroup (4 simdgroups x 32 threads). Using
      // BLOCK_SIZE_M is self-documenting and matches PyTorch's convention.
      for (int j = tiitg; j < n_rows_A; j += BLOCK_SIZE_M) {
        float val = *(temp_str + i + j * BLOCK_SIZE_N);
        *(C + i + j * N) = (device T)val;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Instantiation for all bit widths (1-8), group sizes, and dtypes.
// The function name is intNgemm_mm_<nbit>bit_<group_size>_<dtype> so the
// dispatch layer can select the right kernel.
// ---------------------------------------------------------------------------
#define INSTANTIATE_INTNGEMM(NBIT, DTYPE, GSIZE)                               \
  template [[host_name("intNgemm_mm_" #NBIT "bit_" #GSIZE "_" #DTYPE)]]        \
  kernel void intNgemm_mm<DTYPE, GSIZE, NBIT>(                                 \
      constant DTYPE *A [[buffer(0)]],                                         \
      constant uchar *B [[buffer(1)]],                                         \
      constant DTYPE *scales_ptr [[buffer(2)]],                                \
      constant DTYPE *zeros_ptr [[buffer(3)]],                                 \
      device DTYPE *output_data [[buffer(4)]],                                 \
      constant uint3 &sizes [[buffer(5)]],                                     \
      threadgroup char *shared_memory [[threadgroup(0)]],                      \
      uint3 tgpig [[threadgroup_position_in_grid]],                            \
      uint tiitg [[thread_index_in_threadgroup]],                              \
      uint sgitg [[simdgroup_index_in_threadgroup]])

// Helper macro: instantiate one bit width for all group sizes and float/half
#define INSTANTIATE_NBIT_FLOAT_HALF(NBIT)                                      \
  INSTANTIATE_INTNGEMM(NBIT, float, 32);                                       \
  INSTANTIATE_INTNGEMM(NBIT, half, 32);                                        \
  INSTANTIATE_INTNGEMM(NBIT, float, 64);                                       \
  INSTANTIATE_INTNGEMM(NBIT, half, 64);                                        \
  INSTANTIATE_INTNGEMM(NBIT, float, 128);                                      \
  INSTANTIATE_INTNGEMM(NBIT, half, 128);                                       \
  INSTANTIATE_INTNGEMM(NBIT, float, 256);                                      \
  INSTANTIATE_INTNGEMM(NBIT, half, 256)

// Helper macro: add bfloat instantiations (requires Metal 3.1+)
#define INSTANTIATE_NBIT_BFLOAT(NBIT)                                          \
  INSTANTIATE_INTNGEMM(NBIT, bfloat, 32);                                      \
  INSTANTIATE_INTNGEMM(NBIT, bfloat, 64);                                      \
  INSTANTIATE_INTNGEMM(NBIT, bfloat, 128);                                     \
  INSTANTIATE_INTNGEMM(NBIT, bfloat, 256)

INSTANTIATE_NBIT_FLOAT_HALF(1);
INSTANTIATE_NBIT_FLOAT_HALF(2);
INSTANTIATE_NBIT_FLOAT_HALF(3);
INSTANTIATE_NBIT_FLOAT_HALF(4);
INSTANTIATE_NBIT_FLOAT_HALF(5);
INSTANTIATE_NBIT_FLOAT_HALF(6);
INSTANTIATE_NBIT_FLOAT_HALF(7);
INSTANTIATE_NBIT_FLOAT_HALF(8);

#if __METAL_VERSION__ >= 310
INSTANTIATE_NBIT_BFLOAT(1);
INSTANTIATE_NBIT_BFLOAT(2);
INSTANTIATE_NBIT_BFLOAT(3);
INSTANTIATE_NBIT_BFLOAT(4);
INSTANTIATE_NBIT_BFLOAT(5);
INSTANTIATE_NBIT_BFLOAT(6);
INSTANTIATE_NBIT_BFLOAT(7);
INSTANTIATE_NBIT_BFLOAT(8);
#endif
