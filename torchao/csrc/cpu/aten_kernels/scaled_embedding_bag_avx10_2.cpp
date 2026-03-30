// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
//
// AVX10.2 implementation of the scaled_embedding_bag kernel.
//
// Compiled with -march=diamondrapids, which defines __AVX10_2__.  When that
// macro is set, at::vec::CPU_CAPABILITY::cvtfp8e4m3_fp32 and
// cvtfp32_fp8e4m3 (defined in vec512_float8.h) use the native AVX10.2
// hardware instructions (_mm256_cvthf8_ph / _mm256_cvtph_hf8) instead of
// the multi-step AVX512 software emulation.  This file therefore provides
// faster fp8<->fp32 conversion on DMR (Diamond Rapids) and later CPUs.
//
// Only called at runtime when __builtin_cpu_supports("avx10.2") is true.

#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/vec512/vec512_float8.h>
#include <ATen/native/EmbeddingBag.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Unroll.h>
#include <cstdio>
#include <mutex>
#include <torch/all.h>
#include <immintrin.h>

// ---------------------------------------------------------------------------
// Type dispatch macros (mirrors scaled_embedding_bag.cpp)
// ---------------------------------------------------------------------------
#define QTYPE_DISPATCH(TYPE, ...)                                              \
  [&]() {                                                                      \
    switch (TYPE) {                                                            \
    case c10::ScalarType::Float8_e4m3fn: {                                     \
      using data_t = at::Float8_e4m3fn;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case c10::ScalarType::Char: {                                              \
      using data_t = int8_t;                                                   \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(false, "scaled_embeding_bag: unsupport qtype");              \
    }                                                                          \
  }()

#define OUTTYPE_DISPATCH(TYPE, ...)                                            \
  [&]() {                                                                      \
    switch (TYPE) {                                                            \
    case c10::ScalarType::Float: {                                             \
      using output_t = float;                                                  \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case c10::ScalarType::Char: {                                              \
      using output_t = int8_t;                                                 \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case c10::ScalarType::Float8_e4m3fn: {                                     \
      using output_t = at::Float8_e4m3fn;                                      \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(false, "scaled_embedding_bag: unsupported output type");     \
    }                                                                          \
  }()

namespace torchao {
namespace cpu_avx10_2 {

// ---------------------------------------------------------------------------
// CHUNK type and helpers
// With -march=diamondrapids, __AVX10_2__ is defined, so cvtfp8e4m3_fp32 and
// cvtfp32_fp8e4m3 use the hardware _mm256_cvthf8_ph / _mm256_cvtph_hf8 path.
// ---------------------------------------------------------------------------

using CHUNK = std::tuple<__m512, __m512, __m512, __m512,
                         __m512, __m512, __m512, __m512>;

static inline __m512 _mm512_load_e4m3_cvt_ps(const at::Float8_e4m3fn* x) {
  __m512 o;
  __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x));
  // Uses _mm256_cvthf8_ph hardware path because __AVX10_2__ is defined
  at::vec::CPU_CAPABILITY::cvtfp8e4m3_fp32(v, o);
  return o;
}

static inline __m512 _mm512_cvt_s8_ps(__m128i x) {
  return _mm512_cvt_roundepi32_ps(
      _mm512_cvtepi8_epi32(x),
      (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// load_chunk overloads ---------------------------------------------------------

static inline CHUNK load_chunk(const at::Float8_e4m3fn* x) {
  return {_mm512_load_e4m3_cvt_ps(x +   0),
          _mm512_load_e4m3_cvt_ps(x +  16),
          _mm512_load_e4m3_cvt_ps(x +  32),
          _mm512_load_e4m3_cvt_ps(x +  48),
          _mm512_load_e4m3_cvt_ps(x +  64),
          _mm512_load_e4m3_cvt_ps(x +  80),
          _mm512_load_e4m3_cvt_ps(x +  96),
          _mm512_load_e4m3_cvt_ps(x + 112)};
}

static inline CHUNK load_chunk(const int8_t* x) {
  __m512i x00 = _mm512_load_si512(x);
  __m512i x64 = _mm512_load_si512(x + 64);
  return {_mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x00, 0)),
          _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x00, 1)),
          _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x00, 2)),
          _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x00, 3)),
          _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x64, 0)),
          _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x64, 1)),
          _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x64, 2)),
          _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x64, 3))};
}

// store_chunk overloads -------------------------------------------------------

static inline void store_chunk(float* output, CHUNK chunk) {
  auto [x0, x1, x2, x3, x4, x5, x6, x7] = chunk;
  _mm512_store_ps(output +   0, x0);
  _mm512_store_ps(output +  16, x1);
  _mm512_store_ps(output +  32, x2);
  _mm512_store_ps(output +  48, x3);
  _mm512_store_ps(output +  64, x4);
  _mm512_store_ps(output +  80, x5);
  _mm512_store_ps(output +  96, x6);
  _mm512_store_ps(output + 112, x7);
}

static inline void store_chunk(int8_t* output, CHUNK chunk) {
  auto [f0, f1, f2, f3, f4, f5, f6, f7] = chunk;
  auto to_i32 = [](__m512 f) {
    return _mm512_cvt_roundps_epi32(
        f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  };
  __m512i x00 = _mm512_undefined_epi32();
  __m512i x64 = _mm512_undefined_epi32();
  x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(to_i32(f0)), 0);
  x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(to_i32(f1)), 1);
  x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(to_i32(f2)), 2);
  x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(to_i32(f3)), 3);
  x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(to_i32(f4)), 0);
  x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(to_i32(f5)), 1);
  x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(to_i32(f6)), 2);
  x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(to_i32(f7)), 3);
  _mm512_store_si512(output,      x00);
  _mm512_store_si512(output + 64, x64);
}

static inline void store_chunk(at::Float8_e4m3fn* output, CHUNK chunk) {
  auto [x0, x1, x2, x3, x4, x5, x6, x7] = chunk;
  // Uses _mm256_cvtph_hf8 hardware path because __AVX10_2__ is defined
  _mm_storeu_si128(reinterpret_cast<__m128i*>(output +   0),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x0));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(output +  16),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x1));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(output +  32),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x2));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(output +  48),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x3));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(output +  64),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x4));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(output +  80),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x5));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(output +  96),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x6));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 112),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x7));
}

static inline void store_elem(float& out, float input) { out = input; }

static inline void store_elem(int8_t& out, float input) {
  float rounded = std::round(input);
  float clamped = std::max(-128.0f, std::min(127.0f, rounded));
  out = static_cast<int8_t>(static_cast<int32_t>(clamped));
}

static inline void store_elem(at::Float8_e4m3fn& out, float input) {
  out = static_cast<at::Float8_e4m3fn>(input);
}

// ---------------------------------------------------------------------------
// Kernel: same algorithm as _krnl_avx512, but compiled for AVX10.2 so the
// fp8 load/store helpers use hardware instructions.
// ---------------------------------------------------------------------------

template <typename index_t, typename data_t, typename output_t>
static void _krnl(
    const int64_t bs_begin, const int64_t bs_end,
    const int64_t num_emb, const int64_t emb_dim,
    const index_t last_offset, const index_t* indices,
    const index_t* offsets, const data_t* weight,
    const double scale, output_t* result,
    const int64_t num_batch) {
  if (emb_dim % 128 == 0) {
    constexpr int64_t block_dim = 128;
    const int64_t num_blocks = emb_dim / block_dim;
    __m512 scale_v = _mm512_set1_ps(static_cast<float>(scale));
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m512 x0, x1, x2, x3, x4, x5, x6, x7;
      __m512 y0, y1, y2, y3, y4, y5, y6, y7;
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == num_batch && last_offset != -1)
                            ? last_offset
                            : offsets[b + 1];
      for (int64_t block_id = 0; block_id < num_blocks; block_id++) {
        int64_t idx = indices[start_idx] * emb_dim + block_dim * block_id;
        output_t* block_result = result + block_dim * block_id;
        std::tie(x0, x1, x2, x3, x4, x5, x6, x7) = load_chunk(weight + idx);
        for (int64_t j = start_idx + 1; j < end_idx; ++j) {
          idx = indices[j] * emb_dim + block_dim * block_id;
          std::tie(y0, y1, y2, y3, y4, y5, y6, y7) = load_chunk(weight + idx);
          x0 = _mm512_add_ps(x0, y0); x1 = _mm512_add_ps(x1, y1);
          x2 = _mm512_add_ps(x2, y2); x3 = _mm512_add_ps(x3, y3);
          x4 = _mm512_add_ps(x4, y4); x5 = _mm512_add_ps(x5, y5);
          x6 = _mm512_add_ps(x6, y6); x7 = _mm512_add_ps(x7, y7);
        }
        x0 = _mm512_mul_ps(x0, scale_v); x1 = _mm512_mul_ps(x1, scale_v);
        x2 = _mm512_mul_ps(x2, scale_v); x3 = _mm512_mul_ps(x3, scale_v);
        x4 = _mm512_mul_ps(x4, scale_v); x5 = _mm512_mul_ps(x5, scale_v);
        x6 = _mm512_mul_ps(x6, scale_v); x7 = _mm512_mul_ps(x7, scale_v);
        store_chunk(block_result, {x0, x1, x2, x3, x4, x5, x6, x7});
      }
      result += num_emb * emb_dim;
    }
    return;
  }
  // Scalar fallback for emb_dim not a multiple of 128
  for (int64_t b = bs_begin; b < bs_end; ++b) {
    int64_t start_idx = offsets[b];
    int64_t end_idx = ((b + 1) == num_batch && last_offset != -1)
                          ? last_offset
                          : offsets[b + 1];
    for (int64_t d = 0; d < emb_dim; d++) {
      int64_t idx = indices[start_idx] * emb_dim;
      float value = float(weight[idx + d]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        idx = indices[j] * emb_dim;
        value += float(weight[idx + d]);
      }
      value = value * scale;
      store_elem(result[d], value);
    }
    result += num_emb * emb_dim;
  }
}

template <typename index_t, typename data_t, typename output_t>
static void _run(
    void* o_ptr, void* w_ptr, void* indices_ptr, void* offsets_ptr,
    int64_t num_batch, int64_t emb_dim, int64_t last_offset,
    double w_scale) {
  constexpr int64_t b_block = 512;
  const int64_t n_b_blocks = (num_batch - 1) / b_block + 1;
  const int64_t num_emb = 1;
#pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < n_b_blocks; ++b) {
    for (int64_t n = 0; n < num_emb; ++n) {
      const int64_t bs_begin = b * b_block;
      const int64_t bs_end = std::min(num_batch, (b + 1) * b_block);
      output_t* r = reinterpret_cast<output_t*>(o_ptr) +
                    b * b_block * num_emb * emb_dim + n * emb_dim;
      _krnl<index_t, data_t, output_t>(
          bs_begin, bs_end, num_emb, emb_dim,
          static_cast<index_t>(last_offset),
          reinterpret_cast<const index_t*>(indices_ptr),
          reinterpret_cast<const index_t*>(offsets_ptr),
          reinterpret_cast<const data_t*>(w_ptr),
          w_scale, r, num_batch);
    }
  }
}

// ---------------------------------------------------------------------------
// Public entry point — called from scaled_embedding_bag.cpp when
// __builtin_cpu_supports("avx10.2") is true.
// ---------------------------------------------------------------------------
at::Tensor _scaled_embedding_bag_avx10_2(
    const at::Tensor& qweight, const at::Tensor& indices,
    const at::Tensor& offsets, const at::Tensor& w_scales, double o_scale,
    const int64_t mode, bool include_last_offset,
    at::ScalarType output_dtype) {
  static std::once_flag _flag;
  std::call_once(_flag, []() {
    fprintf(stderr, "[torchao] scaled_embedding_bag: AVX10.2 path selected "
            "(hardware fp8 conversion)\n");
  });

  int64_t batch_size =
      include_last_offset ? offsets.size(0) - 1 : offsets.size(0);
  int64_t emb_dim = qweight.size(1);
  float w_scale = w_scales.data_ptr<float>()[0];
  w_scale /= static_cast<float>(o_scale);

  int64_t last_offset = indices.numel();

  at::Tensor output =
      at::empty({batch_size, emb_dim}, qweight.options().dtype(output_dtype));

  auto qtype = qweight.scalar_type();
  auto index_type = indices.scalar_type();

  OUTTYPE_DISPATCH(output_dtype, [&] {
    QTYPE_DISPATCH(qtype, [&] {
      AT_DISPATCH_INDEX_TYPES(index_type, "_scaled_embedding_bag_avx10_2", [&] {
        _run<index_t, data_t, output_t>(
            output.data_ptr(), qweight.data_ptr(),
            indices.data_ptr(), offsets.data_ptr(),
            batch_size, emb_dim, last_offset, w_scale);
      });
    });
  });
  return output;
}

} // namespace cpu_avx10_2
} // namespace torchao
