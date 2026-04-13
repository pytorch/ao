#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/vec512/vec512_float8.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/EmbeddingBag.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Unroll.h>
#include <torch/all.h>
#include "utils.h"

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

// =============================================================================
// The AVX10.2 variant of this file is compiled as a temp copy with:
//   -DCPU_CAPABILITY_AVX10_2 -march=diamondrapids
// When __AVX10_2__ is set by -march=diamondrapids, the PyTorch helpers
// cvtfp8e4m3_fp32 / cvtfp32_fp8e4m3 (vec512_float8.h) use the native
// hardware instructions _mm256_cvthf8_ph / _mm256_cvtph_hf8 instead of the
// multi-step AVX512 software emulation. All other kernel logic is identical.
// =============================================================================

// Forward-declare the AVX10.2 entry point so the runtime dispatcher can call
// it when __builtin_cpu_supports("avx10.2") is true. Only needed in the
// default (non-AVX10.2) build; in the AVX10.2 temp copy this TU defines it.
#ifndef CPU_CAPABILITY_AVX10_2
namespace torchao {
namespace cpu_avx10_2 {
at::Tensor _scaled_embedding_bag_avx10_2(
    const at::Tensor& qweight, const at::Tensor& indices,
    const at::Tensor& offsets, const at::Tensor& w_scales, double o_scale,
    int64_t mode, bool include_last_offset, at::ScalarType output_dtype);
} // namespace cpu_avx10_2
} // namespace torchao
#endif

// All kernel code is compiled with AVX512 enabled. When compiled with
// -march=diamondrapids (-DCPU_CAPABILITY_AVX10_2), these flags are a subset
// of what the target already provides, so the pragma is harmless.
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,amx-int8,amx-tile,amx-bf16")
#pragma GCC optimize("O3,tree-vectorize")
#include <immintrin.h>

namespace torchao {

// In the AVX10.2 temp copy, emit into the cpu_avx10_2 namespace so the linker
// sees a distinct symbol from the main build.
#ifdef CPU_CAPABILITY_AVX10_2
namespace cpu_avx10_2 {
#else
namespace { // anonymous namespace for internal linkage in the default build
#endif

using CHUNK = std::tuple<__m512, __m512, __m512, __m512,
                         __m512, __m512, __m512, __m512>;

static inline __m512 _mm512_load_e4m3_cvt_ps(const at::Float8_e4m3fn *x) {
  __m512 o;
  __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x));
  at::vec::CPU_CAPABILITY::cvtfp8e4m3_fp32(v, o);
  return o;
}

static inline __m512 _mm512_cvt_s8_ps(__m128i x) {
  return _mm512_cvt_roundepi32_ps(
      _mm512_cvtepi8_epi32(x), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

static inline CHUNK load_chunk(const at::Float8_e4m3fn *x) {
  return {_mm512_load_e4m3_cvt_ps(x +   0), _mm512_load_e4m3_cvt_ps(x +  16),
          _mm512_load_e4m3_cvt_ps(x +  32), _mm512_load_e4m3_cvt_ps(x +  48),
          _mm512_load_e4m3_cvt_ps(x +  64), _mm512_load_e4m3_cvt_ps(x +  80),
          _mm512_load_e4m3_cvt_ps(x +  96), _mm512_load_e4m3_cvt_ps(x + 112)};
}

static inline CHUNK load_chunk(const int8_t *x) {
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

static inline void store_chunk(float *output, CHUNK chunk) {
  auto [x0, x1, x2, x3, x4, x5, x6, x7] = chunk;
  _mm512_store_ps(output +   0, x0); _mm512_store_ps(output +  16, x1);
  _mm512_store_ps(output +  32, x2); _mm512_store_ps(output +  48, x3);
  _mm512_store_ps(output +  64, x4); _mm512_store_ps(output +  80, x5);
  _mm512_store_ps(output +  96, x6); _mm512_store_ps(output + 112, x7);
}

static inline void store_chunk(int8_t *output, CHUNK chunk) {
  auto [f0, f1, f2, f3, f4, f5, f6, f7] = chunk;
  auto cvt = [](__m512 f) {
    return _mm512_cvtsepi32_epi8(_mm512_cvt_roundps_epi32(
        f, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
  };
  __m512i x00 = _mm512_undefined_epi32(), x64 = _mm512_undefined_epi32();
  x00 = _mm512_inserti32x4(x00, cvt(f0), 0); x00 = _mm512_inserti32x4(x00, cvt(f1), 1);
  x00 = _mm512_inserti32x4(x00, cvt(f2), 2); x00 = _mm512_inserti32x4(x00, cvt(f3), 3);
  x64 = _mm512_inserti32x4(x64, cvt(f4), 0); x64 = _mm512_inserti32x4(x64, cvt(f5), 1);
  x64 = _mm512_inserti32x4(x64, cvt(f6), 2); x64 = _mm512_inserti32x4(x64, cvt(f7), 3);
  _mm512_store_si512(output,      x00);
  _mm512_store_si512(output + 64, x64);
}

static inline void store_chunk(at::Float8_e4m3fn *output, CHUNK chunk) {
  auto [x0, x1, x2, x3, x4, x5, x6, x7] = chunk;
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output +   0), at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x0));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output +  16), at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x1));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output +  32), at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x2));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output +  48), at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x3));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output +  64), at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x4));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output +  80), at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x5));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output +  96), at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x6));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 112), at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x7));
}

// Prefetch all cache lines of an embedding row to hide DRAM latency.
template <typename data_t>
static inline void _prefetch_emb_row(const data_t *base, int64_t emb_dim) {
  const char *ptr = reinterpret_cast<const char *>(base);
  const int64_t emb_bytes = emb_dim * static_cast<int64_t>(sizeof(data_t));
  for (int64_t off = 0; off < emb_bytes; off += 64)
    _mm_prefetch(ptr + off, _MM_HINT_T0);
}

static inline void store_elem(float &out, float input) { out = input; }
static inline void store_elem(int8_t &out, float input) {
  out = static_cast<int8_t>(static_cast<int32_t>(
      std::max(-128.0f, std::min(127.0f, std::round(input)))));
}
static inline void store_elem(at::Float8_e4m3fn &out, float input) {
  out = static_cast<at::Float8_e4m3fn>(input);
}

template <typename index_t, typename data_t, typename output_t>
static void _krnl(
    int64_t bs_begin, int64_t bs_end, int64_t num_emb, int64_t emb_dim,
    index_t last_offset, const index_t *indices, const index_t *offsets,
    const data_t *weight, double scale, output_t *result, int64_t num_batch) {
  // How many batch entries ahead to prefetch to overlap DRAM latency with compute.
  constexpr int64_t PREFETCH_DIST = 8;
  if (kHasAVX512 && emb_dim % 128 == 0) {
    constexpr int64_t block_dim = 128;
    const int64_t num_blocks = emb_dim / block_dim;
    __m512 scale_v = _mm512_set1_ps(scale);
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      // Software prefetch for batch entries ahead to overlap DRAM latency.
      const int64_t pref_b = b + PREFETCH_DIST;
      if (pref_b < bs_end) {
        const int64_t pref_start = offsets[pref_b];
        const int64_t pref_end = (pref_b + 1 == num_batch && last_offset != -1)
                                     ? last_offset : offsets[pref_b + 1];
        for (int64_t pj = pref_start; pj < pref_end; ++pj)
          _prefetch_emb_row(weight + indices[pj] * emb_dim, emb_dim);
      }
      __m512 x0, x1, x2, x3, x4, x5, x6, x7;
      __m512 y0, y1, y2, y3, y4, y5, y6, y7;
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == num_batch && last_offset != -1)
                            ? last_offset : offsets[b + 1];
      for (int64_t block_id = 0; block_id < num_blocks; ++block_id) {
        int64_t idx = indices[start_idx] * emb_dim + block_dim * block_id;
        output_t *block_result = result + block_dim * block_id;
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
  for (int64_t b = bs_begin; b < bs_end; ++b) {
    int64_t start_idx = offsets[b];
    int64_t end_idx = ((b + 1) == num_batch && last_offset != -1)
                          ? last_offset : offsets[b + 1];
    for (int64_t d = 0; d < emb_dim; ++d) {
      float value = float(weight[indices[start_idx] * emb_dim + d]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j)
        value += float(weight[indices[j] * emb_dim + d]);
      store_elem(result[d], value * scale);
    }
    result += num_emb * emb_dim;
  }
}

template <typename index_t, typename data_t, typename output_t>
static void _run(
    at::Tensor& output, const at::Tensor& qweight,
    const at::Tensor& indices, const at::Tensor& offsets,
    float w_scale, int64_t batch_size, int64_t emb_dim, int64_t last_offset) {
  constexpr int64_t b_block = 512;
  const int64_t n_b_blocks = (batch_size - 1) / b_block + 1;
#pragma omp parallel for
  for (int64_t b = 0; b < n_b_blocks; ++b) {
    int64_t bs_begin = b * b_block;
    int64_t bs_end = std::min(batch_size, (b + 1) * b_block);
    output_t *r = output.data_ptr<output_t>() + b * b_block * emb_dim;
    _krnl<index_t, data_t, output_t>(
        bs_begin, bs_end, 1, emb_dim, static_cast<index_t>(last_offset),
        indices.data_ptr<index_t>(), offsets.data_ptr<index_t>(),
        qweight.data_ptr<data_t>(), w_scale, r, batch_size);
  }
}

// Entry-point function. Name and namespace differ by compile variant:
//   default build  → torchao::{anonymous}::_scaled_embedding_bag_impl
//   AVX10.2 copy   → torchao::cpu_avx10_2::_scaled_embedding_bag_avx10_2
#ifdef CPU_CAPABILITY_AVX10_2
at::Tensor _scaled_embedding_bag_avx10_2(
#else
at::Tensor _scaled_embedding_bag_impl(
#endif
    const at::Tensor& qweight, const at::Tensor& indices,
    const at::Tensor& offsets, const at::Tensor& w_scales, double o_scale,
    int64_t mode, bool include_last_offset, at::ScalarType output_dtype) {
#ifndef CPU_CAPABILITY_AVX10_2
  // Runtime dispatch to hardware fp8 path when running on AVX10.2 CPU.
#if __GNUC__ >= 15
  if (__builtin_cpu_supports("avx10.2")) {
    return cpu_avx10_2::_scaled_embedding_bag_avx10_2(
        qweight, indices, offsets, w_scales, o_scale,
        mode, include_last_offset, output_dtype);
  }
#endif
  TORCH_CHECK(include_last_offset,
              "_scaled_embedding_bag: only suppport include_last_offset");
  TORCH_CHECK(mode == at::native::EmbeddingBagMode::SUM,
              "_scaled_embedding_bag: only suppport sum mode");
  TORCH_CHECK(indices.is_contiguous() && offsets.is_contiguous(),
              "_scaled_embedding_bag: only accept contiguous input");
  TORCH_CHECK(offsets.scalar_type() == indices.scalar_type(),
              "_scaled_embedding_bag: index and offset must be of the same type");
  TORCH_CHECK(qweight.is_contiguous(),
              "_scaled_embedding_bag: only accept contiguous weight");
  TORCH_CHECK(qweight.dim() == 2,
              "_scaled_embedding_bag: only accept weight with dim == 2");
  TORCH_CHECK(qweight.scalar_type() == c10::ScalarType::Float8_e4m3fn ||
              qweight.scalar_type() == c10::ScalarType::Char,
              "_scaled_embedding_bag: only support e4m3fn and int8 weight");
#endif
  int64_t batch_size = include_last_offset ? offsets.size(0) - 1 : offsets.size(0);
  int64_t emb_dim = qweight.size(1);
  float w_scale = w_scales.data_ptr<float>()[0] / static_cast<float>(o_scale);
  int64_t last_offset = indices.numel();
  at::Tensor output = at::empty({batch_size, emb_dim},
                                qweight.options().dtype(output_dtype));
  OUTTYPE_DISPATCH(output_dtype, [&] {
    QTYPE_DISPATCH(qweight.scalar_type(), [&] {
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_scaled_embedding_bag", [&] {
        _run<index_t, data_t, output_t>(
            output, qweight, indices, offsets, w_scale, batch_size, emb_dim,
            last_offset);
      });
    });
  });
  return output;
}

#ifdef CPU_CAPABILITY_AVX10_2
} // namespace cpu_avx10_2
#else
} // anonymous namespace
#endif

#pragma GCC pop_options

#ifndef CPU_CAPABILITY_AVX10_2
TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::_scaled_embedding_bag", &_scaled_embedding_bag_impl);
}
#endif

} // namespace torchao
