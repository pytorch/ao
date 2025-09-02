#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/vec512/vec512_float8.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/EmbeddingBag.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Unroll.h>
#include <torch/all.h>

namespace torchao {

namespace {

#if defined(CPU_CAPABILITY_AVX512)
static inline __m512 _mm512_load_e4m3_cvt_ps(const at::Float8_e4m3fn *x) {
  __m512 o;
  __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x));
  at::vec::CPU_CAPABILITY::cvtfp8e4m3_fp32(v, o);
  return o;
}
#endif

template <typename index_t>
inline void _scaled_embedding_bag_krnl(
    const int64_t bs_begin, const int64_t bs_end, const int64_t num_emb,
    const int64_t emb_dim, const index_t last_offset, const index_t *indices,
    const index_t *offsets, const at::Float8_e4m3fn *weight, const double scale,
    float *result, const int64_t num_batch) {
#if defined(CPU_CAPABILITY_AVX512)
  if (emb_dim % 128 == 0) {
    constexpr int64_t block_dim = 128;
    const int64_t num_blocks = emb_dim / block_dim;
    __m512 scale_v = _mm512_set1_ps(scale);
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m512 x0, x1, x2, x3, x4, x5, x6, x7;
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == num_batch && last_offset != -1)
                            ? last_offset
                            : offsets[b + 1];
      for (int64_t block_id = 0; block_id < num_blocks; block_id++) {
        // load first indices
        int64_t idx = indices[start_idx] * emb_dim + block_dim * block_id;
        float *block_result = result + block_dim * block_id;
        x0 = _mm512_load_e4m3_cvt_ps(&weight[idx]);
        x1 = _mm512_load_e4m3_cvt_ps(&weight[idx + 16]);
        x2 = _mm512_load_e4m3_cvt_ps(&weight[idx + 32]);
        x3 = _mm512_load_e4m3_cvt_ps(&weight[idx + 48]);
        x4 = _mm512_load_e4m3_cvt_ps(&weight[idx + 64]);
        x5 = _mm512_load_e4m3_cvt_ps(&weight[idx + 80]);
        x6 = _mm512_load_e4m3_cvt_ps(&weight[idx + 96]);
        x7 = _mm512_load_e4m3_cvt_ps(&weight[idx + 112]);
        for (int64_t j = start_idx + 1; j < end_idx; ++j) {
          // add following idx
          idx = indices[j] * emb_dim + block_dim * block_id;
          x0 = _mm512_add_ps(x0, _mm512_load_e4m3_cvt_ps(&weight[idx]));
          x1 = _mm512_add_ps(x1, _mm512_load_e4m3_cvt_ps(&weight[idx + 16]));
          x2 = _mm512_add_ps(x2, _mm512_load_e4m3_cvt_ps(&weight[idx + 32]));
          x3 = _mm512_add_ps(x3, _mm512_load_e4m3_cvt_ps(&weight[idx + 48]));
          x4 = _mm512_add_ps(x4, _mm512_load_e4m3_cvt_ps(&weight[idx + 64]));
          x5 = _mm512_add_ps(x5, _mm512_load_e4m3_cvt_ps(&weight[idx + 80]));
          x6 = _mm512_add_ps(x6, _mm512_load_e4m3_cvt_ps(&weight[idx + 96]));
          x7 = _mm512_add_ps(x7, _mm512_load_e4m3_cvt_ps(&weight[idx + 112]));
        }
        x0 = _mm512_mul_ps(x0, scale_v);
        x1 = _mm512_mul_ps(x1, scale_v);
        x2 = _mm512_mul_ps(x2, scale_v);
        x3 = _mm512_mul_ps(x3, scale_v);
        x4 = _mm512_mul_ps(x4, scale_v);
        x5 = _mm512_mul_ps(x5, scale_v);
        x6 = _mm512_mul_ps(x6, scale_v);
        x7 = _mm512_mul_ps(x7, scale_v);
        // store
        _mm512_store_ps(block_result, x0);
        _mm512_store_ps(block_result + 16, x1);
        _mm512_store_ps(block_result + 32, x2);
        _mm512_store_ps(block_result + 48, x3);
        _mm512_store_ps(block_result + 64, x4);
        _mm512_store_ps(block_result + 80, x5);
        _mm512_store_ps(block_result + 96, x6);
        _mm512_store_ps(block_result + 112, x7);
      }
      result += num_emb * emb_dim;
    }
    return;
  }
#endif
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
      result[d] = value;
    }
    result += num_emb * emb_dim;
  }
}

template <typename index_t>
inline void _scaled_embedding_bag_krnl(
    const int64_t bs_begin, const int64_t bs_end, const int64_t num_emb,
    const int64_t emb_dim, const index_t last_offset, const index_t *indices,
    const index_t *offsets, const int8_t *weight, const double scale,
    float *result, const int64_t num_batch) {
#if defined(CPU_CAPABILITY_AVX512)
  if (emb_dim % 128 == 0) {
    constexpr int64_t block_dim = 128;
    const int64_t num_blocks = emb_dim / block_dim;
    __m512 scale_v = _mm512_set1_ps(scale);
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m512i x00, x64;
      __m512i y0, y1, y2, y3, y4, y5, y6, y7;
      __m512 f0, f1, f2, f3, f4, f5, f6, f7;
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == num_batch && last_offset != -1)
                            ? last_offset
                            : offsets[b + 1];
      for (int64_t block_id = 0; block_id < num_blocks; block_id++) {
        // load first indices
        int64_t idx = indices[start_idx] * emb_dim + block_dim * block_id;
            float *block_result = result + block_dim * block_id;
        x00 = _mm512_load_si512(&weight[idx]);
        x64 = _mm512_load_si512(&weight[idx + 64]);
        y0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 0));
        y1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 1));
        y2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 2));
        y3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 3));
        y4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 0));
        y5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 1));
        y6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 2));
        y7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 3));
        for (int64_t j = start_idx + 1; j < end_idx; ++j) {
            // add following idx
            idx = indices[j] * emb_dim + block_dim * block_id;
            x00 = _mm512_load_si512(&weight[idx]);
            x64 = _mm512_load_si512(&weight[idx + 64]);
            y0 = _mm512_add_epi32(
                y0, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 0)));
            y1 = _mm512_add_epi32(
                y1, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 1)));
            y2 = _mm512_add_epi32(
                y2, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 2)));
            y3 = _mm512_add_epi32(
                y3, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 3)));
            y4 = _mm512_add_epi32(
                y4, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 0)));
            y5 = _mm512_add_epi32(
                y5, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 1)));
            y6 = _mm512_add_epi32(
                y6, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 2)));
            y7 = _mm512_add_epi32(
                y7, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 3)));
        }
        f0 = _mm512_cvt_roundepi32_ps(
            y0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        f1 = _mm512_cvt_roundepi32_ps(
            y1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        f2 = _mm512_cvt_roundepi32_ps(
            y2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        f3 = _mm512_cvt_roundepi32_ps(
            y3, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        f4 = _mm512_cvt_roundepi32_ps(
            y4, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        f5 = _mm512_cvt_roundepi32_ps(
            y5, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        f6 = _mm512_cvt_roundepi32_ps(
            y6, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        f7 = _mm512_cvt_roundepi32_ps(
            y7, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        f0 = _mm512_mul_ps(f0, scale_v);
        f1 = _mm512_mul_ps(f1, scale_v);
        f2 = _mm512_mul_ps(f2, scale_v);
        f3 = _mm512_mul_ps(f3, scale_v);
        f4 = _mm512_mul_ps(f4, scale_v);
        f5 = _mm512_mul_ps(f5, scale_v);
        f6 = _mm512_mul_ps(f6, scale_v);
        f7 = _mm512_mul_ps(f7, scale_v);
        // store
        _mm512_store_ps(block_result, f0);
        _mm512_store_ps(block_result + 16, f1);
        _mm512_store_ps(block_result + 32, f2);
        _mm512_store_ps(block_result + 48, f3);
        _mm512_store_ps(block_result + 64, f4);
        _mm512_store_ps(block_result + 80, f5);
        _mm512_store_ps(block_result + 96, f6);
        _mm512_store_ps(block_result + 112, f7);
      }
      result += num_emb * emb_dim;
    }
    return;
  }
#endif
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
      result[d] = value;
    }
    result += num_emb * emb_dim;
  }
}

template <typename index_t, typename data_t>
void _scaled_embedding_bag(float *o_ptr, data_t *w_ptr, index_t *indices_ptr,
                           index_t *offsets_ptr, int64_t num_batch,
                           int64_t emb_dim, index_t last_offset, double w_scale,
                           double o_scale) {
  constexpr int64_t b_block = 512;
  const int64_t n_b_blocks = (num_batch - 1) / b_block + 1;
  w_scale /= o_scale;
  const int64_t num_emb = 1;
#pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < n_b_blocks; ++b) {
    for (int64_t n = 0; n < num_emb; ++n) {
      const int64_t bs_begin = b * b_block;
      const int64_t bs_end = std::min(num_batch, (b + 1) * b_block);
      float *r = &o_ptr[b * b_block * num_emb * emb_dim + n * emb_dim];
      // avoid offsets not include last batch
      _scaled_embedding_bag_krnl(bs_begin, bs_end, num_emb, emb_dim,
                                 last_offset, indices_ptr, offsets_ptr, w_ptr,
                                 w_scale, r, num_batch);
    }
  }
}

at::Tensor _scaled_embedding_bag_impl(const at::Tensor &qweight,
                                      const at::Tensor &indices,
                                      const at::Tensor &offsets,
                                      const at::Tensor &w_scales,
                                      double o_scale, const int64_t mode,
                                      bool include_last_offset) {
  // Only support include_last_offset == True and mode ==
  // at::native::EmbeddingBagMode::SUM
  // TODO: Support more case
  TORCH_CHECK(include_last_offset,
              "_scaled_embedding_bag: only suppport include_last_offset");
  TORCH_CHECK(mode == at::native::EmbeddingBagMode::SUM,
              "_scaled_embedding_bag: only suppport sum mode");
  int64_t batch_size =
      include_last_offset ? offsets.size(0) - 1 : offsets.size(0);
  int64_t emb_dim = qweight.size(1);

  auto index_type = indices.scalar_type();
  auto qtype = qweight.scalar_type();
  float w_scale = w_scales.data_ptr<float>()[0];

  TORCH_CHECK(indices.is_contiguous() && offsets.is_contiguous(),
              "_scaled_embedding_bag: only accept contiguous input");
  TORCH_CHECK(
      offsets.scalar_type() == index_type,
      "_scaled_embedding_bag: index and offset must be of the same type");
  TORCH_CHECK(qweight.is_contiguous(),
              "_scaled_embedding_bag: only accept contiguous weight");
  TORCH_CHECK(qweight.dim() == 2,
              "_scaled_embedding_bag: only accept weight with dim == 2");
  TORCH_CHECK(qweight.scalar_type() == c10::ScalarType::Float8_e4m3fn || qweight.scalar_type() == c10::ScalarType::Char,
              "_scaled_embedding_bag: only support e4m3fn and int8 weight")
  // handle last offsets
  int64_t last_offset = indices.numel();

  at::Tensor output =
      at::empty({batch_size, emb_dim}, qweight.options().dtype(at::kFloat));
  if (qweight.scalar_type() == c10::ScalarType::Float8_e4m3fn) {
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_scaled_embedding_bag", [&] {
      at::Float8_e4m3fn *qweight_ptr = qweight.data_ptr<at::Float8_e4m3fn>();
      index_t *indices_ptr = indices.data_ptr<index_t>();
      index_t *offsets_ptr = offsets.data_ptr<index_t>();
      float *output_ptr = output.data_ptr<float>();
      _scaled_embedding_bag<index_t, at::Float8_e4m3fn>(
          output_ptr, qweight_ptr, indices_ptr, offsets_ptr, batch_size, emb_dim,
          last_offset, w_scale, o_scale);
    });
  } else {
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_scaled_embedding_bag", [&] {
      int8_t *qweight_ptr = qweight.data_ptr<int8_t>();
      index_t *indices_ptr = indices.data_ptr<index_t>();
      index_t *offsets_ptr = offsets.data_ptr<index_t>();
      float *output_ptr = output.data_ptr<float>();
      _scaled_embedding_bag<index_t, int8_t>(
          output_ptr, qweight_ptr, indices_ptr, offsets_ptr, batch_size, emb_dim,
          last_offset, w_scale, o_scale);
    });
  }

  return output;
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::_scaled_embedding_bag", &_scaled_embedding_bag_impl);
}

} // namespace torchao
