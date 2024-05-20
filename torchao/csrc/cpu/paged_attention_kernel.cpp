#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <ATen/AccumulateType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <ATen/Tensor.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <limits>
#include <omp.h>

namespace torchao {

namespace {

template <typename scalar_t, typename accum_t>
void reduce_head(
    const scalar_t* q_ptr_start,
    const scalar_t* k_cache_start,
    accum_t* attn_w_pos,
    int64_t head_size) {
  attn_w_pos[0] = 0;  
  for (long i = 0; i < head_size; i++) {
    attn_w_pos[0] += q_ptr_start[i] * k_cache_start[i];
  }
}

//BF16
template <>
void reduce_head<at::BFloat16, float>(
    const at::BFloat16* q_ptr_start,
    const at::BFloat16* k_cache_start,
    float* attn_w_pos,
    int64_t head_size) {
  attn_w_pos[0] = 0;
  using lpVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = lpVec::size();
  auto vec_tmp_sum = fVec(0.0f);
  for (long i = 0; i < vec_size * (head_size / vec_size); i += vec_size) {
    auto tmpq = lpVec::loadu(q_ptr_start + i);
    auto tmpk = lpVec::loadu(k_cache_start + i);
    fVec tmpq1, tmpq2, tmpk1, tmpk2;
    //convert to float 
    std::tie(tmpq1, tmpq2) = at::vec::convert_to_float(tmpq);
    std::tie(tmpk1, tmpk2) = at::vec::convert_to_float(tmpk);
    vec_tmp_sum = vec_tmp_sum + tmpq1 * tmpk1 + tmpq2 * tmpk2;
  }
  attn_w_pos[0] = at::vec::vec_reduce_all<>(
      [](fVec& x, fVec& y) {
        return x + y;
      },
  vec_tmp_sum);
}

template <typename scalar_t, typename accum_t>
inline void mul_attenion_weights_and_value_of_head(
    const accum_t& attn_w,
    const scalar_t* v_cache_start,
    accum_t* attn_out_start,
    int64_t head_size,
    bool accumulated) {
  for (auto hsi = 0; hsi < head_size; hsi++) {
    if (accumulated) {
      attn_out_start[hsi] += attn_w * (float)v_cache_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * (float)v_cache_start[hsi];
    }
  }
}

template <>
inline void mul_attenion_weights_and_value_of_head<at::BFloat16, float>(
    const float& attn_w,
    const at::BFloat16* v_cache_start,
    float* attn_out_start,
    int64_t head_size,
    bool accumulated) {
  using lpVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  auto lpVec_size = lpVec::size();
  auto fVec_size = fVec::size();
  auto vec_attn_w = fVec(attn_w);
  auto vec_tmp_sum = fVec(0.0f);
  long i = 0;
  for (; i < lpVec_size *(head_size/lpVec_size) ; i += lpVec_size) {
    auto tmpv = lpVec::loadu(v_cache_start + i);
    fVec tmpv1, tmpv2;
    //convert to float 
    std::tie(tmpv1, tmpv2) = at::vec::convert_to_float(tmpv);
    auto tmp1 = tmpv1 * vec_attn_w;
    auto tmp2 = tmpv2 * vec_attn_w;
    if (accumulated) {
      tmp1 = fVec::loadu(attn_out_start + i) + tmp1;
      tmp1.store(attn_out_start + i);
      tmp2 = fVec::loadu(attn_out_start + i + fVec_size) + tmp2;
      tmp2.store(attn_out_start + i + fVec_size);
    } else {
      tmp1.store(attn_out_start + i);
      tmp2.store(attn_out_start + i + fVec_size);
    }
  }
  for (; i < head_size; i++) {
    if (accumulated) {
      attn_out_start[i] += attn_w * (float)v_cache_start[i];
    } else {
      attn_out_start[i] = attn_w * (float)v_cache_start[i];
    }
  }
}

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
#if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

template <typename scalar_t>
inline void _mul_div_add_softmax(
    const scalar_t* a,
    const scalar_t& scale,
    const float* mask,
    const int& size,
    scalar_t* out) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  auto tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  long i = 0;
  // max(a * scale + mask)
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    if (mask != nullptr) {
      auto tmp_mask = at::vec::Vectorized<scalar_t>::loadu(mask + i);
      tmp1 = tmp1 + tmp_mask;
    }
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    tmp1.store(out + i);
  }
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    if (mask != nullptr) {
      tmp1 = tmp1 + mask[i];
    }
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  auto max = std::max(
      tmp_max,
      at::vec::vec_reduce_all<scalar_t>(
          [](at::vec::Vectorized<scalar_t>& x,
             at::vec::Vectorized<scalar_t>& y) {
            return at::vec::maximum(x, y);
          },
          vec_tmp_max));
  // exp and sum
  scalar_t sum = 0;
  auto max_vec = at::vec::Vectorized<scalar_t>(max);
  i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(out + i);
    auto tmp1 = tmp0 - max_vec;
    tmp1 = tmp1.exp_u20();
    sum += at::vec::vec_reduce_all<scalar_t>(
        [](at::vec::Vectorized<scalar_t>& x, at::vec::Vectorized<scalar_t>& y) {
          return x + y;
        },
        tmp1);
    tmp1.store(out + i);
  }
  for (; i < size; i++) {
    auto tmp0 = out[i];
    auto tmp1 = std::exp(tmp0 - max);
    sum += tmp1;
    out[i] = tmp1;
  }
  auto scale_vec = at::vec::Vectorized<scalar_t>(1.0f / sum);
  // normalize
  i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(out + i);
    auto tmp1 = tmp0 * scale_vec;
    tmp1.store(out + i);
  }
  for (; i < size; i++) {
    out[i] = out[i] * (1.0f / sum);
  }
}

void reshape_attn_mask_to_4d(
    at::Tensor& attn_mask,
    int64_t batchSize,
    int64_t num_head,
    int64_t qSize,
    int64_t kvSize) {
  // Support mask shapes:
  // 2d: ({Q_seq_len, 1}  x {KV_seq_len, 1})
  // 4d: ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})
  // Guaranteed in check_attn_mask_shape
  int64_t attn_mask_size_0 = 1;
  int64_t attn_mask_size_1 = 1;
  if (attn_mask.dim() == 4) {
    if (attn_mask.size(0) == batchSize) {
      attn_mask_size_0 = batchSize;
    }
    if (attn_mask.size(1) == num_head) {
      attn_mask_size_1 = num_head;
    }
  }
  attn_mask = attn_mask
                  .view(
                      {attn_mask_size_0,
                       attn_mask_size_1,
                       attn_mask.size(-2),
                       attn_mask.size(-1)})
                  .expand({attn_mask_size_0, attn_mask_size_1, qSize, kvSize});
}

/**
 * Performs scale-dot-product for the next token based on cached key-value
 * attention.
 *
 * This function computes the attention weights and applies the attention
 * mechanism to obtain the final output. It takes in tensors representing the
 * query, key cache, value cache, head mapping, scale, block tables, context
 * lengths, block size, max context length, and optional alibi slopes. The
 * output tensor is updated with the computed attention values.
 *
 * @param out           Output tensor [num_seqs, 1, num_heads, head_size].
 * @param query         Query tensor [num_seqs, 1, num_heads, head_size].
 * @param key_cache     The pre-allocated buffer to store the key cache. The
 * shape should be [num_blocks, block_size, num_heads, head_size].
 * @param value_cache   The pre-allocated buffer to store the value cache. The
 * shape should be [num_blocks, block_size, num_heads, head_size].
 * @param head_mapping  Head mapping tensor [num_heads]. The mapping from the
 * query head to the kv head to support GQA/MQA. The shape should be the number
 * of query heads.
 * @param scale         Scaling factor for attention weights. In general, it is:
 * float(1.0 / (head_size ** 0.5)).
 * @param block_tables  Block tables tensor [num_seqs, max_num_blocks_per_seq].
 * @param context_lens  Context lengths tensor [num_seqs].
 * @param block_size    The block size which means the number of token in every
 * block.
 * @param max_context_len Maximum context length.
 * @param attn_mask  Optional tensor of alibi slopes with the shape of
 * (num_heads).
 */
template <typename scalar_t>
void paged_attention_kernel(
    at::Tensor& out,
    at::Tensor& query,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& head_mapping,
    const double scale,
    at::Tensor& block_tables,
    at::Tensor& context_lens,
    int64_t block_size,
    c10::optional<at::Tensor> attn_mask) {
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  auto num_seqs = query.size(0);
  auto query_size = query.size(1);
  auto num_heads = query.size(2);
  auto head_size = query.size(3);
  auto num_kv_heads = key_cache.size(2);
  auto max_num_blocks_per_seq = block_tables.size(1);
  auto max_context_len = context_lens.max().item<int64_t>();

  bool has_attn_mask = attn_mask.has_value() && attn_mask.value().numel();
  if (has_attn_mask) {
    attn_mask.value() = attn_mask.value().to(at::kFloat);
    reshape_attn_mask_to_4d(
        attn_mask.value(), num_seqs, num_heads, query_size, attn_mask.value().size(-1));
  }

  auto attn_weights = at::empty(
      {num_seqs, num_heads, max_context_len},
      query.options().dtype(accumulate_dtype));

  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_cache_ptr = key_cache.data_ptr<scalar_t>();
  auto value_cache_ptr = value_cache.data_ptr<scalar_t>();
  auto head_mapping_ptr = head_mapping.data_ptr<int>();
  auto block_tables_ptr = block_tables.data_ptr<int>();
  auto context_lens_ptr = context_lens.data_ptr<int>();
  auto attn_mask_ptr =
      attn_mask.has_value() ? attn_mask.value().data_ptr<float>() : nullptr;

  auto attn_weights_ptr = attn_weights.data_ptr<accum_t>();
  auto kv_block_strideB = key_cache.stride(0);
  auto q_stride = query.stride(0);
  auto attn_weights_strideB = attn_weights.stride(0);
  int64_t mStrideB = (has_attn_mask && attn_mask.value().size(0) > 1)
      ? attn_mask.value().stride(0)
      : 0;
  int64_t mStrideH = (has_attn_mask && attn_mask.value().size(1) > 1)
      ? attn_mask.value().stride(1)
      : 0;
  int64_t mStrideM = has_attn_mask ? attn_mask.value().stride(2) : 0;

#pragma omp parallel for collapse(3)
for (auto token_id = 0; token_id < max_context_len; token_id++) {
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {      
        auto context_len = context_lens_ptr[seq_id];
        if (token_id >= context_len)
          continue;
        auto attn_w_pos = attn_weights_ptr + seq_id * attn_weights_strideB +
            head_id * max_context_len + token_id;
        auto q_ptr_start = query_ptr + seq_id * q_stride + head_id * head_size;
        auto block_id = block_tables_ptr
            [seq_id * max_num_blocks_per_seq + token_id / block_size];
        auto block_offset = token_id % block_size;
        auto k_cache_start = key_cache_ptr + block_id * kv_block_strideB +
            block_offset * num_kv_heads * head_size +
            head_mapping_ptr[head_id] * head_size;
        reduce_head<scalar_t, accum_t>(
            q_ptr_start, k_cache_start, attn_w_pos, head_size);
      }
    }
  }

#pragma omp parallel for collapse(2)
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {
      auto max_val = -10000.0f;
      float sum = 0.0f;
      auto qid = 0;
      auto context_len = context_lens_ptr[seq_id];
      auto attn_w_start = attn_weights_ptr + seq_id * attn_weights_strideB +
          head_id * max_context_len;
      auto head_mask_start = has_attn_mask
          ? attn_mask_ptr + mStrideB * seq_id + mStrideH * head_id
          : nullptr;
      _mul_div_add_softmax<accum_t>(
          attn_w_start, scale, head_mask_start, context_len, attn_w_start);
    }
  }
  auto thread_numbers = omp_get_max_threads();
  auto private_attn_outs = at::empty(
      {thread_numbers, num_seqs, num_heads, head_size}, accumulate_dtype);
  auto private_attn_out_flag =
      at::zeros({thread_numbers, num_seqs, num_heads}, at::kByte);

  auto flag_access = private_attn_out_flag.accessor<uint8_t, 3>();
  auto private_attn_out_ptr = private_attn_outs.data_ptr<accum_t>();
  auto private_attn_out_stride = private_attn_outs.stride(0);
// mul and accumulate
#pragma omp parallel for collapse(3)
for (auto token_id = 0; token_id < max_context_len; token_id++) {
  for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {      
        auto context_len = context_lens_ptr[seq_id];
        auto thread_id = omp_get_thread_num();
        if (token_id >= context_len)
          continue;
        auto attn_w = attn_weights_ptr
            [seq_id * attn_weights_strideB + head_id * max_context_len +
             token_id];
        auto block_id = block_tables_ptr
            [seq_id * max_num_blocks_per_seq + token_id / block_size];
        auto block_offset = token_id % block_size;
        auto v_cache_start = value_cache_ptr + block_id * kv_block_strideB +
            block_offset * num_kv_heads * head_size +
            head_mapping_ptr[head_id] * head_size;
        auto attn_out_start = private_attn_out_ptr +
            thread_id * private_attn_out_stride + seq_id * q_stride +
            head_id * head_size;
        mul_attenion_weights_and_value_of_head<scalar_t, accum_t>(
            attn_w,
            v_cache_start,
            attn_out_start,
            head_size,
            flag_access[thread_id][seq_id][head_id]);
        if (flag_access[thread_id][seq_id][head_id] == 0) {
          flag_access[thread_id][seq_id][head_id] = 1;
        }
      } // for token_id
    } // for head_id
  } // for seq_id
  {
    RECORD_FUNCTION(
        "ipex::paged_attention::reduction_private_result",
        c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(2)
    for (auto seq_id = 0; seq_id < num_seqs; seq_id++) {
      for (auto hi = 0; hi < num_heads; hi++) {
        auto thr0_head_start =
            private_attn_out_ptr + (seq_id * num_heads + hi) * head_size;
        if (flag_access[0][seq_id][hi] == 0) {
          fill_stub<accum_t>(thr0_head_start, 0.0f, head_size);
        }
        for (auto thread_id = 1; thread_id < thread_numbers; thread_id++) {
          if (flag_access[thread_id][seq_id][hi] == 0) {
            continue;
          }
          auto attn_out_head_stride = thread_id * private_attn_out_stride +
              (seq_id * num_heads + hi) * head_size;
          auto private_attn_out_start =
              private_attn_out_ptr + attn_out_head_stride;
          at::vec::map2<accum_t>(
              [](Vec a, Vec b) { return a + b; },
              thr0_head_start,
              private_attn_out_start,
              thr0_head_start,
              head_size);
        }
        auto out_start = out_ptr + (seq_id * num_heads + hi) * head_size;
        at::vec::map<scalar_t>(
            [](Vec a) { return a; }, out_start, thr0_head_start, head_size);
      }
    }
  }

} // paged_attention_kernel

void paged_attention_kernel_impl(
    at::Tensor& out, // [num_seqs, 1,  num_heads, head_size]
    at::Tensor& query, // [num_seqs, 1, num_heads, head_size]
    at::Tensor& key_cache, // [num_blocks,  block_size, num_heads, head_size]
    at::Tensor& value_cache, // [num_blocks,  block_size, num_heads, head_size]
    at::Tensor& head_mapping, // [num_heads]
    const double scale,
    at::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    at::Tensor& context_lens, // [num_seqs]
    int64_t block_size,
    c10::optional<at::Tensor> attn_mask) {
  TORCH_CHECK(
      query.size(1) == 1,
      "Paged attention: only seqlen 1 is supported for query");
  TORCH_CHECK(
      query.scalar_type() == key_cache.scalar_type() &&
          query.scalar_type() == value_cache.scalar_type(),
      "Paged attention: Q/K/V should have the same data type");
  TORCH_CHECK(
      !attn_mask.has_value() ||
          query.scalar_type() == attn_mask.value().scalar_type() ||
          attn_mask.value().scalar_type() != at::ScalarType::Bool,
      "Paged attention: Mask should have the same data type as Q/K/V and should not be Bool");
  TORCH_CHECK(
      query.dim() == 4 && key_cache.dim() == 4 && value_cache.dim() == 4,
      "Paged attention: Accept only 4 dims inputs shape of {B, H, T, K}");
  TORCH_CHECK(
      (query.stride(-1) == 1) && (key_cache.stride(-1) == 1) &&
          (value_cache.stride(-1) == 1) &&
          (!attn_mask.has_value() || attn_mask.value().stride(-1) == 1),
      "Paged attention: Q/KV cache/Mask should be continuous on the last dim");

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      query.scalar_type(),
      "paged_attention",
      [&] {
        paged_attention_kernel<scalar_t>(
            out,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            attn_mask);
      });
}


} // namespace
TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::paged_attention", &paged_attention_kernel_impl);
}

} // namespace torchao