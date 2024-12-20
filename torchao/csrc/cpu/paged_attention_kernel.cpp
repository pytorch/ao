#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <ATen/Tensor.h>
#include <limits>
#include <omp.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

#define SEQ_PARTITION_SIZE 256

namespace torchao {

namespace {

template <typename scalar_t, typename accum_t>
void reduce_head(const scalar_t *q_ptr_start, const scalar_t *k_cache_start,
                 accum_t *attn_w_pos, int64_t head_size) {
  attn_w_pos[0] = 0;
  for (long i = 0; i < head_size; i++) {
    attn_w_pos[0] += q_ptr_start[i] * k_cache_start[i];
  }
}

// BF16
template <>
void reduce_head<at::BFloat16, float>(const at::BFloat16 *q_ptr_start,
                                      const at::BFloat16 *k_cache_start,
                                      float *attn_w_pos, int64_t head_size) {
  attn_w_pos[0] = 0;
  using lpVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = lpVec::size();
  auto vec_tmp_sum = fVec(0.0f);
  for (long i = 0; i < vec_size * (head_size / vec_size); i += vec_size) {
    auto tmpq = lpVec::loadu(q_ptr_start + i);
    auto tmpk = lpVec::loadu(k_cache_start + i);
    fVec tmpq1, tmpq2, tmpk1, tmpk2;
    // convert to float
    std::tie(tmpq1, tmpq2) = at::vec::convert_to_float(tmpq);
    std::tie(tmpk1, tmpk2) = at::vec::convert_to_float(tmpk);
    vec_tmp_sum = vec_tmp_sum + tmpq1 * tmpk1 + tmpq2 * tmpk2;
  }
  attn_w_pos[0] = at::vec::vec_reduce_all<>(
      [](fVec &x, fVec &y) { return x + y; }, vec_tmp_sum);
}

template <typename scalar_t, typename accum_t>
inline void mul_attenion_weights_and_value_of_head(
    const accum_t &attn_w, const scalar_t *v_cache_start,
    accum_t *attn_out_start, int64_t head_size, bool accumulated) {
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
    const float &attn_w, const at::BFloat16 *v_cache_start,
    float *attn_out_start, int64_t head_size, bool accumulated) {
  using lpVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  auto lpVec_size = lpVec::size();
  auto fVec_size = fVec::size();
  auto vec_attn_w = fVec(attn_w);
  auto vec_tmp_sum = fVec(0.0f);
  long i = 0;
  for (; i < lpVec_size * (head_size / lpVec_size); i += lpVec_size) {
    auto tmpv = lpVec::loadu(v_cache_start + i);
    fVec tmpv1, tmpv2;
    // convert to float
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

// out = val * a + b
template <typename T1, typename T2>
inline void _scale_attn_mask_fusion_kernel(T1 *a, float *b, const int &size,
                                           T2 *out, float val) {
  const auto vec_size = at::vec::Vectorized<float>::size();
  const auto vec_scale = at::vec::Vectorized<float>(val);
  int64_t i = 0;
  for (; i < size - (size % vec_size); i += vec_size) {
    auto a_v = at::vec::Vectorized<float>::loadu(a + i);
    auto b_v = at::vec::Vectorized<float>::loadu(b + i);
    auto res = a_v * vec_scale + b_v;
    res.store(out + i);
  }
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = b[i];
    out[i] = tmp0 * val + tmp1;
  }
}

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void _exp_reduce_sum_fusion_kernel(T1 *a, const int &size, T2 *out,
                                          T1 &val) {
  auto vec_size = at::vec::Vectorized<T1>::size();
  auto vec_max = at::vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<T1>(tmp_sum);
  long i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum += tmp2;
    tmp2.store(out + i);
  }
  tmp_sum = at::vec::vec_reduce_all<T1>(
      [](at::vec::Vectorized<T1> &x, at::vec::Vectorized<T1> &y) {
        return x + y;
      },
      vec_tmp_sum);
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(scalar_t *a, const scalar_t &scale,
                                          const int &size, scalar_t *out,
                                          scalar_t &max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  long i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    tmp1.store(out + i);
  }
  tmp_max = at::vec::vec_reduce_all<scalar_t>(
      [](at::vec::Vectorized<scalar_t> &x, at::vec::Vectorized<scalar_t> &y) {
        return at::vec::maximum(x, y);
      },
      vec_tmp_max);
  for (; i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = tmp_max;
}

void reshape_attn_mask_to_4d(at::Tensor &attn_mask, int64_t batchSize,
                             int64_t num_head, int64_t qSize, int64_t kvSize) {
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
                  .view({attn_mask_size_0, attn_mask_size_1, attn_mask.size(-2),
                         attn_mask.size(-1)})
                  .expand({attn_mask_size_0, attn_mask_size_1, qSize, kvSize});
}

/**
 * Performs scale-dot-product for the next token based on paged cached key-value
 * @param out           Output tensor [batch_size, num_heads, 1, head_size].
 * @param query         Query tensor [batch_size, num_heads, 1, head_size].
 * @param key_cache     The pre-allocated buffer to store the key cache. The
 * shape should be [num_blocks, num_heads, block_size, head_size].
 * @param value_cache   The pre-allocated buffer to store the value cache. The
 * shape should be [num_blocks, num_heads, block_size, head_size].
 * @param scale         Scaling factor for attention weights. In general, it is:
 * float(1.0 / (head_size ** 0.5)).
 * @param block_tables  Block tables tensor [batch_size, max_num_blocks_per_seq].
 * @param context_lens  Context lengths tensor [batch_size].
 * @param attn_mask  Optional tensor of attention_mask
 */
template <typename scalar_t>
void paged_attention_kernel(at::Tensor &out, at::Tensor &query,
                            at::Tensor &key_cache, at::Tensor &value_cache,
                            const double scale, at::Tensor &block_tables, 
                            at::Tensor &context_lens,
                            c10::optional<at::Tensor> attn_mask) {
  
  TORCH_CHECK(query.size(2) == 1,
              "Paged attention: only seqlen 1 is supported for query");
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);
  auto max_context_len = context_lens.max().item<int64_t>();
  auto batch_size = query.size(0);
  auto q_len = query.size(2);
  auto num_heads = query.size(1);
  auto head_size = query.size(3);
  auto block_size = key_cache.size(2);
  auto num_kv_heads = key_cache.size(1);
  auto max_num_blocks_per_seq = block_tables.size(1);
  auto kv_head_group_size = num_heads / num_kv_heads;
  bool has_attn_mask = attn_mask.has_value() && attn_mask.value().numel();
  if (has_attn_mask) {
    attn_mask.value() = attn_mask.value().to(at::kFloat);
    reshape_attn_mask_to_4d(attn_mask.value(), batch_size, num_heads, q_len,
                            attn_mask.value().size(-1));
  }

  auto out_ptr = out.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto key_cache_ptr = key_cache.data_ptr<scalar_t>();
  auto value_cache_ptr = value_cache.data_ptr<scalar_t>();
  auto block_tables_ptr = block_tables.data_ptr<int>();
  auto context_lens_ptr = context_lens.data_ptr<int>();

  auto kv_block_strideN = key_cache.stride(0);
  auto kv_block_strideP = key_cache.stride(2);
  auto kv_block_strideH = key_cache.stride(1);

  auto out_strideN = out.stride(0);
  auto out_strideH = out.stride(1);
  auto out_strideS = out.stride(2);

  auto q_strideN = query.stride(0);
  auto q_strideH = query.stride(1);
  auto q_strideS = query.stride(2);

  auto attn_mask_ptr =
      attn_mask.has_value() ? attn_mask.value().data_ptr<float>() : nullptr;

  int64_t mStrideB = (has_attn_mask && attn_mask.value().size(0) > 1)
                         ? attn_mask.value().stride(0)
                         : 0;
  int64_t mStrideH = (has_attn_mask && attn_mask.value().size(1) > 1)
                         ? attn_mask.value().stride(1)
                         : 0;
  int64_t mStrideM = has_attn_mask ? attn_mask.value().stride(2) : 0;
  
  auto max_num_seq_partitions =
      (max_context_len + SEQ_PARTITION_SIZE - 1) / SEQ_PARTITION_SIZE;

  auto max_logits = at::empty({batch_size, num_heads, max_num_seq_partitions + 1},
                              query.options().dtype(accumulate_dtype));

  auto exp_sum = at::empty({batch_size, num_heads, max_num_seq_partitions + 1},
                           query.options().dtype(accumulate_dtype));

  auto tmp_out = at::empty({batch_size, num_heads, max_num_seq_partitions, head_size},
                           query.options().dtype(accumulate_dtype));

  auto tmp_out_ptr = tmp_out.data_ptr<accum_t>();
  auto max_logits_ptr = max_logits.data_ptr<accum_t>();
  auto exp_sum_ptr = exp_sum.data_ptr<accum_t>();

  auto max_logits_strideN = max_logits.stride(0);
  auto max_logits_strideH = max_logits.stride(1);
  auto exp_sum_strideN = exp_sum.stride(0);
  auto exp_sum_strideH = exp_sum.stride(1);
  auto tmp_out_strideN = tmp_out.stride(0);
  auto tmp_out_strideH = tmp_out.stride(1);
  auto tmp_out_strideS = tmp_out.stride(2);
#pragma omp parallel for collapse(3) schedule(static, 1)
  for (auto partition_id = 0; partition_id < max_num_seq_partitions;
       partition_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {
      for (auto seq_id = 0; seq_id < batch_size; seq_id++) {
        auto context_len = context_lens_ptr[seq_id];
        auto partition_start = partition_id * SEQ_PARTITION_SIZE;
        if (partition_start >= context_len)
          continue;
        auto partition_end =
            std::min(partition_start + SEQ_PARTITION_SIZE, context_len);
        auto token_num = partition_end - partition_start;
        auto block_num = (token_num + block_size - 1) / block_size;
        auto logical_block_start = partition_start / block_size;
        auto logical_block_end = logical_block_start + block_num;
        auto need_update = block_num > 1;
        auto kv_head_id = head_id / kv_head_group_size;
        auto q_ptr_start = query_ptr + seq_id * q_strideN + head_id * q_strideH;
        auto max_logits_offset = seq_id * max_logits_strideN +
                                 head_id * max_logits_strideH + partition_id;
        auto exp_sum_offset =
            seq_id * exp_sum_strideN + head_id * exp_sum_strideH + partition_id;
        auto tmp_out_start = tmp_out_ptr + seq_id * tmp_out_strideN +
                             head_id * tmp_out_strideH +
                             partition_id * tmp_out_strideS;
        accum_t alignas(64) logits[SEQ_PARTITION_SIZE] = {0};
        auto logits_position = 0;
        // 1)calculate the matmul(query, key) for this partition
        for (auto logical_block_id = logical_block_start;
             logical_block_id < logical_block_end; logical_block_id++) {
          auto physical_block_id =
              block_tables_ptr[seq_id * max_num_blocks_per_seq +
                               logical_block_id];
          auto tokens_in_block =
              std::min(block_size, context_len - logical_block_id * block_size);
          auto token_start = logical_block_id * block_size;
          auto token_end = token_start + tokens_in_block;
          for (auto token_id = token_start; token_id < token_end; token_id++) {
            auto block_offset = token_id - token_start;
            auto k_cache_start =
                key_cache_ptr + physical_block_id * kv_block_strideN +
                block_offset * kv_block_strideP + kv_head_id * kv_block_strideH;
            reduce_head<scalar_t, accum_t>(q_ptr_start, k_cache_start,
                                           &(logits[logits_position]),
                                           head_size);
            logits_position++;
          }
        }
        // 2) calculate the max and exp_sum for this partition
        auto partition_max = -std::numeric_limits<accum_t>::infinity();
        if (has_attn_mask) {
          _scale_attn_mask_fusion_kernel<accum_t, accum_t>(
              logits,
              attn_mask_ptr + seq_id * mStrideB + head_id * mStrideH +
                  partition_start,
              token_num, logits, scale);
          partition_max = at::vec::reduce_all<accum_t>(
              [](Vec &x, Vec &y) { return at::vec::maximum(x, y); }, logits,
              token_num);
        } else {
          _mul_reduce_max_fusion_kernel<accum_t>(logits, scale, token_num,
                                                 logits, partition_max);
        }
        max_logits_ptr[max_logits_offset] = partition_max;
        _exp_reduce_sum_fusion_kernel<accum_t, accum_t>(logits, token_num,
                                                        logits, partition_max);
        exp_sum_ptr[exp_sum_offset] = partition_max;

        // 3) calculate the matmul(exp(logits-partition_max), value) for this
        // partition, need to divide the global exp_sum in the final result.
        logits_position = 0;
        for (auto logical_block_id = logical_block_start;
             logical_block_id < logical_block_end; logical_block_id++) {
          auto physical_block_id =
              block_tables_ptr[seq_id * max_num_blocks_per_seq +
                               logical_block_id];
          auto tokens_in_block =
              std::min(block_size, context_len - logical_block_id * block_size);
          auto token_start = logical_block_id * block_size;
          auto token_end = token_start + tokens_in_block;
          for (auto token_id = token_start; token_id < token_end; token_id++) {
            auto block_offset = token_id - token_start;
            auto v_cache_start =
                value_cache_ptr + physical_block_id * kv_block_strideN +
                block_offset * kv_block_strideP + kv_head_id * kv_block_strideH;
            auto accumulated = logits_position > 0;
            mul_attenion_weights_and_value_of_head<scalar_t, accum_t>(
                logits[logits_position], v_cache_start, tmp_out_start,
                head_size, accumulated);
            logits_position++;
          }
        }
      }
    }
  }

// calculate the final output
#pragma omp parallel for collapse(2)
  for (auto seq_id = 0; seq_id < batch_size; seq_id++) {
    for (auto head_id = 0; head_id < num_heads; head_id++) {
      auto global_max = -std::numeric_limits<accum_t>::infinity();
      auto global_exp_sum = 0.0;
      auto context_len = context_lens_ptr[seq_id];
      auto partition_num = (context_len + SEQ_PARTITION_SIZE - 1) / SEQ_PARTITION_SIZE;
      // calculate the global max and exp_sum for this head
      for (auto partition_id = 0; partition_id < max_num_seq_partitions;
           partition_id++) {
        if (partition_id >= partition_num)
          break;
        auto max_logit =
            max_logits_ptr[seq_id * max_logits_strideN +
                           head_id * max_logits_strideH + partition_id];
        global_max = std::max(global_max, max_logit);
      }
      // update the partition 0 result with the global max
      auto partition0_out_start =
          tmp_out_ptr + seq_id * tmp_out_strideN + head_id * tmp_out_strideH;
      auto max_logit0 = max_logits_ptr[seq_id * max_logits_strideN +
                                       head_id * max_logits_strideH];
      float exp_val = expf(max_logit0 - global_max);
      global_exp_sum +=
          exp_sum_ptr[seq_id * exp_sum_strideN + head_id * exp_sum_strideH] *
          exp_val;
      at::vec::Vectorized<accum_t> exp_val_vec0(exp_val);
      at::vec::map<accum_t>([&](auto a) { return a * exp_val_vec0; },
                            partition0_out_start, partition0_out_start,
                            head_size);

      // accumulate the partition 1 to partition n result into partition 0
      if (partition_num > 1) {
        for (auto partition_id = 1; partition_id < partition_num;
             partition_id++) {
          if (partition_id * SEQ_PARTITION_SIZE >= context_len)
            break;
          auto tmp_out_start = tmp_out_ptr + seq_id * tmp_out_strideN +
                               head_id * tmp_out_strideH +
                               partition_id * tmp_out_strideS;
          auto max_logit =
              max_logits_ptr[seq_id * max_logits_strideN +
                             head_id * max_logits_strideH + partition_id];
          auto exp_sum = exp_sum_ptr[seq_id * exp_sum_strideN +
                                     head_id * exp_sum_strideH + partition_id];
          exp_val = expf(max_logit - global_max);
          global_exp_sum += exp_sum * exp_val;
          at::vec::Vectorized<accum_t> exp_val_vec(exp_val);
          at::vec::map2<accum_t>(
              [&](auto a, auto b) { return a + exp_val_vec * b; },
              partition0_out_start, partition0_out_start, tmp_out_start,
              head_size);
        }
      }

      // copy the partition 0 result into attn_outs
      auto attn_out_start =
          out_ptr + seq_id * out_strideN + head_id * out_strideH;
      float inverse_global_sum = 1.0 / (global_exp_sum + 1e-8);
      at::vec::Vectorized<accum_t> inverse_global_sum_vec(inverse_global_sum);
      // rescale the partition 0 result with global exp_sum
      at::vec::map<accum_t>([&](auto a) { return a * inverse_global_sum_vec; },
                            partition0_out_start, partition0_out_start,
                            head_size);
      // copy the partition 0 result into attn_outs
      at::vec::map<scalar_t>([&](auto a) { return a; }, attn_out_start,
                             partition0_out_start, head_size);
    }
  }
} // paged_attention_kernel

void paged_attention_kernel_impl(
    at::Tensor &out,          // [batch_size, num_heads, 1, head_size]
    at::Tensor &query,        // [batch_size, num_heads, 1, head_size]
    at::Tensor &key_cache,    // [num_blocks, num_heads, block_size, head_size]
    at::Tensor &value_cache,  // [num_blocks, num_heads, block_size, head_size]
    const double scale,
    at::Tensor &block_tables, // [batch_size, max_num_blocks_per_seq]
    at::Tensor &context_lens, // [batch_size]
    c10::optional<at::Tensor> attn_mask) {
  TORCH_CHECK(SEQ_PARTITION_SIZE % key_cache.size(2) == 0,
              "Paged attention: The PARTION_SIZE:%d should be divisible by block_size: %d", SEQ_PARTITION_SIZE, key_cache.size(2));
  TORCH_CHECK(query.size(2) == 1,
              "Paged attention: only seqlen 1 is supported for query");
  TORCH_CHECK(query.scalar_type() == key_cache.scalar_type() &&
                  query.scalar_type() == value_cache.scalar_type(),
              "Paged attention: Q/K/V should have the same data type");
  TORCH_CHECK(!attn_mask.has_value() ||
                  query.scalar_type() == attn_mask.value().scalar_type() ||
                  attn_mask.value().scalar_type() != at::ScalarType::Bool,
              "Paged attention: Mask should have the same data type as Q/K/V "
              "and should not be Bool");
  TORCH_CHECK(
      query.dim() == 4 && key_cache.dim() == 4 && value_cache.dim() == 4,
      "Paged attention: Accept only 4 dims inputs shape of {B, H, T, K}");
  TORCH_CHECK(
      (query.stride(-1) == 1) && (key_cache.stride(-1) == 1) &&
          (value_cache.stride(-1) == 1) &&
          (!attn_mask.has_value() || attn_mask.value().stride(-1) == 1),
      "Paged attention: Q/KV cache/Mask should be continuous on the last dim");

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, query.scalar_type(), "paged_attention", [&] {
        paged_attention_kernel<scalar_t>(out, query, key_cache, value_cache,
                                         scale, block_tables,
                                         context_lens, attn_mask);
      });
}

} // namespace
TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::paged_attention", &paged_attention_kernel_impl);
}

} // namespace torchao