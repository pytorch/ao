#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/vec_quant.h>
#include <ATen/cpu/Utils.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <limits>
#include <omp.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torchao {

namespace {

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    std::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

#ifdef CPU_CAPABILITY_AVX512

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  const int32_t vec_size = at::vec::Vectorized<scalar_t>::size();
  auto data_vec = at::vec::Vectorized<scalar_t>(val);
  int64_t d = 0;
  for (; d < size - (size % vec_size); d += vec_size) {
    data_vec.store(data + d);
  }
  if (d < size) {
    data_vec.store(data + d, size - d);
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
                .view({attn_mask_size_0, attn_mask_size_1, attn_mask.size(-2), attn_mask.size(-1)})
                .expand({attn_mask_size_0, attn_mask_size_1, qSize, kvSize});
}

// TODO: Use at::native::_store instead when it supports Half.
template <typename scalar_t>
inline void _store(scalar_t* dst, at::vec::Vectorized<scalar_t> src, int size=at::vec::Vectorized<scalar_t>::size()) {
  src.store(dst, size);
}

template <typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char> || std::is_same_v<scalar_t, signed char> || std::is_same_v<scalar_t, at::Float8_e4m3fn>, void>
_store(scalar_t* dst, at::vec::Vectorized<float> src, int size=at::vec::Vectorized<float>::size()) {
  auto res = at::vec::convert<scalar_t>(src);
  res.store(dst, size);
}

/*
out = val * a + b
is_b_stride_zero: If the stride of b is 0 (mask broadcasting case),
                take b as a scalar pointer.
*/
template <bool is_b_stride_zero, typename T1, typename T2>
inline void _scale_dequant_attn_mask_fusion_kernel(
    T1* a,
    T2* b,
    const int& size,
    T1* out,
    const T1& val) {
  const auto vec_size1 = at::vec::Vectorized<T1>::size();
  const auto vec_size2 = at::vec::Vectorized<T2>::size();
  constexpr int64_t T1_n =
      (vec_size2 == vec_size1 * 2 && at::vec::is_reduced_floating_point_v<T2>) ? 2 : 1;
  constexpr int64_t T2_n = 1;
  auto vec_scale = at::vec::VectorizedN<T1, T1_n>(val);
  int64_t i = 0;
  for (; i < size - (size % vec_size2); i += vec_size2) {
    auto a_n = at::vec::VectorizedN<T1, T1_n>::loadu(a + i);
    at::vec::VectorizedN<T2, T2_n> b_n;
    if constexpr(is_b_stride_zero) {
      b_n = at::vec::VectorizedN<T2, T2_n>((T1)b[0]);
    } else {
      b_n = at::vec::VectorizedN<T2, T2_n>::loadu(b + i);
    }
    auto b_n_convert = at::vec::convert<T1, T1_n, T2, T2_n, true>(b_n);
    auto res = a_n * vec_scale + b_n_convert;
    res.store(out + i);
  }
  for (; i < size; i++) {
    auto tmp0 = a[i];
    T1 tmp1;
    if constexpr(is_b_stride_zero) {
      tmp1 = (T1)b[0];
    } else {
      tmp1 = (T1)b[i];
    }
    out[i] = tmp0 * val + tmp1;
  }
}

/*
1. dequant
2. add mask
3. max reduce for softmax
*/
template <typename mask_t>
inline void _dequant_mask_max_fusion_kernel(
    const int32_t* in,
    const mask_t* mask_ptr,
    const int32_t* sum_a_ptr,
    const int32_t* sum_b_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldm, // leading dimension mask
    const int& ldo,
    const int32_t& beta, // zp_a*zp_b*k
    const float& alpha, // scale_a*scale_b*scale_sdpa
    float* out,
    float* sfm_max_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  auto vec_beta = at::vec::Vectorized<int32_t>(beta);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    float* tmp_out = out + row * ldo;
    const mask_t* mask_data_ptr = mask_ptr + row * ldm;
    float tmp_max = -std::numeric_limits<float>::infinity();
    auto vec_tmp_max = at::vec::Vectorized<float>(tmp_max);
    long col = 0;
    for (; col < vec_size * (N / vec_size); col += vec_size) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = at::vec::Vectorized<mask_t>::loadu(mask_data_ptr + col);
      auto tmp7 = at::vec::convert<float>(tmp6);
      auto tmp8 = tmp5 + tmp7;
      vec_tmp_max = at::vec::clamp_min(vec_tmp_max, tmp8);
      _store(tmp_out + col, tmp8);
    }
    if (col < N) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col, N - col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col, N - col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = at::vec::Vectorized<mask_t>::loadu(mask_data_ptr + col, N - col);
      auto tmp7 = at::vec::convert<float>(tmp6);
      auto tmp8 = tmp5 + tmp7;
      _store(tmp_out + col, tmp8, N - col);
      vec_tmp_max = at::vec::Vectorized<float>::set(vec_tmp_max, at::vec::clamp_min(vec_tmp_max, tmp8), N - col);
    }
    sfm_max_ptr[row] = std::max(sfm_max_ptr[row], vec_tmp_max.reduce_max());
  }
}

/*
1. dequant
2. max reduce for softmax
*/
inline void _dequant_max_fusion_kernel(
    const int32_t* in,
    const int32_t* sum_a_ptr,
    const int32_t* sum_b_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldo,
    const int32_t& beta, // zp_a*zp_b*k
    const float& alpha, // scale_a*scale_b*scale_sdpa
    float* out,
    float* sfm_max_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  auto vec_beta = at::vec::Vectorized<int32_t>(beta);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    float* tmp_out = out + row * ldo;
    float tmp_max = -std::numeric_limits<float>::infinity();
    auto vec_tmp_max = at::vec::Vectorized<float>(tmp_max);
    long col = 0;
    for (; col < vec_size * (N / vec_size); col += vec_size) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      vec_tmp_max = at::vec::clamp_min(vec_tmp_max, tmp5);
      _store(tmp_out + col, tmp5);
    }
    if (col < N) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col, N - col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col, N - col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      _store(tmp_out + col, tmp5, N - col);
      vec_tmp_max = at::vec::Vectorized<float>::set(vec_tmp_max, at::vec::clamp_min(vec_tmp_max, tmp5), N - col);
    }
    sfm_max_ptr[row] = std::max(sfm_max_ptr[row], vec_tmp_max.reduce_max());
  }
}

/*
1. Softmax: sub max, exp, sum reduce, div sum
2. quant
3. sum for attention
*/
template <typename scalar_t>
inline void _sub_exp_sum_div_quant_sum_fusion_kernel(
    const float* in,
    const int64_t& M,
    const int64_t& N_step,
    const int64_t& NSlice,
    const int& ldi,
    const int& ldo,
    const int& kvSize,
    const int& rndkvSplitSize,
    const int& av_gemm_K,
    const int32_t& beta1, // zp_a
    const int32_t& beta2, // zp_b
    const float& alpha, // scale_a
    float* local,
    scalar_t* out,
    float* sfm_max_ptr,
    float* sfm_sum_ptr,
    int32_t* sum_a_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  scalar_t zero = 0;
  auto vec_zero = at::vec::Vectorized<scalar_t>(zero);
  float beta1_float = (float) beta1;
  auto vec_beta1 = at::vec::Vectorized<float>(beta1_float);
  for (int64_t row = 0; row < M; ++row) {
    auto sfm_max = sfm_max_ptr[row];
    auto vec_max = at::vec::Vectorized<float>(sfm_max);
    // sub max, exp, sum reduce
    const float* qk_block_data = in + row * rndkvSplitSize;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      const float* tmp_in = qk_block_data + l * ldi;
      float tmp_sum = 0;
      auto vec_tmp_sum = at::vec::Vectorized<float>(tmp_sum);
      float* tmp_out = local + n;
      long col = 0;
      for (; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        vec_tmp_sum += tmp2;
        _store(tmp_out + col, tmp2);
      }
      if (col < kvBlockSize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col, kvBlockSize - col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        _store(tmp_out + col, tmp2, kvBlockSize - col);
        vec_tmp_sum = at::vec::Vectorized<float>::set(vec_tmp_sum, vec_tmp_sum + tmp2, kvBlockSize - col);
      }
      sfm_sum_ptr[row] += vec_tmp_sum.reduce_add();
    }
    // div sum, sum for attention
    auto sum_scale = 1 / sfm_sum_ptr[row] / alpha;
    auto vec_sum_scale = at::vec::Vectorized<float>(sum_scale);
    scalar_t* qk_reduced_block_data = out + row * av_gemm_K;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      int32_t tmp_sum = 0;
      auto vec_tmp_sum = at::vec::Vectorized<int32_t>(tmp_sum);
      float* tmp_in = local + n;
      scalar_t* tmp_out = qk_reduced_block_data + l * ldo;
      long col = 0;
      for (; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
        _store(tmp_out + col, tmp4);
        auto tmp6 = at::vec::convert<int32_t>(tmp4);
        vec_tmp_sum += tmp6;
      }
      if (col < kvBlockSize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col, kvBlockSize - col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
        _store(tmp_out + col, tmp4, kvBlockSize - col);
        auto tmp6 = at::vec::convert<int32_t>(tmp4);
        vec_tmp_sum = at::vec::Vectorized<int32_t>::set(vec_tmp_sum, vec_tmp_sum + tmp6, kvBlockSize - col);
      }
      sum_a_ptr[row] += vec_tmp_sum.reduce_add() * beta2;
      // set zero
      col = kvBlockSize;
      for (; col <  vec_size * (av_gemm_K / vec_size); col += vec_size) {
        _store(tmp_out + col, vec_zero);
      }
      if (col < av_gemm_K) {
        _store(tmp_out + col, vec_zero, av_gemm_K - col);
      }
    }
  }
}

/*
1. Softmax: sub max, exp, sum reduce, div sum
2. quant
*/
template <typename scalar_t>
inline void _sub_exp_sum_div_quant_fusion_kernel(
    const float* in,
    const int64_t& M,
    const int64_t& N_step,
    const int64_t& NSlice,
    const int& ldi,
    const int& ldo,
    const int& kvSize,
    const int& rndkvSplitSize,
    const int& av_gemm_K,
    const int32_t& beta1, // zp_a
    const float& alpha, // scale_a
    float* local,
    scalar_t* out,
    float* sfm_max_ptr,
    float* sfm_sum_ptr) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  scalar_t zero = 0;
  auto vec_zero = at::vec::Vectorized<scalar_t>(zero);
  float beta1_float = (float) beta1;
  auto vec_beta1 = at::vec::Vectorized<float>(beta1_float);
  for (int64_t row = 0; row < M; ++row) {
    auto sfm_max = sfm_max_ptr[row];
    auto vec_max = at::vec::Vectorized<float>(sfm_max);
    // sub max, exp, sum reduce
    const float* qk_block_data = in + row * rndkvSplitSize;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      const float* tmp_in = qk_block_data + l * ldi;
      float tmp_sum = 0;
      auto vec_tmp_sum = at::vec::Vectorized<float>(tmp_sum);
      float* tmp_out = local + n;
      long col = 0;
      for (; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        vec_tmp_sum += tmp2;
        _store(tmp_out + col, tmp2);
      }
      if (col < kvBlockSize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col, kvBlockSize - col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        vec_tmp_sum = at::vec::Vectorized<float>::set(vec_tmp_sum, vec_tmp_sum + tmp2, kvBlockSize - col);
        _store(tmp_out + col, tmp2, kvBlockSize - col);
      }
      sfm_sum_ptr[row] += vec_tmp_sum.reduce_add();
    }
    // div sum, sum for attention
    auto sum_scale = 1 / sfm_sum_ptr[row] / alpha;
    auto vec_sum_scale = at::vec::Vectorized<float>(sum_scale);
    scalar_t* qk_reduced_block_data = out + row * av_gemm_K;
    for (int64_t l = 0; l < NSlice; l ++) {
      int64_t n = l * N_step;
      int64_t kvBlockSize = std::min(N_step, kvSize - n);
      float* tmp_in = local + n;
      scalar_t* tmp_out = qk_reduced_block_data + l * ldo;
      long col = 0;
      for (; col < vec_size * (kvBlockSize / vec_size); col += vec_size) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
        _store(tmp_out + col, tmp4);
      }
      if (col < kvBlockSize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col, kvBlockSize - col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
        _store(tmp_out + col, tmp4, kvBlockSize - col);
      }
      // set zero
      col = kvBlockSize;
      for (; col < vec_size * (av_gemm_K / vec_size); col += vec_size) {
        _store(tmp_out + col, vec_zero);
      }
      if (col < av_gemm_K) {
        _store(tmp_out + col, vec_zero, av_gemm_K - col);
      }
    }
  }
}

/*
1. dequant
2. quant
*/
template <typename scalar_t>
inline void _dequant_quant_fusion_kernel(
    const int32_t* in,
    const int32_t* sum_a_ptr,
    const int32_t* sum_b_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldo,
    const int32_t& beta1, // zp_a*zp_b*k
    const int32_t& beta2, // zp_c
    const float& alpha, // scale_a*scale_b/scale_c
    scalar_t* out) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  auto vec_beta1 = at::vec::Vectorized<int32_t>(beta1);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  float beta2_float = (float) beta2;
  auto vec_beta2 = at::vec::Vectorized<float>(beta2_float);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    scalar_t* tmp_out = out + row * ldo;
    long col = 0;
    for (; col < vec_size * (N / vec_size); col += vec_size) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta1;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = tmp5.round();
      auto tmp7 = tmp6 + vec_beta2;
      auto tmp8 = at::vec::clamp(tmp7, vec_min_val, vec_max_val);
      _store(tmp_out + col, tmp8);
    }
    if (col < N) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col, N - col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col, N - col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta1;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = tmp5.round();
      auto tmp7 = tmp6 + vec_beta2;
      auto tmp8 = at::vec::clamp(tmp7, vec_min_val, vec_max_val);
      _store(tmp_out + col, tmp8, N - col);
    }
  }
}

/*
1. dequant
2. quant
*/
template <typename scalar_t>
inline void _dequant_quant_fusion_kernel(
    const int32_t* in,
    const int32_t* sum_a_ptr,
    const int& M,
    const int& N,
    const int& ldi,
    const int& ldo,
    const int32_t& beta2, // zp_c
    const float& alpha, // scale_a*scale_b/scale_c
    scalar_t* out) {
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  float min_val = 0;
  float max_val = 255;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  // auto vec_beta1 = at::vec::Vectorized<int32_t>(beta1);
  auto vec_alpha = at::vec::Vectorized<float>(alpha);
  float beta2_float = (float) beta2;
  auto vec_beta2 = at::vec::Vectorized<float>(beta2_float);
  for (long row = 0; row < M; row += 1) {
    auto sum_a = sum_a_ptr[row];
    auto vec_sum_a = at::vec::Vectorized<int32_t>(sum_a);
    const int32_t* tmp_in = in + row * ldi;
    scalar_t* tmp_out = out + row * ldo;
    long col = 0;
    for (; col < vec_size * (N / vec_size); col += vec_size) {
      auto tmp1 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col);
      auto tmp3 = tmp1 - vec_sum_a;
      // auto tmp3 = tmp2 + vec_beta1;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = tmp5.round();
      auto tmp7 = tmp6 + vec_beta2;
      auto tmp8 = at::vec::clamp(tmp7, vec_min_val, vec_max_val);
      _store(tmp_out + col, tmp8);
    }
    if (col < N) {
      auto tmp1 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col, N - col);
      auto tmp3 = tmp1 - vec_sum_a;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = tmp5.round();
      auto tmp7 = tmp6 + vec_beta2;
      auto tmp8 = at::vec::clamp(tmp7, vec_min_val, vec_max_val);
      _store(tmp_out + col, tmp8, N - col);
    }
  }
}

template <typename scalar_t>
inline void _int_sum_b_contiguous_kernel_helper(
    const scalar_t* in,
    int32_t* out,
    const int& N,
    const int32_t& scale) {
  const int32_t vec_size = at::vec::Vectorized<int32_t>::size();
  int32_t tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<int32_t>(tmp_sum);
  long i = 0;
  for (; i < vec_size * (N / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(in + i);
    auto tmp1 = at::vec::convert<int32_t>(tmp0);
    vec_tmp_sum = vec_tmp_sum + tmp1;
  }
  if (i < N) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(in + i, N - i);
    auto tmp1 = at::vec::convert<int32_t>(tmp0);
    vec_tmp_sum = at::vec::Vectorized<int32_t>::set(vec_tmp_sum, vec_tmp_sum + tmp1, N - i);
  }
  out[0] = vec_tmp_sum.reduce_add() * scale;
}

// reduce along dim b for shape [a, b], with sum shape [a]
template <typename scalar_t>
inline void _int_sum_b_contiguous_kernel(
    const scalar_t* in,
    int32_t* out,
    const int& M,
    const int& N,
    const int& ld,
    const int32_t& scale) {
  for (long r = 0; r < M; r += 1) {
    _int_sum_b_contiguous_kernel_helper(in + r * ld, out + r, N, scale);
  }
}

// reduce along dim a for shape [a, b], with sum shape [b]
template <typename scalar_t>
inline void _int_sum_a_contiguous_kernel(
    const scalar_t* in,
    int32_t* out,
    const int& M,
    const int& N,
    const int& ld,
    const int32_t& scale) {
  const int32_t vec_size = at::vec::Vectorized<int32_t>::size();
  auto vec_scale = at::vec::Vectorized<int32_t>(scale);
  // initialization with 0
  int32_t zero = 0;
  auto vec_zero = at::vec::Vectorized<int32_t>(zero);
  long i = 0;
  for (; i < vec_size * (M / vec_size); i += vec_size) {
    _store(out + i, vec_zero);
  }
  if (i < M) {
    _store(out + i, vec_zero, M - i);
  }
  // sum
  for (long j = 0; j < N; j++) {
    const scalar_t* tmp_in = in + j * ld;
    long k = 0;
    for (; k < vec_size * (M / vec_size); k += vec_size) {
      auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(tmp_in + k);
      auto tmp1 = at::vec::Vectorized<int32_t>::loadu(out + k);
      auto tmp2 = at::vec::convert<int32_t>(tmp0);
      auto tmp3 = tmp1 + tmp2;
      _store(out + k, tmp3);
    }
    if (k < M) {
      auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(tmp_in + k, M - k);
      auto tmp1 = at::vec::Vectorized<int32_t>::loadu(out + k, M - k);
      auto tmp2 = at::vec::convert<int32_t>(tmp0);
      auto tmp3 = tmp1 + tmp2;
      _store(out + k, tmp3, M - k);
    }
  }
  // scale
  i = 0;
  for (; i < vec_size * (M / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out + i);
    auto tmp1 = tmp0 * vec_scale;
    _store(out + i, tmp1);
  }
  if (i < M) {
    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out + i, M - i);
    auto tmp1 = tmp0 * vec_scale;
    _store(out + i, tmp1, M - i);
  }
}

// do the transpose: [in_rows, in_cols] -> [in_cols, in_rows]
template <typename scalar_t>
inline void do_transpose(
    const scalar_t* src,
    scalar_t* dst,
    int64_t in_rows,
    int64_t in_cols,
    int64_t ldi,
    int64_t ldo) {
  for (int64_t r=0; r<in_rows; r++) {
    for (int64_t c=0; c<in_cols; c++) {
      *(dst + c * ldo + r) = *(src + r * ldi + c);
    }
  }
}

// padding with pad_val: [rows, cols] -> [prows, pcols]
template <typename scalar_t>
inline void pad_remain_row_col(
    scalar_t* value_ptr,
    int rows,
    int cols,
    int prows,
    int pcols,
    int ldi,
    scalar_t pad_val=0) {
  auto psize = pcols - cols;
  if (psize == 0 && prows == rows) {
    return;
  }
  const int32_t vec_size = at::vec::Vectorized<scalar_t>::size();
  auto pad = at::vec::Vectorized<scalar_t>(pad_val);
  if (psize > 0) {
    for (int i = 0; i < rows; i++) {
      int j = 0;
      for (; j < psize - (psize % vec_size); j += vec_size) {
        pad.store(value_ptr + i * ldi + cols + j);
      }
      if (j < psize) {
        pad.store(value_ptr + i * ldi + cols + j, psize - j);
      }
    }
  }

  for (int i = rows; i < prows; i++) {
    int j = 0;
    for (; j < pcols - (pcols % vec_size); j += vec_size) {
      pad.store(value_ptr + i * ldi + j);
    }
    if (j < pcols) {
      pad.store(value_ptr + i * ldi + j, pcols - j);
    }
  }
}

// copy value_ptr to dst_ptr with padding: [rows, cols] -> [prows, pcols]
template <typename scalar_t>
inline void copy_value_with_pad(
    const scalar_t* value_ptr,
    scalar_t* dst_ptr,
    int rows,
    int cols,
    int prows,
    int pcols,
    int ldi,
    scalar_t pad_val=0) {
  const int32_t vec_size = at::vec::Vectorized<scalar_t>::size();
  auto pad = at::vec::Vectorized<scalar_t>(pad_val);
  int i = 0;
  for (; i < rows; i++) {
    int j = 0;
    for (; j < cols - (cols % vec_size); j += vec_size) {
      auto vec_v =
          at::vec::Vectorized<scalar_t>::loadu(value_ptr + i * ldi + j);
      vec_v.store(dst_ptr + i * pcols + j);
    }

    if (j < cols) {
      auto vec_v = at::vec::Vectorized<scalar_t>::loadu(
          value_ptr + i * ldi + j, cols - j);
      vec_v.store(dst_ptr + i * pcols + j, cols - j);
    }

    // col padding
    auto psize = pcols - cols;
    if (psize > 0) {
      int pj = 0;
      for (; pj < psize - (psize % vec_size); pj += vec_size) {
        pad.store(dst_ptr + i * pcols + cols + pj);
      }
      if (pj < psize) {
        pad.store(dst_ptr + i * pcols + cols + pj, psize - pj);
      }
    }
  }

  // row padding
  for (; i < prows; i++) {
    int j = 0;
    for (; j < pcols - (pcols % vec_size); j += vec_size) {
      pad.store(dst_ptr + i * pcols + j);
    }
    if (j < pcols) {
      pad.store(dst_ptr + i * pcols + j, pcols - j);
    }

  }

}

/*
1. out = a * scale
2. max = max(out)
*/
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t& scale,
    const int& size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = at::vec::Vectorized<scalar_t>::size();
  auto vec_scale = at::vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = at::vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = at::vec::maximum(vec_tmp_max, tmp1);
    _store(out + i, tmp1);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  auto reduced_tmp_max = at::vec::vec_reduce_all<scalar_t>(
      [](at::vec::Vectorized<scalar_t>& x, at::vec::Vectorized<scalar_t>& y) {
        return at::vec::maximum(x, y);
      },
      vec_tmp_max);
  // Guard against Q*K^T being NaN
  max = std::isnan(reduced_tmp_max) ? std::numeric_limits<scalar_t>::quiet_NaN()
                                    : std::max(tmp_max, reduced_tmp_max);
}

/*
1. out = exp(a - val)
2. val = sum(out)
3. quant
*/
inline void _fp8_exp_reduce_sum_quant_fusion_kernel(
    float* a,
    const int& size,
    at::Float8_e4m3fn* out,
    float& val,
    const float& scale) {
  auto vec_size = at::vec::Vectorized<float>::size();
  auto vec_max = at::vec::Vectorized<float>(val);
  float tmp_sum = 0;
  auto vec_tmp_sum = at::vec::Vectorized<float>(tmp_sum);
  float min_val = -448;
  float max_val = 448;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  auto vec_scale = at::vec::Vectorized<float>(scale);
  long i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<float>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum += tmp2;
    auto tmp3 = tmp2 * vec_scale;
    auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
    _store(out + i, tmp4);
  }
  if (i < size) {
    auto tmp0 = at::vec::Vectorized<float>::loadu(a + i, size - i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum = at::vec::Vectorized<float>::set(vec_tmp_sum, vec_tmp_sum + tmp2, size - i);
    auto tmp3 = tmp2 * vec_scale;
    auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
    _store(out + i, tmp4, size - i);
  }
  val = vec_tmp_sum.reduce_add();
}

/*
1. dequant
2. quant
*/
inline void _fp8_dequant_quant_fusion_kernel(
    float* a,
    const int& size,
    at::Float8_e4m3fn* out,
    const float& scale) {
  auto vec_size = at::vec::Vectorized<float>::size();
  float min_val = -448;
  float max_val = 448;
  auto vec_min_val = at::vec::Vectorized<float>(min_val);
  auto vec_max_val = at::vec::Vectorized<float>(max_val);
  auto vec_scale = at::vec::Vectorized<float>(scale);
  long i = 0;
  for (; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<float>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    auto tmp2 = at::vec::clamp(tmp1, vec_min_val, vec_max_val);
    _store(out + i, tmp2);
  }
  if (i < size) {
    auto tmp0 = at::vec::Vectorized<float>::loadu(a + i, size - i);
    auto tmp1 = tmp0 * vec_scale;
    auto tmp2 = at::vec::clamp(tmp1, vec_min_val, vec_max_val);
    _store(out + i, tmp2, size - i);
  }
}

// UINT8 - one parallel loop with u8u8s32 GEMM 
template <typename scalar_t, typename mask_t,
          int64_t q_split_size, int64_t kv_split_size,
          bool use_one_parallel_loop,
          typename std::enable_if_t<use_one_parallel_loop, int> = 0>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char>, void>
int8_sdpa_fused_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    std::optional<at::Tensor> attention_mask,
    std::optional<double> scale,
    float q_scale,
    int32_t q_zp,
    float k_scale,
    int32_t k_zp,
    float v_scale,
    int32_t v_zp,
    float a_scale,
    int32_t a_zp,
    float o_scale,
    int32_t o_zp) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  using accum_t = float;
  accum_t scaling_factor = calculate_scale(query, scale).expect_float();
  int block_64 = 64;
  auto u8_dt = at::ScalarType::Byte;

  // Sizes
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_sdpa: Q/K/V should have the same head size");
  TORCH_CHECK(
      kv_split_size % block_64 == 0, "kv_split_size is not divisble by ", block_64);

  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  bool has_attn_mask = attention_mask.has_value() && attention_mask.value().numel();
  if (has_attn_mask) {
    reshape_attn_mask_to_4d(attention_mask.value(), batchSize, num_head, qSize, kvSize);
  }

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t mStrideB =
      (has_attn_mask && attention_mask.value().size(0) > 1)
      ? attention_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (has_attn_mask && attention_mask.value().size(1) > 1)
      ? attention_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      (has_attn_mask && attention_mask.value().size(2) > 1)
      ? attention_mask.value().stride(2)
      : 0;
  int64_t mStrideN =
      (has_attn_mask && attention_mask.value().size(3) > 1)
      ? attention_mask.value().stride(3)
      : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  int64_t rndHeadSize = (headSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSplitSize = (kvSplitSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvTail = (kvTail + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSize = kv_split_size > kvSize ? rndkvTail : rndkvSplitSize * kvSlice + rndkvTail;

  bool av_gemm_K_mul4 = kvSplitSize % 4 == 0;
  int av_gemm_K_padding = av_gemm_K_mul4 ? 0 : 4 - kvSplitSize % 4;
  int av_gemm_K = kvSplitSize + av_gemm_K_padding;

  // Data ptrs
  const scalar_t* q_data = query.data_ptr<scalar_t>();
  const scalar_t* k_data = key.data_ptr<scalar_t>();
  const scalar_t* v_data = value.data_ptr<scalar_t>();
  mask_t* mask_data = attention_mask.has_value()
      ? attention_mask.value().data_ptr<mask_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();

  bool headSize_mul64 = headSize % 64 == 0;
  int qk_gemm_K_padding = headSize_mul64 ? 0 : 64 - headSize % 64;
  int qk_gemm_K = headSize + qk_gemm_K_padding;

  int64_t qk_reduce_strideL = qSplitSize * av_gemm_K;
  int64_t v_reorder_strideL = av_gemm_K * rndHeadSize;

  int64_t total_size_uint8_per_thread =
    /* qk */ kvSlice * qSplitSize * rndkvSplitSize * 4 +
    /* qk_local  */ kvSlice * av_gemm_K * 4 +
    /* qk_reduce  */ kvSlice * qk_reduce_strideL +
    /* qk_s32   */ qSplitSize * rndkvSplitSize * 4 +
    /* dst_s32  */ qSplitSize * rndHeadSize * 4 +
    /* softmax_sum   */ qSplitSize * 4 +
    /* query_sum     */ qSplitSize * 4 +
    /* attention_sum */ qSplitSize * 4 +
    /* softmax max */ qSplitSize * 4 +
    /* query_padding_data */ qSplitSize * qk_gemm_K +
    /* key_sum */ kvSize * 4 +
    /* value_sum */ headSize * 4 +
    /* key_t_reorder */ qk_gemm_K * rndkvSize +
    /* value_t_reorder */ kvSlice * v_reorder_strideL;

  at::Tensor total_buf = at::empty(
      {num_thread, total_size_uint8_per_thread},
      query.options());
  scalar_t* total_buf_data = total_buf.data_ptr<scalar_t>();

  at::parallel_for(
      0, batchSize * num_head, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head);
        int ompIdx = at::get_thread_num();
        scalar_t* total_buf_ptr = total_buf_data + ompIdx * total_size_uint8_per_thread;
        int32_t offset = 0;
        accum_t* qk_data = reinterpret_cast<accum_t*>(total_buf_ptr);
        offset += kvSlice * qSplitSize * rndkvSplitSize * 4;
        accum_t* qk_local_data = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += kvSlice * av_gemm_K * 4;
        scalar_t* qk_reduced_data = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);
        offset += kvSlice * qk_reduce_strideL;
        int32_t* qk_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndkvSplitSize * 4;
        int32_t* dst_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndHeadSize * 4;
        accum_t* sfm_sum_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* q_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* a_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        accum_t* sfm_max_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        scalar_t* query_t_padding_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);
        offset += qSplitSize * qk_gemm_K;

        int32_t* k_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += kvSize * 4;
        int32_t* v_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += headSize * 4;
        scalar_t* key_reorder_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);
        offset += qk_gemm_K * rndkvSize;
        scalar_t* value_reorder_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);

        uint8_t* B_blocked_xform_u8 = new uint8_t[qk_gemm_K * block_64];

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable

          // sum k and v
          if (q_zp == 0) {
            fill_stub(k_sum_ptr, static_cast<int32_t>(0), kvSize);
          } else {
            _int_sum_b_contiguous_kernel(k_data + i * kStrideB + j * kStrideH,
              k_sum_ptr,
              kvSize, headSize, kStrideN, q_zp);
          }
          if (a_zp == 0) {
            fill_stub(v_sum_ptr, static_cast<int32_t>(0), headSize);
          } else {
            _int_sum_a_contiguous_kernel(v_data + i * vStrideB + j * vStrideH,
              v_sum_ptr,
              headSize, kvSize, vStrideN, a_zp);
          }

          // transpose and packing
          for (int64_t n = 0; n < kvSize; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            for (int64_t b = 0; b < kvBlockSize; b += block_64) {
              bool istail = kvBlockSize - b < block_64;
              int64_t trans_rows = istail ? kvBlockSize - b : block_64;
              do_transpose(
                  reinterpret_cast<const uint8_t*>(k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN),
                  B_blocked_xform_u8,
                  trans_rows,
                  headSize,
                  kStrideN,
                  block_64);
              if (!headSize_mul64 || istail) {
                pad_remain_row_col(
                    B_blocked_xform_u8,
                    headSize,
                    trans_rows,
                    qk_gemm_K,
                    block_64,
                    block_64
                  );
              }
              at::native::cpublas::pack(
                      qk_gemm_K, // K
                      block_64, // N
                      block_64, // ld_in
                      block_64, // ld_out
                      u8_dt, // dt_in
                      u8_dt, // dt_out
                      B_blocked_xform_u8,
                      key_reorder_ptr + n * qk_gemm_K +
                          b * qk_gemm_K);
            }
            // split headSize to block_64, block_64, block_64 ...
            // [av_gemm_K, headSize] -> [av_gemm_K,  block_64 ...]
            for (int64_t b = 0; b < rndHeadSize; b += block_64) {
              at::native::cpublas::pack(
                      av_gemm_K,
                      block_64,
                      vStrideN,
                      block_64,
                      u8_dt,
                      u8_dt,
                      v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                      value_reorder_ptr + n * rndHeadSize +
                          av_gemm_K * b);
            }
          }

          // sdpa core
          for (int64_t k = 0; k < qSlice; k++) {
            int64_t m = k * qSplitSize;
            int64_t qBlockSize = std::min(qSplitSize, qSize - m);
            // Initialize sum and max
            fill_stub(
                sfm_sum_ptr, static_cast<accum_t>(0), qSplitSize);
            fill_stub(
                a_sum_ptr, static_cast<int32_t>(0), qSplitSize);
            fill_stub(
                sfm_max_ptr, static_cast<accum_t>(-std::numeric_limits<accum_t>::infinity()), qSplitSize);
            int64_t num_keys =
                is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
            copy_value_with_pad(
                q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                query_t_padding_ptr,
                qBlockSize,
                headSize,
                qBlockSize,
                qk_gemm_K,
                qStrideM);
            // sum q
            if (k_zp != 0) {
              _int_sum_b_contiguous_kernel(q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                    q_sum_ptr, qBlockSize, headSize, qStrideM, k_zp);
            } else {
              fill_stub(
                q_sum_ptr, static_cast<int32_t>(0), qSplitSize);
            }
            const int64_t rkvSlice = (num_keys - 1) / kvSplitSize + 1;
            for (int64_t l = 0; l < rkvSlice; l++) {
              int64_t n = l * kvSplitSize;
              int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
              // Calculate q @ k.T
              for (int64_t b = 0; b < kvBlockSize; b += block_64) {
                at::native::cpublas::brgemm(
                      qSplitSize, block_64, qk_gemm_K,
                      qk_gemm_K, // lda
                      block_64, //ldb
                      rndkvSplitSize, //ldc,
                      false,
                      query_t_padding_ptr,
                      key_reorder_ptr + n * qk_gemm_K +
                          b * qk_gemm_K,
                      qk_s32_data + b);
              }

              // do dequant compensation, add mask, max reduce for softmax, and convert qk from s32 to fp32
              accum_t* qk_block_data = qk_data + l * qSplitSize * rndkvSplitSize;
              if (has_attn_mask) {
                mask_t* mask_data_offset = mask_data + i * mStrideB + j * mStrideH + m * mStrideM + (mStrideN == 0 ? 0 : n);
                _dequant_mask_max_fusion_kernel(
                  qk_s32_data, //in
                  mask_data_offset, //mask_ptr
                  q_sum_ptr, //sum_a_ptr
                  k_sum_ptr + n, //sum_b_ptr
                  qBlockSize, //M
                  kvBlockSize, //N
                  rndkvSplitSize, //ldi
                  mStrideM, //ldm
                  rndkvSplitSize, //ldo
                  q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                  q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                  qk_block_data, //out
                  sfm_max_ptr // sfm_max_ptr
                );
              } else {
                _dequant_max_fusion_kernel(
                  qk_s32_data, //in
                  q_sum_ptr, //sum_a_ptr
                  k_sum_ptr + n, //sum_b_ptr
                  qBlockSize, //M
                  kvBlockSize, //N
                  rndkvSplitSize, //ldi
                  rndkvSplitSize, //ldo
                  q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                  q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                  qk_block_data, //out
                  sfm_max_ptr // sfm_max_ptr
                );
              }
            }
            // sub max, exp, sum reduce, div sum for softmax
            // and quant
            // and sum for attention
            if (v_zp == 0) {
              _sub_exp_sum_div_quant_fusion_kernel(
                qk_data, //in
                qBlockSize, //M
                kvSplitSize, //N_step
                rkvSlice, //NSlices
                qSplitSize * rndkvSplitSize, //ldi
                qk_reduce_strideL, //ldo
                kvSize, //kvSize
                rndkvSplitSize, //rndkvSplitSize
                av_gemm_K, //av_gemm_K
                a_zp, // zp_a=beta1
                a_scale, // scale_a=alpha
                qk_local_data, //local
                qk_reduced_data, //out
                sfm_max_ptr, //sfm_max_ptr
                sfm_sum_ptr //sfm_sum_ptr
              );
            } else {
              _sub_exp_sum_div_quant_sum_fusion_kernel(
                qk_data, //in
                qBlockSize, //M
                kvSplitSize, //N_step
                rkvSlice, //NSlice
                qSplitSize * rndkvSplitSize, //ldi
                qk_reduce_strideL, //ldo
                kvSize, //kvSize
                rndkvSplitSize, //rndkvSplitSize
                av_gemm_K, //av_gemm_K
                a_zp, // zp_a=beta1
                v_zp, // zp_b=beta2
                a_scale, // scale_a=alpha
                qk_local_data, //local
                qk_reduced_data, //out
                sfm_max_ptr, //sfm_max_ptr
                sfm_sum_ptr, //sfm_sum_ptr
                a_sum_ptr //a_sum_ptr
              );
            }
            // Calculate Softmax(q @ k.T) @ v
            for (int64_t b = 0; b < headSize; b += block_64) {
              auto value_reorder_b = value_reorder_ptr + b * av_gemm_K;
              auto dst_s32_b = dst_s32_data + b;
              for (int64_t s = 0; s < kvSlice; s++) {
                at::native::cpublas::brgemm(
                    qSplitSize, block_64, av_gemm_K,
                    av_gemm_K, // lda
                    rndHeadSize, //ldb
                    rndHeadSize, //ldc
                    s != 0,
                    qk_reduced_data + s * qk_reduce_strideL,
                    value_reorder_b + s * v_reorder_strideL,
                    dst_s32_b);
              }
            }

            // After the last gemm,
            // do dequant compensation, quant and convert from s32 to int8
            if (a_zp == 0) {
              _dequant_quant_fusion_kernel(
                dst_s32_data, //in
                a_sum_ptr, //sum_a_ptr
                qBlockSize, //M
                headSize, //N
                rndHeadSize, //ldi
                oStrideM, //ldo
                o_zp, //zp_c=beta2
                a_scale * v_scale / o_scale, //scale_a*scale_b/scale_c=alpha
                out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
              );
            } else {
              _dequant_quant_fusion_kernel(
                dst_s32_data, //in
                a_sum_ptr, //sum_a_ptr
                v_sum_ptr, //sum_b_ptr
                qBlockSize, //M
                headSize, //N
                rndHeadSize, //ldi
                oStrideM, //ldo
                a_zp * v_zp * kvSize, //zp_a*zp_b*k=beta1
                o_zp, //zp_c=beta2
                a_scale * v_scale / o_scale, //scale_a*scale_b/scale_c=alpha
                out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
              );
            }
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head);
        }
      });
  // Once all computations are done, need to release HW context.
  at::native::cpublas::brgemm_release();
}

// UINT8 - several parallel loops with u8u8s32 GEMM
template <typename scalar_t, typename mask_t,
          int64_t q_split_size, int64_t kv_split_size,
          bool use_one_parallel_loop,
          typename std::enable_if_t<!use_one_parallel_loop, int> = 0>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char>, void>
int8_sdpa_fused_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    std::optional<at::Tensor> attention_mask,
    std::optional<double> scale,
    float q_scale,
    int32_t q_zp,
    float k_scale,
    int32_t k_zp,
    float v_scale,
    int32_t v_zp,
    float a_scale,
    int32_t a_zp,
    float o_scale,
    int32_t o_zp) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  using accum_t = float;
  accum_t scaling_factor = calculate_scale(query, scale).expect_float();
  int block_64 = 64;
  auto u8_dt = at::ScalarType::Byte;

  // Sizes
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_sdpa: Q/K/V should have the same head size");
  TORCH_CHECK(
      kv_split_size % block_64 == 0, "kv_split_size is not divisble by ", block_64);

  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  bool has_attn_mask = attention_mask.has_value() && attention_mask.value().numel();
  if (has_attn_mask) {
    reshape_attn_mask_to_4d(attention_mask.value(), batchSize, num_head, qSize, kvSize);
  }

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t mStrideB =
      (has_attn_mask && attention_mask.value().size(0) > 1)
      ? attention_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (has_attn_mask && attention_mask.value().size(1) > 1)
      ? attention_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      (has_attn_mask && attention_mask.value().size(2) > 1)
      ? attention_mask.value().stride(2)
      : 0;
  int64_t mStrideN =
      (has_attn_mask && attention_mask.value().size(3) > 1)
      ? attention_mask.value().stride(3)
      : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  int64_t rndHeadSize = (headSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSplitSize = (kvSplitSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvTail = (kvTail + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSize = kv_split_size > kvSize ? rndkvTail : rndkvSplitSize * kvSlice + rndkvTail;

  bool av_gemm_K_mul4 = kvSplitSize % 4 == 0;
  int av_gemm_K_padding = av_gemm_K_mul4 ? 0 : 4 - kvSplitSize % 4;
  int av_gemm_K = kvSplitSize + av_gemm_K_padding;

  // Data ptrs
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  mask_t* mask_data = attention_mask.has_value()
      ? attention_mask.value().data_ptr<mask_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();

  bool headSize_mul64 = headSize % 64 == 0;
  int qk_gemm_K_padding = headSize_mul64 ? 0 : 64 - headSize % 64;
  int qk_gemm_K = headSize + qk_gemm_K_padding;

  int64_t qk_reduce_strideL = qSplitSize * av_gemm_K;
  int64_t v_reorder_strideL = av_gemm_K * rndHeadSize;

  int64_t total_size_uint8_per_thread =
    /* qk */ kvSlice * qSplitSize * rndkvSplitSize * 4 +
    /* qk_local  */ kvSlice * av_gemm_K * 4 +
    /* qk_reduce  */ kvSlice * qk_reduce_strideL +
    /* qk_s32   */ qSplitSize * rndkvSplitSize * 4 +
    /* dst_s32  */ qSplitSize * rndHeadSize * 4 +
    /* softmax_sum   */ qSplitSize * 4 +
    /* query_sum     */ qSplitSize * 4 +
    /* attention_sum */ qSplitSize * 4 +
    /* softmax max */ qSplitSize * 4 +
    /* query_padding_data */ qSplitSize * qk_gemm_K;

  at::Tensor total_buf = at::empty(
      {num_thread, total_size_uint8_per_thread},
      query.options());
  scalar_t* total_buf_data = total_buf.data_ptr<scalar_t>();

  int64_t kv_sum_size_per_BH =
    /* key_sum */ kvSize +
    /* value_sum */ headSize;

  at::Tensor kv_sum_buf = at::empty(
      {batchSize, num_head, kv_sum_size_per_BH},
      query.options().dtype(at::kInt));
  int32_t* kv_sum_buf_data = kv_sum_buf.data_ptr<int32_t>();

  int64_t kv_reorder_size_per_BH =
    /* key_t_reorder */ qk_gemm_K * rndkvSize +
    /* value_t_reorder */ kvSlice * v_reorder_strideL;

  at::Tensor kv_reorder_buf = at::empty(
      {batchSize, num_head, kv_reorder_size_per_BH},
      query.options());
  scalar_t* kv_reorder_buf_data = kv_reorder_buf.data_ptr<scalar_t>();
  scalar_t* key_reorder_ptr = kv_reorder_buf_data;
  scalar_t* value_reorder_ptr = kv_reorder_buf_data + batchSize * num_head * qk_gemm_K * rndkvSize;

  // sum k and v
  at::parallel_for(
      0, batchSize * num_head, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head);
        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          int32_t* kv_sum_ptr = kv_sum_buf_data
              + i * num_head * kv_sum_size_per_BH
              + j * kv_sum_size_per_BH;
          int32_t* k_sum_ptr = kv_sum_ptr;
          int32_t* v_sum_ptr = kv_sum_ptr + kvSize;
          if (q_zp == 0) {
            fill_stub(k_sum_ptr, static_cast<int32_t>(0), kvSize);
          } else {
            _int_sum_b_contiguous_kernel(k_data + i * kStrideB + j * kStrideH,
              k_sum_ptr,
              kvSize, headSize, kStrideN, q_zp);
          }
          if (a_zp == 0) {
            fill_stub(v_sum_ptr, static_cast<int32_t>(0), headSize);
          } else {
            _int_sum_a_contiguous_kernel(v_data + i * vStrideB + j * vStrideH,
              v_sum_ptr,
              headSize, kvSize, vStrideN, a_zp);
          }
        // Move to the next query
        at::native::data_index_step(i, batchSize, j, num_head);
      }
    });

  // transpose and packing
  at::parallel_for(
    0, batchSize * num_head * kvSlice, 1, [&](int64_t begin, int64_t end) {
      int64_t i = 0, j = 0, l = 0, n = 0;
      at::native::data_index_init(
          begin, i, batchSize, j, num_head, l, kvSlice);
      uint8_t* B_blocked_xform_u8 = new uint8_t[qk_gemm_K * block_64];
      for (const auto z : c10::irange(begin, end)) {
        (void)z; // Suppress unused variable
        n = l * kvSplitSize;
        auto k_reorder = key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                      j * qk_gemm_K * rndkvSize + n * qk_gemm_K;
        auto v_reorder = value_reorder_ptr +
                      i * num_head * kvSlice * v_reorder_strideL +
                      j * kvSlice * v_reorder_strideL + n * rndHeadSize;
        int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
        for (int64_t b = 0; b < kvBlockSize; b += block_64) {
          bool istail = kvBlockSize - b < block_64;
          int64_t trans_rows = istail ? kvBlockSize - b : block_64;
          do_transpose(
              k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
              B_blocked_xform_u8,
              trans_rows,
              headSize,
              kStrideN,
              block_64);
          if (!headSize_mul64 || istail) {
            pad_remain_row_col(
                B_blocked_xform_u8,
                headSize,
                trans_rows,
                qk_gemm_K,
                block_64,
                block_64
              );
          }
          at::native::cpublas::pack(
                  qk_gemm_K, // K
                  block_64, // N
                  block_64, // ld_in
                  block_64, // ld_out
                  u8_dt, // dt_in
                  u8_dt, // dt_out
                  B_blocked_xform_u8,
                  k_reorder + b * qk_gemm_K);
        }
        // split headSize to block_64, block_64, block_64 ...
        // [av_gemm_K, headSize] -> [av_gemm_K,  block_64 ...]
        for (int64_t b = 0; b < rndHeadSize; b += block_64) {
          at::native::cpublas::pack(
                  av_gemm_K,
                  block_64,
                  vStrideN,
                  block_64,
                  u8_dt,
                  u8_dt,
                  v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                  v_reorder + av_gemm_K * b);
        }
        // Move to the next query
        at::native::data_index_step(i, batchSize, j, num_head, l, kvSlice);
      }
    });

  at::parallel_for(
      0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, k = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        scalar_t* total_buf_ptr = total_buf_data + ompIdx * total_size_uint8_per_thread;
        int32_t offset = 0;
        accum_t* qk_data = reinterpret_cast<accum_t*>(total_buf_ptr);
        offset += kvSlice * qSplitSize * rndkvSplitSize * 4;
        accum_t* qk_local_data = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += kvSlice * av_gemm_K * 4;
        scalar_t* qk_reduced_data = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);
        offset += kvSlice * qk_reduce_strideL;
        int32_t* qk_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndkvSplitSize * 4;
        int32_t* dst_s32_data = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * rndHeadSize * 4;
        accum_t* sfm_sum_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* q_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        int32_t* a_sum_ptr = reinterpret_cast<int32_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        accum_t* sfm_max_ptr = reinterpret_cast<accum_t*>(total_buf_ptr + offset);
        offset += qSplitSize * 4;
        scalar_t* query_t_padding_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable

          int32_t* kv_sum_ptr = kv_sum_buf_data
              + i * num_head * kv_sum_size_per_BH
              + j * kv_sum_size_per_BH;
          int32_t* k_sum_ptr = kv_sum_ptr;
          int32_t* v_sum_ptr = kv_sum_ptr + kvSize;

          // sdpa core
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize sum and max
          fill_stub(
              sfm_sum_ptr, static_cast<accum_t>(0), qSplitSize);
          fill_stub(
              a_sum_ptr, static_cast<int32_t>(0), qSplitSize);
          fill_stub(
              sfm_max_ptr, static_cast<accum_t>(-std::numeric_limits<accum_t>::infinity()), qSplitSize);
          copy_value_with_pad(
              q_data + i * qStrideB + j * qStrideH + m * qStrideM,
              query_t_padding_ptr,
              qBlockSize,
              headSize,
              qSplitSize,
              qk_gemm_K,
              qStrideM);
          // sum q
          if (k_zp != 0) {
            _int_sum_b_contiguous_kernel(query_t_padding_ptr,
                  q_sum_ptr, qBlockSize, headSize, qk_gemm_K, k_zp);
          } else {
            fill_stub(
              q_sum_ptr, static_cast<int32_t>(0), qSplitSize);
          }
          const int64_t rkvSlice = (kvSize - 1) / kvSplitSize + 1;
          for (int64_t l = 0; l < rkvSlice; l++) {
            int64_t n = l * kvSplitSize;
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            auto k_reorder = key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                      j * qk_gemm_K * rndkvSize + n * qk_gemm_K;
            // Calculate q @ k.T
            for (int64_t b = 0; b < kvBlockSize; b += block_64) {
              at::native::cpublas::brgemm(
                    qSplitSize, block_64, qk_gemm_K,
                    qk_gemm_K, // lda
                    block_64, //ldb
                    rndkvSplitSize, //ldc,
                    false,
                    query_t_padding_ptr,
                    k_reorder + b * qk_gemm_K,
                    qk_s32_data + b);
            }

            // do dequant compensation, add mask, max reduce for softmax, and convert qk from s32 to fp32
            accum_t* qk_block_data = qk_data + l * qSplitSize * rndkvSplitSize;
            if (has_attn_mask) {
              mask_t* mask_data_offset = mask_data + i * mStrideB + j * mStrideH + m * mStrideM + (mStrideN == 0 ? 0 : n);
              _dequant_mask_max_fusion_kernel(
                qk_s32_data, //in
                mask_data_offset, //mask_ptr
                q_sum_ptr, //sum_a_ptr
                k_sum_ptr + n, //sum_b_ptr
                qBlockSize, //M
                kvBlockSize, //N
                rndkvSplitSize, //ldi
                mStrideM, //ldm
                rndkvSplitSize, //ldo
                q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                qk_block_data, //out
                sfm_max_ptr // sfm_max_ptr
              );
            } else {
              _dequant_max_fusion_kernel(
                qk_s32_data, //in
                q_sum_ptr, //sum_a_ptr
                k_sum_ptr + n, //sum_b_ptr
                qBlockSize, //M
                kvBlockSize, //N
                rndkvSplitSize, //ldi
                rndkvSplitSize, //ldo
                q_zp * k_zp * headSize, //zp_a*zp_b*k=beta
                q_scale * k_scale * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                qk_block_data, //out
                sfm_max_ptr // sfm_max_ptr
              );
            }
          }
          // sub max, exp, sum reduce, div sum for softmax
          // and quant
          // and sum for attention
          if (v_zp == 0) {
            _sub_exp_sum_div_quant_fusion_kernel(
              qk_data, //in
              qBlockSize, //M
              kvSplitSize, //N_step
              rkvSlice, //NSlices
              qSplitSize * rndkvSplitSize, //ldi
              qk_reduce_strideL, //ldo
              kvSize, //kvSize
              rndkvSplitSize, //rndkvSplitSize
              av_gemm_K, //av_gemm_K
              a_zp, // zp_a=beta1
              a_scale, // scale_a=alpha
              qk_local_data, //local
              qk_reduced_data, //out
              sfm_max_ptr, //sfm_max_ptr
              sfm_sum_ptr //sfm_sum_ptr
            );
          } else {
            _sub_exp_sum_div_quant_sum_fusion_kernel(
              qk_data, //in
              qBlockSize, //M
              kvSplitSize, //N_step
              rkvSlice, //NSlice
              qSplitSize * rndkvSplitSize, //ldi
              qk_reduce_strideL, //ldo
              kvSize, //kvSize
              rndkvSplitSize, //rndkvSplitSize
              av_gemm_K, //av_gemm_K
              a_zp, // zp_a=beta1
              v_zp, // zp_b=beta2
              a_scale, // scale_a=alpha
              qk_local_data, //local
              qk_reduced_data, //out
              sfm_max_ptr, //sfm_max_ptr
              sfm_sum_ptr, //sfm_sum_ptr
              a_sum_ptr //a_sum_ptr
            );
          }
          // Calculate Softmax(q @ k.T) @ v
          auto v_reorder = value_reorder_ptr +
                  i * num_head * kvSlice * v_reorder_strideL +
                  j * kvSlice * v_reorder_strideL;
          for (int64_t b = 0; b < headSize; b += block_64) {
            auto value_reorder_b = v_reorder + b * av_gemm_K;
            auto dst_s32_b = dst_s32_data + b;
            for (int64_t s = 0; s < kvSlice; s++) {
              at::native::cpublas::brgemm(
                  qSplitSize, block_64, av_gemm_K,
                  av_gemm_K, // lda
                  rndHeadSize, //ldb
                  rndHeadSize, //ldc
                  s != 0,
                  qk_reduced_data + s * qk_reduce_strideL,
                  value_reorder_b + s * v_reorder_strideL,
                  dst_s32_b);
            }
          }

          // After the last gemm,
          // do dequant compensation, quant and convert from s32 to int8
          if (a_zp == 0) {
            _dequant_quant_fusion_kernel(
              dst_s32_data, //in
              a_sum_ptr, //sum_a_ptr
              qBlockSize, //M
              headSize, //N
              rndHeadSize, //ldi
              oStrideM, //ldo
              o_zp, //zp_c=beta2
              a_scale * v_scale / o_scale, //scale_a*scale_b/scale_c=alpha
              out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
            );
          } else {
            _dequant_quant_fusion_kernel(
              dst_s32_data, //in
              a_sum_ptr, //sum_a_ptr
              v_sum_ptr, //sum_b_ptr
              qBlockSize, //M
              headSize, //N
              rndHeadSize, //ldi
              oStrideM, //ldo
              a_zp * v_zp * kvSize, //zp_a*zp_b*k=beta1
              o_zp, //zp_c=beta2
              a_scale * v_scale / o_scale, //scale_a*scale_b/scale_c=alpha
              out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
            );
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
        }
      });
  // Once all computations are done, need to release HW context.
  at::native::cpublas::brgemm_release();
}

// FP8 - one parallel loop with f8f8f8 GEMM 
template <typename scalar_t, typename mask_t,
          int64_t q_split_size, int64_t kv_split_size>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::Float8_e4m3fn>, void>
fp8_sdpa_fused_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    std::optional<at::Tensor> attn_mask,
    std::optional<double> scale,
    float q_scale,
    float k_scale,
    float v_scale,
    float a_scale,
    float o_scale) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  using accum_t = float;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).expect_float();

  // Sizes
  TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
        "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  bool has_attn_mask = attn_mask.has_value() && attn_mask.value().numel();
  if (has_attn_mask) {
    reshape_attn_mask_to_4d(attn_mask.value(), batchSize, num_head, qSize, kvSize);
  }

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t mStrideB =
      (has_attn_mask && attn_mask.value().size(0) > 1)
      ? attn_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (has_attn_mask && attn_mask.value().size(1) > 1)
      ? attn_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      (has_attn_mask && attn_mask.value().size(2) > 1)
      ? attn_mask.value().stride(2)
      : 0;
  int64_t mStrideN =
      (has_attn_mask && attn_mask.value().size(3) > 1)
      ? attn_mask.value().stride(3)
      : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
  int64_t kvSlice = (kvSize + kvSplitSize - 1) / kvSplitSize;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  // Pad is needed for packing when K is not even
  bool headSize_even = headSize % 4 == 0;
  int64_t eheadSize = !headSize_even ? headSize + 4 - headSize % 4: headSize;
  int64_t ekvSplitSize = (kvSplitSize % 4 != 0) ? kvSplitSize + 4 - kvSplitSize % 4 : kvSplitSize;
  int64_t ekvTail = (kvTail % 4 != 0) ? kvTail + 4 - kvTail % 4 : kvTail;

  // Allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;

  at::Tensor buf = at::empty({num_thread, size_per_thread}, query.options().dtype(at::kFloat));
  at::Tensor buf_reduced = at::empty(
    {num_thread,
     qSplitSize,
     ekvSplitSize},
     query.options());

  // Data ptrs
  const scalar_t* q_data = query.const_data_ptr<scalar_t>();
  const scalar_t* k_data = key.const_data_ptr<scalar_t>();
  const scalar_t* v_data = value.const_data_ptr<scalar_t>();
  mask_t* mask_data = has_attn_mask
      ? attn_mask.value().data_ptr<mask_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();
  // accum_t* lse_data = logsumexp.data_ptr<accum_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data = buf_reduced.data_ptr<scalar_t>();

  // Buffer to store padding query and packing key/value
  int64_t kv_padding_size = (kvSize - 1) / kvSplitSize * ekvSplitSize + ekvTail;
  at::Tensor key_t_reorder = at::empty(
    {batchSize, num_head, eheadSize, kvSize},
    c10::CppTypeToScalarType<scalar_t>::value);
  at::Tensor value_t_reorder = at::empty(
    {batchSize, num_head, kv_padding_size, headSize},
    c10::CppTypeToScalarType<scalar_t>::value);
  scalar_t* key_reorder_ptr = key_t_reorder.data_ptr<scalar_t>();
  scalar_t* value_reorder_ptr = value_t_reorder.data_ptr<scalar_t>();

  scalar_t* query_padding_ptr = nullptr;
  at::Tensor query_t_padding;
  if (!headSize_even) {
    query_t_padding = at::empty(
      {num_thread, qSplitSize, eheadSize},
      c10::CppTypeToScalarType<scalar_t>::value);
    query_padding_ptr = query_t_padding.data_ptr<scalar_t>();
  }

  // Reorder K, V
  at::Tensor tranpose_t_reorder = at::empty(
    {num_thread, kvSplitSize, headSize},
    c10::CppTypeToScalarType<scalar_t>::value);
  scalar_t* transpose_buffer_ptr = tranpose_t_reorder.data_ptr<scalar_t>();
  at::parallel_for(0, batchSize * num_head * kvSlice, 1, [&](int64_t begin, int64_t end) {
      int ompIdx = at::get_thread_num();
      int64_t i = 0, j = 0, l = 0, n = 0;
      scalar_t* transpose_ptr = transpose_buffer_ptr + ompIdx * kvSplitSize * headSize;
      at::native::data_index_init(begin, i, batchSize, j, num_head, l, kvSlice);
      for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
        n = l * kvSplitSize;
        int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);

        // transpose [kvBlockSize, headSize] -> [headSize, kvBlockSize]
        at::native::utils::transpose<uint8_t>(
            kvBlockSize,
            headSize,
            /* src */ reinterpret_cast<const uint8_t*>(k_data + i * kStrideB + j * kStrideH + n * kStrideN),
            /* ld_src */ kStrideN,
            /* dst */ reinterpret_cast<uint8_t*>(transpose_ptr),
            /* ld_dst */ kvBlockSize);

        // Pack [headSize, kvBlockSize]
        at::vec::pack_vnni4(
          /* src */ reinterpret_cast<const uint8_t*>(transpose_ptr),
          /* dst */ reinterpret_cast<uint8_t*>(key_reorder_ptr + i * num_head * eheadSize * kvSize +
                  j * eheadSize * kvSize + n * eheadSize),
          /* ld_src */ kvBlockSize,
          /* K */ headSize,
          /* N */ kvBlockSize);

        // Pack [kvBlockSize, headSize]
        at::vec::pack_vnni4(
          /* src */ reinterpret_cast<const uint8_t*>(v_data + i * vStrideB + j * vStrideH + n * vStrideN),
          /* dst */ reinterpret_cast<uint8_t*>(value_reorder_ptr +
                  i * num_head * kv_padding_size * headSize +
                  j * kv_padding_size * headSize + n * headSize),
          /* ld_src */ vStrideN,
          /* K */ kvBlockSize,
          /* N */ headSize);

        // Move to the next query
        at::native::data_index_step(i, batchSize, j, num_head, l, kvSlice);
      }
    });

  at::parallel_for(0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
    int64_t i = 0, j = 0, k = 0;
    at::native::data_index_init(begin, i, batchSize, j, num_head, k, qSlice);
    int ompIdx = at::get_thread_num();
    accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
    accum_t* qk_data = buf_ptr;
    accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize;
    accum_t* dst_data = qk_sum_data + qSplitSize;
    scalar_t* qk_reduced_data = buf_reduced_data + ompIdx * qSplitSize * ekvSplitSize;
    scalar_t* query_t_padding_ptr = !headSize_even
            ? query_padding_ptr + ompIdx * qSplitSize * eheadSize
            : nullptr;

    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t m = k * qSplitSize;
      int64_t qBlockSize = std::min(qSplitSize, qSize - m);
      // Initialize max and sum
      fill_stub(qk_max_data,
          -std::numeric_limits<accum_t>::infinity(), qBlockSize);
      fill_stub(qk_sum_data,
          static_cast<accum_t>(0), qBlockSize);
      int64_t num_keys = is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
      if (!headSize_even) {
        // Pad query if headSize is not even
        // [qBlockSize, headSize] -> [qBlockSize, eheadSize]
        copy_value_with_pad<scalar_t>(
          q_data + i * qStrideB + j * qStrideH + m * qStrideM,
          query_t_padding_ptr,
          qBlockSize,
          headSize,
          qBlockSize,
          eheadSize,
          qStrideM
        );
      }
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
        int64_t ekvBlockSize = (kvBlockSize % 4 != 0) ? kvBlockSize + 4 - kvBlockSize % 4 : kvBlockSize;
        // Calculate scale * q @ k.T
        at::native::cpublas::brgemm(
          qBlockSize,
          kvBlockSize,
          eheadSize,
          headSize_even ? qStrideM : eheadSize,
          kvBlockSize,
          kvBlockSize,
          false,
          !headSize_even
              ? query_t_padding_ptr
              : q_data + i * qStrideB + j * qStrideH + m * qStrideM,
          key_reorder_ptr + i * num_head * eheadSize * kvSize +
              j * eheadSize * kvSize + n * eheadSize,
          qk_data);
        // Apply causal mask, fill unused with -inf
        if (is_causal && num_keys - n <= kvSplitSize) {
          for (const auto row : c10::irange(qBlockSize)) {
            int64_t last_col = m + row - n;
            accum_t* row_ptr = qk_data + row * kvBlockSize;
            fill_stub(row_ptr + last_col + 1,
                -std::numeric_limits<accum_t>::infinity(),
                kvBlockSize - last_col - 1);
          }
        }
        // Update attention weights with attention mask
        // And apply scaling factor
        // qk <- qk * scaling + attn_mask
        if (has_attn_mask) {
          for (int64_t row = 0; row < qBlockSize; ++row) {
              if (mStrideN == 0) {
                _scale_dequant_attn_mask_fusion_kernel</*is_stride_0*/ true>(
                  qk_data + row * kvBlockSize,
                  mask_data + i * mStrideB + j * mStrideH +
                      (m + row) * mStrideM,
                  kvBlockSize,
                  qk_data + row * kvBlockSize,
                  scaling_factor * q_scale * k_scale);
              } else {
                _scale_dequant_attn_mask_fusion_kernel</*is_stride_0*/ false>(
                  qk_data + row * kvBlockSize,
                  mask_data + i * mStrideB + j * mStrideH +
                      (m + row) * mStrideM + n,
                  kvBlockSize,
                  qk_data + row * kvBlockSize,
                  scaling_factor * q_scale * k_scale);
              }
          }
        }
        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
        for (int64_t row = 0; row < qBlockSize; ++row) {
          if (has_attn_mask) {
            // max per row
            tmp_max = at::vec::reduce_all<accum_t>(
                [](Vec& x, Vec& y) { return at::vec::maximum(x, y); },
                qk_data + row * kvBlockSize,
                kvBlockSize);
          } else {
            // apply scaling factor and max per row in fusion
            _mul_reduce_max_fusion_kernel(
                qk_data + row * kvBlockSize,
                scaling_factor * q_scale * k_scale,
                kvBlockSize,
                qk_data + row * kvBlockSize,
                tmp_max);
          }
          tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
          if (tmp_max == -std::numeric_limits<accum_t>::infinity()) {
            // to avoid `nan = exp2f(-inf - (-inf))`
            fill_stub(qk_reduced_data + row * ekvBlockSize,
              static_cast<scalar_t>(0), kvBlockSize);
          } else {
            tmp_sum = tmp_max;
            // qk <- exp(qk - max) and sum per row
            _fp8_exp_reduce_sum_quant_fusion_kernel(
                qk_data + row * kvBlockSize, kvBlockSize,
                qk_reduced_data + row * ekvBlockSize,
                tmp_sum,
                1.0 / a_scale);
            // exp_tmp <- exp(max[row] - max)
            exp_tmp = std::exp(qk_max_data[row] - tmp_max);
            // sum[row] <- sum + exp_tmp * sum[row]
            qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
            // max[row] <- max
            qk_max_data[row] = tmp_max;
            // dst <- dst * exp_tmp
            if (n > 0) {
              at::vec::map<accum_t>(
                [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
                dst_data + row * headSize,
                dst_data + row * headSize,
                headSize);
            }
          }
          if (kvBlockSize % 4 != 0) {
            // Pad: [qSplitSize, kvBlockSize] -> [qSplitSize, kvBlockSize + 4 - kvBlockSize / 4]
            for (int64_t psize = kvBlockSize; psize < ekvBlockSize; ++psize) {
              *(qk_reduced_data + row * ekvBlockSize + psize) = scalar_t(0);
            }
          }
        }
        // Calculate Softmax(q @ k.T) @ v
        int64_t psize = n / kvSplitSize * ekvSplitSize;
        at::native::cpublas::brgemm(
          qBlockSize,
          headSize,
          ekvBlockSize,
          ekvBlockSize,
          headSize,
          headSize,
          n > 0,
          qk_reduced_data,
          value_reorder_ptr +
              i * num_head * kv_padding_size * headSize +
              j * kv_padding_size * headSize + psize * headSize,
          dst_data);
      }

      // dst <- dst / sum[row]
      // reorder MHA output with strides
      for (int64_t row = 0; row < qBlockSize; ++row) {
        // Row sums for full masked out rows are 0, we set them to 1
        // in order to avoid NaNs in the output and instead set fully
        // masked out rows to 0
        qk_max_data[row] = qk_max_data[row] == -std::numeric_limits<accum_t>::infinity() ? 0 : qk_max_data[row];
        qk_sum_data[row] = qk_sum_data[row] == 0 ? 1 : qk_sum_data[row];
        accum_t sum_reciprocal = 1 / qk_sum_data[row];
        _fp8_dequant_quant_fusion_kernel(
          dst_data + row * headSize,
          headSize,
          out_data + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM,
          sum_reciprocal * a_scale * v_scale / o_scale);
      }
      // Move to the next query
      at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
    at::native::cpublas::brgemm_release();
  });
}

template <typename scalar_t, typename mask_t, int64_t q_split_size, int64_t kv_split_size>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char>, void>
int8_sdpa_fused_kernel_impl(
    bool use_one_parallel_loop,
    const at::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    std::optional<at::Tensor> attn_mask,
    std::optional<double> scale,
    float q_scale,
    int32_t q_zp,
    float k_scale,
    int32_t k_zp,
    float v_scale,
    int32_t v_zp,
    float a_scale,
    int32_t a_zp,
    float o_scale,
    int32_t o_zp) {
  if (use_one_parallel_loop) {
    int8_sdpa_fused_kernel_impl<scalar_t, mask_t, q_split_size, kv_split_size,
          /* use_one_parallel_loop */ true>(
      output, query, key, value,
      dropout_p, is_causal, attn_mask, scale,
      q_scale, q_zp,
      k_scale, k_zp,
      v_scale, v_zp,
      a_scale, a_zp,
      o_scale, o_zp);
  } else {
    int8_sdpa_fused_kernel_impl<scalar_t, mask_t, q_split_size, kv_split_size,
          /* use_one_parallel_loop */ false>(
      output, query, key, value,
      dropout_p, is_causal, attn_mask, scale,
      q_scale, q_zp,
      k_scale, k_zp,
      v_scale, v_zp,
      a_scale, a_zp,
      o_scale, o_zp);
  }
}

#define AT_DISPATCH_MASK_TYPES(TYPE, NAME, ...)            \
  AT_DISPATCH_SWITCH(                                      \
      TYPE,                                                \
      NAME,                                                \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Bool, mask_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Float, mask_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Double, mask_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::BFloat16, mask_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Half, mask_t, __VA_ARGS__))

void int8_sdpa_fused_kernel(
    const at::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    std::optional<at::Tensor> attn_mask,
    std::optional<double> scale,
    float q_scale,
    int32_t q_zp,
    float k_scale,
    int32_t k_zp,
    float v_scale,
    int32_t v_zp,
    float a_scale,
    int32_t a_zp,
    float o_scale,
    int32_t o_zp) {
  TORCH_CHECK(query.scalar_type() == c10::kByte);
  int64_t batchSize = query.size(0);
  int64_t num_head = query.size(1);
  int64_t q_seq_len = query.size(2);
  int64_t kv_seq_len = key.size(2);
  int64_t q_split_size = 32;
  if (q_seq_len >= 768) {
    q_split_size = 256;
  } else if (q_seq_len >= 192) {
    q_split_size = 64;
  }
  // Heuristic to decide whether to use one parallel loop or not
  //    true: one parallel loop for sum+packing+core
  //    false: three parallel loops for sum, packing, core
  uint32_t l2_cache_size = at::cpu::L2_cache_size();
  int64_t num_thread = at::get_num_threads();
  int64_t attn_size = q_split_size * kv_seq_len * sizeof(int32_t) * num_thread;
  bool use_one_parallel_loop = (batchSize * num_head > num_thread) &&
      (attn_size > 1.5 * l2_cache_size);
  if (!attn_mask.has_value()) {
    if (q_split_size == 256) {
      int8_sdpa_fused_kernel_impl<unsigned char, float, 256, 64>(
        use_one_parallel_loop,
        output, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_scale, q_zp,
        k_scale, k_zp,
        v_scale, v_zp,
        a_scale, a_zp,
        o_scale, o_zp);
    } else if (q_split_size == 64) {
      int8_sdpa_fused_kernel_impl<unsigned char, float, 64, 64>(
        use_one_parallel_loop,
        output, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_scale, q_zp,
        k_scale, k_zp,
        v_scale, v_zp,
        a_scale, a_zp,
        o_scale, o_zp);
    } else {
      int8_sdpa_fused_kernel_impl<unsigned char, float, 32, 64>(
        use_one_parallel_loop,
        output, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_scale, q_zp,
        k_scale, k_zp,
        v_scale, v_zp,
        a_scale, a_zp,
        o_scale, o_zp);
    }
  } else {
    AT_DISPATCH_MASK_TYPES(attn_mask.value().scalar_type(), "sdpa_mask", [&]() {
      if (q_split_size == 256) {
        int8_sdpa_fused_kernel_impl<unsigned char, mask_t, 256, 64>(
          use_one_parallel_loop,
          output, query, key, value,
          dropout_p, is_causal, attn_mask, scale,
          q_scale, q_zp,
          k_scale, k_zp,
          v_scale, v_zp,
          a_scale, a_zp,
          o_scale, o_zp);
      } else if (q_split_size == 64) {
        int8_sdpa_fused_kernel_impl<unsigned char, mask_t, 64, 64>(
          use_one_parallel_loop,
          output, query, key, value,
          dropout_p, is_causal, attn_mask, scale,
          q_scale, q_zp,
          k_scale, k_zp,
          v_scale, v_zp,
          a_scale, a_zp,
          o_scale, o_zp);
      } else {
        int8_sdpa_fused_kernel_impl<unsigned char, mask_t, 32, 64>(
          use_one_parallel_loop,
          output, query, key, value,
          dropout_p, is_causal, attn_mask, scale,
          q_scale, q_zp,
          k_scale, k_zp,
          v_scale, v_zp,
          a_scale, a_zp,
          o_scale, o_zp);
      }
    });
  }
}

void fp8_sdpa_fused_kernel(
    const at::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    std::optional<at::Tensor> attn_mask,
    std::optional<double> scale,
    float q_scale,
    float k_scale,
    float v_scale,
    float a_scale,
    float o_scale) {
  TORCH_CHECK(query.scalar_type() == c10::kFloat8_e4m3fn);
  int64_t batchSize = query.size(0);
  int64_t num_head = query.size(1);
  int64_t q_seq_len = query.size(2);
  int64_t kv_seq_len = key.size(2);
  int64_t q_split_size = 32;
  if (q_seq_len >= 768) {
    q_split_size = 256;
  } else if (q_seq_len >= 192) {
    q_split_size = 64;
  }

  if (!attn_mask.has_value()) {
    if (q_split_size == 256) {
      fp8_sdpa_fused_kernel_impl<at::Float8_e4m3fn, float, 256, 64>(
        output, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_scale, k_scale,
        v_scale, a_scale,
        o_scale);
    } else if (q_split_size == 64) {
      fp8_sdpa_fused_kernel_impl<at::Float8_e4m3fn, float, 64, 64>(
        output, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_scale, k_scale,
        v_scale, a_scale,
        o_scale);
    } else {
      fp8_sdpa_fused_kernel_impl<at::Float8_e4m3fn, float, 32, 64>(
        output, query, key, value,
        dropout_p, is_causal, attn_mask, scale,
        q_scale, k_scale,
        v_scale, a_scale,
        o_scale);
    }
  } else {
    AT_DISPATCH_MASK_TYPES(attn_mask.value().scalar_type(), "sdpa_mask", [&]() {
      if (q_split_size == 256) {
        fp8_sdpa_fused_kernel_impl<at::Float8_e4m3fn, mask_t, 256, 64>(
          output, query, key, value,
          dropout_p, is_causal, attn_mask, scale,
          q_scale, k_scale,
          v_scale, a_scale,
          o_scale);
      } else if (q_split_size == 64) {
        fp8_sdpa_fused_kernel_impl<at::Float8_e4m3fn, mask_t, 64, 64>(
          output, query, key, value,
          dropout_p, is_causal, attn_mask, scale,
          q_scale, k_scale,
          v_scale, a_scale,
          o_scale);
      } else {
        fp8_sdpa_fused_kernel_impl<at::Float8_e4m3fn, mask_t, 32, 64>(
          output, query, key, value,
          dropout_p, is_causal, attn_mask, scale,
          q_scale, k_scale,
          v_scale, a_scale,
          o_scale);
      }
    });
  }
}
#endif // CPU_CAPABILITY_AVX512

at::Tensor int8_sdpa_math_kernel(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    std::optional<at::Tensor> attn_mask,
    std::optional<double> scale,
    float q_scale,
    int32_t q_zp,
    float k_scale,
    int32_t k_zp,
    float v_scale,
    int32_t v_zp,
    float a_scale,
    int32_t a_zp,
    float o_scale,
    int32_t o_zp) {
  // dequant q/k/v
  auto q = (query.to(at::kFloat) - q_zp) * q_scale;
  auto k = (key.to(at::kFloat) - k_zp) * k_scale;
  auto v = (value.to(at::kFloat) - v_zp) * v_scale;
  const auto scaling_factor = calculate_scale(q, scale);
  auto attn = at::matmul(q, k.transpose(-2, -1)) * scaling_factor;
  if (attn_mask.has_value() && attn_mask.value().numel()) {
    attn = attn.add(attn_mask.value().to(at::kFloat));
  }
  attn = at::softmax(attn, -1);
  // quant attn
  attn = at::clamp_max(
      at::clamp_min(at::round(attn / a_scale) + a_zp, 0), 255
  );
  // dequant attn
  attn = (attn - a_zp) * a_scale;
  auto output = at::matmul(attn, v);
  // quant output
  output = at::clamp_max(
      at::clamp_min(at::round(output / o_scale) + o_zp, 0), 255
  ).to(at::kByte);
  return output;
}

at::Tensor fp8_sdpa_math_kernel(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    std::optional<at::Tensor> attn_mask,
    std::optional<double> scale,
    float q_scale,
    float k_scale,
    float v_scale,
    float a_scale,
    float o_scale) {
  // dequant q/k/v
  auto q = query.to(at::kFloat) * q_scale;
  auto k = key.to(at::kFloat) * k_scale;
  auto v = value.to(at::kFloat) * v_scale;
  const auto scaling_factor = calculate_scale(q, scale);
  auto attn = at::matmul(q, k.transpose(-2, -1)) * scaling_factor;
  if (attn_mask.has_value() && attn_mask.value().numel()) {
    attn = attn.add(attn_mask.value().to(at::kFloat));
  }
  attn = at::softmax(attn, -1);
  // quant attn
  attn = at::clamp_max(
      at::clamp_min(attn / a_scale, -448), 448
  );
  attn = attn.to(at::kFloat8_e4m3fn).to(at::kFloat);
  // dequant attn
  attn = attn * a_scale;
  auto output = at::matmul(attn, v);
  // quant output
  output = at::clamp_max(
      at::clamp_min(output / o_scale, -448), 448
  ).to(at::kFloat8_e4m3fn);
  return output;
}

at::Tensor _qscaled_dot_product_cpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    std::optional<at::Tensor> attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    double q_scale,
    int64_t q_zp,
    double k_scale,
    int64_t k_zp,
    double v_scale,
    int64_t v_zp,
    double a_scale,
    int64_t a_zp,
    double o_scale,
    int64_t o_zp) {
  const auto dtype = query.scalar_type();
  TORCH_CHECK(!query.is_nested() && !key.is_nested() && !value.is_nested(),
    "_qscaled_dot_product_cpu: Only accept plain inputs");
  TORCH_CHECK(!is_causal,
    "_qscaled_dot_product_cpu: is_causal not supported.");
  TORCH_CHECK(dtype == at::ScalarType::Byte || dtype == at::ScalarType::Float8_e4m3fn,
    "_qscaled_dot_product_cpu: Expected data type be U8 or Float8_e4m3, but got ", dtype, " instead.");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
    "_qscaled_dot_product_cpu: Accept only 4 dims inputs shape of {B, H, T, K}");
  TORCH_CHECK(dropout_p == 0.0,
    "_qscaled_dot_product_cpu: Currently do not support dropout > 0");
  TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
    "_qscaled_dot_product_cpu: Q/K/V should have the same head size");
  TORCH_CHECK(!attn_mask.has_value() ||
          attn_mask.value().scalar_type() == at::kFloat ||
          attn_mask.value().scalar_type() == at::kBFloat16,
    "_qscaled_dot_product_cpu: Expected attention mask be float or bf16");
  TORCH_CHECK(!attn_mask.has_value() ||
          (attn_mask.value().dim() == 2 || attn_mask.value().dim() == 4),
    "_qscaled_dot_product_cpu: Attention mask dim in {2, 4}");
  if (dtype == at::ScalarType::Float8_e4m3fn) {
    TORCH_CHECK(q_zp == 0 && k_zp == 0 && v_zp == 0 && a_zp == 0 && o_zp == 0,
      "_qscaled_dot_product_cpu: Don't accept zero point for Float8_e4m3");
  }

  #ifdef CPU_CAPABILITY_AVX512
    if (at::native::cpublas::could_pack(dtype)) {
        at::Tensor output = at::empty_like(query, query.options()).transpose(1, 2);
        if (dtype == at::ScalarType::Byte) {
          int8_sdpa_fused_kernel(output, query, key, value,
            dropout_p, is_causal, attn_mask, scale,
            q_scale, q_zp,
            k_scale, k_zp,
            v_scale, v_zp,
            a_scale, a_zp,
            o_scale, o_zp);
        } else if (dtype == at::ScalarType::Float8_e4m3fn) {
          fp8_sdpa_fused_kernel(output, query, key, value,
            dropout_p, is_causal, attn_mask, scale,
            q_scale, k_scale,
            v_scale, a_scale,
            o_scale);
        } else {
          TORCH_CHECK(false, "_qscaled_dot_product_cpu: Unsupported data type ", dtype);
        }
        return output.transpose(1, 2);
    } else {
  #endif // CPU_CAPABILITY_AVX512
        if (dtype == at::ScalarType::Byte) {
          return int8_sdpa_math_kernel(query, key, value,
              dropout_p, is_causal, attn_mask, scale,
              q_scale, q_zp,
              k_scale, k_zp,
              v_scale, v_zp,
              a_scale, a_zp,
              o_scale, o_zp).transpose(1, 2).contiguous().transpose(1, 2);
        } else if (dtype == at::ScalarType::Float8_e4m3fn) {
          return fp8_sdpa_math_kernel(query, key, value,
              dropout_p, is_causal, attn_mask, scale,
              q_scale, k_scale,
              v_scale, a_scale,
              o_scale).transpose(1, 2).contiguous().transpose(1, 2);
        }
        TORCH_CHECK(false, "_qscaled_dot_product_cpu: Unsupported data type ", dtype);
  #ifdef CPU_CAPABILITY_AVX512
    }
  #endif // CPU_CAPABILITY_AVX512
}


} // anonymous namespace

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::qscaled_dot_product", &_qscaled_dot_product_cpu);
}

// } // at::native
} // namespace torchao
