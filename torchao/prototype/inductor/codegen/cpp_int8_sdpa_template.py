from typing import List, Optional

import torch
import torch.utils
from sympy import sympify
from torch._inductor import ir
from torch._inductor.codegen.cpp_flex_attention_template import CppFlexAttentionTemplate
from torch._inductor.codegen.cpp_template import CppTemplate
from torch._inductor.ir import TensorBox
from torch._inductor.select_algorithm import DataProcessorTemplateWrapper
from torch._inductor.utils import parallel_num_threads

from .utils import expand

USEFUL_FUNCTIONS = r"""
inline float calculate_scale(
    int64_t headSize,
    std::optional<double> scale) {
  return scale.has_value()
      ? scale.value()
      : (1.0 / std::sqrt(headSize));
}

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

template <typename scalar_t>
inline void store(scalar_t* dst, at::vec::Vectorized<scalar_t> src, int size=at::vec::Vectorized<scalar_t>::size()) {
  src.store(dst, size);
}

template <typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, unsigned char> || std::is_same_v<scalar_t, signed char>, void>
store(scalar_t* dst, at::vec::Vectorized<float> src, int size=at::vec::Vectorized<float>::size()) {
  auto res = at::vec::convert<scalar_t>(src);
  res.store(dst, size);
}

/*
1. dequant
2. add mask
3. max reduce for softmax
*/
template <typename mask_t>
inline void dequant_mask_max_fusion_kernel(
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
      store(tmp_out + col, tmp8);
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
      store(tmp_out + col, tmp8, N - col);
      vec_tmp_max = at::vec::Vectorized<float>::set(vec_tmp_max, at::vec::clamp_min(vec_tmp_max, tmp8), N - col);
    }
    sfm_max_ptr[row] = std::max(sfm_max_ptr[row], vec_tmp_max.reduce_max());
  }
}

/*
1. dequant
2. max reduce for softmax
*/
inline void dequant_max_fusion_kernel(
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
      store(tmp_out + col, tmp5);
    }
    if (col < N) {
      auto vec_sum_b = at::vec::Vectorized<int32_t>::loadu(sum_b_ptr + col, N - col);
      auto tmp0 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col, N - col);
      auto tmp1 = tmp0 - vec_sum_b;
      auto tmp2 = tmp1 - vec_sum_a;
      auto tmp3 = tmp2 + vec_beta;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      store(tmp_out + col, tmp5, N - col);
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
inline void sub_exp_sum_div_quant_sum_fusion_kernel(
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
        store(tmp_out + col, tmp2);
      }
      if (col < kvBlockSize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col, kvBlockSize - col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        store(tmp_out + col, tmp2, kvBlockSize - col);
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
        store(tmp_out + col, tmp4);
        auto tmp6 = at::vec::convert<int32_t>(tmp4);
        vec_tmp_sum += tmp6;
      }
      if (col < kvBlockSize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col, kvBlockSize - col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
        store(tmp_out + col, tmp4, kvBlockSize - col);
        auto tmp6 = at::vec::convert<int32_t>(tmp4);
        vec_tmp_sum = at::vec::Vectorized<int32_t>::set(vec_tmp_sum, vec_tmp_sum + tmp6, kvBlockSize - col);
      }
      sum_a_ptr[row] += vec_tmp_sum.reduce_add() * beta2;
      // set zero
      col = kvBlockSize;
      for (; col <  vec_size * (av_gemm_K / vec_size); col += vec_size) {
        store(tmp_out + col, vec_zero);
      }
      if (col < av_gemm_K) {
        store(tmp_out + col, vec_zero, av_gemm_K - col);
      }
    }
  }
}

/*
1. Softmax: sub max, exp, sum reduce, div sum
2. quant
*/
template <typename scalar_t>
inline void sub_exp_sum_div_quant_fusion_kernel(
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
        store(tmp_out + col, tmp2);
      }
      if (col < kvBlockSize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col, kvBlockSize - col);
        auto tmp1 = tmp0 - vec_max;
        auto tmp2 = tmp1.exp_u20();
        vec_tmp_sum = at::vec::Vectorized<float>::set(vec_tmp_sum, vec_tmp_sum + tmp2, kvBlockSize - col);
        store(tmp_out + col, tmp2, kvBlockSize - col);
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
        store(tmp_out + col, tmp4);
      }
      if (col < kvBlockSize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(tmp_in + col, kvBlockSize - col);
        auto tmp1 = tmp0 * vec_sum_scale;
        auto tmp2 = tmp1.round();
        auto tmp3 = tmp2 + vec_beta1;
        auto tmp4 = at::vec::clamp(tmp3, vec_min_val, vec_max_val);
        store(tmp_out + col, tmp4, kvBlockSize - col);
      }
      // set zero
      col = kvBlockSize;
      for (; col < vec_size * (av_gemm_K / vec_size); col += vec_size) {
        store(tmp_out + col, vec_zero);
      }
      if (col < av_gemm_K) {
        store(tmp_out + col, vec_zero, av_gemm_K - col);
      }
    }
  }
}

/*
1. dequant
2. quant
*/
template <typename scalar_t>
inline void dequant_quant_fusion_kernel(
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
      store(tmp_out + col, tmp8);
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
      store(tmp_out + col, tmp8, N - col);
    }
  }
}

/*
1. dequant
2. quant
*/
template <typename scalar_t>
inline void dequant_quant_fusion_kernel(
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
      store(tmp_out + col, tmp8);
    }
    if (col < N) {
      auto tmp1 = at::vec::Vectorized<int32_t>::loadu(tmp_in + col, N - col);
      auto tmp3 = tmp1 - vec_sum_a;
      auto tmp4 = at::vec::convert<float>(tmp3);
      auto tmp5 = tmp4 * vec_alpha;
      auto tmp6 = tmp5.round();
      auto tmp7 = tmp6 + vec_beta2;
      auto tmp8 = at::vec::clamp(tmp7, vec_min_val, vec_max_val);
      store(tmp_out + col, tmp8, N - col);
    }
  }
}

template <typename scalar_t>
inline void int_sum_b_contiguous_kernel_helper(
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
inline void int_sum_b_contiguous_kernel(
    const scalar_t* in,
    int32_t* out,
    const int& M,
    const int& N,
    const int& ld,
    const int32_t& scale) {
  for (long r = 0; r < M; r += 1) {
    int_sum_b_contiguous_kernel_helper(in + r * ld, out + r, N, scale);
  }
}

// reduce along dim a for shape [a, b], with sum shape [b]
template <typename scalar_t>
inline void int_sum_a_contiguous_kernel(
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
    store(out + i, vec_zero);
  }
  if (i < M) {
    store(out + i, vec_zero, M - i);
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
      store(out + k, tmp3);
    }
    if (k < M) {
      auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(tmp_in + k, M - k);
      auto tmp1 = at::vec::Vectorized<int32_t>::loadu(out + k, M - k);
      auto tmp2 = at::vec::convert<int32_t>(tmp0);
      auto tmp3 = tmp1 + tmp2;
      store(out + k, tmp3, M - k);
    }
  }
  // scale
  i = 0;
  for (; i < vec_size * (M / vec_size); i += vec_size) {
    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out + i);
    auto tmp1 = tmp0 * vec_scale;
    store(out + i, tmp1);
  }
  if (i < M) {
    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out + i, M - i);
    auto tmp1 = tmp0 * vec_scale;
    store(out + i, tmp1, M - i);
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
"""


ALLOCATE_BUFFER = r"""
  auto& {{buffer_name}}_allocator = *at::getCPUAllocator();
  auto {{buffer_name}}_work_data = {{buffer_name}}_allocator.allocate({{buffer_size}} * sizeof({{buffer_dtype}}));
  void* {{buffer_name}}_data_ptr = {{buffer_name}}_work_data.get();
  {{buffer_dtype}}* {{buffer_name}} = ({{buffer_dtype}}*){{buffer_name}}_data_ptr;
"""


INT8_SDPA_ONE_LOOP_TEMPLATE = r"""
#ifndef HEADER_DEFINED
#define HEADER_DEFINED

{{template.header().getvalue()}}
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

#include <ATen/Tensor.h>
#include <limits>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

{{template.codegen_useful_function(kernel.kernel_name)}}

#endif

{%- if has_attention_mask %}
{%- set kernel_args = {"query": query, "key": key, "value": value,
                       "attention_mask": attention_mask} %}
{%- else %}
{%- set kernel_args = {"query": query, "key": key, "value": value} %}
{%- endif %}

// UINT8 - u8u8s32
extern "C"
{{kernel.def_kernel(inputs=kernel_args, outputs={"output": output})}}
{
  {{ kernel.maybe_codegen_profile() }}
  int64_t num_thread = {{num_thread}};
  using accum_t = float;
  using scalar_t = {{kernel.dtype(query)}};
  int block_64 = 64;
  auto u8_dt = at::ScalarType::Byte;

  // Sizes
  int64_t batchSize = {{kernel.size(query, 0)}};
  int64_t qSize = {{kernel.size(query, 1)}};
  int64_t kvSize = {{kernel.size(value, 1)}};
  int64_t num_head = {{kernel.size(query, 2)}};
  int64_t headSize = {{kernel.size(query, 3)}};
  float scaling_factor =
      calculate_scale(headSize, {{scale}});

  // Strides
  int64_t qStrideB = {{kernel.stride(query, 0)}};
  int64_t qStrideM = {{kernel.stride(query, 1)}};
  int64_t qStrideH = {{kernel.stride(query, 2)}};
  int64_t kStrideB = {{kernel.stride(key, 0)}};
  int64_t kStrideN = {{kernel.stride(key, 1)}};
  int64_t kStrideH = {{kernel.stride(key, 2)}};
  int64_t vStrideB = {{kernel.stride(value, 0)}};
  int64_t vStrideN = {{kernel.stride(value, 1)}};
  int64_t vStrideH = {{kernel.stride(value, 2)}};
  int64_t oStrideB = {{kernel.stride(output, 0)}};
  int64_t oStrideM = {{kernel.stride(output, 2)}};
  int64_t oStrideH = {{kernel.stride(output, 1)}};

  int64_t qSplitSize = {{q_split_size}} > qSize ? qSize : {{q_split_size}};
  int64_t kvSplitSize = {{kv_split_size}} > kvSize ? kvSize : {{kv_split_size}};
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;

  int64_t rndHeadSize = (headSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSplitSize = (kvSplitSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvTail = (kvTail + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSize = {{kv_split_size}} > kvSize ? rndkvTail : rndkvSplitSize * kvSlice + rndkvTail;

  bool av_gemm_K_mul4 = kvSplitSize % 4 == 0;
  int av_gemm_K_padding = av_gemm_K_mul4 ? 0 : 4 - kvSplitSize % 4;
  int av_gemm_K = kvSplitSize + av_gemm_K_padding;

{%- if has_attention_mask %}
  // attention mask
  using mask_t = {{kernel.dtype(attention_mask)}};
  const mask_t* mask_data = attention_mask;
  int64_t mStrideB =
      {{kernel.size(attention_mask, 0)}} > 1
      ? {{kernel.stride(attention_mask, 0)}}
      : 0;
  int64_t mStrideH =
      {{kernel.size(attention_mask, 1)}} > 1
      ? {{kernel.stride(attention_mask, 1)}}
      : 0;
  int64_t mStrideM =
      {{kernel.size(attention_mask, 2)}}> 1
      ? {{kernel.stride(attention_mask, 2)}}
      : 0;
  int64_t mStrideN =
      {{kernel.size(attention_mask, 3)}} > 1
      ? {{kernel.stride(attention_mask, 3)}}
      : 0;
{%- endif %}

  // Data ptrs
  const scalar_t* q_data = query;
  const scalar_t* k_data = key;
  const scalar_t* v_data = value;
  scalar_t* out_data = output;

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
  {{template.codegen_allocate_buffer("total_buf_data", "scalar_t", "num_thread * total_size_uint8_per_thread")}}

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
{%- if q_zp == 0 %}
          fill_stub(k_sum_ptr, static_cast<int32_t>(0), kvSize);
{%- else %}
          int_sum_b_contiguous_kernel(k_data + i * kStrideB + j * kStrideH,
            k_sum_ptr,
            kvSize, headSize, kStrideN, {{q_zp}});
{%- endif %}
{%- if a_zp == 0 %}
          fill_stub(v_sum_ptr, static_cast<int32_t>(0), headSize);
{%- else %}
          int_sum_a_contiguous_kernel(v_data + i * vStrideB + j * vStrideH,
            v_sum_ptr,
            headSize, kvSize, vStrideN, {{a_zp}});
{%- endif %}

          // transpose and packing
          for (int64_t n = 0; n < kvSize; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            for (int64_t b = 0; b < kvBlockSize; b += block_64) {
              bool istail = kvBlockSize - b < block_64;
              int64_t trans_rows = istail ? kvBlockSize - b : block_64;
              at::native::utils::transpose<uint8_t>(
                  headSize,
                  trans_rows,
                  k_data + i * kStrideB + j * kStrideH + n * kStrideN + b * kStrideN,
                  kStrideN,
                  B_blocked_xform_u8,
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
              at::vec::pack_vnni4(
                      /* src */ B_blocked_xform_u8,
                      /* dst */ key_reorder_ptr + n * qk_gemm_K +
                          b * qk_gemm_K,
                      /* ld_src */ block_64,
                      /* K */ qk_gemm_K,
                      /* N */ block_64);
            }
            // split headSize to block_64, block_64, block_64 ...
            // [av_gemm_K, headSize] -> [av_gemm_K,  block_64 ...]
            for (int64_t b = 0; b < rndHeadSize; b += block_64) {
              at::vec::pack_vnni4(
                      /* src */ v_data + i * vStrideB + j * vStrideH + n * vStrideN + b,
                      /* dst */ value_reorder_ptr + n * rndHeadSize +
                          av_gemm_K * b,
                      /* ld_src */ vStrideN,
                      /* K */ av_gemm_K,
                      /* N */ block_64);
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
            int64_t num_keys = kvSize;
            copy_value_with_pad(
                q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                query_t_padding_ptr,
                qBlockSize,
                headSize,
                qBlockSize,
                qk_gemm_K,
                qStrideM);
            // sum q
{%- if k_zp != 0 %}
            int_sum_b_contiguous_kernel(q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                  q_sum_ptr, qBlockSize, headSize, qStrideM, {{k_zp}});
{%- else %}
            fill_stub(
              q_sum_ptr, static_cast<int32_t>(0), qSplitSize);
{%- endif %}
            const int64_t rkvSlice = (num_keys - 1) / kvSplitSize + 1;
            for (int64_t l = 0; l < rkvSlice; l++) {
              int64_t n = l * kvSplitSize;
              int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
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
{%- if has_attention_mask %}
              const mask_t* mask_data_offset = mask_data + i * mStrideB + j * mStrideH + m * mStrideM + (mStrideN == 0 ? 0 : n);
              dequant_mask_max_fusion_kernel(
                qk_s32_data, //in
                mask_data_offset, //mask_ptr
                q_sum_ptr, //sum_a_ptr
                k_sum_ptr + n, //sum_b_ptr
                qBlockSize, //M
                kvBlockSize, //N
                rndkvSplitSize, //ldi
                mStrideM, //ldm
                rndkvSplitSize, //ldo
                {{q_zp}} * {{k_zp}} * headSize, //zp_a*zp_b*k=beta
                {{q_scale}} * {{k_scale}} * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                qk_block_data, //out
                sfm_max_ptr //sfm_max_ptr
              );
{%- else %}
              dequant_max_fusion_kernel(
                qk_s32_data, //in
                q_sum_ptr, //sum_a_ptr
                k_sum_ptr + n, //sum_b_ptr
                qBlockSize, //M
                kvBlockSize, //N
                rndkvSplitSize, //ldi
                rndkvSplitSize, //ldo
                {{q_zp}} * {{k_zp}} * headSize, //zp_a*zp_b*k=beta
                {{q_scale}} * {{k_scale}} * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
                qk_block_data, //out
                sfm_max_ptr //sfm_max_ptr
              );
{%- endif %}
            }
            // sub max, exp, sum reduce, div sum for softmax
            // and quant
            // and sum for attention
{%- if v_zp == 0 %}
            sub_exp_sum_div_quant_fusion_kernel(
              qk_data, //in
              qBlockSize, //M
              kvSplitSize, //N_step
              rkvSlice, //NSlices
              qSplitSize * rndkvSplitSize, //ldi
              qk_reduce_strideL, //ldo
              kvSize, //kvSize
              rndkvSplitSize, //rndkvSplitSize
              av_gemm_K, //av_gemm_K
              {{a_zp}}, // zp_a=beta1
              {{a_scale}}, // scale_a=alpha
              qk_local_data, //local
              qk_reduced_data, //out
              sfm_max_ptr, //sfm_max_ptr
              sfm_sum_ptr //sfm_sum_ptr
            );
{%- else %}
            sub_exp_sum_div_quant_sum_fusion_kernel(
              qk_data, //in
              qBlockSize, //M
              kvSplitSize, //N_step
              rkvSlice, //NSlice
              qSplitSize * rndkvSplitSize, //ldi
              qk_reduce_strideL, //ldo
              kvSize, //kvSize
              rndkvSplitSize, //rndkvSplitSize
              av_gemm_K, //av_gemm_K
              {{a_zp}}, // zp_a=beta1
              {{v_zp}}, // zp_b=beta2
              {{a_scale}}, // scale_a=alpha
              qk_local_data, //local
              qk_reduced_data, //out
              sfm_max_ptr, //sfm_max_ptr
              sfm_sum_ptr, //sfm_sum_ptr
              a_sum_ptr //a_sum_ptr
            );
{%- endif %}
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
{%- if a_zp == 0 %}
          dequant_quant_fusion_kernel(
            dst_s32_data, //in
            a_sum_ptr, //sum_a_ptr
            qBlockSize, //M
            headSize, //N
            rndHeadSize, //ldi
            oStrideM, //ldo
            {{o_zp}}, //zp_c=beta2
            {{a_scale}} * {{v_scale}} / {{o_scale}}, //scale_a*scale_b/scale_c=alpha
            out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
          );
{%- else %}
          dequant_quant_fusion_kernel(
            dst_s32_data, //in
            a_sum_ptr, //sum_a_ptr
            v_sum_ptr, //sum_b_ptr
            qBlockSize, //M
            headSize, //N
            rndHeadSize, //ldi
            oStrideM, //ldo
            {{a_zp}} * {{v_zp}} * kvSize, //zp_a*zp_b*k=beta1
            {{o_zp}}, //zp_c=beta2
            {{a_scale}} * {{v_scale}} / {{o_scale}}, //scale_a*scale_b/scale_c=alpha
            out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
          );
{%- endif %}
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head);
        }
      });
  // Once all computations are done, need to release HW context.
  at::native::cpublas::brgemm_release();
}

"""


INT8_SDPA_SEVERAL_LOOPS_TEMPLATE = r"""
#ifndef HEADER_DEFINED
#define HEADER_DEFINED

{{template.header().getvalue()}}
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

#include <ATen/Tensor.h>
#include <limits>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

{{template.codegen_useful_function(kernel.kernel_name)}}

#endif

{%- if has_attention_mask %}
{%- set kernel_args = {"query": query, "key": key, "value": value,
                       "attention_mask": attention_mask} %}
{%- else %}
{%- set kernel_args = {"query": query, "key": key, "value": value} %}
{%- endif %}

// UINT8 - u8u8s32
extern "C"
{{kernel.def_kernel(inputs=kernel_args, outputs={"output": output})}}
{
  {{ kernel.maybe_codegen_profile() }}
  int64_t num_thread = {{num_thread}};
  using accum_t = float;
  using scalar_t = {{kernel.dtype(query)}};
  int block_64 = 64;
  auto u8_dt = at::ScalarType::Byte;

  // Sizes
  int64_t batchSize = {{kernel.size(query, 0)}};
  int64_t qSize = {{kernel.size(query, 1)}};
  int64_t kvSize = {{kernel.size(value, 1)}};
  int64_t num_head = {{kernel.size(query, 2)}};
  int64_t headSize = {{kernel.size(query, 3)}};
  float scaling_factor =
      calculate_scale(headSize, {{scale}});

  // Strides
  int64_t qStrideB = {{kernel.stride(query, 0)}};
  int64_t qStrideM = {{kernel.stride(query, 1)}};
  int64_t qStrideH = {{kernel.stride(query, 2)}};
  int64_t kStrideB = {{kernel.stride(key, 0)}};
  int64_t kStrideN = {{kernel.stride(key, 1)}};
  int64_t kStrideH = {{kernel.stride(key, 2)}};
  int64_t vStrideB = {{kernel.stride(value, 0)}};
  int64_t vStrideN = {{kernel.stride(value, 1)}};
  int64_t vStrideH = {{kernel.stride(value, 2)}};
  int64_t oStrideB = {{kernel.stride(output, 0)}};
  int64_t oStrideM = {{kernel.stride(output, 2)}};
  int64_t oStrideH = {{kernel.stride(output, 1)}};

  int64_t qSplitSize = {{q_split_size}} > qSize ? qSize : {{q_split_size}};
  int64_t kvSplitSize = {{kv_split_size}} > kvSize ? kvSize : {{kv_split_size}};
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;

  int64_t rndHeadSize = (headSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSplitSize = (kvSplitSize + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvTail = (kvTail + block_64 - 1L) / block_64 * block_64;
  int64_t rndkvSize = {{kv_split_size}} > kvSize ? rndkvTail : rndkvSplitSize * kvSlice + rndkvTail;

  bool av_gemm_K_mul4 = kvSplitSize % 4 == 0;
  int av_gemm_K_padding = av_gemm_K_mul4 ? 0 : 4 - kvSplitSize % 4;
  int av_gemm_K = kvSplitSize + av_gemm_K_padding;

{%- if has_attention_mask %}
  // attention mask
  using mask_t = {{kernel.dtype(attention_mask)}};
  const mask_t* mask_data = attention_mask;
  int64_t mStrideB =
      {{kernel.size(attention_mask, 0)}} > 1
      ? {{kernel.stride(attention_mask, 0)}}
      : 0;
  int64_t mStrideH =
      {{kernel.size(attention_mask, 1)}} > 1
      ? {{kernel.stride(attention_mask, 1)}}
      : 0;
  int64_t mStrideM =
      {{kernel.size(attention_mask, 2)}}> 1
      ? {{kernel.stride(attention_mask, 2)}}
      : 0;
  int64_t mStrideN =
      {{kernel.size(attention_mask, 3)}} > 1
      ? {{kernel.stride(attention_mask, 3)}}
      : 0;
{%- endif %}

  // Data ptrs
  const scalar_t* q_data = query;
  const scalar_t* k_data = key;
  const scalar_t* v_data = value;
  scalar_t* out_data = output;

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
  {{template.codegen_allocate_buffer("total_buf_data", "scalar_t", "num_thread * total_size_uint8_per_thread")}}

  int64_t kv_sum_size_per_BH =
    /* key_sum */ kvSize +
    /* value_sum */ headSize;
  {{template.codegen_allocate_buffer("kv_sum_buf_data", "int32_t", "batchSize * num_head * kv_sum_size_per_BH")}}

  int64_t kv_reorder_size_per_BH =
    /* key_t_reorder */ qk_gemm_K * rndkvSize +
    /* value_t_reorder */ kvSlice * v_reorder_strideL;
  {{template.codegen_allocate_buffer("kv_reorder_buf_data", "scalar_t", "batchSize * num_head * kv_reorder_size_per_BH")}}
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
{%- if q_zp == 0 %}
          fill_stub(k_sum_ptr, static_cast<int32_t>(0), kvSize);
{%- else %}
          int_sum_b_contiguous_kernel(k_data + i * kStrideB + j * kStrideH,
            k_sum_ptr,
            kvSize, headSize, kStrideN, {{q_zp}});
{%- endif %}
{%- if a_zp == 0 %}
          fill_stub(v_sum_ptr, static_cast<int32_t>(0), headSize);
{%- else %}
          int_sum_a_contiguous_kernel(v_data + i * vStrideB + j * vStrideH,
            v_sum_ptr,
            headSize, kvSize, vStrideN, {{a_zp}});
{%- endif %}
        // Move to the next query
        at::native::data_index_step(i, batchSize, j, num_head);
      }
    });

  // packing
  at::parallel_for(
    0, batchSize * num_head * kvSlice, 1, [&](int64_t begin, int64_t end) {
      int64_t i = 0, j = 0, l = 0, n = 0;
      at::native::data_index_init(
          begin, i, batchSize, j, num_head, l, kvSlice);
      uint8_t* B_blocked_xform_u8 = new uint8_t[qk_gemm_K * kvSplitSize];
      for (const auto z : c10::irange(begin, end)) {
        (void)z; // Suppress unused variable
        n = l * kvSplitSize;
        auto k_reorder = key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                      j * qk_gemm_K * rndkvSize + n * qk_gemm_K;
        auto v_reorder = value_reorder_ptr +
                      i * num_head * kvSlice * v_reorder_strideL +
                      j * kvSlice * v_reorder_strideL + n * rndHeadSize;
        int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
        at::native::utils::transpose<uint8_t>(
              kvBlockSize,
              headSize,
              k_data + i * kStrideB + j * kStrideH + n * kStrideN,
              kStrideN,
              B_blocked_xform_u8,
              kvBlockSize);
        at::vec::pack_vnni4(
              /* src */ B_blocked_xform_u8,
              /* dst */ k_reorder,
              /* ld_src */ kvBlockSize,
              /* K */ qk_gemm_K,
              /* N */ kvBlockSize);
        at::vec::pack_vnni4(
              /* src */ v_data + i * vStrideB + j * vStrideH + n * vStrideN,
              /* dst */ v_reorder,
              /* ld_src */ vStrideN,
              /* K */ av_gemm_K,
              /* N */ rndHeadSize);
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
        //offset += qSplitSize * 4;
        //scalar_t* query_t_padding_ptr = reinterpret_cast<scalar_t*>(total_buf_ptr + offset);

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
          int64_t num_keys = kvSize;
          // sum q
          const scalar_t* q_tmp = q_data + i * qStrideB + j * qStrideH + m * qStrideM;
{%- if k_zp != 0 %}
          int_sum_b_contiguous_kernel(q_tmp,
                q_sum_ptr, qBlockSize, headSize, qStrideM, {{k_zp}});
{%- else %}
          fill_stub(
            q_sum_ptr, static_cast<int32_t>(0), qSplitSize);
{%- endif %}
          const int64_t rkvSlice = (num_keys - 1) / kvSplitSize + 1;
          
          for (int64_t l = 0; l < rkvSlice; l++) {
            int64_t n = l * kvSplitSize;
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            auto k_reorder = key_reorder_ptr + i * num_head * qk_gemm_K * rndkvSize +
                      j * qk_gemm_K * rndkvSize + n * qk_gemm_K;
            // Calculate q @ k.T
            at::native::cpublas::brgemm(
                    qSplitSize, kvBlockSize, headSize,
                    qStrideM, // lda
                    kvBlockSize, //ldb
                    rndkvSplitSize, //ldc,
                    false,
                    q_tmp,
                    k_reorder,
                    qk_s32_data);

            // do dequant compensation, add mask, max reduce for softmax, and convert qk from s32 to fp32
            accum_t* qk_block_data = qk_data + l * qSplitSize * rndkvSplitSize;
{%- if has_attention_mask %}
            const mask_t* mask_data_offset = mask_data + i * mStrideB + j * mStrideH + m * mStrideM + (mStrideN == 0 ? 0 : n);
            dequant_mask_max_fusion_kernel(
              qk_s32_data, //in
              mask_data_offset, //mask_ptr
              q_sum_ptr, //sum_a_ptr
              k_sum_ptr + n, //sum_b_ptr
              qBlockSize, //M
              kvBlockSize, //N
              rndkvSplitSize, //ldi
              mStrideM, //ldm
              rndkvSplitSize, //ldo
              {{q_zp}} * {{k_zp}} * headSize, //zp_a*zp_b*k=beta
              {{q_scale}} * {{k_scale}} * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
              qk_block_data, //out
              sfm_max_ptr //sfm_max_ptr
            );
{%- else %}
            dequant_max_fusion_kernel(
              qk_s32_data, //in
              q_sum_ptr, //sum_a_ptr
              k_sum_ptr + n, //sum_b_ptr
              qBlockSize, //M
              kvBlockSize, //N
              rndkvSplitSize, //ldi
              rndkvSplitSize,//kvBlockSize, //ldo
              {{q_zp}} * {{k_zp}} * headSize, //zp_a*zp_b*k=beta
              {{q_scale}} * {{k_scale}} * scaling_factor, //scale_a*scale_b*scale_sdpa=alpha
              qk_block_data, //out
              sfm_max_ptr //sfm_max_ptr
            );
{%- endif %}
          }
          // sub max, exp, sum reduce, div sum for softmax
          // and quant
          // and sum for attention
{%- if v_zp == 0 %}
          sub_exp_sum_div_quant_fusion_kernel(
            qk_data, //in
            qBlockSize, //M
            kvSplitSize, //N_step
            rkvSlice, //NSlices
            qSplitSize * rndkvSplitSize, //ldi
            qk_reduce_strideL, //ldo
            kvSize, //kvSize
            rndkvSplitSize, //rndkvSplitSize
            av_gemm_K, //av_gemm_K
            {{a_zp}}, // zp_a=beta1
            {{a_scale}}, // scale_a=alpha
            qk_local_data, //local
            qk_reduced_data, //out
            sfm_max_ptr, //sfm_max_ptr
            sfm_sum_ptr //sfm_sum_ptr
          );
{%- else %}
          sub_exp_sum_div_quant_sum_fusion_kernel(
            qk_data, //in
            qBlockSize, //M
            kvSplitSize, //N_step
            rkvSlice, //NSlice
            qSplitSize * rndkvSplitSize, //ldi
            qk_reduce_strideL, //ldo
            kvSize, //kvSize
            rndkvSplitSize, //rndkvSplitSize
            av_gemm_K, //av_gemm_K
            {{a_zp}}, // zp_a=beta1
            {{v_zp}}, // zp_b=beta2
            {{a_scale}}, // scale_a=alpha
            qk_local_data, //local
            qk_reduced_data, //out
            sfm_max_ptr, //sfm_max_ptr
            sfm_sum_ptr, //sfm_sum_ptr
            a_sum_ptr //a_sum_ptr
          );
{%- endif %}
          // Calculate Softmax(q @ k.T) @ v
          auto v_reorder = value_reorder_ptr +
                  i * num_head * kvSlice * v_reorder_strideL +
                  j * kvSlice * v_reorder_strideL;
          for (int64_t s = 0; s < kvSlice; s++) {
            at::native::cpublas::brgemm(
                qSplitSize, headSize, av_gemm_K,
                av_gemm_K, // lda
                rndHeadSize, //ldb
                rndHeadSize, //ldc
                s != 0,
                qk_reduced_data + s * qk_reduce_strideL,
                v_reorder + s * v_reorder_strideL,
                dst_s32_data);
          }

          // After the last gemm,
          // do dequant compensation, quant and convert from s32 to int8
{%- if a_zp == 0 %}
          dequant_quant_fusion_kernel(
            dst_s32_data, //in
            a_sum_ptr, //sum_a_ptr
            qBlockSize, //M
            headSize, //N
            rndHeadSize, //ldi
            oStrideM, //ldo
            {{o_zp}}, //zp_c=beta2
            {{a_scale}} * {{v_scale}} / {{o_scale}}, //scale_a*scale_b/scale_c=alpha
            out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
          );
{%- else %}
          dequant_quant_fusion_kernel(
            dst_s32_data, //in
            a_sum_ptr, //sum_a_ptr
            v_sum_ptr, //sum_b_ptr
            qBlockSize, //M
            headSize, //N
            rndHeadSize, //ldi
            oStrideM, //ldo
            {{a_zp}} * {{v_zp}} * kvSize, //zp_a*zp_b*k=beta1
            {{o_zp}}, //zp_c=beta2
            {{a_scale}} * {{v_scale}} / {{o_scale}}, //scale_a*scale_b/scale_c=alpha
            out_data + i * oStrideB + j * oStrideH + m * oStrideM //out
          );
{%- endif %}
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
        }
      });
  // Once all computations are done, need to release HW context.
  at::native::cpublas::brgemm_release();
}
"""


class CppInt8SdpaTemplate(CppFlexAttentionTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        scale,
        q_scale,
        q_zp,
        k_scale,
        k_zp,
        v_scale,
        v_zp,
        a_scale,
        a_zp,
        o_scale,
        o_zp,
    ) -> None:
        assert layout.dtype in [torch.uint8]
        CppTemplate.__init__(
            self, "int8_sdpa", input_nodes, layout, parallel_num_threads()
        )
        self.scale = scale
        self.q_scale = q_scale
        self.q_zp = q_zp
        self.k_scale = k_scale
        self.k_zp = k_zp
        self.v_scale = v_scale
        self.v_zp = v_zp
        self.a_scale = a_scale
        self.a_zp = a_zp
        self.o_scale = o_scale
        self.o_zp = o_zp

    @staticmethod
    def add_choices(
        choices,
        input_nodes,
        layout,
        scale,
        q_scale,
        q_zp,
        k_scale,
        k_zp,
        v_scale,
        v_zp,
        a_scale,
        a_zp,
        o_scale,
        o_zp,
    ):
        def preprocessor(input_nodes, layout):
            return input_nodes, layout

        def postprocessor(output):
            return output

        template = DataProcessorTemplateWrapper(
            CppInt8SdpaTemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            scale=scale,
            q_scale=q_scale,
            q_zp=q_zp,
            k_scale=k_scale,
            k_zp=k_zp,
            v_scale=v_scale,
            v_zp=v_zp,
            a_scale=a_scale,
            a_zp=a_zp,
            o_scale=o_scale,
            o_zp=o_zp,
        )
        template.maybe_append_choice(choices)
        return template

    def reshape_attn_mask_to_4d(
        self,
        kernel,
        attn_mask: ir.Buffer,
        batchSize,
        num_head,
        qSize,
        kvSize,
    ):
        # Support mask shapes:
        # 2d: ({Q_seq_len, 1}  x {KV_seq_len, 1})
        # 4d: ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})
        # Guaranteed in check_attn_mask_shape
        attn_mask_size_0 = 1
        attn_mask_size_1 = 1
        layout = attn_mask.get_layout()
        if len(layout.size) == 4:
            if layout.size[0] == batchSize:
                attn_mask_size_0 = batchSize
            if layout.size[1] == num_head:
                attn_mask_size_1 = num_head
        attn_mask = kernel.view(
            attn_mask,
            [
                attn_mask_size_0,
                attn_mask_size_1,
                layout.size[-2],
                layout.size[-1],
            ],
        )
        attn_mask = expand(
            attn_mask, [attn_mask_size_0, attn_mask_size_1, qSize, kvSize]
        )
        return attn_mask

    def get_options(
        self,
        query: ir.Buffer,
        key: ir.Buffer,
        value: ir.Buffer,
        qSize,
        kvSize,
        headSize,
        batchSize,
        num_head,
        num_threads,
    ):
        q_split_size = 32
        if qSize >= 768:
            q_split_size = 256
        elif qSize >= 192:
            q_split_size = 128
        kv_split_size = 512

        qSplitSize = min(qSize, q_split_size)
        l2_cache_size = torch._C._cpu._L2_cache_size()
        attn_size = qSplitSize * kvSize * 4 * num_threads
        use_one_parallel_loop = True
        if all(
            sympify(val).is_number
            for val in [batchSize, num_head, num_threads, attn_size, l2_cache_size]
        ):
            # if not symbolic shape
            use_one_parallel_loop = (batchSize * num_head > num_threads) and (
                attn_size > 1.5 * l2_cache_size
            )

        options = dict(
            q_split_size=q_split_size,
            kv_split_size=kv_split_size,
            use_one_parallel_loop=use_one_parallel_loop,
        )
        return options

    def render(  # type: ignore[override,return]
        self,
        kernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        if epilogue_nodes is not None and epilogue_nodes != []:
            raise NotImplementedError(
                "Unsupported for `epilogue_nodes` in CppInt8SdpaTemplate."
            )
        # Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
        #     -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
        #  Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
        #  Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
        #     -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)

        query = kernel.permute(self.input_nodes[0], [0, 2, 1, 3])
        key = kernel.permute(self.input_nodes[1], [0, 2, 1, 3])
        value = kernel.permute(self.input_nodes[2], [0, 2, 1, 3])

        batchSize = query.layout.size[0]
        qSize = query.layout.size[1]
        kvSize = value.layout.size[1]
        num_head = query.layout.size[2]
        headSize = query.layout.size[3]

        has_attention_mask = len(self.input_nodes) == 4
        attention_mask = (
            self.reshape_attn_mask_to_4d(
                kernel, self.input_nodes[3], batchSize, num_head, qSize, kvSize
            )
            if has_attention_mask
            else None
        )

        num_threads = parallel_num_threads()
        buf_out = TensorBox.create(self.output_node)
        if template_buffer_node is not None:
            buf_out = template_buffer_node
        options = dict(
            query=query,
            key=key,
            value=value,
            has_attention_mask=has_attention_mask,
            attention_mask=attention_mask,
            scale=self.scale,
            q_scale=self.q_scale,
            q_zp=self.q_zp,
            k_scale=self.k_scale,
            k_zp=self.k_zp,
            v_scale=self.v_scale,
            v_zp=self.v_zp,
            a_scale=self.a_scale,
            a_zp=self.a_zp,
            o_scale=self.o_scale,
            o_zp=self.o_zp,
            template=self,
            output=buf_out,
            kernel=kernel,
            num_thread=num_threads,
        )
        new_options = self.get_options(
            query=query,
            key=key,
            value=value,
            qSize=qSize,
            kvSize=kvSize,
            headSize=headSize,
            batchSize=batchSize,
            num_head=num_head,
            num_threads=num_threads,
        )
        options.update(new_options)
        INT8_SDPA_TEMPLATE = (
            INT8_SDPA_ONE_LOOP_TEMPLATE
            if options["use_one_parallel_loop"]
            else INT8_SDPA_SEVERAL_LOOPS_TEMPLATE
        )
        return self._template_from_string(INT8_SDPA_TEMPLATE).render(**options)

    def codegen_useful_function(self, kernel_name: str):
        return self._template_from_string(USEFUL_FUNCTIONS).render(
            dict(kernel_name=kernel_name)
        )

    def codegen_allocate_buffer(self, buffer_name: str, buffer_dtype, buffer_size):
        return self._template_from_string(ALLOCATE_BUFFER).render(
            dict(
                buffer_name=buffer_name,
                buffer_dtype=buffer_dtype,
                buffer_size=buffer_size,
            )
        )
