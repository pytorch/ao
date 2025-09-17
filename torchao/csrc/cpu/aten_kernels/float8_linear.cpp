#include <torch/all.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <c10/util/Unroll.h>
#include "utils.h"
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace torchao {

namespace {

#define BLOCK_N 32

#define PER_TENSOR 1
#define PER_ROW 2
#define PER_GROUP 3

static bool cpublas_checked = false;
static bool cpublas_can_pack = false;

bool cpublas_could_pack() {
  // the could_pack check requires AMX support implicitly
  if (cpublas_checked) {
    return cpublas_can_pack;
  }
#ifdef CPUBLAS_BRGEMM_F8F8F32
  cpublas_can_pack = at::native::cpublas::could_pack(at::kFloat8_e4m3fn);
#else
  cpublas_can_pack = at::native::cpublas::could_pack(at::kBFloat16);
#endif
  cpublas_checked = true;
  return cpublas_can_pack;
}

/*
return: packed_weight, packed_scales
*/
std::tuple<at::Tensor, at::Tensor>
float8_linear_prepack_impl(
    const at::Tensor& weight,
    const at::Tensor& scales) {
  // weight shape = [N, K]
  // scales shape = [N, G]
  TORCH_CHECK(weight.dim() == 2,
              "Float8 linear CPU: Weight should be a 2D tensor for packing");
  int N = weight.size(0);
  int K = weight.size(1);
  constexpr int block_n = BLOCK_N;
  // Case to fall back
  if (N % block_n != 0 || K % 32 != 0) {
    return std::make_tuple(weight, scales);
  }

  auto new_scales = scales;
  bool is_per_tensor = new_scales.numel() == 1;
  if (new_scales.dim() == 1 && !is_per_tensor) {
    new_scales.unsqueeze_(1);
  }
  new_scales = new_scales.to(at::kFloat);
  int G = is_per_tensor ? 1 : new_scales.size(1);
  TORCH_CHECK(K % G == 0, "K should be divisible by num_groups");
  int group_size = K / G;
  int block_k = group_size > 128 ? 128 : group_size;
  while (K % block_k != 0) {
    block_k /= 2;
  }
  TORCH_CHECK(block_k > 0 && block_k <= group_size,
              "Float8 linear CPU: Invalid block_k size, should be in (0, group_size]");
  int Nc = N / block_n;
  int Kc = K / block_k;

  // Reorder weight to [N/block_n, K/block_k, block_k, block_n]
  // Reorder scales to [N/block_n, G, block_n]
  auto weight_view = weight.view({Nc, block_n, Kc, block_k});
  at::Tensor weight_reordered = weight_view.permute({0, 2, 3, 1}).contiguous();
  at::Tensor blocked_weight;
  at::Tensor blocked_scales = is_per_tensor ? new_scales.view({1}) : new_scales.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();

#if defined(CPU_CAPABILITY_AVX512)
  if (cpublas_could_pack()) {
#ifdef CPUBLAS_BRGEMM_F8F8F32
    constexpr int vnni_size = get_vnni_size<at::Float8_e4m3fn>(); // for fp8
#else
    constexpr int vnni_size = get_vnni_size<at::BFloat16>(); // for bfloat16
#endif
    blocked_weight = at::empty({Nc, Kc, block_k, block_n}, weight.options());
    auto weight_ptr = reinterpret_cast<uint8_t*>(weight_reordered.data_ptr());
    auto blocked_weight_ptr = reinterpret_cast<uint8_t*>(blocked_weight.data_ptr());
    int64_t num_blocks = Nc * Kc;
    at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        auto in_ptr = weight_ptr + i * block_k * block_n;
        auto out_ptr = blocked_weight_ptr + i * block_k * block_n;

        // Reorder weight block to VNNI
        // plain shape = [block_k, block_n]
        // packed shape = [block_k / VNNI_SIZE, block_n, VNNI_SIZE] viewed as [block_k, block_n]
        constexpr int n_group_size = 8;
        constexpr int n_group = block_n / n_group_size; // 4
        for (int nb = 0; nb < n_group; ++nb) {
          for (int k = 0; k < block_k; k += vnni_size) {
            for (int ni = 0; ni < n_group_size; ++ni) {
              for (int ki = 0; ki < vnni_size; ++ki) {
                int src_idx = nb * n_group_size + ni + (k + ki) * block_n;
                int dst_idx = (nb * n_group_size + ni) * vnni_size + k * block_n + ki;
                *(out_ptr + dst_idx) = *(in_ptr + src_idx);
              }
            }
          }
        }
      }
    });
  } else
#endif
  {
    blocked_weight = weight_reordered;
  }

  return std::make_tuple(std::move(blocked_weight), std::move(blocked_scales));
}

#if defined(CPU_CAPABILITY_AVX512)
// this doesn't handle NaN.
inline __m512bh cvt_e4m3_bf16_intrinsic_no_nan(__m256i fp8_vec) {
  const __m512i x = _mm512_cvtepu8_epi16(fp8_vec);

  const __m512i mant = _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(0x07)), 4);
  const __m512i raw_exp = _mm512_srli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(0x78)), 3);
  const __m512i exp = _mm512_slli_epi16(_mm512_add_epi16(raw_exp, _mm512_set1_epi16(120)), 7);
  const __m512i nonsign = _mm512_or_si512(exp, mant);

  const __m512i sign = _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(0x80)), 8);
  const __m512i combined = _mm512_or_si512(nonsign, sign);

  const __mmask32 is_nonzero = _mm512_cmpneq_epi16_mask(x, _mm512_setzero_si512());
  return (__m512bh)_mm512_maskz_mov_epi16(is_nonzero, combined);
}

static void cvt_f8e4m3_to_bf16(
    const at::Float8_e4m3fn* __restrict__ in,
    at::BFloat16* out,
    int64_t rows,
    int64_t cols,
    int64_t stride) {
  if (stride == cols) {
    // A contiguous buffer
    size_t len = rows * cols;
    size_t i = 0;
    for (; i < len; i += 32) {
      __m256i fp8_vec = _mm256_loadu_si256((__m256i*)&in[i]);
      __m512bh bf16_vec = cvt_e4m3_bf16_intrinsic_no_nan(fp8_vec);
      _mm512_storeu_si512((__m512i*)(out + i), (__m512i)bf16_vec);
    }
    for (; i < len; ++i) {
      out[i] = (at::BFloat16)in[i];
    }
  } else {
    // Non-contiguous. Access each row with stride
    TORCH_CHECK(stride > cols);
    for (int r = 0; r < rows; ++r) {
      size_t i = 0;
      size_t vec_len = cols / 32 * 32;
      for (; i < vec_len; i += 32) {
        __m256i fp8_vec = _mm256_loadu_si256((__m256i*)&in[r * stride + i]);
        __m512bh bf16_vec = cvt_e4m3_bf16_intrinsic_no_nan(fp8_vec);
        _mm512_storeu_si512((__m512i*)(out + r * cols + i), (__m512i)bf16_vec);
      }
      for (; i < cols; ++i) {
        out[r * cols + i] = (at::BFloat16)in[r * stride + i];
      }
    }
  }
}


// accumulate and store result to buffer
// if act/wei are per_group quantized, apply scales
template <bool accum, int64_t N, int act_quant_mode, int wei_quant_mode>
static void _accumulate_result(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int M,
    int ldi,
    int ldo,
    int ldsa = 1) {
  float a_scale, b_scale;
  __m512 va_scale;
  __m512 vb_scale;
  for (int m = 0; m < M; ++m) {
    if constexpr (act_quant_mode == PER_GROUP) {
      a_scale = *(scale_a + m * ldsa);
      va_scale = _mm512_set1_ps(a_scale);
    }
    constexpr int N_UNROLL = N / 16;
    c10::ForcedUnroll<N_UNROLL>{}([&](auto i) {
      constexpr int n = i * 16;
      __m512 vc_f = _mm512_loadu_ps(input + m * ldi + n);
      if constexpr (act_quant_mode == PER_GROUP) {
        vc_f = _mm512_mul_ps(vc_f, va_scale);
      }
      if constexpr (wei_quant_mode == PER_GROUP) {
        vb_scale = _mm512_loadu_ps(scale_b + n);
        vc_f = _mm512_mul_ps(vc_f, vb_scale);
      }
      if constexpr (accum) {
        __m512 vo = _mm512_loadu_ps(output + m * ldo + n);
        _mm512_storeu_ps(output + m * ldo + n, _mm512_add_ps(vo, vc_f));
      } else {
        _mm512_storeu_ps(output + m * ldo + n, vc_f);
      }
    });
    constexpr int tail_start = N / 16 * 16;
    for (int n = tail_start; n < N; ++n) {
      float dq_val = input[m * ldi + n];
      if constexpr (act_quant_mode == PER_GROUP) {
        dq_val = dq_val * a_scale;
      }
      if constexpr (wei_quant_mode == PER_GROUP) {
        b_scale = scale_b[n];
        dq_val = dq_val * b_scale;
      }
      if constexpr (accum) {
        output[m * ldo + n] += dq_val;
      } else {
        output[m * ldo + n] = dq_val;
      }
    }
  }
}

// Store result to output buffer with dtype conversion
// If act/wei are per_row or per_tensor quantized, apply scales
// If bias is not null, add bias
template<typename out_dtype, int64_t N, int act_quant_mode, int wei_quant_mode>
inline void store_out(
    const float* y_buf,
    out_dtype* c_ptr,
    int64_t M,
    int64_t lda,
    const float* scales_a,
    const float* scales_b,
    const float* bias) {
  float a_scale = 1.0, b_scale = 1.0;
    __m512 va_scale, vb_scale;
  if constexpr (act_quant_mode == PER_TENSOR) {
    a_scale = *scales_a;
  }
  if constexpr (wei_quant_mode == PER_TENSOR) {
    b_scale = *scales_b;
    vb_scale = _mm512_set1_ps(b_scale);
  }
  for (int i = 0; i < M; ++i) {
    if constexpr (act_quant_mode == PER_ROW) {
      a_scale = *(scales_a + i);
    }
    if constexpr (act_quant_mode != PER_GROUP) {
      va_scale = _mm512_set1_ps(a_scale);
    }
    constexpr int N_UNROLL = N / 16;
    c10::ForcedUnroll<N_UNROLL>{}([&](auto idx) {
      constexpr int j = idx * 16;
      __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
      __m512 bias_vec = bias ? _mm512_loadu_ps(bias + j) : _mm512_setzero_ps();
      if constexpr (act_quant_mode != PER_GROUP) {
        y_vec = _mm512_mul_ps(y_vec, va_scale);
      }
      if constexpr (wei_quant_mode == PER_ROW) {
        vb_scale = _mm512_loadu_ps(scales_b + j);
      }
      if constexpr (wei_quant_mode != PER_GROUP) {
        y_vec = _mm512_mul_ps(y_vec, vb_scale);
      }
      y_vec = _mm512_add_ps(y_vec, bias_vec);
      if constexpr (std::is_same<out_dtype, float>::value) {
        _mm512_storeu_ps(c_ptr + i * lda + j, y_vec);
      } else if constexpr (std::is_same<out_dtype, at::BFloat16>::value) {
        __m256i y_bf16_vec = at::vec::cvtfp32_bf16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_bf16_vec);
      } else if constexpr (std::is_same<out_dtype, at::Half>::value) {
        __m256i y_fp16_vec = at::vec::cvtfp32_fp16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_fp16_vec);
      } else {
        TORCH_CHECK(false, "Unsupported output dtype");
      }
    });
    constexpr int tail_start = N / 16 * 16;
    for (int j = tail_start; j < N; ++j) {
      if constexpr (wei_quant_mode == PER_ROW) {
        b_scale = scales_b[j];
      }
      c_ptr[i * lda + j] = static_cast<out_dtype>(y_buf[i * N + j] * a_scale * b_scale);
    }
  } // for M
}

#else // no AVX512

static void cvt_f8e4m3_to_bf16(
    const at::Float8_e4m3fn* __restrict__ in,
    at::BFloat16* out,
    int64_t rows,
    int64_t cols,
    int64_t stride) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[r * cols + c] = (at::BFloat16)in[r * stride + c];
    }
  }
}

// Store result to output buffer with dtype conversion
// If act/wei are per_row or per_tensor quantized, apply scales
// If bias is not null, add bias
template<typename out_dtype, int64_t N, int act_quant_mode, int wei_quant_mode>
inline void store_out(
    const float* y_buf,
    out_dtype* c_ptr,
    int64_t M,
    int64_t lda,
    const float* scales_a,
    const float* scales_b,
    const float* bias) {
  float a_scale = 1.0, b_scale = 1.0;
  if constexpr (act_quant_mode == PER_TENSOR) {
    a_scale = *scales_a;
  }
  if constexpr (wei_quant_mode == PER_TENSOR) {
    b_scale = *scales_b;
  }
  for (int i = 0; i < M; ++i) {
    if constexpr (act_quant_mode == PER_ROW) {
      a_scale = *(scales_a + i);
    }
    for (int j = 0; j < N; ++j) {
      if constexpr (wei_quant_mode == PER_ROW) {
        b_scale = scales_b[j];
      }
      c_ptr[i * lda + j] = static_cast<out_dtype>(y_buf[i * N + j] * a_scale * b_scale);
    }
  } // for M
}

#endif // CPU_CAPABILITY_AVX512

template <bool cpublas_can_pack, int64_t N, int act_quant_mode, int wei_quant_mode>
void _micro_gemm(
    float* C,
    const at::Float8_e4m3fn* A,
    const float* scales_a,
    const at::Float8_e4m3fn* B,
    const float* scales_b,
    int64_t M,
    int64_t K,
    int64_t lda,
    int64_t ldc,
    int64_t ldsa,
    float* ukernel_buf,
    at::BFloat16* dqA_buf,
    at::BFloat16* dqB_buf) {
  // If FP8 brgemm is not available, convert A/B to bf16 for computation
  // Compute GEMM fp8 * fp8 -> fp32 (or bf16 * bf16 -> fp32)
  // If per_group quant, apply scales. Otherwise, don't apply scales here
  // Finally accumulate and store results
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (cpublas_can_pack) {
#ifdef CPUBLAS_BRGEMM_F8F8F32
    at::native::cpublas::brgemm(
        M,
        N,
        K,
        lda /*lda*/,
        N /*ldb*/,
        N /*ldc*/,
        false /* add_C */,
        A,
        B,
        ukernel_buf,
        true /* is_vnni */);
#else
    cvt_f8e4m3_to_bf16(A, dqA_buf, M, K, lda);
    cvt_f8e4m3_to_bf16(B, dqB_buf, K, N, N);
    at::native::cpublas::brgemm(
        M,
        N,
        K,
        K /*lda*/,
        N /*ldb*/,
        N /*ldc*/,
        false /* add_C */,
        dqA_buf,
        dqB_buf,
        ukernel_buf,
        true /* is_vnni */);
#endif
    _mm_prefetch(B + N * (K + 128), _MM_HINT_T0);
    _mm_prefetch(A + K + 128, _MM_HINT_T0);
    _accumulate_result<true, N, act_quant_mode, wei_quant_mode>(
        C,
        ukernel_buf,
        scales_a,
        scales_b,
        M,
        N /*ldi*/,
        ldc,
        ldsa);
  } else
#endif
  {
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        float sum = 0;
        for (int64_t k = 0; k < K; ++k) {
          sum += ((float)A[i * lda + k] * (float)B[k * N + j]);
        }
        if constexpr (act_quant_mode == PER_GROUP) {
          sum *= scales_a[i * ldsa];
        }
        if constexpr (wei_quant_mode == PER_GROUP) {
          sum *= scales_b[j];
        }
        C[i * ldc + j] += sum;
      }
    }
  }
}

template<typename out_dtype, bool cpublas_can_pack, int act_quant_mode, int wei_quant_mode>
void _float8_linear_impl(
    const at::Tensor& input,
    const at::Tensor& input_scales,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    at::Tensor& output) {
  // input shape = [..., K]
  // input is per token quantized
  int64_t K = input.size(-1);
  auto input_view = input.view({-1, K});
  int64_t M = input_view.size(0);

  // weight shape = [Nc, Kc, block_k, block_n]
  // scales shape = [Nc, G, block_n]
  int64_t Nc = weight.size(0);
  int64_t Kc = weight.size(1);
  int64_t block_k = weight.size(2);
  constexpr int64_t block_n = BLOCK_N;
  TORCH_CHECK(weight.size(3) == block_n, "Float8 linear: unexpected weight shape");
  int64_t N = Nc * block_n;
  TORCH_CHECK(K == Kc * block_k, "Float8 linear: weight and input shapes mismatch");
  auto [parallel_on_M, block_m, Mc, Mc_parallel] = get_m_blocking(M);
  int64_t num_parallel_blocks = Mc_parallel * Nc;

  // scales shape = [Nc, G, block_n]
  int64_t num_groups = wei_quant_mode == PER_TENSOR ? 1 : weight_scales.size(1);
  TORCH_CHECK(K % num_groups == 0, "K should be divisible by num_groups");
  int64_t group_size = K / num_groups;
  TORCH_CHECK(group_size % block_k == 0,
              "Float8 linear: group_size should be divisible by block_k");
  int64_t block_per_group = group_size / block_k;
  TORCH_CHECK(input_scales.numel() == 1 || input_scales.numel() == M || input_scales.numel() == M * num_groups, "Float8 linear: unexpected input scales shape");
  auto ldsa = act_quant_mode == PER_TENSOR ? 0 : act_quant_mode == PER_ROW ? 1 : num_groups;

  const at::Float8_e4m3fn* a_ptr = input_view.data_ptr<at::Float8_e4m3fn>();
  const float* a_scales_ptr = input_scales.data_ptr<float>();
  const at::Float8_e4m3fn* b_ptr = weight.data_ptr<at::Float8_e4m3fn>();
  const float* b_scales_ptr = weight_scales.data_ptr<float>();
  out_dtype* c_ptr = output.data_ptr<out_dtype>();
  const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

  int64_t block_size = block_m * block_n;
  int64_t num_thread = at::get_num_threads();
  at::Tensor y_buffer = at::empty({num_thread, block_size}, output.options().dtype(at::kFloat));
  // Create buffer for brgemm output and dqA/dqB (optional)
#if defined(CPU_CAPABILITY_AVX512)
  // buffer for brgemm output in float32
  int64_t buffer_size = block_size * 2; // float32 = bfloat16 * 2
#ifndef CPUBLAS_BRGEMM_F8F8F32
  // buffers for dqA & dqB in bf16
  buffer_size += (block_k * block_n + block_m * block_k);
#endif
  at::Tensor micro_gemm_buffer = at::empty({num_thread, buffer_size}, output.options().dtype(at::kBFloat16));
#endif

  at::parallel_for(0, num_parallel_blocks, 1, [&](int64_t begin, int64_t end) {
    // Get the address of pre-allocated buffers
    float* y_buf = y_buffer.data_ptr<float>() + at::get_thread_num() * block_size;
    at::BFloat16 *dqA_buffer = nullptr, *dqB_buffer = nullptr;
    float* ukernel_buf = nullptr;
#if defined(CPU_CAPABILITY_AVX512)
    at::BFloat16* micro_gemm_buf = micro_gemm_buffer.data_ptr<at::BFloat16>() + at::get_thread_num() * buffer_size;
    ukernel_buf = reinterpret_cast<float*>(micro_gemm_buf);
#ifndef CPUBLAS_BRGEMM_F8F8F32
    dqA_buffer = micro_gemm_buf;
    dqB_buffer = micro_gemm_buf + block_m * block_k;
    ukernel_buf = reinterpret_cast<float*>(micro_gemm_buf + block_m * block_k + block_k * block_n);
#endif
#endif
    int64_t mc = 0, nc = 0;
    at::native::data_index_init(begin, mc, Mc_parallel, nc, Nc);
    for (const auto i : c10::irange(begin, end)) {
      (void)i; // Suppress unused variable
      int64_t mc_end = parallel_on_M ? mc + 1 : Mc;

      for (int mci = mc; mci < mc_end; ++mci) {
        int64_t m_size = mci * block_m + block_m > M ? M - mci * block_m : block_m;
        zero_buffer(y_buf, m_size * block_n);
        for (int kci = 0; kci < Kc; ++kci) {
          auto scales_a = a_scales_ptr + mci * block_m * num_groups + kci / block_per_group;
          auto scales_b = b_scales_ptr + nc * block_n * num_groups + kci / block_per_group * block_n;
          _micro_gemm<cpublas_can_pack, block_n, act_quant_mode, wei_quant_mode>(
            /* C */           y_buf,
            /* A */           a_ptr + mci * block_m * K + kci * block_k,
            /* scales_a */    scales_a,
            /* B */           b_ptr + (nc * Kc + kci) * block_n * block_k,
            /* scales_b */    scales_b,
            /* M */           m_size,
            /* K */           block_k,
            /* lda */         K,
            /* ldc */         block_n,
            /* ldsa */        ldsa,
            /* ukernel_buf */ ukernel_buf,
            /* dqA_buf */     dqA_buffer,
            /* dqB_buf */     dqB_buffer);
        }
        // store y_buf to output with dtype conversion
        auto scales_a = act_quant_mode == PER_TENSOR ? a_scales_ptr :
            act_quant_mode == PER_ROW ? a_scales_ptr + mci * block_m : nullptr;
        auto scales_b = wei_quant_mode == PER_TENSOR ? b_scales_ptr :
          wei_quant_mode == PER_ROW ? b_scales_ptr + nc * block_n : nullptr;
        auto bias_data = bias_ptr ? bias_ptr + nc * block_n : nullptr;
        store_out<out_dtype, block_n, act_quant_mode, wei_quant_mode>(
          y_buf,
          c_ptr + mci * block_m * N + nc * block_n,
          m_size,
          N /*lda*/,
          scales_a,
          scales_b,
          bias_data);
      }
      at::native::data_index_step(mc, Mc_parallel, nc, Nc);
    }
    if constexpr (cpublas_can_pack) {
      at::native::cpublas::brgemm_release();
    }
  });
}

at::Tensor float8_linear_impl(
    const at::Tensor& input,
    const at::Tensor& input_scales,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    at::ScalarType output_dtype) {
  int64_t N = weight.dim() == 4 ? weight.size(0) * weight.size(-1) : weight.size(0);
  int act_quant_mode = input_scales.numel() == 1 ? PER_TENSOR :
                       input_scales.numel() == input.numel() / input.size(-1) ? PER_ROW :
                       PER_GROUP;
  int wei_quant_mode = weight_scales.numel() == 1 ? PER_TENSOR :
                       weight_scales.numel() == N ? PER_ROW :
                       PER_GROUP;
  // Case to fall back
  if (weight.dim() == 2) {
    TORCH_CHECK(act_quant_mode != PER_GROUP && wei_quant_mode != PER_GROUP,
      "FP8 linear: Per-group quantization is not supported in the fallback path");
    auto y_fp32 = at::linear(
      input.to(at::kFloat).mul_(input_scales),
      weight.to(at::kFloat).mul_(weight_scales),
      bias);
    return y_fp32.to(output_dtype);
  }

  static bool cpublas_can_pack = cpublas_could_pack();
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  auto output = at::empty(out_sizes, input.options().dtype(output_dtype));

  AT_DISPATCH_LINEAR_KERNEL(output_dtype, cpublas_can_pack, act_quant_mode, wei_quant_mode, [&](){
    _float8_linear_impl<out_t, can_pack, a_quant_mode, b_quant_mode>(
      input,
      input_scales,
      weight,
      weight_scales,
      bias,
      output);
  });
  return output;
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::float8_linear_prepack_cpu", &float8_linear_prepack_impl);
  m.impl("torchao::float8_linear_cpu", &float8_linear_impl);
}

} // namespace torchao
