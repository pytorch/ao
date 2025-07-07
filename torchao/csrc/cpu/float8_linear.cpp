#include <torch/all.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <c10/util/Unroll.h>

namespace torchao {

namespace {

#define BLOCK_N 32

static bool cpublas_checked = false;
static bool cpublas_can_pack = false;

bool cpublas_could_pack() {
  // the could_pack check requires AMX support implicitly
  if (cpublas_checked) {
    return cpublas_can_pack;
  }
  cpublas_can_pack = at::native::cpublas::could_pack(at::kBFloat16);
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
  TORCH_CHECK(weight.size(1) % 2 == 0,
              "Float8 linear CPU: Weight should have even number of columns for packing");

  auto new_scales = scales;
  if (new_scales.dim() == 1) {
    new_scales.unsqueeze_(1);
  }
  new_scales = new_scales.to(at::kFloat);
  int N = weight.size(0);
  int K = weight.size(1);
  int G = scales.size(1);
  int group_size = K / G;
  int block_k = group_size > 128 ? 128 : group_size;
  while (K % block_k != 0) {
    block_k /= 2;
  }
  TORCH_CHECK(block_k > 0 && block_k <= group_size,
              "Float8 linear CPU: Invalid block_k size, should be in (0, group_size]");
  constexpr int block_n = BLOCK_N;
  int Nc = N / block_n;
  int Kc = K / block_k;

  // Reorder weight to [N/block_n, K/block_k, block_k, block_n]
  // Reorder scales to [N/block_n, G, block_n]
  auto weight_view = weight.view({Nc, block_n, Kc, block_k});
  at::Tensor weight_reordered = weight_view.permute({0, 2, 3, 1}).contiguous();
  at::Tensor blocked_weight;
  at::Tensor blocked_scales = new_scales.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();

#if defined(CPU_CAPABILITY_AVX512)
  if (cpublas_could_pack()) {
    constexpr int vnni_size = 2; // for float16
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
alignas(64) static uint16_t e4m3_to_16bit[256];

template <typename T>
static void initialize_e4m3_to_16bit_tables() {
  // run only once
  static bool initialized_16bit = false;
  if (!initialized_16bit) {
    for (uint8_t u8 = 0; u8 < 256; ++u8) {
      auto value = static_cast<T>(c10::bit_cast<at::Float8_e4m3fn>(u8));
      uint16_t value_bits = c10::bit_cast<uint16_t>(value);
      e4m3_to_16bit[u8] = value_bits;
      if (u8 == 255) {
        break;
      }
    }
    initialized_16bit = true;
  }
}

template <typename T>
static void cvt_e4m3_16bit_intrinsic_lut(
    const at::Float8_e4m3fn* __restrict__ in,
    T* out,
    int64_t len) {
  for (size_t i = 0; i < len; i += 64) {
    __m512i fp8_vec = _mm512_loadu_si512((__m512i*)&in[i]);
    __m128i group0 = _mm512_castsi512_si128(fp8_vec);
    __m128i group1 = _mm512_extracti32x4_epi32(fp8_vec, 1);
    __m128i group2 = _mm512_extracti32x4_epi32(fp8_vec, 2);
    __m128i group3 = _mm512_extracti32x4_epi32(fp8_vec, 3);

    __m512i indices0 = _mm512_cvtepu8_epi32(group0);
    __m512i indices1 = _mm512_cvtepu8_epi32(group1);
    __m512i indices2 = _mm512_cvtepu8_epi32(group2);
    __m512i indices3 = _mm512_cvtepu8_epi32(group3);

    // Gather BF16 conversion results from the lookup table.
    __m512i bf16_i32_vec0 = _mm512_i32gather_epi32(indices0, e4m3_to_16bit, 2);
    __m512i bf16_i32_vec1 = _mm512_i32gather_epi32(indices1, e4m3_to_16bit, 2);
    __m512i bf16_i32_vec2 = _mm512_i32gather_epi32(indices2, e4m3_to_16bit, 2);
    __m512i bf16_i32_vec3 = _mm512_i32gather_epi32(indices3, e4m3_to_16bit, 2);

    // Helper lambda: Convert 16 32-bit ints (in a __m512i) to 16 16-bit ints.
    auto convert_32_to_16 = [](__m512i vec) -> __m256i {
      return _mm512_cvtepi32_epi16(vec);
    };

    __m256i bf16_i16_vec0 = convert_32_to_16(bf16_i32_vec0);
    __m256i bf16_i16_vec1 = convert_32_to_16(bf16_i32_vec1);
    __m256i bf16_i16_vec2 = convert_32_to_16(bf16_i32_vec2);
    __m256i bf16_i16_vec3 = convert_32_to_16(bf16_i32_vec3);

    _mm256_storeu_si256((__m256i*)(out + i + 0), bf16_i16_vec0);
    _mm256_storeu_si256((__m256i*)(out + i + 16), bf16_i16_vec1);
    _mm256_storeu_si256((__m256i*)(out + i + 32), bf16_i16_vec2);
    _mm256_storeu_si256((__m256i*)(out + i + 48), bf16_i16_vec3);
  }
}

static void _convert_B_to_bf16(
    const at::Float8_e4m3fn* __restrict__ B,
    at::BFloat16* dqB,
    int64_t len) {
  initialize_e4m3_to_16bit_tables<at::BFloat16>();
  int tail = len % 64;
  cvt_e4m3_16bit_intrinsic_lut<at::BFloat16>(B, dqB, len - tail);
  for (int i = len - tail; i < len; ++i) {
    dqB[i] = (at::BFloat16)B[i];
  }
}

static void _convert_A_to_bf16(
    const at::Float8_e4m3fn* __restrict__ A,
    at::BFloat16* dqA,
    int64_t M,
    int64_t K,
    int64_t lda) {
  initialize_e4m3_to_16bit_tables<at::BFloat16>();
  for (int m = 0; m < M; ++m) {
    int tail = K % 64;
    int body = K - tail;
    cvt_e4m3_16bit_intrinsic_lut<at::BFloat16>(A + m * lda, dqA + m * K, body);
    for (int k = body; k < K; ++k) {
      dqA[m * K + k] = (at::BFloat16)A[m * lda + k];
    }
  }
}

template <bool accum, int64_t N>
static void _dequant_and_store(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int M,
    int ldi,
    int ldo,
    int ldsa = 1) {
  for (int m = 0; m < M; ++m) {
    float a_scale = *(scale_a + m * ldsa);
    __m512 va_scale = _mm512_set1_ps(a_scale);
    int n = 0;
#pragma GCC unroll 2
    for (; n < N; n += 16) {
      __m512 vc_f = _mm512_loadu_ps(input + m * ldi + n);
      __m512 vc_f_mul = _mm512_mul_ps(vc_f, va_scale);
      __m512 vb_s = _mm512_loadu_ps(scale_b + n);
      vc_f_mul = _mm512_mul_ps(vc_f_mul, vb_s);
      if constexpr (accum) {
        __m512 vo = _mm512_loadu_ps(output + m * ldo + n);
        _mm512_storeu_ps(output + m * ldo + n, _mm512_add_ps(vo, vc_f_mul));
      } else {
        _mm512_storeu_ps(output + m * ldo + n, vc_f_mul);
      }
    }
    for (; n < N; ++n) {
      float dq_val = input[m * ldi + n] * a_scale * scale_b[n];
      if constexpr (accum) {
        output[m * ldo + n] += dq_val;
      } else {
        output[m * ldo + n] = dq_val;
      }
    }
  }
}

#else
static void _convert_B_to_bf16(
    const at::Float8_e4m3fn* B,
    at::BFloat16* dqB,
    int64_t len) {
  for (int i = 0; i < len; ++i) {
    dqB[i] = (at::BFloat16)B[i];
  }
}

static void _convert_A_to_bf16(
    const at::Float8_e4m3fn* __restrict__ A,
    at::BFloat16* dqA,
    int64_t M,
    int64_t K,
    int64_t lda) {
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      dqA[m * K + k] = (at::BFloat16)A[m * lda + k];
    }
  }
}
#endif

template <bool cpublas_can_pack, int64_t N>
void _dequant_gemm_accum(
    float* C,
    const at::Float8_e4m3fn* A,
    const float* scales_a,
    const at::Float8_e4m3fn* B,
    const float* scales_b,
    int64_t M,
    int64_t K,
    int64_t lda,
    int64_t ldc) {
  // Compute GEMM fp8 * fp8 -> fp32
  // Then apply scales and store results
  at::BFloat16 dqB[K * N];
  _convert_B_to_bf16(B, dqB, K * N);
  at::BFloat16 dqA[M * K];
  _convert_A_to_bf16(A, dqA, M, K, lda);
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (cpublas_can_pack) {
    float C_f32[M * N];
    at::native::cpublas::brgemm(
        M,
        N,
        K,
        K /*lda*/,
        N /*ldb*/,
        N /*ldc*/,
        false /* add_C */,
        dqA,
        dqB,
        C_f32,
        true /* is_vnni */);
    _mm_prefetch(B + N * K, _MM_HINT_T0);
    _mm_prefetch(A + K, _MM_HINT_T0);
    _dequant_and_store<true, N>(
        C,
        C_f32,
        scales_a,
        scales_b,
        M,
        N /*ldi*/,
        ldc,
        1 /*ldsa*/);
  } else
#endif
  {
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        float sum = 0;
        for (int64_t k = 0; k < K; ++k) {
          sum += ((float)dqA[i * K + k] * dqB[k * N + j]);
        }
        C[i * ldc + j] += sum * scales_a[i] * scales_b[j];
      }
    }
  }
}

template<int64_t N>
inline void copy_bias(const float* bias_ptr, float* y_buf, int64_t m) {
  if (bias_ptr) {
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 bias_vec = _mm512_loadu_ps(bias_ptr + j);
        _mm512_storeu_ps(y_buf + i * N + j, bias_vec);
      }
#endif
      for (; j < N; ++j) {
        y_buf[i * N + j] = bias_ptr[j];
      }
    }
  } else { // initialize to zero
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 zero_vec = _mm512_setzero_ps();
        _mm512_storeu_ps(y_buf + i * N + j, zero_vec);
      }
#endif
      for (; j < N; ++j) {
        y_buf[i * N + j] = 0;
      }
    }
  }
}

template<typename out_dtype, int64_t N>
inline void store_out(const float* y_buf, out_dtype* c_ptr, int64_t m, /* int64_t n, */ int64_t lda) {
  for (int i = 0; i < m; ++i) {
    int j = 0;
    if constexpr (std::is_same<out_dtype, float>::value) {
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
        _mm512_storeu_ps(c_ptr + i * lda + j, y_vec);
      }
#endif
      for (; j < N; ++j) {
        c_ptr[i * lda + j] = y_buf[i * N + j];
      }
    } else if constexpr (std::is_same<out_dtype, at::BFloat16>::value) {
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
        __m256i y_bf16_vec = at::vec::cvtfp32_bf16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_bf16_vec);
      }
#endif
      for (; j < N; ++j) {
        c_ptr[i * lda + j] = at::BFloat16(y_buf[i * N + j]);
      }
    } else if constexpr (std::is_same<out_dtype, at::Half>::value) {
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
        __m256i y_fp16_vec = at::vec::cvtfp32_fp16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_fp16_vec);
      }
#endif
      for (; j < N; ++j) {
        c_ptr[i * lda + j] = at::Half(y_buf[i * N + j]);
      }
    } else {
      TORCH_CHECK(false, "Unsupported output dtype");
    }
  }
}

template<typename out_dtype, bool cpublas_can_pack>
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
  TORCH_CHECK(input_scales.numel() == M, "Float8 linear: unexpected input scales shape");

  // weight shape = [Nc, Kc, block_k, block_n]
  // scales shape = [Nc, G, block_n]
  int64_t Nc = weight.size(0);
  int64_t Kc = weight.size(1);
  int64_t block_k = weight.size(2);
  constexpr int64_t block_n = BLOCK_N;
  TORCH_CHECK(weight.size(3) == block_n, "Float8 linear: unexpected weight shape");
  int64_t N = Nc * block_n;
  TORCH_CHECK(K == Kc * block_k, "Float8 linear: weight and input shapes mismatch");
  int64_t block_m = [&]() -> long {
    if (M <= 48) {
      return M;
    } else if (M < 64) {
      return 32;
    } else if (M < 96) {
      return 64;
    } else {
      return 128;
    }
  }();
  int64_t Mc = (M + block_m - 1) / block_m;
  bool parallel_on_M = M > 128;
  int64_t num_blocks = parallel_on_M ? Mc * Nc : Nc;

  // scales shape = [Nc, G, block_n]
  int64_t num_groups = weight_scales.size(1);
  int64_t group_size = K / num_groups;
  TORCH_CHECK(group_size % block_k == 0,
              "Float8 linear: group_size should be divisible by block_k");
  int64_t block_per_group = group_size / block_k;

  const at::Float8_e4m3fn* a_ptr = input_view.data_ptr<at::Float8_e4m3fn>();
  const float* a_scales_ptr = input_scales.data_ptr<float>();
  const at::Float8_e4m3fn* b_ptr = weight.data_ptr<at::Float8_e4m3fn>();
  const float* b_scales_ptr = weight_scales.data_ptr<float>();
  out_dtype* c_ptr = output.data_ptr<out_dtype>();
  const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

  at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t mc = parallel_on_M ? i / Nc : 0;
      int64_t nc = parallel_on_M ? i % Nc : i;
      int64_t mc_end = parallel_on_M ? mc + 1 : Mc;

      for (int mci = mc; mci < mc_end; ++mci) {
        int64_t m_size = mci * block_m + block_m > M ? M - mci * block_m : block_m;
        alignas(64) float y_buf[m_size][block_n];
        // copy bias to y_buf if bias is not None
        auto bias_data = bias_ptr ? bias_ptr + nc * block_n : nullptr;
        copy_bias<block_n>(bias_data, y_buf[0], m_size);
        for (int kci = 0; kci < Kc; ++kci) {
          _dequant_gemm_accum<cpublas_can_pack, block_n>(
            y_buf[0] /*C*/,
            a_ptr + mci * block_m * K + kci * block_k /*A*/,
            a_scales_ptr + mci * block_m /*scales_a*/,
            b_ptr + (nc * Kc + kci) * block_n * block_k /*B*/,
            b_scales_ptr + nc * block_n * num_groups + kci / block_per_group * block_n /*scales_b*/,
            m_size /*M*/,
            block_k /*K*/,
            K /*lda*/,
            block_n /*ldc*/);
        }
        // store y_buf to output with dtype conversion
        store_out<out_dtype, block_n>(
          y_buf[0],
          c_ptr + mci * block_m * N + nc * block_n,
          m_size,
          N /*lda*/);
      }
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
  static bool cpublas_can_pack = cpublas_could_pack();
  auto out_sizes = input.sizes().vec();
  int64_t N = weight.size(0) * weight.size(-1);
  out_sizes.back() = N;
  auto output = at::empty(out_sizes, input.options().dtype(output_dtype));

#define call__float8_linear_impl(cpublas_can_pack) \
    AT_DISPATCH_FLOATING_TYPES_AND2( \
        at::ScalarType::BFloat16, at::ScalarType::Half, output_dtype, "float8_linear_cpu", [&] { \
          _float8_linear_impl<scalar_t, cpublas_can_pack>( \
              input, \
              input_scales, \
              weight, \
              weight_scales, \
              bias, \
              output); \
        });

  if (cpublas_can_pack) {
    call__float8_linear_impl(true);
  } else {
    call__float8_linear_impl(false);
  }
  return output;
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::float8_linear_prepack_cpu", &float8_linear_prepack_impl);
  m.impl("torchao::float8_linear_cpu", &float8_linear_impl);
}

} // namespace torchao
