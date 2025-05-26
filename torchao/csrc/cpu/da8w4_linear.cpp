#include <ATen/ATen.h>
// #include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/vec.h>
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

static bool use_cpublas_checked = false;
static bool use_cpublas = false;

bool da8w4_can_pack_weight() {
#if defined(CPU_CAPABILITY_AVX512)
  if (use_cpublas_checked) {
    return use_cpublas;
  }
  use_cpublas = at::native::cpublas::could_pack(at::kByte);
  use_cpublas_checked = true;
  return use_cpublas;
#else
  return false;
#endif
}

/*
return: packed_weight, packed_scales, packed_qzeros, compensation
*/
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
da8w4_linear_prepack_impl(
    const at::Tensor& weight,
    const at::Tensor& scales,
    const at::Tensor& qzeros) {
  // weight shape = [N, K]
  // scales shape = [N, G]
  // qzeros shape = [N, G]
  TORCH_CHECK(weight.dim() == 2,
              "DA8W4 CPU: Weight should be a 2D tensor for packing");
  TORCH_CHECK(weight.size(1) % 2 == 0,
              "DA8W4 CPU: Weight should have even number of columns for packing");

  auto new_scales = scales;
  auto new_qzeros = qzeros;
  if (new_scales.dim() == 1) {
    new_scales.unsqueeze_(1);
  }
  new_scales = new_scales.to(at::kFloat);
  if (new_qzeros.dim() == 1) {
    new_qzeros.unsqueeze_(1);
  }
  new_qzeros = new_qzeros.to(at::kChar);
  int N = weight.size(0);
  int K = weight.size(1);
  int G = scales.size(1);
  int group_size = K / G;
  int block_k = group_size > 128 ? 128 : group_size;
  constexpr int block_n = 32;
  int Nc = N / block_n;
  int Kc = K / block_k;

  // Reorder weight to [N/block_n, K/block_k, block_k, block_n]
  // Reorder scales/qzeros to [N/block_n, G, block_n]
  auto weight_view = weight.view({Nc, block_n, Kc, block_k});
  at::Tensor weight_reordered = weight_view.permute({0, 2, 3, 1}).contiguous();
  at::Tensor blocked_weight;
  at::Tensor blocked_scales = new_scales.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();
  at::Tensor blocked_qzeros = new_qzeros.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();
  // weight was increased by 8 during quantization, so we need to subtract 8
  at::Tensor compensation = weight_view.to(at::kInt).sub(8).sum(-1);
  compensation = compensation.permute({0, 2, 1}).contiguous().to(at::kInt);

  if (da8w4_can_pack_weight()) {
    blocked_weight = at::empty({Nc, Kc, block_k, block_n / 2}, weight.options());
    auto weight_ptr = weight_reordered.data_ptr<uint8_t>();
    auto blocked_weight_ptr = blocked_weight.data_ptr<uint8_t>();
    int64_t num_blocks = Nc * Kc;
    at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        auto in_ptr = weight_ptr + i * block_k * block_n;
        auto out_ptr = blocked_weight_ptr + i * block_k * block_n / 2;

        // Reorder weight block to VNNI4 and pack two lanes along N
        // N=16 viewed as two lanes: a0, ...a7, b0, ...b7
        // pack two lanes: [a0, b0], ..., [a7, b7]
        // plain shape = [block_k, block_n]
        // packed shape = [block_k / 4, block_n / 2, 4] viewed as [block_k, block_n / 2]
        constexpr int n_group_size = 8;
        constexpr int vnni_size = 4;
        constexpr int n_group = block_n / n_group_size; // 4
        for (int nb = 0; nb < n_group; nb += 2) {
          for (int k = 0; k < block_k; k += vnni_size) {
            for (int ni = 0; ni < n_group_size; ++ni) {
              for (int ki = 0; ki < vnni_size; ++ki) {
                int src_idx_1 = nb * n_group_size + ni + (k + ki) * block_n;
                int src_idx_2 = (nb + 1) * n_group_size + ni + (k + ki) * block_n;
                int dst_idx = (nb / 2 * n_group_size + ni) * vnni_size + k * block_n / 2 + ki;
                uint8_t src_1 = *(in_ptr + src_idx_1);
                uint8_t src_2 = *(in_ptr + src_idx_2);
                uint8_t dst = (src_1 & 0x0f) | ((src_2 & 0x0f) << 4);
                *(out_ptr + dst_idx) = dst;
              }
            }
          }
        }
      }
    });
  } else {
    // Pack weight: two int4 -> one int8
    using namespace at::indexing;
    at::Tensor even_columns =
        weight_reordered.index({Slice(), Slice(), Slice(), Slice(1, None, 2)});
    even_columns = even_columns.bitwise_left_shift(4);
    at::Tensor odd_columns =
        weight_reordered.index({Slice(), Slice(), Slice(), Slice(None, None, 2)});
    blocked_weight = even_columns.bitwise_or(odd_columns);
  }

  return std::make_tuple(std::move(blocked_weight), std::move(blocked_scales), std::move(blocked_qzeros), std::move(compensation));
}

#if defined(CPU_CAPABILITY_AVX512)
inline std::array<__m256i, 2> load_zps_4vnni(const int8_t* __restrict__ zps) {
  // broadcast 01234567 to
  // 01234567012345670123456701234567
  __m256i vzps_low = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps));
  __m256i vzps_high = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps + 8));
  // shuffle from
  // 01234567012345670123456701234567
  // to
  // 00001111222233334444555566667777
  __m256i shuffle_mask = _mm256_set_epi8(
      7,
      7,
      7,
      7,
      6,
      6,
      6,
      6,
      5,
      5,
      5,
      5,
      4,
      4,
      4,
      4,
      3,
      3,
      3,
      3,
      2,
      2,
      2,
      2,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0);
  vzps_low = _mm256_shuffle_epi8(vzps_low, shuffle_mask);
  vzps_high = _mm256_shuffle_epi8(vzps_high, shuffle_mask);
  return {vzps_low, vzps_high};
}

inline std::array<__m256i, 2> load_uint4_as_int8(const uint8_t* __restrict__ qB) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qB));
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i high = _mm256_srli_epi16(packed, 4);
  high = _mm256_and_si256(high, low_mask);
  __m256i low = _mm256_and_si256(packed, low_mask);
  return {low, high};
}

void _dequant_weight_zp_only(
    const uint8_t* __restrict__ B,
    int8_t* dqB,
    const int8_t* __restrict__ qzeros,
    int64_t N,
    int64_t K,
    int64_t ldb) {
  // unpack weight int8 -> two int4
  // subtract zero point
  // B shape = [K, ldb] = [K, N / 2], actual shape = [K / 4, N / 2, 4]
  // dqB shape = [K, N], actual shape = [K / 4, N, 4]
  for (int n = 0; n < N; n += 16) {
    auto [zps_low, zps_high] = load_zps_4vnni(&qzeros[n]);
    for (int k = 0; k < K; k += 4) {
      auto [vb_low, vb_high] = load_uint4_as_int8(B + ldb * k + n / 2 * 4);
      vb_high = _mm256_sub_epi8(vb_high, zps_high);
      vb_low = _mm256_sub_epi8(vb_low, zps_low);
      // store vb to B
      _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(dqB + N * k + n * 4), vb_low);
      _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(dqB + N * k + (n + 8) * 4), vb_high);
    }
  }
}

template <bool accum>
void _dequant_and_store(
    float* __restrict__ output,
    const int32_t* __restrict__ input,
    const float* __restrict__ scale_a,
    const int32_t* __restrict__ zp_a,
    const float* __restrict__ scale_b,
    const int32_t* __restrict__ comp_b,
    int M,
    int N,
    int ldi,
    int ldo,
    int ldsa = 1) {
#pragma GCC unroll 2
  for (int m = 0; m < M; ++m) {
    float a_scale = *(scale_a + m * ldsa);
    int32_t a_zp = *(zp_a + m * ldsa);
    __m512 va_scale = _mm512_set1_ps(a_scale);
    __m512i va_zp = _mm512_set1_epi32(a_zp);
    int n = 0;
    for (; n < N; n += 16) {
      __m512i va = _mm512_loadu_si512(input + m * ldi + n);
      __m512i vb_comp = _mm512_loadu_si512(comp_b + n);
      __m512i vc = _mm512_sub_epi32(va, _mm512_mullo_epi32(vb_comp, va_zp));
      __m512 vc_f = _mm512_cvtepi32_ps(vc);
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
      float dq_val =
          (float)(input[m * ldi + n] - a_zp * comp_b[n]) * a_scale * scale_b[n];
      if constexpr (accum) {
        output[m * ldo + n] += dq_val;
      } else {
        output[m * ldo + n] = dq_val;
      }
    }
  }
}

#else
void _dequant_weight_zp_only(
    const uint8_t* B,
    int8_t* dqB,
    const int8_t* qzeros,
    int64_t N,
    int64_t K,
    int64_t ldb) {
  // B shape = [K, N / 2]
  // dqB shape = [K, N]
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N / 2; ++n) {
      int32_t b = (int32_t)B[k * ldb + n];
      dqB[k * N + n * 2] = (b & 0xf) - qzeros[n];
      dqB[k * N + n * 2 + 1] = (b >> 4) - qzeros[n];
    }
  }
}
#endif

template <bool use_cpublas>
void _dequant_gemm_accum(
    float* C,
    const uint8_t* A,
    const float* scales_a,
    const int32_t* qzeros_a,
    const uint8_t* B,
    const float* scales_b,
    const int8_t* qzeros_b,
    const int32_t* compensation,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  // Compute GEMM int8 * int8 -> int32
  // dequant result to float by applying scales/qzeros

  int8_t dqB[K * N];
  _dequant_weight_zp_only(B, dqB, qzeros_b, N, K, ldb);
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (use_cpublas) {
    int32_t C_i32[M * N];
    at::native::cpublas::brgemm(
        M,
        N,
        K,
        lda,
        N /*ldb*/,
        N /*ldc*/,
        false /* add_C */,
        A,
        dqB,
        C_i32,
        true /* is_vnni */);
    _dequant_and_store<true>(
        C,
        C_i32,
        scales_a,
        qzeros_a,
        scales_b,
        compensation,
        M,
        N,
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
          sum += ((int32_t)A[i * lda + k] - qzeros_a[i]) * (int32_t)dqB[k * N + j];
        }
        C[i * ldc + j] += sum * scales_a[i] * scales_b[j];
      }
    }
  }
}

inline void copy_bias(const float* bias_ptr, float* y_buf, int64_t m, int64_t n) {
  if (bias_ptr) {
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
      for (; j < n; j += 16) {
        __m512 bias_vec = _mm512_loadu_ps(bias_ptr + j);
        _mm512_storeu_ps(y_buf + i * n + j, bias_vec);
      }
#endif
      for (; j < n; ++j) {
        y_buf[i * n + j] = bias_ptr[j];
      }
    }
  } else { // initialize to zero
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
      for (; j < n; j += 16) {
        __m512 zero_vec = _mm512_setzero_ps();
        _mm512_storeu_ps(y_buf + i * n + j, zero_vec);
      }
#endif
      for (; j < n; ++j) {
        y_buf[i * n + j] = 0;
      }
    }
  }
}

template<typename out_dtype>
inline void store_out(const float* y_buf, out_dtype* c_ptr, int64_t m, int64_t n, int64_t lda) {
  for (int i = 0; i < m; ++i) {
    int j = 0;
    if constexpr (std::is_same<out_dtype, float>::value) {
#if defined(CPU_CAPABILITY_AVX512)
      for (; j < n; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * n + j);
        _mm512_storeu_ps(c_ptr + i * lda + j, y_vec);
      }
#endif
      for (; j < n; ++j) {
        c_ptr[i * lda + j] = y_buf[i * n + j];
      }
    } else if constexpr (std::is_same<out_dtype, at::BFloat16>::value) {
#if defined(CPU_CAPABILITY_AVX512)
      for (; j < n; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * n + j);
        __m256i y_bf16_vec = at::vec::cvtfp32_bf16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_bf16_vec);
      }
#endif
      for (; j < n; ++j) {
        c_ptr[i * lda + j] = at::BFloat16(y_buf[i * n + j]);
      }
    } else if constexpr (std::is_same<out_dtype, at::Half>::value) {
#if defined(CPU_CAPABILITY_AVX512)
      for (; j < n; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * n + j);
        __m256i y_fp16_vec = at::vec::cvtfp32_fp16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_fp16_vec);
      }
#endif
      for (; j < n; ++j) {
        c_ptr[i * lda + j] = at::Half(y_buf[i * n + j]);
      }
    } else {
      TORCH_CHECK(false, "Unsupported output dtype");
    }
  }
}

template<typename out_dtype, bool use_cpublas>
void _da8w4_linear_impl(
    const at::Tensor& input,
    const at::Tensor& input_scales,
    const at::Tensor& input_qzeros,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_qzeros,
    const at::Tensor& compensation,
    const std::optional<at::Tensor>& bias,
    at::Tensor& output) {
  // input shape = [..., K]
  // input is per token quantized
  int64_t K = input.size(-1);
  auto input_view = input.view({-1, K});
  int64_t M = input_view.size(0);
  TORCH_CHECK(input_scales.numel() == M, "DA8W4: unexpected input scales shape");
  TORCH_CHECK(input_scales.sizes() == input_qzeros.sizes(), "DA8W4: unexpected input qzeros shape");

  // weight shape = [Nc, Kc, block_k, block_n/2]
  // scales/qzeros shape = [Nc, G, block_n]
  // compensation shape = [Nc, Kc, block_n]
  int64_t Nc = weight.size(0);
  int64_t Kc = weight.size(1);
  int64_t block_k = weight.size(2);
  int64_t block_n = weight.size(3) * 2;
  int64_t N = Nc * block_n;
  TORCH_CHECK(K == Kc * block_k, "DA8W4: weight and input shapes mismatch");
  int64_t block_m = [&]() -> long {
    if (M <= 48) {
      return M;
    } else if (M < 64) {
      return 32;
    } else if (M < 96) {
      return 48;
    } else {
      return 64;
    }
  }();
  int64_t Mc = (M + block_m - 1) / block_m;
  bool parallel_on_M = M > 128;
  int64_t num_blocks = parallel_on_M ? Mc * Nc : Nc;

  // scales/qzeros shape = [Nc, G, block_n]
  int64_t num_groups = weight_scales.size(1);
  int64_t group_size = K / num_groups;
  TORCH_CHECK(group_size % block_k == 0,
              "DA8W4 CPU: group_size should be divisible by block_k");
  int64_t block_per_group = group_size / block_k;

  const uint8_t* a_ptr = input_view.data_ptr<uint8_t>();
  const float* a_scales_ptr = input_scales.data_ptr<float>();
  const int32_t* a_qzeros_ptr = input_qzeros.data_ptr<int32_t>();
  const uint8_t* b_ptr = weight.data_ptr<uint8_t>();
  const float* b_scales_ptr = weight_scales.data_ptr<float>();
  const int8_t* b_qzeros_ptr = weight_qzeros.data_ptr<int8_t>();
  const int32_t* compensation_ptr = compensation.data_ptr<int32_t>();
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
        copy_bias(bias_data, y_buf[0], m_size, block_n);
        for (int kci = 0; kci < Kc; ++kci) {
          _dequant_gemm_accum<use_cpublas>(
            y_buf[0] /*C*/,
            a_ptr + mci * block_m * K + kci * block_k /*A*/,
            a_scales_ptr + mci * block_m /*scales_a*/,
            a_qzeros_ptr + mci * block_m /*qzeros_a*/,
            b_ptr + (nc * Kc + kci) * block_n * block_k / 2 /*B*/,
            b_scales_ptr + nc * block_n * num_groups + kci / block_per_group * block_n /*scales_b*/,
            b_qzeros_ptr + nc * block_n * num_groups + kci / block_per_group * block_n /*qzeros_b*/,
            compensation_ptr + nc * block_n * Kc + kci * block_n /*compensation*/,
            m_size /*M*/,
            block_n /*N*/,
            block_k /*K*/,
            K /*lda*/,
            block_n / 2 /*ldb*/,
            block_n /*ldc*/);
        }
        // store y_buf to output
        store_out<out_dtype>(y_buf[0], c_ptr + mci * block_m * N + nc * block_n, m_size, block_n, N);
      }
    }
  });
  if constexpr (use_cpublas) {
    at::native::cpublas::brgemm_release();
  }
}

at::Tensor da8w4_linear_impl(
    const at::Tensor& input,
    const at::Tensor& input_scales,
    const at::Tensor& input_qzeros,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_qzeros,
    const at::Tensor& compensation,
    const std::optional<at::Tensor>& bias,
    at::ScalarType output_dtype) {
  static bool use_cpublas = da8w4_can_pack_weight();
  auto out_sizes = input.sizes().vec();
  int64_t N = weight.size(0) * weight.size(-1) * 2;
  out_sizes.back() = N;
  auto output = at::empty(out_sizes, input.options().dtype(output_dtype));
  if (use_cpublas) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half, output_dtype, "da8w4_linear_cpu", [&] {
          _da8w4_linear_impl<scalar_t, true>(
              input,
              input_scales,
              input_qzeros,
              weight,
              weight_scales,
              weight_qzeros,
              compensation,
              bias,
              output);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half, output_dtype, "da8w4_linear_cpu", [&] {
          _da8w4_linear_impl<scalar_t, false>(
              input,
              input_scales,
              input_qzeros,
              weight,
              weight_scales,
              weight_qzeros,
              compensation,
              bias,
              output);
        });
  }
  return output;
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::da8w4_linear_prepack_cpu", &da8w4_linear_prepack_impl);
  m.impl("torchao::da8w4_linear_cpu", &da8w4_linear_impl);
}

} // namespace torchao
