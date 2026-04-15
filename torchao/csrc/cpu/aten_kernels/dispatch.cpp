#include <torch/all.h>
#include "utils.h"
#include <iostream>

namespace torchao {

/********** DA8W4 Linear Kernel Declare **********/
#define declare_da8w4_linear_prepack_impl \
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> \
  da8w4_linear_prepack_impl( \
    const at::Tensor& weight, \
    const at::Tensor& scales, \
    const at::Tensor& qzeros)

#define declare_da8w4_linear_impl \
  at::Tensor da8w4_linear_impl( \
    const at::Tensor& input, \
    const at::Tensor& input_scales, \
    const at::Tensor& input_qzeros, \
    const at::Tensor& weight, \
    const at::Tensor& weight_scales, \
    const at::Tensor& weight_qzeros, \
    const at::Tensor& compensation, \
    const std::optional<at::Tensor>& bias, \
    at::ScalarType output_dtype)

#define call_da8w4_linear_prepack_impl() \
  da8w4_linear_prepack_impl(weight, scales, qzeros)

#define call_da8w4_linear_impl() \
  da8w4_linear_impl( \
    input, \
    input_scales, \
    input_qzeros, \
    weight, \
    weight_scales, \
    weight_qzeros, \
    compensation, \
    bias, \
    output_dtype)

/********** FLOAT8 Linear Kernel Declare **********/
#define declare_float8_linear_prepack_impl \
  std::tuple<at::Tensor, at::Tensor> \
  float8_linear_prepack_impl( \
    const at::Tensor& weight, \
    const at::Tensor& scales)

#define declare_float8_linear_impl \
  at::Tensor float8_linear_impl( \
    const at::Tensor& input, \
    const at::Tensor& input_scales, \
    const at::Tensor& weight, \
    const at::Tensor& weight_scales, \
    const std::optional<at::Tensor>& bias, \
    at::ScalarType output_dtype)

#define call_float8_linear_prepack_impl() \
  float8_linear_prepack_impl(weight, scales)

#define call_float8_linear_impl() \
  float8_linear_impl( \
    input, \
    input_scales, \
    weight, \
    weight_scales, \
    bias, \
    output_dtype)

/********** Scaled Embedding Bag Kernel Declare **********/
#define declare_scaled_embedding_bag_impl \
  at::Tensor _scaled_embedding_bag_impl( \
    const at::Tensor &qweight, \
    const at::Tensor &indices, \
    const at::Tensor &offsets, \
    const at::Tensor &w_scales, \
    double o_scale, \
    const int64_t mode, \
    bool include_last_offset, \
    at::ScalarType output_dtype)

#define call_scaled_embedding_bag_impl() \
  _scaled_embedding_bag_impl( \
    qweight, \
    indices, \
    offsets, \
    w_scales, \
    o_scale, \
    mode, \
    include_last_offset, \
    output_dtype)

/********** Quantized SDPA Kernel Declare **********/
#define declare_qscaled_dot_product_impl \
  at::Tensor _qscaled_dot_product_cpu( \
    const at::Tensor& query, \
    const at::Tensor& key, \
    const at::Tensor& value, \
    std::optional<at::Tensor> attn_mask, \
    double dropout_p, \
    bool is_causal, \
    std::optional<double> scale, \
    double q_scale, \
    int64_t q_zp, \
    double k_scale, \
    int64_t k_zp, \
    double v_scale, \
    int64_t v_zp, \
    double a_scale, \
    int64_t a_zp, \
    double o_scale, \
    int64_t o_zp)

#define call_qscaled_dot_product_impl() \
  _qscaled_dot_product_cpu( \
    query, \
    key, \
    value, \
    attn_mask, \
    dropout_p, \
    is_causal, \
    scale, \
    q_scale, \
    q_zp, \
    k_scale, \
    k_zp, \
    v_scale, \
    v_zp, \
    a_scale, \
    a_zp, \
    o_scale, \
    o_zp)

/********** Declare All Kernels in All Namespaces **********/
#define declare_all_kernels(namespace_name) \
  namespace namespace_name { \
    declare_da8w4_linear_prepack_impl; \
    declare_da8w4_linear_impl; \
    declare_float8_linear_prepack_impl; \
    declare_float8_linear_impl; \
    declare_scaled_embedding_bag_impl; \
    declare_qscaled_dot_product_impl; \
  }

declare_all_kernels(avx10_2)
declare_all_kernels(avx512)
declare_all_kernels(default_scalar)

/********** DA8W4 Linear Kernel Dispatch **********/
declare_da8w4_linear_prepack_impl {
#if defined(BUILD_AVX10_2) && __GNUC__ >= 15
  if (kHasAVX10_2) {
    std::cout << "Using AVX10.2 DA8W4 linear prepack kernel" << std::endl;
    return avx10_2::call_da8w4_linear_prepack_impl();
  }
#endif
#if defined(BUILD_AVX512)
  if (kHasAVX512) {
    std::cout << "Using AVX512 DA8W4 linear prepack kernel" << std::endl;
    return avx512::call_da8w4_linear_prepack_impl();
  }
#endif
  std::cout << "Using default scalar DA8W4 linear prepack kernel" << std::endl;
  return default_scalar::call_da8w4_linear_prepack_impl();
}

declare_da8w4_linear_impl {
#if defined(BUILD_AVX10_2) && __GNUC__ >= 15
  if (kHasAVX10_2) {
    std::cout << "Using AVX10.2 DA8W4 linear kernel" << std::endl;
    return avx10_2::call_da8w4_linear_impl();
  }
#endif
#if defined(BUILD_AVX512)
  if (kHasAVX512) {
    std::cout << "Using AVX512 DA8W4 linear kernel" << std::endl;
    return avx512::call_da8w4_linear_impl();
  }
#endif
  std::cout << "Using default scalar DA8W4 linear kernel" << std::endl;
  return default_scalar::call_da8w4_linear_impl();
}

/********** FLOAT8 Linear Kernel Dispatch **********/
declare_float8_linear_prepack_impl {
#if defined(BUILD_AVX10_2) && __GNUC__ >= 15
  if (kHasAVX10_2) {
    std::cout << "Using AVX10.2 FLOAT8 linear prepack kernel" << std::endl;
    return avx10_2::call_float8_linear_prepack_impl();
  }
#endif
#if defined(BUILD_AVX512)
  if (kHasAVX512) {
    std::cout << "Using AVX512 FLOAT8 linear prepack kernel" << std::endl;
    return avx512::call_float8_linear_prepack_impl();
  }
#endif
  std::cout << "Using default scalar FLOAT8 linear prepack kernel" << std::endl;
  return default_scalar::call_float8_linear_prepack_impl();
}

declare_float8_linear_impl {
#if defined(BUILD_AVX10_2) && __GNUC__ >= 15
  if (kHasAVX10_2) {
    std::cout << "Using AVX10.2 FLOAT8 linear kernel" << std::endl;
    return avx10_2::call_float8_linear_impl();
  }
#endif
#if defined(BUILD_AVX512)
  if (kHasAVX512) {
    std::cout << "Using AVX512 FLOAT8 linear kernel" << std::endl;
    return avx512::call_float8_linear_impl();
  }
#endif
  std::cout << "Using default scalar FLOAT8 linear kernel" << std::endl;
  return default_scalar::call_float8_linear_impl();
}

/********** Scaled Embedding Bag Kernel Dispatch **********/
declare_scaled_embedding_bag_impl {
#if defined(BUILD_AVX10_2) && __GNUC__ >= 15
  if (kHasAVX10_2) {
    std::cout << "Using AVX10.2 Scaled Embedding Bag kernel" << std::endl;
    return avx10_2::call_scaled_embedding_bag_impl();
  }
#endif
#if defined(BUILD_AVX512)
  if (kHasAVX512) {
    std::cout << "Using AVX512 Scaled Embedding Bag kernel" << std::endl;
    return avx512::call_scaled_embedding_bag_impl();
  }
#endif
  std::cout << "Using default scalar Scaled Embedding Bag kernel" << std::endl;
  return default_scalar::call_scaled_embedding_bag_impl();
}

/********** Quantized SDPA Kernel **********/
declare_qscaled_dot_product_impl {
#if defined(BUILD_AVX10_2) && __GNUC__ >= 15
  if (kHasAVX10_2) {
    std::cout << "Using AVX10.2 Quantized SDPA kernel" << std::endl;
    return avx10_2::call_qscaled_dot_product_impl();
  }
#endif
#if defined(BUILD_AVX512)
  if (kHasAVX512) {
    std::cout << "Using AVX512 Quantized SDPA kernel" << std::endl;
    return avx512::call_qscaled_dot_product_impl();
  }
#endif
  std::cout << "Using default scalar Quantized SDPA kernel" << std::endl;
  return default_scalar::call_qscaled_dot_product_impl();
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::da8w4_linear_prepack_cpu", &da8w4_linear_prepack_impl);
  m.impl("torchao::da8w4_linear_cpu", &da8w4_linear_impl);
  m.impl("torchao::float8_linear_prepack_cpu", &float8_linear_prepack_impl);
  m.impl("torchao::float8_linear_cpu", &float8_linear_impl);
  m.impl("torchao::_scaled_embedding_bag", &_scaled_embedding_bag_impl);
  m.impl("torchao::qscaled_dot_product", &_qscaled_dot_product_cpu);
}

} // namespace torchao
