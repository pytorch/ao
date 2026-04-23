#include <torch/all.h>
#include "utils.h"

/*
To add a new kernel:
1. Implement the kernel in the all namespace (e.g., AVX10_2, AVX512, DEFAULT). See existing kernel files for reference.
  Note: Kernel files must be named in the format of <kernel_name>_krnl.cpp (e.g., da8w4_linear_krnl.cpp).
2. Declare the kernel function type as <kernel_name>_fn.
3. Add an entry in the KernelDispatcher struct for the new kernel.
4. Add a declaration of the kernel function in all namespaces.
5. Add an entry in the get_kernel_dispatcher function.
6. Add a wrapper that calls kernel that the dispatcher selects at runtime.
7. Register the python op with the wrapper.
*/
namespace torchao {

/********** Lightweight ISA-based Dispatcher **********/
// Function pointer types for each kernel
using da8w4_linear_prepack_fn = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&);

using da8w4_linear_fn = at::Tensor(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const std::optional<at::Tensor>&, at::ScalarType);

using float8_linear_prepack_fn = std::tuple<at::Tensor, at::Tensor>(*)(
    const at::Tensor&, const at::Tensor&);

using float8_linear_fn = at::Tensor(*)(
    const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, at::ScalarType);

using scaled_embedding_bag_fn = at::Tensor(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, double, int64_t, bool, at::ScalarType);

using qscaled_dot_product_fn = at::Tensor(*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    std::optional<at::Tensor>, double, bool, std::optional<double>,
    double, int64_t, double, int64_t, double, int64_t,
    double, int64_t, double, int64_t);

// Dispatcher table: holds function pointers for all kernels
struct KernelDispatcher {
  da8w4_linear_prepack_fn da8w4_linear_prepack;
  da8w4_linear_fn da8w4_linear;
  float8_linear_prepack_fn float8_linear_prepack;
  float8_linear_fn float8_linear;
  scaled_embedding_bag_fn scaled_embedding_bag;
  qscaled_dot_product_fn qscaled_dot_product;
};

/********** DA8W4 Linear Kernel **********/
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

/********** FLOAT8 Linear Kernel **********/
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

/********** Scaled Embedding Bag Kernel **********/
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

/********** Quantized SDPA Kernel **********/
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

declare_all_kernels(AVX10_2)
declare_all_kernels(AVX512)
declare_all_kernels(DEFAULT)

/********** Dispatcher Selection and Dispatch Functions **********/
// Select the appropriate dispatcher based on runtime ISA capabilities
#define CREATE_DISPATCHER(namespace_name) \
  { \
    namespace_name::da8w4_linear_prepack_impl, \
    namespace_name::da8w4_linear_impl, \
    namespace_name::float8_linear_prepack_impl, \
    namespace_name::float8_linear_impl, \
    namespace_name::_scaled_embedding_bag_impl, \
    namespace_name::_qscaled_dot_product_cpu \
  }

static std::unordered_map<std::string, DispatchMode> dispatch_mode_map = {
  {"DEFAULT", MODE_DEFAULT},
  {"AVX512", MODE_AVX512}, // Build with AVX512. Brgemm disabled manually.
  {"AMX", MODE_AMX}, // Build with AVX512. Brgemm enabled manually.
  {"AVX10_2", MODE_AVX10_2},
  {"AUTO", MODE_AUTO}
};

KernelDispatcher& get_kernel_dispatcher() {
  // Setting dispatch_mode is useful for validation of different ISA on one machine.
  static const char* env_dispatch = std::getenv(TORCHAO_CPU_DISPATCH_ENV);
  static std::string dispatch_str = env_dispatch ? std::string(env_dispatch) : "AUTO";
  static std::once_flag dispatch_init_flag;
  std::call_once(dispatch_init_flag, [&]() {
    if (dispatch_mode == -1) {
      if (dispatch_mode_map.contains(dispatch_str)) {
        dispatch_mode = dispatch_mode_map[dispatch_str];
      } else {
        TORCH_WARN("Torchao X86 Kernel dispatch: Unrecognized TORCHAO_CPU_DISPATCH value: ", dispatch_str, ", defaulting to AUTO");
        dispatch_mode = MODE_AUTO;
      }
    }
  });

  static const char* env_dispatch_debug = std::getenv(TORCHAO_CPU_DISPATCH_DEBUG_ENV);
  static bool dispatch_debug = env_dispatch_debug ? std::string(env_dispatch_debug) == "1" : false;

  static KernelDispatcher dispatcher = []() {
    KernelDispatcher d;
    // Select ISA level based on runtime detection (kHas*) and compile-time checks (BUILD_*)
#if defined(BUILD_AVX10_2)
    if (kHasAVX10_2 && dispatch_mode >= MODE_AVX10_2) {
      PRINT_DEBUG_INFO("AVX10_2");
      d = CREATE_DISPATCHER(AVX10_2);
      return d;
    }
#endif
#if defined(BUILD_AVX512)
    if (kHasAVX512 && dispatch_mode >= MODE_AVX512) {
      auto isa = (dispatch_mode == MODE_AVX512) ? "AVX512" : "AMX";
      PRINT_DEBUG_INFO(isa);
      d = CREATE_DISPATCHER(AVX512);
      return d;
    }
#endif
    // Fall back to DEFAULT (always available)
    PRINT_DEBUG_INFO("DEFAULT");
    d = CREATE_DISPATCHER(DEFAULT);
    return d;
  }();
  return dispatcher;
}

/********** Wrapper functions of kernels for op registration **********/
declare_da8w4_linear_prepack_impl {
  return get_kernel_dispatcher().da8w4_linear_prepack(weight, scales, qzeros);
}

declare_da8w4_linear_impl {
  return get_kernel_dispatcher().da8w4_linear(
      input, input_scales, input_qzeros, weight, weight_scales, weight_qzeros,
      compensation, bias, output_dtype);
}

declare_float8_linear_prepack_impl {
  return get_kernel_dispatcher().float8_linear_prepack(weight, scales);
}

declare_float8_linear_impl {
  return get_kernel_dispatcher().float8_linear(
      input, input_scales, weight, weight_scales, bias, output_dtype);
}

declare_scaled_embedding_bag_impl {
  return get_kernel_dispatcher().scaled_embedding_bag(
      qweight, indices, offsets, w_scales, o_scale, mode, include_last_offset,
      output_dtype);
}

declare_qscaled_dot_product_impl {
  return get_kernel_dispatcher().qscaled_dot_product(
      query, key, value, attn_mask, dropout_p, is_causal, scale, q_scale, q_zp,
      k_scale, k_zp, v_scale, v_zp, a_scale, a_zp, o_scale, o_zp);
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
