// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#endif // defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <optional>
#include <vector>

#if defined(USE_ATEN) && !defined(USE_EXECUTORCH)
#pragma message("USE_ATEN")
#include <torch/library.h>
#include <torch/script.h>
#include <torch/torch.h>
using Tensor = at::Tensor;
#define CHECK_MSG(cond, msg) TORCH_CHECK(cond, msg)

#elif defined(USE_EXECUTORCH) && !defined(USE_ATEN)
#pragma message("USE_EXECUTORCH")
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>
using Tensor = torch::executor::Tensor;
using RuntimeContext = torch::executor::KernelRuntimeContext;
#define CHECK_MSG(cond, msg) ET_CHECK_MSG(cond, msg)
#else
#error "Must define either USE_ATEN or USE_EXECUTORCH"
#endif

namespace {

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
inline torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig
get_ukernel_config() {
  torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig config;

#if defined(__aarch64__) || defined(__ARM_NEON)
  namespace ukernel = torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot;
  config.mr = 1;
  config.nr = 8;
  config.activation_data_size_fn =
      &ukernel::activation_data_size<has_weight_zeros>;
  config.activation_data_alignment = 16; // size of neon register
  config.prepare_activation_data_fn =
      &ukernel::prepare_activation_data<has_weight_zeros>;
  config.weight_data_size_fn =
      &ukernel::weight_data_size<weight_nbit, has_weight_zeros>;
  config.weight_data_alignment = 16; // size of neon register
  config.prepare_weight_data_fn =
      &ukernel::prepare_weight_data<weight_nbit, has_weight_zeros>;
  config.kernel_fn =
      &ukernel::kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>;
#endif // defined(__aarch64__) || defined(__ARM_NEON)

  return config;
}

#ifdef USE_ATEN
template <int weight_nbit, bool has_weight_zeros>
Tensor pack_weights_cpu(
    const Tensor& weight_qvals,
    const Tensor& weight_scales,
    const std::optional<Tensor>& weight_zeros,
    int64_t group_size) {
  CHECK_MSG(weight_qvals.dtype() == torch::kInt8, "weight_qvals must be int8");
  CHECK_MSG(weight_qvals.dim() == 2, "weight_qvals must be 2D");

  // In PyTorch, weights are nxk in row-major format (with activations being
  // right-multiplied).
  // In kernel, activations are left-multiplied by kxn transposed
  // weights in column-major format.
  // Note the underlying data is the same in both cases
  int n = weight_qvals.size(0);
  int k = weight_qvals.size(1);

  CHECK_MSG(
      weight_scales.dtype() == torch::kFloat32,
      "weight_scales must be float32");
  CHECK_MSG(weight_scales.dim() == 1, "weight_scales must be 1D");
  CHECK_MSG(group_size >= 1, "group_size must be >= 1");
  CHECK_MSG(
      weight_scales.size(0) == ((n * k) / group_size),
      "expected 1 scale per group");

  CHECK_MSG(
      has_weight_zeros == weight_zeros.has_value(),
      "has_weight_zeros must match weight_zeros.has_value()");
  const int8_t* weight_zeros_ptr = nullptr;
  if constexpr (has_weight_zeros) {
    CHECK_MSG(
        weight_zeros.value().dtype() == torch::kInt8,
        "weight_zeros must be int8");
    CHECK_MSG(weight_zeros.value().dim() == 1, "weight_zeros must be 1D");
    CHECK_MSG(
        weight_zeros.value().size(0) == ((n * k) / group_size),
        "expected 1 zero per group");
    weight_zeros_ptr = weight_zeros.value().const_data_ptr<int8_t>();
  }

  using namespace torchao::ops::linear_8bit_act_xbit_weight;

  auto ukernel_config = get_ukernel_config<
      weight_nbit,
      has_weight_zeros,
      false /*has_bias*/,
      false /*has_clamp*/>();
  auto pack_weight_tiling_params = get_default_pack_weight_data_tiling_params(
      ukernel_config, n, /*target_panels_per_thread=*/1);

  auto packed_weight_data_size =
      get_packed_weight_data_size(ukernel_config, n, k, group_size);
  Tensor packed_weights = torch::empty({packed_weight_data_size}, torch::kInt8);
  pack_weight_data_operator(
      ukernel_config,
      pack_weight_tiling_params,
      packed_weights.mutable_data_ptr<int8_t>(),
      n,
      k,
      group_size,
      weight_qvals.const_data_ptr<int8_t>(),
      weight_scales.const_data_ptr<float>(),
      weight_zeros_ptr);

  return packed_weights;
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit>
Tensor pack_weights_without_zeros_cpu(
    const Tensor& weight_qvals,
    const Tensor& weight_scales,
    // TODO(T200095131): convert to int64_t when supported by AOTI
    // group_size is a tensor with size (0, group_size)
    const Tensor& group_size_tensor) {
  int64_t group_size = group_size_tensor.size(1);
  return pack_weights_cpu<weight_nbit, /*has_weight_zeros*/ false>(
      weight_qvals, weight_scales, std::nullopt, group_size);
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit>
Tensor pack_weights_with_zeros_cpu(
    const Tensor& weight_qvals,
    const Tensor& weight_scales,
    const Tensor& weight_zeros,
    // TODO(T200095131): convert to int64_t when supported by AOTI
    // group_size is a meta tensor with size (group_size)
    const Tensor& group_size_tensor) {
  int64_t group_size = group_size_tensor.size(1);
  return pack_weights_cpu<weight_nbit, /*has_weight_zeros*/ true>(
      weight_qvals, weight_scales, weight_zeros, group_size);
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit, bool has_weight_zeros>
Tensor pack_weights_meta(
    const Tensor& weight_qvals,
    const Tensor& weight_scales,
    const std::optional<Tensor>& weight_zeros,
    int64_t group_size) {
  CHECK_MSG(group_size >= 1, "group_size must be >= 1");
  int n = weight_qvals.size(0);
  int k = weight_qvals.size(1);

  using namespace torchao::ops::linear_8bit_act_xbit_weight;

  auto ukernel_config = get_ukernel_config<
      weight_nbit,
      has_weight_zeros,
      false /*has_bias*/,
      false /*has_clamp*/>();

  auto packed_weight_data_size =
      get_packed_weight_data_size(ukernel_config, n, k, group_size);
  return torch::empty({packed_weight_data_size}).to("meta");
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit>
Tensor pack_weights_without_zeros_meta(
    const Tensor& weight_qvals,
    const Tensor& weight_scales,
    // TODO(T200095131): convert to int64_t when supported by AOTI
    // group_size is a meta tensor with size (group_size)
    const Tensor& group_size_tensor) {
  int64_t group_size = group_size_tensor.size(1);
  return pack_weights_meta<weight_nbit, /*has_weight_zeros*/ false>(
      weight_qvals, weight_scales, std::nullopt, group_size);
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit>
Tensor pack_weights_with_zeros_meta(
    const Tensor& weight_qvals,
    const Tensor& weight_scales,
    const Tensor& weight_zeros,
    // TODO(T200095131): convert to int64_t when supported by AOTI
    // group_size is a meta tensor with size (group_size)
    const Tensor& group_size_tensor) {
  int64_t group_size = group_size_tensor.size(1);
  return pack_weights_meta<weight_nbit, /*has_weight_zeros*/ true>(
      weight_qvals, weight_scales, weight_zeros, group_size);
}
#endif // USE_ATEN

#if defined(USE_ATEN) || defined(USE_EXECUTORCH)
template <int weight_nbit, bool has_weight_zeros>
Tensor linear_out_cpu(
    const Tensor& activations,
    const Tensor& packed_weights,
    // TODO(T200095131): convert n_tensor, k_tensor, group_size_tensor to
    // int64_t when supported by AOTI Currently they are tensors with size
    // equal to (0, the int they wrap)
    const Tensor& group_size_tensor,
    const Tensor& n_tensor,
    const Tensor& k_tensor,
    Tensor& out) {
  int n = n_tensor.size(1);
  int k = k_tensor.size(1);
  int group_size = group_size_tensor.size(1);
  CHECK_MSG(n >= 1, "n must be >= 1");
  CHECK_MSG(k >= 1, "k must be >= 1");
  CHECK_MSG(group_size >= 1, "group_size must be >= 1");

#ifdef USE_ATEN
  CHECK_MSG(
      activations.dtype() == torch::kFloat32, "activations must be float32");
#endif // USE_ATEN

  CHECK_MSG(activations.dim() == 2, "activations must be 2D");
  int m = activations.size(0);
  int k_ = activations.size(1);
  CHECK_MSG(k == k_, "activation shape is incompatible with packed weights.");

#ifdef USE_ATEN
  CHECK_MSG(out.dtype() == torch::kFloat32, "out must be float32");
#endif // USE_ATEN

#ifdef USE_ATEN
  out.resize_({m, n});
#endif // USE_ATEN

#ifdef USE_EXECUTORCH
  CHECK_MSG(out.dim() == 2, "out must be 2D");
  CHECK_MSG(out.size(0) == m, "out shape is incorrect");
  CHECK_MSG(out.size(1) == n, "out shape is incorrect");
#endif // USE_EXECUTORCH

  using namespace torchao::ops::linear_8bit_act_xbit_weight;

  auto ukernel_config = get_ukernel_config<
      weight_nbit,
      has_weight_zeros /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/>();
  auto linear_tiling_params = get_default_linear_tiling_params(
      ukernel_config,
      m,
      n,
      /*target_tiles_per_thread=*/5);
  auto linear_scheduling_policy =
      LinearTileSchedulingPolicy::single_mc_parallel_nc;

  auto activation_data_buffer_size = get_activation_data_buffer_size(
      ukernel_config,
      linear_tiling_params,
      linear_scheduling_policy,
      m,
      k,
      group_size);
  std::vector<char> activation_data_buffer(activation_data_buffer_size);

  linear_operator(
      ukernel_config,
      linear_tiling_params,
      linear_scheduling_policy,
      activation_data_buffer.data(),
      out.mutable_data_ptr<float>(),
      m,
      n,
      k,
      group_size,
      packed_weights.const_data_ptr<int8_t>(),
      activations.const_data_ptr<float>(),
      /*bias=*/nullptr,
      // Clamp parameters are ignored because config is created from
      // has_clamp = false
      /*clamp_min=*/0.0,
      /*clamp_max=*/0.0);

  return out;
}
#endif // defined(USE_ATEN) || defined(USE_EXECUTORCH)

#ifdef USE_ATEN
template <int weight_nbit, bool has_weight_zeros>
Tensor linear_cpu(
    const Tensor& activations,
    const Tensor& packed_weights,
    // TODO(T200095131): convert n_tensor, k_tensor, group_size_tensor to
    // int64_t when supported by AOTI Currently they are tensors with size
    // equal to (0, the int they wrap)
    const Tensor& group_size_tensor,
    const Tensor& n_tensor,
    const Tensor& k_tensor) {
  Tensor output_tensor = torch::empty({}, torch::kFloat32);
  linear_out_cpu<weight_nbit, has_weight_zeros>(
      activations,
      packed_weights,
      group_size_tensor,
      n_tensor,
      k_tensor,
      output_tensor);
  return output_tensor;
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit, bool has_weight_zeros>
Tensor linear_meta(
    const Tensor& activations,
    const Tensor& packed_weights,
    // TODO(T200095131): convert n_tensor, k_tensor, group_size_tensor to
    // int64_t when supported by AOTI
    // Currently they are tensors with size equal to (0, the int they wrap)
    const Tensor& group_size_tensor,
    const Tensor& n_tensor,
    const Tensor& k_tensor) {
  int n = n_tensor.size(1);
  int k = k_tensor.size(1);
  CHECK_MSG(n >= 1, "n must be >= 1");
  CHECK_MSG(k >= 1, "k must be >= 1");

  CHECK_MSG(activations.dim() == 2, "activations must be 2D");
  int m = activations.size(0);
  int k_ = activations.size(1);
  CHECK_MSG(k == k_, "activation shape is incompatible with packed weights.");
  return torch::empty({m, n}).to("meta");
}
#endif // USE_ATEN

} // namespace
