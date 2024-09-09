// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/library.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchao/experimental/kernels/cpu/linear/channelwise_8bit_activation_groupwise_lowbit_weight.h>
#include <torchao/experimental/kernels/cpu/parallel.h>

template <int weight_nbit>
at::Tensor pack_weights_without_zeros_cpu(
    const at::Tensor& weight_qvals,
    const at::Tensor& weight_scales,
    // TODO(T200095131): convert to int64_t when supported by AOTI
    // group_size is a meta tensor with size (group_size)
    const at::Tensor& group_size_tensor) {
  int64_t group_size = group_size_tensor.size(0);

  TORCH_CHECK(
      weight_qvals.dtype() == torch::kInt8, "weight_qvals must be int8");
  TORCH_CHECK(weight_qvals.dim() == 2, "weight_qvals must be 2D");

  // In PyTorch, weights are nxk in row-major format (with activations being
  // right-multiplied).
  // In kernel, activations are left-multiplied by kxn transposed
  // weights in column-major format.
  // Note the underlying data is the same in both cases
  int n = weight_qvals.size(0);
  int k = weight_qvals.size(1);

  TORCH_CHECK(
      weight_scales.dtype() == torch::kFloat32,
      "weight_scales must be float32");
  TORCH_CHECK(weight_scales.dim() == 1, "weight_scales must be 1D");
  TORCH_CHECK(
      weight_scales.size(0) == ((n * k) / group_size),
      "expected 1 scale per group");

  using namespace torchao::operators::cpu::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  auto ukernel_config = get_ukernel_config<
      weight_nbit,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/>();
  auto pack_weight_tiling_params = get_default_pack_weight_data_tiling_params(
      ukernel_config, n, /*target_panels_per_thread=*/1);

  auto packed_weight_data_size =
      get_packed_weight_data_size(ukernel_config, n, k, group_size);
  at::Tensor packed_weights =
      torch::empty({packed_weight_data_size}, torch::kInt8);
  pack_weight_data_operator(
      ukernel_config,
      pack_weight_tiling_params,
      packed_weights.data_ptr<int8_t>(),
      n,
      k,
      group_size,
      weight_qvals.const_data_ptr<int8_t>(),
      weight_scales.const_data_ptr<float>(),
      /*weight_zeros=*/nullptr);

  return packed_weights;
}

template <int weight_nbit>
at::Tensor pack_weights_with_zeros_cpu(
    const at::Tensor& weight_qvals,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_zeros,
    // TODO(T200095131): convert to int64_t when supported by AOTI
    // group_size is a meta tensor with size (group_size)
    const at::Tensor& group_size_tensor) {
  int64_t group_size = group_size_tensor.size(0);

  TORCH_CHECK(
      weight_qvals.dtype() == torch::kInt8, "weight_qvals must be int8");
  TORCH_CHECK(weight_qvals.dim() == 2, "weight_qvals must be 2D");

  // In PyTorch, weights are nxk in row-major format (with activations being
  // right-multiplied).
  // In kernel, activations are left-multiplied by kxn transposed
  // weights in column-major format.
  // Note the underlying data is the same in both cases
  int n = weight_qvals.size(0);
  int k = weight_qvals.size(1);

  TORCH_CHECK(
      weight_scales.dtype() == torch::kFloat32,
      "weight_scales must be float32");
  TORCH_CHECK(weight_scales.dim() == 1, "weight_scales must be 1D");
  TORCH_CHECK(
      weight_scales.size(0) == ((n * k) / group_size),
      "expected 1 scale per group");
  TORCH_CHECK(
      weight_zeros.dtype() == torch::kInt8, "weight_zeros must be int8");
  TORCH_CHECK(weight_zeros.dim() == 1, "weight_zeros must be 1D");
  TORCH_CHECK(
      weight_zeros.size(0) == ((n * k) / group_size),
      "expected 1 zero per group");

  using namespace torchao::operators::cpu::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  auto ukernel_config = get_ukernel_config<
      weight_nbit,
      true /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/>();
  auto pack_weight_tiling_params = get_default_pack_weight_data_tiling_params(
      ukernel_config, n, /*target_panels_per_thread=*/1);

  auto packed_weight_data_size =
      get_packed_weight_data_size(ukernel_config, n, k, group_size);
  at::Tensor packed_weights =
      torch::empty({packed_weight_data_size}, torch::kInt8);
  pack_weight_data_operator(
      ukernel_config,
      pack_weight_tiling_params,
      packed_weights.data_ptr<int8_t>(),
      n,
      k,
      group_size,
      weight_qvals.const_data_ptr<int8_t>(),
      weight_scales.const_data_ptr<float>(),
      weight_zeros.const_data_ptr<int8_t>());

  return packed_weights;
}

template <int weight_nbit>
at::Tensor pack_weights_without_zeros_meta(
    const at::Tensor& weight_qvals,
    const at::Tensor& weight_scales,
    // TODO(T200095131): convert to int64_t when supported by AOTI
    // group_size is a meta tensor with size (group_size)
    const at::Tensor& group_size_tensor) {
  int64_t group_size = group_size_tensor.size(0);

  int n = weight_qvals.size(0);
  int k = weight_qvals.size(1);

  using namespace torchao::operators::cpu::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  auto ukernel_config = get_ukernel_config<
      weight_nbit,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/>();

  auto packed_weight_data_size =
      get_packed_weight_data_size(ukernel_config, n, k, group_size);
  return torch::empty({packed_weight_data_size}).to("meta");
}

template <int weight_nbit>
at::Tensor pack_weights_with_zeros_meta(
    const at::Tensor& weight_qvals,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_zeros,
    // TODO(T200095131): convert to int64_t when supported by AOTI
    // group_size is a meta tensor with size (group_size)
    const at::Tensor& group_size_tensor) {
  int64_t group_size = group_size_tensor.size(0);

  int n = weight_qvals.size(0);
  int k = weight_qvals.size(1);

  using namespace torchao::operators::cpu::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  auto ukernel_config = get_ukernel_config<
      weight_nbit,
      true /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/>();

  auto packed_weight_data_size =
      get_packed_weight_data_size(ukernel_config, n, k, group_size);
  return torch::empty({packed_weight_data_size}).to("meta");
}

template <int weight_nbit, bool has_weight_zeros>
at::Tensor linear_cpu(
    const at::Tensor& packed_weights,
    // TODO(T200095131): convert n_tensor, k_tensor, group_size_tensor to
    // int64_t when supported by AOTI Currently they are meta tensors with size
    // equal to the int they wrap
    const at::Tensor& n_tensor,
    const at::Tensor& k_tensor,
    const at::Tensor& group_size_tensor,
    const at::Tensor& activations) {
  int n = n_tensor.size(0);
  int k = k_tensor.size(0);
  int group_size = group_size_tensor.size(0);

  TORCH_CHECK(
      activations.dtype() == torch::kFloat32, "activations must be float32");
  TORCH_CHECK(activations.dim() == 2, "activations must be 2D");
  int m = activations.size(0);
  int k_ = activations.size(1);
  TORCH_CHECK(k == k_, "activation shape is incompatible with packed weights.");

  using namespace torchao::operators::cpu::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

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

  at::Tensor output_tensor = torch::empty({m, n}, torch::kFloat32);
  linear_operator(
      ukernel_config,
      linear_tiling_params,
      linear_scheduling_policy,
      activation_data_buffer.data(),
      output_tensor.data_ptr<float>(),
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

  return output_tensor;
}

template <int weight_nbit, bool has_weight_zeros>
at::Tensor linear_meta(
    const at::Tensor& packed_weights,
    // TODO(T200095131): convert n_tensor, k_tensor, group_size_tensor to
    // int64_t when supported by AOTI
    // Currently they are meta tensors with size equal to the int they wrap
    const at::Tensor& n_tensor,
    const at::Tensor& k_tensor,
    const at::Tensor& group_size_tensor,
    const at::Tensor& activations) {
  int n = n_tensor.size(0);
  int k = k_tensor.size(0);

  int m = activations.size(0);
  int k_ = activations.size(1);
  TORCH_CHECK(k == k_, "activation shape is incompatible with packed weights.");
  return torch::empty({m, n}).to("meta");
}

TORCH_LIBRARY(torchao, m) {
  // Pack weights without zeros
  m.def(
      "_pack_weights_a8sz_w2s(Tensor weight_qvals, Tensor weight_scales, Tensor group_size) -> Tensor");
  m.def(
      "_pack_weights_a8sz_w3s(Tensor weight_qvals, Tensor weight_scales, Tensor group_size) -> Tensor");
  m.def(
      "_pack_weights_a8sz_w4s(Tensor weight_qvals, Tensor weight_scales, Tensor group_size) -> Tensor");
  m.def(
      "_pack_weights_a8sz_w5s(Tensor weight_qvals, Tensor weight_scales, Tensor group_size) -> Tensor");
  // Pack weights with zeros
  m.def(
      "_pack_weights_a8sz_w2sz(Tensor weight_qvals, Tensor weight_scales, Tensor weight_zeros, Tensor group_size) -> Tensor");
  m.def(
      "_pack_weights_a8sz_w3sz(Tensor weight_qvals, Tensor weight_scales, Tensor weight_zeros, Tensor group_size) -> Tensor");
  m.def(
      "_pack_weights_a8sz_w4sz(Tensor weight_qvals, Tensor weight_scales, Tensor weight_zeros, Tensor group_size) -> Tensor");
  m.def(
      "_pack_weights_a8sz_w5sz(Tensor weight_qvals, Tensor weight_scales, Tensor weight_zeros, Tensor group_size) -> Tensor");
  // Linear weights without zeros
  m.def(
      "_linear_a8sz_w2s(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");
  m.def(
      "_linear_a8sz_w3s(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");
  m.def(
      "_linear_a8sz_w4s(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");
  m.def(
      "_linear_a8sz_w5s(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");
  // Linear weights with zeros
  m.def(
      "_linear_a8sz_w2sz(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");
  m.def(
      "_linear_a8sz_w3sz(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");
  m.def(
      "_linear_a8sz_w4sz(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");
  m.def(
      "_linear_a8sz_w5sz(Tensor packed_weights, Tensor n, Tensor k, Tensor group_size, Tensor activations) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("_pack_weights_a8sz_w2s", &pack_weights_without_zeros_cpu<2>);
  m.impl("_pack_weights_a8sz_w3s", &pack_weights_without_zeros_cpu<3>);
  m.impl("_pack_weights_a8sz_w4s", &pack_weights_without_zeros_cpu<4>);
  m.impl("_pack_weights_a8sz_w5s", &pack_weights_without_zeros_cpu<5>);
  m.impl("_pack_weights_a8sz_w2sz", &pack_weights_with_zeros_cpu<2>);
  m.impl("_pack_weights_a8sz_w3sz", &pack_weights_with_zeros_cpu<3>);
  m.impl("_pack_weights_a8sz_w4sz", &pack_weights_with_zeros_cpu<4>);
  m.impl("_pack_weights_a8sz_w5sz", &pack_weights_with_zeros_cpu<5>);
  m.impl("_linear_a8sz_w2s", &linear_cpu<2, false>);
  m.impl("_linear_a8sz_w3s", &linear_cpu<3, false>);
  m.impl("_linear_a8sz_w4s", &linear_cpu<4, false>);
  m.impl("_linear_a8sz_w5s", &linear_cpu<5, false>);
  m.impl("_linear_a8sz_w2sz", &linear_cpu<2, true>);
  m.impl("_linear_a8sz_w3sz", &linear_cpu<3, true>);
  m.impl("_linear_a8sz_w4sz", &linear_cpu<4, true>);
  m.impl("_linear_a8sz_w5sz", &linear_cpu<5, true>);
}

TORCH_LIBRARY_IMPL(torchao, Meta, m) {
  m.impl("_pack_weights_a8sz_w2s", &pack_weights_without_zeros_meta<2>);
  m.impl("_pack_weights_a8sz_w3s", &pack_weights_without_zeros_meta<3>);
  m.impl("_pack_weights_a8sz_w4s", &pack_weights_without_zeros_meta<4>);
  m.impl("_pack_weights_a8sz_w5s", &pack_weights_without_zeros_meta<5>);
  m.impl("_pack_weights_a8sz_w2sz", &pack_weights_with_zeros_meta<2>);
  m.impl("_pack_weights_a8sz_w3sz", &pack_weights_with_zeros_meta<3>);
  m.impl("_pack_weights_a8sz_w4sz", &pack_weights_with_zeros_meta<4>);
  m.impl("_pack_weights_a8sz_w5sz", &pack_weights_with_zeros_meta<5>);
  m.impl("_linear_a8sz_w2s", &linear_meta<2, false>);
  m.impl("_linear_a8sz_w3s", &linear_meta<3, false>);
  m.impl("_linear_a8sz_w4s", &linear_meta<4, false>);
  m.impl("_linear_a8sz_w5s", &linear_meta<5, false>);
  m.impl("_linear_a8sz_w2sz", &linear_meta<2, true>);
  m.impl("_linear_a8sz_w3sz", &linear_meta<3, true>);
  m.impl("_linear_a8sz_w4sz", &linear_meta<4, true>);
  m.impl("_linear_a8sz_w5sz", &linear_meta<5, true>);
}
