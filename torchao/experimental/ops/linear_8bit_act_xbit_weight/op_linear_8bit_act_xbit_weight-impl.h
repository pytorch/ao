// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/kernel_selector.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <torchao/experimental/ops/packed_weights_header.h>
#include <optional>
#include <vector>

namespace {

#ifdef USE_ATEN
template <int weight_nbit>
Tensor pack_weights_cpu(
    const Tensor& weight_qvals,
    const Tensor& weight_scales,
    const std::optional<Tensor>& weight_zeros,
    int64_t group_size,
    const std::optional<Tensor>& bias,
    const std::optional<std::string>& target) {
  bool has_bias = bias.has_value();
  bool has_weight_zeros = weight_zeros.has_value();

  TORCHAO_CHECK(
      weight_qvals.dtype() == torch::kInt8, "weight_qvals must be int8");
  TORCHAO_CHECK(weight_qvals.dim() == 2, "weight_qvals must be 2D");

  // In PyTorch, weights are nxk in row-major format (with activations being
  // right-multiplied).
  // In kernel, activations are left-multiplied by kxn transposed
  // weights in column-major format.
  // Note the underlying data is the same in both cases
  int n = weight_qvals.size(0);
  int k = weight_qvals.size(1);

  TORCHAO_CHECK(
      weight_scales.dtype() == torch::kFloat32,
      "weight_scales must be float32");
  TORCHAO_CHECK(weight_scales.dim() == 1, "weight_scales must be 1D");
  TORCHAO_CHECK(group_size >= 1, "group_size must be >= 1");
  TORCHAO_CHECK(
      weight_scales.size(0) == ((n * k) / group_size),
      "expected 1 scale per group");

  const int8_t* weight_zeros_ptr = nullptr;
  if (has_weight_zeros) {
    TORCHAO_CHECK(
        weight_zeros.value().dtype() == torch::kInt8,
        "weight_zeros must be int8");
    TORCHAO_CHECK(weight_zeros.value().dim() == 1, "weight_zeros must be 1D");
    TORCHAO_CHECK(
        weight_zeros.value().size(0) == ((n * k) / group_size),
        "expected 1 zero per group");
    weight_zeros_ptr = weight_zeros.value().const_data_ptr<int8_t>();
  }

  const float* bias_ptr = nullptr;
  if (has_bias) {
    TORCHAO_CHECK(
        bias.value().dtype() == torch::kFloat32, "bias must be float32");
    TORCHAO_CHECK(bias.value().dim() == 1, "bias must be 1D");
    TORCHAO_CHECK(bias.value().size(0) == n, "expected 1 bias per row");
    bias_ptr = bias.value().const_data_ptr<float>();
  }

  auto packed_weights_format =
      torchao::ops::linear_8bit_act_xbit_weight::select_packed_weights_format<
          weight_nbit>(target, has_weight_zeros, has_bias);
  auto packed_weights_header = packed_weights_format.to_packed_weights_header();
  auto uk = torchao::ops::linear_8bit_act_xbit_weight::select_ukernel_config<
      weight_nbit>(packed_weights_header);

  auto packed_weight_data_size = torchao::ops::PackedWeightsHeader::size() +
      uk.packed_weights_size(
          n,
          k,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          uk.nr,
          uk.kr,
          uk.sr);

  Tensor packed_weights = torch::empty(
      {static_cast<int64_t>(packed_weight_data_size)}, torch::kInt8);
  packed_weights_header.write(packed_weights.mutable_data_ptr<int8_t>());

  torchao::ops::linear_8bit_act_xbit_weight::pack_weights_operator(
      uk,
      packed_weights.mutable_data_ptr<int8_t>() +
          torchao::ops::PackedWeightsHeader::size(),
      n,
      k,
      group_size,
      weight_qvals.const_data_ptr<int8_t>(),
      weight_scales.const_data_ptr<float>(),
      weight_zeros_ptr,
      bias_ptr);

  return packed_weights;
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit>
Tensor pack_weights_meta(
    const Tensor& weight_qvals,
    const Tensor& weight_scales,
    const std::optional<Tensor>& weight_zeros,
    int64_t group_size,
    const std::optional<Tensor>& bias,
    const std::optional<std::string>& target) {
  bool has_bias = bias.has_value();
  bool has_weight_zeros = weight_zeros.has_value();

  TORCHAO_CHECK(group_size >= 1, "group_size must be >= 1");
  int n = weight_qvals.size(0);
  int k = weight_qvals.size(1);

  auto packed_weights_format =
      torchao::ops::linear_8bit_act_xbit_weight::select_packed_weights_format<
          weight_nbit>(target, has_weight_zeros, has_bias);
  auto uk = torchao::ops::linear_8bit_act_xbit_weight::select_ukernel_config<
      weight_nbit>(packed_weights_format);

  auto packed_weight_data_size = torchao::ops::PackedWeightsHeader::size() +
      uk.packed_weights_size(
          n,
          k,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          uk.nr,
          uk.kr,
          uk.sr);

  auto options =
      torch::TensorOptions().device(c10::DeviceType::Meta).dtype(torch::kInt8);
  return torch::empty({static_cast<int64_t>(packed_weight_data_size)}, options);
}
#endif // USE_ATEN

#if defined(USE_ATEN) || defined(USE_EXECUTORCH)
template <int weight_nbit>
Tensor linear_out_cpu(
    const Tensor& activations,
    const Tensor& packed_weights,
    const int64_t& group_size,
    const int64_t& n,
    const int64_t& k,
    Tensor& out) {
  TORCHAO_CHECK(n >= 1, "n must be >= 1");
  TORCHAO_CHECK(k >= 1, "k must be >= 1");
  TORCHAO_CHECK(group_size >= 1, "group_size must be >= 1");

#ifdef USE_ATEN
  TORCHAO_CHECK(
      activations.dtype() == torch::kFloat32, "activations must be float32");
#endif // USE_ATEN

  TORCHAO_CHECK(activations.dim() == 2, "activations must be 2D");
  int m = activations.size(0);
  int k_ = activations.size(1);
  TORCHAO_CHECK(
      k == k_, "activation shape is incompatible with packed weights.");

#ifdef USE_ATEN
  TORCHAO_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
#endif // USE_ATEN

  // Explicit cast from int64_t to int is required for Executorch
  TORCHAO_RESIZE_TENSOR(out, {(int)m, (int)n});

  TORCHAO_CHECK(packed_weights.dim() == 1, "packed_weights must be 1D");
#ifdef USE_ATEN
  TORCHAO_CHECK(
      packed_weights.dtype() == torch::kInt8, "packed_weights must be int8");
#endif // USE_ATEN
  TORCHAO_CHECK(
      packed_weights.size(0) >= torchao::ops::PackedWeightsHeader::size(),
      "packed_weights is not big enough to read the header.");
  auto header =
      torchao::ops::PackedWeightsHeader::read(packed_weights.const_data_ptr());

  auto uk = torchao::ops::linear_8bit_act_xbit_weight::select_ukernel_config<
      weight_nbit>(header);

  torchao::ops::linear_8bit_act_xbit_weight::linear_operator(
      uk,
      std::nullopt,
      out.mutable_data_ptr<float>(),
      m,
      n,
      k,
      group_size,
      packed_weights.const_data_ptr<int8_t>() +
          torchao::ops::PackedWeightsHeader::size(),
      activations.const_data_ptr<float>(),
      /*has_clamp=*/false,
      /*clamp_min=*/0.0,
      /*clamp_max=*/0.0);

  return out;
}
#endif // defined(USE_ATEN) || defined(USE_EXECUTORCH)

#ifdef USE_ATEN
template <int weight_nbit>
Tensor linear_cpu(
    const Tensor& activations,
    const Tensor& packed_weights,
    const int64_t& group_size,
    const int64_t& n,
    const int64_t& k) {
  Tensor output_tensor = torch::empty({}, torch::kFloat32);
  linear_out_cpu<weight_nbit>(
      activations, packed_weights, group_size, n, k, output_tensor);
  return output_tensor;
}
#endif // USE_ATEN

} // namespace
