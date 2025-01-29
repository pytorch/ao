// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <torchao/experimental/ops/memory.h>
#include <torchao/experimental/ops/parallel.h>
#include <iostream>
// This file contains an example of wrapping the torchao weight packing and
// linear operators into two operators: one for weight packing and another
// for running the linear operator.  Each surface (PyTorch custom class, PyTorch
// operator, ExecuTorch operator, ExecuTorch delegate) will need to write its
// own wrapper).  In the example here, std::vector is used for storage, but in
// PyTorch a PyTorch Tensor would be used and in ExecuTorch, an ExecuTorch
// Tensor would be used.
//
// It is more efficient to combine weight-packing and the linear operator into
// one stateful class, but not all surfaces support this (see
// examples/stateful_class_wrapper.cpp for an example of this).

namespace torchao::ops::linear_8bit_act_xbit_weight {

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
UKernelConfig get_ukernel_config() {
  UKernelConfig config;

  namespace ukernel = torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot;
  config.mr = 1;
  config.nr = 8;
  config.activation_data_size_fn =
      &ukernel::activation_data_size<has_weight_zeros>;
  config.preferred_activation_data_alignment = 16; // size of neon register
  config.prepare_activation_data_fn =
      &ukernel::prepare_activation_data<has_weight_zeros>;
  config.weight_data_size_fn =
      &ukernel::weight_data_size<weight_nbit, has_weight_zeros>;
  config.preferred_weight_data_alignment = 16; // size of neon register
  config.prepare_weight_data_fn =
      &ukernel::prepare_weight_data<weight_nbit, has_weight_zeros>;
  config.kernel_fn =
      &ukernel::kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>;

  return config;
}

torchao::aligned_byte_ptr pack_weight_data_operator(
    UKernelConfig ukernel_config,
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    const int8_t* weight_zeros,
    std::optional<PackWeightDataTilingParams> tiling_params = {}) {
  PackWeightDataTilingParams tiling_params_;
  if (tiling_params.has_value()) {
    tiling_params_ = tiling_params.value();
  } else {
    tiling_params_ = get_default_pack_weight_data_tiling_params(
        ukernel_config, n, /*target_panels_per_thread=*/1);
  }

  auto packed_weight_data_size =
      get_packed_weight_data_size(ukernel_config, n, k, group_size);
  auto preferred_packed_weight_data_alignment =
      get_preferred_packed_weight_data_alignment(ukernel_config);
  auto packed_weight_data = torchao::make_aligned_byte_ptr(
      preferred_packed_weight_data_alignment, packed_weight_data_size);

  pack_weight_data_operator(
      ukernel_config,
      tiling_params_,
      packed_weight_data.get(),
      n,
      k,
      group_size,
      weight_qvals,
      weight_scales,
      weight_zeros);

  return packed_weight_data;
}

void linear_operator(
    UKernelConfig ukernel_config,
    float* output,
    int m,
    int n,
    int k,
    int group_size,
    void* packed_weight_data,
    float* activations,
    const float* bias,
    float clamp_min,
    float clamp_max,
    std::optional<LinearTilingParams> tiling_params = {},
    std::optional<LinearTileSchedulingPolicy> scheduling_policy = {}) {
  LinearTilingParams tiling_params_;
  if (tiling_params.has_value()) {
    tiling_params_ = tiling_params.value();
  } else {
    tiling_params_ = get_default_linear_tiling_params(
        ukernel_config, m, n, /*target_tiles_per_thread=*/5);
  }

  LinearTileSchedulingPolicy scheduling_policy_;
  if (scheduling_policy.has_value()) {
    scheduling_policy_ = scheduling_policy.value();
  } else {
    scheduling_policy_ = LinearTileSchedulingPolicy::single_mc_parallel_nc;
  }

  auto activation_data_buffer_size = get_activation_data_buffer_size(
      ukernel_config, tiling_params_, scheduling_policy_, m, k, group_size);
  auto activation_data_buffer_alignment =
      get_preferred_activation_data_buffer_alignment(ukernel_config);
  auto activation_data_buffer = torchao::make_aligned_byte_ptr(
      activation_data_buffer_alignment, activation_data_buffer_size);

  linear_operator(
      ukernel_config,
      tiling_params_,
      scheduling_policy_,
      activation_data_buffer.get(),
      output,
      m,
      n,
      k,
      group_size,
      packed_weight_data,
      activations,
      bias,
      clamp_min,
      clamp_max);
}

} // namespace
  // torchao::ops::linear_8bit_act_xbit_weight

int main() {
  using namespace torchao::ops::linear_8bit_act_xbit_weight;

  torchao::set_num_threads(8);
  std::cout << "Using " << torchao::get_num_threads() << " threads."
            << std::endl;

  constexpr int weight_nbit = 3;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;

  int m = 1;
  int n = 4096 + 1;
  int k = 4096;
  int group_size = 16;

  std::cout << "Generating random test case." << std::endl;
  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp);

  auto output = std::vector<float>(m * n);

  auto ukernel_config =
      get_ukernel_config<weight_nbit, has_weight_zeros, has_bias, has_clamp>();

  std::cout << "Running pack_weight_data_operator." << std::endl;
  auto packed_weight_data = pack_weight_data_operator(
      ukernel_config,
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      test_case.weight_zeros.data());

  std::cout << "Running linear_operator." << std::endl;
  linear_operator(
      ukernel_config,
      output.data(),
      m,
      n,
      k,
      group_size,
      packed_weight_data.get(),
      test_case.activations.data(),
      test_case.bias.data(),
      test_case.clamp_min,
      test_case.clamp_max);

  std::cout << "Checking results." << std::endl;

  bool passed = true;
  float tol = 0.001;
  for (int i = 0; i < output.size(); i++) {
    if (std::abs(test_case.expected_output[i] - output[i]) > tol) {
      std::cout << "Bad result at index " << i << ".";
      std::cout << " Output: " << output[i]
                << ". Expected: " << test_case.expected_output[i] << "."
                << std::endl;
      passed = false;
    }
  }
  if (passed) {
    std::cout << "Test passed." << std::endl;
  } else {
    std::cout << "Test failed." << std::endl;
  }

  return 0;
}
