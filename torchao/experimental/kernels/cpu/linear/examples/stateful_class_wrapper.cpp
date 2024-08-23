// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/kernels/cpu/linear/examples/Channelwise8BitActivationGroupwiseLowbitWeightLinearOperator.h>
#include <torchao/experimental/kernels/cpu/parallel.h>
#include <iostream>
#include <vector>

// This file contains an example of wrapping the torchao weight packing and
// linear operators into one stateful LinearOperator class. Each surface
// (PyTorch custom class, PyTorch operator, ExecuTorch operator, ExecuTorch
// delegate) will need to write its own wrapper.  In the example here,
// std::vector is used for storage, but in PyTorch a PyTorch Tensor would be
// used and in ExecuTorch, an ExecuTorch Tensor would be used.
//
// Although more efficient, not all surfaces support stateful operators.  See
// examples/separate_function_wrappers.cpp for an example of how to split the
// operations into two steps.

using namespace torchao::operators::cpu::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight;

int main() {
  int m = 13;
  int n = 4096 + 1;
  int k = 4096;
  int group_size = 16;

  constexpr int weight_nbit = 4;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;

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

  torchao::set_num_threads(8);
  std::cout << "Using " << torchao::get_num_threads() << " threads."
            << std::endl;

  std::cout << "Initializing linear_operator." << std::endl;
  auto ukernel_config =
      get_ukernel_config<weight_nbit, has_weight_zeros, has_bias, has_clamp>();
  auto linear_operator =
      Channelwise8BitActivationGroupwiseLowbitWeightLinearOperator(
          ukernel_config,
          n,
          k,
          group_size,
          test_case.weight_qvals.data(),
          test_case.weight_scales.data(),
          test_case.weight_zeros.data(),
          // m may be resized during call to support dynamic shapes
          /*initial_m=*/1);

  linear_operator.initialize();

  std::cout << "Calling linear_operator." << std::endl;
  auto output = std::vector<float>(m * n);
  linear_operator(
      output.data(),
      test_case.activations.data(),
      m,
      k,
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
      break;
    }
  }
  if (passed) {
    std::cout << "Test passed." << std::endl;
  } else {
    std::cout << "Test failed." << std::endl;
  }

  return 0;
}
