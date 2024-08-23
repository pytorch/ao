// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/kernels/cpu/linear/channelwise_8bit_activation_groupwise_lowbit_weight.h>
#include <torchao/experimental/kernels/cpu/memory.h>
#include <vector>

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
static void channelwise_8bit_activation_groupwise_lowbit_weight(
    benchmark::State& state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int group_size = state.range(3);
  int num_threads = state.range(4);

  // OMP appears to cache when repeating the same task in the benchmark
  // To prevent this, we benchmark a number of tasks
  int num_test_cases = state.range(5);

  // Initialize config and tiling params
  using namespace torchao::operators::cpu::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  auto ukernel_config =
      get_ukernel_config<weight_nbit, has_weight_zeros, has_bias, has_clamp>();
  auto pack_weight_data_tiling_params =
      get_default_pack_weight_data_tiling_params(ukernel_config, n);
  auto linear_tiling_params =
      get_default_linear_tiling_params(ukernel_config, m, n);
  auto linear_scheduling_policy =
      LinearTileSchedulingPolicy::single_mc_parallel_nc;

  // Set number of threads
  torchao::set_num_threads(num_threads);
  assert(num_threads == torchao::get_num_threads());

  // Generate test cases
  std::vector<
      torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case>
      test_cases;
  for (int i = 0; i < num_test_cases; ++i) {
    test_cases.emplace_back(
        torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case::
            generate(
                m,
                k,
                n,
                group_size,
                weight_nbit,
                has_weight_zeros,
                has_bias,
                has_clamp));
  }

  // Pack test case weights
  size_t packed_weight_data_size =
      get_packed_weight_data_size(ukernel_config, n, k, group_size);
  size_t packed_weight_data_alignment =
      get_packed_weight_data_alignment(ukernel_config);

  std::vector<std::unique_ptr<char[], void (*)(void*)>> packed_weight_data;
  for (int i = 0; i < test_cases.size(); i++) {
    packed_weight_data.emplace_back(torchao::make_aligned_byte_array_unique_ptr(
        packed_weight_data_alignment, packed_weight_data_size));
    pack_weight_data_operator(
        ukernel_config,
        pack_weight_data_tiling_params,
        packed_weight_data[i].get(),
        n,
        k,
        group_size,
        test_cases[i].weight_qvals.data(),
        test_cases[i].weight_scales.data(),
        test_cases[i].weight_zeros.data());
  }

  // Allocate activation data buffer for test cases
  size_t activation_data_buffer_size = get_activation_data_buffer_size(
      ukernel_config,
      linear_tiling_params,
      linear_scheduling_policy,
      m,
      k,
      group_size);
  size_t activation_data_buffer_alignment =
      get_activation_data_buffer_alignment(ukernel_config);

  auto activation_data_buffer = torchao::make_aligned_byte_array_unique_ptr(
      activation_data_buffer_alignment, activation_data_buffer_size);

  auto output = std::vector<float>(m * n);
  for (auto _ : state) {
    for (int i = 0; i < test_cases.size(); i++) {
      linear_operator(
          ukernel_config,
          linear_tiling_params,
          linear_scheduling_policy,
          activation_data_buffer.get(),
          output.data(),
          m,
          n,
          k,
          group_size,
          packed_weight_data[i].get(),
          test_cases[i].activations.data(),
          test_cases[i].bias.data(),
          test_cases[i].clamp_min,
          test_cases[i].clamp_max);
    }
  }
}

#define BENCHMARK_PARAMS                                                 \
  {                                                                      \
    /*m*/ {1}, /*n*/ {4096}, /*k*/ {4096}, /*group_size*/ {16, 32, 256}, \
        /*num_threads*/ {1, 2, 4, 6, 8}, /*num_test_cases*/ {            \
      10                                                                 \
    }                                                                    \
  }

#define BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT( \
    weight_nbit)                                                       \
  BENCHMARK(channelwise_8bit_activation_groupwise_lowbit_weight<       \
                weight_nbit,                                           \
                false /*has_weight_zeros*/,                            \
                false /*has_bias*/,                                    \
                false /*has_clamp*/>)                                  \
      ->ArgsProduct(BENCHMARK_PARAMS)                                  \
      ->ArgNames(                                                      \
          {"m", "n", "k", "group_size", "num_threads", "num_test_cases"});

BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT(3);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT(4);

// Run the benchmark
BENCHMARK_MAIN();
