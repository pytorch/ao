// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <torchao/experimental/ops/memory.h>
#include <torchao/experimental/ops/parallel.h>
#include <vector>

using namespace torchao::ops::linear_8bit_act_xbit_weight;

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

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
static void linear_8bit_act_xbit_weight(benchmark::State& state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int group_size = state.range(3);
  int num_threads = state.range(4);

  // OMP appears to cache when repeating the same task in the benchmark
  // To prevent this, we benchmark a number of tasks
  int num_test_cases = state.range(5);

  // Initialize config and tiling params
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
  size_t preferred_packed_weight_data_alignment =
      get_preferred_packed_weight_data_alignment(ukernel_config);

  std::vector<std::unique_ptr<char[], void (*)(void*)>> packed_weight_data;
  for (int i = 0; i < test_cases.size(); i++) {
    packed_weight_data.emplace_back(torchao::make_aligned_byte_ptr(
        preferred_packed_weight_data_alignment, packed_weight_data_size));
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
  size_t preferred_activation_data_buffer_alignment =
      get_preferred_activation_data_buffer_alignment(ukernel_config);

  auto activation_data_buffer = torchao::make_aligned_byte_ptr(
      preferred_activation_data_buffer_alignment, activation_data_buffer_size);

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

#define BENCHMARK_LINEAR_8BIT_ACT_XBIT_WEIGHT(weight_nbit) \
  BENCHMARK(linear_8bit_act_xbit_weight<                   \
                weight_nbit,                 \
                false /*has_weight_zeros*/,  \
                false /*has_bias*/,          \
                false /*has_clamp*/>)        \
      ->ArgsProduct(BENCHMARK_PARAMS)        \
      ->ArgNames(                            \
          {"m", "n", "k", "group_size", "num_threads", "num_test_cases"});

BENCHMARK_LINEAR_8BIT_ACT_XBIT_WEIGHT(2);
BENCHMARK_LINEAR_8BIT_ACT_XBIT_WEIGHT(3);
BENCHMARK_LINEAR_8BIT_ACT_XBIT_WEIGHT(4);
BENCHMARK_LINEAR_8BIT_ACT_XBIT_WEIGHT(5);

// Run the benchmark
BENCHMARK_MAIN();
