// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#include <torchao/experimental/kernels/cpu/aarch64/quantization/quantize.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <vector>

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
static void
channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot(
    benchmark::State& state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int group_size = state.range(3);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot;

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

  std::vector<char> activation_data(
      activation_data_size(m, k, group_size, has_weight_zeros));
  prepare_activation_data(
      (void*)activation_data.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros);

  std::vector<char> weight_data(weight_data_size<weight_nbit>(
      n, k, group_size, has_weight_zeros, has_bias));
  int8_t* weight_zeros_ptr = nullptr;
  if (has_weight_zeros) {
    weight_zeros_ptr = test_case.weight_zeros.data();
  }
  float* bias_ptr = nullptr;
  if (has_bias) {
    bias_ptr = test_case.bias.data();
  }
  prepare_weight_data<weight_nbit>(
      (void*)weight_data.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      weight_zeros_ptr,
      bias_ptr);

  std::vector<float> output(m * k);
  for (auto _ : state) {
    kernel<weight_nbit>(
        output.data(),
        /*output_m_stride=*/n,
        m,
        n,
        k,
        group_size,
        weight_data.data(),
        activation_data.data(),
        test_case.clamp_min,
        test_case.clamp_max,
        has_weight_zeros,
        has_bias,
        has_clamp);
  }
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
static void
channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot(
    benchmark::State& state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int group_size = state.range(3);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot;

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

  std::vector<char> activation_data(
      activation_data_size(m, k, group_size, has_weight_zeros));
  prepare_activation_data(
      (void*)activation_data.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros);

  std::vector<char> weight_data(weight_data_size<weight_nbit>(
      n, k, group_size, has_weight_zeros, has_bias));
  int8_t* weight_zeros_ptr = nullptr;
  if (has_weight_zeros) {
    weight_zeros_ptr = test_case.weight_zeros.data();
  }
  float* bias_ptr = nullptr;
  if (has_bias) {
    bias_ptr = test_case.bias.data();
  }
  prepare_weight_data<weight_nbit>(
      (void*)weight_data.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      weight_zeros_ptr,
      bias_ptr);

  std::vector<float> output(m * k);
  for (auto _ : state) {
    kernel<weight_nbit>(
        output.data(),
        /*output_m_stride=*/n,
        m,
        n,
        k,
        group_size,
        weight_data.data(),
        activation_data.data(),
        test_case.clamp_min,
        test_case.clamp_max,
        has_weight_zeros,
        has_bias,
        has_clamp);
  }
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
static void
channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot(
    benchmark::State& state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int group_size = state.range(3);

  using namespace torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot;

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

  std::vector<char> activation_data(
      activation_data_size(m, k, group_size, has_weight_zeros));
  prepare_activation_data(
      (void*)activation_data.data(),
      m,
      k,
      group_size,
      test_case.activations.data(),
      has_weight_zeros);

  std::vector<char> weight_data(weight_data_size<weight_nbit>(
      n, k, group_size, has_weight_zeros, has_bias));
  int8_t* weight_zeros_ptr = nullptr;
  if (has_weight_zeros) {
    weight_zeros_ptr = test_case.weight_zeros.data();
  }
  float* bias_ptr = nullptr;
  if (has_bias) {
    bias_ptr = test_case.bias.data();
  }
  prepare_weight_data<weight_nbit>(
      (void*)weight_data.data(),
      n,
      k,
      group_size,
      test_case.weight_qvals.data(),
      test_case.weight_scales.data(),
      weight_zeros_ptr,
      bias_ptr);

  std::vector<float> output(m * k);
  for (auto _ : state) {
    kernel<weight_nbit>(
        output.data(),
        /*output_m_stride=*/n,
        m,
        n,
        k,
        group_size,
        weight_data.data(),
        activation_data.data(),
        test_case.clamp_min,
        test_case.clamp_max,
        has_weight_zeros,
        has_bias,
        has_clamp);
  }
}

#define BENCHMARK_PARAMS                                            \
  {                                                                 \
    /*m*/ {1}, /*n*/ {8}, /*k*/ {4096, 8192, 16384, 32768, 131072}, \
    /*group_size*/ {                                                \
      32, 256                                                       \
    }                                                               \
  }

#define BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x1x32_F32_NEONDOT( \
    weight_nbit)                                                                          \
  BENCHMARK(                                                                              \
      channelwise_8bit_activation_groupwise_lowbit_weight_1x1x32_f32_neondot<             \
          weight_nbit,                                                                    \
          false,                                                                          \
          false,                                                                          \
          false>)                                                                         \
      ->ArgsProduct(BENCHMARK_PARAMS)

#define BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x4x16_F32_NEONDOT( \
    weight_nbit)                                                                          \
  BENCHMARK(                                                                              \
      channelwise_8bit_activation_groupwise_lowbit_weight_1x4x16_f32_neondot<             \
          weight_nbit,                                                                    \
          false,                                                                          \
          false,                                                                          \
          false>)                                                                         \
      ->ArgsProduct(BENCHMARK_PARAMS)

#define BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x8x16_F32_NEONDOT( \
    weight_nbit)                                                                          \
  BENCHMARK(                                                                              \
      channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot<             \
          weight_nbit,                                                                    \
          false,                                                                          \
          false,                                                                          \
          false>)                                                                         \
      ->ArgsProduct(BENCHMARK_PARAMS)

BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x1x32_F32_NEONDOT(
    1);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x1x32_F32_NEONDOT(
    2);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x1x32_F32_NEONDOT(
    3);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x1x32_F32_NEONDOT(
    4);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x1x32_F32_NEONDOT(
    5);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x1x32_F32_NEONDOT(
    6);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x1x32_F32_NEONDOT(
    7);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x4x16_F32_NEONDOT(
    1);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x4x16_F32_NEONDOT(
    2);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x4x16_F32_NEONDOT(
    3);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x4x16_F32_NEONDOT(
    4);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x4x16_F32_NEONDOT(
    5);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x4x16_F32_NEONDOT(
    6);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x4x16_F32_NEONDOT(
    7);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x4x16_F32_NEONDOT(
    1);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x8x16_F32_NEONDOT(
    2);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x8x16_F32_NEONDOT(
    3);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x8x16_F32_NEONDOT(
    4);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x8x16_F32_NEONDOT(
    5);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x8x16_F32_NEONDOT(
    6);
BENCHMARK_CHANNELWISE_8BIT_ACTIVATION_GROUPWISE_LOWBIT_WEIGHT_1x8x16_F32_NEONDOT(
    7);

// Run the benchmark
BENCHMARK_MAIN();
