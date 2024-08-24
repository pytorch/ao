// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
// TODO: move test_utils.h out of aarch64
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/kernels/cpu/linear/channelwise_8bit_activation_groupwise_lowbit_weight.h>
#include <torchao/experimental/kernels/cpu/memory.h>
#include <torchao/experimental/kernels/cpu/parallel.h>

const float kTol = 1.0e-5;

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void test_channelwise_8bit_activation_groupwise_lowbit_weight(
    int m,
    int n,
    int k,
    int group_size) {
  using namespace torchao::operators::cpu::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;
  auto ukernel_config =
      get_ukernel_config<weight_nbit, has_weight_zeros, has_bias, has_clamp>();

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

  for (auto linear_scheduling_policy :
       {LinearTileSchedulingPolicy::single_mc_parallel_nc,
        LinearTileSchedulingPolicy::parallel_mc_parallel_nc}) {
    for (auto num_threads : {1, 4, 500}) {
      torchao::set_num_threads(num_threads);
      EXPECT_EQ(torchao::get_num_threads(), num_threads);

      // Pack weights
      auto pack_weight_data_tiling_params =
          get_default_pack_weight_data_tiling_params(ukernel_config, n);
      auto packed_weight_data_size =
          get_packed_weight_data_size(ukernel_config, n, k, group_size);
      auto packed_weight_data_alignment =
          get_packed_weight_data_alignment(ukernel_config);
      auto packed_weight_data = torchao::make_aligned_byte_array_unique_ptr(
          packed_weight_data_alignment, packed_weight_data_size);

      pack_weight_data_operator(
          ukernel_config,
          pack_weight_data_tiling_params,
          packed_weight_data.get(),
          n,
          k,
          group_size,
          test_case.weight_qvals.data(),
          test_case.weight_scales.data(),
          test_case.weight_zeros.data());

      // Allocate activation buffer
      auto linear_tiling_params =
          get_default_linear_tiling_params(ukernel_config, m, n);

      auto activation_data_buffer_size = get_activation_data_buffer_size(
          ukernel_config,
          linear_tiling_params,
          linear_scheduling_policy,
          m,
          k,
          group_size);
      auto activation_data_buffer_alignment =
          get_activation_data_buffer_alignment(ukernel_config);
      auto activation_data_buffer = torchao::make_aligned_byte_array_unique_ptr(
          activation_data_buffer_alignment, activation_data_buffer_size);

      // Run linear
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
          packed_weight_data.get(),
          test_case.activations.data(),
          test_case.bias.data(),
          test_case.clamp_min,
          test_case.clamp_max);

      // Test correctness
      for (int i = 0; i < m * n; i++) {
        EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
      }
    }
  }
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, Standard) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, HasWeightZeros) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight<
      4 /*weight_nbit*/,
      true /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, HasBias) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, HasClamp) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_channelwise_8bit_activation_groupwise_lowbit_weight, SmallDimension) {
  test_channelwise_8bit_activation_groupwise_lowbit_weight<
      3 /*weight_nbit*/,
      true /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/>(
      /*m=*/1, /*n=*/1, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight,
    KNotDivisibleByGroupSize) {
  int n = 1;
  int k = 16 + 1;
  int group_size = 16;

  using namespace torchao::operators::cpu::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;
  auto ukernel_config = get_ukernel_config<
      3 /*weight_nbit*/,
      true /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/>();
  auto pack_weight_data_tiling_params =
      get_default_pack_weight_data_tiling_params(ukernel_config, n);

  EXPECT_THROW(
      {
        pack_weight_data_operator(
            ukernel_config,
            pack_weight_data_tiling_params,
            /*packed_weight_data=*/nullptr,
            n,
            k,
            group_size,
            /*weight_qvals=*/nullptr,
            /*weight_scales=*/nullptr,
            /*weight_zeros=*/nullptr);
      },
      std::runtime_error);
}

TEST(
    test_channelwise_8bit_activation_groupwise_lowbit_weight,
    GroupSizeNotDivisibleBy16) {
  int n = 1;
  int k = 20;
  int group_size = 10;

  using namespace torchao::operators::cpu::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;
  auto ukernel_config = get_ukernel_config<
      3 /*weight_nbit*/,
      true /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/>();
  auto pack_weight_data_tiling_params =
      get_default_pack_weight_data_tiling_params(ukernel_config, n);

  EXPECT_THROW(
      {
        pack_weight_data_operator(
            ukernel_config,
            pack_weight_data_tiling_params,
            /*packed_weight_data=*/nullptr,
            n,
            k,
            group_size,
            /*weight_qvals=*/nullptr,
            /*weight_scales=*/nullptr,
            /*weight_zeros=*/nullptr);
      },
      std::runtime_error);
}
