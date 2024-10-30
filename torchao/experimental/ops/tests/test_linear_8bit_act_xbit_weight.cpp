// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
// TODO: move test_utils.h out of aarch64
#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <torchao/experimental/ops/memory.h>
#include <torchao/experimental/ops/parallel.h>

#if defined(TORCHAO_ENABLE_KLEIDI)
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#endif // TORCHAO_ENABLE_KLEIDI

const float kTol = 1.0e-5;

using namespace torchao::ops::linear_8bit_act_xbit_weight;

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp, bool has_kleidi=false>
UKernelConfig get_ukernel_config() {
  UKernelConfig config;

  if constexpr (!has_kleidi) {
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
      &ukernel::weight_data_size<weight_nbit, has_weight_zeros, has_bias>;
  config.preferred_weight_data_alignment = 16; // size of neon register
  config.prepare_weight_data_fn =
      &ukernel::prepare_weight_data<weight_nbit, has_weight_zeros, has_bias>;
  config.kernel_fn =
      &ukernel::kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>;
  } else {
#if defined(TORCHAO_ENABLE_KLEIDI)
    assert (weight_nbit == 4);
    assert (!has_weight_zeros);

    namespace kernel = torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x8x32;

    auto uk = kernel::get_ukernel();
    config.mr = uk.get_mr();
    config.nr = uk.get_nr();

    config.activation_data_size_fn = &kernel::activation_data_size;
    config.weight_data_size_fn = &kernel::weight_data_size;

    config.preferred_activation_data_alignment = kernel::get_preferred_alignement();
    config.preferred_weight_data_alignment = kernel::get_preferred_alignement();

    config.prepare_activation_data_fn = &kernel::prepare_activation_data;
    config.prepare_weight_data_fn = &kernel::prepare_weight_data; 

    config.kernel_fn = &kernel::kernel;
#else
    assert (false);
#endif // TORCHAO_ENABLE_KLEIDI
  }

  return config;
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp, bool has_kleidi=false>
void test_linear_8bit_act_xbit_weight(int m, int n, int k, int group_size) {
  auto ukernel_config = 
      get_ukernel_config<weight_nbit, has_weight_zeros, has_bias, has_clamp, has_kleidi>();

  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          m,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          has_clamp,
          /*round_weight_scales_to_bf16=*/has_kleidi);

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
      auto preferred_packed_weight_data_alignment =
          get_preferred_packed_weight_data_alignment(ukernel_config);
      auto packed_weight_data = torchao::make_aligned_byte_ptr(
          preferred_packed_weight_data_alignment, packed_weight_data_size);

      pack_weight_data_operator(
          ukernel_config,
          pack_weight_data_tiling_params,
          packed_weight_data.get(),
          n,
          k,
          group_size,
          test_case.weight_qvals.data(),
          test_case.weight_scales.data(),
          test_case.weight_zeros.data(),
          test_case.bias.data());

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
          get_preferred_activation_data_buffer_alignment(ukernel_config);
      auto activation_data_buffer = torchao::make_aligned_byte_ptr(
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
          test_case.clamp_min,
          test_case.clamp_max);

      // Test correctness
      for (int i = 0; i < m * n; i++) {
        EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
      }
    }
  }
}

TEST(test_linear_8bit_act_xbit_weight, Standard) {
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, HasWeightZeros) {
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      true /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, HasBias) {
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, HasClamp) {
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, SmallDimension) {
  test_linear_8bit_act_xbit_weight<
      3 /*weight_nbit*/,
      true /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/>(
      /*m=*/1, /*n=*/1, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, KNotDivisibleByGroupSize) {
  int n = 1;
  int k = 16 + 1;
  int group_size = 16;
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
            /*weight_zeros=*/nullptr,
            /*bias=*/nullptr);
      },
      std::runtime_error);
}

TEST(test_linear_8bit_act_xbit_weight, GroupSizeNotDivisibleBy16) {
  int n = 1;
  int k = 20;
  int group_size = 10;

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
            /*weight_zeros=*/nullptr,
            /*bias=*/nullptr);
      },
      std::runtime_error);
}

#if defined(TORCHAO_ENABLE_KLEIDI)
TEST(test_linear_8bit_act_xbit_weight, KleidiSmall) {
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/2, /*k=*/32, /*group_size=*/32);
}

TEST(test_linear_8bit_act_xbit_weight, KleidiStandard) {
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/13, /*n=*/20, /*k=*/32, /*group_size=*/32);
}

TEST(test_linear_8bit_act_xbit_weight, KleidiHasClamp) {
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/17, /*n=*/10, /*k=*/32 * 2, /*group_size=*/32);
}

TEST(test_linear_8bit_act_xbit_weight, KleidiHasBias) {
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/18, /*k=*/32 * 3, /*group_size=*/32);
}
#endif // TORCHAO_ENABLE_KLEIDI
