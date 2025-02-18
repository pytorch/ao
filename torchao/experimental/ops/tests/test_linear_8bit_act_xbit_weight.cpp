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
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#if defined(TORCHAO_ENABLE_ARM_I8MM)
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h>
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h>
#endif // TORCHAO_ENABLE_ARM_I8MM
#endif // TORCHAO_ENABLE_KLEIDI

const float kTol = 1.0e-5;

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
      &ukernel::weight_data_size<weight_nbit, has_weight_zeros, has_bias>;
  config.preferred_weight_data_alignment = 16; // size of neon register
  config.prepare_weight_data_fn =
      &ukernel::prepare_weight_data<weight_nbit, has_weight_zeros, has_bias>;
  config.kernel_fn =
      &ukernel::kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>;

  return config;
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp,
          bool has_kleidi = false>
void test_linear_8bit_act_xbit_weight(
    int m, int n, int k, int group_size,
    const UKernelConfig *ukernel_config_arg = nullptr) {
  UKernelConfig ukernel_config;
  if (ukernel_config_arg != nullptr) {
    ukernel_config = *ukernel_config_arg;
  } else {
    ukernel_config = get_ukernel_config<weight_nbit, has_weight_zeros, has_bias,
                                        has_clamp>();
  }

  auto test_case =
      torchao::channelwise_8bit_activation_groupwise_lowbit_weight_test_case::
          generate(m, k, n, group_size, weight_nbit, has_weight_zeros, has_bias,
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
          ukernel_config, pack_weight_data_tiling_params,
          packed_weight_data.get(), n, k, group_size,
          test_case.weight_qvals.data(), test_case.weight_scales.data(),
          test_case.weight_zeros.data(), test_case.bias.data());

      // Allocate activation buffer
      auto linear_tiling_params =
          get_default_linear_tiling_params(ukernel_config, m, n);

      auto activation_data_buffer_size = get_activation_data_buffer_size(
          ukernel_config, linear_tiling_params, linear_scheduling_policy, m, k,
          group_size);
      auto activation_data_buffer_alignment =
          get_preferred_activation_data_buffer_alignment(ukernel_config);
      auto activation_data_buffer = torchao::make_aligned_byte_ptr(
          activation_data_buffer_alignment, activation_data_buffer_size);

      // Run linear
      linear_operator(ukernel_config, linear_tiling_params,
                      linear_scheduling_policy, activation_data_buffer.get(),
                      output.data(), m, n, k, group_size,
                      packed_weight_data.get(), test_case.activations.data(),
                      test_case.clamp_min, test_case.clamp_max);

      // Test correctness
      for (int i = 0; i < m * n; i++) {
        EXPECT_NEAR(output[i], test_case.expected_output[i], kTol);
      }
    }
  }
}

#if defined(TORCHAO_ENABLE_KLEIDI)

enum kai_kernel_id {
  dotprod_1x4x32 = 0,
  dotprod_1x8x32,
  i8mm_4x8x32,
  i8mm_8x4x32
};

#define KAI_GEN_UKERNEL(kernel_ns)                                             \
  namespace kernel = kernel_ns;                                                \
  auto uk = kernel::get_ukernel();                                             \
  config.mr = uk.get_m_step();                                                 \
  config.nr = uk.get_n_step();                                                 \
  config.activation_data_size_fn = &kernel::activation_data_size;              \
  config.weight_data_size_fn = &kernel::weight_data_size;                      \
  config.preferred_activation_data_alignment =                                 \
      kernel::get_preferred_alignement();                                      \
  config.preferred_weight_data_alignment = kernel::get_preferred_alignement(); \
  config.prepare_activation_data_fn = &kernel::prepare_activation_data;        \
  config.prepare_weight_data_fn = &kernel::prepare_weight_data;                \
  config.kernel_fn = &kernel::kernel;

template <kai_kernel_id kernel_id> UKernelConfig get_ukernel_config_kleidi() {
  UKernelConfig config;
#if defined(TORCHAO_ENABLE_ARM_I8MM)
  if constexpr (kernel_id == i8mm_4x8x32) {
    KAI_GEN_UKERNEL(
        torchao::kernels::cpu::aarch64::kleidi::
            kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_i8mm_4x8x32);
    return config;
  }
  if constexpr (kernel_id == i8mm_8x4x32) {
    KAI_GEN_UKERNEL(
        torchao::kernels::cpu::aarch64::kleidi::
            kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_i8mm_8x4x32);
    return config;
  }
#endif // TORCHAO_ENABLE_ARM_I8MM
  if constexpr (kernel_id == dotprod_1x8x32) {
    KAI_GEN_UKERNEL(
        torchao::kernels::cpu::aarch64::kleidi::
            kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x8x32);
    return config;
  }
  KAI_GEN_UKERNEL(
      torchao::kernels::cpu::aarch64::kleidi::
          kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x4x32);
  return config;
}

#endif // TORCHAO_ENABLE_KLEIDI

TEST(test_linear_8bit_act_xbit_weight, Standard) {
  test_linear_8bit_act_xbit_weight<4 /*weight_nbit*/,
                                   false /*has_weight_zeros*/,
                                   false /*has_bias*/, false /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, HasWeightZeros) {
  test_linear_8bit_act_xbit_weight<4 /*weight_nbit*/, true /*has_weight_zeros*/,
                                   true /*has_bias*/, false /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, HasBias) {
  test_linear_8bit_act_xbit_weight<4 /*weight_nbit*/,
                                   false /*has_weight_zeros*/,
                                   true /*has_bias*/, false /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, HasClamp) {
  test_linear_8bit_act_xbit_weight<4 /*weight_nbit*/,
                                   false /*has_weight_zeros*/,
                                   false /*has_bias*/, true /*has_clamp*/>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, SmallDimension) {
  test_linear_8bit_act_xbit_weight<3 /*weight_nbit*/, true /*has_weight_zeros*/,
                                   true /*has_bias*/, true /*has_clamp*/>(
      /*m=*/1, /*n=*/1, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight, KNotDivisibleByGroupSize) {
  int n = 1;
  int k = 16 + 1;
  int group_size = 16;
  auto ukernel_config =
      get_ukernel_config<3 /*weight_nbit*/, true /*has_weight_zeros*/,
                         true /*has_bias*/, true /*has_clamp*/>();
  auto pack_weight_data_tiling_params =
      get_default_pack_weight_data_tiling_params(ukernel_config, n);

  EXPECT_THROW(
      {
        pack_weight_data_operator(
            ukernel_config, pack_weight_data_tiling_params,
            /*packed_weight_data=*/nullptr, n, k, group_size,
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

  auto ukernel_config =
      get_ukernel_config<3 /*weight_nbit*/, true /*has_weight_zeros*/,
                         true /*has_bias*/, true /*has_clamp*/>();
  auto pack_weight_data_tiling_params =
      get_default_pack_weight_data_tiling_params(ukernel_config, n);

  EXPECT_THROW(
      {
        pack_weight_data_operator(
            ukernel_config, pack_weight_data_tiling_params,
            /*packed_weight_data=*/nullptr, n, k, group_size,
            /*weight_qvals=*/nullptr,
            /*weight_scales=*/nullptr,
            /*weight_zeros=*/nullptr,
            /*bias=*/nullptr);
      },
      std::runtime_error);
}

// begin
/* Generated by generate_tests.py */
/* Do not modify */

#if defined(TORCHAO_ENABLE_KLEIDI)

/*****************/
// dotprod_1x4x32 tests
/*****************/

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m1xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m1xn4xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m1xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m1xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m1xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m1xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m1xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m2xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/2, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m2xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/2, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m3xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m4xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/4, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m3xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m31xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/31, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m32xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/32, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m33xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/33, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m34xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/34, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m35xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/35, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m7xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/7, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m17xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/17, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m23xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/23, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m41xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/41, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m19xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/19, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m23xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/23, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m29xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/29, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x4x32_m101xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/101, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

/*****************/
// dotprod_1x8x32 tests
/*****************/

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m1xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m1xn4xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m1xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m1xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m1xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m1xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m1xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m2xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/2, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m2xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/2, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m3xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m4xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/4, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m3xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m31xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/31, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m32xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/32, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m33xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/33, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m34xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/34, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m35xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/35, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m7xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/7, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m17xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/17, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m23xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/23, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m41xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/41, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m19xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/19, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m23xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/23, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m29xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/29, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_dotprod_1x8x32_m101xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/101, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

/*****************/
// i8mm_4x8x32 tests
/*****************/
#if defined(TORCHAO_ENABLE_ARM_I8MM)

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m1xn4xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m1xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m1xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m1xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m2xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/2, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m2xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/2, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m3xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m4xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/4, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m3xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m31xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/31, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m32xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/32, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m33xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/33, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m34xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/34, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m35xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/35, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m7xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/7, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m17xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/17, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m23xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/23, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m41xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/41, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m19xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/19, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m23xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/23, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_4x8x32_m29xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/29, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m101xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/101, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

#endif // TORCHAO_ENABLE_ARM_I8MM

/*****************/
// i8mm_8x4x32 tests
/*****************/
#if defined(TORCHAO_ENABLE_ARM_I8MM)

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m1xn4xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m1xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m1xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m1xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/1, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m2xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/2, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m2xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/2, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m3xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m4xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/4, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m3xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m31xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/31, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m32xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/32, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m33xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/33, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m34xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/34, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m35xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/35, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m7xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/7, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m17xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/17, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m23xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/23, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m41xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/41, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m19xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/19, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m23xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, true /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/23, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight,
     Kleidi_i8mm_8x4x32_m29xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      true /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/29, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m101xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/, false /*has_weight_zeros*/, false /*has_bias*/,
      false /*has_clamp*/, true /*has_kleidi*/>(
      /*m=*/101, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

#endif // TORCHAO_ENABLE_ARM_I8MM

#endif // TORCHAO_ENABLE_KLEIDI
