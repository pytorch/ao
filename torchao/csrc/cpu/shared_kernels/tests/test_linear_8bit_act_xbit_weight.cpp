// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
// TODO: move test_utils.h out of aarch64
#if defined(TORCHAO_BUILD_CPU_AARCH64)
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/channelwise_8bit_activation_groupwise_lowbit_weight.h>
#endif // TORCHAO_BUILD_CPU_AARCH64
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/tests/test_utils.h>
#include <torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <torchao/csrc/cpu/shared_kernels/internal/memory.h>
#include <torchao/csrc/cpu/shared_kernels/internal/parallel.h>

#if defined(TORCHAO_ENABLE_KLEIDI)
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp_qsi4c32p.h>
using namespace torchao::kernels::cpu::aarch64::kleidi::
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p;
#endif // TORCHAO_ENABLE_KLEIDI

const float kTol = 1.0e-5;
const float kTolKleidiAI = 5.0e-2;

using namespace torchao::ops::linear_8bit_act_xbit_weight;

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp, bool has_lut = false>
UKernelConfig get_ukernel_config() {
  namespace kernel = torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight;

  int preferred_alignment = 16;
  int n_step = 8;
  constexpr int nr = 8;
  constexpr int kr = 16;
  constexpr int sr = 2;
  constexpr int mr = 1;
  int m_step = 1;

  auto uk = UKernelConfig::make(
      preferred_alignment,
      n_step,
      nr,
      kr,
      sr,
      weight_nbit,
      has_weight_zeros,
      has_bias,
      &kernel::packed_weights_size,
      &kernel::packed_weights_offset,
      &kernel::pack_weights<weight_nbit, nr, kr, sr>,
      /*linear_configs*/ {});

  uk.linear_configs[0] = UKernelConfig::linear_config_type{
      m_step,
      mr,
      &kernel::packed_activations_size,
      &kernel::packed_activations_offset,
      &kernel::pack_activations<mr, kr, sr>,
      &kernel::
          kernel_1x8x16_f32_neondot<weight_nbit, has_weight_zeros, has_lut>};

  if constexpr (has_lut) {
    uk.packed_weights_size = &kernel::packed_weights_with_lut_size;
    uk.packed_weights_offset = &kernel::packed_weights_with_lut_offset;
    uk.pack_weights = nullptr;
    uk.pack_weights_with_lut = &kernel::pack_weights_with_lut<weight_nbit, nr, kr, sr>;
  }

  return uk;
}

template <
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    bool has_clamp,
    bool has_kleidi = false,
    bool has_lut = false>
void test_linear_8bit_act_xbit_weight(
    int m,
    int n,
    int k,
    int group_size,
    const UKernelConfig* ukernel_config_arg = nullptr) {
  UKernelConfig ukernel_config;
  if (ukernel_config_arg != nullptr) {
    ukernel_config = *ukernel_config_arg;
  } else {
    ukernel_config = get_ukernel_config<
        weight_nbit,
        has_weight_zeros,
        has_bias,
        has_clamp,
        has_lut>();
  }

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

  for (auto num_threads : {1, 4, 500}) {
    torchao::parallel::set_num_threads_in_test_dummy(num_threads);
    EXPECT_EQ(torchao::get_num_threads(), num_threads);

    // Pack weights
    auto packed_weight_data_size = ukernel_config.packed_weights_size(
        n,
        k,
        group_size,
        weight_nbit,
        has_weight_zeros,
        has_bias,
        ukernel_config.nr,
        ukernel_config.kr,
        ukernel_config.sr);
    auto preferred_packed_weight_data_alignment =
        ukernel_config.preferred_alignment;
    auto packed_weights = torchao::make_aligned_byte_ptr(
        preferred_packed_weight_data_alignment, packed_weight_data_size);

    int8_t* weight_zeros_ptr = nullptr;
    if (has_weight_zeros) {
      weight_zeros_ptr = test_case.weight_zeros.data();
    }
    float* bias_ptr = nullptr;
    // kleidi always has bias in these tests
    if (has_bias || has_kleidi) {
      bias_ptr = test_case.bias.data();
    }

    if constexpr (has_lut) {
      // Define equivalent LUT for affine quantization
      constexpr int lut_size = (1 << weight_nbit);
      std::vector<int8_t> weight_qval_idxs(test_case.weight_qvals.size());
      std::vector<int8_t> lut(lut_size, 0);
      constexpr int offset = (1 << (weight_nbit - 1));
      for (int i = 0; i < test_case.weight_qvals.size(); i++) {
        weight_qval_idxs[i] = test_case.weight_qvals[i] + offset;
      }
      for (int i = 0; i < lut_size; i++) {
        lut[i] = i - offset;
      }
      pack_weights_with_lut_operator(
          ukernel_config,
          packed_weights.get(),
          n,
          k,
          group_size,
          weight_qval_idxs.data(),
          /*n_luts*/ 1,
          lut.data(),
          test_case.weight_scales.data(),
          weight_zeros_ptr,
          bias_ptr);
    } else {
    pack_weights_operator(
        ukernel_config,
        packed_weights.get(),
        n,
        k,
        group_size,
        test_case.weight_qvals.data(),
        test_case.weight_scales.data(),
        weight_zeros_ptr,
        bias_ptr);
    }

    linear_operator(
        ukernel_config,
        std::nullopt,
        output.data(),
        m,
        n,
        k,
        group_size,
        packed_weights.get(),
        test_case.activations.data(),
        has_clamp,
        test_case.clamp_min,
        test_case.clamp_max);

    // Test correctness
    float tol = kTol;
    if (has_kleidi) {
      tol = kTolKleidiAI;
    }
    for (int i = 0; i < m * n; i++) {
      EXPECT_NEAR(output[i], test_case.expected_output[i], tol);
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

template <typename kernel_struct>
UKernelConfig get_ukernel_config_kleidi_impl() {
  namespace op = torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p;

  auto uk = kernel_struct::get_ukernel();
  auto ukernel_config = UKernelConfig::make(
      op::get_preferred_alignement(),
      uk.get_n_step(),
      uk.get_nr(),
      uk.get_kr(),
      uk.get_sr(),
      /*weight_nbit*/ 4,
      /*has_weight_zeros*/ false,
      /*has_bias*/ true,
      &op::packed_weights_size,
      &op::packed_weights_offset,
      &op::pack_weights,
      /*linear_configs*/ {});

  ukernel_config.linear_configs[0] = UKernelConfig::linear_config_type{
      static_cast<int>(uk.get_m_step()),
      static_cast<int>(uk.get_mr()),
      &op::packed_activations_size,
      &op::packed_activations_offset,
      &op::pack_activations,
      &kernel_struct::kernel};

  return ukernel_config;
}

template <typename kleidiai_kernel_struct>
void test_linear_8bit_act_xbit_weight_kleidiai() {
  constexpr int weight_nbit = 4;
  constexpr bool has_kleidi = true;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = true;
  auto uk = get_ukernel_config_kleidi_impl<kleidiai_kernel_struct>();

  for (auto m : {1, 3, 4, 8, 9, 13, 21, 43, 101}) {
    for (auto n :
         {1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          4 * 13,
          4 * 13 + 3,
          8 * 13,
          8 * 13 + 3,
          16 * 13,
          16 * 13 + 3}) {
      for (auto k : {32, 64, 128}) {
        int group_size = 32;
        test_linear_8bit_act_xbit_weight<
            weight_nbit,
            has_weight_zeros,
            has_bias,
            /*has_clamp*/ true,
            has_kleidi>(m, n, k, group_size, &uk);
        test_linear_8bit_act_xbit_weight<
            weight_nbit,
            has_weight_zeros,
            has_bias,
            /*has_clamp*/ false,
            has_kleidi>(m, n, k, group_size, &uk);

        if (k >= 64) {
          group_size = 64;
          test_linear_8bit_act_xbit_weight<
              weight_nbit,
              has_weight_zeros,
              has_bias,
              /*has_clamp*/ true,
              has_kleidi>(m, n, k, group_size, &uk);
        }
      }
    }
  }
}

#if defined(TORCHAO_ENABLE_ARM_NEON_DOT)
TEST(
    test_linear_8bit_act_xbit_weight,
    matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod) {
  test_linear_8bit_act_xbit_weight_kleidiai<
      matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod>();
}
TEST(
    test_linear_8bit_act_xbit_weight,
    matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod) {
  test_linear_8bit_act_xbit_weight_kleidiai<
      matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod>();
}
TEST(
    test_linear_8bit_act_xbit_weight,
    matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod) {
  test_linear_8bit_act_xbit_weight_kleidiai<
      matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod>();
}
TEST(
    test_linear_8bit_act_xbit_weight,
    matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod) {
  test_linear_8bit_act_xbit_weight_kleidiai<
      matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod>();
}
#endif // TORCHAO_ENABLE_ARM_NEON_DOT

template <kai_kernel_id kernel_id>
UKernelConfig get_ukernel_config_kleidi() {
#if defined(TORCHAO_ENABLE_ARM_I8MM)
  if constexpr (kernel_id == i8mm_4x8x32) {
    return get_ukernel_config_kleidi_impl<
        matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm>();
  }
  if constexpr (kernel_id == i8mm_8x4x32) {
    return get_ukernel_config_kleidi_impl<
        matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm>();
  }
#endif // TORCHAO_ENABLE_ARM_I8MM
  if constexpr (kernel_id == dotprod_1x8x32) {
    return get_ukernel_config_kleidi_impl<
        matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod>();
  }
  if constexpr (kernel_id == dotprod_1x4x32) {
    return get_ukernel_config_kleidi_impl<
        matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod>();
  }
  throw std::runtime_error("Unsupported kernel_id");
}

#endif // TORCHAO_ENABLE_KLEIDI

TEST(test_linear_8bit_act_xbit_weight_lut, Standard) {
  constexpr bool has_kleidi = false;
  constexpr bool has_lut = true;
  constexpr int weight_nbit = 3;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;
  test_linear_8bit_act_xbit_weight<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp,
      has_kleidi,
      has_lut>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight_lut, HasWeightZeros) {
  constexpr bool has_kleidi = false;
  constexpr bool has_lut = true;
  constexpr int weight_nbit = 3;
  constexpr bool has_weight_zeros = true;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = false;
  test_linear_8bit_act_xbit_weight<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp,
      has_kleidi,
      has_lut>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight_lut, HasBias) {
  constexpr bool has_kleidi = false;
  constexpr bool has_lut = true;
  constexpr int weight_nbit = 3;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = true;
  constexpr bool has_clamp = false;
  test_linear_8bit_act_xbit_weight<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp,
      has_kleidi,
      has_lut>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
}

TEST(test_linear_8bit_act_xbit_weight_lut, HasClamp) {
  constexpr bool has_kleidi = false;
  constexpr bool has_lut = true;
  constexpr int weight_nbit = 3;
  constexpr bool has_weight_zeros = false;
  constexpr bool has_bias = false;
  constexpr bool has_clamp = true;
  test_linear_8bit_act_xbit_weight<
      weight_nbit,
      has_weight_zeros,
      has_bias,
      has_clamp,
      has_kleidi,
      has_lut>(
      /*m=*/13, /*n=*/8 * 10 + 3, /*k=*/16 * 3, /*group_size=*/16);
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
  EXPECT_THROW(
      {
        pack_weights_operator(
            ukernel_config,
            /*packed_weights=*/nullptr,
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

  EXPECT_THROW(
      {
        pack_weights_operator(
            ukernel_config,
            /*packed_weights=*/nullptr,
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

/* Generated by generate_tests.py */
/* Do not modify */

#if defined(TORCHAO_ENABLE_KLEIDI)

/*****************/
// dotprod_1x4x32 tests
/*****************/

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m1xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m1xn4xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m1xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m1xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m1xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn11xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/11, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m1xn13xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/13, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m1xn51xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/51, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn111xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/111, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m1xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m1xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m1xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m2xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/2, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m2xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/2, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m3xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m4xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/4, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m3xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m31xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/31, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m32xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/32, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m33xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/33, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m34xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/34, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m35xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/35, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m7xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/7, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m17xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/17, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m23xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m41xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/41, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m7xn11xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/7, /*n=*/11, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m17xn13xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/17, /*n=*/13, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m23xn51xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/51, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m41xn111xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/41, /*n=*/111, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x4x32_m19xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/19, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m23xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m29xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/29, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x4x32_m101xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/101, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

/*****************/
// dotprod_1x8x32 tests
/*****************/

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m1xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m1xn4xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m1xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m1xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m1xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn11xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/11, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m1xn13xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/13, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m1xn51xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/51, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn111xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/111, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m1xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m1xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m1xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m2xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/2, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m2xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/2, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m3xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m4xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/4, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m3xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m31xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/31, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m32xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/32, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m33xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/33, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m34xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/34, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m35xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/35, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m7xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/7, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m17xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/17, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m23xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m41xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/41, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m7xn11xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/7, /*n=*/11, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m17xn13xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/17, /*n=*/13, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m23xn51xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/51, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m41xn111xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/41, /*n=*/111, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_dotprod_1x8x32_m19xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/19, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m23xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m29xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/29, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_dotprod_1x8x32_m101xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<dotprod_1x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/101, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

/*****************/
// i8mm_4x8x32 tests
/*****************/
#if defined(TORCHAO_ENABLE_ARM_I8MM)

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m1xn4xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m1xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn11xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/11, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn13xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/13, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m1xn51xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/51, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn111xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/111, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m1xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m1xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m1xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m2xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/2, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m2xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/2, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m3xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m4xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/4, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m3xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m31xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/31, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m32xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/32, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m33xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/33, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m34xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/34, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m35xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/35, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m7xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/7, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m17xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/17, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m23xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m41xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/41, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m7xn11xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/7, /*n=*/11, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m17xn13xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/17, /*n=*/13, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m23xn51xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/51, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m41xn111xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/41, /*n=*/111, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m19xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/19, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m23xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_4x8x32_m29xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/29, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_4x8x32_m101xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_4x8x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
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
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m1xn4xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m1xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn11xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/11, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn13xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/13, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m1xn51xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/51, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn111xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/111, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m1xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m1xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m1xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/1, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m2xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/2, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m2xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/2, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m3xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m4xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/4, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m3xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/3, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m31xn2xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/31, /*n=*/2, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m32xn4xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/32, /*n=*/4, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m33xn6xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/33, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m34xn8xk32xg32_bias_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/34, /*n=*/8, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m35xn6xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/35, /*n=*/6, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m7xn22xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/7, /*n=*/22, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m17xn26xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/17, /*n=*/26, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m23xn102xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/102, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m41xn222xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/41, /*n=*/222, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m7xn11xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/7, /*n=*/11, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m17xn13xk32xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/17, /*n=*/13, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m23xn51xk32xg32_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/51, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m41xn111xk32xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/41, /*n=*/111, /*k=*/32, /*group_size=*/32, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m19xn14xk64xg32) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/19, /*n=*/14, /*k=*/64, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m23xn22xk128xg32_bias) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      true /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/23, /*n=*/22, /*k=*/128, /*group_size=*/32, &ukernel_config);
}

TEST(
    test_linear_8bit_act_xbit_weight,
    Kleidi_i8mm_8x4x32_m29xn26xk64xg64_clamp) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      true /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/29, /*n=*/26, /*k=*/64, /*group_size=*/64, &ukernel_config);
}

TEST(test_linear_8bit_act_xbit_weight, Kleidi_i8mm_8x4x32_m101xn34xk128xg64) {
  UKernelConfig ukernel_config = get_ukernel_config_kleidi<i8mm_8x4x32>();
  test_linear_8bit_act_xbit_weight<
      4 /*weight_nbit*/,
      false /*has_weight_zeros*/,
      false /*has_bias*/,
      false /*has_clamp*/,
      true /*has_kleidi*/>(
      /*m=*/101, /*n=*/34, /*k=*/128, /*group_size=*/64, &ukernel_config);
}

#endif // TORCHAO_ENABLE_ARM_I8MM

#endif // TORCHAO_ENABLE_KLEIDI
