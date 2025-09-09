// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#if defined(TORCHAO_BUILD_CPU_AARCH64)
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/groupwise_lowbit_weight/groupwise_lowbit_weight_lut.h>
#endif // TORCHAO_BUILD_CPU_AARCH64
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/tests/test_utils.h>
#include <torchao/csrc/cpu/shared_kernels/groupwise_lowbit_weight_lut/groupwise_lowbit_weight_lut.h>
#include <torchao/csrc/cpu/shared_kernels/internal/memory.h>
#include <torchao/csrc/cpu/shared_kernels/internal/parallel.h>

const float kTol = 1.0e-5;
using namespace torchao::ops::groupwise_lowbit_weight_lut;

template <int weight_nbit, bool has_scales>
UKernelConfig get_ukernel_config(bool has_bias) {
  namespace kernel =
      torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut;

  int preferred_alignment = 16;
  int n_step = 8;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr int mr = 1;
  int m_step = 1;

  auto uk = UKernelConfig::make(
      preferred_alignment,
      n_step,
      nr,
      kr,
      sr,
      weight_nbit,
      has_scales,
      has_bias,
      &kernel::packed_weights_size,
      &kernel::packed_weights_offset,
      &kernel::pack_weights<weight_nbit, nr, kr, sr>,
      /*configs*/ {});

  uk.configs[0] = UKernelConfig::config_type{
      m_step,
      mr,
      &kernel::packed_activations_size,
      &kernel::packed_activations_offset,
      &kernel::pack_activations<mr, kr, sr>,
      &kernel::
          groupwise_lowbit_weight_lut_kernel_1x4x32<weight_nbit, has_scales>};
  return uk;
}

template <int weight_nbit, bool has_scales>
void test_groupwise_lowbit_weight_lut(
    int m,
    int k,
    int n,
    int scale_group_size,
    int lut_group_size,
    bool has_bias,
    bool has_clamp,
    const UKernelConfig* ukernel_config_arg = nullptr) {
  UKernelConfig ukernel_config;
  if (ukernel_config_arg != nullptr) {
    ukernel_config = *ukernel_config_arg;
  } else {
    ukernel_config = get_ukernel_config<weight_nbit, has_scales>(has_bias);
  }

  auto test_case = torchao::groupwise_lowbit_weight_lut_test_case::
      generate_with_decoupled_grouping(
          m,
          k,
          n,
          scale_group_size,
          lut_group_size,
          weight_nbit,
          has_scales,
          has_bias,
          has_clamp);

  auto output = std::vector<float>(m * n);

  for (auto num_threads : {1, 4, 500}) {
    torchao::parallel::set_num_threads_in_test_dummy(num_threads);
    EXPECT_EQ(torchao::get_num_threads(), num_threads);
    auto packed_weight_data_size = ukernel_config.packed_weights_size(
        n,
        k,
        weight_nbit,
        scale_group_size,
        has_scales,
        has_bias,
        ukernel_config.nr,
        ukernel_config.kr,
        ukernel_config.sr);
    auto preferred_packed_weight_data_alignment =
        ukernel_config.preferred_alignment;
    auto packed_weights = torchao::make_aligned_byte_ptr(
        preferred_packed_weight_data_alignment, packed_weight_data_size);

    pack_weights_operator(
        ukernel_config,
        // Outputs
        packed_weights.get(),
        // Inputs
        n,
        k,
        scale_group_size,
        lut_group_size,
        test_case.weight_qval_indices.data(),
        test_case.weight_scales.data(),
        test_case.weight_luts.data(),
        test_case.bias.data());

    groupwise_lowbit_weight_lut_parallel_operator(
        ukernel_config,
        std::nullopt,
        output.data(),
        m,
        n,
        k,
        scale_group_size,
        lut_group_size,
        packed_weights.get(),
        test_case.activations.data(),
        has_clamp,
        test_case.clamp_min,
        test_case.clamp_max);

    float tol = kTol;
    for (int i = 0; i < m * n; i++) {
      EXPECT_NEAR(output[i], test_case.expected_output[i], tol);
    }
  }
}

struct KernelTestParams {
  int m;
  int k;
  int n;
  int scale_group_size;
  int lut_group_size;
  bool has_bias;
  bool has_clamp;
};

class ComprehensiveKernelTest
    : public ::testing::TestWithParam<KernelTestParams> {};

TEST_P(ComprehensiveKernelTest, kernel_test_has_scales_true) {
  const KernelTestParams& params = GetParam();

  constexpr bool has_scales = true;

  for (int weight_nbit : {1, 2, 3, 4}) {
    switch (weight_nbit) {
      case 1:
        test_groupwise_lowbit_weight_lut<1, has_scales>(
            params.m,
            params.k,
            params.n,
            params.scale_group_size,
            params.lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      case 2:
        test_groupwise_lowbit_weight_lut<2, has_scales>(
            params.m,
            params.k,
            params.n,
            params.scale_group_size,
            params.lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      case 3:
        test_groupwise_lowbit_weight_lut<3, has_scales>(
            params.m,
            params.k,
            params.n,
            params.scale_group_size,
            params.lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      case 4:
        test_groupwise_lowbit_weight_lut<4, has_scales>(
            params.m,
            params.k,
            params.n,
            params.scale_group_size,
            params.lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      default:
        FAIL() << "Unsupported weight_nbit value: " << weight_nbit;
    }
  }
}

TEST_P(ComprehensiveKernelTest, kernel_test_has_scales_false) {
  const KernelTestParams& params = GetParam();

  constexpr bool has_scales = false;

  for (int weight_nbit : {1, 2, 3, 4}) {
    switch (weight_nbit) {
      case 1:
        test_groupwise_lowbit_weight_lut<1, has_scales>(
            params.m,
            params.k,
            params.n,
            params.scale_group_size,
            params.lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      case 2:
        test_groupwise_lowbit_weight_lut<2, has_scales>(
            params.m,
            params.k,
            params.n,
            params.scale_group_size,
            params.lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      case 3:
        test_groupwise_lowbit_weight_lut<3, has_scales>(
            params.m,
            params.k,
            params.n,
            params.scale_group_size,
            params.lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      case 4:
        test_groupwise_lowbit_weight_lut<4, has_scales>(
            params.m,
            params.k,
            params.n,
            params.scale_group_size,
            params.lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      default:
        FAIL() << "Unsupported weight_nbit value: " << weight_nbit;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    KernelEdgeCases,
    ComprehensiveKernelTest,
    ::testing::Values(
        // Flag-specific tests
        KernelTestParams{
            8,
            64,
            16,
            32,
            256,
            /*has_bias=*/true,
            /*has_clamp=*/true},
        KernelTestParams{
            8,
            64,
            16,
            32,
            256,
            /*has_bias=*/true,
            /*has_clamp=*/false},
        KernelTestParams{
            8,
            64,
            16,
            32,
            256,
            /*has_bias=*/false,
            /*has_clamp=*/true},
        KernelTestParams{
            8,
            64,
            16,
            32,
            256,
            /*has_bias=*/false,
            /*has_clamp=*/false},

        // Prime number dimensions for m and n
        KernelTestParams{
            7,
            64,
            13,
            32,
            256,
            /*has_bias=*/true,
            /*has_clamp=*/true},
        KernelTestParams{
            13,
            128,
            17,
            64,
            512,
            /*has_bias=*/false,
            /*has_clamp=*/false},
        KernelTestParams{
            1,
            32,
            5,
            32,
            128,
            /*has_bias=*/true,
            /*has_clamp=*/false},

        // Varying Dimensions and Group Sizes
        KernelTestParams{8, 64, 16, 32, 256, true, true},
        KernelTestParams{8, 64, 12, 32, 256, true, false},
        KernelTestParams{7, 128, 24, 64, 512, false, true},
        KernelTestParams{1, 32, 4, 32, 128, true, true},

        // Unaligned M
        KernelTestParams{7, 64, 16, 32, 256, true, false},
        KernelTestParams{5, 64, 16, 32, 256, false, true},
        KernelTestParams{1, 64, 16, 32, 256, true, true}));

void PrintTo(const KernelTestParams& params, std::ostream* os) {
  *os << "KernelTestParams(m=" << params.m << ", k=" << params.k
      << ", n=" << params.n << ", scale_gs=" << params.scale_group_size
      << ", lut_gs=" << params.lut_group_size
      << ", has_bias=" << (params.has_bias ? "true" : "false")
      << ", has_clamp=" << (params.has_clamp ? "true" : "false") << ")";
}
