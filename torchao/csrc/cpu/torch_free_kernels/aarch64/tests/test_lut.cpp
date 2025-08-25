// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <gtest/gtest.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/groupwise_lowbit_weight/groupwise_lowbit_weight_lut.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/lut/lut.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/tests/test_utils.h>
#include <cstring>
#include <vector>

namespace lut_utils = torchao::lut;
namespace kernel_api =
    torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut;

TEST(test_fp32_lut, LutLookup) {
  auto lut = torchao::get_random_vector(16, -1.0, 1.0);
  auto idx = torchao::get_random_lowbit_vector(16, 4);

  uint8x16_t idx_vec = vld1q_u8(idx.data());
  uint8x16x4_t lut_vec;
  torchao::lut::load_fp32_lut(lut_vec, lut.data());

  float32x4_t out0, out1, out2, out3;
  torchao::lut::lookup_from_fp32_lut(out0, out1, out2, out3, lut_vec, idx_vec);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(out0[i], lut[idx[i]]);
    EXPECT_EQ(out1[i], lut[idx[i + 4]]);
    EXPECT_EQ(out2[i], lut[idx[i + 8]]);
    EXPECT_EQ(out3[i], lut[idx[i + 12]]);
  }
}

template <
    int weight_nbit_,
    bool has_scales_,
    int mr_,
    int nr_,
    int kr_,
    int sr_>
void test_groupwise_lowbit_lut_kernel(
    int m,
    int k,
    int n,
    int flat_scale_group_size,
    int flat_lut_group_size,
    bool has_bias,
    bool has_clamp) {
  namespace kernel_api =
      torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut;
  // 1. Generate test case
  auto test_case = torchao::groupwise_lowbit_weight_lut_test_case::
      generate_with_decoupled_grouping(
          m,
          k,
          n,
          /*scale_group_size=*/flat_scale_group_size,
          /*lut_group_size=*/flat_lut_group_size,
          /*weight_nbit=*/weight_nbit_,
          /*has_scales=*/has_scales_,
          has_bias,
          has_clamp);
  // 2. Pack Activations
  const auto& source_activations = test_case.activations;
  std::vector<float> packed_activations_buffer(
      kernel_api::packed_activations_size(m, k, mr_, kr_, sr_));
  kernel_api::pack_activations<mr_, kr_, sr_>(
      packed_activations_buffer.data(),
      m,
      k,
      source_activations.data(),
      mr_,
      kr_,
      sr_);
  // 3. Pack Weights
  std::vector<char> packed_weights(kernel_api::packed_weights_size(
      n,
      k,
      weight_nbit_,
      flat_scale_group_size,
      has_scales_,
      has_bias,
      nr_,
      kr_,
      sr_));
  kernel_api::pack_weights<weight_nbit_, nr_, kr_, sr_>(
      packed_weights.data(),
      test_case.weight_qval_indices.data(),
      test_case.weight_scales.data(),
      test_case.weight_luts.data(),
      n,
      k,
      flat_scale_group_size,
      flat_lut_group_size,
      has_scales_,
      has_bias,
      test_case.bias.data(),
      nr_,
      kr_,
      sr_);

  // 4. Run the kernel
  std::vector<float> output(m * n);
  kernel_api::
      groupwise_lowbit_weight_lut_kernel_1x4x32<weight_nbit_, has_scales_>(
          output.data(),
          n,
          m,
          n,
          k,
          flat_scale_group_size,
          flat_lut_group_size,
          packed_weights.data(),
          packed_activations_buffer.data(),
          test_case.clamp_min,
          test_case.clamp_max,
          has_bias,
          has_clamp);

  //   5. Compare results
  constexpr float kTol = 1e-4;
  for (int i = 0; i < m * n; i++) {
    EXPECT_NEAR(output[i], test_case.expected_output[i], kTol)
        << "Mismatch at index " << i;
  }
}

TEST(test_groupwise_lowbit_lut_kernel, 4bit_aligned) {
  constexpr int weight_nbit_ = 4;
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr bool has_scales = true;

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/false);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
}

TEST(test_groupwise_lowbit_lut_kernel, 4bit_mismatch) {
  constexpr int weight_nbit_ = 4;
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr bool has_scales = true;

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/false);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
}

TEST(test_groupwise_lowbit_lut_kernel, 3bit_mismatch) {
  constexpr int weight_nbit_ = 3;
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr bool has_scales = true;

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/false);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/512, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
}

TEST(test_groupwise_lowbit_lut_kernel, 2bit_mismatch) {
  constexpr int weight_nbit_ = 2;
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr bool has_scales = true;

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/true,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/false,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/true,
      /*has_clamp=*/false);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/true,
      /*has_clamp=*/true);
}

TEST(test_groupwise_lowbit_lut_kernel, 1bit_mismatch) {
  constexpr int weight_nbit_ = 1;
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr bool has_scales = true;

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/true,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/false,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/true,
      /*has_clamp=*/false);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32,
      /*flat_lut_group_size=*/512,
      /*has_bias=*/true,
      /*has_clamp=*/true);
}

TEST(test_groupwise_lowbit_lut_kernel, 3bit_aligned) {
  constexpr int weight_nbit_ = 3;
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr bool has_scales = true;

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/false);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
}

TEST(test_groupwise_lowbit_lut_kernel, 2bit_aligned) {
  constexpr int weight_nbit_ = 2;
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr bool has_scales = true;

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/false);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
}

TEST(test_groupwise_lowbit_lut_kernel, 1bit_aligned) {
  constexpr int weight_nbit_ = 1;
  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr bool has_scales = true;

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/false);

  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/false,
      /*has_clamp=*/true);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/false);
  test_groupwise_lowbit_lut_kernel<weight_nbit_, has_scales, mr, nr, kr, sr>(
      /*m=*/8,
      /*k=*/64,
      /*n=*/16,
      /*flat_scale_group_size=*/32, // Must be multiple of k*NR = 256
      /*flat_lut_group_size=*/256, // Must be multiple of k*NR = 256
      /*has_bias=*/true,
      /*has_clamp=*/true);
}

struct KernelTestParams {
  int m;
  int k;
  int n;
  int flat_scale_group_size;
  int flat_lut_group_size;
  bool has_bias;
  bool has_clamp;
};

class ComprehensiveKernelTest
    : public ::testing::TestWithParam<KernelTestParams> {};

TEST_P(ComprehensiveKernelTest, kernel_test) {
  const KernelTestParams& params = GetParam();

  constexpr int mr = 1;
  constexpr int nr = 4;
  constexpr int kr = 32;
  constexpr int sr = 8;
  constexpr bool has_scales = true;

  for (int weight_nbit : {1, 2, 3, 4}) {
    switch (weight_nbit) {
      case 1:
        test_groupwise_lowbit_lut_kernel<1, has_scales, mr, nr, kr, sr>(
            params.m,
            params.k,
            params.n,
            params.flat_scale_group_size,
            params.flat_lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      case 2:
        test_groupwise_lowbit_lut_kernel<2, has_scales, mr, nr, kr, sr>(
            params.m,
            params.k,
            params.n,
            params.flat_scale_group_size,
            params.flat_lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      case 3:
        test_groupwise_lowbit_lut_kernel<3, has_scales, mr, nr, kr, sr>(
            params.m,
            params.k,
            params.n,
            params.flat_scale_group_size,
            params.flat_lut_group_size,
            params.has_bias,
            params.has_clamp);
        break;
      case 4:
        test_groupwise_lowbit_lut_kernel<4, has_scales, mr, nr, kr, sr>(
            params.m,
            params.k,
            params.n,
            params.flat_scale_group_size,
            params.flat_lut_group_size,
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
        // --- Varying Dimensions ---
        // Test cases where n is a multiple of 4 (since lut_group_size = 256)
        KernelTestParams{8, 64, 16, 32, 256, true, true},
        KernelTestParams{8, 64, 12, 32, 256, true, true},
        KernelTestParams{8, 64, 8, 32, 256, true, true},
        KernelTestParams{8, 64, 4, 32, 256, true, true},

        // Test cases where n is a multiple of 8 (since lut_group_size = 512)
        KernelTestParams{8, 64, 24, 32, 512, true, true},
        KernelTestParams{8, 64, 16, 32, 512, true, true},
        KernelTestParams{8, 64, 8, 32, 512, true, true},

        // Test cases where n is a multiple of 16 (since lut_group_size = 1024)
        KernelTestParams{8, 64, 32, 32, 1024, true, true},
        KernelTestParams{8, 64, 16, 32, 1024, true, true},

        // Test unaligned M
        KernelTestParams{7, 64, 16, 32, 256, true, true},
        KernelTestParams{6, 64, 16, 32, 256, true, true},
        KernelTestParams{5, 64, 16, 32, 256, true, true},
        KernelTestParams{4, 64, 16, 32, 256, true, true},
        KernelTestParams{3, 64, 16, 32, 256, true, true},
        KernelTestParams{2, 64, 16, 32, 256, true, true},
        KernelTestParams{1, 64, 16, 32, 256, true, true},

        // --- Varying Group Sizes ---
        // Test where one LUT group covers multiple scale groups
        KernelTestParams{8, 64, 16, 32, 512, true, true},
        // Test with different group sizes that are not equal
        KernelTestParams{8, 64, 16, 32, 1024, true, true},
        KernelTestParams{8, 64, 16, 32, 1024, true, true},
        KernelTestParams{8, 64, 16, 32, 1024, true, true},
        // A single scale group is exactly one row of tiles.
        KernelTestParams{8, 64, 16, 32, 256, true, true},
        // All flags off (the simplest path)
        KernelTestParams{8, 64, 16, 32, 256, false, false},

        // All flags on
        KernelTestParams{8, 64, 16, 32, 256, true, true},

        // Other combinations
        KernelTestParams{8, 64, 16, 32, 256, true, true},
        KernelTestParams{8, 64, 16, 32, 256, true, false},
        // A single group covers the entire matrix.

        // --- Varying Boolean Flags ---
        // Test with only scales enabled
        KernelTestParams{8, 64, 16, 32, 256, false, false},
        // Test with only bias enabled
        KernelTestParams{8, 64, 16, 32, 256, true, false},
        // Test with only clamp enabled
        KernelTestParams{8, 64, 16, 32, 256, false, true},
        // Test with scales and clamp
        KernelTestParams{8, 64, 16, 32, 256, false, true},

        // --- Edges cases ---
        KernelTestParams{8, 64, 16, 32, 1024, true, true},
        // A single tile matrix.
        KernelTestParams{1, 32, 4, 32, 128, true, true},
        // Group sizes are exactly equal to the padded matrix size.
        KernelTestParams{8, 64, 16, 32, 1024, true, true}));

void PrintTo(const KernelTestParams& params, std::ostream* os) {
  *os << "KernelTestParams(m=" << params.m << ", k=" << params.k
      << ", n=" << params.n << ", scale_gs=" << params.flat_scale_group_size
      << ", lut_gs=" << params.flat_lut_group_size
      << ", bias=" << std::boolalpha << params.has_bias
      << ", clamp=" << std::boolalpha << params.has_clamp << ")";
}

#endif // defined(aarch64) || defined(__ARM_NEON)
