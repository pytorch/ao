// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight/pack_weights.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/tests/test_utils.h>

template <int weight_nbit, int nr, int kr, int sr>
void test_weight_packing(
    int k,
    int n,
    int group_size,
    bool has_weight_zeros,
    bool has_bias) {
  auto test_case = torchao::
      channelwise_8bit_activation_groupwise_lowbit_weight_test_case::generate(
          /*m*/ 1,
          k,
          n,
          group_size,
          weight_nbit,
          has_weight_zeros,
          has_bias,
          /*has_clamp*/ false);

  //   using namespace torchao::kernels::cpu::aarch64::linear::packing;

  std::vector<char> packed_weights(
      torchao::kernels::cpu::aarch64::linear::
          channelwise_8bit_activation_groupwise_lowbit_weight::weight_packing::
              packed_weights_size(
                  n,
                  k,
                  group_size,
                  weight_nbit,
                  has_weight_zeros,
                  has_bias,
                  nr));

  int8_t* weight_qvals_in = test_case.weight_qvals.data();
  float* weight_scales_in = test_case.weight_scales.data();
  int8_t* weight_zeros_in = nullptr;
  if (has_weight_zeros) {
    weight_zeros_in = test_case.weight_zeros.data();
  }
  float* bias_in = nullptr;
  if (has_bias) {
    bias_in = test_case.bias.data();
  }

  std::vector<int8_t> weight_qvals_out(test_case.weight_qvals.size());
  std::vector<float> weight_scales_out(test_case.weight_scales.size());
  std::vector<int8_t> weight_zeros_out(test_case.weight_zeros.size());
  std::vector<float> bias_out(test_case.bias.size());

  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::weight_packing::
          pack_weights<weight_nbit, nr, kr, sr>(
              packed_weights.data(),
              n,
              k,
              group_size,
              weight_qvals_in,
              weight_scales_in,
              weight_zeros_in,
              bias_in);
  torchao::kernels::cpu::aarch64::linear::
      channelwise_8bit_activation_groupwise_lowbit_weight::weight_packing::
          unpack_weights<weight_nbit, nr, kr, sr>(
              weight_qvals_out.data(),
              weight_scales_out.data(),
              weight_zeros_out.data(),
              bias_out.data(),
              n,
              k,
              group_size,
              has_weight_zeros,
              has_bias,
              packed_weights.data());

  for (int i = 0; i < test_case.weight_qvals.size(); ++i) {
    EXPECT_EQ(weight_qvals_out[i], test_case.weight_qvals[i]);
  }
  for (int i = 0; i < test_case.weight_scales.size(); ++i) {
    EXPECT_EQ(weight_scales_out[i], test_case.weight_scales[i]);
  }
  for (int i = 0; i < test_case.weight_zeros.size(); ++i) {
    EXPECT_EQ(weight_zeros_out[i], test_case.weight_zeros[i]);
  }
  for (int i = 0; i < test_case.bias.size(); ++i) {
    EXPECT_EQ(bias_out[i], test_case.bias[i]);
  }
}

TEST(TestWeightPacking, PackUnpackAreSame) {
  int n = 13;
  int group_size = 32;
  int k = group_size * 17;

  test_weight_packing<4, /*nr*/ 8, /*kr*/ 16, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ true, /*has_bias*/ true);
  test_weight_packing<4, /*nr*/ 8, /*kr*/ 16, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ false, /*has_bias*/ true);
  test_weight_packing<4, /*nr*/ 8, /*kr*/ 16, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ true, /*has_bias*/ false);
  test_weight_packing<4, /*nr*/ 8, /*kr*/ 16, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ false, /*has_bias*/ false);

  test_weight_packing<3, /*nr*/ 4, /*kr*/ 16, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ true, /*has_bias*/ true);
  test_weight_packing<3, /*nr*/ 4, /*kr*/ 16, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ false, /*has_bias*/ true);
  test_weight_packing<3, /*nr*/ 4, /*kr*/ 16, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ true, /*has_bias*/ false);
  test_weight_packing<3, /*nr*/ 4, /*kr*/ 16, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ false, /*has_bias*/ false);

  test_weight_packing<2, /*nr*/ 1, /*kr*/ 32, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ true, /*has_bias*/ true);
  test_weight_packing<2, /*nr*/ 1, /*kr*/ 32, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ false, /*has_bias*/ true);
  test_weight_packing<2, /*nr*/ 1, /*kr*/ 32, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ true, /*has_bias*/ false);
  test_weight_packing<2, /*nr*/ 1, /*kr*/ 32, /*sr*/ 2>(
      k, n, group_size, /*has_weight_zeros*/ false, /*has_bias*/ false);
}
