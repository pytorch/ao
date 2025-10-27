// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <gtest/gtest.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/embedding/embedding_lut.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/tests/test_utils.h>
#include <vector>

float kTol = 0.0001;

template <int weight_nbit>
void test_embedding(
    int num_embeddings,
    int embedding_dim,
    int scale_group_size,
    int lut_group_size,
    bool has_scales) {
  auto test_case = torchao::lut_embedding_test_case<weight_nbit>::generate(
      num_embeddings,
      embedding_dim,
      scale_group_size,
      lut_group_size,
      has_scales);

  const size_t packed_embedding_size =
      torchao::kernels::cpu::aarch64::embedding::packed_embedding_size(
          weight_nbit,
          num_embeddings,
          embedding_dim,
          scale_group_size,
          lut_group_size,
          has_scales);

  auto packed = std::vector<uint8_t>(packed_embedding_size, 0);
  auto output = std::vector<float>(num_embeddings * embedding_dim, 0.0);

  for (int i = 0; i < num_embeddings; i++) {
    torchao::kernels::cpu::aarch64::embedding::pack_embedding_row_at_index_lut<
        weight_nbit>(
        packed.data(),
        i,
        test_case.weight_qval_idxs.data(),
        test_case.weight_scales.data(),
        test_case.weight_luts.data(),
        num_embeddings,
        embedding_dim,
        scale_group_size,
        lut_group_size,
        has_scales);
  }

  for (int i = 0; i < num_embeddings; i++) {
    torchao::kernels::cpu::aarch64::embedding::
        dequantize_embedding_row_at_idx_lut<weight_nbit>(
            output.data() + i * embedding_dim,
            packed.data(),
            i,
            num_embeddings,
            embedding_dim,
            scale_group_size,
            lut_group_size,
            has_scales);
  }

  for (int i = 0; i < num_embeddings * embedding_dim; i++) {
    EXPECT_NEAR(output[i], test_case.expected_outputs[i], kTol);
  }
}

struct LutEmbeddingBaseParams {
  int num_embeddings;
  int embedding_dim;
  int scale_group_size;
  int lut_group_size;
  bool has_scales;
};

class LutEmbeddingParamTest
    : public ::testing::TestWithParam<std::tuple<LutEmbeddingBaseParams, int>> {
 protected:
  // run_test now correctly accepts the base parameters
  template <int weight_nbit>
  void run_test(const LutEmbeddingBaseParams& params) {
    test_embedding<weight_nbit>(
        params.num_embeddings,
        params.embedding_dim,
        params.scale_group_size,
        params.lut_group_size,
        params.has_scales);
  };
};

TEST_P(LutEmbeddingParamTest, PackDequantizeEndToEnd) {
  const auto& base_params = std::get<0>(GetParam());
  const int weight_nbit = std::get<1>(GetParam());

  switch (weight_nbit) {
    case 4:
      run_test<4>(base_params);
      break;
    case 3:
      run_test<3>(base_params);
      break;
    case 2:
      run_test<2>(base_params);
      break;
    case 1:
      run_test<1>(base_params);
      break;
    default:
      FAIL() << "Unsupported weight_nbit: " << weight_nbit;
  }
}

INSTANTIATE_TEST_SUITE_P(
    LutEmbeddingParamSweep,
    LutEmbeddingParamTest,
    ::testing::Combine(
        ::testing::Values(
            LutEmbeddingBaseParams{8, 128, 64, 32, true},
            LutEmbeddingBaseParams{8, 128, 32, 32, true},
            LutEmbeddingBaseParams{4, 256, 128, 64, false},
            LutEmbeddingBaseParams{1, 64, 64, 64, true},
            LutEmbeddingBaseParams{16, 512, 64, 32, true},
            LutEmbeddingBaseParams{3, 96, 32, 32, true},
            LutEmbeddingBaseParams{8, 128, 64, 128, true},
            LutEmbeddingBaseParams{8, 128, 64, 256, true},
            LutEmbeddingBaseParams{8, 128, 64, 512, true}),
        ::testing::Values(1, 2, 3, 4)));
#endif // defined(__aarch64__) || defined(__ARM_NEON)
