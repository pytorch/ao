// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/aarch64/embedding/embedding.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/pack_weights.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <vector>

float kTol = 0.0001;

template <int weight_nbit>
void test_embedding(
    int num_embeddings,
    int embedding_dim,
    int group_size,
    bool has_weight_zeros) {
  auto test_case = torchao::lowbit_embedding_test_case<weight_nbit>::generate(
      num_embeddings, embedding_dim, group_size, has_weight_zeros);

  auto packed = std::vector<unsigned char>(
      num_embeddings * embedding_dim * weight_nbit / 8, 0);
  auto output = std::vector<float>(num_embeddings * embedding_dim, 0.0);

  for (int i = 0; i < num_embeddings; i++) {
    torchao::kernels::cpu::aarch64::embedding::pack_embedding_weight_qvals<
        weight_nbit>(
        packed.data(), embedding_dim, test_case.weight_qvals.data(), i);
  }

  int8_t* weight_zeros = nullptr;
  if (has_weight_zeros) {
    weight_zeros = test_case.weight_zeros.data();
  }

  for (int i = 0; i < num_embeddings; i++) {
    torchao::kernels::cpu::aarch64::embedding::embedding<weight_nbit>(
        output.data() + i * embedding_dim,
        embedding_dim,
        group_size,
        packed.data(),
        test_case.weight_scales.data(),
        weight_zeros,
        i);
  }

  for (int i = 0; i < num_embeddings * embedding_dim; i++) {
    EXPECT_NEAR(output[i], test_case.expected_outputs[i], kTol);
  }
}

template <int weight_nbit, int nr, int kr, int sr>
void test_shared_embedding(
    int num_embeddings,
    int embedding_dim,
    int group_size,
    bool has_weight_zeros) {
  auto test_case = torchao::lowbit_embedding_test_case<weight_nbit>::generate(
      num_embeddings, embedding_dim, group_size, has_weight_zeros);

  // Pack weights for linear op
  int n = num_embeddings;
  int k = embedding_dim;
  bool has_bias = false;
  float* bias = nullptr;
  std::vector<char> packed_weights(
      torchao::kernels::cpu::aarch64::linear::packing::packed_weights_size(
          n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr));
  torchao::kernels::cpu::aarch64::linear::packing::
      pack_weights<weight_nbit, nr, kr, sr>(
          packed_weights.data(),
          n,
          k,
          group_size,
          test_case.weight_qvals.data(),
          test_case.weight_scales.data(),
          has_weight_zeros ? test_case.weight_zeros.data() : nullptr,
          bias);

  // Call shared_embedding
  auto output = std::vector<float>(num_embeddings * embedding_dim, 0.0);
  for (int i = 0; i < num_embeddings; i++) {
    torchao::kernels::cpu::aarch64::embedding::
        shared_embedding<weight_nbit, nr, kr, sr>(
            output.data() + i * embedding_dim,
            packed_weights.data(),
            n,
            k,
            group_size,
            has_weight_zeros,
            has_bias,
            i);
  }

  for (int i = 0; i < num_embeddings * embedding_dim; i++) {
    EXPECT_NEAR(output[i], test_case.expected_outputs[i], kTol);
  }
}

TEST(test_embedding, NBit1) {
  constexpr int num_embeddings = 5;
  constexpr int group_size = 128 * 3 + 64 + 32;
  constexpr int embedding_dim = group_size * 7;

  test_embedding<1>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/true);
  test_embedding<1>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/false);
}

TEST(test_embedding, NBit2) {
  constexpr int num_embeddings = 5;
  constexpr int group_size = 128 * 3 + 64 + 32;
  constexpr int embedding_dim = group_size * 7;

  test_embedding<2>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/true);
  test_embedding<2>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/false);
}

TEST(test_embedding, NBit3) {
  constexpr int num_embeddings = 5;
  constexpr int group_size = 128 * 3 + 64 + 32;
  constexpr int embedding_dim = group_size * 7;

  test_embedding<3>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/true);
  test_embedding<3>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/false);
}

TEST(test_embedding, NBit4) {
  constexpr int num_embeddings = 5;
  constexpr int group_size = 128 * 3 + 64 + 32;
  constexpr int embedding_dim = group_size * 7;

  test_embedding<4>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/true);
  test_embedding<4>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/false);

  // More detailed testing for 4-bit case

  test_embedding<4>(
      num_embeddings,
      /*embedding_dim=*/256,
      /*group_size=*/32,
      /*has_weight_zeros=*/true);
  test_embedding<4>(
      num_embeddings,
      /*embedding_dim=*/256,
      /*group_size=*/32,
      /*has_weight_zeros=*/false);
  test_embedding<4>(
      num_embeddings,
      /*embedding_dim=*/256,
      /*group_size=*/64,
      /*has_weight_zeros=*/true);
  test_embedding<4>(
      num_embeddings,
      /*embedding_dim=*/256,
      /*group_size=*/64,
      /*has_weight_zeros=*/false);
  test_embedding<4>(
      num_embeddings,
      /*embedding_dim=*/256,
      /*group_size=*/128,
      /*has_weight_zeros=*/true);
  test_embedding<4>(
      num_embeddings,
      /*embedding_dim=*/256,
      /*group_size=*/128,
      /*has_weight_zeros=*/false);
}

TEST(test_embedding, NBit5) {
  constexpr int num_embeddings = 5;
  constexpr int group_size = 128 * 3 + 64 + 32;
  constexpr int embedding_dim = group_size * 7;

  test_embedding<5>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/true);
  test_embedding<5>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/false);
}

TEST(test_embedding, NBit6) {
  constexpr int num_embeddings = 5;
  constexpr int group_size = 128 * 3 + 64 + 32;
  constexpr int embedding_dim = group_size * 7;

  test_embedding<6>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/true);
  test_embedding<6>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros=*/false);
}

TEST(test_embedding, SharedEmbeddingTest) {
  constexpr int weight_nbit = 3;
  constexpr int num_embeddings = 17;
  constexpr int group_size = 64;
  constexpr int embedding_dim = group_size * 7;

  test_shared_embedding<weight_nbit, /*nr*/ 8, /*kr*/ 16, /*sr*/ 2>(
      num_embeddings, embedding_dim, group_size, /*has_weight_zeros*/true);
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
