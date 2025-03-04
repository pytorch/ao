// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(TORCHAO_BUILD_CPU_AARCH64)
#include <torchao/experimental/kernels/cpu/aarch64/embedding/embedding.h>
#endif // TORCHAO_BUILD_CPU_AARCH64

#include <torchao/experimental/ops/embedding_xbit/packed_weights_header.h>
#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/packed_weights_header.h>
#include <torchao/experimental/ops/parallel.h>

template <int weight_nbit>
void check_embedding_inputs(
    const Tensor& packed_weight_qvals,
    int num_embeddings,
    int embedding_dim,
    const Tensor& weight_scales,
    const Tensor& weight_zeros,
    const Tensor& indices,
    int& group_size) {
  TORCHAO_CHECK(
      packed_weight_qvals.dim() == 1, "packed_weight_qvals must be 1D");
#ifdef USE_ATEN
  TORCHAO_CHECK(
      packed_weight_qvals.dtype() == torch::kInt8,
      "packed_weight_qvals must be byte");
#endif // USE_ATEN
  TORCHAO_CHECK(
      (embedding_dim * weight_nbit) % 8 == 0,
      "embedding_dim * weight_nbit must be a multiple of 8");
  int packed_embedding_dim = (embedding_dim * weight_nbit) / 8;
  TORCHAO_CHECK(
      packed_weight_qvals.size(0) ==
          (torchao::ops::PackedWeightsHeader::size() +
           (num_embeddings * packed_embedding_dim)),
      "packed_weight_qvals is not the correct size");

  // Check header
  auto header = torchao::ops::PackedWeightsHeader::read(
      packed_weight_qvals.const_data_ptr());
  TORCHAO_CHECK(
      header ==
          torchao::ops::embedding_xbit::get_packed_weights_header_universal(
              weight_nbit,
              /*min_value_chunk_size=*/32,
              /*max_value_chunk_size=*/128),
      "packed_weights are not compatible with the kernel");

#ifdef USE_ATEN
  TORCHAO_CHECK(
      weight_scales.dtype() == torch::kFloat32,
      "weight_scales must be float32");
#endif // USE_ATEN
  TORCHAO_CHECK(weight_scales.dim() == 2, "weight_scales must be 2D");
  TORCHAO_CHECK(
      weight_scales.size(0) == num_embeddings,
      "weight_scales must be same shape as packed_weight_qvals in dim0 (num_embeddings)");
  int num_groups = weight_scales.size(1);
  TORCHAO_CHECK(
      num_groups >= 1, "weight_scales must be at least 1 in dim1 (num_groups)");
  TORCHAO_CHECK(
      embedding_dim % num_groups == 0,
      "embedding_dim must be a multiple of num_groups");
  group_size = embedding_dim / num_groups;
  TORCHAO_CHECK(group_size % 32 == 0, "group_size must be a multiple of 32");

#ifdef USE_ATEN
  TORCHAO_CHECK(
      weight_zeros.dtype() == torch::kInt8, "weight_zeros must be int8");
#endif // USE_ATEN
  TORCHAO_CHECK(weight_zeros.dim() == 2, "weight_zeros must be 2D");
  TORCHAO_CHECK(
      weight_zeros.size(0) == weight_scales.size(0) &&
          weight_zeros.size(1) == weight_scales.size(1),
      "zeros must be same shape as scales");

  TORCHAO_CHECK(indices.dim() == 1, "indices must be 1D");
  TORCHAO_CHECK(
      (indices.dtype() == Tensor_dtype_kInt32) ||
          (indices.dtype() == Tensor_dtype_kInt64),
      "indices must be int32 or int64");
}

#if defined(USE_ATEN) || defined(USE_EXECUTORCH)
template <int weight_nbit>
Tensor embedding_out_cpu(
    const Tensor& packed_weight_qvals,
    // TODO(T200095131): convert to
    // int64_t when supported by AOTI
    // Currently they are tensors with size
    // equal to (0, the int they wrap)
    const Tensor& num_embeddings_tensor,
    const Tensor& embedding_dim_tensor,
    const Tensor& weight_scales,
    const Tensor& weight_zeros,
    const Tensor& indices,
    Tensor& out) {
  int num_embeddings = num_embeddings_tensor.size(1);
  int embedding_dim = embedding_dim_tensor.size(1);
  int group_size;
  check_embedding_inputs<weight_nbit>(
      packed_weight_qvals,
      num_embeddings,
      embedding_dim,
      weight_scales,
      weight_zeros,
      indices,
      group_size);

  int num_out = indices.size(0);
  const int8_t* weight_zeros_ptr = weight_zeros.const_data_ptr<int8_t>();

#ifdef USE_ATEN
  TORCHAO_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
  out.resize_({num_out, embedding_dim});
#endif // USE_ATEN

#ifdef USE_EXECUTORCH
  TORCHAO_CHECK(out.dim() == 2, "out must be 2D");
  TORCHAO_CHECK(out.size(0) == num_out, "out shape is incorrect");
  TORCHAO_CHECK(out.size(1) == embedding_dim, "out shape is incorrect");
#endif // USE_EXECUTORCH

  const int32_t* index32_ptr = nullptr;
  const int64_t* index64_ptr = nullptr;
  if (indices.dtype() == Tensor_dtype_kInt32) {
    index32_ptr = indices.const_data_ptr<int32_t>();
  } else {
    TORCHAO_CHECK(
        indices.dtype() == Tensor_dtype_kInt64,
        "indices must be int32 or int64");
    index64_ptr = indices.const_data_ptr<int64_t>();
  }
  torchao::parallel_1d(0, num_out, [&](int64_t idx) {
    int index = -1;
    if (index32_ptr != nullptr) {
      index = index32_ptr[idx];
    } else {
      index = index64_ptr[idx];
    }
    TORCHAO_CHECK(index >= 0 && index < num_embeddings, "index out of bounds");
#if defined(TORCHAO_BUILD_CPU_AARCH64)
    torchao::kernels::cpu::aarch64::embedding::embedding<weight_nbit>(
        out.mutable_data_ptr<float>() + idx * embedding_dim,
        embedding_dim,
        group_size,
        packed_weight_qvals.const_data_ptr<int8_t>() +
            torchao::ops::PackedWeightsHeader::size(),
        weight_scales.const_data_ptr<float>(),
        weight_zeros_ptr,
        index);
#else
    TORCHAO_CHECK(false, "Unsupported platform");
#endif // TORCHAO_BUILD_CPU_AARCH64
  });

  return out;
}
#endif // defined(USE_ATEN) || defined(USE_EXECUTORCH)

#ifdef USE_ATEN
template <int weight_nbit>
Tensor embedding_cpu(
    const Tensor& packed_weight_qvals,
    // TODO(T200095131): convert to
    // int64_t when supported by AOTI
    // Currently they are tensors with size
    // equal to (0, the int they wrap)
    const Tensor& num_embeddings_tensor,
    const Tensor& embedding_dim_tensor,
    const Tensor& weight_scales,
    const Tensor& weight_zeros,
    const Tensor& indices) {
  Tensor output_tensor = torch::empty({}, torch::kFloat32);
  embedding_out_cpu<weight_nbit>(
      packed_weight_qvals,
      num_embeddings_tensor,
      embedding_dim_tensor,
      weight_scales,
      weight_zeros,
      indices,
      output_tensor);
  return output_tensor;
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit>
Tensor embedding_meta(
    const Tensor& packed_weight_qvals,
    // TODO(T200095131): convert to
    // int64_t when supported by AOTI
    // Currently they are tensors with size
    // equal to (0, the int they wrap)
    const Tensor& num_embeddings_tensor,
    const Tensor& embedding_dim_tensor,
    const Tensor& weight_scales,
    const Tensor& weight_zeros,
    const Tensor& indices) {
  int embedding_dim = embedding_dim_tensor.size(1);
  int num_out = indices.size(0);
  return torch::empty({num_out, embedding_dim}).to("meta");
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit>
Tensor pack_embedding_cpu(const Tensor& weight_qvals) {
  TORCHAO_CHECK(weight_qvals.dim() == 2, "weight_qvals must be 2D");
  int num_embeddings = weight_qvals.size(0);
  int embedding_dim = weight_qvals.size(1);
  TORCHAO_CHECK(
      embedding_dim % 8 == 0, "embedding_dim must be a multiple of 8 to pack");
  int packed_embedding_dim = embedding_dim * weight_nbit / 8;
  TORCHAO_CHECK(
      weight_qvals.dtype() == torch::kInt8, "weight_qvals must be int8");

  auto out = torch::empty(
                 torchao::ops::PackedWeightsHeader::size() +
                 (num_embeddings * packed_embedding_dim))
                 .to(torch::kInt8);

  auto header =
      torchao::ops::embedding_xbit::get_packed_weights_header_universal(
          weight_nbit,
          /*min_value_chunk_size=*/32,
          /*max_value_chunk_size=*/128);
  header.write(out.mutable_data_ptr());

  torchao::parallel_1d(0, num_embeddings, [&](int64_t idx) {
#if defined(TORCHAO_BUILD_CPU_AARCH64)
    torchao::kernels::cpu::aarch64::embedding::pack_embedding_weight_qvals<
        weight_nbit>(
        out.mutable_data_ptr<int8_t>() +
            torchao::ops::PackedWeightsHeader::size(),
        embedding_dim,
        weight_qvals.const_data_ptr<int8_t>(),
        idx);
#else
    TORCHAO_CHECK(false, "Unsupported platform");
#endif // defined(TORCHAO_BUILD_CPU_AARCH64)
  });

  return out;
}
#endif // USE_ATEN

#ifdef USE_ATEN
template <int weight_nbit>
Tensor pack_embedding_meta(const Tensor& weight_qvals) {
  TORCHAO_CHECK(weight_qvals.dim() == 2, "weight_qvals must be 2D");
  int num_embeddings = weight_qvals.size(0);
  int embedding_dim = weight_qvals.size(1);
  TORCHAO_CHECK(
      embedding_dim % 8 == 0, "embedding_dim must be a multiple of 8 to pack");
  int packed_embedding_dim = embedding_dim * weight_nbit / 8;
  return torch::empty(
             torchao::ops::PackedWeightsHeader::size() +
             (num_embeddings * packed_embedding_dim))
      .to("meta");
}
#endif // USE_ATEN
