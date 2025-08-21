// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(TORCHAO_BUILD_CPU_AARCH64)
#include <torchao/experimental/kernels/cpu/aarch64/embedding/embedding_lut.h>
#endif // TORCHAO_BUILD_CPU_AARCH64

#include <torchao/experimental/ops/embedding_lut/packed_weights_header.h>
#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/parallel.h>

template <int weight_nbit>
void check_embedding_lut_inputs(
    const Tensor& packed_weight_indices,
    const Tensor& indices,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t scale_group_size,
    int64_t lut_group_size,
    bool has_scales) {
  // Check packed weights header
  TORCHAO_CHECK(
      packed_weight_indices.dim() == 1, "packed_weight_indices must be 1D");
#ifdef USE_ATEN
  TORCHAO_CHECK(
      packed_weight_indices.dtype() == torch::kInt8,
      "packed_weight_indices must be byte");
#endif // USE_ATEN
  TORCHAO_CHECK(
      packed_weight_indices.size(0) >=
          torchao::ops::PackedWeightsHeader::size(),
      "packed_weight_indices is not large enough to contain a header");

  // Check indices tensor
  TORCHAO_CHECK(indices.dim() == 1, "indices must be 1D");
  TORCHAO_CHECK(
      (indices.dtype() == Tensor_dtype_kInt32) ||
          (indices.dtype() == Tensor_dtype_kInt64),
      "indices must be int32 or int64");

  // Check header
  auto header = torchao::ops::PackedWeightsHeader::read(
      packed_weight_indices.const_data_ptr());
  TORCHAO_CHECK(
      header ==
          torchao::ops::embedding_lut::get_packed_weights_header(
              /*version=*/1,
              weight_nbit,
              num_embeddings,
              embedding_dim,
              scale_group_size,
              lut_group_size,
              has_scales),
      "packed_weights are not compatible with the kernel");
}

#if defined(USE_ATEN) || defined(USE_EXECUTORCH)
template <int weight_nbit>
Tensor embedding_out_cpu(
    const Tensor& packed_weights,
    const Tensor& indices,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t scale_group_size,
    int64_t lut_group_size,
    bool has_scales,
    Tensor& out) {
  check_embedding_lut_inputs<weight_nbit>(
      packed_weights,
      indices,
      num_embeddings,
      embedding_dim,
      scale_group_size,
      lut_group_size,
      has_scales);

  const int num_out = indices.size(0);
  TORCHAO_RESIZE_TENSOR(out, {(int)num_out, (int)embedding_dim});

  const int32_t* index32_ptr = nullptr;
  const int64_t* index64_ptr = nullptr;
  if (indices.dtype() == Tensor_dtype_kInt32) {
    index32_ptr = indices.const_data_ptr<int32_t>();
  } else {
    index64_ptr = indices.const_data_ptr<int64_t>();
  }

  // The actual packed data starts after the header
  const void* packed_data_ptr = packed_weights.const_data_ptr<int8_t>() +
      torchao::ops::PackedWeightsHeader::size();

  torchao::parallel_1d(0, num_out, [&](int64_t idx) {
    int index = (index32_ptr != nullptr) ? index32_ptr[idx] : index64_ptr[idx];
    TORCHAO_CHECK(index >= 0 && index < num_embeddings, "Index out of bounds");

#if defined(TORCHAO_BUILD_CPU_AARCH64)
    torchao::kernels::cpu::aarch64::embedding::
        dequantize_embedding_row_at_idx_lut<weight_nbit>(
            out.mutable_data_ptr<float>() + idx * embedding_dim,
            packed_data_ptr,
            index,
            num_embeddings,
            embedding_dim,
            scale_group_size,
            lut_group_size,
            has_scales);
#else
    TORCHAO_CHECK(false, "Unsupported platform for embedding_lut kernel");
#endif // TORCHAO_BUILD_CPU_AARCH64
  });

  return out;
}
#endif // defined(USE_ATEN) || defined(USE_EXECUTORCH)

#ifdef USE_ATEN
template <int weight_nbit>
Tensor embedding_cpu(
    const Tensor& packed_weights,
    const Tensor& indices,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t scale_group_size,
    int64_t lut_group_size,
    bool has_scales) {
  Tensor output_tensor = torch::empty({0}, torch::kFloat32);
  embedding_out_cpu<weight_nbit>(
      packed_weights,
      indices,
      num_embeddings,
      embedding_dim,
      scale_group_size,
      lut_group_size,
      has_scales,
      output_tensor);
  return output_tensor;
}

template <int weight_nbit>
Tensor pack_embedding_cpu(
    const Tensor& weight_qval_idxs,
    const Tensor& luts,
    int64_t scale_group_size,
    int64_t lut_group_size,
    const std::optional<Tensor>& weight_scales) {
  const bool has_scales = weight_scales.has_value();
  TORCHAO_CHECK(weight_qval_idxs.dim() == 2, "weight_qval_idxs must be 2D");
  const int64_t num_embeddings = weight_qval_idxs.size(0);
  const int64_t embedding_dim = weight_qval_idxs.size(1);

  TORCHAO_CHECK(
      (embedding_dim * weight_nbit) % 8 == 0,
      "Total bits must be a multiple of 8.");

  const size_t packed_embedding_size =
      torchao::kernels::cpu::aarch64::embedding::packed_embedding_size(
          weight_nbit,
          num_embeddings,
          embedding_dim,
          scale_group_size,
          lut_group_size,
          has_scales);
  const size_t total_packed_size =
      torchao::ops::PackedWeightsHeader::size() + packed_embedding_size;

  // Allocate and Pack
  auto out = torch::empty({(long)total_packed_size}, torch::kInt8);

  // Write header
  auto header = torchao::ops::embedding_lut::get_packed_weights_header(
      /*version=*/1,
      weight_nbit,
      num_embeddings,
      embedding_dim,
      scale_group_size,
      lut_group_size,
      has_scales);
  header.write(out.mutable_data_ptr());

  void* packed_table_ptr = out.mutable_data_ptr<int8_t>() +
      torchao::ops::PackedWeightsHeader::size();

  // Pack each row
  torchao::parallel_1d(0, num_embeddings, [&](int64_t i) {
#if defined(TORCHAO_BUILD_CPU_AARCH64)
    torchao::kernels::cpu::aarch64::embedding::pack_embedding_row_at_index_lut<
        weight_nbit>(
        packed_table_ptr,
        i,
        weight_qval_idxs.const_data_ptr<uint8_t>(),
        has_scales ? weight_scales->const_data_ptr<float>() : nullptr,
        luts.const_data_ptr<float>(),
        num_embeddings,
        embedding_dim,
        scale_group_size,
        lut_group_size,
        has_scales);
#else
    TORCHAO_CHECK(false, "Unsupported platform for pack_embedding kernel");
#endif // defined(TORCHAO_BUILD_CPU_AARCH64)
  });

  return out;
}

template <int weight_nbit>
Tensor pack_embedding_meta(
    const Tensor& weight_qval_idxs,
    const Tensor& luts,
    int64_t scale_group_size,
    int64_t lut_group_size,
    const std::optional<Tensor>& weight_scales) {
  const int64_t num_embeddings = weight_qval_idxs.size(0);
  const int64_t embedding_dim = weight_qval_idxs.size(1);
  const bool has_scales = weight_scales.has_value();

  TORCHAO_CHECK(
      (embedding_dim * weight_nbit) % 8 == 0,
      "Total bits must be a multiple of 8 for meta function.");

  const size_t packed_embedding_size =
      torchao::kernels::cpu::aarch64::embedding::packed_embedding_size(
          weight_nbit,
          num_embeddings,
          embedding_dim,
          scale_group_size,
          lut_group_size,
          has_scales);
;
  const size_t total_packed_size = torchao::ops::PackedWeightsHeader::size() + packed_embedding_size;

  auto options =
      torch::TensorOptions().device(c10::DeviceType::Meta).dtype(torch::kInt8);
  return torch::empty({(long)total_packed_size}, options);
}
#endif // USE_ATEN
