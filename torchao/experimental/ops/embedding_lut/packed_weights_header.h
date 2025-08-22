// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/packed_weights_header.h>

namespace torchao::ops::embedding_lut {

inline torchao::ops::PackedWeightsHeader get_packed_weights_header(
    int version,
    int weight_nbit,
    int num_embeddings,
    int embedding_dim,
    int scale_group_size,
    int lut_group_size,
    bool has_scales) {
  return torchao::ops::PackedWeightsHeader(
      torchao::ops::PackedWeightsType::groupwise_lowbit_embedding_lut,
      {
          version,
          weight_nbit,
          num_embeddings,
          embedding_dim,
          scale_group_size,
          lut_group_size,
          has_scales,
      });
}

} // namespace torchao::ops::embedding_lut
