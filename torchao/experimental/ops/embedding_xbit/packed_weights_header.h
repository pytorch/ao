// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/packed_weights_header.h>

namespace torchao::ops::embedding_xbit {

inline torchao::ops::PackedWeightsHeader get_packed_weights_header_universal(
    int weight_nbit,
    int min_value_chunk_size,
    int max_value_chunk_size,
    int version = 1) {
  return torchao::ops::PackedWeightsHeader(
      torchao::ops::PackedWeightsType::embedding_xbit_universal,
      {version,
       weight_nbit,
       min_value_chunk_size,
       max_value_chunk_size,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0});
}

} // namespace torchao::ops::embedding_xbit
