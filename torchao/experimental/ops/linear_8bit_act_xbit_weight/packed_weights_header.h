// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/packed_weights_header.h>

namespace torchao::ops::linear_8bit_act_xbit_weight {

inline torchao::ops::PackedWeightsHeader get_packed_weights_header_universal(
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int version = 1) {
  return torchao::ops::PackedWeightsHeader(
      torchao::ops::PackedWeightsFormat::linear_8bit_act_xbit_weight_universal,
      {version,
       weight_nbit,
       has_weight_zeros,
       has_bias,
       nr,
       kr,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0});
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
