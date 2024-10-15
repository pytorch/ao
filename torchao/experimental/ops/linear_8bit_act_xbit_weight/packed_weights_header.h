// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/experimental/ops/macro.h>
#include <torchao/experimental/ops/packed_weights_header.h>

namespace torchao::ops::linear_8bit_act_xbit_weight {

torchao::ops::PackedWeightsHeader get_packed_weights_header_universal(
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int version = 1) {
  TORCHAO_CHECK(
      version >= 0 && version < 256, "version must be between 0 and 255");
  TORCHAO_CHECK(
      weight_nbit >= 1 && weight_nbit < 256,
      "weight_nbit must be between 1 and 255");
  return torchao::ops::PackedWeightsHeader(
      torchao::ops::PackedWeightsFormat::linear_8bit_act_xbit_weight_universal,
      {((static_cast<unsigned short>(version) << 8) |
        static_cast<unsigned short>(weight_nbit)),
       ((static_cast<unsigned short>(has_weight_zeros) << 8) |
        static_cast<unsigned short>(has_bias)),
       static_cast<unsigned short>(nr),
       static_cast<unsigned short>(kr),
       0,
       0,
       0,
       0});
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
