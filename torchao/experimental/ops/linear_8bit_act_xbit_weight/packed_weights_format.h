// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/experimental/ops/packed_weights_header.h>

namespace torchao::ops::linear_8bit_act_xbit_weight {

struct PackedWeightsFormat {
  torchao::ops::PackedWeightsType type;
  int weight_nbit;
  bool has_weight_zeros;
  bool has_bias;
  int nr;
  int kr;
  int sr;

  PackedWeightsFormat(
      torchao::ops::PackedWeightsType type,
      int weight_nbit,
      bool has_weight_zeros,
      bool has_bias,
      int nr,
      int kr,
      int sr)
      : type{type},
        weight_nbit{weight_nbit},
        has_weight_zeros{has_weight_zeros},
        has_bias{has_bias},
        nr{nr},
        kr{kr},
        sr{sr} {}

  static PackedWeightsFormat from_packed_weights_header(
      torchao::ops::PackedWeightsHeader header) {
    return PackedWeightsFormat(
        header.type,
        header.params[0],
        static_cast<bool>(header.params[1]),
        static_cast<bool>(header.params[2]),
        header.params[3],
        header.params[4],
        header.params[5]);
  }

  inline torchao::ops::PackedWeightsHeader to_packed_weights_header() const {
    return torchao::ops::PackedWeightsHeader(
        type, {weight_nbit, has_weight_zeros, has_bias, nr, kr, sr});
  }
};

template <int weight_nbit>
void check_format(
    PackedWeightsFormat format,
    torchao::ops::PackedWeightsType type) {
  if (format.type != type) {
    throw std::runtime_error(
        "Kernel expects packed_weights type=" +
        std::to_string(static_cast<int>(type)) +
        ", but got packed_weights with type=" +
        std::to_string(static_cast<int>(format.type)));
  }
  if (format.weight_nbit != weight_nbit) {
    throw std::runtime_error(
        "Kernel expects weight_nbit=" + std::to_string(weight_nbit) +
        ", but got packed_weights with weight_nbit=" +
        std::to_string(format.weight_nbit));
  }
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
