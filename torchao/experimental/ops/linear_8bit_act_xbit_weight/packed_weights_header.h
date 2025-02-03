// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/packed_weights_header.h>

namespace torchao::ops::linear_8bit_act_xbit_weight {

inline torchao::ops::PackedWeightsFormat get_packed_weights_format_universal(
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int version = 1) {
  return torchao::ops::PackedWeightsFormat(
      torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal,
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

struct UniversalPackingParams {
  int version;
  int weight_nbit;
  bool has_weight_zeros;
  bool has_bias;
  int nr;
  int kr;
};

inline UniversalPackingParams get_universal_packing_params(torchao::ops::PackedWeightsFormat format) {
  if (format.type != torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal) {
    throw std::runtime_error("Packed weights are not in universal packing format.");
  }
  return UniversalPackingParams{
      format.params[0],
      format.params[1],
      static_cast<bool>(format.params[2]),
      static_cast<bool>(format.params[3]),
      format.params[4],
      format.params[5],
  };
}


inline torchao::ops::PackedWeightsFormat get_packed_weights_format_kleidi_ai(
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  return torchao::ops::PackedWeightsFormat(
      torchao::ops::PackedWeightsType::kleidi_ai,
      {weight_nbit,
       has_weight_zeros,
       has_bias,
       nr,
       kr,
       sr,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0});
}

struct KleidiAIPackingParams {
  int weight_nbit;
  bool has_weight_zeros;
  bool has_bias;
  int nr;
  int kr;
  int sr;
};

inline KleidiAIPackingParams get_kleidi_ai_packing_params(torchao::ops::PackedWeightsFormat format) {
  if (format.type != torchao::ops::PackedWeightsType::kleidi_ai) {
    throw std::runtime_error("Packed weights are not in kleidi_ai packing format.");
  }
  return KleidiAIPackingParams{
      format.params[0],
      static_cast<bool>(format.params[1]),
      static_cast<bool>(format.params[2]),
      format.params[3],
      format.params[4],
      format.params[5]
    };
}


} // namespace torchao::ops::linear_8bit_act_xbit_weight
