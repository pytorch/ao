// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <array>

#include <cassert>
namespace torchao::ops {

enum PackedWeightsFormat : unsigned short {
  unknown = 0,
  linear_8bit_act_xbit_weight_universal = 1
};

class PackedWeightsHeader {
 public:
  using params_type = std::array<unsigned short, 7>;
  PackedWeightsFormat format;

  // 14 bytes of format specific params
  params_type params;

  PackedWeightsHeader(
      PackedWeightsFormat format = PackedWeightsFormat::unknown,
      params_type params = {0, 0, 0, 0, 0, 0, 0})
      : format{format}, params{params} {}

  inline static constexpr int size() {
    static_assert(sizeof(format) + sizeof(params) == 16);
    return 16;
  }

  inline void write(void* packed_weights) const {
    auto header = (unsigned short*)(packed_weights);
    header[0] = (unsigned short)format;
    for (int i = 0; i < params.size(); i++) {
      header[i + 1] = params[i];
    }
  }

  static PackedWeightsHeader read(const void* packed_weights) {
    auto header = (unsigned short*)(packed_weights);
    params_type params;
    for (int i = 0; i < params.size(); i++) {
      params[i] = header[i + 1];
    }
    return PackedWeightsHeader((PackedWeightsFormat)header[0], params);
  }

  bool operator==(const PackedWeightsHeader& other) const {
    if (format != other.format) {
      return false;
    }
    for (int i = 0; i < params.size(); i++) {
      if (params[i] != other.params[i]) {
        return false;
      }
    }
    return true;
  }
};

} // namespace torchao::ops
