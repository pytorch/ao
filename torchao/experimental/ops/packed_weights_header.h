// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <array>

#include <stdint.h>
#include <cassert>

namespace torchao::ops {

enum class PackedWeightsFormat : uint32_t {
  unknown = 0,
  linear_8bit_act_xbit_weight_universal = 1,
  embedding_xbit_universal = 2
};

class PackedWeightsHeader {
 public:
  using params_type = std::array<int, 14>;
  const static int magic = 6712;
  PackedWeightsFormat format;

  // 14 bytes of format specific params
  params_type params;

  PackedWeightsHeader(
      PackedWeightsFormat format = PackedWeightsFormat::unknown,
      params_type params = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
      : format{format}, params{params} {}

  inline static constexpr int size() {
    static_assert(sizeof(magic) + sizeof(format) + sizeof(params) == 64);
    return 64;
  }

  inline void write(void* packed_weights) const {
    auto header = reinterpret_cast<int*>(packed_weights);
    header[0] = magic;
    header[1] = static_cast<int>(format);
    for (int i = 0; i < params.size(); i++) {
      header[i + 2] = params[i];
    }
  }

  static PackedWeightsHeader read(const void* packed_weights) {
    auto header = reinterpret_cast<const int*>(packed_weights);
    assert(header[0] == PackedWeightsHeader::magic);
    params_type params;
    for (int i = 0; i < params.size(); i++) {
      params[i] = header[i + 2];
    }
    return PackedWeightsHeader(
        static_cast<PackedWeightsFormat>(header[1]), params);
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
