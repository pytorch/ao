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

enum class PackedWeightsType : uint32_t {
  unknown = 0,
  linear_8bit_act_xbit_weight_universal = 1,
  embedding_xbit_universal = 2,
  linear_8bit_act_xbit_weight_kleidi_ai = 3,
  linear_8bit_act_xbit_weight_lut = 4,
  groupwise_lowbit_weight_lut = 5,
  groupwise_lowbit_embedding_lut = 6,
};

class PackedWeightsHeader {
 public:
  using params_type = std::array<int, 14>;
  const static int magic = 6712;
  PackedWeightsType type;

  // 14 bytes of type specific params
  params_type params;

  PackedWeightsHeader(
      PackedWeightsType type = PackedWeightsType::unknown,
      params_type params = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
      : type{type}, params{params} {}

  inline static constexpr int size() {
    static_assert(sizeof(magic) + sizeof(type) + sizeof(params) == 64);
    return 64;
  }

  inline void write(void* packed_weights) const {
    auto header = reinterpret_cast<int*>(packed_weights);
    header[0] = magic;
    header[1] = static_cast<int>(type);
    for (size_t i = 0; i < params.size(); i++) {
      header[i + 2] = params[i];
    }
  }

  static PackedWeightsHeader read(const void* packed_weights) {
    auto header = reinterpret_cast<const int*>(packed_weights);
    assert(header[0] == PackedWeightsHeader::magic);
    params_type params;
    for (size_t i = 0; i < params.size(); i++) {
      params[i] = header[i + 2];
    }
    return PackedWeightsHeader(
        static_cast<PackedWeightsType>(header[1]), params);
  }

  bool operator==(const PackedWeightsHeader& other) const {
    if (type != other.type) {
      return false;
    }
    for (size_t i = 0; i < params.size(); i++) {
      if (params[i] != other.params[i]) {
        return false;
      }
    }
    return true;
  }
};

} // namespace torchao::ops

namespace std {
    template <>
    struct hash<torchao::ops::PackedWeightsHeader> {
        std::size_t operator()(const torchao::ops::PackedWeightsHeader& f) const {
          std::size_t hash =  std::hash<int>()(static_cast<int>(f.type));
          for (size_t i = 0; i < f.params.size(); i++) {
            hash ^= std::hash<int>()(f.params[i]);
          }
          return hash;
    };
};
}
