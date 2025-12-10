// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/shared_kernels/internal/packed_weights_header.h>
#include <stdexcept>

namespace torchao::ops::groupwise_lowbit_weight_lut {

/**
 * @brief Defines the format parameters for the packed weights of the
 * groupwise LUT kernel.
 */
struct PackedWeightsFormat {
  torchao::ops::PackedWeightsType type;
  int weight_nbit;
  int scale_group_size;
  int lut_group_size;
  bool has_scales;
  bool has_bias;
  int nr;
  int kr;
  int sr;

  PackedWeightsFormat(
      torchao::ops::PackedWeightsType type,
      int weight_nbit,
      int scale_group_size,
      int lut_group_size,
      bool has_scales,
      bool has_bias,
      int nr,
      int kr,
      int sr)
      : type{type},
        weight_nbit{weight_nbit},
        scale_group_size{scale_group_size},
        lut_group_size{lut_group_size},
        has_scales{has_scales},
        has_bias{has_bias},
        nr{nr},
        kr{kr},
        sr{sr} {}

  /**
   * @brief Converts a generic PackedWeightsHeader into this specific format.
   *
   * This assumes the generic header's `params` array is populated in the
   * correct order.
   */
  static PackedWeightsFormat from_packed_weights_header(
      const torchao::ops::PackedWeightsHeader& header) {
    return PackedWeightsFormat(
        header.type,
        header.params[0], // weight_nbit
        header.params[1], // scale_group_size
        header.params[2], // lut_group_size
        static_cast<bool>(header.params[3]), // has_scales
        static_cast<bool>(header.params[4]), // has_bias
        header.params[5], // nr
        header.params[6], // kr
        header.params[7] // sr
    );
  }

  /**
   * @brief Converts this specific format into a generic PackedWeightsHeader.
   */
  inline torchao::ops::PackedWeightsHeader to_packed_weights_header() const {
    return torchao::ops::PackedWeightsHeader(
        type,
        {weight_nbit,
         scale_group_size,
         lut_group_size,
         has_scales,
         has_bias,
         nr,
         kr,
         sr});
  }
};

/**
 * @brief Helper function to validate that the provided format matches the
 * expectations of a specific kernel.
 */
inline void check_format(
    const PackedWeightsFormat& format,
    torchao::ops::PackedWeightsType expected_type,
    int expected_weight_nbit) {
  if (format.type != expected_type) {
    throw std::runtime_error(
        "Kernel expects packed_weights type=" +
        std::to_string(static_cast<int>(expected_type)) +
        ", but got packed_weights with type=" +
        std::to_string(static_cast<int>(format.type)));
  }
  if (format.weight_nbit != expected_weight_nbit) {
    throw std::runtime_error(
        "Kernel expects weight_nbit=" + std::to_string(expected_weight_nbit) +
        ", but got packed_weights with weight_nbit=" +
        std::to_string(format.weight_nbit));
  }
}

} // namespace torchao::ops::groupwise_lowbit_weight_lut
