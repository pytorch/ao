// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/csrc/cpu/shared_kernels/internal/library.h>
#include <torchao/csrc/cpu/shared_kernels/linear_8bit_act_xbit_weight/kernel_config.h>
#include <array>
#include <cassert>

namespace torchao::ops::linear_8bit_act_xbit_weight {

constexpr int kMaxLinearConfigs = 4;
struct UKernelConfig {
  // Size of packed_activations buffer
  using packed_activations_size_fn_type = size_t (*)(
      int m,
      int k,
      int group_size,
      bool has_weight_zeros,
      int mr,
      int kr,
      int sr);

  // Offset in packed_activations buffer for a given m_idx
  // m_idx is index in unpacked activations matrix; it will be a multiple of
  // m_step
  using packed_activations_offset_fn_type = size_t (*)(
      int m_idx,
      int k,
      int group_size,
      bool has_weight_zeros,
      int mr,
      int kr,
      int sr);

  // Pack activations into packed_activations buffer
  using pack_activations_fn_type = void (*)(
      void* packed_activations,
      int m,
      int k,
      int group_size,
      const float* activations,
      bool has_weight_zeros,
      int mr,
      int kr,
      int sr);

  // Size of packed_weights buffer
  using packed_weights_size_fn_type = size_t (*)(
      int n,
      int k,
      int group_size,
      int weight_nbit,
      bool has_weight_zeros,
      bool has_bias,
      int nr,
      int kr,
      int sr);

  // Offset in packed_weights buffer for a given n_idx
  // n_inx is index in unpacked weights matrix; it will be a multiple of n_step
  using packed_weights_offset_fn_type = size_t (*)(
      int n_idx,
      int k,
      int group_size,
      int weight_nbit,
      bool has_weight_zeros,
      bool has_bias,
      int nr,
      int kr,
      int sr);

  // Pack weights into packed_weights buffer
  using pack_weights_fn_type = void (*)(
      void* packed_weights,
      int n,
      int k,
      int group_size,
      const int8_t* weight_qvals,
      const float* weight_scales,
      const int8_t* weight_zeros,
      const float* bias,
      int nr,
      int kr,
      int sr);

  // Pack weights into packed_weights buffer with int8-valued LUT
  using pack_weights_with_lut_fn_type = void (*)(
    void* packed_weights,
      int n,
      int k,
      int group_size,
      const int8_t* weight_qval_idxs,
      int n_luts,
      const int8_t* luts,
      const float* weight_scales,
      const int8_t* weight_zeros,
      const float* bias,
      int nr,
      int kr,
      int sr
    );

  // Run matmul kernel
  using kernel_fn_type = void (*)(
      float* output,
      int output_m_stride,
      int m,
      int n,
      int k,
      int group_size,
      const void* packed_weights,
      const void* packed_activations,
      float clamp_min,
      float clamp_max,
      bool has_weight_zeros,
      bool has_bias,
      bool has_clamp);

  struct linear_config_type {
    int m_step{0}; // m_idx will be a multiple of this
    int mr{0};
    packed_activations_size_fn_type packed_activations_size{nullptr};
    packed_activations_offset_fn_type packed_activations_offset{nullptr};
    pack_activations_fn_type pack_activations{nullptr};
    kernel_fn_type kernel{nullptr};
  };

  // preferred_alignment for packed_activations and packed_weights
  // Integration surfaces are not required to respect this alignment, and the
  // kernel must behave correctly no matter how buffers are aligned
  size_t preferred_alignment{0};
  int n_step{0}; // n_idx will be a multiple of this
  int nr{0};
  int kr{0};
  int sr{0};
  int weight_nbit{0};
  bool has_weight_zeros{false};
  bool has_bias{false};
  packed_weights_size_fn_type packed_weights_size{nullptr};
  packed_weights_offset_fn_type packed_weights_offset{nullptr};
  pack_weights_fn_type pack_weights{nullptr};
  pack_weights_with_lut_fn_type pack_weights_with_lut{nullptr};

  // linear_configs must be sorted in ascending m_step
  std::array<linear_config_type, kMaxLinearConfigs> linear_configs;

  static UKernelConfig make(
      size_t preferred_alignment,
      int n_step,
      int nr,
      int kr,
      int sr,
      int weight_nbit,
      bool has_weight_zeros,
      bool has_bias,
      packed_weights_size_fn_type packed_weights_size,
      packed_weights_offset_fn_type packed_weights_offset,
      pack_weights_fn_type pack_weights,
      std::array<linear_config_type, kMaxLinearConfigs> linear_configs);

  static UKernelConfig make_with_lut(
      size_t preferred_alignment,
      int n_step,
      int nr,
      int kr,
      int sr,
      int weight_nbit,
      bool has_weight_zeros,
      bool has_bias,
      packed_weights_size_fn_type packed_weights_with_lut_size,
      packed_weights_offset_fn_type packed_weights_with_lut_offset,
      pack_weights_with_lut_fn_type pack_weights_with_lut,
      std::array<linear_config_type, kMaxLinearConfigs> linear_configs);

  inline void validate() const {
    TORCHAO_CHECK(preferred_alignment >= 1, "preferred_alignment must be >= 1");
    TORCHAO_CHECK(n_step >= 1, "n_step must be >= 1");
    TORCHAO_CHECK(nr >= 1, "nr must be >= 1");
    TORCHAO_CHECK(kr >= 1, "kr must be >= 1");
    TORCHAO_CHECK(sr >= 1, "sr must be >= 1");
    TORCHAO_CHECK(weight_nbit >= 1, "weight_nbit must be >= 1");
    TORCHAO_CHECK(
        packed_weights_size != nullptr, "packed_weights_size must be set");
    TORCHAO_CHECK(
        packed_weights_offset != nullptr, "packed_weights_offset must be set");
    TORCHAO_CHECK(pack_weights != nullptr || pack_weights_with_lut != nullptr, "pack_weights or pack_weights_with_lut must be set");

    bool linear_configs_set = true; // first linear config must be set
    for (size_t i = 0; i < linear_configs.size(); i++) {
      if (linear_configs_set) {
        TORCHAO_CHECK(
            linear_configs[i].m_step >= 1,
            "linear_configs[i].m_step must be >= 1");
        TORCHAO_CHECK(
            linear_configs[i].mr >= 1, "linear_configs[i].mr must be >= 1");
        TORCHAO_CHECK(
            linear_configs[i].packed_activations_size != nullptr,
            "linear_configs[i].packed_activations_size must be set");
        TORCHAO_CHECK(
            linear_configs[i].packed_activations_offset != nullptr,
            "linear_configs[i].packed_activations_offset must be set");
        TORCHAO_CHECK(
            linear_configs[i].pack_activations != nullptr,
            "linear_configs[i].pack_activations must be set");
        TORCHAO_CHECK(
            linear_configs[i].kernel != nullptr,
            "linear_configs[i].kernel must be set");
        if (i >= 1) {
          TORCHAO_CHECK(
              linear_configs[i - 1].m_step < linear_configs[i].m_step,
              "set linear_configs must be increasing in m_step");
        }
        if (i + 1 < linear_configs.size()) {
          linear_configs_set = (linear_configs[i + 1].m_step >= 1);
        }
      }
    }
  }

  inline int select_linear_config_idx(int m) const {
    assert(m >= 1);
    assert(linear_configs[0].m_step >= 1);

    size_t i = 0;
    while (i + 1 < linear_configs.size() && linear_configs[i + 1].m_step >= 1 &&
           linear_configs[i + 1].m_step <= m) {
      assert(linear_configs[i].m_step < linear_configs[i + 1].m_step);
      i++;
    }

    assert(i < linear_configs.size());
    assert(linear_configs[i].m_step >= 1);
    assert(i == 0 || linear_configs[i].m_step <= m);
    return static_cast<int>(i);
  }
};

inline UKernelConfig UKernelConfig::make(
    size_t preferred_alignment,
    int n_step,
    int nr,
    int kr,
    int sr,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    packed_weights_size_fn_type packed_weights_size,
    packed_weights_offset_fn_type packed_weights_offset,
    pack_weights_fn_type pack_weights,
    std::array<linear_config_type, kMaxLinearConfigs> linear_configs) {
  return UKernelConfig{
      preferred_alignment,
      n_step,
      nr,
      kr,
      sr,
      weight_nbit,
      has_weight_zeros,
      has_bias,
      packed_weights_size,
      packed_weights_offset,
      pack_weights,
      /*pack_weights_with_lut*/nullptr,
      std::move(linear_configs)};
}

inline UKernelConfig UKernelConfig::make_with_lut(
    size_t preferred_alignment,
    int n_step,
    int nr,
    int kr,
    int sr,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    packed_weights_size_fn_type packed_weights_with_lut_size,
    packed_weights_offset_fn_type packed_weights_with_lut_offset,
    pack_weights_with_lut_fn_type pack_weights_with_lut,
    std::array<linear_config_type, kMaxLinearConfigs> linear_configs) {
  return UKernelConfig{
      preferred_alignment,
      n_step,
      nr,
      kr,
      sr,
      weight_nbit,
      has_weight_zeros,
      has_bias,
      packed_weights_with_lut_size,
      packed_weights_with_lut_offset,
      /*pack_weights*/nullptr,
      /*pack_weights_with_lut*/pack_weights_with_lut,
      std::move(linear_configs)};
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
