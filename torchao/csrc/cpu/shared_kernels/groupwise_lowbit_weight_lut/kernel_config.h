// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/csrc/cpu/shared_kernels/internal/library.h>
#include <array>
#include <cassert>
#include <vector>

namespace torchao::ops::groupwise_lowbit_weight_lut {

constexpr int kMaxConfigs = 4;

/**
 * @brief Defines the configuration for a Universal Kernel (UKernel) for the
 * groupwise low-bit LUT-based kernel.
 */
struct UKernelConfig {
  // Calculates the required size for the packed activation.
  using packed_activations_size_fn_type =
      size_t (*)(int m, int k, int mr, int kr, int sr);

  // Calculates the required size for the packed weights buffer.
  using packed_weights_size_fn_type = size_t (*)(
      int n,
      int k,
      int weight_nbit,
      int scale_group_size,
      bool has_scales,
      bool has_bias,
      int nr,
      int kr,
      int sr);

  // Packs activations into a kernel-friendly layout.
  using pack_activations_fn_type = void (*)(
      float* packed_activations,
      int m,
      int k,
      const float* activations,
      int mr,
      int kr,
      int sr);

  // Packs weights, scales, and LUTs into the target buffer.
  using pack_weights_fn_type = void (*)(
      void* packed_weights_ptr,
      const uint8_t* weight_qvals_indices,
      const float* weight_scales,
      const float* weight_luts,
      int n,
      int k,
      int scale_group_size,
      int lut_group_size,
      bool has_scales,
      bool has_bias,
      const float* bias,
      int nr,
      int kr,
      int sr);

  // Offset in packed_activation buffer for multithread.
  using packed_activations_offset_fn_type =
      size_t (*)(int m_idx, int k, int mr, int kr, int sr);

  // Offset in packed_weight buffer for multithread.
  using packed_weights_offset_fn_type = size_t (*)(
      int n_idx,
      int k,
      int weight_nbit,
      int scale_group_size,
      bool has_scales,
      bool has_bias,
      int nr,
      int kr,
      int sr);

  // The main computation kernel.
  using kernel_fn_type = void (*)(
      float* output,
      int output_m_stride,
      int m,
      int n,
      int k,
      int scale_group_size,
      int lut_group_size,
      const void* packed_weights,
      const void* packed_activations,
      float clamp_min,
      float clamp_max,
      bool has_bias,
      bool has_clamp);

  // Configuration for a single kernel.
  struct config_type {
    int m_step{0};
    int mr{0};
    packed_activations_size_fn_type packed_activations_size{nullptr};
    packed_activations_offset_fn_type packed_activations_offset{nullptr};
    pack_activations_fn_type pack_activations{nullptr};
    kernel_fn_type kernel{nullptr};
  };

  // Preferred memory alignment for buffers.
  size_t preferred_alignment{0};
  int n_step{0};
  int nr{0};
  int kr{0};
  int sr{0};
  int weight_nbit{0};
  bool has_scales{false};
  bool has_bias{false};

  packed_weights_size_fn_type packed_weights_size{nullptr};
  packed_weights_offset_fn_type packed_weights_offset{nullptr};
  pack_weights_fn_type pack_weights{nullptr};

  std::array<config_type, kMaxConfigs> configs;

  static UKernelConfig make(
      size_t preferred_alignment,
      int n_step,
      int nr,
      int kr,
      int sr,
      int weight_nbit,
      bool has_scales,
      bool has_bias,
      packed_weights_size_fn_type packed_weights_size,
      packed_weights_offset_fn_type packed_weights_offset,
      pack_weights_fn_type pack_weights,
      std::array<config_type, kMaxConfigs> configs);

  // Validation function to ensure all pointers are properly initialized.
  inline void validate() const {
    // 1. Validate Top-Level UKernelConfig Parameters
    TORCHAO_CHECK(preferred_alignment >= 1, "preferred_alignment must be >= 1");
    TORCHAO_CHECK(nr >= 1, "nr must be >= 1");
    TORCHAO_CHECK(kr >= 1, "kr must be >= 1");
    TORCHAO_CHECK(sr >= 1, "sr must be >= 1");
    TORCHAO_CHECK(weight_nbit >= 1, "weight_nbit must be >= 1");
    TORCHAO_CHECK(weight_nbit <= 4, "weight_nbit must be <= 4");
    TORCHAO_CHECK(
        packed_weights_size != nullptr,
        "packed_weights_size_fn_type must be set");
    TORCHAO_CHECK(
        packed_weights_offset != nullptr,
        "packed_weights_offset_fn_type must be set");
    TORCHAO_CHECK(pack_weights != nullptr, "pack_weights must be set");
    // 2. Validate the Array of Linear Configurations
    // At least one configuration must be defined.
    TORCHAO_CHECK(
        !configs.empty(),
        "At least one valid kernel configuration must be provided.");

    bool configs_set = true; // first linear config must be set
    for (size_t i = 0; i < configs.size(); ++i) {
      if (configs_set) {
        const auto& config = configs[i];

        TORCHAO_CHECK(
            config.packed_activations_size != nullptr,
            "config.packed_activations_size must be set");
        TORCHAO_CHECK(
            config.pack_activations != nullptr,
            "config.pack_activations must be set");
        TORCHAO_CHECK(config.kernel != nullptr, "config.kernel must be set");

        if (i > 0) {
          const auto& prev_config = configs[i - 1];
          TORCHAO_CHECK(
              prev_config.m_step > 0,
              "There cannot be a gap in configurations (m_step=0 followed by m_step>0)");
          TORCHAO_CHECK(
              prev_config.m_step < config.m_step,
              "m_step values in configs must be strictly increasing.");
        }
        if (i + 1 < configs.size()) {
          configs_set = (configs[i + 1].m_step >= 1);
        }
      }
    }
  }

  // Selects the appropriate configuration based on m.
  inline int select_config_idx(int m) const {
    assert(m >= 1);
    assert(configs[0].m_step >= 1);

    size_t i = 0;
    while (i + 1 < configs.size() && configs[i + 1].m_step >= 1 &&
           configs[i + 1].m_step <= m) {
      assert(configs[i].m_step < configs[i + 1].m_step);
      i++;
    }

    assert(i < configs.size());
    assert(configs[i].m_step >= 1);
    assert(i == 0 || configs[i].m_step <= m);
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
    bool has_scales,
    bool has_bias,
    packed_weights_size_fn_type packed_weights_size,
    packed_weights_offset_fn_type packed_weights_with_lut_offset,
    pack_weights_fn_type pack_weights,
    std::array<config_type, kMaxConfigs> configs) {
  return UKernelConfig{
      preferred_alignment,
      n_step,
      nr,
      kr,
      sr,
      weight_nbit,
      has_scales,
      has_bias,
      packed_weights_size,
      packed_weights_with_lut_offset,
      pack_weights,
      std::move(configs)};
}
} // namespace torchao::ops::groupwise_lowbit_weight_lut
