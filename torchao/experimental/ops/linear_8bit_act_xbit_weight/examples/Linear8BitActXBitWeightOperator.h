// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <torchao/experimental/ops/library.h>
#include <torchao/experimental/ops/memory.h>
#include <cassert>
#include <optional>

namespace torchao::ops::linear_8bit_act_xbit_weight {

class Linear8BitActXBitWeightOperator {
 private:
  torchao::aligned_byte_ptr packed_weight_data_{nullptr, nullptr};
  int packed_weight_data_size_{0};
  int preferred_packed_weight_data_alignment_{0};

  torchao::aligned_byte_ptr activation_data_buffer_{nullptr, nullptr};

  int m_{0};
  int n_{0};
  int k_{0};
  int group_size_{0};

  // The class does not own this data
  const int8_t* weight_qvals_{nullptr};
  const float* weight_scales_{nullptr};
  const int8_t* weight_zeros_{nullptr};

  bool initialized_{false};

  UKernelConfig ukernel_config_;
  PackWeightDataTilingParams pack_weight_tiling_params_;
  LinearTilingParams linear_tiling_params_;
  LinearTileSchedulingPolicy linear_scheduling_policy_;

 public:
  Linear8BitActXBitWeightOperator(
      UKernelConfig ukernel_config,
      int n,
      int k,
      int group_size,
      const int8_t* weight_qvals,
      const float* weight_scales,
      const int8_t* weight_zeros,
      int initial_m = 1,
      std::optional<PackWeightDataTilingParams> pack_weight_tiling_params = {},
      std::optional<LinearTilingParams> linear_tiling_params = {},
      std::optional<LinearTileSchedulingPolicy> linear_scheduling_policy = {})
      : m_{initial_m},
        n_{n},
        k_{k},
        group_size_(group_size),
        weight_qvals_{weight_qvals},
        weight_scales_{weight_scales},
        weight_zeros_{weight_zeros} {
    TORCHAO_CHECK(n_ >= 1, "n must be >= 1");
    TORCHAO_CHECK(k_ >= 1, "k must be >= 1");
    TORCHAO_CHECK(group_size_ >= 1, "group_size must be >= 1");
    TORCHAO_CHECK(m_ >= 1, "initial_m must be >= 1");

    ukernel_config_ = ukernel_config;
    if (pack_weight_tiling_params.has_value()) {
      pack_weight_tiling_params_ = pack_weight_tiling_params.value();
    } else {
      pack_weight_tiling_params_ = get_default_pack_weight_data_tiling_params(
          ukernel_config_, n_, /*target_panels_per_thread=*/1);
    }

    if (linear_tiling_params.has_value()) {
      linear_tiling_params_ = linear_tiling_params.value();
    } else {
      linear_tiling_params_ = get_default_linear_tiling_params(
          ukernel_config_, m_, n_, /*target_tiles_per_thread=*/5);
    }

    if (linear_scheduling_policy.has_value()) {
      linear_scheduling_policy_ = linear_scheduling_policy.value();
    } else {
      linear_scheduling_policy_ =
          LinearTileSchedulingPolicy::single_mc_parallel_nc;
    }
  }

  int get_m() {
    return m_;
  }
  int get_n() {
    return n_;
  }
  int get_k() {
    return k_;
  }
  int get_group_size() {
    return group_size_;
  }

  void initialize() {
    if (initialized_) {
      return;
    }

    // Pack weight data
    auto packed_weight_data_size =
        get_packed_weight_data_size(ukernel_config_, n_, k_, group_size_);
    auto preferred_packed_weight_data_alignment =
        get_preferred_packed_weight_data_alignment(ukernel_config_);

    packed_weight_data_size_ = packed_weight_data_size;
    preferred_packed_weight_data_alignment_ = preferred_packed_weight_data_alignment;
    packed_weight_data_ = torchao::make_aligned_byte_ptr(
        preferred_packed_weight_data_alignment, packed_weight_data_size);

    pack_weight_data_operator(
        ukernel_config_,
        pack_weight_tiling_params_,
        packed_weight_data_.get(),
        n_,
        k_,
        group_size_,
        weight_qvals_,
        weight_scales_,
        weight_zeros_);

    // Pre-allocate space for quantized/packed activations
    // This buffer may be resized when calling the operator if m is changed
    auto activation_data_buffer_size = get_activation_data_buffer_size(
        ukernel_config_,
        linear_tiling_params_,
        linear_scheduling_policy_,
        m_,
        k_,
        group_size_);
    auto activation_data_buffer_alignment =
        get_preferred_activation_data_buffer_alignment(ukernel_config_);
    activation_data_buffer_ = torchao::make_aligned_byte_ptr(
        activation_data_buffer_alignment, activation_data_buffer_size);

    // Mark as initialized
    initialized_ = true;
  }

  void operator()(
      float* output,
      const float* activations,
      int m,
      int k,
      const float* bias,
      float clamp_min,
      float clamp_max) {
    TORCHAO_CHECK(initialized_, "kernel is not initialized.");
    TORCHAO_CHECK(
        k == this->k_,
        "activations have incompatible size with initialized kernel.");

    // Resize activation buffer if needed
    if (m > m_) {
      m_ = m;
      auto activation_data_buffer_size = get_activation_data_buffer_size(
          ukernel_config_,
          linear_tiling_params_,
          linear_scheduling_policy_,
          m_,
          k_,
          group_size_);
      auto activation_data_buffer_alignment =
          get_preferred_activation_data_buffer_alignment(ukernel_config_);
      activation_data_buffer_ = torchao::make_aligned_byte_ptr(
          activation_data_buffer_alignment, activation_data_buffer_size);
    }

    // Run linear operator
    linear_operator(
        ukernel_config_,
        linear_tiling_params_,
        linear_scheduling_policy_,
        activation_data_buffer_.get(),
        output,
        // To support dynamic shapes, we use m from args, not m_
        // Note m_ can be larger than m
        m,
        n_,
        k_,
        group_size_,
        packed_weight_data_.get(),
        activations,
        bias,
        clamp_min,
        clamp_max);
  }
};
} // namespace
  // torchao::ops::linear_8bit_act_xbit_weight
