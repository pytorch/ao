// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cpuinfo.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/packed_weights_header.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#endif // defined(__aarch64__) || defined(__ARM_NEON)

#include <string>
#include <optional>
#include <unordered_map>

namespace torchao::ops::linear_8bit_act_xbit_weight {

namespace {
// If needed in future, we can add uarch to the UKernelConfigCacheKey if needed
using UKernelConfigCacheKey = torchao::ops::PackedWeightsHeader;
struct UKernelConfigCacheKeyHash {
  std::size_t operator()(const UKernelConfigCacheKey& k) const {
    std::size_t hash =  std::hash<int>()(static_cast<int>(k.format));
    for (int i = 0; i < k.params.size(); i++) {
      hash ^= std::hash<int>()(k.params[i]);
    }
    return hash;
  }
};
using UKernelConfigCacheType = std::unordered_map<UKernelConfigCacheKey, torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig, UKernelConfigCacheKeyHash>;
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void register_ukernel_config_universal(UKernelConfigCacheType& ukernel_config_cache, int nr, int kr, int version) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  UKernelConfigCacheKey key = torchao::ops::linear_8bit_act_xbit_weight::get_packed_weights_header_universal(weight_nbit, has_weight_zeros, has_bias, nr, kr);

  if (cpuinfo_has_arm_neon_dot()) {
    if (nr == 8 && kr == 16) {
  ukernel_config_cache[key] = torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig{
    &torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot::activation_data_size<has_weight_zeros>,
    /*preferred_activation_data_alignment*/16,
    &torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot::prepare_activation_data<has_weight_zeros>,
    &torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot::weight_data_size<weight_nbit, has_weight_zeros, has_bias>,
    /*preferred_weight_data_alignment*/16,
    &torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot::prepare_weight_data<weight_nbit, has_weight_zeros, has_bias>,
    /*nr*/8,
    {{{/*mr*/1, &torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot::kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>}}}
  };
  return;
  }
  }

  throw std::runtime_error("Cannot register ukernel_config for packing format ukernel because no implementation is available on this platform");
}


template <int weight_nbit, bool has_weight_zeros>
void register_ukernel_config(UKernelConfigCacheType& ukernel_config_cache, torchao::ops::PackedWeightsHeader header) {
  auto it = ukernel_config_cache.find(header);
  if (it != ukernel_config_cache.end()) {
    throw std::runtime_error("UKernel config already registered");
  }
  
  switch (header.format) {
    case torchao::ops::PackedWeightsFormat::linear_8bit_act_xbit_weight_universal: {
      auto packing_params = torchao::ops::linear_8bit_act_xbit_weight::get_universal_packing_params(header);
      if (packing_params.weight_nbit != weight_nbit) {
        throw std::runtime_error("Packed weights are not in the expected format");
      }
      if (packing_params.has_weight_zeros != has_weight_zeros) {
        throw std::runtime_error("Packed weights are not in the expected format");
      }
      if (packing_params.has_bias) {
        register_ukernel_config_universal<weight_nbit, has_weight_zeros, /*has_bias*/ true, /*has_clamp*/false>(ukernel_config_cache, packing_params.nr, packing_params.kr, packing_params.version);
      } else {
        register_ukernel_config_universal<weight_nbit, has_weight_zeros, /*has_bias*/ false, /*has_clamp*/false>(ukernel_config_cache, packing_params.nr, packing_params.kr, packing_params.version);
      }
      break;
    }
    default:
      throw std::runtime_error("No implementation for packed weights format");
  }

  it = ukernel_config_cache.find(header);
  if (it == ukernel_config_cache.end()) {
    throw std::runtime_error("UKernel config did not register");
  }
}


template <int weight_nbit, bool has_weight_zeros>
torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig select_ukernel_config(torchao::ops::PackedWeightsHeader header) {
  static UKernelConfigCacheType ukernel_config_cache;

  // Check cache
  auto it = ukernel_config_cache.find(header);
  if (it != ukernel_config_cache.end()) {
    std::cout << "UKERNEL CONFIG FROM CACHE: " << std::endl;
    return it->second;
  }

  std::cout << "REGISTERING UKERNEL CONFIG: " << std::endl;
  register_ukernel_config<weight_nbit, has_weight_zeros>(ukernel_config_cache, header);
  it = ukernel_config_cache.find(header);
  assert(it != ukernel_config_cache.end());  
  auto config = it->second;
  return config;
}

// TODO: make packing format and header separate concepts
// Header is a serialized packing format
template <int weight_nbit, bool has_weight_zeros, bool has_bias>
torchao::ops::PackedWeightsHeader select_header(std::optional<std::string> target = std::nullopt) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  // Select KleidiAI header
  if (!target || *target == "kleidi_ai") {
    if (weight_nbit == 4 && !has_weight_zeros) {
    }
  }
  
  // Select universal header
  if (!target || *target == "universal") {
    if (cpuinfo_has_arm_neon_dot()) {
      return torchao::ops::linear_8bit_act_xbit_weight::get_packed_weights_header_universal(weight_nbit, has_weight_zeros, has_bias, /*nr*/8, /*kr*/16, /*version*/1);
    }
  }

  throw std::runtime_error("No header was selected");
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
