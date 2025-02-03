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

#if defined(TORCHAO_ENABLE_KLEIDI)
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h>
#if defined (TORCHAO_ENABLE_ARM_I8MM)
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h>
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h>
#endif  // TORCHAO_ENABLE_ARM_I8MM
#endif // TORCHAO_ENABLE_KLEIDI


// #if defined(TORCHAO_ENABLE_KLEIDI)

// enum kai_kernel_id {
//     dotprod_1x4x32 = 0,
//     dotprod_1x8x32,
//     i8mm_4x8x32,
//     i8mm_8x4x32
// };

// #define KAI_GEN_UKERNEL(kernel_ns)                                                       \
//         namespace kernel = kernel_ns;                                                    \
//         auto uk = kernel::get_ukernel();                                                 \
//         config.mr = uk.get_m_step();                                                     \
//         config.nr = uk.get_n_step();                                                     \
//         config.activation_data_size_fn = &kernel::activation_data_size;                  \
//         config.weight_data_size_fn = &kernel::weight_data_size;                          \
//         config.preferred_activation_data_alignment = kernel::get_preferred_alignement(); \
//         config.preferred_weight_data_alignment = kernel::get_preferred_alignement();     \
//         config.prepare_activation_data_fn = &kernel::prepare_activation_data;            \
//         config.prepare_weight_data_fn = &kernel::prepare_weight_data;                    \
//         config.kernel_fn = &kernel::kernel;                                              \

// template <kai_kernel_id kernel_id>
// UKernelConfig get_ukernel_config_kleidi() {
//     UKernelConfig config;
// #if defined (TORCHAO_ENABLE_ARM_I8MM)
//     if constexpr (kernel_id == i8mm_4x8x32) {
//         KAI_GEN_UKERNEL(torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_i8mm_4x8x32);
//         return config;
//     }
//     if constexpr (kernel_id == i8mm_8x4x32) {
//         KAI_GEN_UKERNEL(torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_i8mm_8x4x32);
//         return config;
//     }
// #endif // TORCHAO_ENABLE_ARM_I8MM
//     if constexpr (kernel_id == dotprod_1x8x32) {
//         KAI_GEN_UKERNEL(torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x8x32);
//         return config;
//     }
//     KAI_GEN_UKERNEL(torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x4x32);
//     return config;
// }

// #endif // TORCHAO_ENABLE_KLEIDI



namespace torchao::ops::linear_8bit_act_xbit_weight {

namespace {
using UKernelConfigCacheKey = torchao::ops::PackedWeightsFormat;
using UKernelConfigCacheType = std::unordered_map<UKernelConfigCacheKey, torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig>;
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void register_ukernel_config_universal(UKernelConfigCacheType& ukernel_config_cache, int nr, int kr, int version) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  UKernelConfigCacheKey key = torchao::ops::linear_8bit_act_xbit_weight::get_packed_weights_format_universal(weight_nbit, has_weight_zeros, has_bias, nr, kr);

  if (cpuinfo_has_arm_neon_dot()) {
    if (nr == 8 && kr == 16) {
      namespace kernel = torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot;
  ukernel_config_cache[key] = torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig{
    &kernel::activation_data_size<has_weight_zeros>,
    /*preferred_activation_data_alignment*/16,
    &kernel::prepare_activation_data<has_weight_zeros>,
    &kernel::weight_data_size<weight_nbit, has_weight_zeros, has_bias>,
    /*preferred_weight_data_alignment*/16,
    &kernel::prepare_weight_data<weight_nbit, has_weight_zeros, has_bias>,
    /*nr*/8,
    {{{/*mr*/1, &kernel::kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>}}}
  };
  return;
  }
  }

  throw std::runtime_error("Cannot register ukernel_config for packing format ukernel because no implementation is available on this platform");
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias>
void register_ukernel_config_kleidi_ai(UKernelConfigCacheType& ukernel_config_cache, int nr, int kr, int sr) {
  std::cout << "register_ukernel_config_kleidi_ai" << std::endl;
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  // TODO: make better
  UKernelConfigCacheKey key = torchao::ops::linear_8bit_act_xbit_weight::get_packed_weights_format_kleidi_ai(weight_nbit, has_weight_zeros, has_bias, nr, kr, sr);

  #if defined (TORCHAO_ENABLE_ARM_I8MM)
  if (cpuinfo_has_arm_i8mm()) {
    namespace kernel = torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_i8mm_4x8x32;                                                    \
        auto uk = kernel::get_ukernel();
        ukernel_config_cache[key] = torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig{
          &kernel::activation_data_size,
          kernel::get_preferred_alignement(),
          &kernel::prepare_activation_data,
          &kernel::weight_data_size,
          kernel::get_preferred_alignement(),
          &kernel::prepare_weight_data,
          /*nr*/static_cast<int>(uk.get_n_step()),
          {{{/*mr*/static_cast<int>(uk.get_m_step()), &kernel::kernel}}}
        };
    return;
  }
  #endif // TORCHAO_ENABLE_ARM_I8MM


  if (cpuinfo_has_arm_neon_dot()) {
        if (nr == 8 && kr == 16 && sr == 2) {
          namespace kernel = torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x8x32;
          auto uk = kernel::get_ukernel();
          assert (nr == uk.get_nr());
          assert (kr == uk.get_kr());
          assert (sr == uk.get_sr());
          ukernel_config_cache[key] = torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig{
            &kernel::activation_data_size,
            kernel::get_preferred_alignement(),
            &kernel::prepare_activation_data,
            &kernel::weight_data_size,
            kernel::get_preferred_alignement(),
            &kernel::prepare_weight_data,
            /*nr*/static_cast<int>(uk.get_n_step()),
            {{{/*mr*/static_cast<int>(uk.get_m_step()), &kernel::kernel}}}
          };
          return;
        }

        if (nr == 4 && kr == 8 && sr == 2) {
          namespace kernel = torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x4x32;
          auto uk = kernel::get_ukernel();
          assert (nr == uk.get_nr());
          assert (kr == uk.get_kr());
          assert (sr == uk.get_sr());
          ukernel_config_cache[key] = torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig{
            &kernel::activation_data_size,
            kernel::get_preferred_alignement(),
            &kernel::prepare_activation_data,
            &kernel::weight_data_size,
            kernel::get_preferred_alignement(),
            &kernel::prepare_weight_data,
            /*nr*/static_cast<int>(uk.get_n_step()),
            {{{/*mr*/static_cast<int>(uk.get_m_step()), &kernel::kernel}}}
          };
          return;
        }
  }


throw std::runtime_error("Cannot register ukernel_config for packing format kleidi_ai because no implementation is available on this platform");
}


template <int weight_nbit, bool has_weight_zeros>
void register_ukernel_config(UKernelConfigCacheType& ukernel_config_cache, torchao::ops::PackedWeightsFormat format) {
  auto it = ukernel_config_cache.find(format);
  if (it != ukernel_config_cache.end()) {
    throw std::runtime_error("UKernel config already registered");
  }
  
  switch (format.type) {
    case torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal: {
      auto packing_params = torchao::ops::linear_8bit_act_xbit_weight::get_universal_packing_params(format);
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
    case torchao::ops::PackedWeightsType::kleidi_ai: {
      auto packing_params = torchao::ops::linear_8bit_act_xbit_weight::get_kleidi_ai_packing_params(format);
      assert (packing_params.has_bias == true);
      register_ukernel_config_kleidi_ai<weight_nbit, has_weight_zeros, /*has_bias*/true>(ukernel_config_cache, packing_params.nr, packing_params.kr, packing_params.sr);
      break;
    }
    default:
      throw std::runtime_error("No implementation for packed weights format");
  }

  it = ukernel_config_cache.find(format);
  if (it == ukernel_config_cache.end()) {
    throw std::runtime_error("UKernel config did not register");
  }
}


template <int weight_nbit, bool has_weight_zeros>
torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig select_ukernel_config(torchao::ops::PackedWeightsFormat format) {
  static UKernelConfigCacheType ukernel_config_cache;

  // Check cache
  auto it = ukernel_config_cache.find(format);
  if (it != ukernel_config_cache.end()) {
    std::cout << "UKERNEL CONFIG FROM CACHE: " << std::endl;
    return it->second;
  }

  std::cout << "REGISTERING UKERNEL CONFIG: " << std::endl;
  register_ukernel_config<weight_nbit, has_weight_zeros>(ukernel_config_cache, format);
  it = ukernel_config_cache.find(format);
  assert(it != ukernel_config_cache.end());  
  auto config = it->second;
  return config;
}

// TODO: make packing format and format separate concepts
// format is a serialized packing format
template <int weight_nbit, bool has_weight_zeros, bool has_bias>
torchao::ops::PackedWeightsFormat select_packed_weights_format(std::optional<std::string> target = std::nullopt) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  // Select KleidiAI format
  #if defined(TORCHAO_ENABLE_KLEIDI)
  if (!target || *target == "kleidi_ai") {
    if (weight_nbit == 4 && !has_weight_zeros) {
      return torchao::ops::linear_8bit_act_xbit_weight::get_packed_weights_format_kleidi_ai(weight_nbit, has_weight_zeros, /*has_bias*/true, /*nr*/8, /*kr*/16, /*sr*/2);
    }
  }
  #endif // defined(TORCHAO_ENABLE_KLEIDI)
  
  // Select universal format
  if (!target || *target == "universal") {
    if (cpuinfo_has_arm_neon_dot()) {
      return torchao::ops::linear_8bit_act_xbit_weight::get_packed_weights_format_universal(weight_nbit, has_weight_zeros, has_bias, /*nr*/8, /*kr*/16, /*version*/1);
    }
  }

  throw std::runtime_error("No format was selected");
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
