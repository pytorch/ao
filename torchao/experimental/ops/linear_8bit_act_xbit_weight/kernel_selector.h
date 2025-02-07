// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cpuinfo.h>
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

namespace torchao::ops::linear_8bit_act_xbit_weight {

struct UniversalPackedWeightsFormat {
  int weight_nbit;
  bool has_weight_zeros;
  bool has_bias;
  int nr;
  int kr;

  static UniversalPackedWeightsFormat from_packed_weights_format(torchao::ops::PackedWeightsFormat format) {
    if (format.type != torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal) {
      throw std::runtime_error("Packed weights are not in universal packing format.");
    }
    return UniversalPackedWeightsFormat{
      format.params[0],
      static_cast<bool>(format.params[1]),
      static_cast<bool>(format.params[2]),
      format.params[3],
      format.params[4],
    };
  }
  inline torchao::ops::PackedWeightsFormat to_packed_weights_format() const {
    return torchao::ops::PackedWeightsFormat(
      torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal,
        {
          weight_nbit,
          has_weight_zeros,
          has_bias,
          nr,
          kr
        });
  } 
};

struct KleidiAIPackedWeightsFormat {
  int weight_nbit;
  bool has_weight_zeros;
  bool has_bias;
  int nr;
  int kr;
  int sr;

  static KleidiAIPackedWeightsFormat from_packed_weights_format(torchao::ops::PackedWeightsFormat format) {
    if (format.type != torchao::ops::PackedWeightsType::kleidi_ai) {
      throw std::runtime_error("Packed weights are not in kleidi_ai packing format.");
    }
    return KleidiAIPackedWeightsFormat{
        format.params[0],
        static_cast<bool>(format.params[1]),
        static_cast<bool>(format.params[2]),
        format.params[3],
        format.params[4],
        format.params[5]
      };
    }
    inline torchao::ops::PackedWeightsFormat to_packed_weights_format() const {
      return torchao::ops::PackedWeightsFormat(
        torchao::ops::PackedWeightsType::kleidi_ai,
          {weight_nbit,
          has_weight_zeros,
          has_bias,
          nr,
          kr,
          sr});
  } 
};

struct UKernelConfigRegistrationTable {
  private:
    using Key = std::pair<torchao::ops::PackedWeightsFormat, cpuinfo_uarch>;
    struct KeyHasher {
      std::size_t operator()(const Key& k) const {
        return std::hash<torchao::ops::PackedWeightsFormat>()(k.first) ^ std::hash<int>()(static_cast<int>(k.second));
    }
    };
    std::unordered_map<Key, torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig, KeyHasher> registration_table_;
  public:
    void register_ukernel_config(torchao::ops::PackedWeightsFormat format, cpuinfo_uarch uarch, torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig config) {
      auto key = std::make_pair(format, uarch);
      if (registration_table_.find(key) != registration_table_.end()) {
        throw std::runtime_error("UKernelConfig is already registered for this format");
      }
      registration_table_[key] = config;
    }
    std::optional<torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig> get_ukernel_config(torchao::ops::PackedWeightsFormat format, cpuinfo_uarch uarch) const {
      auto key = std::make_pair(format, uarch);
      auto it = registration_table_.find(key);
      if (it == registration_table_.end()) {
        return std::nullopt;
      }
      return it->second;
    }
};

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void register_ukernel_config_universal(UKernelConfigRegistrationTable& table, torchao::ops::PackedWeightsFormat format, cpuinfo_uarch uarch) {
  std::cout << "Calling register_ukernel_config_universal" << std::endl; // TODO: remove
  
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  auto universal_format = UniversalPackedWeightsFormat::from_packed_weights_format(format);
  if (universal_format.weight_nbit != weight_nbit) {
    throw std::runtime_error(
      "Kernel expects weight_nbit=" + std::to_string(weight_nbit) + 
      ", but packed_weights have weight_nbit=" + std::to_string(universal_format.weight_nbit)
    );
  }
  if (universal_format.has_weight_zeros != has_weight_zeros) {
    throw std::runtime_error(
      "Kernel expects has_weight_zeros=" + std::to_string(has_weight_zeros) + 
      ", but packed_weights have has_weight_zeros=" + std::to_string(universal_format.has_weight_zeros)
    );
  }
  if (universal_format.has_bias != has_bias) {
   throw std::runtime_error(
      "Kernel expects has_bias=" + std::to_string(has_bias) + 
      ", but packed_weights have has_bias=" + std::to_string(universal_format.has_bias)
    );
  }

  if (universal_format.nr == 8 && universal_format.kr == 16) {
    #if defined(__aarch64__) || defined(__ARM_NEON)
    if (cpuinfo_has_arm_neon_dot()) {
      namespace kernel = torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot;
      table.register_ukernel_config(
        format,
        uarch,
        torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig{
          /*preferred_alignment*/16,
          /*weight_packing*/
          {
            /*nr*/8,
            /*weight_data_size_fn*/&kernel::weight_data_size<weight_nbit, has_weight_zeros, has_bias>,
            /*prepare_weight_data_fn*/&kernel::prepare_weight_data<weight_nbit, has_weight_zeros, has_bias>
          },
          /*kernels*/
          {{
          {
          /*mr*/1,
          /*activation_data_size_fn*/&kernel::activation_data_size<has_weight_zeros>,
          /*prepare_activation_data_fn*/&kernel::prepare_activation_data<has_weight_zeros>,
          /*kernel*/&kernel::kernel<weight_nbit, has_weight_zeros, has_bias, has_clamp>
          }
          }}
        }
      );
      return;
    }
    #endif // defined(__aarch64__) || defined(__ARM_NEON)
  }
}

template <int weight_nbit, bool has_weight_zeros>
void register_ukernel_config_kleidi_ai(UKernelConfigRegistrationTable& table, torchao::ops::PackedWeightsFormat format, cpuinfo_uarch uarch) {
#ifdef TORCHAO_ENABLE_KLEIDI    
  std::cout << "Calling register_ukernel_config_kleidi_ai" << std::endl; // TODO: remove

  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  auto kleidi_ai_format = KleidiAIPackedWeightsFormat::from_packed_weights_format(format);
  int nr = kleidi_ai_format.nr;
  int kr = kleidi_ai_format.kr;
  int sr = kleidi_ai_format.sr;
  if (kleidi_ai_format.weight_nbit != weight_nbit) {
    throw std::runtime_error(
      "Kernel expects weight_nbit=" + std::to_string(weight_nbit) + 
      ", but packed_weights have weight_nbit=" + std::to_string(kleidi_ai_format.weight_nbit)
    );
  }
  if (kleidi_ai_format.has_weight_zeros != has_weight_zeros) {
    throw std::runtime_error(
      "Kernel expects has_weight_zeros=" + std::to_string(has_weight_zeros) + 
      ", but packed_weights have has_weight_zeros=" + std::to_string(kleidi_ai_format.has_weight_zeros)
    );
  }
  if (kleidi_ai_format.has_bias != true) {
   throw std::runtime_error(
      "Kernel expects has_bias=true, but packed_weights have has_bias=" + std::to_string(kleidi_ai_format.has_bias)
    );
  }

  if (nr == 8 && kr == 16 && sr == 2) {
    #if defined (TORCHAO_ENABLE_ARM_I8MM)
    if (cpuinfo_has_arm_i8mm()) {
        namespace kernel = torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_i8mm_4x8x32;
        auto uk = kernel::get_ukernel();
        assert (nr == uk.get_nr());
        assert (kr == uk.get_kr());
        assert (sr == uk.get_sr());
        table.register_ukernel_config(
          format,
          uarch,
          torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig{
            /*preferred_alignment*/kernel::get_preferred_alignement(),
            /*weight_packing*/
            {
              /*nr*/static_cast<int>(uk.get_n_step()),
              /*weight_data_size_fn*/&kernel::weight_data_size,
              /*prepare_weight_data_fn*/&kernel::prepare_weight_data
            },
            /*kernels*/
            {{
            {
            /*mr*/static_cast<int>(uk.get_m_step()),
            /*activation_data_size_fn*/&kernel::activation_data_size,
            /*prepare_activation_data_fn*/&kernel::prepare_activation_data,
            /*kernel*/&kernel::kernel
            }
           }}
          }
        );
        return;
    }
    #endif // TORCHAO_ENABLE_ARM_I8MM

    if (cpuinfo_has_arm_neon_dot()) {
      namespace kernel = torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x8x32;
      auto uk = kernel::get_ukernel();
      assert (nr == uk.get_nr());
      assert (kr == uk.get_kr());
      assert (sr == uk.get_sr());
      table.register_ukernel_config(
        format,
        uarch,
        torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig{
          /*preferred_alignment*/kernel::get_preferred_alignement(),
          /*weight_packing*/
          {
          /*nr*/static_cast<int>(uk.get_n_step()),
          /*weight_data_size_fn*/&kernel::weight_data_size,
          /*prepare_weight_data_fn*/&kernel::prepare_weight_data
          },
          /*kernels*/
          {{
            {
            /*mr*/static_cast<int>(uk.get_m_step()),
            /*activation_data_size_fn*/&kernel::activation_data_size,
            /*prepare_activation_data_fn*/&kernel::prepare_activation_data,
            /*kernel*/&kernel::kernel
            }
          }}
        }
      );
      return;
    }
  }

  if (nr == 4 && kr == 16 && sr == 2) {
    if (cpuinfo_has_arm_neon_dot()) {
      namespace kernel = torchao::kernels::cpu::aarch64::kleidi::kai_matmul_clamp_f32_qai8dxp_qsi4c32p::neon_dotprod_1x4x32;
      auto uk = kernel::get_ukernel();
      assert (nr == uk.get_nr());
      assert (kr == uk.get_kr());
      assert (sr == uk.get_sr());
      table.register_ukernel_config(
        format,
        uarch,
        torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig{
          /*preferred_alignment*/kernel::get_preferred_alignement(),
          /*weight_packing*/
          {
          /*nr*/static_cast<int>(uk.get_n_step()),
          /*weight_data_size_fn*/&kernel::weight_data_size,
          /*prepare_weight_data_fn*/&kernel::prepare_weight_data
          },
          /*kernels*/
          {{
            {
            /*mr*/static_cast<int>(uk.get_m_step()),
            /*activation_data_size_fn*/&kernel::activation_data_size,
            /*prepare_activation_data_fn*/&kernel::prepare_activation_data,
            /*kernel*/&kernel::kernel
            }
          }}
        }
      );
      return;
    }
  }
#endif // TORCHAO_ENABLE_KLEIDI
}


template <int weight_nbit, bool has_weight_zeros>
void register_ukernel_config(UKernelConfigRegistrationTable& table, torchao::ops::PackedWeightsFormat format, cpuinfo_uarch uarch) { 
  switch (format.type) {
    case torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal: {
      auto universal_format = UniversalPackedWeightsFormat::from_packed_weights_format(format);
      if (universal_format.has_bias) {
        register_ukernel_config_universal<weight_nbit, has_weight_zeros, /*has_bias*/ true, /*has_clamp*/false>(table, format, uarch);
      } else {
        register_ukernel_config_universal<weight_nbit, has_weight_zeros, /*has_bias*/ false, /*has_clamp*/false>(table, format, uarch);
      }
      break;
    }
    case torchao::ops::PackedWeightsType::kleidi_ai: {
      register_ukernel_config_kleidi_ai<weight_nbit, has_weight_zeros>(table, format, uarch);
      break;
    }
    default:
      throw std::runtime_error("No registration available for packed_weights_type=" + std::to_string(static_cast<int>(format.type)));
  }

  auto config = table.get_ukernel_config(format, uarch);
  if (!config.has_value()) {
    throw std::runtime_error("ukernel_config did not register");
  }
}


template <int weight_nbit, bool has_weight_zeros>
torchao::ops::linear_8bit_act_xbit_weight::UKernelConfig select_ukernel_config(torchao::ops::PackedWeightsFormat format) {
  static UKernelConfigRegistrationTable table;

  // In future, we can populate this with the current thread's uarch
  // That will require that select_ukernel_config be called in the lambda
  // instead of before it on the main thread
  // Note, cpuinfo_get_current_core() is not currently implemeted outside of linux
  // XNNPACK often uses non-core specific logic like cpuinfo_get_core(0)->uarch in configs
  auto uarch = cpuinfo_uarch_unknown;
  auto ukernel = table.get_ukernel_config(format, uarch);
  if (ukernel.has_value()) {
    std::cout << "Found ukernel_config in cache" << std::endl;  // TODO: remove cout
    return ukernel.value();
  }

  std::cout << "Registering ukernel config" << std::endl;  // TODO: remove cout
  register_ukernel_config<weight_nbit, has_weight_zeros>(table, format, uarch);

  ukernel = table.get_ukernel_config(format, uarch);
  assert(ukernel.has_value());
  return ukernel.value();
}


template <int weight_nbit, bool has_weight_zeros, bool has_bias>
torchao::ops::PackedWeightsFormat select_packed_weights_format(std::optional<std::string> target = std::nullopt) {
  // Select KleidiAI format
  #if defined(TORCHAO_ENABLE_KLEIDI)
  if (!target || *target == "kleidi_ai") {
    if constexpr (weight_nbit == 4 && (!has_weight_zeros)) { // TODO: add has_bias here
      return KleidiAIPackedWeightsFormat({weight_nbit, has_weight_zeros, /*has_bias*/true, /*nr*/8, /*kr*/16, /*sr*/2}).to_packed_weights_format();
    }
  }
  #endif // defined(TORCHAO_ENABLE_KLEIDI)
  
  // Select universal format
  if (!target || *target == "universal") {
    return UniversalPackedWeightsFormat({weight_nbit, has_weight_zeros, has_bias, /*nr*/8, /*kr*/16}).to_packed_weights_format();
  }

  throw std::runtime_error("No packed_weights_format was selected");
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
