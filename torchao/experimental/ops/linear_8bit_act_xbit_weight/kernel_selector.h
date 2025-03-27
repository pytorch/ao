// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cpuinfo.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/packed_weights_format.h>

#if defined(TORCHAO_BUILD_CPU_AARCH64)
#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#endif // TORCHAO_BUILD_CPU_AARCH64

#include <optional>
#include <string>
#include <unordered_map>

#if defined(TORCHAO_ENABLE_KLEIDI)
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp_qsi4c32p.h>
#endif // TORCHAO_ENABLE_KLEIDI

namespace torchao::ops::linear_8bit_act_xbit_weight {

struct UKernelConfigRegistrationTable {
 private:
  using Key = std::pair<torchao::ops::PackedWeightsHeader, cpuinfo_uarch>;
  struct KeyHasher {
    std::size_t operator()(const Key& k) const {
      return std::hash<torchao::ops::PackedWeightsHeader>()(k.first) ^
          std::hash<int>()(static_cast<int>(k.second));
    }
  };
  std::unordered_map<Key, UKernelConfig, KeyHasher> registration_table_;
  inline Key make_key(
      torchao::ops::PackedWeightsHeader header,
      cpuinfo_uarch uarch) const {
    return std::make_pair(header, uarch);
  }

 public:
  void register_ukernel_config(
      PackedWeightsFormat format,
      cpuinfo_uarch uarch,
      UKernelConfig config) {
    auto header = format.to_packed_weights_header();
    auto key = make_key(header, uarch);
    if (registration_table_.find(key) != registration_table_.end()) {
      throw std::runtime_error(
          "UKernelConfig is already registered for this format");
    }
    registration_table_[key] = config;
  }
  std::optional<UKernelConfig> get_ukernel_config(
      torchao::ops::PackedWeightsHeader header,
      cpuinfo_uarch uarch) const {
    auto key = make_key(header, uarch);
    auto it = registration_table_.find(key);
    if (it == registration_table_.end()) {
      return std::nullopt;
    }
    return it->second;
  }
};

void log_registration(PackedWeightsFormat format, std::string description) {
  // Logging is only supported in ATen mode
#ifdef USE_ATEN
  LOG(INFO) << "Registering ukernel config for linear_8bit_act_xbit_weight"
            << std::endl
            << "\tDescription: " << description << std::endl
            << "\tformat.type=" << static_cast<int>(format.type) << std::endl
            << "\tformat.weight_nbit=" << format.weight_nbit << std::endl
            << "\tformat.has_weight_zeros=" << format.has_weight_zeros
            << std::endl
            << "\tformat.has_bias=" << format.has_bias << std::endl
            << "\tformat.nr=" << format.nr << std::endl
            << "\tformat.kr=" << format.kr << std::endl
            << "\tformat.sr=" << format.sr << std::endl;
#endif // USE_ATEN
}

template <int weight_nbit>
void register_ukernel_config_universal(
    UKernelConfigRegistrationTable& table,
    PackedWeightsFormat format,
    cpuinfo_uarch uarch) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  check_format<weight_nbit>(
      format,
      torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal);

  if (format.nr == 8 && format.kr == 16 && format.sr == 2) {
#if defined(TORCHAO_BUILD_CPU_AARCH64)
    if (cpuinfo_has_arm_neon_dot()) {
      log_registration(format, "universal");
      namespace kernel = torchao::kernels::cpu::aarch64::linear::
          channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot;
      table.register_ukernel_config(
          format,
          uarch,
          UKernelConfig{
              /*preferred_alignment*/ 16,
              /*nr*/ 8,
              /*weight_packing_config*/
              {/*weight_data_size_fn*/
               &kernel::weight_data_size<weight_nbit>,
               /*prepare_weight_data_fn*/
               &kernel::prepare_weight_data<weight_nbit>},
              /*linear_configs*/
              {{{/*mr*/ 1,
                 /*activation_data_size_fn*/
                 &kernel::activation_data_size,
                 /*prepare_activation_data_fn*/
                 &kernel::prepare_activation_data,
                 /*kernel*/
                 &kernel::kernel<weight_nbit>}}}});
      return;
    }
#endif // TORCHAO_BUILD_CPU_AARCH64
  }
}

#if defined(TORCHAO_ENABLE_KLEIDI)
template <
    typename kernel_struct,
    int m_step,
    int mr,
    int n_step,
    int nr,
    int kr,
    int sr>
UKernelConfig::linear_config_type get_linear_config_kleidi() {
  namespace op = torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p;
  assert(m_step == kernel_struct::get_ukernel().get_m_step());
  assert(mr == kernel_struct::get_ukernel().get_mr());
  assert(n_step == kernel_struct::get_ukernel().get_n_step());
  assert(nr == kernel_struct::get_ukernel().get_nr());
  assert(kr == kernel_struct::get_ukernel().get_kr());
  assert(sr == kernel_struct::get_ukernel().get_sr());
  return UKernelConfig::linear_config_type{
      /*mr*/ m_step,
      /*activation_data_size_fn*/ &op::activation_data_size<mr, kr, sr>,
      /*prepare_activation_data_fn*/ &op::prepare_activation_data<mr, kr, sr>,
      /*kernel*/ &kernel_struct::kernel};
}

template <int nr, int kr, int sr>
UKernelConfig::weight_packing_config_type get_weight_packing_config_kleidi() {
  namespace op = torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p;
  return UKernelConfig::weight_packing_config_type(
      {/*weight_data_size_fn*/ &op::weight_data_size<nr, kr, sr>,
       /*prepare_weight_data_fn*/ &op::prepare_weight_data<nr, kr, sr>});
}

template <int weight_nbit>
void register_ukernel_config_kleidi(
    UKernelConfigRegistrationTable& table,
    PackedWeightsFormat format,
    cpuinfo_uarch uarch) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  check_format<weight_nbit>(format, torchao::ops::PackedWeightsType::kleidi_ai);
  namespace op = torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p;

  if (format.nr == 8 && format.kr == 16 && format.sr == 2) {
    constexpr int nr = 8;
    constexpr int kr = 16;
    constexpr int sr = 2;
#if defined(TORCHAO_ENABLE_ARM_I8MM)
    if (cpuinfo_has_arm_i8mm()) {
      constexpr int n_step = 8;
      log_registration(
          format,
          "kleidiai: matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm");
      table.register_ukernel_config(
          format,
          uarch,
          UKernelConfig{
              /*preferred_alignment*/ op::get_preferred_alignement(),
              /*nr*/ n_step,
              /*weight_packing_config*/
              get_weight_packing_config_kleidi<nr, kr, sr>(),
              /*linear_configs*/
              {{get_linear_config_kleidi<
                  op::matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                  /*m_step*/ 4,
                  /*mr*/ 4,
                  n_step,
                  nr,
                  kr,
                  sr>()}}});
      return;
    }
#endif // TORCHAO_ENABLE_ARM_I8MM

    if (cpuinfo_has_arm_neon_dot()) {
      constexpr int n_step = 8;
      log_registration(
          format,
          "kleidiai: matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod");
      table.register_ukernel_config(
          format,
          uarch,
          UKernelConfig{
              /*preferred_alignment*/ op::get_preferred_alignement(),
              /*nr*/ n_step,
              /*weight_packing_config*/
              get_weight_packing_config_kleidi<nr, kr, sr>(),
              /*linear_configs*/
              {{get_linear_config_kleidi<
                  op::matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                  /*m_step*/ 1,
                  /*mr*/ 1,
                  n_step,
                  nr,
                  kr,
                  sr>()}}});
      return;
    }
  }

  if (format.nr == 4 && format.kr == 16 && format.sr == 2) {
    constexpr int nr = 4;
    constexpr int kr = 16;
    constexpr int sr = 2;
    if (cpuinfo_has_arm_neon_dot()) {
      constexpr int n_step = 4;
      log_registration(
          format,
          "kleidiai: matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod");
      table.register_ukernel_config(
          format,
          uarch,
          UKernelConfig{
              /*preferred_alignment*/ op::get_preferred_alignement(),
              /*nr*/ n_step,
              /*weight_packing_config*/
              get_weight_packing_config_kleidi<nr, kr, sr>(),
              /*linear_configs*/
              {{get_linear_config_kleidi<
                  op::matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
                  /*m_step*/ 1,
                  /*mr*/ 1,
                  n_step,
                  nr,
                  kr,
                  sr>()}}});
      return;
    }
  }
}
#endif // TORCHAO_ENABLE_KLEIDI

template <int weight_nbit>
void register_ukernel_config(
    UKernelConfigRegistrationTable& table,
    PackedWeightsFormat format,
    cpuinfo_uarch uarch) {
  switch (format.type) {
    case torchao::ops::PackedWeightsType::
        linear_8bit_act_xbit_weight_universal: {
      register_ukernel_config_universal<weight_nbit>(table, format, uarch);
      break;
    }
    case torchao::ops::PackedWeightsType::kleidi_ai: {
#ifdef TORCHAO_ENABLE_KLEIDI
      register_ukernel_config_kleidi<weight_nbit>(table, format, uarch);
#endif // TORCHAO_ENABLE_KLEIDI
      break;
    }
    default:
      throw std::runtime_error(
          "No registration available for packed_weights_type=" +
          std::to_string(static_cast<int>(format.type)));
  }

  auto config =
      table.get_ukernel_config(format.to_packed_weights_header(), uarch);
  if (!config.has_value()) {
    throw std::runtime_error("ukernel_config did not register");
  }
}

// Not thread safe
template <int weight_nbit>
UKernelConfig select_ukernel_config(torchao::ops::PackedWeightsHeader header) {
  static UKernelConfigRegistrationTable table;

  // In future, we can populate this with the current thread's uarch
  // That will require that select_ukernel_config be called in the lambda
  // instead of before it on the main thread
  // Note, cpuinfo_get_current_core() is not currently implemeted outside of
  // linux XNNPACK often uses non-core specific logic like
  // cpuinfo_get_core(0)->uarch in configs
  auto uarch = cpuinfo_uarch_unknown;
  auto ukernel = table.get_ukernel_config(header, uarch);
  if (ukernel.has_value()) {
    return ukernel.value();
  }

  auto format = PackedWeightsFormat::from_packed_weights_header(header);
  register_ukernel_config<weight_nbit>(table, format, uarch);

  ukernel = table.get_ukernel_config(header, uarch);
  assert(ukernel.has_value());
  return ukernel.value();
}

template <int weight_nbit>
UKernelConfig select_ukernel_config(PackedWeightsFormat format) {
  return select_ukernel_config<weight_nbit>(format.to_packed_weights_header());
}

template <int weight_nbit>
PackedWeightsFormat select_packed_weights_format(
    std::optional<std::string> target,
    bool has_weight_zeros,
    bool has_bias) {
// Select KleidiAI format
#if defined(TORCHAO_ENABLE_KLEIDI)
  if (!target || *target == "kleidiai") {
    if (weight_nbit == 4 && (!has_weight_zeros)) {
      // KleidiAI will pack bias with weights always,
      // even if bias is not provided 0s will be packed
      return PackedWeightsFormat(
          torchao::ops::PackedWeightsType::kleidi_ai,
          weight_nbit,
          has_weight_zeros,
          /*has_bias*/ true,
          /*nr*/ 8,
          /*kr*/ 16,
          /*sr*/ 2);
    }
  }
#endif // defined(TORCHAO_ENABLE_KLEIDI)

  // Select universal format
  if (!target || *target == "universal") {
    return PackedWeightsFormat(
        torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal,
        weight_nbit,
        has_weight_zeros,
        has_bias,
        /*nr*/ 8,
        /*kr*/ 16,
        /*sr*/ 2);
  }

  throw std::runtime_error("No packed_weights_format was selected");
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
