// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cpuinfo.h>
#include <torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h>
#include <torchao/experimental/ops/packed_weights_header.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <torchao/experimental/kernels/cpu/aarch64/linear/linear.h>
#endif // defined(__aarch64__) || defined(__ARM_NEON)

#include <optional>
#include <string>
#include <unordered_map>

#if defined(TORCHAO_ENABLE_KLEIDI)
#include <torchao/experimental/kernels/cpu/aarch64/kleidi/kai_matmul_clamp_f32_qai8dxp_qsi4c32p.h>
#endif // TORCHAO_ENABLE_KLEIDI

namespace torchao::ops::linear_8bit_act_xbit_weight {

struct PackedWeightsFormat {
  torchao::ops::PackedWeightsType type;
  int weight_nbit;
  bool has_weight_zeros;
  bool has_bias;
  int nr;
  int kr;
  int sr;

  PackedWeightsFormat(torchao::ops::PackedWeightsType type, int weight_nbit,
                      bool has_weight_zeros, bool has_bias, int nr, int kr,
                      int sr)
      : type{type}, weight_nbit{weight_nbit},
        has_weight_zeros{has_weight_zeros}, has_bias{has_bias}, nr{nr}, kr{kr},
        sr{sr} {}

  static PackedWeightsFormat
  from_packed_weights_header(torchao::ops::PackedWeightsHeader header) {
    return PackedWeightsFormat(
        header.type, header.params[0], static_cast<bool>(header.params[1]),
        static_cast<bool>(header.params[2]), header.params[3], header.params[4],
        header.params[5]);
  }

  inline torchao::ops::PackedWeightsHeader to_packed_weights_header() const {
    return torchao::ops::PackedWeightsHeader(
        type, {weight_nbit, has_weight_zeros, has_bias, nr, kr, sr});
  }
};

struct UKernelConfigRegistrationTable {
private:
  using Key = std::pair<torchao::ops::PackedWeightsHeader, cpuinfo_uarch>;
  struct KeyHasher {
    std::size_t operator()(const Key &k) const {
      return std::hash<torchao::ops::PackedWeightsHeader>()(k.first) ^
             std::hash<int>()(static_cast<int>(k.second));
    }
  };
  std::unordered_map<Key, UKernelConfig, KeyHasher> registration_table_;
  inline Key make_key(torchao::ops::PackedWeightsHeader header,
                      cpuinfo_uarch uarch) const {
    return std::make_pair(header, uarch);
  }

public:
  void register_ukernel_config(PackedWeightsFormat format, cpuinfo_uarch uarch,
                               UKernelConfig config) {
    auto header = format.to_packed_weights_header();
    auto key = make_key(header, uarch);
    if (registration_table_.find(key) != registration_table_.end()) {
      throw std::runtime_error(
          "UKernelConfig is already registered for this format");
    }
    registration_table_[key] = config;
  }
  std::optional<UKernelConfig>
  get_ukernel_config(torchao::ops::PackedWeightsHeader header,
                     cpuinfo_uarch uarch) const {
    auto key = make_key(header, uarch);
    auto it = registration_table_.find(key);
    if (it == registration_table_.end()) {
      return std::nullopt;
    }
    return it->second;
  }
};

template <int weight_nbit, bool has_weight_zeros, bool has_bias>
void check_format(PackedWeightsFormat format,
                  torchao::ops::PackedWeightsType type) {
  if (format.type != type) {
    throw std::runtime_error("Kernel expects packed_weights type=" +
                             std::to_string(static_cast<int>(type)) +
                             ", but got packed_weights with type=" +
                             std::to_string(static_cast<int>(format.type)));
  }
  if (format.weight_nbit != weight_nbit) {
    throw std::runtime_error(
        "Kernel expects weight_nbit=" + std::to_string(weight_nbit) +
        ", but got packed_weights with weight_nbit=" +
        std::to_string(format.weight_nbit));
  }
  if (format.has_weight_zeros != has_weight_zeros) {
    throw std::runtime_error(
        "Kernel expects has_weight_zeros=" + std::to_string(has_weight_zeros) +
        ", but got packed_weights with has_weight_zeros=" +
        std::to_string(format.has_weight_zeros));
  }
  if (format.has_bias != has_bias) {
    throw std::runtime_error(
        "Kernel expects has_bias=" + std::to_string(has_bias) +
        ", but got packed_weights with has_bias=" +
        std::to_string(format.has_bias));
  }
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias, bool has_clamp>
void register_ukernel_config_universal(UKernelConfigRegistrationTable &table,
                                       PackedWeightsFormat format,
                                       cpuinfo_uarch uarch) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  check_format<weight_nbit, has_weight_zeros, has_bias>(
      format,
      torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal);

  if (format.nr == 8 && format.kr == 16 && format.sr == 2) {
#if defined(__aarch64__) || defined(__ARM_NEON)
    if (cpuinfo_has_arm_neon_dot()) {
      namespace kernel = torchao::kernels::cpu::aarch64::linear::
          channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot;
      table.register_ukernel_config(
          format, uarch,
          UKernelConfig{
              /*preferred_alignment*/ 16,
              /*nr*/ 8,
              /*weight_packing_config*/
              {/*weight_data_size_fn*/
               &kernel::weight_data_size<weight_nbit, has_weight_zeros,
                                         has_bias>,
               /*prepare_weight_data_fn*/
               &kernel::prepare_weight_data<weight_nbit, has_weight_zeros,
                                            has_bias>},
              /*linear_configs*/
              {{{/*mr*/ 1,
                 /*activation_data_size_fn*/
                 &kernel::activation_data_size<has_weight_zeros>,
                 /*prepare_activation_data_fn*/
                 &kernel::prepare_activation_data<has_weight_zeros>,
                 /*kernel*/
                 &kernel::kernel<weight_nbit, has_weight_zeros, has_bias,
                                 has_clamp>}}}});
      return;
    }
#endif // defined(__aarch64__) || defined(__ARM_NEON)
  }
}

#if defined(TORCHAO_ENABLE_KLEIDI)
template <typename kernel_struct, int m_step, int mr, int n_step, int nr,
          int kr, int sr>
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

template <int weight_nbit, bool has_weight_zeros>
void register_ukernel_config_kleidi(UKernelConfigRegistrationTable &table,
                                    PackedWeightsFormat format,
                                    cpuinfo_uarch uarch) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  check_format<weight_nbit, has_weight_zeros, /*has_bias*/ true>(
      format, torchao::ops::PackedWeightsType::kleidi_ai);
  namespace op = torchao::kernels::cpu::aarch64::kleidi::
      kai_matmul_clamp_f32_qai8dxp_qsi4c32p;

  if (format.nr == 8 && format.kr == 16 && format.sr == 2) {
    constexpr int nr = 8;
    constexpr int kr = 16;
    constexpr int sr = 2;
#if defined(TORCHAO_ENABLE_ARM_I8MM)
    if (cpuinfo_has_arm_i8mm()) {
      constexpr int n_step = 8;
      table.register_ukernel_config(
          format, uarch,
          UKernelConfig{
              /*preferred_alignment*/ op::get_preferred_alignement(),
              /*nr*/ n_step,
              /*weight_packing_config*/
              get_weight_packing_config_kleidi<nr, kr, sr>(),
              /*linear_configs*/
              {{get_linear_config_kleidi<
                  op::matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
                  /*m_step*/ 4, /*mr*/ 4, n_step, nr, kr, sr>()}}});
      return;
    }
#endif // TORCHAO_ENABLE_ARM_I8MM

    if (cpuinfo_has_arm_neon_dot()) {
      constexpr int n_step = 8;
      table.register_ukernel_config(
          format, uarch,
          UKernelConfig{
              /*preferred_alignment*/ op::get_preferred_alignement(),
              /*nr*/ n_step,
              /*weight_packing_config*/
              get_weight_packing_config_kleidi<nr, kr, sr>(),
              /*linear_configs*/
              {{get_linear_config_kleidi<
                  op::matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
                  /*m_step*/ 1, /*mr*/ 1, n_step, nr, kr, sr>()}}});
      return;
    }
  }

  if (format.nr == 4 && format.kr == 16 && format.sr == 2) {
    constexpr int nr = 4;
    constexpr int kr = 16;
    constexpr int sr = 2;
    if (cpuinfo_has_arm_neon_dot()) {
      constexpr int n_step = 4;
      table.register_ukernel_config(
          format, uarch,
          UKernelConfig{
              /*preferred_alignment*/ op::get_preferred_alignement(),
              /*nr*/ n_step,
              /*weight_packing_config*/
              get_weight_packing_config_kleidi<nr, kr, sr>(),
              /*linear_configs*/
              {{get_linear_config_kleidi<
                  op::matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
                  /*m_step*/ 1, /*mr*/ 1, n_step, nr, kr, sr>()}}});
      return;
    }
  }
}
#endif // TORCHAO_ENABLE_KLEIDI

template <int weight_nbit, bool has_weight_zeros>
void register_ukernel_config(UKernelConfigRegistrationTable &table,
                             PackedWeightsFormat format, cpuinfo_uarch uarch) {
  switch (format.type) {
  case torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal: {
    if (format.has_bias) {
      register_ukernel_config_universal<weight_nbit, has_weight_zeros,
                                        /*has_bias*/ true, /*has_clamp*/ false>(
          table, format, uarch);
    } else {
      register_ukernel_config_universal<weight_nbit, has_weight_zeros,
                                        /*has_bias*/ false,
                                        /*has_clamp*/ false>(table, format,
                                                             uarch);
    }
    break;
  }
  case torchao::ops::PackedWeightsType::kleidi_ai: {
#ifdef TORCHAO_ENABLE_KLEIDI
    register_ukernel_config_kleidi<weight_nbit, has_weight_zeros>(table, format,
                                                                  uarch);
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
template <int weight_nbit, bool has_weight_zeros>
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
  register_ukernel_config<weight_nbit, has_weight_zeros>(table, format, uarch);

  ukernel = table.get_ukernel_config(header, uarch);
  assert(ukernel.has_value());
  return ukernel.value();
}

template <int weight_nbit, bool has_weight_zeros>
UKernelConfig select_ukernel_config(PackedWeightsFormat format) {
  return select_ukernel_config<weight_nbit, has_weight_zeros>(
      format.to_packed_weights_header());
}

template <int weight_nbit, bool has_weight_zeros, bool has_bias>
PackedWeightsFormat
select_packed_weights_format(std::optional<std::string> target = std::nullopt) {
// Select KleidiAI format
#if defined(TORCHAO_ENABLE_KLEIDI)
  if (!target || *target == "kleidi_ai") {
    if constexpr (weight_nbit == 4 &&
                  (!has_weight_zeros)) { // TODO: add has_bias here
      return PackedWeightsFormat(
          torchao::ops::PackedWeightsType::kleidi_ai, weight_nbit,
          has_weight_zeros, /*has_bias*/ true, /*nr*/ 8, /*kr*/ 16, /*sr*/ 2);
    }
  }
#endif // defined(TORCHAO_ENABLE_KLEIDI)

  // Select universal format
  if (!target || *target == "universal") {
    return PackedWeightsFormat(
        torchao::ops::PackedWeightsType::linear_8bit_act_xbit_weight_universal,
        weight_nbit, has_weight_zeros, has_bias, /*nr*/ 8, /*kr*/ 16, /*sr*/ 2);
  }

  throw std::runtime_error("No packed_weights_format was selected");
}

} // namespace torchao::ops::linear_8bit_act_xbit_weight
