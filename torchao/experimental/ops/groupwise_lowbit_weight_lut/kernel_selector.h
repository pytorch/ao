// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cpuinfo.h>
#include <torchao/experimental/ops/groupwise_lowbit_weight_lut/kernel_config.h>
#include <torchao/experimental/ops/groupwise_lowbit_weight_lut/packed_weights_format.h>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#if defined(TORCHAO_BUILD_CPU_AARCH64)
#if defined(TORCHAO_ENABLE_ARM_NEON_DOT)
#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight_lut/groupwise_lowbit_weight_lut.h>
#endif // TORCHAO_ENABLE_ARM_NEON_DOT
#endif // TORCHAO_BUILD_CPU_AARCH64

namespace torchao::ops::groupwise_lowbit_weight_lut {

/**
 * @brief A thread-unsafe registration table for kernel configurations.
 *
 * This table maps a combination of a weight format (header) and a CPU
 * microarchitecture to a specific UKernelConfig.
 */
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
  // resgist a kernel config for a given format and uarch.
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
    config.validate();
    registration_table_[key] = config;
  }
  // get the kernel config for a given format and uarch.
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
  LOG(INFO) << "Registering ukernel config for groupwise_lowbit_weight_lut"
            << std::endl
            << "\tDescription: " << description << std::endl
            << "\tformat.type=" << static_cast<int>(format.type) << std::endl
            << "\tformat.weight_nbit=" << format.weight_nbit << std::endl
            << "\tformat.has_bias=" << format.has_bias << std::endl
            << "\tformat.has_scales=" << format.has_scales << std::endl
            << "\tformat.lut_group_size=" << format.lut_group_size << std::endl
            << "\tformat.scale_group_size=" << format.scale_group_size
            << "\tformat.nr=" << format.nr << std::endl
            << "\tformat.kr=" << format.kr << std::endl
            << "\tformat.sr=" << format.sr << std::endl
            << std::endl;
#endif // USE_ATEN
}

#if defined(TORCHAO_BUILD_CPU_AARCH64)
/**
 * @brief Registers all available AArch64 kernels for a given format.
 *
 * @tparam weight_nbit The bit-width of the weights.
 * @tparam has_scales Whether the packed buffer contains scale factors.
 * @param table The registration table to add the kernel config to.
 * @param format The format header describing the weights.
 * @param uarch The target CPU microarchitecture.
 */
template <int weight_nbit>
void register_ukernel_config(
    UKernelConfigRegistrationTable& table,
    PackedWeightsFormat format,
    cpuinfo_uarch uarch) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if (!cpuinfo_has_arm_v8()) {
    // This CPU doesn't support the kernel, so do nothing.
    return;
  }

  check_format(
      format,
      torchao::ops::PackedWeightsType::groupwise_lowbit_weight_lut,
      weight_nbit);
  int preferred_alignment = 16;

  namespace kernel_api =
      torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut;

  using kernel_fn_ptr_t =
      decltype(&kernel_api::kernel_lowbit_1x4x32_f32<weight_nbit, true>);
  kernel_fn_ptr_t kernel_dispatcher;

  if (format.has_scales) {
    kernel_dispatcher =
        &kernel_api::kernel_lowbit_1x4x32_f32<weight_nbit, /*has_scales=*/true>;
  } else {
    kernel_dispatcher =
        &kernel_api::
            kernel_lowbit_1x4x32_f32<weight_nbit, /*has_scales=*/false>;
  }
  if (format.nr == 4 && format.kr == 32 && format.sr == 8) {
    log_registration(format, "lut: kernel_lowbit_1x4x32_f32");
    constexpr int nr = 4;
    constexpr int kr = 32;
    constexpr int sr = 8;
    constexpr int mr = 1;
    constexpr int m_step = 1;
    constexpr int n_step = 4;

    auto uk = UKernelConfig::make(
        /*preferred_alignment=*/preferred_alignment,
        /*n_step=*/n_step,
        /*nr=*/format.nr,
        /*kr=*/format.kr,
        /*sr=*/format.sr,
        /*weight_nbit=*/format.weight_nbit,
        /*has_scales=*/format.has_scales,
        /*has_bias=*/format.has_bias,
        /*packed_weights_size_fn_type=*/
        &kernel_api::packed_weights_size<weight_nbit, nr, kr, sr>,
        /*pack_weights_fn_type=*/
        &kernel_api::
            pack_weights_for_groupwise_lut_kernel<weight_nbit, nr, kr, sr>,
        /*configs=*/{});

    uk.configs[0] = UKernelConfig::group_config_type(
        {m_step,
         mr,
         &kernel_api::packed_activations_size,
         &kernel_api::packed_activations_offset,
         &kernel_api::pack_activations<mr, kr, sr>,
         kernel_dispatcher});

    // Resgister the kernel config.
    table.register_ukernel_config(format, uarch, std::move(uk));
  }
}
#endif // TORCHAO_BUILD_CPU_AARCH64

/**
 * @brief Selects the best UKernelConfig for the given format header.
 *
 * This function is the main entry point for the op. It manages a static
 * registration table and, if a kernel is not already registered for the
 * current CPU, it will perform the registration.
 *
 * @tparam weight_nbit The bit-width of the weights.
 * @param header A header describing the packed weight format.
 * @return The appropriate UKernelConfig for the current environment.
 */
template <int weight_nbit>
UKernelConfig select_ukernel_config(torchao::ops::PackedWeightsHeader header) {
#if defined(TORCHAO_BUILD_CPU_AARCH64)
  // Static table ensures we only register kernels once per session.
  static UKernelConfigRegistrationTable table;

  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  auto uarch = cpuinfo_uarch_unknown;

  auto ukernel = table.get_ukernel_config(header, uarch);
  if (ukernel.has_value()) {
    return ukernel.value();
  }

  // Create a new format object from the header.
  auto format = PackedWeightsFormat::from_packed_weights_header(header);

  register_ukernel_config<weight_nbit>(table, format, uarch);

  ukernel = table.get_ukernel_config(header, uarch);
  assert(ukernel.has_value() && "Kernel registration failed for the current CPU microarchitecture.");
  return ukernel.value();
#else
  throw std::runtime_error(
      "select_ukernel_config for groupwise_lowbit_weight_lut is only supported "
      "when TORCHAO_BUILD_CPU_AARCH64 is defined.");
#endif
}

template <int weight_nbit>
PackedWeightsFormat select_packed_weights_format(
    std::optional<std::string> target,
    int scale_group_size,
    int lut_group_size,
    bool has_scales,
    bool has_bias) {
  if (!target) {
    return PackedWeightsFormat(
        torchao::ops::PackedWeightsType::groupwise_lowbit_weight_lut,
        weight_nbit,
        scale_group_size,
        lut_group_size,
        has_scales,
        has_bias,
        /*nr*/ 4,
        /*kr*/ 32,
        /*sr*/ 8);
  }
  throw std::runtime_error("No packed_weights_format was selected");
}

} // namespace torchao::ops::groupwise_lowbit_weight_lut
