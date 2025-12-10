#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut::
    activation_packing {

inline size_t packed_activations_size(int m, int k) {
  return m * k * sizeof(float);
}

template <int mr_, int kr_, int sr_>
void pack_activations(
    // Output
    float* packed_activations,
    // Inputs
    int m,
    int k,
    const float* activations) {
  static_assert(mr_ == 1);
  std::memcpy(packed_activations, activations, sizeof(float) * m * k);
}
} // namespace
  // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut::activation_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
