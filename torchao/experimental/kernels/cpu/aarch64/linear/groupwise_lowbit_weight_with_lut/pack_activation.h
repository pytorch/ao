#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <cassert>
#include <cstddef>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::activation_packing {

inline size_t packed_activations_size(int m, int k, int MR) {
  const int m_padded = ((m + MR - 1) / MR) * MR;
  return (size_t)m_padded * k * sizeof(float);
}

template <int MR>
void pack_activations_for_kernel(
    // Output
    void* packed_activations,
    // Inputs
    int m,
    int k,
    const float* activations) {

  // --- 1. Initialization ---
  float* packed_ptr = static_cast<float*>(packed_activations);
  const int m_padded = ((m + MR - 1) / MR) * MR;

  // --- 2. Main Packing Loops ---
  for (int m_start = 0; m_start < m_padded; m_start += MR) {

    for (int k_idx = 0; k_idx < k; ++k_idx) {

      for (int m_offset = 0; m_offset < MR; ++m_offset) {

        // --- 3. Handle Padding and Copy Data ---
        const int current_m = m_start + m_offset;

        if (current_m < m) {
          *packed_ptr = activations[(size_t)current_m * k + k_idx];
        } else {
          *packed_ptr = 0.0f;
        }

        packed_ptr++;
      }
    }
  }
}

} // namespace torchao::kernels::cpu::aarch64::linear::activation_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
