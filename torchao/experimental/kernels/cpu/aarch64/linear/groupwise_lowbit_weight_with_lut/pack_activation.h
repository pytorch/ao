#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <cassert>
#include <cstddef>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut::activation_packing {

/**
 * @brief Calculates the total memory in bytes required for the packed activations,
 * including any necessary padding.
 *
 * @param m The number of rows in the source activation matrix.
 * @param k The number of columns in the source activation matrix.
 * @param MR The row-tiling factor of the micro-kernel (e.g., 4).
 * @return The size of the required buffer in bytes.
 */
inline size_t packed_activations_size(int m, int k, int MR) {
  // The kernel processes MR rows at a time. If 'm' is not a multiple of MR,
  // we must pad it to the next multiple to create full blocks.
  const int m_padded = ((m + MR - 1) / MR) * MR;
  return (size_t)m_padded * k * sizeof(float);
}

/**
 * @brief (Corrected) Packs the activation matrix by reordering it into a
 * transposed-block layout optimal for a micro-kernel that processes MR rows
 * at a time.
 *
 * @details
 * This function transforms a standard row-major matrix A(m, k) into a
 * blocked layout of (M/MR, K, MR). This layout ensures that the MR
 * activation values for any given column 'k' are contiguous in memory,
 * allowing the micro-kernel to load them with a single efficient vector
 * instruction (`vld1q_f32`).
 *
 * Example Transformation (MR=4):
 * Source Layout (Row-Major):
 *   Row 0: [a00, a01, a02, ...]
 *   Row 1: [a10, a11, a12, ...]
 *   Row 2: [a20, a21, a22, ...]
 *   Row 3: [a30, a31, a32, ...]
 *
 * Packed Layout (Transposed-Block):
 *   [a00, a10, a20, a30], [a01, a11, a21, a31], [a02, a12, a22, a32], ...
 *   (--- k=0 block ---), (--- k=1 block ---), (--- k=2 block ---), ...
 *
 * @tparam MR The row-tiling factor of the micro-kernel.
 * @param packed_activations Pointer to the destination buffer. Must be pre-allocated
 *        with the size given by `packed_activations_size`.
 * @param m The number of rows in the source activation matrix.
 * @param k The number of columns in the source activation matrix.
 * @param activations Pointer to the source activation matrix (float32, row-major).
 */
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
  // Iterate over the matrix in vertical chunks (tiles) of MR rows.
  for (int m_start = 0; m_start < m_padded; m_start += MR) {

    // This is the key to the transposition: iterate through columns first.
    for (int k_idx = 0; k_idx < k; ++k_idx) {

      // For each column, gather the MR values from their respective rows.
      for (int m_offset = 0; m_offset < MR; ++m_offset) {

        // --- 3. Handle Padding and Copy Data ---
        const int current_m = m_start + m_offset;

        if (current_m < m) {
          // If the current row is within the original matrix bounds, copy the value.
          *packed_ptr = activations[(size_t)current_m * k + k_idx];
        } else {
          // If we are in a padded row, write a zero.
          *packed_ptr = 0.0f;
        }

        // Advance the destination pointer to the next slot.
        packed_ptr++;
      }
    }
  }
}

} // namespace torchao::kernels::cpu::aarch64::linear::activation_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
