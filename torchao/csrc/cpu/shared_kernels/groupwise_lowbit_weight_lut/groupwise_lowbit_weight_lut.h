// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torchao/csrc/cpu/shared_kernels/groupwise_lowbit_weight_lut/kernel_config.h>
#include <optional>
#include <vector>

namespace torchao::ops::groupwise_lowbit_weight_lut {

/**
 * @brief Orchestrates the packing of quantized weights into a kernel-specific
 * memory layout.
 *
 * @details This function acts as a high-level operator that parallelizes the
 * weight packing process across the N dimension. It partitions the work into
 * tiles, calculates the correct memory offsets for each tile's source and
 * destination pointers, and then invokes the low-level `pack_weights` function
 * provided by the kernel configuration (`uk`).
 *
 * @param uk The kernel configuration, providing layout details, function
 * pointers, and dimension constraints (nr, kr).
 * @param packed_weights_ptr [out] The destination buffer for the packed weight
 * data.
 * @param n The N dimension of the weight matrix (e.g., output channels).
 * @param k The K dimension of the weight matrix (e.g., input channels).
 * @param scale_group_size The group size for weight quantization scales.
 * @param lut_group_size The group size for weight lookup tables (LUTs).
 * @param weight_qval_indices [in] Pointer to the raw quantized weight indices.
 * @param weight_scales [in] Pointer to the raw weight quantization scales.
 * @param weight_luts [in] Pointer to the raw weight lookup tables.
 * @param bias [in] Pointer to the raw bias values; can be nullptr if the kernel
 * configuration indicates no bias is used.
 */
void pack_weights_operator(
    const UKernelConfig& uk,
    // Outputs
    void* packed_weights_ptr,
    // Inputs
    int n,
    int k,
    int scale_group_size,
    int lut_group_size,
    const uint8_t* weight_qval_indices,
    const float* weight_scales,
    const float* weight_luts,
    const float* bias);

struct GroupwiseTilingParams {
  int mc;
  int nc;

  /**
   * @brief Calculates groupwise tiling parameters based on a target number of
   * tiles per thread.
   *
   * @details This function implements a heuristic to determine optimal tile
   * sizes (`mc`, `nc`) for balancing a computational workload across multiple
   * threads. It calculates the number of tiles needed to cover the M dimension
   * and uses this, along with the target number of tiles per thread, to derive
   * a suitable tile count in the N dimension. This count is then scaled by
   * `n_step` to get the final `nc` value. The resulting tile sizes are clamped
   * to not exceed the original problem dimensions.
   *
   * @param m The total size of the M dimension (e.g., rows).
   * @param m_step The required step size for tiling in the M dimension.
   * @param n The total size of the N dimension (e.g., columns).
   * @param n_step The required step size for tiling in the N dimension.
   * @param target_tiles_per_thread A tuning parameter that suggests how many
   * tiles each thread should ideally process, influencing the calculated tile
   * sizes.
   * @return A `GroupwiseTilingParams` struct containing the computed `mc` and
   * `nc`.
   */
  static GroupwiseTilingParams from_target_tiles_per_thread(
      int m,
      int m_step,
      int n,
      int n_step,
      int target_tiles_per_thread);
};

/**
 * @brief Executes a parallel linear operation using a groupwise low-bit LUT
 * kernel.
 *
 * @details This function acts as a high-level operator for performing a linear
 * operation (GEMM-like) with quantized weights.
 *
 * @param uk The kernel configuration, providing layout details and function
 * pointers.
 * @param tiling_params [in] Optional. User-provided tiling parameters (mc, nc).
 * If not provided, the operator will calculate them dynamically.
 * @param output [out] The destination buffer for the output matrix.
 * @param m The M dimension of the output matrix (e.g., rows).
 * @param n The N dimension of the output matrix (e.g., columns).
 * @param k The K dimension, shared between the weights and activations.
 * @param scale_group_size The group size for weight quantization scales.
 * @param lut_group_size The group size for weight lookup tables (LUTs).
 * @param packed_weights [in] Pointer to the pre-packed weight data.
 * @param activations [in] Pointer to the raw activation data.
 * @param has_clamp A boolean flag indicating whether to apply clamping to the
 * output.
 * @param clamp_min The minimum value for output clamping.
 * @param clamp_max The maximum value for output clamping.
 */
void groupwise_lowbit_weight_lut_parallel_operator(
    const UKernelConfig& uk,
    const std::optional<GroupwiseTilingParams>& tiling_params,
    // Outputs
    float* output,
    // Inputs
    int m,
    int n,
    int k,
    int scale_group_size,
    int lut_group_size,
    const void* packed_weights,
    const float* activations,
    bool has_clamp,
    float clamp_min,
    float clamp_max);
} // namespace torchao::ops::groupwise_lowbit_weight_lut
