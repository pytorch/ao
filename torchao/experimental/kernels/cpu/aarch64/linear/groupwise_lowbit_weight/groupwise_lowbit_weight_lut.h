#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <stddef.h>
#include <cassert>
#include <stdexcept>

#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight/kernel_f32-impl.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight/pack_activations.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight/pack_weights.h>

namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut {

/**
 * @brief Calculates the total size in bytes required for the packed weight.
 *
 * @param m The number of rows in the source activation matrix.
 * @param k The number of columns in the source activation matrix.
 * @param mr The row-tiling factor of the micro-kernel.
 * @param kr The column-tiling factor of the micro-kernel.
 * @param sr The split ratio of the micro-kernel.
 */
inline size_t packed_activations_size(int m, int k, int mr, int kr, int sr) {
  (void)mr; // unused
  (void)kr; // unused
  (void)sr; // unused
  return activation_packing::packed_activations_size(m, k);
}

/**
 * @brief Packs a row-major activation matrix into a kernel-optimized blocked
layout.
 *
 * @tparam mr_ The row-tiling factor of the micro-kernel (Currently only have
1).
 * @tparam kr_ The column-tiling factor of the micro-kernel (e.g., 32).
 * @tparam sr_ Split ratio determine how the k dimension of a weight tile is
chunked and interleaved during the packing process.
 * @param output Pointer to the destination buffer.
 * @param m The number of rows in the source activation matrix.
 * @param k The number of columns in the source activation matrix.
 * @param input Pointer to the source activation matrix (float32, row-major).
 */
template <int mr_, int kr_, int sr_>
inline void pack_activations(float* output, int m, int k, const float* input) {
  activation_packing::pack_activations<mr_, kr_, sr_>(output, m, k, input);
}

/**
 * @brief Calculates the total size in bytes required for the packed weight
 * buffer for the groupwise LUT kernel format.
 *
 * @param n The number of columns in the weight matrix.
 * @param k The number of rows in the weight matrix.
 * @param weight_nbit The number of bits per weight (e.g., 2, 3, 4).
 * @param scale_group_size The number of weights along the K dim that share a
 * scale factor.
 * @param has_scales If true, the packed buffer will contain scale factors.
 * @param has_bias If true, the packed buffer will contain bias terms.
 * @param nr The column-tiling factor for the kernel (e.g., 16).
 * @param kr The column-tiling factor for the kernel (e.g., 16).
 * @param sr The split ratio of the micro-kernel.
 * @return The total required size of the packed buffer in bytes.
 */
inline size_t packed_weights_size(
    int n,
    int k,
    int weight_nbit,
    int scale_group_size,
    bool has_scales,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  (void)sr; // unused
  return weight_packing::packed_weights_size(
      n, k, weight_nbit, scale_group_size, has_scales, has_bias, nr, kr);
}

/**
 * @brief Packs weights, LUTs, scales and bias into a kernel-optimized format.
 * @tparam weight_nbit_ The true bit-width of the weights.
 * @tparam nr_ The column-tiling factor for the kernel (e.g., 4).
 * @tparam kr_ The column-tiling factor of the micro-kernel (e.g., 32).
 * @tparam sr_ Split ratio determine how the k dimension of a weight tile is
chunked and interleaved during the packing process.
 * @param packed_weights_ptr Pointer to the destination buffer.
 * @param weight_qvals_indices Pointer to the quantized weight matrix (uint8,
row-major).
 * @param weight_scales Pointer to the scale factors (float32, row-major).
 * @param weight_luts Pointer to the LUTs (float32, row-major).
 * @param n The number of columns in the weight matrix.
 * @param k The number of rows in the weight matrix.
 * @param scale_group_size The number of weights that share a scale factor.
 * @param lut_group_size The number of weights that share a LUT.
 * @param has_scales If true, the packed buffer will contain scale factors.
 * @param has_bias If true, the packed buffer will contain bias terms.
 * @param bias Pointer to the bias vector (float32, row-major).
 */
template <int weight_nbit_, int nr_, int kr_, int sr_>
void pack_weights_for_groupwise_lut_kernel(
    /*output*/
    void* packed_weights_ptr,
    /*inputs*/
    const uint8_t* weight_qvals_indices,
    const float* weight_scales,
    const float* weight_luts,
    int n,
    int k,
    int scale_group_size,
    int lut_group_size,
    bool has_scales,
    bool has_bias,
    const float* bias) {
  weight_packing::pack_weights<weight_nbit_, nr_, kr_, sr_>(
      packed_weights_ptr,
      weight_qvals_indices,
      weight_scales,
      weight_luts,
      n,
      k,
      scale_group_size,
      lut_group_size,
      has_scales,
      has_bias,
      bias);
}

/**
 * @brief Computes a group-wise low-bit GEMM using an optimized NEON kernel.
 *
 * This function selects the best available micro-kernel based on the provided
 * tile sizes (MR and NR) and dispatches the computation.
 * @tparam weight_nbit_ The true bit-width of the weights (e.g., 2, 3, 4).
 * @tparam has_scales_ If true, applies the scales.
 * @param output Pointer to the output matrix C.
 * @param output_m_stride The stride (in elements) between rows of the output
 * matrix.
 * @param m Number of rows in A and C.
 * @param n Number of columns in B and C.
 * @param k Number of columns in A and rows in B.
 * @param scale_group_size The grouping factor for scales.
 * @param lut_group_size The grouping factor for LUTs.
 * @param packed_weights Pointer to the pre-packed weight buffer.
 * @param packed_activations Pointer to the pre-packed activation buffer.
 * @param biases Pointer to the bias vector.
 * @param clamp_min Minimum value for the fused clamp (ReLU) operation.
 * @param clamp_max Maximum value for the fused clamp (ReLU6) operation.
 * @param has_bias If true, applies the bias.
 * @param has_clamp If true, applies the clamping.
 */
template <int weight_nbit_, bool has_scales_>
inline void groupwise_lowbit_weight_lut_kernel_1x4x32(
    float* output,
    int output_m_stride,
    int m,
    int n,
    int k,
    int scale_group_size,
    int lut_group_size,
    const void* packed_weights,
    const void* packed_activations,
    float clamp_min,
    float clamp_max,
    bool has_bias,
    bool has_clamp) {
  kernel::groupwise_lowbit_weight_lut_kernel_1x4x32<weight_nbit_, has_scales_>(
      output,
      output_m_stride,
      m,
      n,
      k,
      scale_group_size,
      lut_group_size,
      packed_weights,
      packed_activations,
      clamp_min,
      clamp_max,
      has_bias,
      has_clamp);
}
} // namespace
  // torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_lut

#endif // defined(__aarch64__) || defined(__ARM_NEON)
