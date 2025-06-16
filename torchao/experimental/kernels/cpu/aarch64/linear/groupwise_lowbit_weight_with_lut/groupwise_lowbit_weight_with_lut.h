// Filename: torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight_with_lut.h

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <cassert>
#include <stddef.h>
#include <stdexcept> // For std::invalid_argument

#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight_with_lut/kernel_f32.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight_with_lut/pack_activation.h>
#include <torchao/experimental/kernels/cpu/aarch64/linear/groupwise_lowbit_weight_with_lut/pack_weights.h>

namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut {

/**
 * @brief Calculates the total memory in bytes required for a packed activation buffer.
 *
 * This function must be called to determine the correct buffer size to allocate
 * before calling `pack_activations`. It accounts for any padding needed to
 * make the 'm' dimension a multiple of the kernel's row-tiling factor (MR).
 *
 * @param m The number of rows in the source activation matrix.
 * @param k The number of columns in the source activation matrix.
 * @param MR The row-tiling factor of the micro-kernel that will consume this
 *           packed data (e.g., 4 or 8).
 * @return The required size of the buffer in bytes.
 */
inline size_t packed_activations_size(int m, int k, int MR) {
    // This simply calls the underlying implementation.
    return activation_packing::packed_activations_size(m, k, MR);
}

/**
 * @brief Packs a row-major activation matrix into a kernel-optimized blocked layout.
 *
 * This function rearranges the source matrix into a (M/MR, K, MR) format,
 * which allows the compute kernel to load activation data for MR rows with a
 * single vector instruction, dramatically improving performance.
 *
 * The destination buffer `packed_activations_out` must be pre-allocated by the
 * caller with the size returned by `packed_activations_size()`.
 *
 * @param packed_activations_out Pointer to the destination buffer.
 * @param m The number of rows in the source activation matrix.
 * @param k The number of columns in the source activation matrix.
 * @param activations_in Pointer to the source activation matrix (float32, row-major).
 * @param MR The row-tiling factor of the target micro-kernel. This function
 *           currently supports MR values of 4.
 */
inline void pack_activations(
    void* packed_activations_out,
    int m,
    int k,
    const float* activations_in,
    int MR) {

    switch (MR) {
        case 4:
            activation_packing::pack_activations_for_kernel<4>(packed_activations_out, m, k, activations_in);
            break;
        default:
            throw std::invalid_argument("Unsupported MR value for activation packing. Supported values: [4].");
    }
}

/**
 * @brief Calculates the total size in bytes required for the packed weight buffer.
 *
 * This function must be called to allocate a sufficiently large buffer before
 * calling `pack_weights`.
 *
 * @param weight_nbit The number of bits per weight (e.g., 2, 3, 4).
 * @param n The number of output channels (columns of the weight matrix).
 * @param k The number of input channels (rows of the weight matrix).
 * @param has_bias Whether the packed buffer should include space for a bias vector.
 * @param scale_group_size The number of weights that share a single scale factor.
 * @param lut_group_size The number of weights that share a single Look-Up Table (LUT).
 * @param nr The column-tiling factor of the micro-kernel (e.g., 16 or 8).
 * @return The required size of the buffer in bytes.
 */
 inline size_t packed_weights_size(
    int weight_nbit,
    int n,
    int k,
    bool has_bias,
    int scale_group_size,
    int lut_group_size,
    int NR, bool promote_to_4bit_layout) {

    // This dispatcher calls the correct templated implementation.
    // The user doesn't need to know about the internal templates.
    switch (weight_nbit) {
        case 1:
            return weight_packing::packed_weights_size_for_fused_lut_kernel<1, NR>(n, k, has_bias, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout);
        case 2:
            return weight_packing::packed_weights_size_for_fused_lut_kernel<2, NR>(n, k, has_bias, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout);
        case 3:
            return weight_packing::packed_weights_size_for_fused_lut_kernel<3, NR>(n, k, has_bias, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout)
        case 4:
            return weight_packing::packed_weights_size_for_fused_lut_kernel<4, NR>(n, k, has_bias, scale_group_size, lut_group_size, weight_nbit, promote_to_4bit_layout);
        default:
            throw std::invalid_argument("Unsupported weight_nbit. Must be 1, 2, 3, or 4.");
    }
}

/**
 * @brief Packs quantized weights, scales, LUTs, and biases into a single
 *        contiguous buffer optimized for the target micro-kernel.
 *
 * This function is the primary entry point for preparing weights. It handles
 * transposition, bit-packing, metadata layout, and padding. The caller must
 * pre-allocate `packed_weights_ptr` with the size returned by `packed_weights_size`.
 *
 * @param packed_weights_ptr Pointer to the destination buffer.
 * @param B_qvals Pointer to the source quantized weights, stored as uint8_t values
 *        in a (K, N) row-major layout.
 * @param weight_scales A vector of all unique scale factors.
 * @param weight_luts A vector of all unique Look-Up Tables (LUTs).
 * @param biases A vector of bias values, one for each of the N output channels.
 * @param weight_nbit The number of bits per weight (e.g., 2, 3, 4).
 * @param has_bias True if bias values should be packed.
 * @param N The number of output channels (columns of weights).
 * @param K The number of input channels (rows of weights).
 * @param scale_group_size The grouping factor for scales.
 * @param lut_group_size The grouping factor for LUTs.
 * @param NR The column-tiling factor for the kernel (e.g., 16).
 * @param kr The k-unrolling factor for the kernel (e.g., 16).
 * @param sr The sub-register unrolling factor for the kernel (e.g., 2).
 */
inline void pack_weights(
    // Output
    void* packed_weights_ptr,
    // Inputs
    const uint8_t* B_qvals,
    const std::vector<float>& weight_scales,
    const std::vector<float>& weight_luts,
    const std::vector<float>& biases,
    int weight_nbit,
    bool has_bias,
    int N,
    int K,
    int scale_group_size,
    int lut_group_size,
    int NR,
    int kr,
    int sr) {

    // Dispatcher to call the correct templated implementation.
    if (NR == 16 && kr == 16 && sr == 2) {
        switch(weight_nbit) {
            case 1: weight_packing::pack_weights<1, 16, 16, 2>(packed_weights_ptr, B_qvals, weight_scales, weight_luts, biases, has_bias, N, K, scale_group_size, lut_group_size); break;
            case 2: weight_packing::pack_weights<2, 16, 16, 2>(packed_weights_ptr, B_qvals, weight_scales, weight_luts, biases, has_bias, N, K, scale_group_size, lut_group_size); break;
            case 3: weight_packing::pack_weights<3, 16, 16, 2>(packed_weights_ptr, B_qvals, weight_scales, weight_luts, biases, has_bias, N, K, scale_group_size, lut_group_size); break;
            case 4: weight_packing::pack_weights<4, 16, 16, 2>(packed_weights_ptr, B_qvals, weight_scales, weight_luts, biases, has_bias, N, K, scale_group_size, lut_group_size); break;
            default: throw std::invalid_argument("Unsupported weight_nbit.");
        }
    } else {
        throw std::invalid_argument("Unsupported NR/kr/sr combination for weight packing.");
    }
}

/**
 * @brief Computes a group-wise low-bit GEMM using a LUT-based NEON kernel.
 *
 * @param output Pointer to the output matrix C.
 * @param output_m_stride The stride (in elements) between rows of the output matrix.
 * @param m Number of rows in A and C. Must be a multiple of the kernel's MR (e.g., 4).
 * @param n Number of columns in B and C. Must be a multiple of the kernel's NR (e.g., 16).
 * @param k Number of columns in A and rows in B. Must be a multiple of `lut_group`.
 * @param scale_group (Not used by this kernel, but part of a wider API)
 * @param lut_group The physical number of K-rows packed into a single weight group.
 * @param packed_weights Pointer to the pre-packed weight buffer.
 * @param packed_activations Pointer to the pre-packed activation buffer, prepared by
 *        `pack_activations()`.
 * @param clamp_min Minimum value for the fused clamp (ReLU) operation.
 * @param clamp_max Maximum value for the fused clamp (ReLU6) operation.
 * @param has_bias If true, applies the fused bias from the packed weights.
 * @param has_clamp If true, applies the fused clamping.
 * @param has_scale (Not used by this kernel)
 */
inline void groupwise_lowbit_lut_kernel(
        float* output,
        int output_m_stride,
        int m,
        int n,
        int k,
        int scale_group, // Note: This kernel's packer fuses scale into the LUT.
        int lut_group,   // This is the packing_group_size for the kernel.
        const void* packed_weights,
        const void* packed_activations,
        float clamp_min,
        float clamp_max,
        bool has_bias,
        bool has_clamp,
        bool has_scale) // Note: ignored, scale is assumed fused into the LUT.
{
    constexpr int MR = 4;
    constexpr int NR = 16;

    // The public API can perform checks before calling the specialized kernel.
    assert(m % MR == 0);
    assert(n % NR == 0);

    // Call the actual templated kernel implementation from kernel_f32.h
    kernel::groupwise_lowbit_lut_kernel<MR, NR>(
        output,
        output_m_stride,
        m, n, k,
        lut_group,
        packed_weights,
        packed_activations,
        clamp_min, clamp_max,
        has_bias, has_clamp
    );
}

} // namespace torchao::kernels::cpu::aarch64::linear::groupwise_lowbit_weight_with_lut

#endif // defined(__aarch64__) || defined(__ARM_NEON)
