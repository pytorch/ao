// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <torchao/csrc/cpu/torch_free_kernels/weight_packing/weight_packing.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace torchao::kernels::cpu::fallback::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight {

// Stub functions for activation packing - required for UKernelConfig validation
// but will throw at runtime since fallback doesn't support linear execution.

inline size_t packed_activations_size(
    int m,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  (void)m;
  (void)k;
  (void)group_size;
  (void)has_weight_zeros;
  (void)mr;
  (void)kr;
  (void)sr;
  throw std::runtime_error(
      "packed_activations_size not implemented for fallback (x86). "
      "Linear execution requires ARM NEON.");
}

inline size_t packed_activations_offset(
    int m_idx,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  (void)m_idx;
  (void)k;
  (void)group_size;
  (void)has_weight_zeros;
  (void)mr;
  (void)kr;
  (void)sr;
  throw std::runtime_error(
      "packed_activations_offset not implemented for fallback (x86). "
      "Linear execution requires ARM NEON.");
}

template <int mr_, int kr_, int sr_>
void pack_activations(
    void* packed_activations,
    int m,
    int k,
    int group_size,
    const float* activations,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  (void)packed_activations;
  (void)m;
  (void)k;
  (void)group_size;
  (void)activations;
  (void)has_weight_zeros;
  (void)mr;
  (void)kr;
  (void)sr;
  throw std::runtime_error(
      "pack_activations not implemented for fallback (x86). "
      "Linear execution requires ARM NEON.");
}

template <int weight_nbit, bool has_weight_zeros, bool has_lut>
void kernel_1x8x16_f32_fallback(
    float* output,
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* packed_weights,
    const void* packed_activations,
    float clamp_min,
    float clamp_max,
    bool has_weight_zeros_runtime,
    bool has_bias,
    bool has_clamp) {
  (void)output;
  (void)output_m_stride;
  (void)m;
  (void)n;
  (void)k;
  (void)group_size;
  (void)packed_weights;
  (void)packed_activations;
  (void)clamp_min;
  (void)clamp_max;
  (void)has_weight_zeros_runtime;
  (void)has_bias;
  (void)has_clamp;
  throw std::runtime_error(
      "kernel_1x8x16_f32_fallback not implemented for fallback (x86). "
      "Linear execution requires ARM NEON.");
}

namespace internal {

TORCHAO_ALWAYS_INLINE inline void dequantize_and_store_values(
    float* out,
    const int8_t* qvals,
    int count,
    float scale,
    float zero) {
  for (int i = 0; i < count; ++i) {
    out[i] = (static_cast<float>(qvals[i]) - zero) * scale;
  }
}

} // namespace internal

// Shared embedding op that uses linear packed weights
template <int weight_nbit, int nr, int kr, int sr>
inline void shared_embedding(
    float* out,
    const void* packed_weights,
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias,
    int index) {
  assert(k % group_size == 0);
  assert(group_size % 16 == 0);

  int groups_per_k = k / group_size;
  std::vector<int8_t> weight_qvals(k * nr);
  std::vector<float> weight_scales(groups_per_k * nr);
  std::vector<int8_t> weight_zeros(groups_per_k * nr);
  std::vector<float> bias(nr);

  int n_idx = index / nr;
  n_idx = n_idx * nr;
  int j = index - n_idx;

  torchao::weight_packing::unpack_weights_at_n_idx<weight_nbit, nr, kr, sr>(
      weight_qvals.data(),
      weight_scales.data(),
      has_weight_zeros ? weight_zeros.data() : nullptr,
      has_bias ? bias.data() : nullptr,
      n_idx,
      n,
      k,
      group_size,
      has_weight_zeros,
      has_bias,
      packed_weights);

  for (int i = 0; i < k; i += 16) {
    int chunk_size = std::min(16, k - i);
    float scale = weight_scales[j * groups_per_k + i / group_size];
    float zero = 0.0f;
    if (has_weight_zeros) {
      zero =
          static_cast<float>(weight_zeros[j * groups_per_k + i / group_size]);
    }
    internal::dequantize_and_store_values(
        out + i, weight_qvals.data() + j * k + i, chunk_size, scale, zero);
  }
}

} // namespace
  // torchao::kernels::cpu::fallback::linear::channelwise_8bit_activation_groupwise_lowbit_weight
