// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/quantization/quantize.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/reduction.h>
#include <cassert>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::activation_packing {

// Prepares activation data for kernel_impl.
//   Per m_idx (row), activations are stored as follows:
//     scale (float), zero (int8_t),
//     group0_qvals (int8_t[group_size]), [group0_qvals_sum (int32_t)]?
//     group1_qvals (int8_t[group_size]), [group1_qvals_sum (int32_t)]?
//     ...
//   The groupi_qvals_sum is only present if has_weight_zeros = true.

// Returns number of bytes required for activation_data
size_t inline packed_activations_size(
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    bool has_weight_zeros) {
  int row_size = 0;

  // scale
  row_size += sizeof(float);

  // zero
  row_size += sizeof(int8_t);

  // qvals
  row_size += sizeof(int8_t) * k;

  // qvals_sum
  if (has_weight_zeros) {
    assert(k % group_size == 0);
    int groups_per_row = k / group_size;
    row_size += sizeof(int32_t) * groups_per_row;
  }

  return row_size * m;
}

template <int mr, int kr, int sr>
void inline pack_activations(
    // Output
    void* activation_data,
    // Inputs
    int m,
    int k,
    // Ignored if has_weight_zeros = false
    int group_size,
    const float* activations,
    bool has_weight_zeros) {
  // The universal multi-row kernels use the same row-major packing.
  // kr/sr do not affect activation packing for these kernels.
  static_assert(mr == 1 || mr == 2 || mr == 4 || mr == 8);

  auto activation_data_byte_ptr = (char*)activation_data;

  float vmin, vmax, scale;
  int qmin, qmax, zero, qvals_sum;
  torchao::quantization::get_qvals_range(
      qmin, qmax, /*nbit=*/8, /*is_symmetric=*/false);

  for (int m_idx = 0; m_idx < m; m_idx++) {
    torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
        vmin, vmax, activations, k);
    torchao::quantization::get_scale_and_zero(
        scale, zero, vmin, vmax, qmin, qmax);

    // Save scale and zero
    *(float32_t*)activation_data_byte_ptr = scale;
    activation_data_byte_ptr += sizeof(float32_t);

    *(int8_t*)activation_data_byte_ptr = (int8_t)zero;
    activation_data_byte_ptr += sizeof(int8_t);

    if (has_weight_zeros) {
      for (int k_idx = 0; k_idx < k; k_idx += group_size) {
        torchao::kernels::cpu::aarch64::quantization::quantize(
            /*qvals=*/(int8_t*)activation_data_byte_ptr,
            /*vals=*/activations,
            /*size=*/group_size,
            /*scale=*/scale,
            /*zero=*/zero,
            /*qmin=*/qmin,
            /*qmax=*/qmax);

        qvals_sum = torchao::kernels::cpu::aarch64::reduction::compute_sum(
            /*vals=*/(int8_t*)activation_data_byte_ptr,
            /*size=*/group_size);

        activation_data_byte_ptr += group_size;

        *(int32_t*)activation_data_byte_ptr = qvals_sum;
        activation_data_byte_ptr += sizeof(int32_t);

        activations += group_size;
      }
    } else {
      torchao::kernels::cpu::aarch64::quantization::quantize(
          /*qvals=*/(int8_t*)activation_data_byte_ptr,
          /*vals=*/activations,
          /*size=*/k,
          /*scale=*/scale,
          /*zero=*/zero,
          /*qmin=*/qmin,
          /*qmax=*/qmax);
      activation_data_byte_ptr += k;
      activations += k;
    }
  }
}

namespace internal {

template <int mr>
void inline pack_interleaved_block(
    char*& packed,
    int k,
    int group_size,
    const float* activations,
    bool has_weight_zeros) {
  static_assert(mr == 4 || mr == 8);

  float scales[mr];
  int zeros[mr];
  int qmin, qmax;
  torchao::quantization::get_qvals_range(
      qmin, qmax, /*nbit=*/8, /*is_symmetric=*/false);

  for (int row = 0; row < mr; row++) {
    float vmin, vmax;
    torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
        vmin, vmax, activations + row * k, k);
    torchao::quantization::get_scale_and_zero(
        scales[row], zeros[row], vmin, vmax, qmin, qmax);
    std::memcpy(packed, &scales[row], sizeof(float));
    packed += sizeof(float);
  }
  for (int row = 0; row < mr; row++) {
    *reinterpret_cast<int8_t*>(packed++) = static_cast<int8_t>(zeros[row]);
  }

  for (int k_idx = 0; k_idx < k; k_idx += group_size) {
    int32_t qvals_sums[mr]{};
    for (int i = 0; i < group_size; i += 16) {
      int8_t qvals[mr][16];
      for (int row = 0; row < mr; row++) {
        torchao::kernels::cpu::aarch64::quantization::quantize(
            qvals[row],
            activations + row * k + k_idx + i,
            /*size=*/16,
            scales[row],
            zeros[row],
            qmin,
            qmax);
        if (has_weight_zeros) {
          qvals_sums[row] +=
              torchao::kernels::cpu::aarch64::reduction::compute_sum(
                  qvals[row], 16);
        }
      }

      // Store four adjacent K values from each of four rows in one vector.
      // This is the layout consumed directly by vdotq_laneq_s32.
      for (int row_block = 0; row_block < mr; row_block += 4) {
        for (int k_block = 0; k_block < 4; k_block++) {
          for (int row = 0; row < 4; row++) {
            std::memcpy(
                packed,
                qvals[row_block + row] + k_block * 4,
                /*count=*/4);
            packed += 4;
          }
        }
      }
    }
    if (has_weight_zeros) {
      std::memcpy(packed, qvals_sums, mr * sizeof(int32_t));
      packed += mr * sizeof(int32_t);
    }
  }
}

} // namespace internal

template <int mr, int kr, int sr>
void inline pack_activations_interleaved(
    void* activation_data,
    int m,
    int k,
    int group_size,
    const float* activations,
    bool has_weight_zeros) {
  static_assert(mr == 4 || mr == 8);
  (void)kr;
  (void)sr;

  auto* packed = static_cast<char*>(activation_data);
  int m_idx = 0;
  for (; m_idx + mr <= m; m_idx += mr) {
    internal::pack_interleaved_block<mr>(
        packed, k, group_size, activations + m_idx * k, has_weight_zeros);
  }
  if constexpr (mr == 8) {
    if (m_idx + 4 <= m) {
      internal::pack_interleaved_block<4>(
          packed, k, group_size, activations + m_idx * k, has_weight_zeros);
      m_idx += 4;
    }
  }
  if (m_idx < m) {
    pack_activations<1, kr, sr>(
        packed,
        m - m_idx,
        k,
        group_size,
        activations + m_idx * k,
        has_weight_zeros);
  }
}

} // namespace torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::activation_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
