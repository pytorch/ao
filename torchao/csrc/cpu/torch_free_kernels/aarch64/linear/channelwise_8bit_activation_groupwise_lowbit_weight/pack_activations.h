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
  // when mr == 1, kr/sr do not matter
  static_assert(mr == 1);

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

} // namespace torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::activation_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
