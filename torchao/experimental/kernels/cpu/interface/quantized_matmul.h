// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>

#include <torchao/experimental/kernels/cpu/fallback/matmul/channelwise_8bit_a_channelwise_8bit_b.h>
#include <torchao/experimental/kernels/cpu/fallback/matmul/fp32_a_channelwise_8bit_b_fp32_c.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <torchao/experimental/kernels/cpu/aarch64/matmul/matmul.h>
#endif // defined(__aarch64__) && defined(__ARM_NEON)

namespace torchao::kernels::cpu::quantized_matmul {

/*
a_stride_m: stride of a in memory to indiciate how far apart each row is.
b_stride_n: stride of b in memory to indiciate how far apart each row is.
If b is transposed (n x k), then this is how many bytes to skip to get to the
next row. If b is not transposed (k x n), then this is how many bytes to skip to
get to the next row.

It also returns the stride of a and b, that should be used in the kernel.

Will need to think of a better way to find the right
ukernel. Perhaps via ukernelconfig + registry?.
*/
using int8_a_int8_b_channelwise_fp32_c_qmatmul_type = void (*)(
    int,
    int,
    int,
    const void*,
    int,
    const void*,
    int,
    float*,
    int,
    const int8_t*,
    const int8_t*,
    const float*,
    const float*,
    const int,
    const int);

int8_a_int8_b_channelwise_fp32_c_qmatmul_type
get_int8_a_int8_b_channelwise_qmatmul(
    int m,
    int n,
    int k,
    bool a_transposed,
    bool b_transposed,
    int& a_stride_m,
    int& b_stride_n);

int8_a_int8_b_channelwise_fp32_c_qmatmul_type
get_int8_a_int8_b_channelwise_qmatmul(
    int m,
    int n,
    int k,
    bool a_transposed,
    bool b_transposed,
    int& a_stride_m,
    int& b_stride_n) {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (!a_transposed && b_transposed && n >= 8) {
    a_stride_m = k;
    b_stride_n = k;
    return aarch64::quantized_matmul::
        channelwise_8bit_a_channelwise_8bit_b_f32::
            kernel<true, true, false, true>;
  }
#endif // defined(__aarch64__) && defined(__ARM_NEON)
  assert(!a_transposed);
  if (b_transposed) {
    a_stride_m = k;
    b_stride_n = k;
    return torchao::kernels::cpu::fallback::quantized_matmul::
        channelwise_8bit_a_channelwise_8bit_b::kernel<true, true, false, true>;
  } else {
    return torchao::kernels::cpu::fallback::quantized_matmul::
        channelwise_8bit_a_channelwise_8bit_b::kernel<true, true, false, false>;
  }
}

/*
a_stride_m: stride of a in memory to indiciate how far apart each row is.
b_stride_n: stride of b in memory to indiciate how far apart each row is.
If b is transposed (n x k), then this is how many bytes to skip to get to the
next row. If b is not transposed (k x n), then this is how many bytes to skip to
get to the next row.

It also returns the stride of a and b, that should be used in the kernel.

Will need to think of a better way to find the right
ukernel. Perhaps via ukernelconfig + registry?.
*/
using fp32_a_input_channelwise_8bit_b_f32_c_matmul_type = void (*)(
    int,
    int,
    int,
    const float*,
    int,
    const int8_t*,
    int,
    float*,
    int,
    const int8_t*,
    const float*,
    const float,
    const int);

fp32_a_input_channelwise_8bit_b_f32_c_matmul_type
get_fp32_a_input_channelwise_8bit_b_f32_c_matmul(
    int m,
    int n,
    int k,
    bool a_transposed,
    bool b_transposed,
    int& a_stride_m,
    int& b_stride_n);

fp32_a_input_channelwise_8bit_b_f32_c_matmul_type
get_fp32_a_input_channelwise_8bit_b_f32_c_matmul(
    int m,
    int n,
    int k,
    bool a_transposed,
    bool b_transposed,
    int& a_stride_m,
    int& b_stride_n) {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (!a_transposed && !b_transposed && n >= 16) {
    a_stride_m = k;
    b_stride_n = n;
    return aarch64::quantized_matmul::fp32_a_input_channelwise_8bit_b_f32::
        kernel<true, false, false>;
  }
#endif // defined(__aarch64__) && defined(__ARM_NEON)
  assert(!a_transposed);
  if (b_transposed) {
    a_stride_m = k;
    b_stride_n = k;
    return torchao::kernels::cpu::fallback::quantized_matmul::
        fp32_a_input_channelwise_8bit_b_fp32::kernel<true, false, true>;
  } else {
    a_stride_m = k;
    b_stride_n = n;
    return torchao::kernels::cpu::fallback::quantized_matmul::
        fp32_a_input_channelwise_8bit_b_fp32::kernel<true, false, false>;
  }
}
} // namespace torchao::kernels::cpu::quantized_matmul
