// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>

int32_t torchao::kernels::cpu::aarch64::reduction::compute_sum(
    const int8_t* vals,
    int size) {
  int32_t res = 0;
  int i = 0;

#pragma unroll(4)
  for (; i < size; i += 16) {
    int8x16_t vec_vals = vld1q_s8(vals + i);
    res += (int)(vaddlvq_s8(vec_vals));
  }
  for (; i < size; i += 1) {
    res += vals[i];
  }
  return res;
}
