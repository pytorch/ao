// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/reduction.h>
#include <cassert>

int32_t torchao::kernels::cpu::aarch64::reduction::compute_sum(
    const int8_t* vals,
    int size) {
  assert(size >= 1);

  int32_t res = 0;
  int i = 0;

#pragma unroll(4)
  for (; i + 15 < size; i += 16) {
    int8x16_t vec_vals = vld1q_s8(vals + i);
    res += (int)(vaddlvq_s8(vec_vals));
  }
  for (; i < size; i += 1) {
    res += vals[i];
  }
  return res;
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
