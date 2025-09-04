// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#include <utility>

namespace torchao {
namespace kernels {
namespace cpu {
namespace aarch64 {
namespace reduction {
void find_min_and_max(
    float32_t& min,
    float32_t& max,
    const float32_t* vals,
    int size);

int32_t compute_sum(const int8_t* vals, int size);

} // namespace reduction
} // namespace aarch64
} // namespace cpu
} // namespace kernels
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
