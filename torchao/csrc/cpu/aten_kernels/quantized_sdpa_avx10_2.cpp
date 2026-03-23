// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
//
// AVX10.2-specific implementations of quantized SDPA kernels.
// This file is compiled with -march=diamondrapids to enable AVX10.2.
// Only called at runtime when __builtin_cpu_supports("avx10.2") is true.
//
// TODO: Add AVX10.2-optimized implementations here.

#pragma GCC push_options
#pragma GCC target("avx10.2")
#include <immintrin.h>

namespace torchao {
namespace cpu_avx10_2 {

// Placeholder: AVX10.2 optimized quantized SDPA kernel
// Currently falls back to AVX512 path; will be optimized for AVX10.2 (DMR)
// when AVX10.2-specific optimizations are implemented.
inline bool is_avx10_2_available() {
  return __builtin_cpu_supports("avx10.2");
}

} // namespace cpu_avx10_2
} // namespace torchao
#pragma GCC pop_options
