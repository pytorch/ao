// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>

namespace torchao::kernels::mps::lowbit::dispatch {

inline void dispatch_mm(
    id<MTLComputeCommandEncoder> encoder,
    int32_t maxThreadsPerGroup,
    int32_t M,
    int32_t N,
    [[maybe_unused]] int32_t K) {
  [encoder dispatchThreads:MTLSizeMake(N, M, 1)
      threadsPerThreadgroup:MTLSizeMake(std::min(maxThreadsPerGroup, M), 1, 1)];
}

} // namespace torchao::kernels::mps::lowbit::dispatch
