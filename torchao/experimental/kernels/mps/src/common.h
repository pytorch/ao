// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#ifdef USE_ATEN
#include <ATen/native/mps/OperationUtils.h>
using namespace at::native::mps;
#elif defined(USE_EXECUTORCH)
#include <executorch/backends/apple/mps/runtime/MPSStream.h>
using namespace executorch::backends::mps::delegate;
#else
#include <torchao/experimental/kernels/mps/src/OperationUtils.h>
#endif

inline void dispatch_block(
    MPSStream* mpsStream,
    void (^block)()) {
#if defined(USE_ATEN)
  dispatch_sync_with_rethrow(mpsStream->queue(), block);
#elif defined(USE_EXECUTORCH)
  dispatch_sync(mpsStream->queue(), block);
#else
  (void)mpsStream;
  block();
#endif
}

inline void optionally_wait_for_command_completion(MPSStream* mpsStream) {
#if defined(USE_ATEN)
#elif defined(USE_EXECUTORCH)
  ET_CHECK(mpsStream->synchronize(SyncType::COMMIT_AND_WAIT) == executorch::runtime::Error::Ok);
#else
  id<MTLCommandEncoder> encoder = mpsStream->commandEncoder();
  id<MTLCommandBuffer> cmdBuffer = mpsStream->commandBuffer();
  [encoder endEncoding];
  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];
#endif
}

inline id<MTLDevice> get_metal_device() {
#if defined(USE_ATEN) || defined(USE_EXECUTORCH)
  return MPSDevice::getInstance()->device();
#else
  return getMetalDevice();
#endif
}
