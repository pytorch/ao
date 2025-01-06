// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdexcept>

id<MTLDevice> getMetalDevice() {
  @autoreleasepool {
    NSArray* devices = [MTLCopyAllDevices() autorelease];
    if (devices.count == 0) {
      throw std::runtime_error("Metal is not supported");
    }
    static id<MTLDevice> MTL_DEVICE = devices[0];
    return MTL_DEVICE;
  }
}
