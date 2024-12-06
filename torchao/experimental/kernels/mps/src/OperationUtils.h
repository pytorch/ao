// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <iostream>
#include <stdexcept>

static void throw_exception(const std::string& str) {
  std::cerr << str << std::endl;
  throw std::runtime_error(str);
}

inline void dispatch_block(
    [[maybe_unused]] id<MTLCommandQueue> queue,
    void (^block)()) {
  __block std::optional<std::exception_ptr> block_exception;
  try {
    block();
  } catch (...) {
    block_exception = std::current_exception();
  }
  if (block_exception) {
    std::rethrow_exception(*block_exception);
  }
}

inline id<MTLDevice> getMetalDevice() {
  @autoreleasepool {
    NSArray* devices = [MTLCopyAllDevices() autorelease];
    if (devices.count == 0) {
      throw_exception("Metal is not supported");
    }
    return devices[0];
  }
}

static id<MTLDevice> MTL_DEVICE = getMetalDevice();

class MPSStream {
 public:
  MPSStream() {
    _commandQueue = [MTL_DEVICE newCommandQueue];
  }

  ~MPSStream() {
    [_commandQueue release];
    _commandQueue = nil;

    assert(_commandBuffer == nil);
  }

  id<MTLCommandQueue> queue() const {
    return _commandQueue;
  }

  id<MTLCommandBuffer> commandBuffer() {
    if (!_commandBuffer) {
      auto desc = [MTLCommandBufferDescriptor new];
      desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
      _commandBuffer = [_commandQueue commandBufferWithDescriptor:desc];
    }
    return _commandBuffer;
  }

  id<MTLComputeCommandEncoder> commandEncoder() {
    if (!_commandEncoder) {
      _commandEncoder = [commandBuffer() computeCommandEncoder];
    }
    return _commandEncoder;
  }

 private:
  id<MTLCommandQueue> _commandQueue = nil;
  id<MTLCommandBuffer> _commandBuffer = nil;
  id<MTLComputeCommandEncoder> _commandEncoder = nil;
};

inline MPSStream* getCurrentMPSStream() {
  return new MPSStream();
}
