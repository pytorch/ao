// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

id<MTLDevice> getMetalDevice();

class MPSStream {
 public:
  MPSStream() {
    _commandQueue = [getMetalDevice() newCommandQueue];
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
