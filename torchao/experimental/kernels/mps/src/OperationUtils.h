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

static id<MTLLibrary> compileLibraryFromSource(
    id<MTLDevice> device,
    const std::string& source) {
  NSError* error = nil;
  MTLCompileOptions* options = [MTLCompileOptions new];
  [options setLanguageVersion:MTLLanguageVersion3_1];
  NSString* kernel_source = [NSString stringWithUTF8String:source.c_str()];
  id<MTLLibrary> library = [device newLibraryWithSource:kernel_source
                                                options:options
                                                  error:&error];
  if (library == nil) {
    throw_exception(
        "Failed to compile: " + std::string(error.description.UTF8String));
  }
  return library;
}

class MetalShaderLibrary {
 public:
  MetalShaderLibrary(const std::string& src) : shaderSource(src) {
    lib = compileLibraryFromSource(device, shaderSource);
  }
  MetalShaderLibrary(const MetalShaderLibrary&) = delete;
  MetalShaderLibrary(MetalShaderLibrary&&) = delete;

  id<MTLComputePipelineState> getPipelineStateForFunc(
      const std::string& fname) {
    return get_compute_pipeline_state(load_func(fname));
  }

 private:
  std::string shaderSource;
  id<MTLDevice> device = MTL_DEVICE;
  id<MTLLibrary> lib = nil;

  id<MTLFunction> load_func(const std::string& func_name) const {
    id<MTLFunction> func = [lib
        newFunctionWithName:[NSString stringWithUTF8String:func_name.c_str()]];
    if (func == nil) {
      throw_exception("Can't get function:" + func_name);
    }
    return func;
  }

  id<MTLComputePipelineState> get_compute_pipeline_state(
      id<MTLFunction> func) const {
    NSError* error = nil;
    auto cpl = [device newComputePipelineStateWithFunction:func error:&error];
    if (cpl == nil) {
      throw_exception(
          "Failed to construct pipeline state: " +
          std::string(error.description.UTF8String));
    }
    return cpl;
  }
};

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

inline void finalize_block(MPSStream* mpsStream) {
  id<MTLCommandEncoder> encoder = mpsStream->commandEncoder();
  id<MTLCommandBuffer> cmdBuffer = mpsStream->commandBuffer();
  [encoder endEncoding];
  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];
}

inline MPSStream* getCurrentMPSStream() {
  return new MPSStream();
}
