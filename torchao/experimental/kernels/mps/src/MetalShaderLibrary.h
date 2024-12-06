// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#ifdef USE_EXECUTORCH
#include <executorch/backends/apple/mps/runtime/MPSDevice.h>
using executorch::backends::mps::delegate::MPSDevice;
static id<MTLDevice> MTL_DEVICE = MPSDevice::getInstance()->device();
#else
#include <torchao/experimental/kernels/mps/src/OperationUtils.h>
#endif

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
#ifndef USE_EXECUTORCH // TODO(mcandales): Unify with ET error handling
  if (library == nil) {
    throw_exception(
        "Failed to compile: " + std::string(error.description.UTF8String));
  }
#endif
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
#ifndef USE_EXECUTORCH // TODO(mcandales): Unify with ET error handling
    if (func == nil) {
      throw_exception("Can't get function:" + func_name);
    }
#endif
    return func;
  }

  id<MTLComputePipelineState> get_compute_pipeline_state(
      id<MTLFunction> func) const {
    NSError* error = nil;
    auto cpl = [device newComputePipelineStateWithFunction:func error:&error];
#ifndef USE_EXECUTORCH // TODO(mcandales): Unify with ET error handling
    if (cpl == nil) {
      throw_exception(
          "Failed to construct pipeline state: " +
          std::string(error.description.UTF8String));
    }
#endif
    return cpl;
  }
};
