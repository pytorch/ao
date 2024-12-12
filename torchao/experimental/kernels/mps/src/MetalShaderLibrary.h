// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/experimental/kernels/mps/src/common.h>

class MetalShaderLibrary {
 public:
  MetalShaderLibrary(const std::string& src) : shaderSource(src) {
    lib = compileLibraryFromSource(shaderSource);
  }
  MetalShaderLibrary(const MetalShaderLibrary&) = delete;
  MetalShaderLibrary(MetalShaderLibrary&&) = delete;

  id<MTLComputePipelineState> getPipelineStateForFunc(
      const std::string& fname) {
    id<MTLFunction> func = loadFunc(fname);

    NSError* error = nil;
    id<MTLDevice> device = get_metal_device();
    auto cpl = [device newComputePipelineStateWithFunction:func error:&error];
    if (cpl == nil) {
      throw std::runtime_error(
          "Failed to construct pipeline state: " +
          std::string(error.description.UTF8String));
    }
    return cpl;

  }

 private:
  std::string shaderSource;
  id<MTLLibrary> lib = nil;

  id<MTLFunction> loadFunc(const std::string& func_name) const {
    id<MTLFunction> func = [lib
        newFunctionWithName:[NSString stringWithUTF8String:func_name.c_str()]];
    if (func == nil) {
      throw std::runtime_error("Can't get function:" + func_name);
    }
    return func;
  }

  id<MTLLibrary> compileLibraryFromSource(
      const std::string& source) {
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    [options setLanguageVersion:MTLLanguageVersion3_1];
    NSString* kernel_source = [NSString stringWithUTF8String:source.c_str()];
    id<MTLDevice> device = get_metal_device();
    id<MTLLibrary> library = [device newLibraryWithSource:kernel_source
                                                  options:options
                                                    error:&error];
    if (library == nil) {
      throw std::runtime_error(
          "Failed to compile: " + std::string(error.description.UTF8String));
    }
    return library;
  }
};
