// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <deque>
#include <mutex>

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#define CUTLASS_STATUS_CHECK(status, message_prefix)                           \
  {                                                                            \
    STD_TORCH_CHECK(status == cutlass::Status::kSuccess, message_prefix,       \
                    " : Got CUTLASS error: ", cutlassGetStatusString(status)); \
  }

namespace torchao {

namespace {

std::deque<std::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

inline void initDevicePropertiesVectors() {
  static bool init_flag [[maybe_unused]] = []() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
      STD_TORCH_CHECK(false, "cudaGetDeviceCount failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    device_flags.resize(device_count);
    device_properties.resize(device_count);
    return true;
  }();
}

inline void initDeviceProperty(int device_index) {
  cudaDeviceProp device_prop{};
  cudaError_t err = cudaGetDeviceProperties(&device_prop, device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDeviceProperties failed: " +
                               std::string(cudaGetErrorString(err)));
  }
  device_properties[device_index] = device_prop;
}

inline cudaDeviceProp* get_device_prop() {
  initDevicePropertiesVectors();
  int device_index;
  cudaError_t err = cudaGetDevice(&device_index);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaGetDevice failed: " +
                               std::string(cudaGetErrorString(err)));
  }

  std::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}

inline cudaStream_t get_current_cuda_stream(const torch::stable::Tensor& t) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      static_cast<int32_t>(t.get_device_index()), &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

} // anonymous namespace

template <typename Kernel>
struct enable_2x_kernel_for_sm80_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800
    Kernel::invoke(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_3x_kernel_for_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

}  // namespace torchao
