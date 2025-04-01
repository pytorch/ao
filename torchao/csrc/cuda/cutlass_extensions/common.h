// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#define CUTLASS_STATUS_CHECK(status, message_prefix)                           \
  {                                                                            \
    TORCH_CHECK(status == cutlass::Status::kSuccess, message_prefix,           \
                " : Got CUTLASS error: ", cutlassGetStatusString(status));     \
  }

namespace torchao {

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
