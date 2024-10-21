// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(USE_ATEN) && !defined(USE_EXECUTORCH)
#pragma message("USE_ATEN")
#include <torch/library.h>
#include <torch/script.h>
#include <torch/torch.h>
using Tensor = at::Tensor;
#define TORCHAO_CHECK(cond, msg) TORCH_CHECK(cond, msg)

#elif defined(USE_EXECUTORCH) && !defined(USE_ATEN)
#pragma message("USE_EXECUTORCH")
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>
using Tensor = torch::executor::Tensor;
using RuntimeContext = torch::executor::KernelRuntimeContext;
#define TORCHAO_CHECK(cond, msg) ET_CHECK_MSG(cond, msg)

#elif !defined(USE_EXECUTORCH) && !defined(USE_ATEN)
#pragma message("Neither USE_ATEN or USE_EXECUTORCH defined")
#include <stdexcept>

#define TORCHAO_CHECK(cond, message)   \
  if (!(cond)) {                       \
    throw std::runtime_error(message); \
  }

#else
#error "Cannot define both USE_ATEN or USE_EXECUTORCH"
#endif
