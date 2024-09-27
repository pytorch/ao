// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <stdexcept>

#define TORCHAO_CHECK(cond, message)   \
  if (!(cond)) {                       \
    throw std::runtime_error(message); \
  }
