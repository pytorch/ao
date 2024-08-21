// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#define TORCHAO_CHECK(cond, message)   \
  if (!(cond)) {                       \
    throw std::runtime_error(message); \
  }
