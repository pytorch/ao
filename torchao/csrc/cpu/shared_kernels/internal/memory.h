// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace torchao {

using aligned_byte_ptr = std::unique_ptr<char[], void (*)(void*)>;

inline aligned_byte_ptr make_aligned_byte_ptr(size_t alignment, size_t size) {
  // Adjust size to next multiple of alignment >= size
  size_t adjusted_size = ((size + alignment - 1) / alignment) * alignment;

  char* ptr = static_cast<char*>(std::aligned_alloc(alignment, adjusted_size));
  if (!ptr) {
    throw std::runtime_error(
        "Failed to allocate memory. Requested size: " + std::to_string(size) +
        ". Requested alignment: " + std::to_string(alignment) +
        ". Adjusted size: " + std::to_string(adjusted_size) + ".");
  }
  return std::unique_ptr<char[], void (*)(void*)>(ptr, std::free);
}
} // namespace torchao
