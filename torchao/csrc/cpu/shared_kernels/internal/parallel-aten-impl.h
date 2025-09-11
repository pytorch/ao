// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/torch.h>

// F has signature [&](int64_t idx)
template <typename F>
void torchao::parallel_1d(const int64_t begin, const int64_t end, const F& f) {
  at::parallel_for(begin, end, 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; idx++) {
      f(idx);
    }
  });
}

inline int torchao::get_num_threads() {
  return torch::get_num_threads();
}
