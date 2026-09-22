// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/extension/threadpool/threadpool.h>

template <typename F>
void torchao::parallel_1d(const int64_t begin, const int64_t end, const F& f) {
  torch::executorch::threadpool::get_threadpool()->run(
      [&](size_t i) {
        int64_t idx = begin + i;
        f(idx);
      },
      end - begin);
}

inline int torchao::get_num_threads() {
  return torch::executorch::threadpool::get_threadpool()->get_thread_count();
}
