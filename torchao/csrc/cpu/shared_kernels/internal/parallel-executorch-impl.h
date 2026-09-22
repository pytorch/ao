// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/runtime/kernel/thread_parallel_interface.h>

template <typename F>
void torchao::parallel_1d(const int64_t begin, const int64_t end, const F& f) {
  executorch::extension::parallel_for(
      begin, end, 1, [&](int64_t chunk_begin, int64_t chunk_end) {
        for (int64_t idx = chunk_begin; idx < chunk_end; ++idx) {
          f(idx);
        }
      });
}

inline int torchao::get_num_threads() {
  return executorch::extension::get_thread_count();
}
