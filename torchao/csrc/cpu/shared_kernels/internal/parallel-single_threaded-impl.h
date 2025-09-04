// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

template <typename F>
void torchao::parallel_1d(const int64_t begin, const int64_t end, const F& f) {
  for (int i = begin; i < end; i += 1) {
    f(i);
  }
}

inline int torchao::get_num_threads() {
  return 1;
}
