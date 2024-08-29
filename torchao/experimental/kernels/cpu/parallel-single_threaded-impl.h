// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

template <typename F>
void torchao::parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  for (int i = begin; i < end; i += grain_size) {
    f(i, i + grain_size);
  }
}

void torchao::set_num_threads(int num_threads) {}
int torchao::get_num_threads() {
  return 1;
}
