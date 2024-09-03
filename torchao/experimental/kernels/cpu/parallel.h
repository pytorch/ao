// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace torchao {
// F has  [&](int64_t begin, int64_t end)
template <typename F>
void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f);

void set_num_threads(int num_threads);

int get_num_threads();

} // namespace torchao

#include <torchao/experimental/kernels/cpu/parallel-impl.h>
