// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <omp.h>

template <typename F>
void torchao::parallel_1d(const int64_t begin, const int64_t end, const F& f) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = begin; i < end; i += 1) {
      f(i);
    }
  }
}

inline int torchao::get_num_threads() {
  // omp_get_num_threads returns the number of threads
  // in the current code section, which will be 1 in the routines
  // that select tiling params
  return omp_get_max_threads();
}
