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

#ifdef TORCHAO_PARALLEL_ATEN
#pragma message("TORCHAO_PARALLEL_ATEN is set.  Using ATen parallel backend.")

#include <ATen/ATen.h>
#include <torch/torch.h>

template <typename F>
void torchao::parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  at::parallel_for(begin, end, grain_size, f);
}

void torchao::set_num_threads(int num_threads) {
  torch::set_num_threads(num_threads);
}

int torchao::get_num_threads() {
  return torch::get_num_threads();
}

#else
#ifdef TORCHAO_PARALLEL_EXECUTORCH
#pragma message( \
    "TORCHAO_PARALLEL_EXECUTORCH is set.  Using ExecuTorch parallel backend.")

#error "TORCHAO_PARALLEL_EXECUTORCH is not implemented yet"

#else
#ifdef TORCHAO_PARALLEL_PTHREADPOOL
#pragma message( \
    "TORCHAO_PARALLEL_PTHREADPOOL is set.  Using pthreadpool parallel backend.")

#error "TORCHAO_PARALLEL_PTHREADPOOL is not implemented yet"

#else
#ifdef TORCHAO_PARALLEL_OMP
#pragma message("TORCHAO_PARALLEL_OMP is set.  Using OMP parallel backend.")

#include <omp.h>

template <typename F>
void torchao::parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = begin; i < end; i += grain_size) {
      f(i, i + grain_size);
    }
  }
}

void torchao::set_num_threads(int num_threads) {
  omp_set_num_threads(num_threads);
}
int torchao::get_num_threads() {
  // omp_get_num_threads returns the number of threads
  // in the current code section, which will be 1 in the routines
  // that select tiling params
  return omp_get_max_threads();
}

#else
#if defined TORCHAO_PARALLEL_SINGLE_THREADED || \
    defined TORCHAO_PARALLEL_TEST_DUMMY

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

#ifdef TORCHAO_PARALLEL_SINGLE_THREADED
#pragma message( \
    "TORCHAO_PARALLEL_SINGLE_THREADED is set.  Using single-threaded parallel backend.")
void torchao::set_num_threads(int num_threads) {}
int torchao::get_num_threads() {
  return 1;
}
#else // TORCHAO_PARALLEL_TEST_DUMMY
#pragma message( \
    "TORCHAO_PARALLEL_TEST_DUMMY is set.  Using test dummy parallel backend.")

namespace torchao {
static int _dummy_num_threads{1};
}

void torchao::set_num_threads(int num_threads) {
  torchao::_dummy_num_threads = num_threads;
}
int torchao::get_num_threads() {
  return torchao::_dummy_num_threads;
}
#endif // TORCHAO_PARALLEL_SINGLE_THREADED

#else
#error \
    "Set parallel backend by defining one of the following: \
 TORCHAO_PARALLEL_ATEN, \
 TORCHAO_PARALLEL_EXECUTORCH, \
 TORCHAO_PARALLEL_PTHREADPOOL, \
 TORCHAO_PARALLEL_OMP, \
 TORCHAO_PARALLEL_SINGLE_THREADED, \
 TORCHAO_PARALLEL_TEST_DUMMY"
#endif // TORCHAO_PARALLEL_SINGLE_THREADED || TORCHAO_PARALLEL_TEST_DUMMY

#endif // TORCHAO_PARALLEL_OMP
#endif // TORCHAO_PARALLEL_PTHREADPOOL
#endif // TORCHAO_PARALLEL_EXECUTORCH
#endif // TORCHAO_PARALLEL_ATEN
