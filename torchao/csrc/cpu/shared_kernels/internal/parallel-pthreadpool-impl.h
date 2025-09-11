// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <pthreadpool.h>
#include <stdexcept>

namespace torchao::parallel::internal {
class Threadpool {
 private:
  pthreadpool_t pthreadpool_{nullptr};

 public:
  Threadpool(size_t num_threads = 0) {
    pthreadpool_ = pthreadpool_create(num_threads);
    if (pthreadpool_ == nullptr) {
      throw std::runtime_error("Failed to create pthreadpool.");
    }
  }
  ~Threadpool() {
    pthreadpool_destroy(pthreadpool_);
    pthreadpool_ = nullptr;
  }
  pthreadpool_t get() {
    return pthreadpool_;
  }
  size_t get_num_threads() {
    if (pthreadpool_ == nullptr) {
      return 0;
    }
    return pthreadpool_get_threads_count(pthreadpool_);
  }
};

template <typename F>
struct Context {
  const F& f;
  int64_t begin;
  Context(const F& f, int begin) : f{f}, begin{begin} {}
};

template <typename F>
static void task(Context<F>* context, size_t i) {
  int64_t idx = context->begin + i;
  context->f(idx);
}

static Threadpool threadpool;
} // namespace torchao::parallel::internal

inline int torchao::get_num_threads() {
  return torchao::parallel::internal::threadpool.get_num_threads();
}

template <typename F>
void torchao::parallel_1d(const int64_t begin, const int64_t end, const F& f) {
  auto context = torchao::parallel::internal::Context<F>(f, begin);
  pthreadpool_parallelize_1d(
      torchao::parallel::internal::threadpool.get(),
      (pthreadpool_task_1d_t)torchao::parallel::internal::task<F>,
      (void**)&context,
      /*range=*/end - begin,
      /*flags=*/0);
}
