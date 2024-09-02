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
  void set_num_threads(size_t num_threads) {
    if (num_threads == get_num_threads()) {
      return;
    }
    pthreadpool_destroy(pthreadpool_);
    pthreadpool_ = pthreadpool_create(num_threads);
  }
};

template <typename F>
struct Context {
  const F& f;
  int grain_size;
  Context(const F& f, int grain_size) : f{f}, grain_size{grain_size} {}
};

template <typename F>
static void task(Context<F>* context, size_t grain_idx) {
  int i = grain_idx * context->grain_size;
  context->f(i, i + context->grain_size);
}

static Threadpool threadpool;
} // namespace torchao::parallel::internal

int torchao::get_num_threads() {
  return torchao::parallel::internal::threadpool.get_num_threads();
}

void torchao::set_num_threads(int num_threads) {
  torchao::parallel::internal::threadpool.set_num_threads(num_threads);
}

template <typename F>
void torchao::parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  int grain_idx_end = end / grain_size;
  auto context = torchao::parallel::internal::Context<F>(f, grain_size);
  pthreadpool_parallelize_1d(
      torchao::parallel::internal::threadpool.get(),
      (pthreadpool_task_1d_t)torchao::parallel::internal::task<F>,
      (void**)&context,
      grain_idx_end,
      0 /* flags */);
}
