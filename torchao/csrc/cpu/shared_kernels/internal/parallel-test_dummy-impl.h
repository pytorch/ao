// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

namespace torchao::parallel::internal {
static int num_threads_test_dummy_{1};
}

template <typename F>
void torchao::parallel_1d(const int64_t begin, const int64_t end, const F& f) {
  for (int i = begin; i < end; i += 1) {
    f(i);
  }
}

inline int torchao::get_num_threads() {
  return torchao::parallel::internal::num_threads_test_dummy_;
}


namespace torchao::parallel {
inline void set_num_threads_in_test_dummy(int num_threads) {
  torchao::parallel::internal::num_threads_test_dummy_ = num_threads;
}
}
