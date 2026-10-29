// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <torchao/csrc/cpu/shared_kernels/internal/parallel.h>

#ifndef ET_USE_THREADPOOL
#error "The ExecuTorch parallel backend must enable threadpool support."
#endif

TEST(ExecuTorchParallel, VisitsEachIndexExactlyOnce) {
  const int64_t threads = torchao::get_num_threads();
  ASSERT_GT(threads, 0);
  // More work items than threads exercises multi-index callback chunks.
  const int64_t end = 7 + 2 * threads + 1;
  const std::pair<int64_t, int64_t> ranges[] = {
      {0, 0}, {7, 7}, {7, 8}, {7, end}};
  for (const auto& range : ranges) {
    const int64_t begin = range.first;
    const int64_t limit = range.second;
    SCOPED_TRACE(::testing::Message() << begin << ":" << limit);
    std::vector<std::atomic<int>> visits(end + 1);
    for (auto& count : visits) {
      count.store(0);
    }
    std::atomic<int> outside{0};
    torchao::parallel_1d(begin, limit, [&](int64_t i) {
      if (i < begin || i >= limit) {
        outside.fetch_add(1);
        return;
      }
      visits[i].fetch_add(1);
    });
    EXPECT_EQ(outside.load(), 0);
    for (int64_t i = 0; i <= end; ++i) {
      EXPECT_EQ(visits[i].load(), i >= begin && i < limit ? 1 : 0);
    }
  }
}

TEST(ExecuTorchParallel, ReportsPublicThreadCount) {
  EXPECT_GT(torchao::get_num_threads(), 0);
  EXPECT_EQ(
      torchao::get_num_threads(), executorch::extension::get_thread_count());
}
