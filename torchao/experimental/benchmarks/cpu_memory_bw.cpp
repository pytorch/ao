/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

DEFINE_int32(array_size, 1024 * 1024 * 256, "Size of array num int32 elements.");

DEFINE_int32(num_threads, 1, "Number of threads to use");

namespace {
int32_t reduction_add(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    const int32_t numel) {
  int32_t total_sum = 0;
  for (int32_t i = 0; i < numel; i++) {
    total_sum += a[i] + b[i];
  }
  return total_sum;
}

void init_array(int32_t* a, int32_t* b, int32_t size) {
  std::random_device rd; // a seed source for the random number engine
  std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(0);
  for (int32_t i = 0; i < size; i++) {
    a[i] = distrib(gen);
    b[i] = distrib(gen);
  }
}

int32_t thread_parallel_benchmark(
    int32_t* a,
    int32_t* b,
    int32_t array_size,
    int32_t num_threads) {
  if (num_threads == -1) {
    num_threads = std::thread::hardware_concurrency();
  }
  std::vector<std::thread> threads;
  std::atomic<int32_t> total_sum(0);

  int32_t block_size = (array_size + num_threads - 1) / num_threads;
  auto reduction_add_lambda = [&](int32_t* a, int32_t* b, int32_t size) {
    auto sum = reduction_add(a, b, size);
    total_sum.fetch_add(sum);
  };

  for (int32_t i = 1; i < num_threads; i++) {
    auto thread_block_size = ((i + 1) * block_size < array_size)
        ? block_size
        : (array_size - i * block_size);
    threads.emplace_back(
        reduction_add_lambda,
        a + i * block_size,
        b + i * block_size,
        thread_block_size);
  }

  reduction_add_lambda(a, b, block_size);

  for (int32_t i = 0; i < num_threads - 1; i++) {
    threads[i].join();
  }
  return total_sum.load();
}

}

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::unique_ptr<int32_t[]> a;
  std::unique_ptr<int32_t[]> b;
  a = std::make_unique<int32_t[]>(FLAGS_array_size);
  b = std::make_unique<int32_t[]>(FLAGS_array_size);
  init_array(a.get(), b.get(), FLAGS_array_size);
  std::cout << "Number of threads: " << FLAGS_num_threads << std::endl;
  int32_t total_sum =
      thread_parallel_benchmark(a.get(), b.get(), FLAGS_array_size, FLAGS_num_threads);
  auto start = std::chrono::high_resolution_clock::now();
  constexpr int32_t num_iters = 10;
  for (int32_t i = 0; i < num_iters; ++i) {
    total_sum =
        thread_parallel_benchmark(a.get(), b.get(), FLAGS_array_size, FLAGS_num_threads);
  }
  std::cout << "Total sum: " << total_sum << std::endl;
  auto stop = std::chrono::high_resolution_clock::now();
  int64_t bytes = 2 * FLAGS_array_size * sizeof(int32_t) * num_iters;
  std::cout << "Bytes read: " << bytes / 1024 << " KiB" << std::endl;
  double bw_per_ns = bytes /
      std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
          .count();
  double bw_per_ms = bw_per_ns * (1000 * 1000);
  auto bw_per_sec = bw_per_ms * 1000;
  auto MiB_per_sec = bw_per_sec / (1024 * 1024);
  std::cout << "Achieved memory bw: " << MiB_per_sec << " MiB/second"
            << std::endl;
  return 0;
}
