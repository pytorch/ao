// (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>

namespace {
std::random_device rd;
std::mt19937 generator(rd());

// This test is to validate that the benchmarking binary
// can be run on any device. Right now, it is focusing
// on benchmarking on laptop (x86 oe mac) and iOS
static void TestBenchmark(benchmark::State& state) {
  const int32_t K = state.range(0);
  auto a = std::make_unique<float[]>(K);
  auto b = std::make_unique<float[]>(K);
  auto c = std::make_unique<float[]>(K);
  static std::uniform_real_distribution<> real_distrib(-1.0, 1.0);
  for (int ii = 0; ii < K; ++ii) {
    a[ii] = real_distrib(generator);
    b[ii] = real_distrib(generator);
    c[ii] = 0;
  }
  for (auto _ : state) {
    for (int ii = 0; ii < K; ++ii) {
      c[ii] = a[ii] + b[ii];
    }
  }
}

BENCHMARK(TestBenchmark)->Args({4096 * 4})->UseRealTime();
} // namespace

int main(int argc, char** argv) {
  char arg0_default[] = "benchmark";
  char* args_default = arg0_default;
  if (!argv) {
    argc = 1;
    argv = &args_default;
  }
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
