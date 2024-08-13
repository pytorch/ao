// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <benchmark/benchmark.h>
#include <torchao/experimental/kernels/cpu/aarch64/quantization/quantize.h>
#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>

static void benchmark_quantize(benchmark::State& state) {
  int nbit = state.range(0);
  int size = state.range(1);
  auto vals = torchao::get_random_vector(size, -10, 10);
  auto qvals = std::vector<int8_t>(size, 0);

  int qmin, qmax, zero;
  float vmin, vmax, scale;

  for (auto _ : state) {
    torchao::kernels::cpu::aarch64::reduction::find_min_and_max(
        vmin, vmax, vals.data(), vals.size());

    torchao::quantization::get_qvals_range(
        qmin, qmax, nbit, /*is_symmetric=*/false);

    torchao::quantization::get_scale_and_zero(
        scale, zero, vmin, vmax, qmin, qmax);

    torchao::kernels::cpu::aarch64::quantization::quantize(
        qvals.data(), vals.data(), vals.size(), scale, zero, qmin, qmax);
  }
}

BENCHMARK(benchmark_quantize)
    ->ArgsProduct(
        {{3, 4, 8}, benchmark::CreateRange(1024, 131072, /*multi=*/4)});

// Run the benchmark
BENCHMARK_MAIN();
