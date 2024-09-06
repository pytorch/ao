// (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "../fast_hadamard_transform.h"


class FastHadamardTransformFixture : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) {
    data_.resize((1 << state.range(0)) * state.range(1));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist;
    for (int ii = 0; ii < data_.size(); ++ii) {
      data_[ii] = dist(gen);
    }
  }

  std::vector<float> data_;
};

BENCHMARK_DEFINE_F(FastHadamardTransformFixture, FHT)
(benchmark::State& state) {
  for (auto _ : state) {
    torchao::fast_hadamard_transform(
        data_.data(), state.range(0));
  }
  state.SetBytesProcessed(
      state.iterations() * data_.size());
}

BENCHMARK_DEFINE_F(FastHadamardTransformFixture, FHT28)
(benchmark::State& state) {
  for (auto _ : state) {
    torchao::fast_hadamard_transform_28N(
        data_.data(), state.range(0));
  }
  state.SetBytesProcessed(
      state.iterations() * data_.size());
}


class Quantized16BitFastHadamardTransformFixture : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) {
    data_.resize((1 << state.range(0)) * state.range(1));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<int16_t> dist;
    for (int ii = 0; ii < data_.size(); ++ii) {
      data_[ii] = dist(gen);
    }
  }

  std::vector<int16_t> data_;
};

BENCHMARK_DEFINE_F(Quantized16BitFastHadamardTransformFixture, QFHT)(benchmark::State& state) {
  for (auto _ : state) {
    torchao::fast_hadamard_transform_symmetric_quantized_s16(data_.data(), state.range(0));
  }
  state.SetBytesProcessed(state.iterations() * data_.size());
}

int main(int argc, char** argv) {
  BENCHMARK_REGISTER_F(FastHadamardTransformFixture, FHT)
      ->Args({4, 1})
      ->Args({7, 1}) // Important for Llama 3 with SpinQuant. (K last dim = 128, 1})
      ->Args({9, 1}) // Important for Llama 3 with SpinQuant (hidden dim 14336 =
                     // 512 * 28, 1})
      ->Args({10, 1})
      ->Args({12, 1})
      ->Args({20, 1})
      ->Args({22, 1});

  BENCHMARK_REGISTER_F(FastHadamardTransformFixture, FHT28)
      ->Args({4, 28})
      ->Args({7, 28})
      ->Args({9, 28})
      ->Args({10, 28})
      ->Args({12, 28})
      ->Args({20, 28})
      ->Args({22, 28});

  BENCHMARK_REGISTER_F(Quantized16BitFastHadamardTransformFixture, QFHT)
      ->Args({4, 1})
      ->Args({7, 1}) // Important for Llama 3 with SpinQuant. (K last dim = 128, 1})
      ->Args({9, 1}) // Important for Llama 3 with SpinQuant (hidden dim 14336 =
                     // 512 * 28, 1})
      ->Args({10, 1})
      ->Args({12, 1})
      ->Args({20, 1})
      ->Args({22, 1});


  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
