// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <arm_neon.h>
#include <benchmark/benchmark.h>
#include <iostream>

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint3.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint4.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <cassert>

namespace {

// Benchmark utility to compare variants of uint3 packing
void pack_uint3_values(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int nbit = 3;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(packed_size % variant == 0);

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;
  uint8x16_t unpacked4;
  uint8x16_t unpacked5;
  uint8x16_t unpacked6;
  uint8x16_t unpacked7;

  switch (variant) {
    case 8:
      for (int i = 0; i < unpacked_size; i += 8) {
        torchao::bitpacking::internal::pack_8_uint3_values(
            packed + ((i * nbit) / bitsPerByte), unpacked + i);
      }
      break;
    case 64:
      for (int i = 0; i < unpacked_size; i += 64) {
        torchao::bitpacking::internal::vec_load_64_uint8_values(
            unpacked0, unpacked1, unpacked2, unpacked3, unpacked + i);
        torchao::bitpacking::internal::vec_pack_64_uint3_values(
            packed + ((i * nbit) / bitsPerByte),
            unpacked0,
            unpacked1,
            unpacked2,
            unpacked3);
      }
      break;
    case 128:
      for (int i = 0; i < unpacked_size; i += 128) {
        torchao::bitpacking::internal::vec_load_64_uint8_values(
            unpacked0, unpacked1, unpacked2, unpacked3, unpacked + i);
        torchao::bitpacking::internal::vec_load_64_uint8_values(
            unpacked4, unpacked5, unpacked6, unpacked7, unpacked + i + 64);
        torchao::bitpacking::internal::vec_pack_128_uint3_values(
            packed + ((i * nbit) / bitsPerByte),
            unpacked0,
            unpacked1,
            unpacked2,
            unpacked3,
            unpacked4,
            unpacked5,
            unpacked6,
            unpacked7);
      }
      break;
  }
}

// Benchmark utility to compare variants of uint3 unpacking
void unpack_uint3_values(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int nbit = 3;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(packed_size % variant == 0);

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;
  uint8x16_t unpacked4;
  uint8x16_t unpacked5;
  uint8x16_t unpacked6;
  uint8x16_t unpacked7;

  switch (variant) {
    case 8:
      for (int i = 0; i < unpacked_size; i += 8) {
        torchao::bitpacking::internal::unpack_8_uint3_values(
            unpacked + i, packed + ((i * nbit) / bitsPerByte));
      }
      break;
    case 64:
      for (int i = 0; i < unpacked_size; i += 64) {
        torchao::bitpacking::internal::vec_unpack_64_uint3_values(
            unpacked0,
            unpacked1,
            unpacked2,
            unpacked3,
            packed + ((i * nbit) / bitsPerByte));
        torchao::bitpacking::internal::vec_store_64_uint8_values(
            unpacked + i, unpacked0, unpacked1, unpacked2, unpacked3);
      }
      break;
    case 128:
      for (int i = 0; i < unpacked_size; i += 128) {
        torchao::bitpacking::internal::vec_unpack_128_uint3_values(
            unpacked0,
            unpacked1,
            unpacked2,
            unpacked3,
            unpacked4,
            unpacked5,
            unpacked6,
            unpacked7,
            packed + ((i * nbit) / bitsPerByte));
        torchao::bitpacking::internal::vec_store_64_uint8_values(
            unpacked + i, unpacked0, unpacked1, unpacked2, unpacked3);
        torchao::bitpacking::internal::vec_store_64_uint8_values(
            unpacked + i + 64, unpacked4, unpacked5, unpacked6, unpacked7);
      }
      break;
  }
}

// Benchmark utility to compare variants of uint4 packing
void pack_uint4_values(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int nbit = 4;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(packed_size % variant == 0);

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;

  switch (variant) {
    case 2:
      for (int i = 0; i < unpacked_size; i += 2) {
        torchao::bitpacking::internal::pack_2_uint4_values(
            packed + ((i * nbit) / bitsPerByte), unpacked + i);
      }
      break;
    case 16:
      for (int i = 0; i < unpacked_size; i += 16) {
        unpacked0 = vld1q_u8(unpacked + i);
        torchao::bitpacking::internal::vec_pack_16_uint4_values(
            packed + ((i * nbit) / bitsPerByte), unpacked0);
      }
      break;
    case 32:
      for (int i = 0; i < unpacked_size; i += 32) {
        unpacked0 = vld1q_u8(unpacked + i);
        unpacked1 = vld1q_u8(unpacked + 16 + i);
        torchao::bitpacking::internal::vec_pack_32_uint4_values(
            packed + ((i * nbit) / bitsPerByte), unpacked0, unpacked1);
      }
      break;
  }
}

// Benchmark utility to compare variants of uint4 unpacking
void unpack_uint4_values(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int nbit = 4;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(packed_size % variant == 0);

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;

  switch (variant) {
    case 2:
      for (int i = 0; i < unpacked_size; i += 2) {
        torchao::bitpacking::internal::unpack_2_uint4_values(
            unpacked + i, packed + ((i * nbit) / bitsPerByte));
      }
      break;
    case 16:
      for (int i = 0; i < unpacked_size; i += 16) {
        torchao::bitpacking::internal::vec_unpack_16_uint4_values(
            unpacked0, packed + ((i * nbit) / bitsPerByte));
        vst1q_u8(unpacked + i, unpacked0);
      }
      break;
    case 32:
      for (int i = 0; i < unpacked_size; i += 32) {
        torchao::bitpacking::internal::vec_unpack_32_uint4_values(
            unpacked0, unpacked1, packed + ((i * nbit) / bitsPerByte));
        vst1q_u8(unpacked + i, unpacked0);
        vst1q_u8(unpacked + 16 + i, unpacked1);
      }
      break;
  }
}

} // namespace

static void benchmark_pack_uint3_values(benchmark::State& state) {
  int unpacked_size = state.range(0);
  int variant = state.range(1);
  int nbit = 3;

  assert(unpacked_size % 8 == 0);
  int packed_size = (unpacked_size / 8) * nbit;

  auto packed = std::vector<uint8_t>(unpacked_size, 0);
  auto unpacked = torchao::get_random_lowbit_vector(packed_size, 8);

  for (auto _ : state) {
    pack_uint3_values(
        packed.data(), unpacked.data(), packed_size, unpacked_size, variant);
  }
}

static void benchmark_unpack_uint3_values(benchmark::State& state) {
  int unpacked_size = state.range(0);
  int variant = state.range(1);
  int nbit = 3;

  assert(unpacked_size % 8 == 0);
  int packed_size = (unpacked_size / 8) * nbit;

  auto packed = torchao::get_random_lowbit_vector(packed_size, 8);
  auto unpacked = std::vector<uint8_t>(unpacked_size, 0);

  for (auto _ : state) {
    unpack_uint3_values(
        unpacked.data(),
        packed.data(),
        unpacked.size(),
        packed.size(),
        variant);
  }
}

static void benchmark_pack_uint4_values(benchmark::State& state) {
  int unpacked_size = state.range(0);
  int variant = state.range(1);
  int nbit = 4;

  assert(unpacked_size % 8 == 0);
  int packed_size = (unpacked_size / 8) * nbit;

  auto packed = std::vector<uint8_t>(unpacked_size, 0);
  auto unpacked = torchao::get_random_lowbit_vector(packed_size, 8);

  for (auto _ : state) {
    pack_uint4_values(
        packed.data(), unpacked.data(), packed_size, unpacked_size, variant);
  }
}

static void benchmark_unpack_uint4_values(benchmark::State& state) {
  int unpacked_size = state.range(0);
  int variant = state.range(1);
  int nbit = 4;

  assert(unpacked_size % 8 == 0);
  int packed_size = (unpacked_size / 8) * nbit;

  auto packed = torchao::get_random_lowbit_vector(packed_size, 8);
  auto unpacked = std::vector<uint8_t>(unpacked_size, 0);

  for (auto _ : state) {
    unpack_uint4_values(
        unpacked.data(),
        packed.data(),
        unpacked.size(),
        packed.size(),
        variant);
  }
}

BENCHMARK(benchmark_pack_uint3_values)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_unpack_uint3_values)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_pack_uint4_values)->ArgsProduct({{128}, {2, 16, 32}});
BENCHMARK(benchmark_unpack_uint4_values)->ArgsProduct({{128}, {2, 16, 32}});

// Run the benchmark
BENCHMARK_MAIN();
