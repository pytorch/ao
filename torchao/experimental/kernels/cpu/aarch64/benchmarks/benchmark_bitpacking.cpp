// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <benchmark/benchmark.h>

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint1.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint2.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint3.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint4.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint5.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint6.h>
#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/uint7.h>
#include <torchao/experimental/kernels/cpu/aarch64/tests/test_utils.h>
#include <cassert>

namespace {

// Benchmark utility to compare variants of odd bit packing
template <
    typename pack_8_values_fn_type,
    typename vec_pack_64_values_fn_type,
    typename vec_pack_128_values_fn_type>
TORCHAO_ALWAYS_INLINE inline void pack_uint_odd_bit_values(
    pack_8_values_fn_type pack_8_values_func,
    vec_pack_64_values_fn_type vec_pack_64_values_func,
    vec_pack_128_values_fn_type vec_pack_128_values_func,
    const int nbit,
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(unpacked_size % variant == 0);

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
        pack_8_values_func(packed + ((i * nbit) / bitsPerByte), unpacked + i);
      }
      break;
    case 64:
      for (int i = 0; i < unpacked_size; i += 64) {
        torchao::bitpacking::internal::vec_load_64_uint8_values(
            unpacked0, unpacked1, unpacked2, unpacked3, unpacked + i);
        vec_pack_64_values_func(
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
        vec_pack_128_values_func(
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

// Benchmark utility to compare variants of odd bit unpacking
template <
    typename unpack_8_values_fn_type,
    typename vec_unpack_64_values_fn_type,
    typename vec_unpack_128_values_fn_type>
TORCHAO_ALWAYS_INLINE inline void unpack_uint_odd_bit_values(
    unpack_8_values_fn_type unpack_8_values_func,
    vec_unpack_64_values_fn_type vec_unpack_64_values_func,
    vec_unpack_128_values_fn_type vec_unpack_128_values_func,
    const int nbit,
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(unpacked_size % variant == 0);

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
        unpack_8_values_func(unpacked + i, packed + ((i * nbit) / bitsPerByte));
      }
      break;
    case 64:
      for (int i = 0; i < unpacked_size; i += 64) {
        vec_unpack_64_values_func(
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
        vec_unpack_128_values_func(
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

template <int nbit>
void pack_uint_values(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant);

template <int nbit>
void unpack_uint_values(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant);

// Benchmark utility to compare variants of uint1 packing
template <>
void pack_uint_values<1>(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int nbit = 1;
  pack_uint_odd_bit_values(
      torchao::bitpacking::internal::pack_8_uint1_values,
      torchao::bitpacking::internal::vec_pack_64_uint1_values,
      torchao::bitpacking::internal::vec_pack_128_uint1_values,
      nbit,
      packed,
      unpacked,
      packed_size,
      unpacked_size,
      variant);
}

// Benchmark utility to compare variants of uint1 unpacking
template <>
void unpack_uint_values<1>(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int nbit = 1;
  unpack_uint_odd_bit_values(
      torchao::bitpacking::internal::unpack_8_uint1_values,
      torchao::bitpacking::internal::vec_unpack_64_uint1_values,
      torchao::bitpacking::internal::vec_unpack_128_uint1_values,
      nbit,
      unpacked,
      packed,
      unpacked_size,
      packed_size,
      variant);
}

// Benchmark utility to compare variants of uint2 packing
template <>
void pack_uint_values<2>(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int nbit = 2;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(unpacked_size % variant == 0);

  uint8x8_t unpacked0_8x8;
  uint8x8_t unpacked1_8x8;
  uint8x8_t unpacked2_8x8;
  uint8x8_t unpacked3_8x8;

  uint8x16_t unpacked0_8x16;
  uint8x16_t unpacked1_8x16;
  uint8x16_t unpacked2_8x16;
  uint8x16_t unpacked3_8x16;

  switch (variant) {
    case 4:
      for (int i = 0; i < unpacked_size; i += 4) {
        torchao::bitpacking::internal::pack_4_uint2_values(
            packed + ((i * nbit) / bitsPerByte), unpacked + i);
      }
      break;
    case 32:
      for (int i = 0; i < unpacked_size; i += 32) {
        torchao::bitpacking::internal::vec_load_32_uint8_values(
            unpacked0_8x8,
            unpacked1_8x8,
            unpacked2_8x8,
            unpacked3_8x8,
            unpacked + i);
        torchao::bitpacking::internal::vec_pack_32_uint2_values(
            packed + ((i * nbit) / bitsPerByte),
            unpacked0_8x8,
            unpacked1_8x8,
            unpacked2_8x8,
            unpacked3_8x8);
      }
      break;
    case 64:
      for (int i = 0; i < unpacked_size; i += 64) {
        torchao::bitpacking::internal::vec_load_64_uint8_values(
            unpacked0_8x16,
            unpacked1_8x16,
            unpacked2_8x16,
            unpacked3_8x16,
            unpacked + i);
        torchao::bitpacking::internal::vec_pack_64_uint2_values(
            packed + ((i * nbit) / bitsPerByte),
            unpacked0_8x16,
            unpacked1_8x16,
            unpacked2_8x16,
            unpacked3_8x16);
      }
      break;
  }
}

// Benchmark utility to compare variants of uint2 packing
template <>
void unpack_uint_values<2>(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int nbit = 2;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(unpacked_size % variant == 0);

  uint8x8_t unpacked0_8x8;
  uint8x8_t unpacked1_8x8;
  uint8x8_t unpacked2_8x8;
  uint8x8_t unpacked3_8x8;

  uint8x16_t unpacked0_8x16;
  uint8x16_t unpacked1_8x16;
  uint8x16_t unpacked2_8x16;
  uint8x16_t unpacked3_8x16;

  switch (variant) {
    case 4:
      for (int i = 0; i < unpacked_size; i += 4) {
        torchao::bitpacking::internal::unpack_4_uint2_values(
            unpacked + i, packed + ((i * nbit) / bitsPerByte));
      }
      break;
    case 32:
      for (int i = 0; i < unpacked_size; i += 32) {
        torchao::bitpacking::internal::vec_unpack_32_uint2_values(
            unpacked0_8x8,
            unpacked1_8x8,
            unpacked2_8x8,
            unpacked3_8x8,
            packed + ((i * nbit) / bitsPerByte));
        torchao::bitpacking::internal::vec_store_32_uint8_values(
            unpacked + i,
            unpacked0_8x8,
            unpacked1_8x8,
            unpacked2_8x8,
            unpacked3_8x8);
      }
      break;
    case 64:
      for (int i = 0; i < unpacked_size; i += 64) {
        torchao::bitpacking::internal::vec_unpack_64_uint2_values(
            unpacked0_8x16,
            unpacked1_8x16,
            unpacked2_8x16,
            unpacked3_8x16,
            packed + ((i * nbit) / bitsPerByte));
        torchao::bitpacking::internal::vec_store_64_uint8_values(
            unpacked + i,
            unpacked0_8x16,
            unpacked1_8x16,
            unpacked2_8x16,
            unpacked3_8x16);
      }
      break;
  }
}

// Benchmark utility to compare variants of uint3 packing
template <>
void pack_uint_values<3>(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int nbit = 3;
  pack_uint_odd_bit_values(
      torchao::bitpacking::internal::pack_8_uint3_values,
      torchao::bitpacking::internal::vec_pack_64_uint3_values,
      torchao::bitpacking::internal::vec_pack_128_uint3_values,
      nbit,
      packed,
      unpacked,
      packed_size,
      unpacked_size,
      variant);
}

// Benchmark utility to compare variants of uint3 unpacking
template <>
void unpack_uint_values<3>(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int nbit = 3;
  unpack_uint_odd_bit_values(
      torchao::bitpacking::internal::unpack_8_uint3_values,
      torchao::bitpacking::internal::vec_unpack_64_uint3_values,
      torchao::bitpacking::internal::vec_unpack_128_uint3_values,
      nbit,
      unpacked,
      packed,
      unpacked_size,
      packed_size,
      variant);
}

// Benchmark utility to compare variants of uint4 packing
template <>
void pack_uint_values<4>(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int nbit = 4;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(unpacked_size % variant == 0);

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
template <>
void unpack_uint_values<4>(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int nbit = 4;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(unpacked_size % variant == 0);

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

// Benchmark utility to compare variants of uint5 packing
template <>
void pack_uint_values<5>(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int nbit = 5;
  pack_uint_odd_bit_values(
      torchao::bitpacking::internal::pack_8_uint5_values,
      torchao::bitpacking::internal::vec_pack_64_uint5_values,
      torchao::bitpacking::internal::vec_pack_128_uint5_values,
      nbit,
      packed,
      unpacked,
      packed_size,
      unpacked_size,
      variant);
}

// Benchmark utility to compare variants of uint5 unpacking
template <>
void unpack_uint_values<5>(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int nbit = 5;
  unpack_uint_odd_bit_values(
      torchao::bitpacking::internal::unpack_8_uint5_values,
      torchao::bitpacking::internal::vec_unpack_64_uint5_values,
      torchao::bitpacking::internal::vec_unpack_128_uint5_values,
      nbit,
      unpacked,
      packed,
      unpacked_size,
      packed_size,
      variant);
}

// Benchmark utility to compare variants of uint6 packing
template <>
void pack_uint_values<6>(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int nbit = 6;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(unpacked_size % variant == 0);

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;

  switch (variant) {
    case 4:
      for (int i = 0; i < unpacked_size; i += 4) {
        torchao::bitpacking::internal::pack_4_uint6_values(
            packed + ((i * nbit) / bitsPerByte), unpacked + i);
      }
      break;
    case 32:
      for (int i = 0; i < unpacked_size; i += 32) {
        unpacked0 = vld1q_u8(unpacked + i);
        unpacked1 = vld1q_u8(unpacked + 16 + i);
        torchao::bitpacking::internal::vec_pack_32_uint6_values(
            packed + ((i * nbit) / bitsPerByte), unpacked0, unpacked1);
      }
      break;
    case 64:
      for (int i = 0; i < unpacked_size; i += 64) {
        unpacked0 = vld1q_u8(unpacked + i);
        unpacked1 = vld1q_u8(unpacked + 16 + i);
        unpacked2 = vld1q_u8(unpacked + 32 + i);
        unpacked3 = vld1q_u8(unpacked + 48 + i);
        torchao::bitpacking::internal::vec_pack_64_uint6_values(
            packed + ((i * nbit) / bitsPerByte), unpacked0, unpacked1, unpacked2, unpacked3);
      }
      break;
  }
}

// Benchmark utility to compare variants of uint6 unpacking
template <>
void unpack_uint_values<6>(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int nbit = 6;
  constexpr int bitsPerByte = 8;
  assert(unpacked_size * nbit / bitsPerByte == packed_size);
  assert(unpacked_size % variant == 0);

  uint8x16_t unpacked0;
  uint8x16_t unpacked1;
  uint8x16_t unpacked2;
  uint8x16_t unpacked3;

  switch (variant) {
    case 4:
      for (int i = 0; i < unpacked_size; i += 4) {
        torchao::bitpacking::internal::unpack_4_uint6_values(
            unpacked + i, packed + ((i * nbit) / bitsPerByte));
      }
      break;
    case 32:
      for (int i = 0; i < unpacked_size; i += 32) {
        torchao::bitpacking::internal::vec_unpack_32_uint6_values(
            unpacked0, unpacked1, packed + ((i * nbit) / bitsPerByte));
        vst1q_u8(unpacked + i, unpacked0);
        vst1q_u8(unpacked + 16 + i, unpacked1);
      }
      break;
    case 64:
      for (int i = 0; i < unpacked_size; i += 64) {
        torchao::bitpacking::internal::vec_unpack_64_uint6_values(
            unpacked0, unpacked1, unpacked2, unpacked3, packed + ((i * nbit) / bitsPerByte));
        vst1q_u8(unpacked + i, unpacked0);
        vst1q_u8(unpacked + 16 + i, unpacked1);
        vst1q_u8(unpacked + 32 + i, unpacked2);
        vst1q_u8(unpacked + 48 + i, unpacked3);
      }
      break;
  }
}

// Benchmark utility to compare variants of uint7 packing.
template <>
void pack_uint_values<7>(
    uint8_t* packed,
    uint8_t* unpacked,
    int packed_size,
    int unpacked_size,
    int variant) {
  constexpr int nbit = 7;
  pack_uint_odd_bit_values(
      torchao::bitpacking::internal::pack_8_uint7_values,
      torchao::bitpacking::internal::vec_pack_64_uint7_values,
      torchao::bitpacking::internal::vec_pack_128_uint7_values,
      nbit,
      packed,
      unpacked,
      packed_size,
      unpacked_size,
      variant);
}

// Benchmark utility to compare variants of uint7 unpacking.
template <>
void unpack_uint_values<7>(
    uint8_t* unpacked,
    uint8_t* packed,
    int unpacked_size,
    int packed_size,
    int variant) {
  constexpr int nbit = 7;
  unpack_uint_odd_bit_values(
      torchao::bitpacking::internal::unpack_8_uint7_values,
      torchao::bitpacking::internal::vec_unpack_64_uint7_values,
      torchao::bitpacking::internal::vec_unpack_128_uint7_values,
      nbit,
      unpacked,
      packed,
      unpacked_size,
      packed_size,
      variant);
}


} // namespace

template <int nbit>
static void benchmark_pack_uint_values(benchmark::State& state) {
  int unpacked_size = state.range(0);
  int variant = state.range(1);

  assert(unpacked_size % 8 == 0);
  int packed_size = (unpacked_size / 8) * nbit;

  auto packed = std::vector<uint8_t>(packed_size, 0);
  auto unpacked = torchao::get_random_lowbit_vector(unpacked_size, nbit);

  for (auto _ : state) {
    pack_uint_values<nbit>(
        packed.data(), unpacked.data(), packed_size, unpacked_size, variant);
  }
}

template <int nbit>
static void benchmark_unpack_uint_values(benchmark::State& state) {
  int unpacked_size = state.range(0);
  int variant = state.range(1);

  assert(unpacked_size % 8 == 0);
  int packed_size = (unpacked_size / 8) * nbit;

  auto packed = torchao::get_random_lowbit_vector(packed_size, 8);
  auto unpacked = std::vector<uint8_t>(unpacked_size, 0);

  for (auto _ : state) {
    unpack_uint_values<nbit>(
        unpacked.data(),
        packed.data(),
        unpacked.size(),
        packed.size(),
        variant);
  }
}

BENCHMARK(benchmark_pack_uint_values<1>)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_unpack_uint_values<1>)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_pack_uint_values<2>)->ArgsProduct({{128}, {4, 32, 64}});
BENCHMARK(benchmark_unpack_uint_values<2>)->ArgsProduct({{128}, {4, 32, 64}});
BENCHMARK(benchmark_pack_uint_values<3>)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_unpack_uint_values<3>)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_pack_uint_values<4>)->ArgsProduct({{128}, {2, 16, 32}});
BENCHMARK(benchmark_unpack_uint_values<4>)->ArgsProduct({{128}, {2, 16, 32}});
BENCHMARK(benchmark_pack_uint_values<5>)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_unpack_uint_values<5>)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_pack_uint_values<6>)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_unpack_uint_values<6>)->ArgsProduct({{128}, {4, 32, 64}});
BENCHMARK(benchmark_pack_uint_values<7>)->ArgsProduct({{128}, {8, 64, 128}});
BENCHMARK(benchmark_unpack_uint_values<7>)->ArgsProduct({{128}, {8, 64, 128}});

// Run the benchmark
BENCHMARK_MAIN();

#endif // defined(__aarch64__) || defined(__ARM_NEON)
