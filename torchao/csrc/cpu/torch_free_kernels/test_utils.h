// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <functional>
#include <random>
#include <vector>

namespace torchao {
inline std::vector<float>
get_random_vector(int size, float min = -1.0, float max = 1.0) {
  assert(min < max);
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto dist = std::bind(std::uniform_real_distribution<float>(min, max), rng);
  std::vector<float> res(size);
  std::generate(res.begin(), res.end(), std::ref(dist));
  return res;
}

inline std::vector<uint8_t> get_random_lowbit_vector(int size, int nbit) {
  assert(nbit >= 1);
  assert(nbit <= 8);

  int min = 0;
  int max = (1 << nbit) - 1;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto dist = std::bind(std::uniform_int_distribution<>(min, max), rng);

  std::vector<uint8_t> res(size);
  std::generate(res.begin(), res.end(), std::ref(dist));
  return res;
}

inline std::vector<int8_t> get_random_signed_lowbit_vector(int size, int nbit) {
  assert(nbit >= 1);
  assert(nbit <= 8);

  int min = 0;
  int max = (1 << nbit) - 1;
  int offset = (1 << (nbit - 1));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto dist = std::bind(std::uniform_int_distribution<>(min, max), rng);

  std::vector<int8_t> res(size);
  std::vector<int16_t> tmp(size);
  std::generate(tmp.begin(), tmp.end(), std::ref(dist));
  for (int i = 0; i < size; i++) {
    res[i] = tmp[i] - offset;
  }
  return res;
}
} // namespace torchao
