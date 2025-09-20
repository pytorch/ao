// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <tuple>
#include <ATen/native/cpu/utils.h>

int64_t get_m_block(int64_t M) {
  if (M <= 48) {
    return M;
  } else if (M < 64) {
    return 32;
  } else if (M < 96) {
    return 64;
  } else {
    return 128;
  }
}

std::tuple<bool, int64_t, int64_t, int64_t>
get_m_blocking(int64_t M) {
  bool parallel_on_M = M > 128;
  int64_t block_m = get_m_block(M);
  int64_t Mc = (M + block_m - 1) / block_m;
  int64_t Mc_parallel = parallel_on_M ? Mc : 1;
  return std::make_tuple(parallel_on_M, block_m, Mc, Mc_parallel);
}

#if defined(CPU_CAPABILITY_AVX512)
template<typename T>
void zero_buffer(T* data, int64_t size) {
  const int32_t vec_size = at::vec::Vectorized<T>::size();
  auto zero_vec = at::vec::Vectorized<T>(0);
  int64_t d = 0;
  for (; d < size - (size % vec_size); d += vec_size) {
    zero_vec.store(data + d);
  }
  if (d < size) {
    zero_vec.store(data + d, size - d);
  }
}
#else
template<typename T>
void zero_buffer(T* data, int64_t size) {
  memset(data, 0, sizeof(T) * size);
}
#endif

template <typename T> struct vnni_traits;
template <> struct vnni_traits<at::Half> { static constexpr int size = 2; };
template <> struct vnni_traits<at::BFloat16> { static constexpr int size = 2; };
template <> struct vnni_traits<at::Float8_e4m3fn> { static constexpr int size = 4; };
template <> struct vnni_traits<int8_t> { static constexpr int size = 4; };

template <typename T> constexpr int get_vnni_size() { return vnni_traits<T>::size; }


// Utilities for dispatch
#define AT_DISPATCH_OUT_TYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                      \
      TYPE,                                                \
      NAME,                                                \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Float, out_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::BFloat16, out_t, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                     \
          at::ScalarType::Half, out_t, __VA_ARGS__))

#define AT_DISPATCH_CASE_ENUM(VALUE, TYPE, HINT, ...)      \
  case VALUE: {                                            \
    constexpr TYPE HINT = VALUE;                           \
    __VA_ARGS__;                                           \
    break;                                                 \
  }

#define AT_DISPATCH_BOOL(VALUE, NAME, HINT, ...)             \
  [&]() {                                                    \
    switch (VALUE) {                                         \
      AT_DISPATCH_CASE_ENUM(true, bool, HINT, __VA_ARGS__)   \
      AT_DISPATCH_CASE_ENUM(false, bool, HINT, __VA_ARGS__)  \
    }                                                        \
  }()

#define AT_DISPATCH_QUANT_MODE(MODE, NAME, HINT, ...)           \
  [&]() {                                                       \
    switch (MODE) {                                             \
      AT_DISPATCH_CASE_ENUM(PER_TENSOR, int, HINT, __VA_ARGS__) \
      AT_DISPATCH_CASE_ENUM(PER_ROW, int, HINT, __VA_ARGS__)    \
      AT_DISPATCH_CASE_ENUM(PER_GROUP, int, HINT, __VA_ARGS__)  \
    }                                                           \
  }()

#define AT_DISPATCH_LINEAR_KERNEL(OUT_DTYPE, CAN_PACK, A_QUANT_MODE, B_QUANT_MODE, ...) \
    AT_DISPATCH_BOOL(                                                                   \
        CAN_PACK, "cpublas_can_pack", can_pack,                                         \
        AT_DISPATCH_QUANT_MODE(                                                         \
            A_QUANT_MODE, "act_quant_mode", a_quant_mode,                               \
            AT_DISPATCH_QUANT_MODE(                                                     \
              B_QUANT_MODE, "wei_quant_mode", b_quant_mode,                             \
                AT_DISPATCH_OUT_TYPES(                                                  \
                    OUT_DTYPE, "out_dtype", __VA_ARGS__                                 \
                )                                                                       \
            )                                                                           \
        )                                                                               \
    )
