// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <random>

#include <torchao/experimental/kernels/mps/src/lowbit.h>
#include <torchao/experimental/kernels/mps/test/bfloat16.h>

using Float16 = _Float16;

template <typename T>
const std::string_view type_string();
template <>
const std::string_view type_string<BFloat16>() {
  return "bfloat";
}
template <>
const std::string_view type_string<float>() {
  return "float";
}
template <>
const std::string_view type_string<Float16>() {
  return "half";
}

inline id<MTLBuffer> allocSharedBuffer(id<MTLDevice> device, unsigned length) {
  id<MTLBuffer> rc = [device newBufferWithLength:length
                                         options:MTLResourceStorageModeShared];
  if (rc == nil) {
    throw_exception(
        "Can't allocate " + std::to_string(length) + " bytes on GPU");
  }
  return rc;
}

namespace torchao::kernels::mps::lowbit {

// Reference CPU implementation of lowbit quantized linear
template <typename T>
void reference_linear_lowbit_quant_weights_cpu(
    const T* a_ptr,
    const uint8_t* w_ptr,
    int64_t group_size,
    const T* sz_ptr,
    T* out_ptr,
    int32_t M,
    int32_t K,
    int32_t N,
    int64_t nbit) {
  uint8_t zero_shift = 1 << (nbit - 1);

  for (int32_t m = 0; m < M; m++) {
    for (int32_t n = 0; n < N; n++) {
      const int32_t k_block = (K + group_size - 1) / group_size;
      const T* A_ptr = a_ptr + m * K;

      float rc = 0.0;
      int32_t k = 0;
      for (int32_t kb = 0; kb < k_block; kb++) {
        const float scale = float(sz_ptr[(kb * N + n) * 2 + 0]);
        const float zero =
            float(sz_ptr[(kb * N + n) * 2 + 1]) - scale * float(zero_shift);
        for (int32_t idx = 0; idx < group_size && k < K; idx++, k++) {
          const auto a_val = float(A_ptr[k]);
          uint8_t w_val = w_ptr[n * K + k];
          rc += a_val * (scale * float(w_val) + zero);
        }
      }

      out_ptr[m * N + n] = T(rc);
    }
  }
}

template <typename T, int nbit>
class LowBitTester {
 public:
  LowBitTester(int32_t m, int32_t k, int32_t n, int32_t group_size)
      : M(m), K(k), N(n), qGroupSize(group_size) {}

  void init() {
    allocBuffers(MTL_DEVICE);

    T* a_ptr = reinterpret_cast<T*>([buf_A contents]);
    uint8_t* w_ptr = reinterpret_cast<uint8_t*>([buf_W contents]);
    T* c_ptr = reinterpret_cast<T*>([buf_C contents]);
    T* s_ptr = reinterpret_cast<T*>([buf_SZ contents]);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> int_distrib(0, (1 << nbit) - 1);
    std::uniform_real_distribution<> real_distrib(-1.0, 1.0);

    for (int idx = 0; idx < M * K; ++idx) {
      a_ptr[idx] = real_distrib(generator);
    }
    for (int idx = 0; idx < N * K; ++idx) {
      w_ptr[idx] = int_distrib(generator);
    }
    int32_t ceil_K_group_size = (K + qGroupSize - 1) / qGroupSize;
    for (int idx = 0; idx < N * ceil_K_group_size; ++idx) {
      s_ptr[2 * idx] = (idx + 1.0) / N;
      s_ptr[2 * idx + 1] = 0;
    }
    for (int idx = 0; idx < M * N; ++idx) {
      c_ptr[idx] = -1.0;
    }
  }

  void pack() {
    uint8_t* w_ptr = reinterpret_cast<uint8_t*>([buf_W contents]);
    uint8_t* b_ptr = reinterpret_cast<uint8_t*>([buf_B contents]);
    LowBitQuantWeights<nbit>::pack(w_ptr, b_ptr, N, K);
  }

  void linear() {
    LowBitQuantWeights<nbit>::linear(
        buf_A, buf_B, qGroupSize, buf_SZ, buf_C, M, K, N, type_string<T>());
  }

  bool validate(float atol_lim = 5e-3, float rtol_lim = 5e-3) const {
    T* a_ptr = reinterpret_cast<T*>([buf_A contents]);
    uint8_t* w_ptr = reinterpret_cast<uint8_t*>([buf_W contents]);
    T* c_ptr = reinterpret_cast<T*>([buf_C contents]);
    T* sz_ptr = reinterpret_cast<T*>([buf_SZ contents]);

    char* e_ptr_f = new char[M * N * sizeof(T)]; // expected
    T* e_ptr = reinterpret_cast<T*>(e_ptr_f);
    reference_linear_lowbit_quant_weights_cpu<T>(
        a_ptr, w_ptr, qGroupSize, sz_ptr, e_ptr, M, K, N, nbit);

    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        float rc = float(c_ptr[m * N + n]);
        float expected = float(e_ptr[m * N + n]);

        auto atol = std::abs(rc - expected);
        auto rtol =
            atol / std::max(std::min(std::abs(expected), std::abs(rc)), 1e-6f);
        if (rtol > rtol_lim && atol > atol_lim) {
          std::cerr << "Result " << expected << " vs expected " << rc
                    << " (atol=" << atol << " ,rtol=" << rtol << ") at " << m
                    << ":" << n << std::endl;
          return false;
        }
      }
    }
    return true;
  }

 private:
  void allocBuffers(id<MTLDevice> device) {
    int32_t ceil_K_group_size = (K + qGroupSize - 1) / qGroupSize;
    const int32_t elem_size = sizeof(T);
    buf_A = allocSharedBuffer(device, M * K * elem_size);
    buf_W = allocSharedBuffer(device, N * K);
    buf_B = allocSharedBuffer(device, N * nbit * K / 8);
    buf_C = allocSharedBuffer(device, M * N * elem_size);
    buf_SZ = allocSharedBuffer(device, N * ceil_K_group_size * 2 * elem_size);
  }

 public:
  int32_t M, K, N; // Input-output matirx dims
  int32_t qGroupSize;
  id<MTLBuffer> buf_A; // MxK elements
  id<MTLBuffer> buf_W; // NxK elements
  id<MTLBuffer> buf_B; // NxK elements (packed)
  id<MTLBuffer> buf_C; // MxN elements
  id<MTLBuffer> buf_SZ; // (K/group_size)xNx2 elements
};

} // namespace torchao::kernels::mps::lowbit

template <typename T, int nbit>
void run_test(int32_t m, int32_t k, int32_t n, int32_t group_size) {
  torchao::kernels::mps::lowbit::LowBitTester<T, nbit> tester(
      m, k, n, group_size);
  tester.init();
  tester.pack();
  tester.linear();
  bool success = tester.validate();
  std::cout << "Test " << type_string<T>() << " " << nbit << "-bit " << m << "x"
            << k << "x" << n << " group size: " << group_size << " "
            << (success ? "succeeded" : "failed") << std::endl;
}

template <typename T, int nbit>
void run_test_battery() {
  run_test<T, nbit>(1, 8, 1, 32);
  run_test<T, nbit>(1, 32, 1, 32);
  run_test<T, nbit>(1, 32, 1, 64);
  run_test<T, nbit>(1, 56, 1, 64);
  run_test<T, nbit>(1, 64, 1, 64);
  run_test<T, nbit>(1, 72, 1, 64);
  run_test<T, nbit>(1, 1000, 1, 64);
  run_test<T, nbit>(3, 64, 5, 64);
  run_test<T, nbit>(7, 64, 23, 64);
  run_test<T, nbit>(17, 120, 23, 128);
  run_test<T, nbit>(17, 128, 23, 128);
  run_test<T, nbit>(41, 144, 23, 128);
  run_test<T, nbit>(41, 128, 23, 128);
  run_test<T, nbit>(81, 8, 1, 256);
  run_test<T, nbit>(19, 256, 17, 256);
  run_test<T, nbit>(1, 1000, 81, 256);
}

int main() {
  run_test_battery<float, 1>();
  run_test_battery<float, 2>();
  run_test_battery<float, 3>();
  run_test_battery<float, 4>();
  run_test_battery<float, 5>();
  run_test_battery<float, 6>();
  run_test_battery<float, 7>();

  run_test_battery<Float16, 1>();
  run_test_battery<Float16, 2>();
  run_test_battery<Float16, 3>();
  run_test_battery<Float16, 4>();
  run_test_battery<Float16, 5>();
  run_test_battery<Float16, 6>();
  run_test_battery<Float16, 7>();

  run_test_battery<BFloat16, 1>();
  run_test_battery<BFloat16, 2>();
  run_test_battery<BFloat16, 3>();
  run_test_battery<BFloat16, 4>();
  run_test_battery<BFloat16, 5>();
  run_test_battery<BFloat16, 6>();
  run_test_battery<BFloat16, 7>();

  return 0;
}
