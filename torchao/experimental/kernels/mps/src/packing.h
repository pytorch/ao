// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

namespace torchao::kernels::mps::lowbit::packing {

/**
 * Pack weights into a smaller number of bits.
 *
 * @param[in] w_ptr The input weight tensor.
 * @param[out] b_ptr The output packed weight tensor.
 * @param[in] N The number of rows in the weight matrix.
 * @param[in] K The number of columns in the weight matrix.
 */
template <int nbit>
inline void pack(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K);

/**
 * All packing functions are implemented here. All of them pack the weights
 * along the K dimension.
 */

/**
 * 1-bit packing. Each weight is a single bit, so we pack 8 weights into a byte.
 */
template <>
inline void
pack<1>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = n * (K / 8);
    for (int32_t k8 = 0; k8 < K / 8; k8++) {
      uint8_t src_val0 = w_ptr[n * K + k8 * 8];
      uint8_t src_val1 = w_ptr[n * K + k8 * 8 + 1];
      uint8_t src_val2 = w_ptr[n * K + k8 * 8 + 2];
      uint8_t src_val3 = w_ptr[n * K + k8 * 8 + 3];
      uint8_t src_val4 = w_ptr[n * K + k8 * 8 + 4];
      uint8_t src_val5 = w_ptr[n * K + k8 * 8 + 5];
      uint8_t src_val6 = w_ptr[n * K + k8 * 8 + 6];
      uint8_t src_val7 = w_ptr[n * K + k8 * 8 + 7];
      b_ptr[row_base + k8] = (uint8_t(src_val7) << 7) |
          (uint8_t(src_val6) << 6) | (uint8_t(src_val5) << 5) |
          (uint8_t(src_val4) << 4) | (uint8_t(src_val3) << 3) |
          (uint8_t(src_val2) << 2) | (uint8_t(src_val1) << 1) |
          uint8_t(src_val0);
    }
  }
}

/**
 * 2-bit packing. Each weight is two bits, so we pack 4 weights into a byte.
 */
template <>
inline void
pack<2>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = n * (K / 4);
    for (int32_t k4 = 0; k4 < K / 4; k4++) {
      uint8_t src_val0 = w_ptr[n * K + k4 * 4];
      uint8_t src_val1 = w_ptr[n * K + k4 * 4 + 1];
      uint8_t src_val2 = w_ptr[n * K + k4 * 4 + 2];
      uint8_t src_val3 = w_ptr[n * K + k4 * 4 + 3];
      b_ptr[row_base + k4] = (uint8_t(src_val3) << 6) |
          (uint8_t(src_val2) << 4) | (uint8_t(src_val1) << 2) |
          uint8_t(src_val0);
    }
  }
}

/**
 * 3-bit packing. Each weight is 3 bits. We can't pack them into a byte, so we
 * pack 8 weights into 3 bytes.
 */
template <>
inline void
pack<3>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = (n * (K / 8)) * 3;
    for (int32_t k8 = 0; k8 < K / 8; k8++) {
      uint8_t src_val0 = w_ptr[n * K + k8 * 8];
      uint8_t src_val1 = w_ptr[n * K + k8 * 8 + 1];
      uint8_t src_val2 = w_ptr[n * K + k8 * 8 + 2];
      uint8_t src_val3 = w_ptr[n * K + k8 * 8 + 3];
      uint8_t src_val4 = w_ptr[n * K + k8 * 8 + 4];
      uint8_t src_val5 = w_ptr[n * K + k8 * 8 + 5];
      uint8_t src_val6 = w_ptr[n * K + k8 * 8 + 6];
      uint8_t src_val7 = w_ptr[n * K + k8 * 8 + 7];

      b_ptr[row_base + 3 * k8 + 0] = src_val0 | (src_val1 << 3) | (src_val2 << 6);
      b_ptr[row_base + 3 * k8 + 1] = (src_val2 >> 2) | (src_val3 << 1) | (src_val4 << 4) | (src_val5 << 7);
      b_ptr[row_base + 3 * k8 + 2] = (src_val5 >> 1) | (src_val6 << 2) | (src_val7 << 5);
    }
  }
}

/**
 * 4-bit packing. Each weight is four bits, so we pack 2 weights into a byte.
 */
template <>
inline void
pack<4>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = n * (K / 2);
    for (int32_t k2 = 0; k2 < K / 2; k2++) {
      uint8_t src_val0 = w_ptr[n * K + k2 * 2];
      uint8_t src_val1 = w_ptr[n * K + k2 * 2 + 1];
      b_ptr[row_base + k2] = (uint8_t(src_val1) << 4) | uint8_t(src_val0);
    }
  }
}

/**
 * 5-bit packing. Each weight is 5 bits. We pack 8 weights into 5 bytes.
 */
template <>
inline void
pack<5>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = (n * (K / 8)) * 5;
    for (int32_t k8 = 0; k8 < K / 8; k8++) {
      uint8_t src_val0 = w_ptr[n * K + k8 * 8];
      uint8_t src_val1 = w_ptr[n * K + k8 * 8 + 1];
      uint8_t src_val2 = w_ptr[n * K + k8 * 8 + 2];
      uint8_t src_val3 = w_ptr[n * K + k8 * 8 + 3];
      uint8_t src_val4 = w_ptr[n * K + k8 * 8 + 4];
      uint8_t src_val5 = w_ptr[n * K + k8 * 8 + 5];
      uint8_t src_val6 = w_ptr[n * K + k8 * 8 + 6];
      uint8_t src_val7 = w_ptr[n * K + k8 * 8 + 7];

      b_ptr[row_base + 5 * k8 + 0] = src_val0 | (src_val1 << 5);
      b_ptr[row_base + 5 * k8 + 1] = (src_val1 >> 3) | (src_val2 << 2) | (src_val3 << 7);
      b_ptr[row_base + 5 * k8 + 2] = (src_val3 >> 1) | (src_val4 << 4);
      b_ptr[row_base + 5 * k8 + 3] = (src_val4 >> 4) | (src_val5 << 1) | (src_val6 << 6);
      b_ptr[row_base + 5 * k8 + 4] = (src_val6 >> 2) | (src_val7 << 3);
    }
  }
}

/**
 * 6-bit packing. Each weight is 6 bits. We pack 4 weights into 3 bytes.
 */
template <>
inline void
pack<6>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = (n * (K / 4)) * 3;
    for (int32_t k4 = 0; k4 < K / 4; k4++) {
      uint8_t src_val0 = w_ptr[n * K + k4 * 4];
      uint8_t src_val1 = w_ptr[n * K + k4 * 4 + 1];
      uint8_t src_val2 = w_ptr[n * K + k4 * 4 + 2];
      uint8_t src_val3 = w_ptr[n * K + k4 * 4 + 3];

      b_ptr[row_base + 3 * k4 + 0] = src_val0 | (src_val1 << 6);
      b_ptr[row_base + 3 * k4 + 1] = (src_val1 >> 2) | (src_val2 << 4);
      b_ptr[row_base + 3 * k4 + 2] = (src_val2 >> 4) | (src_val3 << 2);
    }
  }
}

/**
 * 7-bit packing. Each weight is 7 bits. We pack 8 weights into 7 bytes.
 */
template <>
inline void
pack<7>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = (n * (K / 8)) * 7;
    for (int32_t k8 = 0; k8 < K / 8; k8++) {
      uint8_t src_val0 = w_ptr[n * K + k8 * 8 + 0];
      uint8_t src_val1 = w_ptr[n * K + k8 * 8 + 1];
      uint8_t src_val2 = w_ptr[n * K + k8 * 8 + 2];
      uint8_t src_val3 = w_ptr[n * K + k8 * 8 + 3];
      uint8_t src_val4 = w_ptr[n * K + k8 * 8 + 4];
      uint8_t src_val5 = w_ptr[n * K + k8 * 8 + 5];
      uint8_t src_val6 = w_ptr[n * K + k8 * 8 + 6];
      uint8_t src_val7 = w_ptr[n * K + k8 * 8 + 7];

      b_ptr[row_base + 7 * k8 + 0] = src_val0 | (src_val1 << 7);
      b_ptr[row_base + 7 * k8 + 1] = (src_val1 >> 1) | (src_val2 << 6);
      b_ptr[row_base + 7 * k8 + 2] = (src_val2 >> 2) | (src_val3 << 5);
      b_ptr[row_base + 7 * k8 + 3] = (src_val3 >> 3) | (src_val4 << 4);
      b_ptr[row_base + 7 * k8 + 4] = (src_val4 >> 4) | (src_val5 << 3);
      b_ptr[row_base + 7 * k8 + 5] = (src_val5 >> 5) | (src_val6 << 2);
      b_ptr[row_base + 7 * k8 + 6] = (src_val6 >> 6) | (src_val7 << 1);
    }
  }
}

} // namespace torchao::kernels::mps::lowbit::packing
