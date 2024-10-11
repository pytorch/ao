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
 * pack 8 weights into 3 bytes. But we can't nicely pack the 8 weights
 * continuously. Instead, we pack the upper bits of all weights into the first
 * byte, then the 2 lower bits of all weights into the other 2 bytes.
 */
template <>
inline void
pack<3>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = (n * (K / 8)) * 3;
    for (int32_t k8 = 0; k8 < K / 8; k8++) {
      uint8_t src_0ab = w_ptr[n * K + k8 * 8 + 0];
      uint8_t src_1cd = w_ptr[n * K + k8 * 8 + 1];
      uint8_t src_2ef = w_ptr[n * K + k8 * 8 + 2];
      uint8_t src_3gh = w_ptr[n * K + k8 * 8 + 3];
      uint8_t src_4ij = w_ptr[n * K + k8 * 8 + 4];
      uint8_t src_5kl = w_ptr[n * K + k8 * 8 + 5];
      uint8_t src_6mn = w_ptr[n * K + k8 * 8 + 6];
      uint8_t src_7op = w_ptr[n * K + k8 * 8 + 7];

      // b0: 7|6|5|4|3|2|1|0 (upper bits for all values)
      b_ptr[row_base + 3 * k8 + 0] = ((src_0ab & 4) >> 2) |
          ((src_1cd & 4) >> 1) | ((src_2ef & 4)) | ((src_3gh & 4) << 1) |
          ((src_4ij & 4) << 2) | ((src_5kl & 4) << 3) | ((src_6mn & 4) << 4) |
          ((src_7op & 4) << 5);

      // b1: gh|ef|cd|ab (lower 2 bits for first 4 values)
      b_ptr[row_base + 3 * k8 + 1] = (src_0ab & 3) | ((src_1cd & 3) << 2) |
          ((src_2ef & 3) << 4) | ((src_3gh & 3) << 6);

      // b2: op|mn|kl|ij (lower 2 bits for last 4 values)
      b_ptr[row_base + 3 * k8 + 2] = (src_4ij & 3) | ((src_5kl & 3) << 2) |
          ((src_6mn & 3) << 4) | ((src_7op & 3) << 6);
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
 * 5-bit packing. Each weight is 5 bits. So we pack 8 weights into 5 bytes. We
 * pack the upper bits of all weights into the first byte, then the 4 lower
 * bits of all weights into the other 4 bytes.
 */
template <>
inline void
pack<5>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = (n * (K / 8)) * 5;
    for (int32_t k8 = 0; k8 < K / 8; k8++) {
      uint8_t src_0abAB = w_ptr[n * K + k8 * 8 + 0];
      uint8_t src_1cdCD = w_ptr[n * K + k8 * 8 + 1];
      uint8_t src_2efEF = w_ptr[n * K + k8 * 8 + 2];
      uint8_t src_3ghGH = w_ptr[n * K + k8 * 8 + 3];
      uint8_t src_4ijIJ = w_ptr[n * K + k8 * 8 + 4];
      uint8_t src_5klKL = w_ptr[n * K + k8 * 8 + 5];
      uint8_t src_6mnMN = w_ptr[n * K + k8 * 8 + 6];
      uint8_t src_7opOP = w_ptr[n * K + k8 * 8 + 7];

      // b0: 7|6|5|4|3|2|1|0 (upper bits for all values)
      b_ptr[row_base + 5 * k8 + 0] = ((src_0abAB & 16) >> 4) |
          ((src_1cdCD & 16) >> 3) | ((src_2efEF & 16) >> 2) |
          ((src_3ghGH & 16) >> 1) | ((src_4ijIJ & 16)) |
          ((src_5klKL & 16) << 1) | ((src_6mnMN & 16) << 2) |
          ((src_7opOP & 16) << 3);

      // b1: cdCD|abAB (lower 4 bits for first 2 values)
      b_ptr[row_base + 5 * k8 + 1] = (src_0abAB & 15) | ((src_1cdCD & 15) << 4);

      // b2: ghGH|efEF (lower 4 bits for second 2 values)
      b_ptr[row_base + 5 * k8 + 2] = (src_2efEF & 15) | ((src_3ghGH & 15) << 4);

      // b3: klKL|ijIJ (lower 4 bits for third 2 values)
      b_ptr[row_base + 5 * k8 + 3] = (src_4ijIJ & 15) | ((src_5klKL & 15) << 4);

      // b4: opOP|mnMN (lower 4 bits for last 2 values)
      b_ptr[row_base + 5 * k8 + 4] = (src_6mnMN & 15) | ((src_7opOP & 15) << 4);
    }
  }
}

/**
 * 6-bit packing. Each weight is 6 bits. So we pack 4 weights into 3 bytes. We
 * pack the upper 2 bits of all 4 weights into the first 2 bytes, then the 4
 * lower bits of all weights into the other 4 bytes.
 */
template <>
inline void
pack<6>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = (n * (K / 4)) * 3;
    for (int32_t k4 = 0; k4 < K / 4; k4++) {
      uint8_t src_10abcd = w_ptr[n * K + k4 * 4 + 0];
      uint8_t src_32efgh = w_ptr[n * K + k4 * 4 + 1];
      uint8_t src_54ijkl = w_ptr[n * K + k4 * 4 + 2];
      uint8_t src_76mnop = w_ptr[n * K + k4 * 4 + 3];

      // b0: 76|54|32|10 (upper 2 bits for all values)
      b_ptr[row_base + 3 * k4 + 0] = ((src_10abcd & 48) >> 4) |
          ((src_32efgh & 48) >> 2) | ((src_54ijkl & 48)) |
          ((src_76mnop & 48) << 2);

      // b1: efgh|abcd (lower 4 bits for first 2 values)
      b_ptr[row_base + 3 * k4 + 1] =
          (src_10abcd & 15) | ((src_32efgh & 15) << 4);

      // b2: mnop|ijkl (lower 4 bits for last 2 values)
      b_ptr[row_base + 3 * k4 + 2] =
          (src_54ijkl & 15) | ((src_76mnop & 15) << 4);
    }
  }
}

/**
 * 7-bit packing. Each weight is 7 bits. So we pack 8 weights into 7 bytes.
 * Each of the 7 bytes contains 1 weight, plus 1 bit from the 8th weight. So,
 * this packing spreads the 8th weight across all 7 bytes. The upper bit of
 * each byte is the bit from the 8th weight.
 */
template <>
inline void
pack<7>(const uint8_t* w_ptr, uint8_t* b_ptr, int32_t N, int32_t K) {
  for (int32_t n = 0; n < N; n++) {
    int32_t row_base = (n * (K / 8)) * 7;
    for (int32_t k8 = 0; k8 < K / 8; k8++) {
      uint8_t src_0 = w_ptr[n * K + k8 * 8 + 0];
      uint8_t src_1 = w_ptr[n * K + k8 * 8 + 1];
      uint8_t src_2 = w_ptr[n * K + k8 * 8 + 2];
      uint8_t src_3 = w_ptr[n * K + k8 * 8 + 3];
      uint8_t src_4 = w_ptr[n * K + k8 * 8 + 4];
      uint8_t src_5 = w_ptr[n * K + k8 * 8 + 5];
      uint8_t src_6 = w_ptr[n * K + k8 * 8 + 6];
      uint8_t src_7 = w_ptr[n * K + k8 * 8 + 7];

      b_ptr[row_base + 7 * k8 + 0] = src_0 | ((src_7 & 1) << 7);
      b_ptr[row_base + 7 * k8 + 1] = src_1 | ((src_7 & 2) << 6);
      b_ptr[row_base + 7 * k8 + 2] = src_2 | ((src_7 & 4) << 5);
      b_ptr[row_base + 7 * k8 + 3] = src_3 | ((src_7 & 8) << 4);
      b_ptr[row_base + 7 * k8 + 4] = src_4 | ((src_7 & 16) << 3);
      b_ptr[row_base + 7 * k8 + 5] = src_5 | ((src_7 & 32) << 2);
      b_ptr[row_base + 7 * k8 + 6] = src_6 | ((src_7 & 64) << 1);
    }
  }
}

} // namespace torchao::kernels::mps::lowbit::packing
