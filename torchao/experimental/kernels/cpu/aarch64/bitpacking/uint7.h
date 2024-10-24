// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>

// This file contains bitpacking and unpacking methods for uint7.
// These are not inteded to be used outside of bitpacking directory.
// See bitpack.h for the interface.

namespace torchao {
namespace bitpacking {
namespace internal {

TORCHAO_ALWAYS_INLINE inline void pack_8_uint7_values(
    uint8_t* packed,
    const uint8_t* unpacked) {
  // Given 8 unpacked uint7 values:
  // aaa aaaa, bbb bbbb, ccc cccc, ddd dddd,
  // eee eeee, fff ffff, ggg gggg, 123 4567,
  // This function produces the following packing:
  // packed[0] = 7aaa aaaa
  // packed[1] = 6bbb bbbb
  // packed[2] = 5ccc cccc
  // packed[3] = 4ddd dddd
  // packed[4] = 3eee eeee
  // packed[5] = 2fff ffff
  // packed[6] = 1ggg gggg
  //
  // Input is 8 bytes
  // Output is 7 * 8 bits = 56 bits = 7 bytes

  // Split the bits of unpacked[7].
  uint8_t mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  uint8x8_t unpacked7 = vtst_u8(vdup_n_u8(unpacked[7]), vld1_u8(mask));
  // At this point, each byte in unpacked7 is all ones or all zeroes, depending
  // on whether the corresponding bit in unpacked7 was one or zero.
  // The next statement combines 7 bits from unpacked[i] with the i-th bit from
  // unpacked7.
  uint8x8_t p = vsli_n_u8(vld1_u8(unpacked), unpacked7, 7);

  packed[0] = vget_lane_u8(p, 0);
  packed[1] = vget_lane_u8(p, 1);
  packed[2] = vget_lane_u8(p, 2);
  packed[3] = vget_lane_u8(p, 3);
  packed[4] = vget_lane_u8(p, 4);
  packed[5] = vget_lane_u8(p, 5);
  packed[6] = vget_lane_u8(p, 6);
}

TORCHAO_ALWAYS_INLINE inline void unpack_8_uint7_values(
    uint8_t* unpacked,
    const uint8_t* packed) {
  // Unpacks 8 uint7 vales packed by pack_8_uint7_values.
  // Load the 7 packed bytes into vector.
  uint8_t temp[8] = {
      packed[0],
      packed[1],
      packed[2],
      packed[3],
      packed[4],
      packed[5],
      packed[6],
      /*ignored*/ 0};
  uint8x8_t v = vld1_u8(temp);
  int8_t shift[8] = {-7, -6, -5, -4, -3, -2, -1, 0};
  // The following and and shift operations will produce a vector with the
  // 0123 4567 bits from the last packed uint7:
  // 0000 0007
  // 0000 0060
  // 0000 0500
  // 0000 4000
  // 0003 0000
  // 0020 0000
  // 0100 0000
  // Which can then be added to obtain unpacked[7].
  unpacked[7] =
      vaddv_u8(vshl_u8(vand_u8(v, vdup_n_u8(0b1000'0000u)), vld1_s8(shift)));
  // All other unpacked values are just the corresponding packed value with the
  // last bit cleared.
  v = vand_u8(v, vdup_n_u8(0b0111'1111u));
  unpacked[0] = vget_lane_u8(v, 0);
  unpacked[1] = vget_lane_u8(v, 1);
  unpacked[2] = vget_lane_u8(v, 2);
  unpacked[3] = vget_lane_u8(v, 3);
  unpacked[4] = vget_lane_u8(v, 4);
  unpacked[5] = vget_lane_u8(v, 5);
  unpacked[6] = vget_lane_u8(v, 6);
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_64_uint7_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3) {
  // This function is a vectorized version of pack_8_uint7_values
  // To understand it, please see pack_8_uint7_values first. The
  // main idea is to use the last bit of each packed uint8_t to
  // store the last uint7.
  //
  // Input is 64 bytes
  // Output is 7*64= 448 bits = 56 bytes
  //
  // Insert one bit from the elements from the last 8-value vector into the
  // first 7 8-value vectors. If bits of `last` are labeled 0123 4567 then
  // bit 7 is inserted into the first packed uint8_t (in a vectorized manner),
  // then bit 6, and so on.
  uint8x8_t last = vget_high_u8(unpacked3);
  // Insert bit 7 of last into the first packed uint8_t.
  vst1_u8(packed, vsli_n_u8(vget_low_u8(unpacked0), last, 7));

  // Repeat for the i-th bit of `last` and the remaining 8-value vectors.
  // Pack bit 6 from 0123 4567.
  vst1_u8(
      packed + 8, vsli_n_u8(vget_high_u8(unpacked0), vshr_n_u8(last, 1), 7));
  // Pack bit 5 from 0123 4567, etc.
  vst1_u8(
      packed + 16, vsli_n_u8(vget_low_u8(unpacked1), vshr_n_u8(last, 2), 7));
  vst1_u8(
      packed + 24, vsli_n_u8(vget_high_u8(unpacked1), vshr_n_u8(last, 3), 7));
  vst1_u8(
      packed + 32, vsli_n_u8(vget_low_u8(unpacked2), vshr_n_u8(last, 4), 7));
  vst1_u8(
      packed + 40, vsli_n_u8(vget_high_u8(unpacked2), vshr_n_u8(last, 5), 7));
  vst1_u8(
      packed + 48, vsli_n_u8(vget_low_u8(unpacked3), vshr_n_u8(last, 6), 7));
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_64_uint7_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    const uint8_t* packed) {
  // Unpacks data packed by vec_pack_64_uint7_values.
  // Please see vec_pack_64_uint7_values first.
  //
  // Input is 7*64 = 448 bits = 56 bytes
  // Output is 64 bytes.
  const uint8x8_t mask = vdup_n_u8(0b0111'1111u);
  // Starting from the last packed byte, extract the most significant bit
  // to reconstruct the last 8-value vector. If the last uint7 value
  // is labeled as 0123 4567 and X are bits we don't care about, we start
  // with last_high = 01XX XXXX
  uint8x8_t last_low = vld1_u8(packed + 48);
  uint8x8_t last_high = vshr_n_u8(last_low, 1);

  uint8x8_t high = vld1_u8(packed + 40);
  uint8x8_t low = vld1_u8(packed + 32);
  // last_high = 012X XXXX
  last_high = vsri_n_u8(last_high, high, 2);
  // last_high = 0123 XXXX
  last_high = vsri_n_u8(last_high, low, 3);
  unpacked2 = vcombine_u8(vand_u8(low, mask), vand_u8(high, mask));

  high = vld1_u8(packed + 24);
  low = vld1_u8(packed + 16);
  // last_high = 0123 4XXX
  last_high = vsri_n_u8(last_high, high, 4);
  // last_high = 0123 45XX
  last_high = vsri_n_u8(last_high, low, 5);
  unpacked1 = vcombine_u8(vand_u8(low, mask), vand_u8(high, mask));

  high = vld1_u8(packed + 8);
  low = vld1_u8(packed);
  // last_high = 0123 456X
  last_high = vsri_n_u8(last_high, high, 6);
  // last_high = 0123 4567
  last_high = vsri_n_u8(last_high, low, 7);
  unpacked0 = vcombine_u8(vand_u8(low, mask), vand_u8(high, mask));

  unpacked3 = vcombine_u8(vand_u8(last_low, mask), vand_u8(last_high, mask));
}

TORCHAO_ALWAYS_INLINE inline void vec_pack_128_uint7_values(
    uint8_t* packed,
    const uint8x16_t& unpacked0,
    const uint8x16_t& unpacked1,
    const uint8x16_t& unpacked2,
    const uint8x16_t& unpacked3,
    const uint8x16_t& unpacked4,
    const uint8x16_t& unpacked5,
    const uint8x16_t& unpacked6,
    const uint8x16_t& unpacked7) {
  // This function is a vectorized version of pack_8_uint7_values
  // To understand it, please see pack_8_uint7_values first. The
  // main idea is to use the last bit of each packed uint8_t to
  // store the last uint7.
  //
  // Input is 128 bytes
  // Output is 7*128= 896 bits = 112 bytes

  // Pack an 8-element vector using the first bit from each element in unpacked7.
  // If those elements are labeled 0123 4567 then the following line does:
  // Shift left insert by 7: 7 | low_7_bits(unpacked0)
  vst1q_u8(packed, vsliq_n_u8(unpacked0, unpacked7, 7));
  // Shift right by 1: 0123 4567 -> 0012 3456
  // Shift left the insert by 7 the above: 6 | low_7_bits(unpacked1)
  vst1q_u8(packed + 16, vsliq_n_u8(unpacked1, vshrq_n_u8(unpacked7, 1), 7));
  // Shift right by 2: 0123 4567 -> 0001 2345
  // Shift left the insert by 7 the above: 5 | low_7_bits(unpacked2)
  vst1q_u8(packed + 16 * 2, vsliq_n_u8(unpacked2, vshrq_n_u8(unpacked7, 2), 7));
  // And so on and so forth...
  vst1q_u8(packed + 16 * 3, vsliq_n_u8(unpacked3, vshrq_n_u8(unpacked7, 3), 7));
  vst1q_u8(packed + 16 * 4, vsliq_n_u8(unpacked4, vshrq_n_u8(unpacked7, 4), 7));
  vst1q_u8(packed + 16 * 5, vsliq_n_u8(unpacked5, vshrq_n_u8(unpacked7, 5), 7));
  vst1q_u8(packed + 16 * 6, vsliq_n_u8(unpacked6, vshrq_n_u8(unpacked7, 6), 7));
}

TORCHAO_ALWAYS_INLINE inline void vec_unpack_128_uint7_values(
    uint8x16_t& unpacked0,
    uint8x16_t& unpacked1,
    uint8x16_t& unpacked2,
    uint8x16_t& unpacked3,
    uint8x16_t& unpacked4,
    uint8x16_t& unpacked5,
    uint8x16_t& unpacked6,
    uint8x16_t& unpacked7,
    const uint8_t* packed) {
  // Unpacks data packed by vec_pack_128_uint7_values
  // Please see vec_pack_128_uint7_values first.
  //
  // Input is 128 bytes
  // Output is 7*128= 896 bits = 112 bytes
  const uint8x16_t mask = vdupq_n_u8(0b0111'1111u);
  // Starting from the last packed byte, extract the most significant bit
  // to reconstruct the last 8-value vector. If the last uint7 value
  // is labeled as 0123 4567 and X are bits we don't care about, we start
  // with unpacked7 = 01XX XXXX
  unpacked6 = vld1q_u8(packed + 16 * 6);
  unpacked7 = vshrq_n_u8(unpacked6, 1);
  unpacked6 = vandq_u8(unpacked6, mask);

  unpacked5 = vld1q_u8(packed + 16 * 5);
  // unpacked7 = 012X XXXX
  unpacked7 = vsriq_n_u8(unpacked7, unpacked5, 2);
  unpacked5 = vandq_u8(unpacked5, mask);

  unpacked4 = vld1q_u8(packed + 16 * 4);
  // unpacked7 = 0123 XXXX
  unpacked7 = vsriq_n_u8(unpacked7, unpacked4, 3);
  unpacked4 = vandq_u8(unpacked4, mask);

  unpacked3 = vld1q_u8(packed + 16 * 3);
  // unpacked7 = 0123 4XXX
  unpacked7 = vsriq_n_u8(unpacked7, unpacked3, 4);
  unpacked3 = vandq_u8(unpacked3, mask);

  unpacked2 = vld1q_u8(packed + 16 * 2);
  // unpacked7 = 0123 45XX
  unpacked7 = vsriq_n_u8(unpacked7, unpacked2, 5);
  unpacked2 = vandq_u8(unpacked2, mask);

  unpacked1 = vld1q_u8(packed + 16);
  // unpacked7 = 0123 456X
  unpacked7 = vsriq_n_u8(unpacked7, unpacked1, 6);
  unpacked1 = vandq_u8(unpacked1, mask);

  unpacked0 = vld1q_u8(packed);
  // unpacked7 = 0123 4567
  unpacked7 = vsriq_n_u8(unpacked7, unpacked0, 7);
  unpacked0 = vandq_u8(unpacked0, mask);
}

} // namespace internal
} // namespace bitpacking
} // namespace torchao

#endif // defined(__aarch64__) || defined(__ARM_NEON)
