// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>

namespace torchao::lut {

TORCHAO_ALWAYS_INLINE inline void load_fp32_lut(uint8x16x4_t& lut, const float* table) {
  lut = {
      vld1q_u8((const uint8_t*)&table[0]),
      vld1q_u8((const uint8_t*)&table[4]),
      vld1q_u8((const uint8_t*)&table[8]),
      vld1q_u8((const uint8_t*)&table[12])
  };
}

// This function looks up float values from a 16-value LUT
// (stored as 16 consecutive floats loaded into uint8x16x4_t)
// The indices of the 16 values being looked up are contained in idx
// These values are output to out0, out1, out2, and out3
TORCHAO_ALWAYS_INLINE inline void lookup_from_fp32_lut(
  float32x4_t& out0,
  float32x4_t& out1,
  float32x4_t& out2,
  float32x4_t& out3,
  const uint8x16x4_t& lut,
  const uint8x16_t idx
) {
  // Performs a vectorized lookup of FP32 values from a 16-element float table.
  // The input `idx` is a uint8x16_t vector containing 16 indices (0–15),
  // each selecting a float from the LUT. Since each float is 4 bytes, we compute
  // the byte offsets for each selected float:
  //    - `idx0` = idx * 4       (byte 0 of each float)
  //    - `idx1` = idx0 + 1      (byte 1)
  //    - `idx2` = idx0 + 2      (byte 2)
  //    - `idx3` = idx0 + 3      (byte 3)
  //
  // These are grouped into a 4-way NEON table `idx_tbl = {idx0, idx1, idx2, idx3}`.
  //
  // To reconstruct full FP32 values (4 bytes each) from the byte lookup, we use
  // `vqtbl4q_u8(idx_tbl, ...)` with a special interleaving `offsets` vector:
  //   - `offsets = { 0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51 }`
  //
  // This offset pattern selects the 4 bytes for float0 (0, 16, 32, 48), float1 (1, 17, 33, 49), etc.
  //
  // We repeat this with offset vectors incremented by 4 and 8 and 12 to produce
  // `out1_idx`, `out2_idx`, and `out3_idx`, each forming the byte indices for
  // the next group of 4 floats.
  //
  // Finally, we use `vqtbl4q_u8(lut, outN_idx)` to gather bytes from the original LUT,
  // and `vreinterpretq_f32_u8(...)` to convert the byte-wise result into
  // actual `float32x4_t` values: `out0`, `out1`, `out2`, and `out3`

  uint8x16_t idx0 = vshlq_n_u8(idx, 2);
  uint8x16_t idx1 = vaddq_u8(idx0, vdupq_n_u8(1));
  uint8x16_t idx2 = vaddq_u8(idx0, vdupq_n_u8(2));
  uint8x16_t idx3 = vaddq_u8(idx0, vdupq_n_u8(3));

  // 4-way interleave idx0, idx1, idx2, idx3 to create out0_idx, out1_idx, out2_idx, out3_idx
  uint8x16x4_t idx_tbl = {idx0, idx1, idx2, idx3};
  uint8x16_t offsets = { 0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51 };
  uint8x16_t out0_idx = vqtbl4q_u8(idx_tbl, offsets);
  uint8x16_t out1_idx = vqtbl4q_u8(idx_tbl, vaddq_u8(offsets, vdupq_n_u8(4)));
  uint8x16_t out2_idx = vqtbl4q_u8(idx_tbl, vaddq_u8(offsets, vdupq_n_u8(8)));
  uint8x16_t out3_idx = vqtbl4q_u8(idx_tbl, vaddq_u8(offsets, vdupq_n_u8(12)));

  out0 = vreinterpretq_f32_u8(vqtbl4q_u8(lut, out0_idx));
  out1 = vreinterpretq_f32_u8(vqtbl4q_u8(lut, out1_idx));
  out2 = vreinterpretq_f32_u8(vqtbl4q_u8(lut, out2_idx));
  out3 = vreinterpretq_f32_u8(vqtbl4q_u8(lut, out3_idx));
}

} // namespace torchao::lut


#endif // defined(__aarch64__) || defined(__ARM_NEON)
