// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/quantization/quantize.h>
#include <algorithm>
#include <cassert>
#include <cfenv>
#include <cmath>

void torchao::quantization::get_qvals_range(
    int& qmin,
    int& qmax,
    int nbit,
    bool is_symmetric) {
  if (is_symmetric) {
    qmin = -(1 << (nbit - 1)) + 1;
    qmax = -qmin;
  } else {
    qmin = -(1 << (nbit - 1));
    qmax = (1 << (nbit - 1)) - 1;
  }
}

float torchao::quantization::get_scale(
    float vmin,
    float vmax,
    int qmin,
    int qmax) {
  assert(qmin < qmax);
  assert(vmin < vmax);
  return (vmax - vmin) / (qmax - qmin);
}

void torchao::quantization::get_scale_and_zero(
    float& scale,
    int& zero,
    float vmin,
    float vmax,
    int qmin,
    int qmax) {
  scale = torchao::quantization::get_scale(vmin, vmax, qmin, qmax);
  zero = qmin - std::round(vmin / scale);
}

namespace {
inline void
_vec_clip_inplace(int32x4_t& vec, int32x4_t vec_min, int32x4_t vec_max) {
  vec = vmaxq_s32(vec, vec_min);
  vec = vminq_s32(vec, vec_max);
}
} // namespace

void torchao::kernels::cpu::aarch64::quantization::quantize(
    // Output
    int8_t* qvals,
    // Inputs
    const float32_t* vals,
    int size,
    float32_t scale,
    int8_t zero,
    int8_t qmin,
    int8_t qmax) {
  float32_t invScale = 1.0 / (scale + 1e-16);
  float32x4_t vec_zero = vdupq_n_f32(zero);
  float32x4_t vec_invScale = vdupq_n_f32(invScale);
  int32x4_t vec_qmin = vdupq_n_s32(qmin);
  int32x4_t vec_qmax = vdupq_n_s32(qmax);

  float32x4_t vec_val;
  float32x4_t vec_qval_f32;
  int32x4_t vec_qval_s32;
  int16x4_t vec_qval_s16_0;
  int16x4_t vec_qval_s16_1;

  int i = 0;
  for (; (i + 8) < size; i += 8) {
    //////////////////////////////////////
    // Quantize first 4 element chunk to int16
    //////////////////////////////////////
    vec_val = vld1q_f32(vals + i);

    // Quantize and round
    vec_qval_f32 = vfmaq_f32(vec_zero, vec_val, vec_invScale);
    vec_qval_s32 = vcvtnq_s32_f32(vec_qval_f32);

    _vec_clip_inplace(vec_qval_s32, vec_qmin, vec_qmax);

    vec_qval_s16_0 = vqmovn_s32(vec_qval_s32);

    //////////////////////////////////////
    // Quantize second 4 element chunk to int16
    //////////////////////////////////////
    vec_val = vld1q_f32(vals + i + 4);

    // Quantize and round
    vec_qval_f32 = vfmaq_f32(vec_zero, vec_val, vec_invScale);
    vec_qval_s32 = vcvtnq_s32_f32(vec_qval_f32);

    _vec_clip_inplace(vec_qval_s32, vec_qmin, vec_qmax);

    vec_qval_s16_1 = vqmovn_s32(vec_qval_s32);

    //////////////////////////////////////
    // Store 8 quantized elements
    //////////////////////////////////////
    int16x8_t vec_qval_s16_01 = vcombine_s16(vec_qval_s16_0, vec_qval_s16_1);
    int8x8_t vec_qval_s8_01 = vqmovn_s16(vec_qval_s16_01);
    vst1_s8(qvals + i, vec_qval_s8_01);
  }
  auto curr_rounding_mode = fegetround();
  fesetround(FE_TONEAREST);
  for (; i < size; ++i) {
    // Quantize remaining elements using scalar code
    float32_t val = vals[i];
    float32_t qval_f32 = zero + val * invScale;
    int32_t qval_s32 = static_cast<int32_t>(std::nearbyint(qval_f32));

    // Clip to qmin and qmax
    qval_s32 = std::max(
        static_cast<int32_t>(qmin),
        std::min(qval_s32, static_cast<int32_t>(qmax)));

    // Store the quantized value
    qvals[i] = static_cast<int8_t>(qval_s32);
  }
  fesetround(int(curr_rounding_mode));
}

#endif // defined(__aarch64__) || defined(__ARM_NEON)
