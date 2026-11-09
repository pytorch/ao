// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <torchao/csrc/cpu/torch_free_kernels/weight_packing/weight_packing.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// Portable (non-NEON) implementation of the channelwise 8-bit activation,
// groupwise low-bit weight linear kernel. It consumes the same packed weights
// as the aarch64 neondot kernels (universal and LUT formats, nr=8, kr=16,
// sr=2), so a PTE exported on Arm runs unchanged on x86, and computes the same
// integer dot products: activations are quantized per row exactly as on Arm.
namespace torchao::kernels::cpu::fallback::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight {

namespace internal {

// Asymmetric int8 per-row activation quantization, matching
// torchao::quantization::get_scale_and_zero and the aarch64 quantize (fused
// multiply-add, round half to even).
TORCHAO_ALWAYS_INLINE inline void
quantize_row(int8_t* qvals, const float* vals, int size, float scale, int8_t zero) {
  constexpr int qmin = -128;
  constexpr int qmax = 127;
  float inv_scale = 1.0 / (scale + 1e-16);
  float zero_f = zero;
  for (int i = 0; i < size; ++i) {
    int32_t q =
        static_cast<int32_t>(std::nearbyint(std::fma(vals[i], inv_scale, zero_f)));
    qvals[i] = static_cast<int8_t>(std::max(qmin, std::min(q, qmax)));
  }
}

TORCHAO_ALWAYS_INLINE inline void
row_scale_and_zero(float& scale, int& zero, const float* vals, int size) {
  constexpr int qmin = -128;
  constexpr int qmax = 127;
  float vmin = vals[0];
  float vmax = vals[0];
  for (int i = 1; i < size; ++i) {
    vmin = std::min(vmin, vals[i]);
    vmax = std::max(vmax, vals[i]);
  }
  if (vmin == vmax) {
    // A constant row (e.g. all zeros) would give a zero scale. Widen the range
    // to include 0 so the row still dequantizes exactly.
    vmin = std::min(vmin, 0.0f);
    vmax = std::max(vmax, 0.0f);
    if (vmin == vmax) {
      scale = 1.0f;
      zero = 0;
      return;
    }
  }
  scale = (vmax - vmin) / (qmax - qmin);
  // Deliberately not clamped to int8: the aarch64 packer narrows the same
  // unclamped value, and packed activations must match it bit for bit.
  zero = qmin - std::round(vmin / scale);
}

inline int32_t compute_sum(const int8_t* vals, int size) {
  int32_t sum = 0;
  for (int i = 0; i < size; ++i) {
    sum += vals[i];
  }
  return sum;
}

inline size_t packed_activations_row_size(
    int k,
    int group_size,
    bool has_weight_zeros) {
  // scale, zero, k qvals, and with weight zeros one qvals sum per group
  size_t row_size = sizeof(float) + sizeof(int8_t) + k;
  if (has_weight_zeros) {
    row_size += sizeof(int32_t) * (k / group_size);
  }
  return row_size;
}

// Unpacks the 8 columns x 16 values that the packing params (nr=8, kr=16,
// sr=2) interleave into one 128-value bitpacked chunk. Output is column major:
// weight_qvals[j * 16 + i] is value i of column j.
template <int weight_nbit, bool has_lut>
TORCHAO_ALWAYS_INLINE inline void unpack_chunk(
    int8_t* weight_qvals,
    const uint8_t* packed,
    const int8_t* lut) {
  std::array<uint8_t, 128> idxs{};
  std::array<int8_t, 128> vals{};
  torchao::kernels::cpu::fallback::bitpacking::internal::unpack_128_uint_values<weight_nbit>(idxs.data(), packed);
  for (int i = 0; i < 128; ++i) {
    if constexpr (has_lut) {
      vals[i] = lut[idxs[i]];
    } else if constexpr (weight_nbit < 8) {
      vals[i] = static_cast<int8_t>(idxs[i]) - (1 << (weight_nbit - 1));
    } else {
      vals[i] = static_cast<int8_t>(idxs[i]);
    }
  }
  torchao::weight_packing::unpack_values(
      weight_qvals, vals.data(), /*nr=*/8, /*kr=*/16, /*sr=*/2);
}

} // namespace internal

inline size_t packed_activations_size(
    int m,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  (void)kr;
  (void)sr;
  if (mr > 1) {
    m = ((m + mr - 1) / mr) * mr;
  }
  return internal::packed_activations_row_size(k, group_size, has_weight_zeros) *
      m;
}

inline size_t packed_activations_offset(
    int m_idx,
    int k,
    int group_size,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  assert(m_idx % mr == 0);
  return (m_idx / mr) *
      packed_activations_size(mr, k, group_size, has_weight_zeros, mr, kr, sr);
}

// Row layout (same as aarch64 pack_activations with mr = 1):
// float scale, int8 zero, then k int8 qvals; with weight zeros, the qvals of
// each group are followed by their int32 sum.
template <int mr_, int kr_, int sr_>
void pack_activations(
    void* packed_activations,
    int m,
    int k,
    int group_size,
    const float* activations,
    bool has_weight_zeros,
    int mr,
    int kr,
    int sr) {
  static_assert(mr_ == 1);
  (void)mr;
  (void)kr;
  (void)sr;
  assert(k % group_size == 0);
  auto curr_rounding_mode = std::fegetround();
  std::fesetround(FE_TONEAREST);
  auto ptr = reinterpret_cast<char*>(packed_activations);
  for (int m_idx = 0; m_idx < m; m_idx++) {
    float scale;
    int zero;
    internal::row_scale_and_zero(scale, zero, activations, k);
    std::memcpy(ptr, &scale, sizeof(float));
    ptr += sizeof(float);
    // Arm stores and quantizes with the zero point narrowed to int8.
    auto zero_i8 = static_cast<int8_t>(zero);
    *reinterpret_cast<int8_t*>(ptr) = zero_i8;
    ptr += sizeof(int8_t);
    if (has_weight_zeros) {
      for (int k_idx = 0; k_idx < k; k_idx += group_size) {
        auto qvals = reinterpret_cast<int8_t*>(ptr);
        internal::quantize_row(
            qvals, activations + k_idx, group_size, scale, zero_i8);
        int32_t qvals_sum = internal::compute_sum(qvals, group_size);
        ptr += group_size;
        std::memcpy(ptr, &qvals_sum, sizeof(int32_t));
        ptr += sizeof(int32_t);
      }
    } else {
      internal::quantize_row(
          reinterpret_cast<int8_t*>(ptr), activations, k, scale, zero_i8);
      ptr += k;
    }
    activations += k;
  }
  std::fesetround(curr_rounding_mode);
}

// Computes output = F(activations * weights + bias) for m rows and n columns,
// F being clamp (has_clamp) or identity. Per group, each column accumulates
//   weight_scale * activation_scale * (qval_dot - activation_zero * weight_qvals_sum
//       - weight_zero * activation_qvals_sum
//       + group_size * weight_zero * activation_zero)
// Each 8 x 16 weight chunk is unpacked once and applied to all m rows.
template <int weight_nbit, bool has_weight_zeros, bool has_lut>
void kernel_1x8x16_f32_fallback(
    float* output,
    int output_m_stride,
    int m,
    int n,
    int k,
    int group_size,
    const void* packed_weights,
    const void* packed_activations,
    float clamp_min,
    float clamp_max,
    bool has_weight_zeros_runtime,
    bool has_bias,
    bool has_clamp) {
  (void)has_weight_zeros_runtime;
  assert(k % group_size == 0);
  assert(group_size % 16 == 0);
  constexpr int nr = 8;
  constexpr int bytes_per_chunk = 16 * weight_nbit;

  const size_t act_row_size =
      internal::packed_activations_row_size(k, group_size, has_weight_zeros);
  const auto act_data = reinterpret_cast<const char*>(packed_activations);
  std::vector<float> act_scales(m);
  std::vector<int32_t> act_zeros(m);
  for (int r = 0; r < m; r++) {
    std::memcpy(&act_scales[r], act_data + r * act_row_size, sizeof(float));
    act_zeros[r] = *reinterpret_cast<const int8_t*>(
        act_data + r * act_row_size + sizeof(float));
  }
  const char* act_qvals = act_data + sizeof(float) + sizeof(int8_t);
  // Offset of a group's qvals within a row
  const size_t act_group_stride =
      has_weight_zeros ? group_size + sizeof(int32_t) : group_size;

  std::vector<float> res(m * nr);
  std::vector<int32_t> acc(m * nr);
  std::array<int8_t, 16> lut{};
  std::array<int8_t, nr * 16> weight_qvals{};

  auto w = reinterpret_cast<const char*>(packed_weights);
  for (int n_idx = 0; n_idx < n; n_idx += nr) {
    if constexpr (has_lut) {
      std::memcpy(lut.data(), w, 16);
      w += 16;
    }
    std::fill(res.begin(), res.end(), 0.0f);
    for (int group_idx = 0; group_idx < k / group_size; group_idx++) {
      std::fill(acc.begin(), acc.end(), 0);
      const size_t group_offset = group_idx * act_group_stride;
      for (int i = 0; i < group_size; i += 16) {
        internal::unpack_chunk<weight_nbit, has_lut>(
            weight_qvals.data(), reinterpret_cast<const uint8_t*>(w), lut.data());
        w += bytes_per_chunk;
        for (int r = 0; r < m; r++) {
          const auto a = reinterpret_cast<const int8_t*>(
              act_qvals + r * act_row_size + group_offset + i);
          int32_t* acc_r = acc.data() + r * nr;
          for (int j = 0; j < nr; j++) {
            const int8_t* wj = weight_qvals.data() + j * 16;
            int32_t dot = 0;
            for (int t = 0; t < 16; t++) {
              dot += static_cast<int32_t>(a[t]) * static_cast<int32_t>(wj[t]);
            }
            acc_r[j] += dot;
          }
        }
      }

      std::array<float, nr> weight_scales{};
      std::array<int32_t, nr> weight_qvals_sums{};
      std::array<int32_t, nr> weight_zeros{};
      std::memcpy(weight_scales.data(), w, sizeof(weight_scales));
      w += sizeof(weight_scales);
      std::memcpy(weight_qvals_sums.data(), w, sizeof(weight_qvals_sums));
      w += sizeof(weight_qvals_sums);
      if constexpr (has_weight_zeros) {
        std::memcpy(weight_zeros.data(), w, sizeof(weight_zeros));
        w += sizeof(weight_zeros);
      }

      for (int r = 0; r < m; r++) {
        const int32_t act_zero = act_zeros[r];
        int32_t act_qvals_sum = 0;
        if constexpr (has_weight_zeros) {
          std::memcpy(
              &act_qvals_sum,
              act_qvals + r * act_row_size + group_offset + group_size,
              sizeof(int32_t));
        }
        for (int j = 0; j < nr; j++) {
          int32_t tmp = acc[r * nr + j] - act_zero * weight_qvals_sums[j];
          if constexpr (has_weight_zeros) {
            tmp = tmp - weight_zeros[j] * act_qvals_sum +
                weight_zeros[j] * group_size * act_zero;
          }
          const float scale_factor = weight_scales[j] * act_scales[r];
          res[r * nr + j] += scale_factor * static_cast<float>(tmp);
        }
      }
    }

    if (has_bias) {
      std::array<float, nr> bias{};
      std::memcpy(bias.data(), w, sizeof(bias));
      w += sizeof(bias);
      for (int r = 0; r < m; r++) {
        for (int j = 0; j < nr; j++) {
          res[r * nr + j] += bias[j];
        }
      }
    }
    const int cols = std::min(nr, n - n_idx);
    for (int r = 0; r < m; r++) {
      float* out = output + r * output_m_stride + n_idx;
      for (int j = 0; j < cols; j++) {
        float v = res[r * nr + j];
        if (has_clamp) {
          v = std::min(std::max(v, clamp_min), clamp_max);
        }
        out[j] = v;
      }
    }
  }
}

// Portable version of the aarch64 pack_weights_with_lut: same layout, so the
// packed weights are byte-identical to an Arm export. Per nr columns: a
// 16-byte LUT, then per group the bitpacked LUT indices (kr-value chunks
// interleaved by pack_values), nr scales, nr sums of the mapped values, and nr
// zeros if present; finally nr biases if present.
template <int weight_nbit, int nr_, int kr_, int sr_>
void pack_weights_with_lut(
    void* packed_weights,
    int n,
    int k,
    int group_size,
    const int8_t* weight_qval_idxs,
    int n_luts,
    const int8_t* luts,
    const float* weight_scales,
    const int8_t* weight_zeros,
    const float* bias,
    int nr,
    int kr,
    int sr) {
  static_assert(weight_nbit >= 1 && weight_nbit <= 4);
  static_assert(nr_ * kr_ == 128);
  (void)nr;
  (void)kr;
  (void)sr;
  assert(k % group_size == 0);
  assert(group_size % kr_ == 0);
  assert(n % n_luts == 0);
  constexpr int lut_size = 1 << weight_nbit;
  constexpr int packed_buffer_bytes = weight_nbit * nr_ * kr_ / 8;
  const bool has_weight_zeros = (weight_zeros != nullptr);
  const bool has_bias = (bias != nullptr);
  const int groups_per_k = k / group_size;
  const int cols_per_lut = n / n_luts;
  assert((cols_per_lut == n) || (cols_per_lut % nr_ == 0));

  std::array<int8_t, 16> lut{};
  std::array<int8_t, nr_ * kr_> buffer;
  std::array<int8_t, nr_ * kr_> packed_values;
  std::array<int32_t, nr_> qvals_sum;
  auto ptr = reinterpret_cast<char*>(packed_weights);

  for (int n_idx = 0; n_idx < n; n_idx += nr_) {
    for (int group_idx = 0; group_idx < groups_per_k; group_idx++) {
      if (group_idx == 0) {
        lut.fill(0);
        std::memcpy(
            lut.data(), luts + (n_idx / cols_per_lut) * lut_size, lut_size);
        std::memcpy(ptr, lut.data(), 16);
        ptr += 16;
      }
      qvals_sum.fill(0);
      const int k_idx = group_idx * group_size;
      for (int idx_in_group = 0; idx_in_group < group_size;
           idx_in_group += kr_) {
        buffer.fill(0);
        for (int j = 0; j < nr_; j++) {
          if (n_idx + j < n) {
            const int8_t* idxs =
                weight_qval_idxs + (n_idx + j) * k + (k_idx + idx_in_group);
            std::memcpy(buffer.data() + kr_ * j, idxs, kr_);
            for (int t = 0; t < kr_; t++) {
              qvals_sum[j] += lut[static_cast<uint8_t>(idxs[t])];
            }
          }
        }
        torchao::weight_packing::pack_values(
            packed_values.data(), buffer.data(), nr_, kr_, sr_);
        torchao::kernels::cpu::fallback::bitpacking::internal::pack_128_uint_values<weight_nbit>(
            reinterpret_cast<uint8_t*>(ptr),
            reinterpret_cast<const uint8_t*>(packed_values.data()));
        ptr += packed_buffer_bytes;
      }
      for (int j = 0; j < nr_; j++) {
        float scale = 0.0f;
        if (n_idx + j < n) {
          scale = weight_scales[(n_idx + j) * groups_per_k + group_idx];
        }
        std::memcpy(ptr, &scale, sizeof(float));
        ptr += sizeof(float);
      }
      for (int j = 0; j < nr_; j++) {
        std::memcpy(ptr, &qvals_sum[j], sizeof(int32_t));
        ptr += sizeof(int32_t);
      }
      if (has_weight_zeros) {
        for (int j = 0; j < nr_; j++) {
          int32_t zero = 0;
          if (n_idx + j < n) {
            zero = weight_zeros[(n_idx + j) * groups_per_k + group_idx];
          }
          std::memcpy(ptr, &zero, sizeof(int32_t));
          ptr += sizeof(int32_t);
        }
      }
    }
    if (has_bias) {
      for (int j = 0; j < nr_; j++) {
        float bias_ = 0.0f;
        if (n_idx + j < n) {
          bias_ = bias[n_idx + j];
        }
        std::memcpy(ptr, &bias_, sizeof(float));
        ptr += sizeof(float);
      }
    }
  }
}

inline size_t packed_weights_with_lut_size(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  (void)kr;
  (void)sr;
  auto col_size = torchao::weight_packing::packed_weights_size_per_n(
      k, group_size, weight_nbit, has_weight_zeros, has_bias);
  n = ((n + nr - 1) / nr) * nr;
  // One 16-byte LUT per nr columns
  return col_size * n + (n / nr) * 16;
}

inline size_t packed_weights_with_lut_offset(
    int n_idx,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr,
    int kr,
    int sr) {
  assert(n_idx % nr == 0);
  return (n_idx / nr) *
      packed_weights_with_lut_size(
             nr, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr);
}

namespace internal {

TORCHAO_ALWAYS_INLINE inline void dequantize_and_store_values(
    float* out,
    const int8_t* qvals,
    int count,
    float scale,
    float zero) {
  for (int i = 0; i < count; ++i) {
    out[i] = (static_cast<float>(qvals[i]) - zero) * scale;
  }
}

} // namespace internal

// Shared embedding op that uses linear packed weights
template <int weight_nbit, int nr, int kr, int sr>
inline void shared_embedding(
    float* out,
    const void* packed_weights,
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias,
    int index) {
  assert(k % group_size == 0);
  assert(group_size % 16 == 0);

  int groups_per_k = k / group_size;
  std::vector<int8_t> weight_qvals(k * nr);
  std::vector<float> weight_scales(groups_per_k * nr);
  std::vector<int8_t> weight_zeros(groups_per_k * nr);
  std::vector<float> bias(nr);

  int n_idx = index / nr;
  n_idx = n_idx * nr;
  int j = index - n_idx;

  torchao::weight_packing::unpack_weights_at_n_idx<weight_nbit, nr, kr, sr>(
      weight_qvals.data(),
      weight_scales.data(),
      has_weight_zeros ? weight_zeros.data() : nullptr,
      has_bias ? bias.data() : nullptr,
      n_idx,
      n,
      k,
      group_size,
      has_weight_zeros,
      has_bias,
      packed_weights);

  for (int i = 0; i < k; i += 16) {
    int chunk_size = std::min(16, k - i);
    float scale = weight_scales[j * groups_per_k + i / group_size];
    float zero = 0.0f;
    if (has_weight_zeros) {
      zero =
          static_cast<float>(weight_zeros[j * groups_per_k + i / group_size]);
    }
    internal::dequantize_and_store_values(
        out + i, weight_qvals.data() + j * k + i, chunk_size, scale, zero);
  }
}

} // namespace
  // torchao::kernels::cpu::fallback::linear::channelwise_8bit_activation_groupwise_lowbit_weight
