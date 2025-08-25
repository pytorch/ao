#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/csrc/cpu/torch_free_kernels/aarch64/bitpacking/bitpack.h>
#include <torchao/csrc/cpu/torch_free_kernels/macro.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/reduction/reduction.h>
#include <torchao/csrc/cpu/torch_free_kernels/aarch64/packing/utils.h>
#include <array>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear::
    channelwise_8bit_activation_groupwise_lowbit_weight::weight_packing {

namespace internal {

// Packs a buffer of (kr * nr) lowbit values (stored as int8_t) down to bits
template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void pack_buffer(
    void* packed_weights,
    const int8_t* buffer) {
  if constexpr (kr * nr == 128) {
    bitpacking::vec_pack_128_lowbit_values<weight_nbit>(
        (uint8_t*)packed_weights,
        vld1q_s8(buffer),
        vld1q_s8(buffer + 16),
        vld1q_s8(buffer + 32),
        vld1q_s8(buffer + 48),
        vld1q_s8(buffer + 64),
        vld1q_s8(buffer + 80),
        vld1q_s8(buffer + 96),
        vld1q_s8(buffer + 112));
    return;
  }
  if constexpr (kr * nr == 64) {
    bitpacking::vec_pack_64_lowbit_values<weight_nbit>(
        (uint8_t*)packed_weights,
        vld1q_s8(buffer),
        vld1q_s8(buffer + 16),
        vld1q_s8(buffer + 32),
        vld1q_s8(buffer + 48));
    return;
  }
  if constexpr (kr * nr == 32) {
    bitpacking::vec_pack_32_lowbit_values<weight_nbit>(
        (uint8_t*)packed_weights, vld1q_s8(buffer), vld1q_s8(buffer + 16));
    return;
  }
  assert(false);
}

template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void pack_buffer_for_lut(
    void* packed_weights,
    const int8_t* buffer) {
  static_assert(weight_nbit >= 1);
  static_assert(weight_nbit <= 4);
  const uint8_t* buffer_u8 = reinterpret_cast<const uint8_t*>(buffer);
  if constexpr (kr * nr == 128) {
    bitpacking::vec_pack_128_uintx_values<weight_nbit>(
        (uint8_t*)packed_weights,
        vld1q_u8(buffer_u8),
        vld1q_u8(buffer_u8 + 16),
        vld1q_u8(buffer_u8 + 32),
        vld1q_u8(buffer_u8 + 48),
        vld1q_u8(buffer_u8 + 64),
        vld1q_u8(buffer_u8 + 80),
        vld1q_u8(buffer_u8 + 96),
        vld1q_u8(buffer_u8 + 112));
    return;
  }
  assert(false);
}

// Unpacks bits to a buffer of (kr * nr) lowbit values (stored as int8_t)
template <int weight_nbit, int kr, int nr>
TORCHAO_ALWAYS_INLINE inline void unpack_buffer(
    int8_t* buffer,
    const void* packed_weights) {
  int8x16_t vals0;
  int8x16_t vals1;
  int8x16_t vals2;
  int8x16_t vals3;
  int8x16_t vals4;
  int8x16_t vals5;
  int8x16_t vals6;
  int8x16_t vals7;

  if constexpr (kr * nr == 128) {
    bitpacking::vec_unpack_128_lowbit_values<weight_nbit>(
        vals0,
        vals1,
        vals2,
        vals3,
        vals4,
        vals5,
        vals6,
        vals7,
        (const uint8_t*)packed_weights);
    vst1q_s8(buffer, vals0);
    vst1q_s8(buffer + 16, vals1);
    vst1q_s8(buffer + 32, vals2);
    vst1q_s8(buffer + 48, vals3);
    vst1q_s8(buffer + 64, vals4);
    vst1q_s8(buffer + 80, vals5);
    vst1q_s8(buffer + 96, vals6);
    vst1q_s8(buffer + 112, vals7);
    return;
  }
  if constexpr (kr * nr == 64) {
    torchao::bitpacking::vec_unpack_64_lowbit_values<weight_nbit>(
        vals0, vals1, vals2, vals3, (const uint8_t*)packed_weights);
    vst1q_s8(buffer, vals0);
    vst1q_s8(buffer + 16, vals1);
    vst1q_s8(buffer + 32, vals2);
    vst1q_s8(buffer + 48, vals3);
    return;
  }
  if constexpr (kr * nr == 32) {
    bitpacking::vec_unpack_32_lowbit_values<weight_nbit>(
        vals0, vals1, (const uint8_t*)packed_weights);
    vst1q_s8(buffer, vals0);
    vst1q_s8(buffer + 16, vals1);
    return;
  }
  assert(false);
}

// Size in bytes of 1 packed weights column
size_t inline packed_weights_size_per_n(
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias) {
  assert(k % group_size == 0);
  int groups_per_col = k / group_size;
  int col_size = 0;

  // qvals
  col_size += (k / 8) * weight_nbit;

  // scales
  col_size += sizeof(float) * groups_per_col;

  // qvals_sum
  col_size += sizeof(int32_t) * groups_per_col;

  // zeros
  if (has_weight_zeros) {
    col_size += sizeof(int32_t) * groups_per_col;
  }

  // bias
  if (has_bias) {
    col_size += sizeof(float);
  }

  return col_size;
}

TORCHAO_ALWAYS_INLINE inline void
map_values(int8_t* dst, int8_t* src, int8x16_t lut, int size) {
  // src will be in range 0 to 16, which fits in int8_t
  assert(size % 16 == 0);
  for (int i = 0; i < size; i += 16) {
    uint8x16_t idx = vreinterpretq_u8_s8(vld1q_s8(src + i));
    vst1q_s8(dst + i, vqtbl1q_s8(lut, idx));
  }
}

// Call pack_weights every n_step columns

template <int weight_nbit, int nr, int kr, int sr, bool has_lut>
TORCHAO_ALWAYS_INLINE inline void pack_weights_impl(
    // Output
    void* packed_weights,
    // Inputs
    int n,
    int k,
    int group_size,
    // weight_ints holds weight_qvals if has_lut is false
    // Otherwise it holds weight_qval_idxs
    const int8_t* weight_ints,
    // number of luts, not used if has_lut is false
    // must be nr group or coarser (per tensor)
    int n_luts,
    const int8_t*
        luts, // luts (each 2**weight_nbit values), not used if has_lut is false
    const float* weight_scales,
    // weight_zeros not packed if nullptr
    const int8_t* weight_zeros,
    // bias not packed if nullptr
    const float* bias) {
  if constexpr (!has_lut) {
    (void)n_luts; // unused
    (void)luts; // unused
  }

  assert(k % group_size == 0);
  assert(group_size % kr == 0);
  bool has_weight_zeros = (weight_zeros != nullptr);
  bool has_bias = (bias != nullptr);
  int groups_per_k = k / group_size;

  // LUT has size lut_size, which is <= 16
  // If lut_size < 16, we extend it with 0s in lut_buffer
  int8x16_t lut;
  constexpr int lut_size = (1 << weight_nbit);
  std::array<int8_t, 16> lut_buffer;
  int cols_per_lut;

  // Buffer to hold kr qvals, mapped with LUT from qval_idxs
  std::array<int8_t, kr> mapped_val_buffer;
  if constexpr (has_lut) {
    static_assert(weight_nbit <= 4);
    static_assert(lut_size <= 16);
    lut_buffer.fill(0);

    assert(n % n_luts == 0);
    cols_per_lut = n / n_luts;
    assert((cols_per_lut == n) || (cols_per_lut % nr == 0));
  } else {
    (void)lut; // unused
    (void)lut_size; // unused
    (void)lut_buffer; // unused
    (void)cols_per_lut; // unused
    (void)mapped_val_buffer; // unused
  }

  // Buffer to hold (kr * nr) values
  std::array<int8_t, nr * kr> buffer;

  // Buffer to hold (kr * nr) values after theose values
  // are packed by params (nr, kr, sr)
  int8_t packed_values[buffer.size()];

  // Bytes of packed buffer of (nr * kr) values
  assert(nr * kr % 8 == 0);
  constexpr int packed_buffer_bytes = weight_nbit * nr * kr / 8;

  // Buffer to hold sum of weight_qvals in each column group
  std::array<int, nr> qvals_sum;

  // Data pointer for packed weights
  auto packed_weights_byte_ptr = (char*)packed_weights;

  // Loop over n by nr
  for (int n_idx = 0; n_idx < n; n_idx += nr) {
    // Look over groups along k
    for (int group_idx = 0; group_idx < groups_per_k; group_idx++) {
      // Populate lut and write it out to packed_weights
      if constexpr (has_lut) {
        // Set lut variable and write lut for nr group
        if (group_idx == 0) {
          int lut_idx = n_idx / cols_per_lut;
          std::memcpy(lut_buffer.data(), luts + lut_idx * lut_size, lut_size);
          lut = vld1q_s8(lut_buffer.data());
          vst1q_s8((int8_t*)packed_weights_byte_ptr, lut);
          packed_weights_byte_ptr += 16;
        }
      }

      // Initialize qvals_sum for each group to 0
      qvals_sum.fill(0);

      // Loop over group by kr and pack the weights for the next nr columns
      int k_idx = group_idx * group_size;
      for (int idx_in_group = 0; idx_in_group < group_size;
           idx_in_group += kr) {
        // Fill buffer with next kr values from the next nr columns
        // If there are fewer than nr columns, 0s are stored
        buffer.fill(0);
        for (int j = 0; j < nr; j++) {
          if (n_idx + j < n) {
            std::memcpy(
                buffer.data() + kr * j,
                weight_ints + (n_idx + j) * k + (k_idx + idx_in_group),
                kr);
            if constexpr (has_lut) {
              internal::map_values(
                  mapped_val_buffer.data(), buffer.data() + kr * j, lut, kr);
              qvals_sum[j] +=
                  reduction::compute_sum(mapped_val_buffer.data(), kr);
            } else {
              qvals_sum[j] +=
                  reduction::compute_sum(buffer.data() + kr * j, kr);
            }
          }
        }

        // Pack buffer
        torchao::packing::pack_values(packed_values, buffer.data(), nr, kr, sr);
        if constexpr (has_lut) {
          internal::pack_buffer_for_lut<weight_nbit, kr, nr>(
              packed_weights_byte_ptr, packed_values);
        } else {
          internal::pack_buffer<weight_nbit, kr, nr>(
              packed_weights_byte_ptr, packed_values);
        }
        packed_weights_byte_ptr += packed_buffer_bytes;
      } // loop over group (idx_in_group)

      // Store group attributes scale, qval_sums, and zeros for next nr columns
      // If there are fewer than nr columns, 0s are stored

      // Store weight scales
      for (int j = 0; j < nr; j++) {
        float32_t scale = 0.0;
        if (n_idx + j < n) {
          scale = weight_scales[(n_idx + j) * groups_per_k + group_idx];
        }
        *((float*)packed_weights_byte_ptr) = scale;
        packed_weights_byte_ptr += sizeof(float);
      }

      // Store weight qval sums
      for (int j = 0; j < nr; j++) {
        *((int*)packed_weights_byte_ptr) = qvals_sum[j];
        packed_weights_byte_ptr += sizeof(int);
      }

      // Store weight zeros
      if (has_weight_zeros) {
        for (int j = 0; j < nr; j++) {
          int32_t zero = 0;
          if (n_idx + j < n) {
            zero = weight_zeros[(n_idx + j) * groups_per_k + group_idx];
          }
          *((int32_t*)packed_weights_byte_ptr) = zero;
          packed_weights_byte_ptr += sizeof(int32_t);
        }
      }
    } // loop over k (group_idx)

    // Store bias for next nr columns
    if (has_bias) {
      for (int j = 0; j < nr; j++) {
        float bias_ = 0.0;
        if (n_idx + j < n) {
          bias_ = bias[n_idx + j];
        }
        *((float*)packed_weights_byte_ptr) = bias_;
        packed_weights_byte_ptr += sizeof(float);
      }
    }
  } // n_idx
}

} // namespace internal

template <int weight_nbit, int nr, int kr, int sr>
void pack_weights(
    // Output
    void* packed_weights,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qvals,
    const float* weight_scales,
    // weight_zeros not packed if nullptr
    const int8_t* weight_zeros,
    // bias not packed if nullptr
    const float* bias) {
  internal::pack_weights_impl<weight_nbit, nr, kr, sr, /*has_lut*/ false>(
      packed_weights,
      n,
      k,
      group_size,
      /*weight_ints*/ weight_qvals,
      /*n_luts*/ 0,
      /*luts*/ nullptr,
      weight_scales,
      weight_zeros,
      bias);
}

// Returns number of bytes required for weight_data
size_t inline packed_weights_size(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr) {
  auto packed_weights_size_per_n = internal::packed_weights_size_per_n(
      k, group_size, weight_nbit, has_weight_zeros, has_bias);

  // Replace n with next multiple of nr >= n
  n = ((n + nr - 1) / nr) * nr;
  return packed_weights_size_per_n * n;
}

// Unpack weights at n_idx to support shared embedding/unembedding
template <int weight_nbit, int nr, int kr, int sr>
void unpack_weights_at_n_idx(
    // Output
    int8_t* weight_qvals, // k * nr values at n_idx
    float* weight_scales, // groups_per_k * nr values at n_idx
    // weight_zeros is not extracted if has_weight_zeros is false
    int8_t* weight_zeros, // groups_per_k * nr values at n_idx
    // bias is not extracted if has_bias is false
    float* bias, // nr values at n_idx
    // Inputs
    int n_idx,
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias,
    const void* packed_weights) {
  assert(k % group_size == 0);
  assert(group_size % kr == 0);
  assert(n_idx % nr == 0);

  int groups_per_k = k / group_size;

  // Buffer to hold (kr * nr) values
  std::array<int8_t, nr * kr> buffer;

  // Buffer to hold (kr * nr) values after theose values
  // are packed by params (nr, kr, sr)
  int8_t packed_values[buffer.size()];

  // Bytes of packed buffer of (nr * kr) values
  assert(nr * kr % 8 == 0);
  constexpr int packed_buffer_bytes = weight_nbit * nr * kr / 8;

  // Data pointer for packed weights
  auto packed_weights_byte_ptr =
      ((char*)packed_weights +
       n_idx *
           internal::packed_weights_size_per_n(
               k, group_size, weight_nbit, has_weight_zeros, has_bias));

  // Look over groups along k
  for (int group_idx = 0; group_idx < groups_per_k; group_idx++) {
    // Loop over group by kr and pack the weights for the next nr columns
    int k_idx = group_idx * group_size;
    for (int idx_in_group = 0; idx_in_group < group_size; idx_in_group += kr) {
      // Unpack qvals
      internal::unpack_buffer<weight_nbit, kr, nr>(
          packed_values, packed_weights_byte_ptr);
      packed_weights_byte_ptr += packed_buffer_bytes;
      torchao::packing::unpack_values(buffer.data(), packed_values, nr, kr, sr);

      // Write weight_qvals
      for (int j = 0; j < nr; j++) {
        if (n_idx + j < n) {
          std::memcpy(
              weight_qvals + j * k + (k_idx + idx_in_group),
              buffer.data() + kr * j,
              kr);
        }
      }

    } // loop over group (idx_in_group)

    // Write group scales and zeros for next nr columns

    // Write weight scales
    for (int j = 0; j < nr; j++) {
      float scale = *((float*)packed_weights_byte_ptr);
      packed_weights_byte_ptr += sizeof(float);
      if (n_idx + j < n) {
        weight_scales[j * groups_per_k + group_idx] = scale;
      }
    }

    // Skip over weight qval sums
    packed_weights_byte_ptr += nr * sizeof(int);

    // Write weight zeros
    if (has_weight_zeros) {
      for (int j = 0; j < nr; j++) {
        int32_t zero = *((int32_t*)packed_weights_byte_ptr);
        packed_weights_byte_ptr += sizeof(int32_t);
        if (n_idx + j < n) {
          weight_zeros[j * groups_per_k + group_idx] = (int8_t)zero;
        }
      }
    }

  } // loop over k (group_idx)

  // Write bias
  if (has_bias) {
    for (int j = 0; j < nr; j++) {
      float bias_ = *((float*)packed_weights_byte_ptr);
      packed_weights_byte_ptr += sizeof(float);
      if (n_idx + j < n) {
        bias[j] = bias_;
      }
    }
  }
}

template <int weight_nbit, int nr, int kr, int sr>
void unpack_weights(
    // Output
    int8_t* weight_qvals,
    float* weight_scales,
    // weight_zeros is not extracted if has_weight_zeros is false
    int8_t* weight_zeros,
    // bias is not extracted if has_bias is false
    float* bias,
    // Inputs
    int n,
    int k,
    int group_size,
    bool has_weight_zeros,
    bool has_bias,
    const void* packed_weights) {
  assert(k % group_size == 0);
  assert(group_size % kr == 0);
  int groups_per_k = k / group_size;

  for (int n_idx = 0; n_idx < n; n_idx += nr) {
    unpack_weights_at_n_idx<weight_nbit, nr, kr, sr>(
        weight_qvals + n_idx * k,
        weight_scales + n_idx * groups_per_k,
        weight_zeros + n_idx * groups_per_k,
        bias + n_idx,
        n_idx,
        n,
        k,
        group_size,
        has_weight_zeros,
        has_bias,
        packed_weights);
  }
}

template <int weight_nbit, int nr, int kr, int sr>
void pack_weights_with_lut(
    // Output
    void* packed_weights,
    // Inputs
    int n,
    int k,
    int group_size,
    const int8_t* weight_qval_idxs,
    int n_luts,
    const int8_t* luts,
    const float* weight_scales,
    // weight_zeros not packed if nullptr
    const int8_t* weight_zeros,
    // bias not packed if nullptr
    const float* bias) {
  internal::pack_weights_impl<weight_nbit, nr, kr, sr, /*has_lut*/ true>(
      packed_weights,
      n,
      k,
      group_size,
      weight_qval_idxs,
      n_luts,
      luts,
      weight_scales,
      weight_zeros,
      bias);
}

size_t inline packed_weights_with_lut_size(
    int n,
    int k,
    int group_size,
    int weight_nbit,
    bool has_weight_zeros,
    bool has_bias,
    int nr) {
  auto packed_weights_col_size = internal::packed_weights_size_per_n(
      k, group_size, weight_nbit, has_weight_zeros, has_bias);

  // Replace n with next multiple of nr >= n
  n = ((n + nr - 1) / nr) * nr;

  // Per nr columns, we have one 16 byte lut
  auto lut_size = (n / nr) * 16;

  return packed_weights_col_size * n + lut_size;
}

} // namespace
  // torchao::kernels::cpu::aarch64::linear::channelwise_8bit_activation_groupwise_lowbit_weight::weight_packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
