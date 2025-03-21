#pragma once

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <torchao/experimental/kernels/cpu/aarch64/bitpacking/bitpack.h>
#include <torchao/experimental/kernels/cpu/aarch64/macro.h>
#include <torchao/experimental/kernels/cpu/aarch64/reduction/reduction.h>
#include <cstring>

namespace torchao::kernels::cpu::aarch64::linear::packing {

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

// Packs nr * kr values for GEMM with packing params (nr, kr, sr)
// It takes (kr / sr) values from each of nr columns and writes to packed_values
// This is repeated sr times
template <typename T>
void pack_values(
    // Output
    T* packed_values,
    // Inputs
    const T* values,
    int nr,
    int kr,
    int sr) {
  assert(kr % sr == 0);
  int kr_per_sr = kr / sr;
  int dst_idx = 0;
  for (int sr_idx = 0; sr_idx < sr; sr_idx++) {
    for (int n_idx = 0; n_idx < nr; n_idx++) {
      // Take kr_per_sr values from column n_idx
      std::memcpy(
          packed_values + dst_idx,
          values + n_idx * kr + sr_idx * kr_per_sr,
          sizeof(T) * kr_per_sr);
      dst_idx += kr_per_sr;
    }
  }
}

// Undoes pack_values
template <typename T>
void unpack_values(
    // Output
    T* values,
    // Inputs
    const T* packed_values,
    int nr,
    int kr,
    int sr) {
  // packed_values and values should have size nr * kr
  // This function takes (kr / sr) from each column of nr columns and writes to
  // output This is repeated sr times
  assert(kr % sr == 0);
  int kr_per_sr = kr / sr;
  int dst_idx = 0;
  for (int sr_idx = 0; sr_idx < sr; sr_idx++) {
    for (int n_idx = 0; n_idx < nr; n_idx++) {
      // Take kr_per_sr values from column n_idx
      std::memcpy(
          values + n_idx * kr + sr_idx * kr_per_sr,
          packed_values + dst_idx,
          sizeof(T) * kr_per_sr);
      dst_idx += kr_per_sr;
    }
  }
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
  assert(k % group_size == 0);
  assert(group_size % kr == 0);
  bool has_weight_zeros = (weight_zeros != nullptr);
  bool has_bias = (bias != nullptr);

  int groups_per_k = k / group_size;

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
                weight_qvals + (n_idx + j) * k + (k_idx + idx_in_group),
                kr);
            qvals_sum[j] += reduction::compute_sum(buffer.data() + kr * j, kr);
          }
        }

        // Pack buffer
        internal::pack_values(packed_values, buffer.data(), nr, kr, sr);
        internal::pack_buffer<weight_nbit, kr, nr>(
            packed_weights_byte_ptr, packed_values);
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
      internal::unpack_values(buffer.data(), packed_values, nr, kr, sr);

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

} // namespace torchao::kernels::cpu::aarch64::linear::packing

#endif // defined(__aarch64__) || defined(__ARM_NEON)
