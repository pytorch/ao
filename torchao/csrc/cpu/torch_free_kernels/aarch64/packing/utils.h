// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstring>

namespace torchao::packing {

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

} // namespace torchao::packing
