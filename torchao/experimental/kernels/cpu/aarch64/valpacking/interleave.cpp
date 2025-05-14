// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torchao/experimental/kernels/cpu/aarch64/valpacking/valpack.h>
#include <cassert>
#include <cstring>
#include <cstdint>

// Interleaves data across channels (row/column) and groups.
// Each channel is the same size (vals_per_channel) and is
// divided into groups (vals_per_channel % vals_per_group == 0).
// Each group is divided into chunks (vals_per_group % vals_per_chunk == 0).
// Chunks are interleaved.
//
// Data is interleaved by iterating over chunks, then groups, and then channels.
//
// For example, given original data (depicted below with channels as
// rows, vals_per_channel=12, vals_per_group = 4, vals_per_chunk=2):
//
//  group0                  group1                  group2
//  chunk0      chunk1      chunk0      chunk1      chunk0      chunk1
// [(v00, v01 | v02, v03) | (v04, v05 | v06, v07) | (v08, v09 | v0a, v0b)] ch0
// [(v10, v11 | v12, v13) | (v14, v15 | v16, v17) | (v18, v19 | v1a, v1b)] ch1
// [(v20, v21 | v22, v23) | (v24, v25 | v26, v27) | (v28, v29 | v2a, v2b)] ch2
// [(v30, v31 | v32, v33) | (v34, v35 | v36, v37) | (v38, v39 | v3a, v3b)] ch3
//
// The output of this method is:
//
// v00, v01 | v10, v11 | v20, v21 | v30, v31 // chunk0, group0 channels
// v04, v05 | v14, v15 | v24, v25 | v34, v35 // chunk0, group1 channels
// v08, v09 | v18, v19 | v28, v29 | v38, v39 // chunk0, group2 channels
// v02, v03 | v12, v13 | v22, v23 | v32, v33 // chunk1, group0 channels
// v06, v07 | v16, v17 | v26, v27 | v36, v37 // chunk1, group1 channels
// v0a, v0b | v1a, v1b | v2a, v2b | v3a, v3b // chunk1, group2 channels
//
// For a given value, the value in the next channel is offset by
// channel_stride_in_vals.
// It may be that channel_stride_in_vals = vals_per_channel,
// but it can be something else if we are applying this method
// to a matrix tile.

void torchao::kernels::cpu::valpacking::interleave_data(
    void* data_interleaved,
    const void* data,
    int bytes_per_val,
    int vals_per_channel,
    int vals_per_group,
    int vals_per_chunk,
    int channels,
    int channel_stride_in_vals) {
  assert(vals_per_channel % vals_per_group == 0);
  assert(vals_per_group % vals_per_chunk == 0);

  int chunks_per_group = vals_per_group / vals_per_chunk;
  int groups_per_channel = vals_per_channel / vals_per_group;
  int bytes_per_chunk = vals_per_chunk * bytes_per_val;

  int8_t* output_byte_ptr = (int8_t*)(data_interleaved);
  const int8_t* input_byte_ptr = (int8_t*)(data);

  for (int chunk_idx = 0; chunk_idx < chunks_per_group; chunk_idx++) {
    for (int group_idx = 0; group_idx < groups_per_channel; group_idx++) {
      for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
        // Index of first value in chunk we're moving
        int val_idx = (channel_idx * channel_stride_in_vals) +
            (group_idx * vals_per_group) + (chunk_idx * vals_per_chunk);

        // Copy chunk to correct location
        std::memcpy(
            output_byte_ptr,
            input_byte_ptr + val_idx * bytes_per_val,
            bytes_per_chunk);
        output_byte_ptr += bytes_per_chunk;
      }
    }
  }
}
