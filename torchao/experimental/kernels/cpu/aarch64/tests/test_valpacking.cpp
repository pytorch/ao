// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include <torchao/experimental/kernels/cpu/aarch64/valpacking/valpack.h>
#include <cassert>

TEST(InterleaveDataTest, InterleaveChannels) {
  // interleave 4 rows of 6 elements
  int bytes_per_val = 4; // int32_t
  int vals_per_channel = 6;
  int vals_per_group = 6;
  int vals_per_chunk = 3;
  int channels = 4;
  int channel_stride_in_vals = vals_per_channel;

  int data_size = channels * vals_per_channel;
  assert(data_size == 24);
  int32_t data[data_size];
  int32_t data_interleaved[data_size];
  for (int i = 0; i < data_size; i++) {
    data[i] = i;
    data_interleaved[i] = 0;
  }
  int32_t expected_data_interleaved[] = {0,  1,  2,  6,  7,  8,  12, 13,
                                         14, 18, 19, 20, 3,  4,  5,  9,
                                         10, 11, 15, 16, 17, 21, 22, 23};

  torchao::kernels::cpu::valpacking::interleave_data(
      data_interleaved,
      data,
      bytes_per_val,
      vals_per_channel,
      vals_per_group,
      vals_per_chunk,
      channels,
      channel_stride_in_vals);

  for (int i = 0; i < data_size; ++i) {
    EXPECT_EQ(data_interleaved[i], expected_data_interleaved[i]);
  }
}

TEST(InterleaveDataTest, InterleaveChannelsAndGroups) {
  // Test this example:
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

  // interleave 4 rows of 6 elements
  int bytes_per_val = 4; // int32_t
  int vals_per_channel = 12;
  int vals_per_group = 4;
  int vals_per_chunk = 2;
  int channels = 4;
  int channel_stride_in_vals = vals_per_channel;

  int data_size = channels * vals_per_channel;
  assert(data_size == 48);
  int32_t data[data_size];
  int32_t data_interleaved[data_size];
  for (int i = 0; i < data_size; i++) {
    data[i] = i;
    data_interleaved[i] = 0;
  }
  int32_t expected_data_interleaved[] = {
      0, 1, 12, 13, 24, 25, 36, 37, 4,  5,  16, 17, 28, 29, 40, 41,
      8, 9, 20, 21, 32, 33, 44, 45, 2,  3,  14, 15, 26, 27, 38, 39,
      6, 7, 18, 19, 30, 31, 42, 43, 10, 11, 22, 23, 34, 35, 46, 47};

  torchao::kernels::cpu::valpacking::interleave_data(
      data_interleaved,
      data,
      bytes_per_val,
      vals_per_channel,
      vals_per_group,
      vals_per_chunk,
      channels,
      channel_stride_in_vals);

  for (int i = 0; i < data_size; ++i) {
    EXPECT_EQ(data_interleaved[i], expected_data_interleaved[i]);
  }
}
