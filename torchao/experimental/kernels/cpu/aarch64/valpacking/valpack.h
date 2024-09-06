// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace torchao {
namespace kernels {
namespace cpu {
namespace valpacking {

// TODO: should this be relocated out of aarch64?
void interleave_data(
    void* data_interleaved,
    const void* data,
    int bytes_per_val,
    int vals_per_channel,
    int vals_per_group,
    int vals_per_chunk,
    int channels,
    int channel_stride_in_vals);

} // namespace valpacking
} // namespace cpu
} // namespace kernels
} // namespace torchao
