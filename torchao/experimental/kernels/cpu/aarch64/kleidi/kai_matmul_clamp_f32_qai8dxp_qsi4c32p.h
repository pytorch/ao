// namespace example
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h>

namespace torchao::kernels::cpu::aarch64::kleidi {
namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p {

using ukernel = struct kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel;

} // namespace kai_matmul_clamp_f32_qai8dxp_qsi4c32p
} // namespace torchao::kernels::cpu::aarch64::kleidi
