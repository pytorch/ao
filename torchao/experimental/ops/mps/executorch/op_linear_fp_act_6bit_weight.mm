// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Unlike ATen, ExecuTorch op registration appears to only allow one
// EXECUTORCH_LIBRARY per cpp file due to a name redefinition error, so a new
// file is needed for each variant

#include <torchao/experimental/ops/mps/executorch/linear_fp_act_xbit_weight.h>

EXECUTORCH_LIBRARY(torchao, "_linear_fp_act_6bit_weight.out", linear_mps_kernel_et_ctx_out<6>);
