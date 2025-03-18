# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch

# This is conceptually an enum of non-core dtypes
# TODO(future PR): change to a cleaner way to represent this without
# regressing torch.compile and while keeping things readable.
DTYPE_FP4 = "fp4_e2m1"
DTYPE_FP6_E3M2 = "fp6_e3m2"
DTYPE_FP6_E2M3 = "fp6_e2m3"

# Supported element dtypes
# TODO(future PR): add support for MX int8
SUPPORTED_ELEM_DTYPES = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    DTYPE_FP4,
]

F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
F8E5M2_MAX = torch.finfo(torch.float8_e5m2).max  # 57344.0

F8E4M3_MAX_POW2 = 8  # 256
F8E5M2_MAX_POW2 = 15  # 32768
F6_E2M3_MAX_POW2 = 2  # 4
F6_E3M2_MAX_POW2 = 4  # 16
F4_E2M1_MAX_POW2 = 2  # 4

E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255

F32_EXP_BIAS = 127
BF16_EXP_BIAS = 127
F6_E2M3_EXP_BIAS = 1
F6_E3M2_EXP_BIAS = 3
F4_E2M1_EXP_BIAS = 1

F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)

F6_E2M3_MAX = 7.5
F6_E2M3_MIN_NORMAL = 1.0
F6_E2M3_MAX_INT = 31  # integer corresponding to 0b00011111

F6_E3M2_MAX = 28.0
F6_E3M2_MIN_NORMAL = 0.25
F6_E3M2_MAX_INT = 31  # integer corresponding to 0b00011111

F4_E2M1_MAX = 6.0
F4_E2M1_MIN_NORMAL = 1.0
F4_E2M1_MAX_INT = 7

BLOCK_SIZE_DEFAULT = 32
