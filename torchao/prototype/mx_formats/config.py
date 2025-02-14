# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class MXLinearConfig:
    # If True, uses a custom triton kernel for fp4 dequantize
    use_fp4_custom_triton_dequant_kernel: bool = False
