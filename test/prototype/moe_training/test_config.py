# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchao.prototype.moe_training.config import Float8TrainingOpConfig
from torchao.quantization.quantize_.common import KernelPreference


def test_blockwise_fp8_config_rejects_unsupported_kernel_preference():
    with pytest.raises(
        ValueError,
        match="fp8_grouped_mm_recipe='blockwise'.*KernelPreference.AUTO.*KernelPreference.EMULATED",
    ):
        Float8TrainingOpConfig(
            fp8_grouped_mm_recipe="blockwise",
            kernel_preference=KernelPreference.TRITON,
        )


def test_rowwise_fp8_config_allows_ignored_kernel_preference():
    Float8TrainingOpConfig(
        fp8_grouped_mm_recipe="rowwise",
        kernel_preference=KernelPreference.TRITON,
    )
