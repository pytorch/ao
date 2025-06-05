# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.quantization import (
    FbgemmConfig,
    quantize_,
)
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_90,
)


@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Need pytorch 2.8+")
class TestFbgemmInt4Tensor(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
    def test_linear(self):
        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        original = linear(input)
        config = FbgemmConfig(
            input_dtype=torch.bfloat16,
            weight_dtype=torch.int4,
            output_dtype=torch.bfloat16,
            block_size=[1, 128],
        )
        quantize_(linear, config)
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 20)


if __name__ == "__main__":
    run_tests()
