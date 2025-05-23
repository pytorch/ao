# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.quantization import (
    FbgemmConfig,
    quantize_,
)


class TestFbgemmInt4Tensor(TestCase):
    def test_linear(self):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        config = FbgemmConfig(io_dtype="bf16i4bf16", is_grouped_mm=False)
        quantize_(linear, config)


if __name__ == "__main__":
    run_tests()
