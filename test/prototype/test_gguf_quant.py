# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchao.prototype.quantization.gguf import (
    GGUFQuantizedTensor,
    GGUFWeightOnlyConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.quant_primitives import _choose_qparams_gguf
from torchao.quantization.utils import compute_error


class TestGGUFQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.input = torch.randn(2, 256, dtype=torch.float32)
        self.n_blocks_per_superblock = 8
        self.block_size = (1, 32)
        self.dtype = torch.uint4

    def test__choose_qparams_gguf(self):
        (
            super_block_scale_scale,
            super_block_min_scale,
            quantized_block_scale,
            quantized_block_min,
        ) = _choose_qparams_gguf(self.input, self.block_size, self.dtype)

        assert super_block_scale_scale.shape, (2, 8)
        assert super_block_min_scale.shape, (2, 8)
        assert quantized_block_scale.shape, (2, 32)

    def test_gguf_quantized_tensor_from_float(self):
        gqt = GGUFQuantizedTensor.from_float(
            self.input,
            self.n_blocks_per_superblock,
            self.dtype,
        )

        dequant = gqt.dequantize()

        sqnr = compute_error(dequant, self.input)
        self.assertGreater(sqnr, 30)

    def test_quantize_api(self):
        m = torch.nn.Sequential(torch.nn.Linear(256, 64))
        quantize_(m, GGUFWeightOnlyConfig())
        assert type(m[0].weight) == GGUFQuantizedTensor


if __name__ == "__main__":
    unittest.main()
