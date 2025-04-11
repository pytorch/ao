# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch

from torchao.prototype.quantization.codebook import (
    CodebookQuantizedTensor,
    CodebookWeightOnlyConfig,
    choose_qparams_codebook,
)
from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error
from torchao.testing.utils import skip_if_no_cuda
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5


class TestCodebookQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.input = torch.randn(100, 256, dtype=torch.float32)
        self.block_size = (1, 1)
        self.scale_block_size = 64
        self.code_dtype = torch.uint8
        self.chunk_size = 1024

    def test_choose_qparams_codebook(self):
        codebook, scales = choose_qparams_codebook(
            self.input,
            block_size=self.block_size,
            scale_block_size=self.scale_block_size,
            code_dtype=self.code_dtype,
        )
        self.assertEqual(codebook.dim(), len(self.block_size) + 1)

        self.assertFalse(torch.isnan(codebook).any())
        self.assertFalse(torch.isnan(scales).any())

    def test_codebook_quantized_tensor_from_float(self):
        cqt = CodebookQuantizedTensor.from_float(
            self.input,
            block_size=self.block_size,
            code_dtype=self.code_dtype,
            scale_block_size=self.scale_block_size,
            chunk_size=self.chunk_size,
        )

        dequant = cqt.dequantize()

        sqnr = compute_error(dequant, self.input)
        self.assertGreater(sqnr, 30)

    def test_codebook_quantized_tensor_from_float2(self):
        block_size = (1, 16)
        code_dtype = torch.int32
        scale_block_size = self.input.shape[1]

        cqt = CodebookQuantizedTensor.from_float(
            self.input,
            block_size=block_size,
            code_dtype=code_dtype,
            scale_block_size=scale_block_size,
            chunk_size=self.chunk_size,
        )

        dequant = cqt.dequantize()

        sqnr = compute_error(dequant, self.input)
        self.assertGreater(sqnr, 30)

    def test_quantize_api(self):
        m = torch.nn.Sequential(torch.nn.Linear(64, 64))
        quantize_(m, CodebookWeightOnlyConfig())
        assert type(m[0].weight) == CodebookQuantizedTensor

    @skip_if_no_cuda()
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_5, "requires 2.5+.")
    def test_export(self):
        m = torch.nn.Sequential(torch.nn.Linear(128, 64)).to(dtype=torch.bfloat16)
        quantize_(m, CodebookWeightOnlyConfig())
        example_inputs = (torch.randn(1, 128, dtype=torch.bfloat16),)
        print("m:", m)
        # torchao.utils.unwrap_tensor_subclass(m)
        m = torch.export.export_for_training(m, example_inputs).module()
        print("m:", m)
        targets = [n.target for n in m.graph.nodes]
        self.assertTrue(torch.ops.quant.quantize_codebook.default in targets)


if __name__ == "__main__":
    unittest.main()
