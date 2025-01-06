import unittest

import torch

from torchao.prototype.quantization.codebook import (
    CodebookQuantizedTensor,
    choose_qparams_codebook,
)
from torchao.quantization.utils import compute_error


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


if __name__ == "__main__":
    unittest.main()
