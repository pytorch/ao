import unittest

import torch
import torch.nn as nn

from torchao.prototype.quantization.mixed_precision.scripts import intN_weight_only
from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error

_CUDA_IS_AVAILABLE = torch.cuda.is_available()


class TestWeightOnlyQuantNaive(unittest.TestCase):
    def test_quantization_intNwo(self):
        # skip test int4wo for now since it is under development in torchao
        for quantization_bit in [2, 3, 5, 6, 8]:
            for symmetric in [False, True]:
                with self.subTest(
                    quantization_bit=quantization_bit, symmetric=symmetric
                ):
                    for x_shape in [[64, 32], [80, 80, 80, 32], [16, 64, 32]]:
                        x = torch.randn(*x_shape, dtype=torch.bfloat16)
                        m = nn.Sequential(nn.Linear(32, 80)).bfloat16()
                        y_ref = m(x)
                        quantize_(
                            m,
                            intN_weight_only(
                                n=quantization_bit, group_size=32, symmetric=symmetric
                            ),
                        )
                        y_wo = m(x)
                        sqnr = compute_error(y_ref, y_wo)
                        # SQNR_dB can be approximated by 6.02n, where n is the bit width of the quantization
                        # e.g., we set sqnr threshold = 44 for 8-bit, so that 6.02 * 8= 48.16 fullfills
                        expected_sqnr_threshold = 44.0 - (8 - quantization_bit) * 6.02
                        self.assertGreater(
                            sqnr, expected_sqnr_threshold, f"sqnr: {sqnr} is too low"
                        )


if __name__ == "__main__":
    unittest.main()
