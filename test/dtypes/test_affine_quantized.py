from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
from torchao.quantization.quant_api import int4wo
import torch
import unittest


class TestAffineQuantized(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_tensor_core_layout_transpose(self):
        t = torch.rand(128, 256, dtype=torch.bfloat16, device="cuda")
        shape = t.shape
        apply_int4wo_quant = int4wo(groupsize=32)
        aqt = apply_int4wo_quant(t)
        aqt_shape = aqt.shape
        self.assertEqual(aqt_shape, shape)

        # transpose shape test
        for _ in range(10):
            t = t.t()
            aqt = aqt.t()
            shape = t.shape
            aqt_shape = aqt.shape
            self.assertEqual(aqt_shape, shape)

if __name__ == "__main__":
    run_tests()
