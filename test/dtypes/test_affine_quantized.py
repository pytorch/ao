from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
from torchao.quantization.quant_api import int4_weight_only
import torch
import unittest
from torchao.utils import (
    TORCH_VERSION_AFTER_2_5,
)


class TestAffineQuantized(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    # @unittest.skipIf(TORCH_VERSION_AFTER_2_5, "int4 skipping 2.5+ for now")
    def test_tensor_core_layout_transpose(self):
        l = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        t = l.weight
        shape = t.shape
        apply_int4_weight_only_quant = int4_weight_only(group_size=32)
        ql = apply_int4_weight_only_quant(l)
        aqt = ql.weight
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
